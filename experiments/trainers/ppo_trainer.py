#https://gymnasium.farama.org/introduction/basic_usage/

# Standard library imports
import os
from functools import reduce
from datetime import datetime

# Third-party imports
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from experiments.logger import Logger
from experiments.environment import Environment


# Absolute imports from your package
from experiments.models import MLPValue, MLPPolicy
from experiments.trajectory import Trajectory, BatchTrajectory
from experiments.config import PPOConfigBase
from experiments.trainers.base_trainer import BaseTrainer

class PPOTrainer(BaseTrainer):
    def __init__(self, config: PPOConfigBase):
        self.config = config
        self.logger = Logger(self.config)

        # Class members
        self.tjs = []
        self.env = Environment(self.config)

        # Models
        self.policy_model = MLPPolicy(obs_dim=self.env.obs_dim, action_dim=self.env.action_dim)
        self.value_model = MLPValue(obs_dim=self.env.obs_dim)
        self.old_policy_model = MLPPolicy(obs_dim=self.env.obs_dim, action_dim=self.env.action_dim)
        self.old_value_model = MLPValue(obs_dim=self.env.obs_dim)

        # Optimizers
        self.optimizer_policy = optim.Adam(self.policy_model.parameters(), lr = self.config.alpha)
        self.optimizer_value = optim.Adam(self.value_model.parameters(), lr = self.config.alpha)

    def gen_trajectory(self):
        observation, info = self.env.reset()
        tj = Trajectory(init_state=observation, obs_dim=self.env.obs_dim, action_dim=self.env.action_dim)

        # Generate an episode
        episode_finished = False
        while not episode_finished:

            old_value = self.old_value_model.forward(torch.from_numpy(observation))
            old_policy = self.old_policy_model.forward(torch.from_numpy(observation))
            
            # Act on old policy
            action = np.random.choice([0,1], size=1, p=old_policy.detach().numpy())[0]
            observation, reward, terminated, truncated, info = self.env.step(action)

            tj.add_step(observation, action, reward, old_policy, old_value, 0)

            if terminated or truncated:
                episode_finished = True


        # Episode finished
        tj.compute_gae(gamma=self.config.gamma, lam=self.config.lam)
        tj.compute_R(gamma=self.config.gamma)

        return tj

    def compute_value_loss_mse(self, states, R):
        new_values = self.value_model.forward(states).squeeze(1)
        loss_value = torch.mean(F.mse_loss(new_values, R))
        return loss_value

    def compute_policy_loss_ppo(self, states, old_actions, old_probs, A):
        new_policies = self.policy_model.forward(states)
        new_probs = new_policies.gather(1, old_actions.long().unsqueeze(1)).squeeze(1)
        r = new_probs / old_probs.detach()

        # Compute ppo loss
        loss_ppo = torch.min(r * A.detach(), \
                            torch.clamp(r, 1-self.config.eps , 1+self.config.eps ) * A.detach())
        loss_ppo = -torch.mean(loss_ppo)

        # Entropy regularization
        entropy = -torch.mean(new_policies * torch.log2(new_policies))
        loss_ppo -= self.config.beta * entropy

        return loss_ppo, entropy

    def train(self):
        for epoch in range(self.config.num_train_iter):
            print("train_iter: ", epoch)
            
            self.global_step = 0

            if self.global_step > self.config.max_env_steps: 
                break 
            
            # 1. N "parallel" actors each generate a trajectory
            #       - runs policy on environment until failure
            #       - computes advantage estimates
            #      Note: Parallelism is simulated for now
            tjs = [self.gen_trajectory() for _ in range(self.config.N)]

            # 1.1 Merge and load recent trajectory data 
            batched_tj = BatchTrajectory(tjs)
            loader = DataLoader(batched_tj, batch_size=self.config.M, shuffle=True)

            # 2. Optimize loss, for K epochs
            for k in range(self.config.K):
                # Update new policy for each minibatch
                for _, data in enumerate(loader):

                    self.optimizer_policy.zero_grad()
                    self.optimizer_value.zero_grad()

                    states, old_actions, rewards, old_policies, old_values, old_probs, R, A = data

                    # 2.1 Compute mse loss for value model
                    loss_value = self.compute_value_loss_mse(states, R)
          
                    # 2.2 Compute ppo loss for policy model
                    loss_ppo, entropy = self.compute_policy_loss_ppo(states, old_actions, old_probs, A)

                    # 2.3 Update models
                    loss_value.backward()
                    loss_ppo.backward()
                    self.optimizer_policy.step()
                    self.optimizer_value.step()

                    self.global_step += len(rewards) # could alternatively just increment by 1
                
                # Logging
                self.logger.log(
                    scalars={
                        "loss_value": loss_value.item(),
                        "loss_ppo": loss_ppo.item(),
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "A": torch.mean(A).item(),
                        "policy_entropy": entropy.item(),
                        "total_reward": (1.0 * len(batched_tj) / self.config.N),
                        "global_step": self.global_step
                        # "lr": self.lr_scheduler.get_last_lr()[0]
                        },
                    models=[self.policy_model, self.value_model]
                )
                
            # 3. Theta old <-- theta new
            self.old_policy_model.load_state_dict(self.policy_model.state_dict())
            self.old_value_model.load_state_dict(self.value_model.state_dict())

        return self.policy_model      

        
    def demonstrate(self, num_demonstrations):
        self.demonstrate_env = gym.make('CartPole-v1', render_mode = 'human')

        for i in range(num_demonstrations):
            observation, info = self.demonstrate_env.reset()

            episode_finished = False
            while not episode_finished:
                policy = self.policy_model.forward(torch.from_numpy(observation))
                action = np.argmax(policy.detach().numpy())
                observation, reward, terminated, truncated, info = self.demonstrate_env.step(action)

                if terminated or truncated:
                    episode_finished = True
        self.demonstrate_env.close()
    
    def record(self, num_videos=10, name_prefix="eval"):
        self.record_env = gym.make('CartPole-v1', render_mode = 'rgb_array')
        self.record_env = RecordVideo(
            self.record_env,
            video_folder=self.config.video_folder_name, 
            name_prefix=name_prefix + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_",
            episode_trigger=lambda x: True
        )
    
        for i in range(num_videos):
            observation, info = self.record_env.reset()

            episode_finished = False
            while not episode_finished:
                policy = self.policy_model.forward(torch.from_numpy(observation))
                action = np.argmax(policy.detach().numpy())
                observation, reward, terminated, truncated, info = self.record_env.step(action)

                if terminated or truncated:
                    episode_finished = True
        self.record_env.close()


