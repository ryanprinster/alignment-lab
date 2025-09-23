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
from experiments.environment import GymEnvironment
from experiments.profiler import profile


# Absolute imports from your package
from experiments.models import MLPSimple
from experiments.trajectory import Trajectory, BatchTrajectory
from experiments.config import PPOConfigBase
from experiments.trainers.base_trainer import BaseTrainer

class PPORLHFTrainer(BaseTrainer):
    def __init__(self, config: PPOConfigBase):
        self.config = config
        self.logger = Logger(self.config)

        # Class members
        self.env = GymEnvironment(self.config)

        # Models
        self.policy_model = MLPSimple(obs_dim=self.env.obs_dim, action_dim=self.env.action_dim)
        self.value_model = MLPSimple(obs_dim=self.env.obs_dim)
        self.old_policy_model = MLPSimple(obs_dim=self.env.obs_dim, action_dim=self.env.action_dim)
        self.old_value_model = MLPSimple(obs_dim=self.env.obs_dim)

        # Optimizers
        self.optimizer_policy = optim.Adam(self.policy_model.parameters(), lr = self.config.alpha)
        self.optimizer_value = optim.Adam(self.value_model.parameters(), lr = self.config.alpha)

    @profile
    def _generate_trajectory(self):
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
        # TODO: Compute values
        # self.old_value_model.forward_parallel_decode()
        tj.compute_gae(gamma=self.config.gamma, lam=self.config.lam)
        tj.compute_R(gamma=self.config.gamma)

        return tj
    
    @profile
    def _generate_n_trajectories(self, N=None, M=None):
        # Parallelism is simulated for now
        # TODO: Review batching logic
        batched_tj = BatchTrajectory([self._generate_trajectory() for _ in range(N or self.config.N)])
        loader = DataLoader(batched_tj, batch_size=(M or self.config.M), shuffle=True)
        return batched_tj, loader
    
    def _zero_grad(self, optimizer_policy, optimizer_value):
        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()

    @profile
    def _forward(self, states):
        # new_values = self.value_model.forward_parallel_decode(states).squeeze(1)
        # new_policies = self.policy_model.forward_parallel_decode(states)

        new_values = self.value_model.forward(states).squeeze(1)
        new_policies = self.policy_model.forward(states)

        return new_values, new_policies


    def compute_value_loss_mse(self, R, new_values):
        loss_value = torch.mean(F.mse_loss(new_values, R))
        return loss_value

    def compute_policy_loss_ppo(self, old_actions, old_probs, A, new_policies):
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
    
    @profile
    def _backward(self, loss_value, loss_ppo):
        loss_value.backward()
        loss_ppo.backward()
    
    @profile
    def _step(self, optimizer_policy, optimizer_value):
        optimizer_policy.step()
        optimizer_value.step()

    @profile
    def _update_old_models(self, old_value_model, old_policy_model, value_model, policy_model):
        old_policy_model.load_state_dict(policy_model.state_dict())
        old_value_model.load_state_dict(value_model.state_dict())

    @profile
    def train(self):
        self.global_step = 0
        for i in range(self.config.num_train_iter):
            print("train_iter: ", i) 

            if self.global_step > self.config.max_env_steps: 
                break 
            
            # 1. N "parallel" actors each generate a trajectory
            #       - runs policy on environment until failure
            #       - computes advantage estimates
            batched_tj, tj_loader = self._generate_n_trajectories(N=self.config.N)

            # 2. Optimize loss, for K epochs
            for k in range(self.config.K):
                # Update new policy for each minibatch
                for _, (states, old_actions, rewards, old_policies, old_values, old_probs, R, A) in enumerate(tj_loader):

                    self._zero_grad(self.optimizer_policy, self.optimizer_value)

                    new_values, new_policies = self._forward(states)

                    # 2.1 Compute mse loss for value model
                    loss_value = self.compute_value_loss_mse(R, new_values)
          
                    # 2.2 Compute ppo loss for policy model
                    loss_ppo, entropy = self.compute_policy_loss_ppo(old_actions, old_probs, A, new_policies)

                    # 2.3 Update models
                    self._backward(loss_value, loss_ppo)
                    self._step(self.optimizer_policy, self.optimizer_value)

                    self.global_step += len(rewards) # could alternatively just increment by 1
                
                # Logging
                self.logger.log(
                    scalars={
                        "loss_value": loss_value.item(),
                        "loss_ppo": loss_ppo.item(),
                        "train_iter": i,
                        "epoch": k,
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
            self._update_old_models(
                self.old_value_model,
                self.old_policy_model,
                self.value_model,
                self.policy_model
            )

        return self.policy_model      

        
    def demonstrate(self, num_demonstrations):
        self.demonstrate_env = GymEnvironment(render_mode = 'human')

        for i in range(num_demonstrations):
            observation, info = self.demonstrate_env.reset()

            episode_finished = False
            while not episode_finished:
                policy = self.policy_model.forward(torch.from_numpy(observation))
                action = np.argmax(policy.detach().numpy())
                observation, reward, terminated, truncated, info = self.demonstrate_env.step(action)

                if terminated or truncated:
                    episode_finished = True
    
    def record(self, num_videos=10, name_prefix="eval"):
        self.record_env = GymEnvironment(render_mode = 'rgb_array')
        self.record_env = RecordVideo(
            self.record_env.env, # Havent made it truly inherit at the moment
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


