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
from experiments.environment import RLHFEnvironment
from experiments.profiler import profile
from experiments.datasets import TLDRFilteredDataPPO


from experiments.models import Llama_3p2_1B_Policy, Llama_3p2_1B_Value
from experiments.trajectory import Trajectory, TrajectorySet
from experiments.config import PPOConfigBase
from experiments.trainers.base_trainer import BaseTrainer

class PPORLHFTrainer(BaseTrainer):
    def __init__(self, config: PPOConfigBase):
        self.config = config
        self.logger = Logger(self.config)

        # Models
        self.policy_model = Llama_3p2_1B_Policy(self.config)
        self.value_model = Llama_3p2_1B_Value(self.config)
        self.old_policy_state_dict = self.policy_model.state_dict()
        self.old_value_state_dict = self.value_model.state_dict()

        # If pre-compute, don't keep these in memory
        self.reward_model = Llama_3p2_1B_Value(self.config) # TODO: This is a placeholder'
        self.sft_model = Llama_3p2_1B_Value(self.config) # TODO: This is a placeholder'


        # Class members
        self.data = TLDRFilteredDataPPO(tokenizer=self.policy_model.tokenizer, batch_size=self.config.batch_size)
        self.env = RLHFEnvironment(self.config, self.data)


        # Optimizers
        self.optimizer_policy = optim.Adam(self.policy_model.parameters(), lr = self.config.alpha)
        self.optimizer_value = optim.Adam(self.value_model.parameters(), lr = self.config.alpha)

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
    def _update_old_models(self):
        self.old_policy_state_dict = self.policy_model.state_dict()
        self.old_value_state_dict = self.value_model.state_dict()


    @profile
    def _to_device(self, batch):
        pass

    @profile
    def train(self):
        self.global_step = 0

        # Go through the data num_epochs times, or max_episodes steps
        for i in range(self.config.num_epochs):
            for _, batch in enumerate(self.data.train_loader):
                if self.global_step * self.data.train_loader.batch_size > self.config.max_episodes: 
                    break 
                
                # TODO:
                # batch = self._to_device(batch)

                # 1. N "parallel" actors each generate a trajectory
                #       - runs policy on environment until failure
                #       - computes advantage estimates
                tjs = self.env.generate_trajectory(
                    batch, 
                    self.policy_model,
                    self.value_model,
                    self.reward_model,
                    #TODO: may need to be able to split this up
                    self.config.generation_temperature)
                
                tj_loader = DataLoader(TrajectorySet(tjs), batch_size=self.config._mini_batch_size, shuffle=True, num_workers=0)

                # 2. Optimize loss, for K epochs
                for k in range(self.config.K):
                    # Update new policy for each minibatch

                    curr_accumulation_steps = 0
                    self._zero_grad(self.optimizer_policy, self.optimizer_value)

                    for _, (states, old_actions, rewards, old_policies, old_values, old_probs, R, A) in enumerate(tj_loader):

                        if curr_accumulation_steps >= self.config.accumulation_steps:
                            self._zero_grad(self.optimizer_policy, self.optimizer_value)

                        new_values, new_policies = self._forward(states)

                        # 2.1 Compute mse loss for value model
                        loss_value = self.compute_value_loss_mse(R, new_values)
            
                        # 2.2 Compute ppo loss for policy model
                        loss_ppo, entropy = self.compute_policy_loss_ppo(old_actions, old_probs, A, new_policies)

                        # 2.3 Update models
                        self._backward(loss_value, loss_ppo)

                        if curr_accumulation_steps >= self.config.accumulation_steps:
                            self._step(self.optimizer_policy, self.optimizer_value)
                            self.global_step += len(rewards) # could alternatively just increment by 1
                            curr_accumulation_steps = 0
                    
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
                            # "total_reward": (1.0 * len(batched_tj) / self.config.N),
                            # TODO: total reward is different here. Should perhaps take total rewards at eos tokens
                            "global_step": self.global_step
                            # "lr": self.lr_scheduler.get_last_lr()[0]
                            },
                        models=[self.policy_model, self.value_model]
                    )
                    
                # 3. Theta old <-- theta new
                self._update_old_models()

        return self.policy_model      

    @profile
    def pre_compute_rewards_and_sft_policies(self):
        print("Pre-computing rewards...")

        # load model
        reward_model = Llama_3p2_1B_Value(self.config)

        for _, data in enumerate(self.data.train_loader):
            rewards = reward_model.forward(data)

            for idx, rm_score in zip(data['idx'], rewards):
                self.data.set_score(idx, rm_score)
