#https://gymnasium.farama.org/introduction/basic_usage/

# Standard library imports
import os
from functools import reduce
from datetime import datetime
import pdb
from contextlib import nullcontext


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
from torch.amp import GradScaler, autocast

from experiments.logger import Logger
from experiments.environment import RLHFEnvironment
from experiments.profiler import profile
from experiments.datasets import TLDRFilteredDataPPO, TLDRFilteredDataSFT


from experiments.models import Llama_3p2_1B_Policy, Llama_3p2_1B_Value, Llama_3p2_1B_SFT, Llama_3p2_1B_RM
from experiments.trajectory import Trajectory, TrajectorySet
from experiments.config import PPOConfigBase
from experiments.trainers.base_trainer import BaseTrainer
from torch.optim.lr_scheduler import LinearLR
from experiments.monitor import detect_nans


class PPORLHFTrainer(BaseTrainer):
    @profile
    def __init__(self, config: PPOConfigBase):
        self.config = config
        self.logger = Logger(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Models
        self.policy_model = Llama_3p2_1B_Policy(self.config, init_model_path=self.config.sft_model_path).to(self.device)
        self.value_model = Llama_3p2_1B_Value(self.config, init_model_path=self.config.rm_model_path).to(self.device)
        self.old_policy_state_dict = self.policy_model.state_dict()
        self.old_value_state_dict = self.value_model.state_dict()
        self.sft_model = Llama_3p2_1B_SFT(self.config, init_model_path=self.config.sft_model_path).to(self.device).requires_grad_(False)
        self.reward_model = Llama_3p2_1B_Value(self.config, init_model_path=self.config.rm_model_path).to(self.device).requires_grad_(False)

        # Class members
        self.data = TLDRFilteredDataPPO(tokenizer=self.policy_model.tokenizer, batch_size=self.config.batch_size)
        self.env = RLHFEnvironment(self.config, self.data)


        # Optimizers
        self.optimizer_policy = optim.Adam(self.policy_model.parameters(), lr = self.config.alpha)
        self.optimizer_value = optim.Adam(self.value_model.parameters(), lr = self.config.alpha)

        self.lr_scheduler_policy = LinearLR(self.optimizer_policy, 
                                        total_iters=int(self.config.max_episodes / self.config.batch_size) * self.config.K,
                                        start_factor=1.0,
                                        end_factor=self.config.lr_final_ratio * self.config.lr)
        self.lr_scheduler_value = LinearLR(self.optimizer_value, 
                                        total_iters=int(self.config.max_episodes / self.config.batch_size) * self.config.K,
                                        start_factor=1.0,
                                        end_factor=self.config.lr_final_ratio * self.config.lr)
        
        # Mixed precision training
        self.mixed_precision_context = autocast("cuda") if self.config.enable_mixed_precision_training else nullcontext()
        self.scaler_policy = GradScaler("cuda") 
        self.scaler_value = GradScaler("cuda") 


    def _zero_grad(self, optimizer_policy, optimizer_value):
        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()

    @profile
    def _forward(self, states):
        new_values = self.value_model.forward(states).squeeze(1)
        new_policy_logits, _ = self.policy_model.forward(states)
        new_policies = torch.softmax(new_policy_logits, dim=-1)

        return new_values, new_policies

    @detect_nans
    def compute_value_loss_mse(self, R, new_values):
        loss_value = torch.mean(F.mse_loss(new_values, R))
        return loss_value

    @detect_nans
    def compute_policy_loss_ppo(self, old_actions, old_probs, A, new_policies):
        pdb.set_trace()
        old_probs = old_probs.detach()
        A = A.detach()
        
        new_probs = torch.gather(new_policies, 2, old_actions.long().unsqueeze(1)).squeeze(1)
        r = new_probs / old_probs

        # Compute ppo loss
        loss_ppo = torch.min(r * A, \
                            torch.clamp(r, 1-self.config.eps , 1+self.config.eps ) * A)
        loss_ppo = -torch.mean(loss_ppo)

        # Entropy regularization
        entropy = -torch.mean(new_policies * torch.log2(new_policies))
        loss_ppo -= self.config.beta * entropy
        
        pdb.set_trace()

        # Problem 1: we need a mask
        # Problem 2: Advantages are all zero

        return loss_ppo, entropy
    
    @profile
    def _backward(self, loss_value, loss_ppo):
        if self.config.enable_mixed_precision_training:
            loss_value = self.scaler_value.scale(loss_value)
            loss_ppo = self.scaler_policy.scale(loss_ppo)
        loss_value.backward()
        loss_ppo.backward()
    
    @profile
    def _step(self, optimizer_policy, optimizer_value):

        if self.config.enable_mixed_precision_training:
            # Unscale gradient, take optimizer step, and update scale factor
            self.scaler_policy.step(self.optimizer_policy)
            self.scaler_value.step(self.optimizer_value)
            self.scaler_policy.update()
            self.scaler_value.update()
        else:
            optimizer_policy.step()
            optimizer_value.step()
        
        self.lr_scheduler_policy.step()
        self.lr_scheduler_value.step()

    @profile
    def _update_old_models(self):
        self.old_policy_state_dict = self.policy_model.state_dict()
        self.old_value_state_dict = self.value_model.state_dict()

    @profile
    def _to_device(self, batch):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)
        return batch

    @profile
    def train(self):     
        self.global_step = 0

        # Go through the data num_epochs times, or max_episodes steps
        for epoch in range(self.config.num_epochs):
            for batch_idx, batch in enumerate(self.data.train_loader):
                if self.global_step * self.data.train_loader.batch_size > self.config.max_episodes: 
                    break 
                
                batch = self._to_device(batch)

                # 1. N "parallel" actors each generate a trajectory
                #       - runs policy on environment until failure
                #       - computes advantage estimates
                
                # FP32 --> FP16 for mixed precision training
                with self.mixed_precision_context:
                    tjs = self.env.generate_trajectory(
                        batch, 
                        self.policy_model,
                        self.value_model,
                        self.sft_model,
                        self.config.generation_temperature,
                        self.reward_model)
                
                tj_loader = DataLoader(TrajectorySet(tjs), batch_size=self.config._mini_batch_size, shuffle=True, num_workers=0)

                # 2. Optimize loss, for K epochs
                for k in range(self.config.K):
                    # Update new policy for each minibatch

                    self._zero_grad(self.optimizer_policy, self.optimizer_value)

                    for _, (states, old_actions, rewards, old_policies, old_values, old_probs, R, A) in enumerate(tj_loader):

                        self._zero_grad(self.optimizer_policy, self.optimizer_value)

                        # FP32 --> FP16 for mixed precision training
                        with self.mixed_precision_context:
                            new_values, new_policies = self._forward(states)

                            # 2.1 Compute mse loss for value model
                            loss_value = self.compute_value_loss_mse(R, new_values)
                
                            # 2.2 Compute ppo loss for policy model
                            loss_ppo, entropy = self.compute_policy_loss_ppo(old_actions, old_probs, A, new_policies)

                        # 2.3 Update models
                        self._backward(loss_value, loss_ppo)
                        self._step(self.optimizer_policy, self.optimizer_value)
                    
                        # TODO: 
                        # 0. MSE Value does not change
                        # 1. new_values probably only changes every batch of N trajectories, should it change? 
                        #   No, since Nmb=1
                        #   Actually yes, slightly, since foward pass should change
                        #   Okay it does change a bit
                        # 2. R should not change
                        #   BUT R should probably not be all -1?
                        #   --> rewards seems to be pretty constant, and reversed... 

                        # Hypotheses
                        """
                        1. Something is wrong in generation of trajectory formatting rewards
                            - rewards values looks sketchy
                        1.1 Policy model is never generating EOS tokens, when it should be 
                            - Loaded model has unset weights that F things up
                        1.2 Operations on rewards not happening in correct order
                            - Whitening of rewards should happen before computation of R, whitening of advantages should happen before GAE

                        2. MSE + PPO loss not computed correctly
                            - MSE value is always the same. If values is different, it shouldnt 
                        3. Mixed precision issues
                        4. Model not updating correctly

                        --

                        Rejected
                        [x] The rewards model is not outputing the same reward for the same token 
                            - is likely predicting autoregressively
                        [x] Verified that the trained sft model IS outputing EOS tokens

                        Verified
                        [x] Needed to manually enforce 
                        """




                        # Logging
                        self.logger.log(
                            scalars={
                                "loss_value": loss_value.item(),
                                "loss_ppo": loss_ppo.item(),
                                "train_iter": epoch,
                                "global_step": self.global_step,
                                "A": torch.mean(A).item(),
                                "policy_entropy": entropy.item(),
                                "total_reward": torch.sum(rewards).item(),
                                "batch_idx": batch_idx,
                                "k": k,
                                "global_step": self.global_step
                                # "lr": self.lr_scheduler.get_last_lr()[0]
                                },
                            models=[self.policy_model, self.value_model]
                            )
                        
                # 3. Theta old <-- theta new
                self._update_old_models()

                self.global_step += 1

            
            self.checkpointer.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.global_step,
                    epoch,
                    loss=0, # placeholder
                    final_checkpoint=True
                )

        return self.policy_model      

    # @profile
    # def pre_compute_rewards(self):
    #     print("Pre-computing rewards...")

    #     data = TLDRFilteredDataSFT(tokenizer=self.policy_model.tokenizer, batch_size=self.config.batch_size)

    #     reward_model = Llama_3p2_1B_RM(self.config, init_model_path=self.config.rm_model_path).to(self.device)
    #     reward_model_v = Llama_3p2_1B_Value(self.config, init_model_path=self.config.rm_model_path).to(self.device)

    #     for i, data in enumerate(data.train_loader):
    #         print(f"i: {i}")
    #         data = self._to_device(data)
    #         rewards = reward_model.forward(data['input_ids'], data['attention_mask'])
    #         rewards_v = reward_model_v.forward(data['input_ids'], data['attention_mask'])
    #         pdb.set_trace()

    #         for idx, rm_score in zip(data['idx'], rewards):
    #             self.data.dataset['train'].set_rm_score(idx, rm_score)
