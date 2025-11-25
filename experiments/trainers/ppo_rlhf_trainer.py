#https://gymnasium.farama.org/introduction/basic_usage/

# Standard library imports
import os
from functools import reduce
from datetime import datetime
import pdb
from contextlib import nullcontext

from experiments.debug import DEBUG

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
from experiments.util import masked_mean, masked_var, masked_whiten, masked_log_softmax, whiten

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

        self.sft_model = Llama_3p2_1B_SFT(self.config, init_model_path=self.config.sft_model_path).to(self.device).requires_grad_(False)
        self.reward_model = Llama_3p2_1B_RM(self.config, init_model_path=self.config.rm_model_path).to(self.device).requires_grad_(False)
        self.reward_model.init_head_bias(self.config.calculated_sft_bias)

        self.policy_model = Llama_3p2_1B_Policy(self.config, init_model_path=self.config.sft_model_path).to(self.device)
        self.value_model = Llama_3p2_1B_Value(self.config, init_model_path=self.config.rm_model_path).to(self.device)
        self.value_model.init_head_bias(self.config.calculated_sft_bias)
        self.old_policy_state_dict = self.policy_model.state_dict()
        self.old_value_state_dict = self.value_model.state_dict()
        # Class members
        self.data = TLDRFilteredDataPPO(tokenizer=self.policy_model.tokenizer, batch_size=self.config.batch_size)
        self.env = RLHFEnvironment(self.config, self.data)


        # Optimizers
        self.optimizer_policy = optim.AdamW(self.policy_model.parameters(), lr = self.config.alpha, eps=self.config.eps_adam)
        self.optimizer_value = optim.AdamW(self.value_model.parameters(), lr = self.config.alpha, eps=self.config.eps_adam)

        self.lr_scheduler_policy = LinearLR(self.optimizer_policy, 
                                        total_iters=int(self.config.max_episodes / self.config.batch_size) * self.config.K,
                                        start_factor=1.0,
                                        end_factor=self.config.lr_final_ratio)
        self.lr_scheduler_value = LinearLR(self.optimizer_value, 
                                        total_iters=int(self.config.max_episodes / self.config.batch_size) * self.config.K,
                                        start_factor=1.0,
                                        end_factor=self.config.lr_final_ratio)
        
        # Mixed precision training
        self.mixed_precision_context = autocast("cuda", dtype=torch.bfloat16) if self.config.enable_mixed_precision_training else nullcontext()
        self.scaler_policy = GradScaler("cuda") 
        self.scaler_value = GradScaler("cuda") 


    def _zero_grad(self, optimizer_policy, optimizer_value):
        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()

    @profile
    def _forward(self, states):
        new_values = self.value_model.forward(
            states, 
            max_query_length_truncate=self.data.SFT_MAX_QUERY_LENGTH - 1).squeeze(1) 
        new_policy_logits, _ = self.policy_model.forward(states, max_query_length_truncate=self.data.SFT_MAX_QUERY_LENGTH-1)
        new_policy_logits = new_policy_logits[:, :-1, :] # slice to action indexing
        return new_values, new_policy_logits

    # @detect_nans
    # def compute_value_loss_mse(self, R, new_values, mask):
    #     loss_value = masked_mean((new_values - R) ** 2, mask)
    #     return loss_value

    def compute_value_loss_mse(self, R, new_values, old_values, value_pad_mask):
        old_values = old_values[:, :-1].masked_fill(~value_pad_mask, 0)
        new_values = new_values[:, :-1].masked_fill(~value_pad_mask, 0)
        
        # Clip the new values relative to old values
        V_clipped = old_values + torch.clamp(
            new_values - old_values, 
            -self.config.eps_value_clipping, 
            self.config.eps_value_clipping
        )
        
        # Compute both losses
        loss_value_unclipped = (new_values - R) ** 2
        loss_value_clipped = (V_clipped - R) ** 2
        
        # Take max (more pessimistic loss)
        loss_value = torch.max(loss_value_unclipped, loss_value_clipped)
        
        # Apply masking and return mean
        return masked_mean(loss_value, value_pad_mask)

    # @detect_nans
    def compute_policy_loss_ppo(self, old_actions, old_log_probs, A, new_policy_logits, action_pad_mask):

        # TODO: double check these pad masks, and all the ones with action_pad_masks, especially with values
        old_log_probs = old_log_probs.detach()
        A = A.detach()

        new_log_policies = masked_log_softmax(new_policy_logits, action_pad_mask.unsqueeze(2), mask_value=0, dim=-1)
        new_log_probs = torch.gather(new_log_policies, dim=-1, index=old_actions.long().unsqueeze(-1)).squeeze(-1)
        diff_log_probs = (new_log_probs - old_log_probs).masked_fill(~action_pad_mask, 0)
        
        ratios = torch.exp(diff_log_probs)

        # Compute ppo loss
        loss_ppo = torch.min(ratios * A, torch.clamp(ratios, 1-self.config.eps_policy_clipping , 1+self.config.eps_policy_clipping) * A)
        loss_ppo = -masked_mean(loss_ppo, action_pad_mask)

        ### For tracking ###
        with torch.no_grad():
            # Entropy for tracking, but KL is doing regularization
            entropy = torch.sum(new_log_policies * torch.exp(new_log_policies), dim=-1)
            entropy = -masked_mean(entropy, action_pad_mask)

            approx_kl = masked_mean(0.5 * diff_log_probs**2, action_pad_mask)

        return loss_ppo, entropy, ratios, approx_kl
    
    @profile
    def _backward(self, loss_value, loss_ppo):
        loss_ppo.backward()
        loss_value.backward()
    
    @profile
    def _step(self, optimizer_policy, optimizer_value):

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
        self.policy_model.train()
        self.value_model.train()

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
                
                tj_loader = DataLoader(TrajectorySet(tjs), batch_size=self.config._mini_batch_size, shuffle=False, num_workers=0)

                # 2. Optimize loss, for K epochs
                for k in range(self.config.K):
                    # Update new policy for each minibatch

                    for _, old_data in enumerate(tj_loader):


                        self._zero_grad(self.optimizer_policy, self.optimizer_value)

                        # FP32 --> FP16 for mixed precision training
                        with self.mixed_precision_context:
                            new_values, new_policy_logits = self._forward(old_data['full_states'], ) # TODO: which pad mask

                            # 2.1 Compute mse loss for value model
                            loss_value = self.compute_value_loss_mse(old_data['R'], new_values, old_data['values'], old_data['value_pad_mask'])

                            # 2.2 Compute ppo loss for policy model
                            # NOTE: all tensors in function below are in fp32?
                            loss_ppo, entropy, ratios, approx_kl = self.compute_policy_loss_ppo(old_data['actions'], old_data['log_probs'], old_data['A'], new_policy_logits, old_data['action_pad_mask'])

                            del new_policy_logits

                        # 2.3 Update models
                        self._backward(loss_value, loss_ppo)
                        self._step(self.optimizer_policy, self.optimizer_value)

                        if self.global_step % 10 == 0:
                            pdb.set_trace()

                        # Logging
                        eos_mask = old_data['states'][:,1:] == self.data.tokenizer.eos_token_id
                        non_eos_mask = (old_data['states'][:,1:] != self.data.tokenizer.eos_token_id) & (old_data['states'][:,1:] != self.data.tokenizer.pad_token_id)
                        clipped_mask = (ratios > 1 + self.config.eps_policy_clipping) | (ratios < 1 - self.config.eps_policy_clipping)
                        policy_grads = [p.grad for p in self.policy_model.parameters() if p.grad is not None]
                        value_grads = [p.grad for p in self.value_model.parameters() if p.grad is not None]

                        self.logger.log(
                            scalars={
                                "loss_value": loss_value.item(),
                                "loss_ppo": loss_ppo.item(),
                                "global_step": self.global_step,
                                "k": k,
                                # Advantage stats
                                "A_max": old_data['A_raw'].max().item(),
                                "A_min": old_data['A_raw'].min().item(),
                                "A_std": old_data['A_raw'].std().item(),
                                "A_abs_eos": masked_mean(old_data['A_raw'].abs(), eos_mask).item(),
                                "A_abs_non_eos": masked_mean(old_data['A_raw'].abs(), non_eos_mask).item(),
                                # 1 - var(A) / var(A + V)
                                "explained_var": 1 - masked_var(old_data['A_raw'], old_data['action_pad_mask']).item() / masked_var(old_data['A'] + old_data['values'][:,:-1], old_data['action_pad_mask']).item(),
                                # State stats
                                "eos_count": eos_mask.float().sum().item(),
                                "eos_pct": (eos_mask.float().sum() / float(eos_mask.size(0))).item(),
                                # Policy stats
                                "pct_clipped": masked_mean(clipped_mask.float(), old_data['action_pad_mask']).item(),
                                "ratio_eos": masked_mean((ratios - 1.0).abs(), eos_mask).item(),
                                "ratio_non_eos": masked_mean((ratios - 1.0).abs(), non_eos_mask).item(),
                                "policy_entropy": entropy.item(),
                                # Reward stats
                                "mean_raw_reward": masked_mean(
                                    old_data['raw_rewards'], 
                                    old_data['reward_mask']
                                ).item(),   
                                "mean_rlhf_reward": masked_mean(
                                    (old_data['raw_rewards'] - (self.config.beta * old_data['kl'])), 
                                    old_data['reward_mask']
                                ).item(),
                                # KL stats 
                                "kl": masked_mean(old_data['kl'], old_data['action_pad_mask']).item(),
                                "kl_mean": old_data['kl'].masked_fill(~old_data['action_pad_mask'],0).sum(1).mean().item(),
                                "kl_beta": torch.mean(old_data['kl']).item() * self.config.beta,
                                "approx_kl": approx_kl.item(),
                                # Returns stats
                                "R": masked_mean(old_data['R'], old_data['action_pad_mask']).item(),
                                # Gradient stats
                                "policy_gradient_norm": torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), float('inf')).item(),
                                "policy_max_grad": max([g.abs().max().item() for g in policy_grads]) if policy_grads else 0.0,
                                "policy_min_grad": min([g.abs().min().item() for g in policy_grads]) if policy_grads else 0.0,
                                "policy_nan_grads": sum([torch.isnan(g).sum().item() for g in policy_grads]),
                                "policy_inf_grads": sum([torch.isinf(g).sum().item() for g in policy_grads]),
                                "value_gradient_norm": torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), float('inf')).item(),
                                "value_max_grad": max([g.abs().max().item() for g in value_grads]) if value_grads else 0.0,
                                "value_min_grad": min([g.abs().min().item() for g in value_grads]) if value_grads else 0.0,
                                "value_nan_grads": sum([torch.isnan(g).sum().item() for g in value_grads]),
                                "value_inf_grads": sum([torch.isinf(g).sum().item() for g in value_grads]),
                                # Other stats
                                "lr_policy": self.lr_scheduler_policy.get_last_lr()[0],
                                "mean_sequence_length": old_data['action_pad_mask'].float().sum(1).mean().item(),
                            },
                            models=[self.policy_model, self.value_model],
                            samples=
                                {
                                    "max_reward": self.data.tokenizer.decode(
                                        old_data['states'][
                                            old_data['raw_rewards'].sum(dim=1).argmax().item()
                                        ]
                                    ),
                                    "min_reward": self.data.tokenizer.decode(
                                        old_data['states'][
                                            old_data['raw_rewards'].sum(dim=1).argmin().item()
                                        ]
                                    ),
                                    "random": self.data.tokenizer.decode(
                                        old_data['states'][
                                            torch.randint(0, old_data['states'].size(0), ())
                                        ]
                                    ),
                                }
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
