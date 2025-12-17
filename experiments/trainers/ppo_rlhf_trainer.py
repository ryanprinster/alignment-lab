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

# from experiments.models_v2 import HFModel_Policy, HFModel_Value, HFModel_SFT, HFModel_Reward
# from experiments.models import HFModel_Policy, HFModel_Value, HFModel_SFT, HFModel_Reward
from experiments.models import HFModel_Policy, HFModel_Value, HFModel_SFT, HFModel_Reward

from experiments.trajectory import Trajectory, TrajectorySet
from experiments.config import PPOConfigBase
from experiments.trainers.base_trainer import BaseTrainer
from experiments.checkpointer import Checkpointer
from torch.optim.lr_scheduler import LinearLR
from experiments.monitor import detect_nans

import anthropic


class PPORLHFTrainer(BaseTrainer):
    @profile
    def __init__(self, config: PPOConfigBase):
        self.config = config
        self.checkpointer = Checkpointer(self.config)

        self.logger = Logger(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Models
        self.sft_model = HFModel_SFT.init_from_hf_pretrained(self.config).to(self.device).requires_grad_(False)
        self.sft_model.set_from_local_state_dict(self.config.sft_model_path)
        
        self.reward_model = HFModel_Reward.init_from_hf_pretrained(self.config).to(self.device).requires_grad_(False)
        self.reward_model.set_from_local_state_dict(self.config.rm_model_path)

        self.policy_model = HFModel_Policy.init_from_hf_pretrained(self.config).to(self.device)
        self.policy_model.set_from_local_state_dict(self.config.sft_model_path)
        
        self.value_model = HFModel_Value.init_from_hf_pretrained(self.config).to(self.device)
        self.value_model.set_from_local_state_dict(self.config.rm_model_path)

        self.old_policy_state_dict = self.policy_model.state_dict()
        self.old_value_state_dict = self.value_model.state_dict()
        
        # Optimizers + LR Schedulers
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
        
        # Class members
        self.data = TLDRFilteredDataPPO(tokenizer=self.policy_model.tokenizer, batch_size=self.config.batch_size)
        self.env = RLHFEnvironment(self.config, self.data)
        self.global_step = 0

        # Mixed precision training
        self.mixed_precision_context = autocast("cuda", dtype=torch.bfloat16) if self.config.enable_mixed_precision_training else nullcontext()


        if self.config.resume_from_checkpoint:
            print("Loading initial checkpoint...")
            self._load_from_checkpoint(policy_checkpoint_path=self.config.policy_checkpoint_path, value_checkpoint_path=self.config.value_checkpoint_path)

    @profile
    def _load_from_checkpoint(self, policy_checkpoint_path=None, value_checkpoint_path=None):
        training_state = {'global_step': 0, 'epoch': 0}
    
        # Policy Model
        if policy_checkpoint_path:
            policy_state = self.checkpointer.load_checkpoint(
                policy_checkpoint_path,
                self.policy_model,
                self.device,
                self.optimizer_policy
            )
            training_state = policy_state
            
            self.old_policy_state_dict = self.policy_model.state_dict()

        # Value Model
        if value_checkpoint_path:
            value_state = self.checkpointer.load_checkpoint(
                value_checkpoint_path,
                self.value_model,
                self.device,
                self.optimizer_value
            )
            if not policy_checkpoint_path:
                training_state = value_state
                
            self.old_value_state_dict = self.value_model.state_dict()
            print("Updated old_value_state_dict")

        #  LR schedulers
        if training_state['global_step'] > 0:
            total_iters = int(self.config.max_episodes / self.config.batch_size) * self.config.K
            
            self.lr_scheduler_policy = LinearLR(
                self.optimizer_policy,
                total_iters=total_iters,
                start_factor=1.0,
                end_factor=self.config.lr_final_ratio
            )

            for _ in range(training_state['global_step']):
                self.lr_scheduler_policy.step()
            
            self.lr_scheduler_value = LinearLR(
                self.optimizer_value,
                total_iters=total_iters,
                start_factor=1.0,
                end_factor=self.config.lr_final_ratio
            )
            for _ in range(training_state['global_step']):
                self.lr_scheduler_value.step()
            
            print(f"LR schedulers advanced to step {training_state['global_step']}")
        
        # Global step
        self.global_step = training_state['global_step']
        
        return training_state
    
    @profile
    def _log(self, loss_ppo, loss_value, k, old_data, ppo_log_data):
        eos_mask = old_data['states'][:,1:] == self.data.tokenizer.eos_token_id
        non_eos_mask = (old_data['states'][:,1:] != self.data.tokenizer.eos_token_id) & (old_data['states'][:,1:] != self.data.tokenizer.pad_token_id)
        clipped_mask = (ppo_log_data['ratios'] > 1 + self.config.eps_policy_clipping) | (ppo_log_data['ratios'] < 1 - self.config.eps_policy_clipping)
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
                "explained_var": 1 - masked_var(old_data['A_raw'], old_data['action_pad_mask']).item() / masked_var(old_data['A'] + old_data['values'][:,:-1], old_data['action_pad_mask']).item(), # 1 - var(A) / var(A + V)
                # State stats
                "eos_count": eos_mask.float().sum().item(),
                "eos_pct": (eos_mask.float().sum() / float(eos_mask.size(0))).item(),
                # Policy stats
                "pct_clipped": masked_mean(clipped_mask.float(), old_data['action_pad_mask']).item(),
                "ratio_eos": masked_mean((ppo_log_data['ratios'] - 1.0).abs(), eos_mask).item(),
                "ratio_non_eos": masked_mean((ppo_log_data['ratios'] - 1.0).abs(), non_eos_mask).item(),
                "policy_entropy": ppo_log_data['entropy'].item(),
                "policy_entropy_paper": ppo_log_data['entropy_paper'].item(),
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
                "approx_kl": ppo_log_data['approx_kl'].item(),
                "approx_kl_paper": ppo_log_data['approx_kl_paper'].item(),
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
                "length_reward_correlation": torch.corrcoef(torch.stack([
                    old_data['action_pad_mask'].float().sum(1),
                    old_data['raw_rewards'].sum(1)
                ]))[0, 1].item(),
                "unique_tokens_per_response": torch.tensor([len(torch.unique(resp)) for resp in old_data['states']]).float().mean().item(),                                
            },
            models=[self.policy_model, self.value_model],
            samples=
                {
                    "max_reward": self.data.tokenizer.decode(
                        old_data['full_states'][
                            old_data['raw_rewards'].sum(dim=1).argmax().item()
                        ]
                    ),
                    "min_reward": self.data.tokenizer.decode(
                        old_data['full_states'][
                            old_data['raw_rewards'].sum(dim=1).argmin().item()
                        ]
                    ),
                    "max_entropy":
                    self.data.tokenizer.decode(
                        old_data['full_states'][
                            ppo_log_data['entropy_per_sequence'].argmax().item()
                        ]
                    ),
                    "min_entropy":
                    self.data.tokenizer.decode(
                        old_data['full_states'][
                            ppo_log_data['entropy_per_sequence'].argmin().item()
                        ]
                    ),
                    "random": self.data.tokenizer.decode(
                        old_data['full_states'][
                            torch.randint(0, old_data['states'].size(0), ())
                        ]
                    ),
                }
        )

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

    def _compute_value_loss_mse(self, R, new_values, old_values, value_pad_mask):
        old_values = old_values[:, :-1].masked_fill(~value_pad_mask, 0)
        new_values = new_values[:, :-1].masked_fill(~value_pad_mask, 0)
        
        V_clipped = old_values + torch.clamp(
            new_values - old_values, 
            -self.config.eps_value_clipping, 
            self.config.eps_value_clipping
        )
        
        loss_value_unclipped = (new_values - R) ** 2
        loss_value_clipped = (V_clipped - R) ** 2
        
        loss_value = torch.max(loss_value_unclipped, loss_value_clipped)
        
        return masked_mean(loss_value * self.config.c1, value_pad_mask)

    def _compute_policy_loss_ppo(self, old_actions, old_log_probs, A, new_policy_logits, action_pad_mask):

        old_log_probs = old_log_probs.detach()
        A = A.detach()

        new_log_policies = masked_log_softmax(new_policy_logits, action_pad_mask.unsqueeze(2), mask_value=0, dim=-1)
        new_log_probs = torch.gather(new_log_policies, dim=-1, index=old_actions.long().unsqueeze(-1)).squeeze(-1)
        diff_log_probs = (new_log_probs - old_log_probs).masked_fill(~action_pad_mask, 0)
        
        ratios = torch.exp(diff_log_probs)

        # Compute ppo loss
        loss_ppo = torch.min(ratios * A, torch.clamp(ratios, 1-self.config.eps_policy_clipping , 1+self.config.eps_policy_clipping) * A)
        loss_ppo = -masked_mean(loss_ppo, action_pad_mask)


        ### For logging ###
        with torch.no_grad():
            entropy = torch.sum(new_log_policies * torch.exp(new_log_policies), dim=-1)
            entropy = -masked_mean(entropy, action_pad_mask)
            approx_kl = masked_mean(0.5 * diff_log_probs**2, action_pad_mask)

            # NOTE: Upon code inspection, Huang et al. computes some of these metrics slightly differently.
            # both were included here for analysis purposes.
            new_log_policies_unmasked = torch.log_softmax(new_policy_logits, dim=-1)
            entropy_per_token = -torch.sum(new_log_policies_unmasked * torch.exp(new_log_policies_unmasked), dim=-1) 
            entropy_paper = entropy_per_token.mean()
            entropy_per_sequence = entropy_per_token.sum(dim=-1) 
            approx_kl_paper = (0.5 * diff_log_probs**2).mean()

            log_data = {
                'ratios': ratios,
                'entropy': entropy,
                'approx_kl': approx_kl,
                'entropy_paper': entropy_paper,
                'entropy_per_sequence': entropy_per_sequence,
                'approx_kl_paper': approx_kl_paper

            }

        return loss_ppo, log_data
    
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
        self.policy_model.train()
        self.value_model.train()

        # Go through the data num_epochs times, or max_episodes steps
        for epoch in range(self.config.num_epochs):
            for batch_idx, batch in enumerate(self.data.train_loader):
                
                batch = self._to_device(batch)

                # 1. N "parallel" actors each generate a trajectory
                #    - runs policy on environment until failure or truncation
                #    - computes advantage estimates
                
                # FP32 --> FP16 (bfloat16) for mixed precision training
                with self.mixed_precision_context:
                    tjs = self.env.generate_trajectory(
                        batch, 
                        self.policy_model,
                        self.value_model,
                        self.sft_model,
                        self.config.generation_temperature,
                        self.reward_model)
                
                tj_loader = DataLoader(TrajectorySet(tjs), batch_size=self.config._mini_batch_size, shuffle=False, num_workers=0)

                # 2. Optimize loss and update models for K epochs 
                for k in range(self.config.K):

                    for _, old_data in enumerate(tj_loader):

                        self._zero_grad(self.optimizer_policy, self.optimizer_value)

                        # FP32 --> FP16 for mixed precision training
                        with self.mixed_precision_context:
                            new_values, new_policy_logits = self._forward(old_data['full_states'])

                            # 2.1 Compute mse loss for value model
                            self.loss_value = self._compute_value_loss_mse(old_data['R'], new_values, old_data['values'], old_data['value_pad_mask'])

                            # 2.2 Compute ppo loss for policy model
                            self.loss_ppo, ppo_log_data = self._compute_policy_loss_ppo(old_data['actions'], old_data['log_probs'], old_data['A'], new_policy_logits, old_data['action_pad_mask'])

                            del new_policy_logits

                        # 2.3 Update models
                        self._backward(self.loss_value, self.loss_ppo)
                        self._step(self.optimizer_policy, self.optimizer_value)


                        self._log(self.loss_ppo, self.loss_value, k, old_data, ppo_log_data)

                self.checkpointer.save_checkpoint(
                    self.policy_model,
                    self.optimizer_policy,
                    self.global_step,
                    epoch,
                    loss=self.loss_ppo,
                    checkpoint_prefix="policy_",
                    final_checkpoint=False
                )

                self.checkpointer.save_checkpoint(
                    self.value_model,
                    self.optimizer_value,
                    self.global_step,
                    epoch,
                    loss=self.loss_value,
                    checkpoint_prefix="value_",
                    final_checkpoint=False
                )
                                            
                # 3. Theta old <-- theta new
                self._update_old_models()

                self.global_step += 1

            # Break after a total number of episodes (train examples) instead of on training data epochs 
                if self.global_step * self.data.train_loader.batch_size >= self.config.max_episodes: 
                    break
            if self.global_step * self.data.train_loader.batch_size >= self.config.max_episodes: 
                break 
        
        # Final checkpoint
        self.checkpointer.save_checkpoint(
                self.policy_model,
                self.optimizer_policy,
                self.global_step,
                epoch,
                loss=self.loss_ppo,
                checkpoint_prefix="policy_",
                final_checkpoint=True
            )

        return self.policy_model  
        

        






