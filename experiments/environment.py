import atexit
import signal
import torch
import sys
import psutil
import pdb

import experiments.debug


from experiments.profiler import profile
from experiments.trajectory import Trajectory
from experiments.monitor import detect_nans
from experiments.util import masked_mean, masked_log_softmax, masked_whiten, whiten

import gymnasium as gym

from abc import ABC, abstractmethod
from typing import Any, Tuple
import warnings
import torch.nn.functional as F

class BaseEnvironment(ABC):
    def __init__(self):
        ### Cleanup 
        # Thanks to Claude here:
        self._closed = False
        atexit.register(self.close)  # Normal exit
        signal.signal(signal.SIGTERM, self._cleanup_signal)  # Pod termination
        signal.signal(signal.SIGINT, self._cleanup_signal)   # Ctrl+C

    # Cleanup handling
        
    def _cleanup_signal(self, signum, frame): # Thanks to Claude here
        print(f"Received signal {signum}, cleaning up...")
        self.close()
        sys.exit(0)

    def __del__(self):
        self.close()
    
    def close(self):
        if not self._closed and hasattr(self, 'env'):
            self._close()
            self._closed = True
            print("Env closed")

    @abstractmethod
    def _close(self):
        pass


class GymEnvironment(BaseEnvironment):
    def __init__(self, config, render_mode=None):
        super().__init__()

        self.config = config
        self.env = gym.make(self.config.gymnasium_env_name, render_mode)
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

    def _close(self):
        self.env.close()
    
    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
    



class RLHFEnvironment(BaseEnvironment):
    def __init__(self, config, data):
        super().__init__()

        self.config = config
        self.data = data
        self.max_sequence_length = self.data.__class__.SFT_MAX_INPUT_LENGTH
        self.max_prompt_length = self.data.__class__.SFT_MAX_QUERY_LENGTH
        self.max_response_length = self.data.__class__.SFT_MAX_REPONSE_LENGTH
        self.action_dim = self.data.tokenizer.vocab_size

    def _close(self):
        self._closed = True

    def _find_first_token_position(self, states, token_id):
        """Find the position of the first occurrence of token_id in each sequence."""
        token_mask = (states == token_id)
        first_token_pos = torch.where(
            token_mask.any(dim=1),
            token_mask.int().argmax(dim=1),
            torch.full((states.size(0),), states.size(1), device=states.device)
        )
        return first_token_pos

    def enforce_padding(self, states, tokenizer):
        """Sets all tokens after the first EOS or PAD token to PAD token."""
        first_eos_pos = self._find_first_token_position(states, tokenizer.eos_token_id)
        first_pad_pos = self._find_first_token_position(states, tokenizer.pad_token_id)
        
        first_termination_pos = torch.min(first_eos_pos, first_pad_pos)
        
        # Set everything after to pad
        pos = torch.arange(states.size(1), device=states.device).unsqueeze(0)
        after_termination_mask = pos > first_termination_pos.unsqueeze(1)
        states[after_termination_mask] = tokenizer.pad_token_id
        
        return states

    def construct_pad_mask(self, states, tokenizer):
        """Constructs a boolean mask where True indicates valid tokens before the first padding token."""
        first_pad_pos = self._find_first_token_position(states, tokenizer.pad_token_id)
        pos = torch.arange(states.size(1), device=states.device).unsqueeze(0)
        return ~(pos >= first_pad_pos.unsqueeze(1))

    def set_reward_for_no_eos(self, states, rewards, tokenizer, pad_mask, penalty=-10.0):
        """Penalizes sequences that don't contain an EOS token by setting the final reward to -1.
        Also marks the last position as EOS in the mask for sequences without EOS."""

        reward_mask = (states == tokenizer.eos_token_id)
        has_eos = reward_mask.any(dim=1)
        
        # For sequences without EOS, set the last valid position to True in the mask
        if (~has_eos).any():
            last_valid_idx = pad_mask.sum(dim=1) - 1  # Get last non-padded position
            batch_idx = torch.arange(states.size(0), device=states.device)
            reward_mask[batch_idx[~has_eos], last_valid_idx[~has_eos]] = True
        
        # Set penalty reward for sequences without EOS
        rewards = rewards.clone()
        rewards[~has_eos] = penalty
        
        return rewards, reward_mask

    def _set_models_to_eval(self, *models):
        """Set all models to evaluation mode."""
        for model in models:
            model.eval()


    def _set_models_to_train(self, *models):
        """Set all models to training mode."""
        for model in models:
            model.train()

    def _generate_and_compute_outputs(self, batch, policy_model, value_model, sft_model, reward_model, temp):
        """Generate sequences and compute all model outputs."""
        full_states, _ = policy_model.generate(
            batch,
            self.max_sequence_length,
            temp,
            max_query_length=self.data.SFT_MAX_QUERY_LENGTH,
        )
        del _

        # NOTE: The attention masks here are critically wrong

        policy_logits, _ = policy_model.forward(full_states)
        sft_policy_logits, _ = sft_model.forward(full_states)
        values = value_model.forward(full_states)
        rewards = reward_model.forward(full_states)
        
        return full_states, policy_logits, sft_policy_logits, values, rewards

    def _truncate_to_response(self, states, values, policy_logits, sft_policy_logits, response_length):
        """Truncate all tensors to the response portion only."""
        states = states[:, -response_length:]
        values = values[:, -response_length:]
        policy_logits = policy_logits[:, -response_length:, :]
        sft_policy_logits = sft_policy_logits[:, -response_length:, :]
        
        return states, values, policy_logits, sft_policy_logits

    def _apply_eos_trick(self, states, rewards, tokenizer):
        """
        Apply EOS trick: truncate and pad after EOS, set -1 reward for missing EOS.
        
        Detail 23.2: Truncate and pad after EOS to ensure valid RM scores
        Detail 23.3: Set -1 reward for sequences without EOS token
        """
        states = self.enforce_padding(states, tokenizer)
        pad_mask = self.construct_pad_mask(states, tokenizer)
        rewards, reward_mask = self.set_reward_for_no_eos(
            states, rewards, tokenizer, pad_mask
        )
        
        return states, pad_mask, rewards, reward_mask
    
    def _create_trajectory(self, states, values, rewards, pad_mask, 
                       reward_mask, full_states, response_length, tokenizer):
        """Create a Trajectory object with base quantities."""
        tj = Trajectory(
            init_state=states.unsqueeze(-1), 
            action_dim=self.action_dim,
            max_sequence_length=response_length,
            tokenizer=tokenizer,
            values=values * pad_mask,
            rewards=rewards,
            pad_mask=pad_mask,
            reward_mask=reward_mask,
        )
        tj.full_states = full_states
        tj.actions=states
        return tj
    
    @profile
    def generate_trajectory(self, 
                            batch, 
                            policy_model, 
                            value_model, 
                            sft_model,
                            temp,
                            reward_model = None):
        with torch.no_grad():

            self._set_models_to_eval(policy_model, value_model, sft_model)

            tokenizer = policy_model.tokenizer

            # Raw model output: generate and forward passes
            full_states, policy_logits, sft_policy_logits, values, rewards = \
                self._generate_and_compute_outputs(
                    batch, policy_model, value_model, sft_model, 
                    reward_model, temp
                )
                        
            response_length = full_states.shape[1] - self.data.SFT_MAX_QUERY_LENGTH
            states, values, policy_logits, sft_policy_logits = \
                self._truncate_to_response(
                    full_states, values, policy_logits, sft_policy_logits, 
                    response_length
                )
            
            # masked_mean(values, ((torch.ones_like(states) * (states == tokenizer.eos_token_id))).bool())
            # masked_mean(values, (~((states == tokenizer.eos_token_id) | (states == tokenizer.pad_token_id))).bool())
            
            # Apply EOS trick and masking
            states, pad_mask, rewards, reward_mask = \
                self._apply_eos_trick(states, rewards, tokenizer)

            # Create trajectory and compute base quantities
            tj = self._create_trajectory(
                states, values, rewards, pad_mask, reward_mask, full_states,
                response_length, tokenizer
            )
        
            total_kl, kl_per_token = tj.compute_kl(policy_logits, sft_policy_logits)
            del sft_policy_logits

            tj.compute_log_probs(policy_logits)
            del policy_logits

            # NOTE: Ordering to reflect the following implementation
            # https://github.com/vwxyzjn/summarize_from_feedback_details/blob/main/summarize_from_feedback_details/ppo.py#L679

            # 1. Apply KL to rewards
            rewards_2d = tj.rewards.unsqueeze(1) * tj.reward_mask
            rewards_2d = (rewards_2d - (self.config.beta * kl_per_token)).masked_fill(~pad_mask, 0)

            pdb.set_trace()

            # 2. Whiten rewards
            if self.config.whiten_rewards:
                rewards_2d = masked_whiten(rewards_2d, pad_mask, shift_mean=False)

            # 3. Compute advantages
            tj.compute_gae(gamma=self.config.gamma, lam=self.config.lam, r=rewards_2d)
            if self.config.whiten_A:
                # NOTE: shift mean here to keep 
                # A > 0 to be "action better than expected", 
                # A < 0 to be "action worse than expected"
                tj.A = masked_whiten(tj.A, pad_mask) 
            
            # 4. Compute returns/rewards-to-go
            tj.compute_R(gamma=self.config.gamma, r=rewards_2d)
                         
        policy_model.train()
        value_model.train()

        return tj