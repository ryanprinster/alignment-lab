import atexit
import signal
import torch
import sys
import psutil
import pdb

from experiments.profiler import profile
from experiments.trajectory import Trajectory
from experiments.monitor import detect_nans
from experiments.util import masked_mean, masked_log_softmax, masked_whiten

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



    @detect_nans
    @profile
    def rewards_with_kl_penalty(self, rewards, policy_logits, sft_policy_logits, pad_mask, reward_mask):
        """
        Averages kl over action_space = vocab_size space, and over sequence space.
        """
        # NOTE: KL could be computed in different ways. 
        # - KL of the full distribution, on the top_p or top_k, or just actions taken.
        # - KL could be averaged or summed across the sequence dimension. 
        # This implementation currently takes KL over top_p=0.9, and summed across the policy dim but averaged across the sequence dim.


        log_P = masked_log_softmax(policy_logits, pad_mask.unsqueeze(2), mask_value=0, dim=-1).masked_fill(~pad_mask.unsqueeze(2), 0)
        P = torch.exp(log_P).masked_fill(~pad_mask.unsqueeze(2), 0)
        log_Q = sft = masked_log_softmax(sft_policy_logits, pad_mask.unsqueeze(2), mask_value=0, dim=-1).masked_fill(~pad_mask.unsqueeze(2), 0)

        kl_div = torch.sum((P * (log_P - log_Q)).masked_fill(~pad_mask.unsqueeze(2), 0), dim=-1)
        kl_div = masked_mean(kl_div, pad_mask, dim=-1)
        # The math technically says to sum over both dims, but averaging over time makes sense for initial stability
        # and is also tunable by beta.

        kl_div = torch.ones_like(rewards) * kl_div.unsqueeze(1)

        return rewards - self.config.beta * kl_div.masked_fill(~reward_mask, 0)

    def construct_mask(self, states, tokenizer):
        
        pad_mask = (states == tokenizer.pad_token_id)

        # In the unlikely case there are random pad tokens with other tokens proceeding it,
        # set all following tokens to pad token. This will effectively end the sequence and
        # penalize the model.
        first_pad_pos = torch.where(
            pad_mask.any(dim=1),
            pad_mask.int().argmax(dim=1),
            torch.full((states.size(0),), states.size(1), device=states.device)
        )

        pos = torch.arange(states.size(1), device=states.device).unsqueeze(0)
        after_pad_mask = ~(pos >= first_pad_pos.unsqueeze(1))

        return after_pad_mask

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
    
    def construct_reward_mask(self, states, tokenizer):
        return (states == tokenizer.eos_token_id)

    
    def set_reward_for_no_eos(self, eos_mask, rewards):
        """Penalizes sequences that don't contain an EOS token by setting the final reward to -1."""
        has_no_eos = ~eos_mask.any(dim=1)
        rewards[has_no_eos, -1] = -1
        return rewards

    # # Taken from https://arxiv.org/pdf/2403.17031 then modified to add masking
    # @profile
    # def whiten(self, values, mask, shift_mean=True):
    #     mean, var = masked_mean(values, mask), masked_var(values, mask, unbiased=False)
    #     whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    #     if not shift_mean:
    #         whitened += mean
    #     return whitened * mask
    


    @profile
    def generate_trajectory(self, 
                            batch, 
                            policy_model, 
                            value_model, 
                            sft_model,
                            temp,
                            reward_model = None):
        with torch.no_grad():

            policy_model.eval()
            value_model.eval()
            sft_model.eval()

            tokenizer = policy_model.tokenizer

            states, policy_logits = policy_model.generate(
                batch,
                self.max_sequence_length,
                temp,
                max_query_length=self.data.SFT_MAX_QUERY_LENGTH,
            )
            sft_policy_logits, _ = sft_model.forward(
                states
            )

            values = value_model.forward(states, batch['attention_mask'])
            rewards = reward_model.forward(states, batch['attention_mask'])

            respose_length = states.shape[1] - self.data.SFT_MAX_QUERY_LENGTH

            states = states[:,-respose_length:]
            values = values[:,-respose_length:]
            policy_logits = policy_logits[:,-respose_length:,:] # don't mask yet
            sft_policy_logits = sft_policy_logits[:,-respose_length:,:] # don't mask yet
            rewards = rewards[:,-respose_length:]

            # Detail 23.2 (PPO Training -> “EOS trick” to ensure scores from the RM is valid ->  truncate and pad after eos)
            states = self.enforce_padding(states, tokenizer)
            pad_mask = self.construct_pad_mask(states, tokenizer)
        
            # Detail 12 (RM Training -> Extract reward from the EOS token)
            reward_mask = self.construct_reward_mask(states, tokenizer)

            # Detail 23.3 (PPO Training -> “EOS trick” to ensure scores from the RM is valid -> set -1 reward for no eos token)
            rewards = self.set_reward_for_no_eos(reward_mask, rewards)
            # NOTE: whitened before computing kl to follow https://arxiv.org/pdf/2403.17031

            tj = Trajectory(init_state=states.unsqueeze(-1), 
                    action_dim=self.action_dim,
                    max_sequence_length=respose_length,
                    tokenizer=tokenizer,
                    values=values * pad_mask,
                    rewards=rewards,
                    pad_mask=pad_mask,
                    reward_mask=reward_mask)
        
            tj.compute_kl(policy_logits, sft_policy_logits)
            del sft_policy_logits

            tj.actions = states
            tj.compute_probs(policy_logits)
            del policy_logits

            tj.compute_gae(gamma=self.config.gamma, lam=self.config.lam)
            
            # rewards = whiten(r) - beta * kl
            transformed_rewards = masked_whiten(rewards, reward_mask, shift_mean=False)
            transformed_rewards -= self.config.beta * tj.kl
            tj.compute_R(gamma=self.config.gamma, r=transformed_rewards)
                         
        policy_model.train()
        value_model.train()

        return tj