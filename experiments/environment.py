import atexit
import signal
import torch
import sys
import psutil
import pdb

from experiments.profiler import profile
from experiments.trajectory import Trajectory
from experiments.monitor import detect_nans

import gymnasium as gym

from abc import ABC, abstractmethod
from typing import Any, Tuple
import warnings
import torch.nn.functional as F


def masked_mean(tensor, mask, dim=None, keepdim=False):
    """
    Compute mean of tensor with a boolean mask.
    
    Args:
        tensor: Input tensor
        mask: Boolean mask (True for valid elements, False for masked out)
        dim: Dimension(s) to reduce. If None, reduces all dimensions.
        keepdim: Whether to keep the reduced dimensions
    
    Returns:
        Masked mean
    """
    masked_tensor = torch.where(mask, tensor, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))
    sum_valid = masked_tensor.sum(dim=dim, keepdim=keepdim)
    count_valid = mask.sum(dim=dim, keepdim=keepdim)
    
    # Avoid division by zero
    count_valid = count_valid.clamp(min=1)
    
    return sum_valid / count_valid


def masked_var(tensor, mask, dim=None, keepdim=False, unbiased=True):
    """
    Compute variance of tensor with a boolean mask.
    
    Args:
        tensor: Input tensor
        mask: Boolean mask (True for valid elements, False for masked out)
        dim: Dimension(s) to reduce. If None, reduces all dimensions.
        keepdim: Whether to keep the reduced dimensions
        unbiased: If True, use Bessel's correction (divide by N-1 instead of N)
    
    Returns:
        Masked variance
    """
    # Compute masked mean
    mean = masked_mean(tensor, mask, dim=dim, keepdim=True)
    
    # Compute squared deviations
    squared_diff = (tensor - mean) ** 2
    masked_squared_diff = torch.where(mask, squared_diff, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))
    
    sum_squared_diff = masked_squared_diff.sum(dim=dim, keepdim=keepdim)
    count_valid = mask.sum(dim=dim, keepdim=keepdim)
    
    # Avoid division by zero
    if unbiased:
        count_valid = (count_valid - 1).clamp(min=1)
    else:
        count_valid = count_valid.clamp(min=1)
    
    return sum_squared_diff / count_valid

def masked_softmax(tensor, mask, dim=-1):
    """
    Compute softmax with a boolean mask.
    
    Args:
        tensor: Input tensor (logits)
        mask: Boolean mask (True for valid elements, False for masked out)
        dim: Dimension along which to apply softmax
    
    Returns:
        Masked softmax probabilities (masked positions will be near-zero)
    """
    # Set masked positions to negative infinity
    masked_tensor = tensor.masked_fill(~mask, float('-inf'))

    # Apply softmax
    return F.softmax(masked_tensor, dim=dim) * mask

def masked_log_softmax(tensor, mask, dim=-1, mask_value=-1e9):
    """
    Compute log_softmax with a boolean mask.
    
    Args:
        tensor: Input tensor (logits)
        mask: Boolean mask (True for valid elements, False for masked out)
        dim: Dimension along which to apply log_softmax
    
    Returns:
        Masked log_softmax (masked positions will be -inf)
    """
    masked_tensor = tensor.masked_fill(~mask, float('-inf'))

    log_probs = F.log_softmax(masked_tensor, dim=dim)

    log_probs = log_probs.masked_fill(~mask, mask_value)
    
    return log_probs

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

    @detect_nans
    def rewards_with_kl_penalty(self, rewards, policy_logits, policies, sft_policy_logits, pad_mask, reward_mask):
        """
        Averages kl over action_space = vocab_size space, and over sequence space.
        """
        pad_mask_3d = pad_mask.unsqueeze(2)

        log_P = masked_log_softmax(policy_logits, pad_mask_3d, mask_value=0, dim=-1).masked_fill(~pad_mask_3d, 0)
        # P = policies.masked_fill(~pad_mask_3d, 0)
        P = torch.exp(log_P).masked_fill(~pad_mask_3d, 0)
        log_Q = sft = masked_log_softmax(sft_policy_logits, pad_mask_3d, mask_value=0, dim=-1).masked_fill(~pad_mask_3d, 0)
        kl_div = torch.sum((P * (log_P - log_Q)), pad_mask_3d, dim=(1,2))

        pdb.set_trace() 
        # TODO: verify masked_log_softmax works as intendend
        has_reward_mask = (rewards != 0)
        kl_div = torch.ones_like(rewards) * kl_div.unsqueeze(1)

        return rewards - self.config.beta * (kl_div * has_reward_mask)

    def set_pad_after_eos(self, states, tokenizer):
        eos_mask = (states == tokenizer.eos_token_id)
        first_eos_pos = torch.where(
            eos_mask.any(dim=1),
            eos_mask.int().argmax(dim=1),
            torch.full((states.size(0),), states.size(1), device=states.device)
        )

        pos = torch.arange(states.size(1), device=states.device).unsqueeze(0)
        after_eos_mask = pos > first_eos_pos.unsqueeze(1)

        states[after_eos_mask] = tokenizer.pad_token_id
        return states
    
    def set_reward_for_no_eos(self, states, rewards):
        """ Assumes rewards as been set to all zeros for a given trajectory if no eos token"""
        # TODO: remove above assumption
        all_zero_no_eos = (rewards == 0).all(dim=1)
        rewards[all_zero_no_eos, -1] = -1
        return rewards

    # Taken from https://arxiv.org/pdf/2403.17031 then modified to add masking
    def whiten(self, values, mask, shift_mean=True):
        mean, var = masked_mean(values, mask), masked_var(values, mask, unbiased=False)
        whitened = (values - mean) * torch.rsqrt(var + 1e-8)
        if not shift_mean:
            whitened += mean
        return whitened * mask
    


    @profile
    def generate_trajectory(self, 
                            batch, 
                            policy_model, 
                            value_model, 
                            sft_model,
                            temp,
                            reward_model = None):
        with torch.no_grad():

            torch.backends.cudnn.deterministic = True
            torch.manual_seed(42)

            policy_model.eval()
            value_model.eval()
            sft_model.eval()

            tokenizer = policy_model.tokenizer

            states, policy_logits = policy_model.generate(
                batch,
                self.max_sequence_length,
                temp,
            )

            _sft_tokens, sft_policy_logits = sft_model.generate(
                batch,
                self.max_sequence_length,
                temp,
            )

            pdb.set_trace()

            policy_response_length = states.shape[1] - self.data.SFT_MAX_QUERY_LENGTH
            _sft_reponse_length = _sft_tokens.shape[1] - self.data.SFT_MAX_QUERY_LENGTH

            if policy_response_length != _sft_reponse_length:
                raise ValueError(f"policy response length {policy_response_length} does not sft response length {_sft_reponse_length}")

            values = value_model.forward(states, batch['attention_mask'])

            rewards = reward_model.forward(states, batch['attention_mask'])
            

            states = states[:,-policy_response_length:]
            # Detail 23.2 (PPO Training -> “EOS trick” to ensure scores from the RM is valid ->  truncate and pad after eos)
            states = self.set_pad_after_eos(states, tokenizer)
    
            mask = self.construct_mask(states, tokenizer)

            values = values[:,-policy_response_length:] * mask

            policy_logits = policy_logits[:,-policy_response_length:,:] # don't mask yet
            sft_policy_logits = sft_policy_logits[:,-_sft_reponse_length:,:] # don't mask yet
            policies = masked_softmax(policy_logits, mask.unsqueeze(2), dim=-1)

            rewards = rewards[:,-policy_response_length:]
            reward_mask = (states == tokenizer.eos_token_id)
            
            rewards = rewards * reward_mask
            rewards = self.whiten(rewards, reward_mask, shift_mean=False)

            """
            reward operations:
                0. modify length of reward vector
                1. get reward mask 
                2. use reward mask to get reward @ eos tokens
            reward @ eos tokens
            whiten over batch <- note that this could come after KL penalty too, following implementation tho

            

            """
            
            # Detail 12 (RM Training -> Extract reward from the EOS token)
        
            rewards = self.rewards_with_kl_penalty(rewards=rewards, 
                                                   policy_logits=policy_logits, 
                                                   policies=policies, 
                                                   sft_policy_logits=sft_policy_logits, 
                                                   mask=mask, 
                                                   reward_mask=reward_mask)
            # Detail 23.3 (PPO Training -> “EOS trick” to ensure scores from the RM is valid -> set -1 reward for no eos token)
            rewards = self.set_reward_for_no_eos(states, rewards)

            tj = Trajectory(init_state=states.unsqueeze(-1), 
                    action_dim=self.action_dim,
                    max_sequence_length=policy_response_length,
                    pad_token_id=self.data.tokenizer.pad_token_id,
                    policies=policies,
                    values=values,
                    rewards=rewards)


            tj.compute_gae(gamma=self.config.gamma, lam=self.config.lam)
            tj.compute_R(gamma=self.config.gamma)
            tj.actions = states
            tj.compute_probs()


            policy_model.train()
            value_model.train()

            return tj