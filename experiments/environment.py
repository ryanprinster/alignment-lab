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
    def rewards_with_kl_penalty(self, rewards, policy_logits, policies, sft_policy_logits):
        """
        Averages kl over action_space = vocab_size space, and over sequence space.
        """
        log_P= torch.nn.functional.log_softmax(policy_logits, dim=-1)
        P = policies
        log_Q = sft = torch.nn.functional.log_softmax(sft_policy_logits, dim=-1)
        kl_div = (P * (log_P - log_Q)).mean(dim=(1,2))

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

    @profile
    def generate_trajectory(self, 
                            batch, 
                            policy_model, 
                            value_model, 
                            sft_model,
                            temp,
                            reward_model = None):
        with torch.no_grad():
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

            policy_response_length = states.shape[1] - self.data.SFT_MAX_QUERY_LENGTH
            _sft_reponse_length = _sft_tokens.shape[1] - self.data.SFT_MAX_QUERY_LENGTH


            if policy_response_length != _sft_reponse_length:
                raise ValueError(f"policy response length {policy_response_length} does not sft response length {_sft_reponse_length}")

            values = value_model.forward(states, batch['attention_mask'])

            rewards = reward_model.forward(states, batch['attention_mask'])
            
            states = states[:,-policy_response_length:]
            # Detail 23.2 (PPO Training -> “EOS trick” to ensure scores from the RM is valid ->  truncate and pad after eos)
            states = self.set_pad_after_eos(states, tokenizer)

            values = values[:,-policy_response_length:]

            policy_logits = policy_logits[:,-policy_response_length:,:]
            policies = torch.softmax(policy_logits, dim=-1)

            # Detail 12 (RM Training -> Extract reward from the EOS token)
            rewards = rewards[:,-policy_response_length:] * (states == tokenizer.eos_token_id)
            rewards = self.rewards_with_kl_penalty(rewards, policy_logits, policies, sft_policy_logits)
            # Detail 23.3 (PPO Training -> “EOS trick” to ensure scores from the RM is valid -> set -1 reward for no eos token)
            rewards = self.set_reward_for_no_eos(states, rewards)

            tj = Trajectory(init_state=states.unsqueeze(-1), 
                    action_dim=self.action_dim,
                    max_sequence_length=policy_response_length,
                    pad_token_id=self.data.tokenizer.pad_token_id,
                    policies=policies,
                    values=values,
                    rewards=rewards)

            pdb.set_trace()
            tj.whiten_rewards()
            tj.whiten_advantages()

            tj.compute_gae(gamma=self.config.gamma, lam=self.config.lam)
            tj.compute_R(gamma=self.config.gamma)
            tj.actions = states
            tj.compute_probs()

            return tj