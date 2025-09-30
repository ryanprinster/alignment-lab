import atexit
import signal
import torch
import sys
import psutil
import pdb

from experiments.profiler import profile
from experiments.trajectory import Trajectory
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

    def rewards_with_kl_penalty(self, rewards, policy, policy_sft):
        # TODO: condense rewards to one value
        pdb.set_trace()
        return rewards - self.config.beta * torch.nn.functional.kl_div(policy, policy_sft)

    def filter_no_eos(self, states, tokenizer, *tensors):
        has_eos = (states == tokenizer.eos_token_id).any(dim=1)
        return (states[has_eos],) + tuple(t[has_eos] for t in tensors)

    @profile
    def generate_trajectory(self, 
                            batch, 
                            policy_model, 
                            value_model, 
                            sft_model,
                            temp,
                            reward_model = None):
        tokenizer = policy_model.tokenizer

        states, policies = policy_model.generate(
            batch,
            self.max_sequence_length,
            temp,
        )

        _, sft_policies = sft_model.generate(
            batch,
            self.max_sequence_length,
            temp,
        )

        values = value_model.forward(
            {'input_ids': states, 'attention_mask': batch['attention_mask']}
        )

        if None in batch['rm_score']:
            # TODO: remove this, this is here for quick iteration testing
            rewards = torch.ones_like(values)

            # rewards = reward_model.forward(
            #     {'input_ids': states, 'attention_mask': batch['attention_mask']}
            # )

        else:
            rewards = batch['rm_score']

        # Filter out any trajectories that did not generate an EOS token as part of
        # Detail 12 (RM Training -> Extract reward from the EOS token)
        # states, policies, values, rewards, sft_policies = self.filter_no_eos(
        #     states, tokenizer, policies, values, rewards, sft_policies
        # )

        states = states[:,-self.max_response_length:]
        policies = policies[:,-self.max_response_length:,:]
        values = values[:,-self.max_response_length:]
        # Detail 12 (RM Training -> Extract reward from the EOS token)
        rewards = rewards[:,-self.max_response_length:] * (states == tokenizer.eos_token_id) # create mask to get eos token rewards
        rewards = self.rewards_with_kl_penalty(rewards, policies, sft_policies)

        tj = Trajectory(init_state=states.unsqueeze(-1), 
                action_dim=self.action_dim,
                max_sequence_length=self.max_response_length,
                pad_token_id=self.data.tokenizer.pad_token_id,
                policies=policies,
                values=values,
                rewards=rewards)
        
        tj.compute_gae(gamma=self.config.gamma, lam=self.config.lam)
        tj.compute_R(gamma=self.config.gamma)
        tj.actions = states
        
        tj.compute_probs()
        pdb.set_trace()

        return tj