import atexit
import signal
import torch
import sys
import psutil
import pdb

from experiments.profiler import profile
from experiments.trajectory import Trajectory, BatchTrajectory
import gymnasium as gym

from abc import ABC, abstractmethod
from typing import Any, Tuple

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
    
    def reset(self, prompt): # --> observation, info 
        # Need a prompt to reset?
        pass

    @profile
    def step(self, action): #  --> observation, reward, terminated, truncated, info
        pass
        # model.generate_autoregressive
        # reward = None (reward to be computed after)
        # if most recent token is EOS, terminated = True
        # if most recent token brings us to max token length, truncated = True

    @profile
    def generate_trajectory(self, 
                            batch, 
                            policy_model, 
                            value_model, 
                            reward_model,
                            temp):
        

        states, policies = policy_model.generate(
            batch,
            self.max_sequence_length,
            temp,
        )

        # TODO: policy_model.generate only gives policies for generated tokens
        # hence 

        values = value_model.forward(
            {'input_ids': states, 'attention_mask': batch['attention_mask']}
        )

        rewards = reward_model.forward(
            {'input_ids': states, 'attention_mask': batch['attention_mask']}
        )

        # Slice to get just the response data
        # TODO: can I do this without creating additional computation or memory?
        states = states[:,-self.max_response_length:]
        policies = policies[:,-self.max_response_length:,:]
        values = values[:,-self.max_response_length:]
        rewards = rewards[:,-self.max_response_length:]

        print(f"Shapes {states.shape}, {policies.shape}, {values.shape}")
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

        # TODO: remove computation of R and GAE on prompt portion?
        # Could just pass in the sliced return values from the models
        
        tj.compute_probs()
        pdb.set_trace()

        return tj
    
    @profile
    def generate_trajectories(self, 
                                tokenized_prompts, 
                                policy_model, 
                                value_model, 
                                max_response_length,
                                N=None, 
                                M=None):
        # Parallelism is simulated for now
        # TODO: Review batching logic

        batched_tj = BatchTrajectory([self._generate_trajectory(tokenized_prompts, 
                                                                policy_model, 
                                                                value_model, 
                                                                max_response_length) 
                                        for _ in range(N or self.config.N)])
        return batched_tj