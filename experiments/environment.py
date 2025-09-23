import atexit
import signal
import torch
import sys
import psutil

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
        self.obs_dim = self.data.__class__.SFT_MAX_INPUT_LENGTH
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
                            temp):


        # tj = Trajectory(init_state=tokenized_prompt, 
        #                 obs_dim=self.obs_dim, 
        #                 action_dim=self.action_dim,
        #                 max_length=max_sequence_length)

        states, policies = policy_model.generate(
            batch,
            self.data.__class__.SFT_MAX_INPUT_LENGTH,
            temp,
        )

        values = value_model.forward(
            batch
        )

        print(f"Shapes {states.shape}, {policies.shape}, {values.shape}")
        assert(False)
        tj.compute_gae(gamma=self.config.gamma, lam=self.config.lam)
        tj.compute_R(gamma=self.config.gamma)

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