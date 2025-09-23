import atexit
import signal
import torch
import sys
import psutil

from experiments.profiler import profile
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

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
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

    @profile
    def step(self, action):
        return self.env.step(action)


class RLHFEnvironment(BaseEnvironment):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

    def _close(self):
        pass
    
    def reset(self): # --> observation, info 
        # Need a prompt to reset?
        pass

    @profile
    def step(self, action): #  --> observation, reward, terminated, truncated, info
        pass
        # model.generate_autoregressive
        # reward = None (reward to be computed after)
        # if most recent token is EOS, terminated = True
        # if most recent token brings us to max token length, truncated = True