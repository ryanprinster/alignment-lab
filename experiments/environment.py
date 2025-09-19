import atexit
import signal
import torch
import sys
import psutil

from experiments.profiler import profile
import gymnasium as gym



class Environment():
    def __init__(self, config):
        ### Cleanup 
        # Thanks to Claude here:
        self._closed = False
        atexit.register(self.close)  # Normal exit
        signal.signal(signal.SIGTERM, self._cleanup_signal)  # Pod termination
        signal.signal(signal.SIGINT, self._cleanup_signal)   # Ctrl+C
    

        self.config = config
        self.env = gym.make(self.config.gymnasium_env_name)

        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n


    ### Cleanup handling

    def __del__(self):
        self.close()
    
    def _cleanup_signal(self, signum, frame): # Thanks to Claude here
        print(f"Received signal {signum}, cleaning up...")
        self.close()
        sys.exit(0)

    def close(self):
        if not self._closed and hasattr(self, 'env'):
            self.env.close()
            self._closed = True
            print("Env closed")

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
