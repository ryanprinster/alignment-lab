from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


@dataclass
class Trajectory():
    TIME_DIM = 0 
    TORCH_FIELDS = ['states', 'actions', 'rewards', 'policies', 'values', 'probs', 'R', 'A'] 

    def __init__(self, init_state, obs_dim, action_dim, max_length=500):
        self._states = torch.zeros((max_length+1, obs_dim))
        self._states[0] = torch.from_numpy(init_state) if isinstance(init_state, np.ndarray) else init_state
        self._actions = torch.zeros((max_length))
        self._rewards = torch.zeros((max_length))
        self._policies = torch.zeros((max_length, action_dim))
        self._values = torch.zeros((max_length))
        self._probs = torch.zeros((max_length))
        self._R = torch.zeros((max_length))
        self._A = torch.zeros((max_length))

        self._length = 0
        self._max_length=max_length

    def add_step(self, state, action, reward, policy, value, prob):
        if self._length >= self._max_length:
            raise ValueError("Trajectory exceeds max length")

        i = self._length
        self._states[i+1] = torch.as_tensor(state)
        self._actions[i] = action
        self._rewards[i] = reward
        self._policies[i] = torch.as_tensor(policy)
        self._values[i] = value
        self._probs[i] = policy[action]

        self._length += 1
    
    def get_trajectory(self):
        return (
            self.states,
            self.actions,
            self.rewards,
            self.policies,
            self.values,
            self.probs,
            self.R,
            self.A
        )

    # Calculate discounted rewards, aka rewards to go
    def compute_R(self, gamma):
        time_dim = Trajectory.TIME_DIM
        r_rev = torch.flip(self.rewards, dims=[time_dim])
        discounts_rev = torch.cumprod(torch.ones_like(self.rewards) * gamma,dim=time_dim) / gamma
        R_rev = torch.cumsum(discounts_rev * r_rev, dim=time_dim)
        self._R = torch.flip(R_rev, dims=[time_dim])
        return self.R
    
    def compute_gae(self, gamma, lam):
        time_dim = Trajectory.TIME_DIM

        # 0. Get V and r
        # len(V) = T+1
        # len(r) = T
        V = self.values.detach()
        r = self.rewards

        # 1. Compute delta_t (TD Error)
        V_next = torch.cat([V[1:], torch.tensor([0])], dim=time_dim) # Would need to change to accommodate chanding input dims / batch dim
        TD_error = r + gamma * V_next - V

        # 2. Get discounts 
        TD_rev = TD_error.flip(dims=[time_dim]) 
        discounts_rev = torch.cumprod(torch.ones(r.size()) * lam * gamma,dim=time_dim)
        
        # 2. Calculate GAE via cumulative sum in reverse
        A_rev = torch.cumsum(discounts_rev * TD_rev, dim=time_dim)
        self._A = torch.flip(TD_rev, dims=[time_dim])

        return self.A
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        return self.states[idx], \
            self.actions[idx], \
            self.rewards[idx], \
            self.policies[idx], \
            self.values[idx], \
            self.probs[idx], \
            self.R[idx], \
            self.A[idx]

    @property
    def states(self):
        return self._states[:self._length]
    
    @property
    def actions(self):
        return self._actions[:self._length]
    
    @property
    def rewards(self):
        return self._rewards[:self._length]
    
    @property
    def policies(self):
        return self._policies[:self._length]
    
    @property
    def values(self):
        return self._values[:self._length]
    
    @property
    def probs(self):
        return self._probs[:self._length]
    
    @property
    def R(self):
        return self._R[:self._length]

    @property
    def A(self):
        return self._A[:self._length]
    
@dataclass
class BatchTrajectory(Dataset):
    def __init__(self, trajectories):
        self._tjs = trajectories
        self._batch()

    def _batch(self):
        for field in Trajectory.TORCH_FIELDS:
            setattr(self, "_" + field, torch.cat([getattr(tj, field) for tj in self._tjs], dim=Trajectory.TIME_DIM))

    def __len__(self):
        return self._rewards.shape[Trajectory.TIME_DIM]
    
    def __getitem__(self, idx):
        return self._states[idx], \
            self._actions[idx], \
            self._rewards[idx], \
            self._policies[idx], \
            self._values[idx], \
            self._probs[idx], \
            self._R[idx], \
            self._A[idx]
