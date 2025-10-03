from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pdb
import warnings


@dataclass
class Trajectory():
    BATCH_DIM, TIME_DIM = 0, 1
    TORCH_FIELDS = ['states', 'actions', 'rewards', 'policies', 'values', 'probs', 'R', 'A'] 

    def __init__(self, init_state, action_dim, max_sequence_length, values, policies, rewards, pad_token_id=None):
        """
        init_state - tensor of shape (batch_size, max_sequence_length, obs_dim)
            --> Assume for now that tensors are padded to be of max_sequence_length
            --> Assume for now that we will not be adding to the trajectories after creation, 
                hence values, policies, rewards are required on init
        """
        
        ### Verify base input
        init_state = torch.from_numpy(init_state) if isinstance(init_state, np.ndarray) else init_state

        if init_state.dim() != 3:
            raise ValueError("init_state.dim() should be 3")
        if init_state.shape[1] > max_sequence_length:
            # Should I require max sequence length? --> Will be needed when making trajectories that we can add to after
            raise ValueError("given trajectory length longer than max_sequence_length")

        device = init_state.device
        self.batch_size = batch_size = init_state.shape[0]
        self.max_sequence_length = max_sequence_length
        self.obs_dim = init_state.shape[2]

        ### Handle padding
        # TODO: double cehck this
        if pad_token_id is not None:
            # Detect and create a mask
            if self.obs_dim == 1: 
                self._mask = ~(init_state.squeeze(-1) == pad_token_id)
                self._mask_3d = ~(init_state == pad_token_id)

                # is_pad = (init_state.squeeze(-1) == pad_token_id)
                # first_pad_pos = torch.argmax(is_pad.int(), dim=1)
                # has_padding = is_pad.any(dim=1)
                # self._length = torch.where(has_padding, first_pad_pos, init_state.shape[1])
            else:
                raise ValueError("Padding detection only supported for obs_dim=1")
        else:
            # Assume no padding - all sequences use full length
            warnings.warn(f"No pad_token_id provided. Assuming all sequences are valid")
            self._mask = torch.ones((batch_size, max_sequence_length), dtype=torch.bool)


        # Each different traj can have a different length, after init
        # self._length = torch.full((batch_size,), init_state.shape[1], dtype=torch.long) 

        # pdb.set_trace()
        self._states = init_state * self._mask_3d
        self._actions = torch.zeros((batch_size, max_sequence_length), device=device) 
        self._rewards = rewards * self._mask if rewards is not None else \
            torch.zeros((batch_size, max_sequence_length), device=device)
        self._policies = policies * self._mask_3d if policies is not None else \
            torch.zeros((batch_size, max_sequence_length, action_dim), device=device)
        self._values = values * self._mask if values is not None else \
            torch.zeros((batch_size, max_sequence_length), device=device)
        self._probs = torch.zeros((batch_size, max_sequence_length), device=device)
        self._R = torch.zeros((batch_size, max_sequence_length), device=device)
        self._A = torch.zeros((batch_size, max_sequence_length), device=device)
    

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
        if torch.all(self.rewards == 0).item():
            raise ValueError("rewards is not set, set non-zero rewards attribute first")

        time_dim = Trajectory.TIME_DIM

        r = self.rewards
        r_rev = torch.flip(r, dims=[time_dim])
        discounts_rev = torch.ones_like(self.rewards) * gamma * self._mask.flip(dims=[time_dim])
        discounts_rev = torch.cumprod(discounts_rev,dim=time_dim) / gamma
        R_rev = torch.cumsum(discounts_rev * r_rev, dim=time_dim)
        self._R = torch.flip(R_rev, dims=[time_dim])
        return self.R
    
    def compute_gae(self, gamma, lam):
        
        if torch.all(self.rewards == 0).item():
            raise ValueError("rewards is not set, set non-zero rewards attribute first")
        if torch.all(self.values == 0).item():
            raise ValueError("values is not set, set non-zero values attribute first")
               
        time_dim = Trajectory.TIME_DIM

        # 0. Get V and r
        # len(V) = T+1
        # len(r) = T
        V = self.values.detach()
        r = self.rewards

        # 1. Compute delta_t (TD Error)
        V_next = torch.cat([V[:,1:], torch.zeros(self.batch_size, 1)], dim=time_dim) # Assumes V(s_{T+1}) = 0 TODO: is this a good assumption for LLMs
        TD_error = r + gamma * V_next - V

        # 2. Get discounts 
        TD_rev = TD_error.flip(dims=[time_dim]) 
        discounts_rev = torch.ones(r.size()) * lam * gamma * self._mask.flip(dims=[time_dim])
        discounts_rev = torch.cumprod(discounts_rev, dim=time_dim) / (lam * gamma)
        
        # 3. Calculate GAE via cumulative sum in reverse
        A_rev = torch.cumsum(discounts_rev * TD_rev, dim=time_dim)
        self._A = torch.flip(A_rev, dims=[time_dim])

        return self.A
    
    def compute_probs(self):
        if torch.all(self.actions == 0).item():
            raise ValueError("actions is not set, set non-zero actions attribute first")
        if torch.all(self.policies == 0).item():
            raise ValueError("policies is not set, set non-zero policies attribute first")
        
        self._probs = torch.gather(self.policies, dim=-1, index=self.actions.long().unsqueeze(-1)).squeeze(-1)
        return self.probs
    
    def whiten_rewards(self):
        self._rewards = self._whiten(self._rewards)
        return self._rewards

    def whiten_advantages(self):
        self._A = self._whiten(self._A)
        return self._A

    # Taken from https://arxiv.org/pdf/2403.17031
    def _whiten(values, shift_mean=True):
        mean, var = torch.mean(values), torch.var(values, unbiased=False)
        whitened = (values - mean) * torch.rsqrt(var + 1e-8)
        if not shift_mean:
            whitened += mean
        return whitened
    
    # def __len__(self):
    #     # TODO: Decide what exactly length should return here
    #     return self.batch_size
    
    # def __getitem__(self, idx):
    #     #TODO: make this work with both PPO cartpole and PPO for RLHF
    #     return self.states[idx,:,:], \
    #         self.actions[idx,:], \
    #         self.rewards[idx,:], \
    #         self.policies[idx,:,:], \
    #         self.values[idx,:], \
    #         self.probs[idx,:], \
    #         self.R[idx,:], \
    #         self.A[idx,:]

    # TODO: Review how returning whole tensor will interact with everything else
    @property
    def states(self):
        return self._states
    
    @property
    def actions(self):
        return self._actions
    
    @actions.setter
    def actions(self, new_actions):
        new_actions = torch.as_tensor(new_actions, device=self._actions.device, dtype=self._actions.dtype)
        if new_actions.shape != self._actions.shape:
            raise ValueError(f"Actions shape {new_actions.shape} doesn't match expected {self._actions.shape}")
        self._actions = new_actions * self._mask

    @property
    def rewards(self):
        return self._rewards
    
    @property
    def policies(self):
        return self._policies
    
    @property
    def values(self):
        return self._values
    
    @property
    def probs(self):
        # TODO: need to compute these
        return self._probs
    
    @property
    def R(self):
        return self._R

    @property
    def A(self):
        return self._A
    

    # def add_step(self, state, action, reward, policy, value, prob):
    #     if self.batch_size != 1:
    #         raise ValueError("add_step only works with batch_size=1")
    #         # Simplify the logic for this case, for now
    #     if self._length >= self._max_sequence_length:
    #         raise ValueError("Trajectory exceeds max length")

    #     i = self._length
    #     self._states[i+1] = torch.as_tensor(state)
    #     self._actions[i] = action
    #     self._rewards[i] = reward
    #     self._policies[i] = torch.as_tensor(policy)
    #     self._values[i] = value
    #     self._probs[i] = policy[action]

    #     self._length += 1

    #### TODO: Build this functionality in for a more flexible API
    # def _init_state(self, init_state):
    #     """
    #     For sequence model rl:
    #     (batch_size=1,init_state_seq_len=t,obs_dim=1) --squeeze--> (t)
    #     (batch_size=1,init_state_seq_len=t,obs_dim=o) --squeeze--> (t, o)
    #     (batch_size=m,init_state_seq_len=t,obs_dim=1) --squeeze--> (m, t)
    #     (batch_size=m,init_state_seq_len=t,obs_dim=1) --squeeze--> (m, t, o)

    #     For non-sequence model rl: (assume observation dimension)
    #     (batch_size=1,init_state_seq_len=1,obs_dim=1) --squeeze--> 1 or scalar #unlikely to have a non-sequence model with obs dim = 1
    #     (batch_size=m,init_state_seq_len=1,obs_dim=1) --squeeze--> (m) #unlikely to have a non-sequence model with obs dim = 1
    #     (batch_size=1,init_state_seq_len=1,obs_dim=o) --squeeze--> (o)
    #     (batch_size=m,init_state_seq_len=1,obs_dim=o) --squeeze--> (m, o)
    #     """
    #     init_state = torch.as_tensor(init_state)

    #     # Determine what the dimensions likely represent based on our target shapes
    #     if init_state.dim() == 1:  # (seq_len,) - common case for single sequence
    #         if init_state.shape[0] == self.obs_dim:
    #             print("Assuming input is of shape (obs_dim)")
    #             init_state = init_state.unsqueeze(0).unsqueeze(1)  # -> (1, 1, obs_dim)
    #         elif init_state.shape[0] == self.max_sequence_length:
    #             print("Assuming input is of shape (max_sequence_length)")
    #             init_state = init_state.unsqueeze(0).unsqueeze(-1)  # -> (1, seq_len, 1)
    #         elif init_state.shape[0] == self.batch_size:
    #             print("Assuming input is of shape (batch_size)")
    #             init_state = init_state.unsqueeze(1).unsqueeze(-1)  # -> (batch_size, 1, 1)
    #         else:
    #             raise ValueError("Unable to infer init_state dimension meanings. " \
    #                             "Check input shape, or give an unsqueezed tensor of shape (batch_size, max_sequence_length, obs_dim)")
    #     elif init_state.dim() == 2:
    #         # Could be (seq_len, obs_dim) or (batch_size, seq_len), or (batch_size, obs_dim)
    #         if init_state.shape == (self.max_sequence_length, self.obs_dim):
    #             print("Assuming input is of shape (max_sequence_length, obs_dim)")
    #             init_state = init_state.unsqueeze(0)  # -> (1, seq_len, obs_dim)
    #         elif init_state.shape == (self.batch_size, self.max_sequence_length):
    #             print("Assuming input is of shape (batch_size, max_sequence_length)")
    #             init_state = init_state.unsqueeze(-1) # -> (batch_size, seq_len, 1)
    #         elif init_state.shape == (self.batch_size, self.obs_dim):
    #             print("Assuming input is of shape (batch_size, obs_dim)")
    #             init_state = init_state.unsqueeze(1) # -> (batch_size, 1, obs_dim)
    #         else:
    #             raise ValueError("Unable to infer init_state dimension meanings. " \
    #                             "Check input shape, or give an unsqueezed tensor of shape (batch_size, max_sequence_length, obs_dim)")
    #     elif init_state.dim() == 3:
    #         if init_state.shape != (self.batch_size, self.max_sequence_length, self.obs_dim):
    #             raise ValueError("Unable to infer init_state dimension meanings. " \
    #                             "Check input shape, or give an unsqueezed tensor of shape (batch_size, max_sequence_length, obs_dim)")
    #     else:
    #         raise ValueError("Unable to infer init_state dimension meanings. " \
    #                             "Check input shape, or give an unsqueezed tensor of shape (batch_size, max_sequence_length, obs_dim)")

    #     # TODO: Now actually init state
    #     return init_state


class TrajectorySet(Dataset):
    def __init__(self, trajectory: Trajectory):
        self._tjs = trajectory

    def __len__(self):
        return self._tjs.batch_size
    
    def __getitem__(self, idx):
        return self._tjs.states[idx,:,:], \
            self._tjs.actions[idx,:], \
            self._tjs.rewards[idx,:], \
            self._tjs.policies[idx,:,:], \
            self._tjs.values[idx,:], \
            self._tjs.probs[idx,:], \
            self._tjs.R[idx,:], \
            self._tjs.A[idx,:]

# TODO: Make another TrajectorySet which shuffles the time dimension 
# into the batch dimension for cartpole or non-sequence environments