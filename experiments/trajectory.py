from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pdb
import warnings
from experiments.profiler import profile
from experiments.util import masked_mean, masked_var, masked_softmax, masked_log_softmax


@dataclass
class Trajectory():
    BATCH_DIM, TIME_DIM = 0, 1
    TORCH_FIELDS = ['states', 'actions', 'rewards', 'policies', 'values', 'log_probs', 'R', 'A', 'kl', 'pad_mask', 'reward_mask'] 

    @profile
    def __init__(self, 
                 batch_size
                #  init_state, 
                #  action_dim, 
                #  max_sequence_length, 
                #  values, 
                #  rewards, 
                #  policies = None, 
                #  tokenizer=None,
                #  pad_mask=None,
                #  reward_mask=None,
                 ):
        
        self.batch_size = batch_size
        """
        init_state - tensor of shape (batch_size, max_sequence_length, obs_dim)
            --> Assume for now that tensors are padded to be of max_sequence_length
            --> Assume for now that we will not be adding to the trajectories after creation, 
                hence values, policies, rewards are required on init
        """
        # if tokenizer is not None:
        #     pad_token_id = tokenizer.pad_token_id
        #     eos_token_id = tokenizer.eos_token_id
        # else:
        #     pad_token_id, eos_token_id = None, None
        
        # ### Verify base input ###
        # init_state = torch.from_numpy(init_state) if isinstance(init_state, np.ndarray) else init_state

        # if init_state.dim() != 3:
        #     raise ValueError("init_state.dim() should be 3")
        # if init_state.shape[1] > max_sequence_length:
        #     # Should I require max sequence length? --> Will be needed when making trajectories that we can add to after
        #     raise ValueError("given trajectory length longer than max_sequence_length")

        # self.device = device = init_state.device
        # self.batch_size = batch_size = init_state.shape[0]
        # self.max_sequence_length = max_sequence_length
        # self.obs_dim = init_state.shape[2]

        # ### Handle padding ###
        # if pad_mask is not None:
        #     self._pad_mask = pad_mask
        # elif pad_token_id is not None:
        #     # Detect and create a mask
        #     if self.obs_dim == 1: 
        #         self._pad_mask = ~(init_state.squeeze(-1) == pad_token_id)
        #     else:
        #         raise ValueError("Padding detection only supported for obs_dim=1")
        # else:
        #     # Assume no padding - all sequences use full length
        #     warnings.warn(f"No pad_token_id provided. Assuming all sequences are valid")
        #     self._pad_mask = torch.ones((batch_size, max_sequence_length), dtype=torch.bool)

        # self._states = init_state * self._pad_mask.unsqueeze(2)
        # self._full_states = None
        # self._actions = torch.zeros((batch_size, max_sequence_length), device=device) 
        # self._rewards = rewards if rewards is not None else \
        #     torch.zeros((batch_size), device=device)
        # self._policies = policies * self._pad_mask.unsqueeze(2) if policies is not None else None
        # self._values = values * self._pad_mask if values is not None else \
        #     torch.zeros((batch_size, max_sequence_length), device=device)
        # self._log_probs = torch.zeros((batch_size, max_sequence_length), device=device)
        # self._R = torch.zeros((batch_size, max_sequence_length), device=device)
        # self._A = torch.zeros((batch_size, max_sequence_length), device=device)
        # self._kl = torch.zeros((batch_size, max_sequence_length), device=device)

        # if reward_mask is not None:
        #     self._reward_mask = reward_mask
        # elif eos_token_id is not None:
        #     self._reward_mask = (self.states == eos_token_id).squeeze(-1)
        # else:
        #     self._reward_mask = None

    def get_trajectory(self):

        return {
            'states': self.states,
            'full_states': self.full_states,
            'actions': self.actions,
            'rewards': self.rewards,
            'raw_rewards': self.raw_rewards,
            'values': self.values,
            'log_probs': self.log_probs,
            'R': self.R,
            'A': self.A,
            'A_raw': self.A_raw,
            'kl': self.kl,
            'pad_mask': self.pad_mask,
            'action_pad_mask': self.action_pad_mask,
            'reward_mask': self.reward_mask,
        }

    @profile
    def compute_R(gamma, r, action_pad_mask):
        """ 
        Calculates discounted rewards, aka rewards to go
        """

        time_dim = Trajectory.TIME_DIM

        # Get discounts
        discounts_rev = torch.ones_like(r, device=r.device) * gamma 
        discounts_rev = discounts_rev.masked_fill(~action_pad_mask.flip(dims=[time_dim]), 1)
        discounts_rev = torch.cumprod(discounts_rev,dim=time_dim) / gamma
        discounts_rev = discounts_rev.masked_fill(~action_pad_mask.flip(dims=[time_dim]), 0)

        # Calculate
        r_rev = torch.flip(r, dims=[time_dim])
        R_rev = torch.cumsum(discounts_rev * r_rev, dim=time_dim)
        R = torch.flip(R_rev, dims=[time_dim])
        return R
    
    @profile
    def compute_gae(V, r, value_pad_mask, gamma, lam):
        
        # if torch.all(self.rewards == 0).item():
        #     raise ValueError("rewards is not set, set non-zero rewards attribute first")
        # if torch.all(self.values == 0).item():
        #     raise ValueError("values is not set, set non-zero values attribute first")
               
        time_dim = Trajectory.TIME_DIM

        # 0. Get V and r
        # len(V) = T+1
        # len(r) = T
        V = V.detach()
        r = r.detach()

        # 0. Calculate V_next by bootstrapping last value
        V_next = V[:, 1:]       # seq_len - 1: value after taking action[t]
        V = V[:, :-1]           # seq_len - 1: value before taking action[t]
        # pad_mask = pad_mask[:, :-1] # align to V, not to actions

        last_valid_idx = value_pad_mask.sum(dim=time_dim) - 1
        batch_idx = torch.arange(V.size(0), device=V.device)
        V_next[batch_idx, last_valid_idx] = 0

        # 1. Compute delta_t (TD Error)
        TD_error = r + gamma * V_next - V
        # TD_error.masked_fill(~self._pad_mask.flip(dims=[time_dim]), 0)

        # 2. Compute recursion in reverse
        # A_t = δ_t + γλ * A_{t+1}
        A = torch.zeros_like(TD_error)
        a = 0
        # Process in reverse (cannot/ hard to be done in a vectorized fashion)
        for t in reversed(range(TD_error.shape[1])):
            a = TD_error[:, t] + gamma * lam * a
            A[:, t] = a
        
        return A
        

        # # 2. Get discounts 
        # discounts_rev = torch.ones(r.size(), device=self.device) * lam * gamma
        # discounts_rev = discounts_rev.masked_fill(~self._pad_mask.flip(dims=[time_dim]), 1)        
        # discounts_rev = torch.cumprod(discounts_rev, dim=time_dim) / (lam * gamma)
        # discounts_rev = discounts_rev.masked_fill(~self._pad_mask.flip(dims=[time_dim]), 0)
        
        # # 3. Calculate GAE via cumulative sum in reverse
        # TD_rev = TD_error.flip(dims=[time_dim]) 
        # A_rev = torch.cumsum(discounts_rev * TD_rev, dim=time_dim)
        # self._A = torch.flip(A_rev, dims=[time_dim])

        # pdb.set_trace()


        # PAPERS GAE FORMULATION
        # lastgaelam = 0
        # advantages_reversed = []
        # for t in reversed(range(gen_length)):
        #     nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
        #     delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        #     lastgaelam = delta + gamma * lam * lastgaelam  # ← Key line
        #     advantages_reversed.append(lastgaelam)
        # advantages = torch.stack(advantages_reversed[::-1], axis=1) 
        

    def compute_log_probs(actions, policy_logits, action_pad_mask):
        
        # NOTE: Mask value here can't be the logically correct large negative number, 
        # as the will cause infs in backward gradient computation later
        return torch.gather(masked_log_softmax(policy_logits, action_pad_mask.unsqueeze(2), mask_value=0, dim=-1), dim=-1, index=actions.long().unsqueeze(-1)).squeeze(-1)
    
    def compute_kl(policy_logits, sft_policy_logits, action_pad_mask):
        """
        Averages kl over action_space = vocab_size space, and over sequence space.
        """
        # NOTE: KL could be computed in different ways. 
        # - KL of the full distribution, KL of the top_p or top_k, or KL on just the actions taken.
        # - KL could be averaged or summed across the sequence dimension. 
        # This implementation currently:
        #  - takes KL over top_p=0.9

        #  - Sums across the policy dim and sums across the sequence dim
        #    -> This is done to reflect https://arxiv.org/pdf/2403.17031 (Figure 10)
        #       Code pointer here: https://github.com/vwxyzjn/summarize_from_feedback_details/blob/main/summarize_from_feedback_details/ppo.py#L798C1

        pad_mask_3d = action_pad_mask.unsqueeze(2)
        log_P = masked_log_softmax(policy_logits, pad_mask_3d, mask_value=0, dim=-1).masked_fill(~pad_mask_3d, 0)
        P = torch.exp(log_P).masked_fill(~pad_mask_3d, 0)
        log_Q = masked_log_softmax(sft_policy_logits, pad_mask_3d, mask_value=0, dim=-1).masked_fill(~pad_mask_3d, 0) # sft

        # Sum over policy dim
        kl_div_per_action = torch.sum((P * (log_P - log_Q)).masked_fill(~pad_mask_3d, 0), dim=-1)
        del pad_mask_3d

        # Sum over sequence dim for tracking
        # kl_div = torch.sum(kl_div_per_token, dim=-1) 
        return kl_div_per_action

    
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
    #         self.log_probs[idx,:], \
    #         self.R[idx,:], \
    #         self.A[idx,:]


    # TODO: Order these
    @property
    def states(self):
        return self._states
    
    @property
    def actions(self):
        return self._actions
    
    @property
    def rewards(self):
        return self._rewards
    
    @property
    def raw_rewards(self):
        return self._raw_rewards

    @property
    def values(self):
        return self._values
    
    @property
    def log_probs(self):
        return self._log_probs
    
    @property
    def R(self):
        return self._R

    @property
    def A(self):
        return self._A
    
    @property
    def A_raw(self):
        return self._A_raw
    
    @property
    def kl(self):
        return self._kl
    
    @property
    def pad_mask(self):
        return self._pad_mask
    
    @property
    def action_pad_mask(self):
        return self._action_pad_mask
    
    @property
    def value_pad_mask(self):
        return self._value_pad_mask

    @property
    def reward_mask(self):
        return self._reward_mask 
    
    @property
    def full_states(self):
        return self._full_states
    


    # === State-indexed tensors (seq_len) ===
    
    @full_states.setter
    def full_states(self, new_full_states):
        new_full_states = torch.as_tensor(new_full_states)
        if hasattr(self, '_full_states') and self._full_states is not None:
            if new_full_states.shape != self._full_states.shape:
                raise ValueError(f"Full states shape {new_full_states.shape} doesn't match expected {self._full_states.shape}")
            new_full_states = new_full_states.to(device=self._full_states.device, dtype=self._full_states.dtype)
        self._full_states = new_full_states

    @states.setter
    def states(self, new_states):
        new_states = torch.as_tensor(new_states)
        if hasattr(self, '_states') and self._states is not None:
            if new_states.shape != self._states.shape:
                raise ValueError(f"States shape {new_states.shape} doesn't match expected {self._states.shape}")
            new_states = new_states.to(device=self._states.device, dtype=self._states.dtype)
        self._states = new_states

    @pad_mask.setter
    def pad_mask(self, new_pad_mask):
        new_pad_mask = torch.as_tensor(new_pad_mask)
        if hasattr(self, '_pad_mask') and self._pad_mask is not None:
            if new_pad_mask.shape != self._pad_mask.shape:
                raise ValueError(f"Pad mask shape {new_pad_mask.shape} doesn't match expected {self._pad_mask.shape}")
            new_pad_mask = new_pad_mask.to(device=self._pad_mask.device, dtype=self._pad_mask.dtype)
        self._pad_mask = new_pad_mask

    @values.setter
    def values(self, new_values):
        new_values = torch.as_tensor(new_values)
        if hasattr(self, '_values') and self._values is not None:
            if new_values.shape != self._values.shape:
                raise ValueError(f"Values shape {new_values.shape} doesn't match expected {self._values.shape}")
            new_values = new_values.to(device=self._values.device, dtype=self._values.dtype)
        self._values = new_values

    @value_pad_mask.setter
    def value_pad_mask(self, new_value_pad_mask):
        new_value_pad_mask = torch.as_tensor(new_value_pad_mask)
        if hasattr(self, '_value_pad_mask') and self._value_pad_mask is not None:
            if new_value_pad_mask.shape != self._value_pad_mask.shape:
                raise ValueError(f"Value pad mask shape {new_value_pad_mask.shape} doesn't match expected {self._value_pad_mask.shape}")
            new_value_pad_mask = new_value_pad_mask.to(device=self._value_pad_mask.device, dtype=self._value_pad_mask.dtype)
        self._value_pad_mask = new_value_pad_mask

    # === Prediction-indexed tensors (seq_len - 1) ===

    @action_pad_mask.setter
    def action_pad_mask(self, new_action_pad_mask):
        new_action_pad_mask = torch.as_tensor(new_action_pad_mask)
        if hasattr(self, '_action_pad_mask') and self._action_pad_mask is not None:
            if new_action_pad_mask.shape != self._action_pad_mask.shape:
                raise ValueError(f"Action pad mask shape {new_action_pad_mask.shape} doesn't match expected {self._action_pad_mask.shape}")
            new_action_pad_mask = new_action_pad_mask.to(device=self._action_pad_mask.device, dtype=self._action_pad_mask.dtype)
        self._action_pad_mask = new_action_pad_mask

    @actions.setter
    def actions(self, new_actions):
        new_actions = torch.as_tensor(new_actions)
        if hasattr(self, '_actions') and self._actions is not None:
            if new_actions.shape != self._actions.shape:
                raise ValueError(f"Actions shape {new_actions.shape} doesn't match expected {self._actions.shape}")
            new_actions = new_actions.to(device=self._actions.device, dtype=self._actions.dtype)
        self._actions = new_actions

    @log_probs.setter
    def log_probs(self, new_log_probs):
        new_log_probs = torch.as_tensor(new_log_probs)
        if hasattr(self, '_log_probs') and self._log_probs is not None:
            if new_log_probs.shape != self._log_probs.shape:
                raise ValueError(f"Log probs shape {new_log_probs.shape} doesn't match expected {self._log_probs.shape}")
            new_log_probs = new_log_probs.to(device=self._log_probs.device, dtype=self._log_probs.dtype)
        self._log_probs = new_log_probs

    @rewards.setter
    def rewards(self, new_rewards):
        new_rewards = torch.as_tensor(new_rewards)
        if hasattr(self, '_rewards') and self._rewards is not None:
            if new_rewards.shape != self._rewards.shape:
                raise ValueError(f"Rewards shape {new_rewards.shape} doesn't match expected {self._rewards.shape}")
            new_rewards = new_rewards.to(device=self._rewards.device, dtype=self._rewards.dtype)
        self._rewards = new_rewards

    @raw_rewards.setter
    def raw_rewards(self, new_raw_rewards):
        new_raw_rewards = torch.as_tensor(new_raw_rewards)
        if hasattr(self, '_raw_rewards') and self._raw_rewards is not None:
            if new_raw_rewards.shape != self._raw_rewards.shape:
                raise ValueError(f"raw_rewards shape {new_raw_rewards.shape} doesn't match expected {self._raw_rewards.shape}")
            new_raw_rewards = new_raw_rewards.to(device=self._raw_rewards.device, dtype=self._raw_rewards.dtype)
        self._raw_rewards = new_raw_rewards

    @reward_mask.setter
    def reward_mask(self, new_reward_mask):
        new_reward_mask = torch.as_tensor(new_reward_mask)
        if hasattr(self, '_reward_mask') and self._reward_mask is not None:
            if new_reward_mask.shape != self._reward_mask.shape:
                raise ValueError(f"Reward mask shape {new_reward_mask.shape} doesn't match expected {self._reward_mask.shape}")
            new_reward_mask = new_reward_mask.to(device=self._reward_mask.device, dtype=self._reward_mask.dtype)
        self._reward_mask = new_reward_mask

    @kl.setter
    def kl(self, new_kl):
        new_kl = torch.as_tensor(new_kl)
        if hasattr(self, '_kl') and self._kl is not None:
            if new_kl.shape != self._kl.shape:
                raise ValueError(f"KL shape {new_kl.shape} doesn't match expected {self._kl.shape}")
            new_kl = new_kl.to(device=self._kl.device, dtype=self._kl.dtype)
        self._kl = new_kl

    @A.setter
    def A(self, new_A):
        new_A = torch.as_tensor(new_A)
        if hasattr(self, '_A') and self._A is not None:
            if new_A.shape != self._A.shape:
                raise ValueError(f"A shape {new_A.shape} doesn't match expected {self._A.shape}")
            new_A = new_A.to(device=self._A.device, dtype=self._A.dtype)
        self._A = new_A

    @A_raw.setter
    def A_raw(self, new_A_raw):
        new_A_raw = torch.as_tensor(new_A_raw)
        if hasattr(self, '_A_raw') and self._A_raw is not None:
            if new_A_raw.shape != self._A_raw.shape:
                raise ValueError(f"A_raw shape {new_A_raw.shape} doesn't match expected {self._A_raw.shape}")
            new_A_raw = new_A_raw.to(device=self._A_raw.device, dtype=self._A_raw.dtype)
        self._A_raw = new_A_raw

    @R.setter
    def R(self, new_R):
        new_R = torch.as_tensor(new_R)
        if hasattr(self, '_R') and self._R is not None:
            if new_R.shape != self._R.shape:
                raise ValueError(f"R shape {new_R.shape} doesn't match expected {self._R.shape}")
            new_R = new_R.to(device=self._R.device, dtype=self._R.dtype)
        self._R = new_R


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
    #     self._log_probs[i] = policy[action]

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
        return {
            'states': self._tjs.states[idx, :],
            'actions': self._tjs.actions[idx, :],
            'rewards': self._tjs.rewards[idx, :],
            'raw_rewards': self._tjs.raw_rewards[idx, :],
            'values': self._tjs.values[idx, :],
            'log_probs': self._tjs.log_probs[idx, :],
            'R': self._tjs.R[idx, :],
            'A': self._tjs.A[idx, :],
            'A_raw': self._tjs.A_raw[idx, :],
            'kl': self._tjs.kl[idx, :],
            'pad_mask': self._tjs.pad_mask[idx, :],
            'action_pad_mask': self._tjs.action_pad_mask[idx, :],
            'value_pad_mask': self._tjs.value_pad_mask[idx, :],
            'reward_mask': self._tjs.reward_mask[idx, :],
            'full_states': self._tjs.full_states[idx, :],
        }

# TODO: Make another TrajectorySet which shuffles the time dimension 
# into the batch dimension for cartpole or non-sequence environments