import atexit
import signal
import torch
import sys
import psutil
import pdb

import experiments.debug


from experiments.profiler import profile
from experiments.trajectory import Trajectory
from experiments.monitor import detect_nans
from experiments.util import masked_mean, masked_log_softmax, masked_whiten, whiten

import gymnasium as gym

from abc import ABC, abstractmethod
from typing import Any, Tuple
import warnings
import torch.nn.functional as F

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
        self.tokenizer = self.data.tokenizer

    def _close(self):
        self._closed = True

    def _find_first_token_position(self, states, token_id):
        """Find the position of the first occurrence of token_id in each sequence."""
        token_mask = (states == token_id)
        first_token_pos = torch.where(
            token_mask.any(dim=1),
            token_mask.int().argmax(dim=1),
            torch.full((states.size(0),), states.size(1), device=states.device)
        )
        return first_token_pos

    def _enforce_padding(self, states):
        """
        Applies EOS trick (pt 1)
        Sets all tokens after the first EOS or PAD token to PAD token.
        Detail 23.2: Truncate and pad after EOS to ensure valid RM scores
        """
        first_eos_pos = self._find_first_token_position(states, self.tokenizer.eos_token_id)
        first_pad_pos = self._find_first_token_position(states, self.tokenizer.pad_token_id)
        
        first_termination_pos = torch.min(first_eos_pos, first_pad_pos)
        
        # Set everything after to pad
        pos = torch.arange(states.size(1), device=states.device).unsqueeze(0)
        after_termination_mask = pos > first_termination_pos.unsqueeze(1)
        states[after_termination_mask] = self.tokenizer.pad_token_id
        
        return states

    def construct_pad_mask(self, states):
        """Constructs a boolean mask where True indicates valid tokens before the first padding token."""
        first_pad_pos = self._find_first_token_position(states, self.tokenizer.pad_token_id)
        pos = torch.arange(states.size(1), device=states.device).unsqueeze(0)
        return ~(pos >= first_pad_pos.unsqueeze(1))
    
    def construct_reward_mask(self, states):
        return (states == self.tokenizer.eos_token_id)
         
    def _set_models_to_eval(self, *models):
        """Set all models to evaluation mode."""
        for model in models:
            model.eval()

    def _set_models_to_train(self, *models):
        """Set all models to training mode."""
        for model in models:
            model.train()

    def _generate_and_compute_outputs(self, batch, policy_model, value_model, sft_model, reward_model, temp):
        """Generate sequences and compute all model outputs."""
        full_states, _ = policy_model.generate(
            batch,
            self.max_sequence_length,
            temp,
            max_query_length=self.data.SFT_MAX_QUERY_LENGTH,
        )
        del _

        policy_logits, _ = policy_model.forward(full_states)
        sft_policy_logits, _ = sft_model.forward(full_states)
        values = value_model.forward(full_states)
        rewards = reward_model.forward(full_states)
        
        return full_states, policy_logits, sft_policy_logits, values, rewards

    def _truncate_to_response(self, states, values, policy_logits, sft_policy_logits, response_length):
        """Truncate all tensors to the response portion only."""
        states = states[:, -response_length:]
        values = values[:, -response_length:]
        policy_logits = policy_logits[:, -response_length:, :]
        sft_policy_logits = sft_policy_logits[:, -response_length:, :]
        
        return states, values, policy_logits, sft_policy_logits
    
    def _create_masks(self, states):
        # Note: construct_pad_mask would not work for full_states
        return self.construct_pad_mask(states), self.construct_reward_mask(states)
    
    def _align_to_action_space(self, states, policy_logits, sft_policy_logits, rewards, pad_mask, reward_mask):
        """
        Create the following alignment

        ----State Indexing---- (len = seq_len)
        Position:            0         1         2        3        4
        states:          [prompt,   token1,   token2,    EOS,     PAD]
        pad_mask:        [   1,        1,        1,       1,       0 ]
        values:          [  V0,       V1,       V2,      V3,      V4 ]

        ----Action Indexing---- (len = seq_len-1)
        Position:            0         1         2        3
        actions:         [token1,   token2,    EOS,     PAD]
        logits:          [   L0,       L1,      L2,      L3 ]
        log_probs:       [  lp0,      lp1,     lp2,     lp3 ]
        rewards:         [    0,        0,      +1,       0 ]
        reward_mask:     [    0,        0,       1,       0 ]
        advantages:      [   A0,       A1,      A2,      A3 ]
        action_pad_mask: [    1,        1,       1,       0 ]
        """

        ### State Indexing ###
        # states = states
        # pad_mask = pad_mask
        # values = values
        value_pad_mask = pad_mask[:, :-1]

        ### Action Indexing ###
        actions = states[:, 1:]
        policy_logits = policy_logits[:, :-1, :] 
        sft_policy_logits = sft_policy_logits[:, :-1, :]
        reward_mask = reward_mask[:, 1:]
        rewards_2d = rewards.unsqueeze(1) * reward_mask
        action_pad_mask = pad_mask[:, 1:]

        return actions, policy_logits, sft_policy_logits, rewards_2d, reward_mask, action_pad_mask, value_pad_mask

    def _set_reward_for_no_eos(self, rewards, reward_mask, action_pad_mask, penalty=-1.0):
        """
        Applies EOS trick (pt 2)
        Detail 23.3: set -1 reward for missing EOS (penalty defaults to -1.0)
        Also marks the last position as EOS in the masks for sequences without EOS.
        """

        has_eos = reward_mask.any(dim=1)
        
        # For sequences without EOS, set the last valid position to True in the mask
        if (~has_eos).any():
            last_valid_idx = action_pad_mask.sum(dim=1) - 1  # Get last non-padded position
            batch_idx = torch.arange(action_pad_mask.size(0), device=action_pad_mask.device)
            reward_mask[batch_idx[~has_eos], last_valid_idx[~has_eos]] = True
        
        # Set penalty reward for sequences without EOS
        rewards = rewards.clone()
        rewards[~has_eos] = penalty
        
        return rewards, reward_mask
    
    def _create_trajectory(
            self,
            states, 
            full_states,
            values, 
            pad_mask,
            actions,
            action_pad_mask,
            log_probs, 
            rewards, 
            reward_mask, 
            kl_per_action,
            A,
            R
        ):
        """Create a Trajectory object with base quantities."""

        tj = Trajectory(batch_size=states.size(0))
        tj.states = states
        tj.full_states = full_states
        tj.values = values
        tj.pad_mask = pad_mask
        tj.actions = actions
        tj.action_pad_mask = action_pad_mask
        tj.log_probs = log_probs
        tj.rewards = rewards
        tj.reward_mask = reward_mask
        tj.kl = kl_per_action
        tj.A = A
        tj.R = R

        return tj
    
    @profile
    def generate_trajectory(self, 
                            batch, 
                            policy_model, 
                            value_model, 
                            sft_model,
                            temp,
                            reward_model = None):
        with torch.no_grad():
            self._set_models_to_eval(policy_model, value_model, sft_model)

            # Raw model output
            full_states, policy_logits, sft_policy_logits, values, rewards = \
                self._generate_and_compute_outputs(
                    batch, policy_model, value_model, sft_model, 
                    reward_model, temp
                )
            
            # full_states may not be at max response length, but the query will be at max
            response_length = full_states.shape[1] - self.data.SFT_MAX_QUERY_LENGTH 
            states, values, policy_logits, sft_policy_logits = \
                self._truncate_to_response(
                    full_states, 
                    values, 
                    policy_logits, 
                    sft_policy_logits, 
                    # we need the last token of the prompt because the policy here dictates 
                    # the action taken == the token generated at the beginning of the response
                    response_length+1 
                )
            
            states = self._enforce_padding(states)
            pad_mask, reward_mask = self._create_masks(states)
                        
            actions, policy_logits, sft_policy_logits, rewards_2d, reward_mask, action_pad_mask, value_pad_mask = \
                self._align_to_action_space(
                    states, 
                    policy_logits, 
                    sft_policy_logits, 
                    rewards,
                    pad_mask,
                    reward_mask,
                )

            # Apply EOS trick
            rewards_2d, reward_mask = \
                self._set_reward_for_no_eos(rewards_2d, reward_mask, action_pad_mask)
            
            
            # TODO: check all tensors are on the right devices
        
            kl_per_action = Trajectory.compute_kl(policy_logits, sft_policy_logits, action_pad_mask)
            del sft_policy_logits

            log_probs = Trajectory.compute_log_probs(actions, policy_logits, action_pad_mask)
            del policy_logits

            # NOTE: Ordering to reflect the following implementation
            # https://github.com/vwxyzjn/summarize_from_feedback_details/blob/main/summarize_from_feedback_details/ppo.py#L679

            # TODO: Should I maintain other states for tracking? (rewards before kl, after whitening, etc)


            # 1. Apply KL to rewards
            rewards_2d = (rewards_2d - (self.config.beta * kl_per_action)).masked_fill(~action_pad_mask, 0)

            # 2. Whiten rewards
            if self.config.whiten_rewards:
                rewards_2d = masked_whiten(rewards_2d, action_pad_mask, shift_mean=False)

            # 3. Compute advantages
            A = Trajectory.compute_gae(values, rewards_2d, value_pad_mask, self.config.gamma, self.config.lam)
            if self.config.whiten_A:
                # NOTE: shift mean here to keep 
                # A > 0 to be "action better than expected", 
                # A < 0 to be "action worse than expected"
                A = masked_whiten(A, value_pad_mask) # TODO: double check
            
            # 4. Compute returns/rewards-to-go
            R = Trajectory.compute_R(gamma=self.config.gamma, r=rewards_2d, action_pad_mask=action_pad_mask)

            tj = self._create_trajectory(
                states,
                full_states,
                values.masked_fill(~pad_mask, 0),
                pad_mask,
                actions.masked_fill(~action_pad_mask, 0),
                action_pad_mask,
                log_probs.masked_fill(~action_pad_mask, 0),
                rewards_2d.masked_fill(~action_pad_mask, 0),
                reward_mask,
                kl_per_action.masked_fill(~action_pad_mask, 0),
                A.masked_fill(~pad_mask[:, :-1], 0),
                R.masked_fill(~action_pad_mask, 0)
            )

                         
        policy_model.train()
        value_model.train()

        return tj