from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from experiments.profiler import profile
from experiments.util import masked_log_softmax


@dataclass
class Trajectory:
    BATCH_DIM, TIME_DIM = 0, 1

    @profile
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def get_trajectory(self):
        return {
            "states": self.states,
            "full_states": self.full_states,
            "actions": self.actions,
            "rlhf_rewards": self.rlhf_rewards,
            "raw_rewards": self.raw_rewards,
            "values": self.values,
            "log_probs": self.log_probs,
            "R": self.R,
            "A": self.A,
            "A_raw": self.A_raw,
            "kl": self.kl,
            "pad_mask": self.pad_mask,
            "action_pad_mask": self.action_pad_mask,
            "reward_mask": self.reward_mask,
        }

    # This is unused, but technically usable. The final algorithm simply uses Returns = Values + Advantages
    @profile
    def compute_R(gamma, r, action_pad_mask):
        """
        Returns calculated as discounted rewards-to-go
        """

        time_dim = Trajectory.TIME_DIM

        # Get discounts
        discounts_rev = torch.ones_like(r, device=r.device) * gamma
        discounts_rev = discounts_rev.masked_fill(~action_pad_mask.flip(dims=[time_dim]), 1)
        discounts_rev = torch.cumprod(discounts_rev, dim=time_dim) / gamma
        discounts_rev = discounts_rev.masked_fill(~action_pad_mask.flip(dims=[time_dim]), 0)

        # Calculate
        r_rev = torch.flip(r, dims=[time_dim])
        R_rev = torch.cumsum(discounts_rev * r_rev, dim=time_dim)
        R = torch.flip(R_rev, dims=[time_dim])
        return R

    @profile
    def compute_gae(V, r, value_pad_mask, gamma, lam):

        time_dim = Trajectory.TIME_DIM

        # 0. Get V and r
        # len(V) = T+1
        # len(r) = T
        V = V.detach()
        r = r.detach()

        # 0. Calculate V_next by bootstrapping last value
        V_next = V[:, 1:]  # seq_len - 1: value after taking action[t]
        V = V[:, :-1]  # seq_len - 1: value before taking action[t]

        last_valid_idx = value_pad_mask.sum(dim=time_dim) - 1
        batch_idx = torch.arange(V.size(0), device=V.device)
        V_next[batch_idx, last_valid_idx] = 0

        # 1. Compute delta_t (TD Error)
        TD_error = r + gamma * V_next - V

        # 2. Compute recursion in reverse
        # A_t = δ_t + γλ * A_{t+1}
        A = torch.zeros_like(TD_error)
        a = 0
        # Process in reverse (hard/ maybe impossible to be done in a vectorized fashion)
        for t in reversed(range(TD_error.shape[1])):
            a = TD_error[:, t] + gamma * lam * a
            A[:, t] = a

        return A

    def compute_log_probs(actions, policy_logits, action_pad_mask):
        # NOTE: Mask value as the logically correct large negative number sometimes will
        # cause infs in masked positions in backward gradient computation later, hence 0 mask
        return torch.gather(
            masked_log_softmax(policy_logits, action_pad_mask.unsqueeze(2), mask_value=0, dim=-1),
            dim=-1,
            index=actions.long().unsqueeze(-1),
        ).squeeze(-1)

    def compute_kl(policy_logits, sft_policy_logits, action_pad_mask):
        # NOTE: KL could be computed in different ways.
        # - KL of the full distribution, KL of the top_p or top_k, or KL on just the actions taken.
        # - KL could be averaged or summed across the sequence dimension.
        # This implementation currently:
        #  - takes KL over top_p=0.9

        #  - Sums across the policy dim and sums across the sequence dim
        #    -> This is done to reflect https://arxiv.org/pdf/2403.17031 (Figure 10)
        #       Code pointer here: https://github.com/vwxyzjn/summarize_from_feedback_details/blob/main/summarize_from_feedback_details/ppo.py#L798C1

        pad_mask_3d = action_pad_mask.unsqueeze(2)
        log_P = masked_log_softmax(policy_logits, pad_mask_3d, mask_value=0, dim=-1).masked_fill(
            ~pad_mask_3d, 0
        )
        P = torch.exp(log_P).masked_fill(~pad_mask_3d, 0)
        log_Q = masked_log_softmax(
            sft_policy_logits, pad_mask_3d, mask_value=0, dim=-1
        ).masked_fill(
            ~pad_mask_3d, 0
        )  # sft

        # Sum over policy dim
        kl_div_per_action = torch.sum((P * (log_P - log_Q)).masked_fill(~pad_mask_3d, 0), dim=-1)
        del pad_mask_3d

        return kl_div_per_action


class TrajectorySet(Dataset):
    def __init__(self, trajectory: Trajectory):
        self._tjs = trajectory

    def __len__(self):
        return self._tjs.batch_size

    def __getitem__(self, idx):
        return {
            "states": self._tjs.states[idx, :],
            "actions": self._tjs.actions[idx, :],
            "rlhf_rewards": self._tjs.rlhf_rewards[idx, :],
            "raw_rewards": self._tjs.raw_rewards[idx, :],
            "values": self._tjs.values[idx, :],
            "log_probs": self._tjs.log_probs[idx, :],
            "R": self._tjs.R[idx, :],
            "A": self._tjs.A[idx, :],
            "A_raw": self._tjs.A_raw[idx, :],
            "kl": self._tjs.kl[idx, :],
            "pad_mask": self._tjs.pad_mask[idx, :],
            "action_pad_mask": self._tjs.action_pad_mask[idx, :],
            "value_pad_mask": self._tjs.value_pad_mask[idx, :],
            "reward_mask": self._tjs.reward_mask[idx, :],
            "full_states": self._tjs.full_states[idx, :],
        }


# TODO: Could make another TrajectorySet which shuffles the time dimension
# into the batch dimension for cartpole or non-sequence environments
