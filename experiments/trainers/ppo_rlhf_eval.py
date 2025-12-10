
# Standard library imports
import os
from functools import reduce
from datetime import datetime
import pdb
from contextlib import nullcontext

from experiments.debug import DEBUG

# Third-party imports
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

from experiments.logger import Logger
from experiments.environment import RLHFEnvironment
from experiments.profiler import profile
from experiments.datasets import TLDRFilteredDataPPO, TLDRFilteredDataSFT
from experiments.util import masked_mean, masked_var, masked_whiten, masked_log_softmax, whiten

from experiments.models_v2 import HFModel_Policy, HFModel_Value, HFModel_SFT, HFModel_Reward
from experiments.models import Llama_3p2_1B_Policy, Llama_3p2_1B_Value, Llama_3p2_1B_SFT, Llama_3p2_1B_RM

from experiments.trajectory import Trajectory, TrajectorySet
from experiments.config import PPOConfigBase
from experiments.trainers.base_trainer import BaseTrainer
from experiments.checkpointer import Checkpointer
from torch.optim.lr_scheduler import LinearLR
from experiments.monitor import detect_nans


import anthropic

class PPORLHFEval(BaseTrainer):
    


    @profile
    def __init__(self, config: PPOConfigBase):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = TLDRFilteredDataPPO(tokenizer=self.policy_model.tokenizer, batch_size=self.config.batch_size)
        self.client = anthropic.Anthropic(api_key="your-api-key")

        # Model that we want to evaluate vs reference summaries
        self.model = Llama_3p2_1B_Policy(self.config, init_model_path=self.config.sft_model_path).to(self.device)

        # self.reference_ppo_model = self._load_model(
        #     HFModel_Policy,
        #     hf_name="vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr",
        #     hf_revision="sft__44413__1708611267",
        # ).to(self.device)

    def _judge_prompt(post, summary_a, summary_b):
        return f"""Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? Judge based on accuracy, coverage, and coherence.

            Post:
            {post}

            Summary A:
            {summary_a}

            Summary B:
            {summary_b}

            FIRST provide a one-sentence comparison of the two summaries, explaining which you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format:
            Comparison: <one-sentence comparison and explanation>
            Preferred: <"A" or "B">"""

    def _claude_request_json(idx, post, summary_a, summary_b):
        return {
            "custom_id": f"comparison-{idx}",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 200,
                "messages": [{
                    "role": "user",
                    "content": PPORLHFEval._judge_prompt(post, summary_a, summary_b)}]
            }
        }

    def _construct_claude_request(self):
        self.policy_model.eval()

        requests = []

        for batch_idx, batch in enumerate(self.data.validation_loader):
            
            # get prompts from batch
            # get reference summaries from batch
            
            full_states, _  = self.policy_model.generate(
                batch,
                self.config.max_sequence_length, # ???
                self.config.generation_temperature,
                max_query_length=self.data.SFT_MAX_QUERY_LENGTH,
            )
            del _
            response_length = full_states.shape[1] - self.data.SFT_MAX_QUERY_LENGTH

            # Truncate to responses
            states = full_states[:, -response_length:]


            for i in enumerate(batch): # enumerate through batch
                # extract post
                request = PPORLHFEval._claude_request_json(post, generated_summary, reference_summary)
                requests.append(request)

        batch = self.client.messages.batches.create(requests=requests)
        print(f"Submitted 7000 comparisons: {batch.id}")

            
            
            



