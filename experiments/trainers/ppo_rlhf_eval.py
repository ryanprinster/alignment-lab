
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
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.checkpointer = Checkpointer(self.config)

        # Model that we want to evaluate vs reference summaries
        self.model = Llama_3p2_1B_Policy(self.config, init_model_path=self.config.policy_checkpoint_path).to(self.device)

        self.checkpointer.load_checkpoint(
                self.config.policy_checkpoint_path,
                self.model,
                self.device
            )
        
        # self.reference_ppo_model = self._load_model(
        #     HFModel_Policy,
        #     hf_name="vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr",
        #     hf_revision="sft__44413__1708611267",
        # ).to(self.device)

        self.data = TLDRFilteredDataPPO(tokenizer=self.model.tokenizer, batch_size=self.config.batch_size)
       

    def _to_device(self, batch):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)
        return batch

    @staticmethod
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

    @staticmethod
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
    
    def tensor_to_formatted_string(self, tensor):
        """
        Trims tokenized tensor of both pad and eos tokens, then decodes to text 
        """
        trim_ids = [self.data.tokenizer.pad_token_id, self.data.tokenizer.eos_token_id, self.data.tokenizer.bos_token_id]

        left = 0
        right = len(tensor) - 1
        
        while left <= right and tensor[left].item() in trim_ids:
            left += 1
        while right >= left and tensor[right].item() in trim_ids:
            right -= 1
            
        return self.data.tokenizer.decode(tensor[left:right+1])
        

    
    def format_batch_for_generation(self, batch, max_query_length):
        input_ids = []
        attention_masks = []
        summary_ids = []
        for subreddit, title, post, summary in zip(batch["subreddit"], batch["title"], batch["post"], batch["summary"]):
            query_text = self.data.get_query_text(subreddit, title, post)
            inputs = self.data.tokenizer(query_text, return_tensors="pt")
            inputs['input_ids'] = inputs['input_ids'].squeeze()
            inputs['attention_mask'] = inputs['attention_mask'].squeeze()
            inputs = self._to_device(inputs)
            input_ids.append(torch.nn.functional.pad(inputs['input_ids'], (max_query_length - inputs['input_ids'].size(0), 0), value=self.data.tokenizer.pad_token_id))
            attention_masks.append(torch.nn.functional.pad(inputs['attention_mask'], (max_query_length - inputs['attention_mask'].size(0), 0), value=0))

            summary_ids.append(self.data.tokenizer(summary, return_tensors="pt")['input_ids'].squeeze())
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks)
        }, summary_ids
    
    @profile
    def torch_batch_to_request(self, prompts, summary_ids, generated_summaries):
        for i in range(len(summary_ids)): # enumerate through batch
            prompt_text = self.tensor_to_formatted_string(prompts[i])
            generated_summary_text = self.tensor_to_formatted_string(generated_summaries[i])
            reference_summary_text = self.tensor_to_formatted_string(summary_ids[i])
            request = PPORLHFEval._claude_request_json(len(self.requests), prompt_text, generated_summary_text, reference_summary_text)
            self.requests.append(request)
        print("\n\n\n", PPORLHFEval._judge_prompt(prompt_text, generated_summary_text, reference_summary_text), "\n\n\n")


    def construct_claude_request(self):
        self.model.eval()

        self.requests = []
        pdb.set_trace()

        for batch_idx, batch in enumerate(self.data.validation_loader):

            input_batch, summary_ids = self.format_batch_for_generation(batch, self.data.__class__.SFT_MAX_QUERY_LENGTH)

            # get prompts from batch
            # get reference summaries from batch

            full_states, _  = self.model.generate(
                input_batch,
                self.data.__class__.SFT_MAX_INPUT_LENGTH, # ???
                self.config.generation_temperature,
                max_query_length=self.data.__class__.SFT_MAX_QUERY_LENGTH,
            )
            del _
            prompts = input_batch['input_ids']
            del input_batch
            generated_summaries = full_states[:, self.data.__class__.SFT_MAX_QUERY_LENGTH:]
            del full_states

            self.torch_batch_to_request(prompts, summary_ids, generated_summaries)

        print("finished creating batched requests")
        pdb.set_trace()
        batch = self.client.messages.batches.create(requests=self.requests)
        print(f"Submitted comparisons: {batch.id}")

    
    def download_batch_results(self, batch_id="msgbatch_01KunPMsZouDwKJsWNh2j3Er", output_file="batch_results.jsonl"):
        """Download batch results and save to file"""
        
        # Check if batch is complete
        batch = self.client.messages.batches.retrieve(batch_id)
        print(f"Status: {batch.processing_status}")
        print(f"Succeeded: {batch.request_counts.succeeded}")
        print(f"Errored: {batch.request_counts.errored}")
        
        if batch.processing_status != "ended":
            print(f"Batch not ready yet. Current status: {batch.processing_status}")
            return None
        
        # Download all results
        results = []
        for result in self.client.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                response_text = result.result.message.content[0].text
                results.append({
                    "custom_id": result.custom_id,
                    "response": response_text,
                    "status": "success"
                })
            else:
                results.append({
                    "custom_id": result.custom_id,
                    "error": str(result.result.error),
                    "status": "error"
                })
        
        # Save to file
        import json
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        print(f"Downloaded {len(results)} results to {output_file}")
        return results


    def parse_preferences(self):
        """Extract A/B preferences from results"""
        results = 
        preferences = []
        for result in results:
            if result['status'] == 'success':
                response = result['response']
                # Extract the "Preferred: A" or "Preferred: B" line
                if "Preferred: A" in response:
                    preference = "A"  # Your model won
                elif "Preferred: B" in response:
                    preference = "B"  # Reference won
                else:
                    preference = "unclear"
                
                preferences.append({
                    "comparison_id": result['custom_id'],
                    "preference": preference,
                    "full_response": response
                })
        
        # Calculate win rate
        model_wins = sum(1 for p in preferences if p['preference'] == 'A')
        total = len(preferences)
        win_rate = model_wins / total if total > 0 else 0
        
        print(f"Model win rate: {win_rate:.2%} ({model_wins}/{total})")
        return preferences

            
            
            



