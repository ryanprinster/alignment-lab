from functools import reduce
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from experiments.profiler import profile
from experiments.monitor import detect_nans


# TODO: Could optionally combine these classes
class MLPValue(nn.Module):
    def __init__(self, obs_dim, hidden_dims=None):
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]

        self.l1 = nn.Linear(obs_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1],1)

    def forward(self, x):
        single_input = x.dim() == 1
        if single_input:
            x = x.unsqueeze(dim=0)
        
        x = self.l1(x)
        x = F.leaky_relu(x)
        x = self.l2(x)
        x = F.leaky_relu(x)
        x = self.l3(x)

        if single_input:
            x = x.squeeze(dim=0)

        return x

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=None):
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]

        self.l1 = nn.Linear(obs_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1], action_dim)

    def forward(self, x):
        # TODO: Make this more clean
        single_input = x.dim() == 1
        if single_input:
            x = x.unsqueeze(dim=0)
        
        x = self.l1(x)
        x = F.leaky_relu(x)
        x = self.l2(x)
        x = F.leaky_relu(x)
        x = self.l3(x)
        # Could do log probs but this works fine
        x = torch.softmax(x, dim=-1)
        if single_input:
            x = x.squeeze(dim=0)
        return x

class Llama_3p2_1B(nn.Module):
    @profile
    def __init__(self, hf_model_name="meta-llama/Llama-3.2-1B"):
        super().__init__()
        self.transformer = AutoModelForCausalLM.from_pretrained(hf_model_name)
        # self.transformer.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        
        # # NOTE: NOT advised by (https://arxiv.org/pdf/2403.17031) detail 3
        # self.tokenizer.pad_token = self.tokenizer.eos_token 
        # Detail 3 (use a special padding token [PAD]; do not use EOS token synonymously as [PAD])
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.transformer.resize_token_embeddings(len(self.tokenizer))

    # @detect_nans
    @profile
    def forward(self, input_ids, attention_mask, labels):
        """
        Note for learning:
        Since this is a Causal LM, therefore in this case labels are likely the data itself
        (input_ids), shifted by one place. 
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        torch.cuda.empty_cache()
        return outputs
