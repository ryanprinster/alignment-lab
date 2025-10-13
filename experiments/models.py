from abc import ABC, abstractmethod
import os
from functools import reduce
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.init as init
import pdb
import warnings


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForTokenClassification
from experiments.profiler import profile
from experiments.monitor import detect_nans


class MLPSimple(nn.Module):
    def __init__(self, obs_dim, action_dim=None, hidden_dims=None):
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]

        self.l1 = nn.Linear(obs_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1], action_dim or 1)

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

class Llama_3p2_1B(nn.Module, ABC):
    HF_MODEL_NAME = "meta-llama/Llama-3.2-1B"
    
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.transformer = self._set_model_class()
        if config.enable_gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(Llama_3p2_1B.HF_MODEL_NAME)
        
        # Detail 3 (use a special padding token [PAD]; do not use EOS token synonymously as [PAD])
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.transformer.config.pad_token_id = self.tokenizer.pad_token_id
        self.transformer.resize_token_embeddings(len(self.tokenizer))

    @abstractmethod
    def _set_model_class(self):
        pass

    def _init_model_weights(self):
        pass

    @abstractmethod
    def forward(self, input_ids, attention_mask, labels):
        pass

    def generate(self, inputs, max_length, temp):
        pass

    def clean_logits(self, logits):
        # clean scores, -inf --> 1e-9
        return torch.where(torch.isinf(logits), torch.tensor(-1e9, device=logits.device), logits)
    
    @profile
    def _init_model_weights(self):
        if self.init_model_path is None:
            return 
        if not os.path.exists(self.init_model_path):
            raise FileNotFoundError(f"Model not found: {self.init_model_path}")

        self.load_state_dict(
            torch.load(self.init_model_path, map_location='cpu')['model_state_dict'])


class Llama_3p2_1B_Causal(Llama_3p2_1B):
    def __init__(self, config, init_model_path=None):
        self.init_model_path = init_model_path
        super().__init__(config)
        self.transformer.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self._init_model_weights()

    @profile
    def generate(self, inputs, max_length, temp, do_sample=True):
        # generate autoregressively
        generation_obj = self.transformer.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            temperature=temp,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True
        )

        # NOTE on Detail 23.1 (PPO Training -> “EOS trick” to ensure scores from the RM is valid -> Always sample a fixed amount of tokens) 
        # It is observed that forcing the model to continue to produce more after EOS token via min_length parameter
        # results in the model never producing EOS tokens. This might be changed if during SFT model training this behavior was trained in.
        # instead, we use the max length of a sequence in the batch, which is functionally very similar.  
        padded_tokens = torch.nn.functional.pad(
            generation_obj.sequences,
            (0, max_length - generation_obj.sequences.size(1)),
            value=self.tokenizer.pad_token_id
        )

        policy_logits = torch.stack(generation_obj.scores, dim=1) #torch.softmax(torch_tensor, dim=-1)
        policy_logits = self.clean_logits(policy_logits)

        return padded_tokens, policy_logits

    @profile
    def forward(self, input_ids, attention_mask=None, labels=None):

        if labels is not None:
            labels = labels.squeeze(-1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids) * (input_ids != self.tokenizer.pad_token_id)

        # Forward parallel decode
        outputs = self.transformer(
            input_ids=input_ids.squeeze(-1),
            attention_mask=attention_mask.squeeze(-1),
            labels=labels
        )
        return outputs.logits, outputs.loss

    @profile   
    def _set_model_class(self):
        return AutoModelForCausalLM.from_pretrained(Llama_3p2_1B.HF_MODEL_NAME)


class Llama_3p2_1B_SFT(Llama_3p2_1B_Causal):
    pass

class Llama_3p2_1B_Policy(Llama_3p2_1B_Causal):
    pass   
    

class Llama_3p2_1B_RM(Llama_3p2_1B):
    def __init__(self, config, init_model_path=None):
        self.init_model_path = init_model_path
        super().__init__(config)
        
        # score layer doesn't come with a bias
        if self.transformer.score.bias is None:
            self.transformer.score.bias = nn.Parameter(torch.zeros(1))

        if init_model_path is None:
            self._init_head_weights(config.calculated_sft_bias)

        
        self._init_model_weights()

    def _init_head_weights(self, calculated_sft_bias):
        # Detail 11 (Reward head initialization)
        print("Initializing Head Weights...")

        d_model = self.transformer.score.in_features
        std = 1.0 / (d_model + 1) ** 0.5
        
        init.normal_(self.transformer.score.weight, mean=0, std=std)
        self.transformer.score.bias.data.fill_(-1.0 * calculated_sft_bias)
        # Calculated from the cu         

    @profile   
    def _set_model_class(self):
        return AutoModelForSequenceClassification.from_pretrained(
            Llama_3p2_1B.HF_MODEL_NAME, 
            num_labels=1
        )
        # Detail 12 (Extract reward from the EOS token) Done by default
        # https://github.com/huggingface/transformers/blob/v4.41.0/src/transformers/models/llama/modeling_llama.py#L1299


    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits.squeeze(-1)


class Llama_3p2_1B_Value(Llama_3p2_1B):
    def __init__(self, config, init_model_path=None):
        self.init_model_path = init_model_path
        super().__init__(config)
        self.transformer.config.pad_token_id = self.tokenizer.pad_token_id
        self._init_model_weights()

    @profile   
    def _set_model_class(self):
        return AutoModelForTokenClassification.from_pretrained(
            Llama_3p2_1B.HF_MODEL_NAME,
            num_labels=1
        )

    def forward(self, input_ids, attention_mask=None):
        # Forward parallel decode

        # Mask pad tokens
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids) * (input_ids != self.tokenizer.pad_token_id)

        outputs = self.transformer(
            input_ids=input_ids.squeeze(-1),
            attention_mask=attention_mask.squeeze(-1),
        )
        
        return outputs.logits.squeeze(-1) # -> (batch, seq_len)
