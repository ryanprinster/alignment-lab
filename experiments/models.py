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

from experiments.debug import DEBUG


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
            self.transformer.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
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
        return logits.masked_fill_(torch.isinf(logits), 1e-9)
    
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
    def generate(self, inputs, max_length, temp, do_sample=True, max_query_length=None):
        # NOTE: generation currently does top_p = 0.9 by default. Pros and cons to this as a design choice.

        generation_obj = self.transformer.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            temperature=temp,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        sequences = generation_obj.sequences
        scores = generation_obj.scores
        del generation_obj
        # torch.cuda.empty_cache()

        # NOTE on Detail 23.1 (PPO Training -> “EOS trick” to ensure scores from the RM is valid -> Always sample a fixed amount of tokens) 
        # It is observed that forcing the model to continue to produce more after EOS token via min_length parameter
        # results in the model never producing EOS tokens. This might be changed if during SFT model training this behavior was trained in.
        # instead, we use the max length of a sequence in the batch, which is functionally very similar.  
        padded_tokens = torch.nn.functional.pad(
            sequences,
            (0, max_length - sequences.size(1)),
            value=self.tokenizer.pad_token_id
        )
        del sequences

        policy_logits = torch.stack(scores, dim=1)
        
        # Truncate policy logits early to save memory
        if max_query_length is not None:
            respose_length = padded_tokens.shape[1] - max_query_length
            policy_logits = policy_logits[:,-respose_length:,:]

        del scores
        policy_logits = policy_logits.half() # float32 -> float16

        policy_logits = torch.nn.functional.pad(
            policy_logits,
            (0, 0, 0, max_length - policy_logits.size(1)),
            value= -float('inf')
        )
        policy_logits = self.clean_logits(policy_logits)

        return padded_tokens, policy_logits

    @profile
    def forward(self, input_ids, attention_mask=None, labels=None, max_query_length_truncate=None):

        if labels is not None:
            labels = labels.squeeze(-1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids) * (input_ids != self.tokenizer.pad_token_id)

        # Forward parallel decode
        outputs = self.transformer(
            input_ids=input_ids.squeeze(-1),
            attention_mask=attention_mask.squeeze(-1),
            labels=labels,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False 
        )

        if max_query_length_truncate is not None:
            return outputs.logits[:,max_query_length_truncate:,:], outputs.loss
        
        return outputs.logits, outputs.loss

    def _set_model_class(self):
        return AutoModelForCausalLM.from_pretrained(Llama_3p2_1B.HF_MODEL_NAME)


class Llama_3p2_1B_SFT(Llama_3p2_1B_Causal):
    pass

class Llama_3p2_1B_Policy(Llama_3p2_1B_Causal):
    pass   
    

class Llama_3p2_1B_RM(Llama_3p2_1B):
    def __init__(self, config, init_model_path=None, calculated_sft_bias=None):
        self.init_model_path = init_model_path
        super().__init__(config)
        
        # score layer doesn't come with a bias
        if self.transformer.score.bias is None:
            self.transformer.score.bias = nn.Parameter(torch.zeros(1))

        if calculated_sft_bias is not None:
            self.init_head_bias(calculated_sft_bias)

        if init_model_path is None:
            self._init_head_weights()

        self._init_model_weights()

    def _init_head_weights(self):
        # Detail 11 (Reward head initialization)
        print("Initializing Head Weights...")

        d_model = self.transformer.score.in_features
        std = 1.0 / (d_model + 1) ** 0.5
        
        init.normal_(self.transformer.score.weight, mean=0, std=std)

    def init_head_bias(self, calculated_sft_bias):
        # Detail 15 (Reward normalization based on SFT demonstrations)
        print("Initializing Head Bias...")
        self.transformer.score.bias.data.fill_(-1.0 * calculated_sft_bias)

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
        return outputs.logits.squeeze(-1)  # -> (batch, )


class Llama_3p2_1B_Value(Llama_3p2_1B):
    def __init__(self, config, init_model_path=None, init_rm_model=None):
        self.init_model_path = init_model_path
        super().__init__(config)
        self.transformer.config.pad_token_id = self.tokenizer.pad_token_id
        self._init_model_weights()
        self._init_head_weights(init_rm_model)


    def _set_model_class(self):
        return AutoModelForTokenClassification.from_pretrained(
            Llama_3p2_1B.HF_MODEL_NAME,
            num_labels=1
        )
    
    def _init_head_weights(self, init_rm_model):
        # TODO: cleanup how this is called
        if init_rm_model is None:
            return 
        # if not os.path.exists(self.init_model_path):
        #     raise FileNotFoundError(f"Model not found: {self.init_model_path}")
        
        # TODO: Finish properly init the value model
        # NOTE: It seems the model does this by default, but 
        self.transformer.score.weight.data = init_rm_model.transformer.score.weight.data.clone()
        self.transformer.score.bias.data = init_rm_model.transformer.score.bias.data.clone()

    @profile
    def forward(self, input_ids, attention_mask=None, max_query_length_truncate=None):
        # Forward parallel decode

        # Mask pad tokens
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids) * (input_ids != self.tokenizer.pad_token_id)

        outputs = self.transformer(
            input_ids=input_ids.squeeze(-1),
            attention_mask=attention_mask.squeeze(-1),
        )
        
        if max_query_length_truncate is not None:
            return outputs.logits[:,max_query_length_truncate:,:].squeeze(-1)
        
        return outputs.logits.squeeze(-1) # -> (batch, seq_len)


#  For some reason, forward in Llama_3p2_1B_Value does not give the same or even close values to forward in Llama_3p2_1B_RM
# Hypotheses include: 
# off by one errors in indexing, 
# some sort of randomness is added, 
# the models are actually not equivalent for some sort of intialization issue 
# This could be messing with ppo rlhf training
# The RM one is not actually returning the reward at EOS token
