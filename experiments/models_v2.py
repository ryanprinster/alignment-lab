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


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoConfig
from experiments.profiler import profile
from experiments.monitor import detect_nans


class HFModel(nn.Module, ABC):
    def __init__(self, config, model, tokenizer, model_config, **kwargs):
        super().__init__()
        self.transformer = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.config = config


    
    @classmethod
    @abstractmethod
    def _get_model_class(cls):
        pass

    @staticmethod
    def _setup_padding_token(model, tokenizer):
        # Detail 3 (use a special padding token [PAD]; do not use EOS token synonymously as [PAD])
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    
    @classmethod
    @profile
    def from_pretrained(cls, config, model_name, revision=None, init_head_weights=True, init_head_bias=True, **kwargs):
        # Download model + tokenizer from HF
        model_class = cls._get_model_class()
    
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = model_class.from_pretrained(model_name, revision=revision, **kwargs)
        model_config = model.config

        cls._setup_padding_token(model, tokenizer)
    
        return cls(config, model, tokenizer, model_config, init_head_weights=init_head_weights, init_head_bias=init_head_bias)
    
    @classmethod
    @profile
    def from_state_dict(cls, config, init_model_path, init_head_weights=False, init_head_bias=False, **kwargs):
        # Load from local state dict
        # Auto-detect tokenizer/config from same directory
        model_class = cls._get_model_class()

        if not os.path.exists(init_model_path):
            raise FileNotFoundError(f"Model not found: {init_model_path}")
        
        save_dir = os.path.dirname(init_model_path)
        model_config = AutoConfig.from_pretrained(save_dir)
        tokenizer = AutoTokenizer.from_pretrained(save_dir)

        # Handle custom architectures like ScalarModel (vwxyzjn's reward models)
        if hasattr(model_config, 'base_config'):
            # This is a wrapped model, extract the base config
            base_config_dict = model_config.base_config
            model_config = AutoConfig.from_pretrained(save_dir)
            # Update with base config values
            for key, value in base_config_dict.items():
                if key != '_name_or_path':  # Skip internal fields
                    setattr(model_config, key, value)
        
        # Apply config overrides (e.g., num_labels for sequence classification)
        for key, value in kwargs.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)

        model = model_class.from_config(model_config)

        cls._setup_padding_token(model, tokenizer)
        
        checkpoint = torch.load(init_model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Strip "transformer." prefix if present (from HFModel wrapper)
        # TODO: remove this if / when retraining and re saving llama models
        if any(key.startswith('transformer.') for key in state_dict.keys()):
            state_dict = {
                key.replace('transformer.', '', 1): value 
                for key, value in state_dict.items()
            }

        # Add bias layer if needed
        if hasattr(model, 'score') and model.score.bias is None:
            model.score.bias = nn.Parameter(torch.zeros(model.score.out_features))
        
        model.load_state_dict(state_dict)

        return cls(config, model, tokenizer, model_config, init_head_weights=init_head_weights, init_head_bias=init_head_bias)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    def save_state_dict(self, path):
        pass
        # Save state dict + tokenizer + config to same directory
    
    def save_pretrained(self, path):
        pass
        # Save full HF-style model

    def clean_logits(logits):
        # clean scores, -inf --> 1e-9
        return logits.masked_fill_(torch.isinf(logits), 1e-9)


class HFModel_Causal(HFModel):
    def __init__(self, config, model, tokenizer, model_config,  **kwargs):
        super().__init__(config, model, tokenizer, model_config)
        self.transformer.generation_config.pad_token_id = self.tokenizer.pad_token_id
                 
    @classmethod
    def _get_model_class(cls):
        return AutoModelForCausalLM
    

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
        
        if self.config.temperature_scale_logits:
            return outputs.logits / (self.config.generation_temperature + 1e-7 ), outputs.loss
        return outputs.logits, outputs.loss
    
    @profile
    def generate(self, inputs, max_length, temp, do_sample=True, max_query_length=None):
        # NOTE: generation currently does top_p = 0.9 by default. Pros and cons to this as a design choice.
        generation_obj = self.transformer.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            top_p=self.config.top_p_generation,
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
        policy_logits = HFModel.clean_logits(policy_logits)

        return padded_tokens, policy_logits
    

# Aliases
class HFModel_SFT(HFModel_Causal):
    pass

class HFModel_Policy(HFModel_Causal):
    pass   


class HFModel_Classification(HFModel):
    def __init__(self, config, model, tokenizer, model_config, **kwargs):
        super().__init__(config, model, tokenizer, model_config)

        # Initialize head weights if loading from pretrained (used as a base model)
        # But not from state dict, as state dicts presumably trained as part of the rlhf pipeline
        if kwargs["init_head_weights"]:
            self._init_head_weights()

        if kwargs["init_head_bias"]:

            score_head = self._get_score_head()
            
            if score_head.bias is None:
                score_head.bias = nn.Parameter(torch.zeros(1))
            
            if self.config.calculated_sft_bias is not None:
                self.init_head_bias(self.config.calculated_sft_bias)
        
    def _get_score_head(self):
        """Get the final classification layer (naming varies by model architecture)"""
        # Add new architectures here as needed
        if hasattr(self.transformer, 'score'):
            return self.transformer.score
        elif hasattr(self.transformer, 'classifier'):
            return self.transformer.classifier
        else:
            raise AttributeError(
                f"Could not find score head in {type(self.transformer).__name__}. "
                f"Add the attribute name to _get_score_head()"
            )
        
    def _init_head_weights(self):
        # Detail 11 (Reward head initialization)
        print("Initializing Head Weights...")
        
        score_head = self._get_score_head()
        d_model = score_head.in_features
        std = 1.0 / (d_model + 1) ** 0.5
        
        init.normal_(score_head.weight, mean=0, std=std)
        
    def init_head_bias(self, calculated_sft_bias):
        # Detail 15 (Reward normalization based on SFT demonstrations)
        print(f"Initializing Head Bias to {calculated_sft_bias}...")
        score_head = self._get_score_head()
        score_head.bias.data.fill_(-1.0 * calculated_sft_bias)


class HFModel_SequenceClassification(HFModel_Classification):
    @classmethod
    def _get_model_class(cls):
        return AutoModelForSequenceClassification
        # Detail 12 (Extract reward from the EOS token) Done by default for Llama models
        # https://github.com/huggingface/transformers/blob/v4.41.0/src/transformers/models/llama/modeling_llama.py#L1299

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids) * (input_ids != self.tokenizer.pad_token_id)

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits.squeeze(-1)  # -> (batch, )

class HFModel_TokenClassification(HFModel_Classification):

    @classmethod
    def _get_model_class(cls):
        return AutoModelForTokenClassification

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


# Aliases
class HFModel_Reward(HFModel_SequenceClassification):
    pass

class HFModel_Value(HFModel_TokenClassification):
    pass   