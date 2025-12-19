import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from experiments.profiler import profile


class HFModel(nn.Module, ABC):

    def __init__(self, config, transformer, tokenizer, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.transformer = transformer
        self.tokenizer = tokenizer

    ### INIT METHODS ###

    @classmethod
    def init_from_hf_pretrained(
        cls, config, hf_model_name="meta-llama/Llama-3.2-1B", revision=None
    ):
        """Inits by downloading a pretrained model from HF"""
        transformer = cls._get_model_class(hf_model_name, revision=revision)

        if config.enable_gradient_checkpointing:
            transformer.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, revision=revision)

        cls._setup_padding_token(transformer, tokenizer)

        return cls(config, transformer, tokenizer)

    ### SET METHODS ###

    def set_from_local_state_dict(self, init_model_path):
        self._set_model_weights(init_model_path)

    # NOTE and TODO: Generally there is a decent amount of inefficiency around how models are loaded.
    # There is likely opportunity save buy-in runtime for each script by downloading the model to disk
    # but with different save formats having different loading speeds, doing the initial downloads,
    # dealing with tokenizers, etc I am putting this off.

    # Example:
    # def init_from_local_hf_pretrained(cls, ...):
    #   # Load model and weights directly from local,
    #   # instead of down loading the model and its weights, then setting weights with a local state dict

    @staticmethod
    def _setup_padding_token(model, tokenizer):
        # Detail 3 (use a special padding token [PAD]; do not use EOS token synonymously as [PAD])
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))

    @abstractmethod
    def _get_model_class(cls, hf_model_name, revision=None):
        pass

    @abstractmethod
    def forward(self, input_ids, attention_mask, labels):
        pass

    @profile
    def _set_model_weights(self, init_model_path):
        print("Setting model weights...")
        if not os.path.exists(init_model_path):
            raise FileNotFoundError(f"Model not found: {init_model_path}")

        checkpoint = torch.load(init_model_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.load_state_dict(state_dict, strict=False)

    def disable_dropout(self):
        # Note that llama models seem to have dropout disabled by default
        self.transformer.config.attention_dropout = 0.0
        self.transformer.config.hidden_dropout = 0.0
        self.transformer.config.resid_dropout = 0.0

        for module in self.transformer.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0


class HFModel_Causal(HFModel):
    def __init__(self, config, transformer, tokenizer):
        super().__init__(config, transformer, tokenizer)
        self.transformer.generation_config.pad_token_id = self.tokenizer.pad_token_id

    @profile
    def generate(self, inputs, max_length, temp, do_sample=True, max_query_length=None):
        # NOTE: generation currently does top_p = 0.9 by default, vs 1.0 in the paper.
        # As this is not used for any loss / advantage computation, the impact to training is limited.
        # There are pros and cons to this as a design choice, but this may contribute to a lower entropy.
        generation_obj = self.transformer.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
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

        # NOTE on Detail 23.1 (PPO Training -> “EOS trick” to ensure scores from the RM is valid -> Always sample a fixed amount of tokens)
        # It is observed that forcing the model to continue to produce more after EOS token via min_length parameter
        # results in the model never producing EOS tokens. This might be changed if during SFT model training this behavior was trained in.
        # instead, we use the max length of a sequence in the batch, which is functionally very similar.
        padded_tokens = torch.nn.functional.pad(
            sequences,
            (0, max_length - sequences.size(1)),
            value=self.tokenizer.pad_token_id,
        )
        del sequences

        policy_logits = torch.stack(scores, dim=1)

        # Truncate policy logits early to save memory
        if max_query_length is not None:
            respose_length = padded_tokens.shape[1] - max_query_length
            policy_logits = policy_logits[:, -respose_length:, :]

        del scores
        policy_logits = policy_logits.half()  # float32 -> float16 to save memory

        policy_logits = torch.nn.functional.pad(
            policy_logits,
            (0, 0, 0, max_length - policy_logits.size(1)),
            value=-float("inf"),
        )

        # clean logit scores, -inf --> 1e-9
        policy_logits = policy_logits.masked_fill_(torch.isinf(policy_logits), 1e-9)

        return padded_tokens, policy_logits

    @profile
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        max_query_length_truncate=None,
    ):

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
            use_cache=False,
        )

        if max_query_length_truncate is not None:
            return outputs.logits[:, max_query_length_truncate:, :], outputs.loss

        if self.config.temperature_scale_logits:
            return (
                outputs.logits / (self.config.generation_temperature + 1e-7),
                outputs.loss,
            )
        return outputs.logits, outputs.loss

    @classmethod
    def _get_model_class(cls, hf_model_name, revision=None):
        return AutoModelForCausalLM.from_pretrained(hf_model_name, revision=revision)


class HFModel_SFT(HFModel_Causal):
    pass


class HFModel_Policy(HFModel_Causal):
    pass


class HFModel_Classification(HFModel):
    def __init__(self, config, transformer, tokenizer):
        super().__init__(config, transformer, tokenizer)

        self.transformer.config.pad_token_id = self.tokenizer.pad_token_id

        # Init head weights when initially loading a pretrained model / untrained head
        self._init_head_weights()

    def _set_model_weights(self, init_model_path):
        # If we are setting the model weights, also set the the head biases

        score_head = self._get_score_head()
        # score layer doesn't come with a bias
        if score_head.bias is None:
            score_head.bias = nn.Parameter(torch.zeros(score_head.out_features).to(self.device))

        super()._set_model_weights(init_model_path)

        if self.config.calculated_sft_bias is not None:
            self.init_head_bias(self.config.calculated_sft_bias)

    def _get_score_head(self):
        """Get the final classification layer (naming varies by model architecture)"""
        # Add new architectures here as needed
        if hasattr(self.transformer, "score"):
            return self.transformer.score
        elif hasattr(self.transformer, "lm_head"):
            return self.transformer.lm_head
        elif hasattr(self.transformer, "classifier"):
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
    def _get_model_class(cls, hf_model_name, revision=None):
        return AutoModelForSequenceClassification.from_pretrained(
            hf_model_name, revision=revision, num_labels=1
        )
        # Detail 12 (Extract reward from the EOS token) Done by default for Llama models
        # https://github.com/huggingface/transformers/blob/v4.41.0/src/transformers/models/llama/modeling_llama.py#L1299

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids) * (input_ids != self.tokenizer.pad_token_id)

        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)  # -> (batch, )


class HFModel_TokenClassification(HFModel_Classification):

    @classmethod
    def _get_model_class(cls, hf_model_name, revision=None):
        return AutoModelForTokenClassification.from_pretrained(
            hf_model_name, revision=revision, num_labels=1
        )

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
            return outputs.logits[:, max_query_length_truncate:, :].squeeze(-1)

        return outputs.logits.squeeze(-1)  # -> (batch, seq_len)


# Aliases
class HFModel_Reward(HFModel_SequenceClassification):
    pass


class HFModel_Value(HFModel_TokenClassification):
    pass
