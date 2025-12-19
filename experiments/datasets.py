import pdb
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from experiments.profiler import profile


class ProfiledDataLoader(DataLoader):

    def __iter__(self):
        self._iterator = super().__iter__()
        return self

    # @profile
    def __next__(self):
        return next(self._iterator)


class TLDRFilteredDataBase(ABC):
    SFT_MAX_QUERY_LENGTH = 512
    SFT_MAX_INPUT_LENGTH = 562
    SFT_MAX_REPONSE_LENGTH = SFT_MAX_INPUT_LENGTH - SFT_MAX_QUERY_LENGTH

    def __init__(self, tokenizer, batch_size):
        self.dataset = load_dataset("vwxyzjn/summarize_from_feedback_tldr_3_filtered")

        self.tokenizer = tokenizer

        preprocess_func = partial(self.preprocess_func, tokenizer=tokenizer)

        dataset = self.dataset.map(preprocess_func, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        self.dataset["train"] = dataset["train"]
        self.dataset["validation"] = dataset["validation"]
        self.dataset["test"] = dataset["test"]

        self.train_loader = DataLoader(
            self.dataset["train"],
            collate_fn=default_collate,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.validation_loader = DataLoader(
            self.dataset["validation"],
            collate_fn=default_collate,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.test_loader = DataLoader(
            self.dataset["test"],
            collate_fn=default_collate,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

    @abstractmethod
    def preprocess_func(self, batch, tokenizer=None):
        pass

    def get_query_text(self, subreddit, title, post, tokenizer=None):
        tokenizer = tokenizer or self.tokenizer
        truncated_post = self._truncate_post(subreddit, title, post, tokenizer)
        formatted_query = self._format_query(subreddit, title, truncated_post)
        return formatted_query

    # Detail 2.1 (Format the query), 2.3 (No trailing space after “TL;DR:”)
    def _format_query(self, subreddit, title, post):
        return f"SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"

    # Detail 2.2 (Clever truncation)
    def _truncate_post(self, subreddit, title, post, tokenizer):

        test_query = self._format_query(subreddit, title, post)
        tokens = tokenizer(test_query, return_tensors="pt")

        if tokens["input_ids"].shape[1] <= TLDRFilteredDataBase.SFT_MAX_QUERY_LENGTH:
            return post

        truncate_char = "\n"
        while tokens["input_ids"].shape[1] > TLDRFilteredDataBase.SFT_MAX_QUERY_LENGTH:
            last_newline = post.rfind(truncate_char)
            if last_newline == -1:
                # In the rare case that a single paragraph has too many tokens,
                # naively truncate that paragraph to the last period instead
                truncate_char = "."
                continue

            post = post[:last_newline]
            test_query = self._format_query(subreddit, title, post)
            tokens = tokenizer(test_query, return_tensors="pt")

        return post


class TLDRFilteredDataSFT(TLDRFilteredDataBase):

    def __init__(self, tokenizer, batch_size):
        super().__init__(tokenizer, batch_size)

    def preprocess_func(self, batch, tokenizer=None):
        tokenizer = tokenizer or self.tokenizer
        #  Detail 1 (Dataset -> Specification)
        texts = []

        for subreddit, title, post, summary in zip(
            batch["subreddit"], batch["title"], batch["post"], batch["summary"]
        ):
            formatted_query = self.get_query_text(subreddit, title, post, tokenizer)
            # Detail 3 (Prepend a leading space to completion; append an EOS token to the completions)
            summary = " " + summary + tokenizer.eos_token
            full_text = formatted_query + summary
            texts.append(full_text)

            # Detail 4 (Dataset -> SFT and preference datasets have different tokenization length)
            #   --> TODO: add check to ensure that full_text is <= SFT_MAX_INPUT_LENGTH
            #   --> TODO: when extending to preference dataset, double cehck summary max token lengths

        # Detail 5 (SFT dataset for SFT training: concatenate the query and the reference summary
        # together and pad from the right)
        return tokenizer(
            texts,
            truncation=False,  # Already did ~clever truncation~
            padding="max_length",  # This should use tokenizer.pad_token_id
            max_length=TLDRFilteredDataBase.SFT_MAX_INPUT_LENGTH,
            return_tensors="pt",
        )


class TLDRFilteredDataPPO(TLDRFilteredDataBase):

    def __init__(self, tokenizer, batch_size):
        super().__init__(tokenizer, batch_size)

    def preprocess_func(self, batch, tokenizer=None):
        tokenizer = tokenizer or self.tokenizer
        texts = []

        for subreddit, title, post, summary in zip(
            batch["subreddit"], batch["title"], batch["post"], batch["summary"]
        ):
            formatted_query = self.get_query_text(subreddit, title, post, tokenizer)
            texts.append(formatted_query)

        # Pad Left
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            texts,
            truncation=False,  # Already did ~clever truncation~
            padding="max_length",  # Should use tokenizer.pad_token_id
            max_length=TLDRFilteredDataBase.SFT_MAX_QUERY_LENGTH,
            return_tensors="pt",
        )

        # Reset original padding
        tokenizer.padding_side = original_padding_side
        return inputs


class OpenAIPreferenceData:

    RM_MAX_QUERY_LENGTH = 512
    RM_MAX_INPUT_LENGTH = 638

    @profile
    def __init__(self, tokenizer, batch_size, subset="comparisons"):
        # openai/summarize_from_feedback is not really supported by hf anymore, using this instead
        self.dataset = load_dataset("HuggingFaceH4/summarize-from-feedback")

        self.tokenizer = tokenizer

        preprocess_func = partial(self._extract_preference_data, tokenizer=tokenizer)

        dataset = self.dataset.map(
            preprocess_func,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
            cache_file_names={
                "train": "./.cache/processed_preference_train.arrow",
                "validation": "./.cache/processed_preference_validation.arrow",
            },
        )

        dataset.set_format(
            type="torch",
            columns=[
                "preferred_input_ids",
                "preferred_attention_mask",
                "rejected_input_ids",
                "rejected_attention_mask",
            ],
        )

        self.train_loader = ProfiledDataLoader(
            dataset["train"], batch_size=batch_size, shuffle=True, num_workers=0
        )
        self.validation_loader = ProfiledDataLoader(
            dataset["validation"], batch_size=batch_size, shuffle=True, num_workers=0
        )

    # @profile
    def _extract_preference_data(self, batch, tokenizer):
        preferred_input_ids = []
        preferred_attention_mask = []
        rejected_input_ids = []
        rejected_attention_mask = []
        queries = []

        for meta, responses, label in zip(
            batch["meta"], batch["responses"], batch["label"]
        ):
            post = meta["post"]
            subreddit = meta["subreddit"]
            title = meta["title"]
            preferred_summary = responses[label]["text"]
            rejected_summary = responses[1 - label]["text"]

            query_text = self._truncate_post(subreddit, title, post, tokenizer)
            query_text = self._format_query(subreddit, title, post)

            preferred_text = (
                query_text + " " + preferred_summary + self.tokenizer.eos_token
            )
            rejected_text = (
                query_text + " " + rejected_summary + self.tokenizer.eos_token
            )

            # Detail 14 (Minor numerical differences between extracting reward with
            #   left and right padded queries)
            # Note that tokenizer right pads by default
            preferred_tokens = self.tokenizer(
                preferred_text,
                truncation=True,
                padding="max_length",
                max_length=OpenAIPreferenceData.RM_MAX_INPUT_LENGTH,
                return_tensors="pt",
            )

            rejected_tokens = self.tokenizer(
                rejected_text,
                truncation=True,
                padding="max_length",
                max_length=OpenAIPreferenceData.RM_MAX_INPUT_LENGTH,
                return_tensors="pt",
            )

            preferred_input_ids.append(preferred_tokens["input_ids"].squeeze(0))
            preferred_attention_mask.append(
                preferred_tokens["attention_mask"].squeeze(0)
            )
            rejected_input_ids.append(rejected_tokens["input_ids"].squeeze(0))
            rejected_attention_mask.append(rejected_tokens["attention_mask"].squeeze(0))
            queries.append(query_text)

        return {
            "preferred_input_ids": torch.stack(preferred_input_ids),
            "preferred_attention_mask": torch.stack(preferred_attention_mask),
            "rejected_input_ids": torch.stack(rejected_input_ids),
            "rejected_attention_mask": torch.stack(rejected_attention_mask),
            "queries": queries,
        }

    def _format_query(self, subreddit, title, post):
        return f"SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"

    def get_query_text(self, subreddit, title, post):
        return self._format_query(subreddit, title, post)

    # TODO: This is copy and pasted from above, make a better structure to make this reusable
    # Detail 2.2 (Clever truncation)
    def _truncate_post(self, subreddit, title, post, tokenizer):

        test_query = self._format_query(subreddit, title, post)
        tokens = tokenizer(test_query, return_tensors="pt")

        if tokens["input_ids"].shape[1] <= TLDRFilteredDataBase.SFT_MAX_QUERY_LENGTH:
            return post

        truncate_char = "\n"
        while tokens["input_ids"].shape[1] > TLDRFilteredDataBase.SFT_MAX_QUERY_LENGTH:
            last_newline = post.rfind(truncate_char)
            if last_newline == -1:
                # In the rare case that a single paragraph has too many tokens,
                # naively truncate that paragraph to the last period instead
                truncate_char = "."
                continue

            post = post[:last_newline]
            test_query = self._format_query(subreddit, title, post)
            tokens = tokenizer(test_query, return_tensors="pt")

        return post


class Dolly15kData:

    @profile
    def __init__(self, tokenizer, batch_size, test_size_pct):
        self.dataset = load_dataset("databricks/databricks-dolly-15k")
        preprocess_func = partial(self.dolly15k_preprocessor, tokenizer=tokenizer)

        dataset = self.dataset.map(preprocess_func, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataset = dataset["train"].train_test_split(test_size=test_size_pct)

        self.train_loader = DataLoader(
            dataset["train"], batch_size=batch_size, shuffle=True, num_workers=0
        )
        self.test_loader = DataLoader(
            dataset["test"], batch_size=batch_size, shuffle=True, num_workers=0
        )

    def dolly15k_preprocessor(self, batch, tokenizer, max_length=512):
        texts = []

        for instruction, context, response in zip(
            batch["instruction"], batch["context"], batch["response"]
        ):
            if context and context.strip():
                text = f"### Instruction:\n{instruction}\n### Context:\n{context}\n\n### Response:\n{response}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            texts.append(text)

        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
