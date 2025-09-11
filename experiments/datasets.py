from functools import partial
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

from experiments.profiler import profile

class ProfiledDataLoader(DataLoader):
    
    def __iter__(self):
        self._iterator = super().__iter__()
        return self
    
    # @profile
    def __next__(self):
        return next(self._iterator)

#TODO make subclasses?

class TLDRFilteredData():
    SFT_MAX_QUERY_LENGTH = 512
    SFT_MAX_INPUT_LENGTH = 562
    
    @profile
    def __init__(self, tokenizer, batch_size):
        self.dataset = load_dataset("vwxyzjn/summarize_from_feedback_tldr_3_filtered")
        self.tokenizer = tokenizer
        
        preprocess_func = partial(self._sft_tldr_filtered_preprocessor, tokenizer=tokenizer)
        
        dataset = self.dataset.map(preprocess_func, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        self.train_loader = ProfiledDataLoader(dataset["train"], batch_size=batch_size, shuffle=True, num_workers=0)
        self.validation_loader = ProfiledDataLoader(dataset["validation"], batch_size=batch_size, shuffle=True, num_workers=0)
        self.test_loader = ProfiledDataLoader(dataset["test"], batch_size=batch_size, shuffle=True, num_workers=0)
        
    def _sft_tldr_filtered_preprocessor(self, batch, tokenizer=None):
        tokenizer = tokenizer or self.tokenizer
        #  Detail 1 (Dataset -> Specification)
        texts = []

        for subreddit, title, post, summary in zip(batch["subreddit"], batch["title"], batch["post"], batch["summary"]):
            formatted_query = self.get_query_text(subreddit, title, post, tokenizer)
            # Detail 3 (Prepend a leading space to completion; append an EOS token to the completions)
            summary = " " + summary + tokenizer.eos_token
            full_text = formatted_query + summary
            texts.append(full_text)

            # Note: Detail 4 (Dataset -> SFT and preference datasets have different tokenization length)
            #   --> TODO: add check to ensure that full_text is <= SFT_MAX_INPUT_LENGTH
            #   --> TODO: when extending to preference dataset, double cehck summary max token lengths

        # Detail 5 (SFT dataset for SFT training: concatenate the query and the reference summary
        # together and pad from the right)
        return tokenizer(
            texts,
            truncation=False, # Already did ~clever truncation~
            padding="max_length", # Should use tokenizer.pad_token_id
            max_length=TLDRFilteredData.SFT_MAX_INPUT_LENGTH,
            return_tensors="pt"
        )
    
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
        
        if tokens['input_ids'].shape[1] <= TLDRFilteredData.SFT_MAX_QUERY_LENGTH:
            return post
        
        truncate_char = '\n'
        while tokens['input_ids'].shape[1] > TLDRFilteredData.SFT_MAX_QUERY_LENGTH:
            last_newline = post.rfind(truncate_char)
            if last_newline == -1:
                # In the rare case that a single paragraph has too many tokens, 
                # naively truncate that paragraph to the last period instead
                truncate_char = '.'
                # print("Post: ", 
                #       repr(self._format_query(subreddit, title, post)),
                #       tokenizer(self._format_query(subreddit, title, post), return_tensors="pt")['input_ids'].shape[1])
                continue
                raise RuntimeError("All paragraphs removed, post will be empty")
                
            
            post = post[:last_newline]
            test_query = self._format_query(subreddit, title, post)
            tokens = tokenizer(test_query, return_tensors="pt")
        
        return post
        
        

class Dolly15kData():
    
    @profile
    def __init__(self, tokenizer, batch_size, test_size_pct):
        self.dataset = load_dataset("databricks/databricks-dolly-15k")
        preprocess_func = partial(self.dolly15k_preprocessor, tokenizer=tokenizer)
        
        dataset = self.dataset.map(preprocess_func, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataset = dataset["train"].train_test_split(test_size=test_size_pct)
        
        self.train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=True, num_workers=0)
        

    def dolly15k_preprocessor(self, batch, tokenizer, max_length=512):
        texts = []

        for instruction, context, response in zip(batch["instruction"], batch["context"], batch["response"]):
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
            return_tensors="pt"
        )
