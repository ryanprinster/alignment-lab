from functools import partial
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

from experiments.profiler import profile


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
