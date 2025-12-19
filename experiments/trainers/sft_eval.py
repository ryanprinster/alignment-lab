import torch

from experiments.config import SFTConfigBase
from experiments.datasets import TLDRFilteredDataSFT
from experiments.models import HFModel_SFT
from experiments.trainers.base_trainer import BaseTrainer


class SFTEval(BaseTrainer):
    def __init__(self, config: SFTConfigBase):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sft = (
            HFModel_SFT.init_from_hf_pretrained(self.config).to(self.device).requires_grad_(False)
        )
        self.sft.set_from_local_state_dict(self.config.sft_model_path)
        self.gpt = (
            HFModel_SFT.init_from_hf_pretrained(self.config).to(self.device).requires_grad_(False)
        )

        self.sft.eval()
        self.gpt.eval()

        self.data = TLDRFilteredDataSFT(
            tokenizer=self.sft.tokenizer, batch_size=self.config.batch_size
        )

    def _to_device(self, batch):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)
        return batch

    def evaluate(self):
        for _batch_idx, batch in enumerate(self.data.test_loader):
            for subreddit, title, post, summary in zip(
                batch["subreddit"], batch["title"], batch["post"], batch["summary"]
            ):

                query_text = self.data.get_query_text(subreddit, title, post)
                inputs = self.data.tokenizer(query_text, return_tensors="pt")
                inputs = self._to_device(inputs)

                with torch.no_grad():
                    sft_gen_ids, _ = self.sft.generate(
                        inputs,
                        TLDRFilteredDataSFT.SFT_MAX_INPUT_LENGTH,
                        self.config.generation_temperature,
                        do_sample=False,
                    )
                    gpt_gen_ids, _ = self.gpt.generate(
                        inputs,
                        TLDRFilteredDataSFT.SFT_MAX_INPUT_LENGTH,
                        self.config.generation_temperature,
                        do_sample=False,
                    )

                gpt_text = self.data.tokenizer.decode(gpt_gen_ids[0]).split("TL;DR:")[-1]
                sft_text = self.data.tokenizer.decode(sft_gen_ids[0]).split("TL;DR:")[-1]

                print(f"Batch #{_batch_idx}\n")
                print(f"Prompt: {query_text}\n\n")
                print(f"Label: {summary}\n")
                print(f"SFT Response: {sft_text}\n")
                print(f"GPT Response: {gpt_text}\n")
                print(f"===================")
