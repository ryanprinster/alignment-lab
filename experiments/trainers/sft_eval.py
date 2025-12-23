import torch

from experiments.config import SFTConfigBase
from experiments.datasets import TLDRFilteredDataPPO
from experiments.models import HFModel_SFT
from experiments.trainers.base_trainer import BaseTrainer

import pdb


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

        self.data = TLDRFilteredDataPPO(
            tokenizer=self.sft.tokenizer, batch_size=self.config.batch_size
        )

    def _to_device(self, batch):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)
        return batch

    def evaluate(self):
        for _batch_idx, batch in enumerate(self.data.test_loader):
            with torch.no_grad():
                batch = self._to_device(batch)

                sft_gen_ids, _ = self.sft.generate(
                    batch,
                    self.data.SFT_MAX_INPUT_LENGTH,
                    self.config.generation_temperature,
                )
                gpt_gen_ids, _ = self.gpt.generate(
                    batch,
                    self.data.SFT_MAX_INPUT_LENGTH,
                    self.config.generation_temperature,
                )
            
    
            full_gpt_text = self.data.tokenizer.decode(gpt_gen_ids[0]).split("TL;DR:")
            prompt, gpt_text = full_gpt_text[0], "".join(full_gpt_text[1:])
            full_sft_text = self.data.tokenizer.decode(sft_gen_ids[0]).split("TL;DR:")
            sft_text = "".join(full_sft_text[1:])

            print(f"Batch #{_batch_idx}\n")
            print(f"Prompt: {prompt}\n\n")
            print(f"Label: {batch['summary'][0]}\n")
            print(f"SFT Response: {sft_text}\n")
            print(f"GPT Response: {gpt_text}\n")
            print(f"===================")
