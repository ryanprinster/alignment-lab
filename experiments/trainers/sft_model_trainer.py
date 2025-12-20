import pdb
from contextlib import nullcontext

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from experiments.checkpointer import Checkpointer
from experiments.config import SFTConfigBase
from experiments.datasets import TLDRFilteredDataSFT
from experiments.logger import Logger
from experiments.models import HFModel_SFT
from experiments.monitor import detect_nans
from experiments.profiler import profile
from experiments.trainers.base_trainer import BaseTrainer


class SFTTrainer(BaseTrainer):
    def __init__(self, config: SFTConfigBase):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HFModel_SFT.init_from_hf_pretrained(self.config).to(self.device)
        if self.config.disable_dropout:
            self.model.disable_dropout()

        self.data = TLDRFilteredDataSFT(
            tokenizer=self.model.tokenizer, batch_size=self.config.batch_size
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.data.dataset["train"]) / self.config._virtual_batch_size,
            eta_min=self.config.lr_final_ratio * self.config.lr,
        )
        self.checkpointer = Checkpointer(self.config)
        self.logger = Logger(self.config)

        # Mixed precision training
        self.mixed_precision_context = (
            autocast("cuda", dtype=torch.bfloat16)
            if self.config.enable_mixed_precision_training
            else nullcontext()
        )
        self.scaler = GradScaler("cuda")

    @profile
    def _to_device(self, batch):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)
        return batch

    @detect_nans
    def _loss(self, loss):
        # This model does CE loss under the hood
        return loss

    @profile
    def _backward(self, loss):
        loss.backward()

    @profile
    def _update_weights(self):
        self.optimizer.step()
        self.lr_scheduler.step()

    @profile
    def _zero_grad(self):
        self.optimizer.zero_grad()

    @profile
    def train(self):
        print("Starting Training!")
        self.global_step = 0
        self.model.train()

        for epoch in range(self.config.num_epochs):

            for _batch_idx, batch in enumerate(self.data.train_loader):

                batch = self._to_device(batch)

                # FP32 --> BF16 for mixed precision training
                with self.mixed_precision_context:
                    _, loss = self.model.forward(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["input_ids"],
                    )

                    loss = self._loss(loss)

                self._backward(loss)

                if (self.global_step + 1) % self.config.accumulation_steps == 0:
                    self._update_weights()

                self.global_step += 1

                self.checkpointer.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.global_step,
                    epoch,
                    loss.item(),
                    checkpoint_prefix="sft_no_dropout_",
                )

                self.logger.log(
                    scalars={
                        "loss": loss.item(),
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "lr": self.lr_scheduler.get_last_lr()[0],
                    },
                    models=[self.model],
                )

                if (self.global_step + 1) % self.config.accumulation_steps == 0:
                    self._zero_grad()


        # Final checkpoint
        self.checkpointer.save_checkpoint(
            self.model,
            None,  # No optimizer needed
            self.global_step,
            epoch,
            loss.item(),
            checkpoint_prefix="sft_no_dropout_",
            final_checkpoint=True,
        )
