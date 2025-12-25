import json
from contextlib import nullcontext
from datetime import datetime

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from experiments.checkpointer import Checkpointer
from experiments.config import RMConfigBase
from experiments.datasets import OpenAIPreferenceData, TLDRFilteredDataSFT
from experiments.logger import Logger
from experiments.models import HFModel_Reward
from experiments.monitor import detect_nans
from experiments.profiler import profile
from experiments.trainers.base_trainer import BaseTrainer


class RMTrainer(BaseTrainer):
    def __init__(self, config: RMConfigBase):
        self.config = config
        self.checkpointer = Checkpointer(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HFModel_Reward.init_from_hf_pretrained(self.config).to(self.device)
        self.model.set_from_local_state_dict(self.config.sft_model_path)
        if self.config.disable_dropout:
            self.model.disable_dropout()

        self.data = OpenAIPreferenceData(
            tokenizer=self.model.tokenizer, batch_size=self.config.batch_size
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.data.dataset["train"]) / self.config._virtual_batch_size,
            #   self.config.num_epochs, # might end up not doing anything, need to do based on global steps
            eta_min=self.config.lr_final_ratio * self.config.lr,
        )
        self.logger = Logger(self.config)

        # Mixed precision training
        self.mixed_precision_context = (
            autocast("cuda", dtype=torch.bfloat16)
            if self.config.enable_mixed_precision_training
            else nullcontext()
        )
        self.scaler = GradScaler("cuda")

    def compute_model_bias(self):
        self.model.eval()
        with torch.no_grad():

            sft_data = TLDRFilteredDataSFT(
                tokenizer=self.model.tokenizer, batch_size=self.config.batch_size
            )

            total_reward = torch.tensor(0.0, device=self.device)
            start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(f"compute_rm_bias{start}.jsonl", "a") as f:

                @profile
                def process_batch(total_reward, _batch_idx, batch):
                    batch["input_ids"] = batch["input_ids"].to(self.device)
                    batch["attention_mask"] = batch["attention_mask"].to(self.device)

                    # Logits are scalar rewards
                    reward_logit = self.model.forward(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    total_reward += torch.sum(reward_logit)

                    running_reward_bias = (
                        (total_reward / ((_batch_idx + 1) * self.config.batch_size)).cpu().item()
                    )

                    log_data = {
                        "batch_idx": _batch_idx,
                        "running_reward_bias": running_reward_bias,
                        "timestamp": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                    }
                    f.write(json.dumps(log_data) + "\n")

                    print(f"running_reward_bias {running_reward_bias}, batch_idx {_batch_idx}")

                    return total_reward

                for _batch_idx, batch in enumerate(sft_data.train_loader):
                    total_reward = process_batch(total_reward, _batch_idx, batch)

    @profile
    def _to_device(self, batch):
        batch["preferred_input_ids"] = batch["preferred_input_ids"].to(self.device)
        batch["preferred_attention_mask"] = batch["preferred_attention_mask"].to(self.device)
        batch["rejected_input_ids"] = batch["rejected_input_ids"].to(self.device)
        batch["rejected_attention_mask"] = batch["rejected_attention_mask"].to(self.device)
        return batch

    @profile
    def _forward(self, batch):
        r_preferred = self.model.forward(
            input_ids=batch["preferred_input_ids"],
            attention_mask=batch["preferred_attention_mask"],
        )
        r_rejected = self.model.forward(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        )
        return (r_preferred, r_rejected)

    @detect_nans
    def _loss(self, r_preferred, r_rejected):
        loss = -torch.mean(torch.log(torch.sigmoid(r_preferred - r_rejected)))
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

    def _accuracy(self, r_preferred, r_rejected):
        correct = (r_preferred > r_rejected).float()
        accuracy = correct.mean()
        return accuracy

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
                    r_preferred, r_rejected = self._forward(batch)
                    loss = self._loss(r_preferred, r_rejected)

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
                    checkpoint_prefix="reward_",
                )

                self.logger.log(
                    scalars={
                        "loss": loss.item(),
                        "accuracy": self._accuracy(r_preferred, r_rejected).item(),
                        "r_preferred": torch.mean(r_preferred).item(),
                        "r_rejected": torch.mean(r_rejected).item(),
                        "r_delta": torch.mean(r_preferred - r_rejected).item(),
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
            checkpoint_prefix="reward_",
            final_checkpoint=True,
        )
