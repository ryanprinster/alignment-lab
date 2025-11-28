import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext


from experiments.models import HFModel_SFT
from experiments.config import SFTConfigBase
from experiments.datasets import TLDRFilteredDataSFT
from experiments.logger import Logger
from experiments.checkpointer import Checkpointer
from experiments.profiler import profile
from experiments.monitor import detect_nans
from experiments.trainers.base_trainer import BaseTrainer

import pdb

class SFTTrainer(BaseTrainer):
    def __init__(self, config: SFTConfigBase):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HFModel_SFT(self.config,
                                hf_model_name=self.config.hf_sft_model_name,
                                hf_model_revision=self.config.hf_sft_model_revision
                                ).to(self.device)
        self.data = TLDRFilteredDataSFT(tokenizer=self.model.tokenizer, batch_size=self.config.batch_size)
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                    lr = self.config.lr)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, 
                                              T_max=len(self.data.dataset["train"]) / self.config._virtual_batch_size,
                                            #   self.config.num_epochs, # might end up not doing anything, need to do based on global steps
                                              eta_min=self.config.lr_final_ratio * self.config.lr)
        self.checkpointer = Checkpointer(self.config)
        self.logger = Logger(self.config)

        # Mixed precision training
        self.mixed_precision_context = autocast("cuda") if self.config.enable_mixed_precision_training else nullcontext()
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
        if self.config.enable_mixed_precision_training:
            # Loss scaling for mixed precision training
            # Note: do this only in backward pass, because otherwise we are logging with a scaled loss 
            loss = self.scaler.scale(loss)
        loss.backward()
    
    @profile
    def _update_weights(self):
        if self.config.enable_mixed_precision_training:
            # Unscale gradient, take optimizer step, and update scale factor
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
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
                
                # FP32 --> FP16 for mixed precision training
                with self.mixed_precision_context: 
                    _, loss = self.model.forward(input_ids=batch['input_ids'], 
                                        attention_mask=batch['attention_mask'], 
                                        labels=batch['input_ids']) 
                    
                    loss = self._loss(loss)

                self._backward(loss)
                
                if (self.global_step+1) % self.config.accumulation_steps == 0:
                    self._update_weights()
                
                self.global_step += 1

                self.checkpointer.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.global_step,
                    epoch,
                    loss.item()
                )

                self.logger.log(
                    scalars={
                        "loss": loss.item(),
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "lr": self.lr_scheduler.get_last_lr()[0]
                        },
                    models=[self.model]
                )

                if (self.global_step+1) % self.config.accumulation_steps == 0:
                    self._zero_grad()
        
        # Final checkpoint
        self.checkpointer.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.global_step,
                    epoch,
                    loss.item(),
                    final_checkpoint=True
                )
                            

    @profile
    def evaluate(self):
        with torch.no_grad():
            max_summary_length = TLDRFilteredDataSFT.SFT_MAX_INPUT_LENGTH

            self.sft = Llama_3p2_1B_SFT(self.config).to(self.device)
            self.gpt = Llama_3p2_1B_SFT(self.config).to(self.device)

            self.sft.eval()
            self.gpt.eval()
            
            self.checkpointer.load_model(self.config.load_checkpoint_path, self.sft, self.device)
        
            for _batch_idx, batch in enumerate(self.data.test_loader):
                for subreddit, title, post, summary in zip(batch["subreddit"], batch["title"], batch["post"], batch["summary"]):

                    query_text = self.data.get_query_text(subreddit, title, post)
                    inputs = self.data.tokenizer(query_text, return_tensors="pt")
                    inputs = self._to_device(inputs)

                    sft_gen_ids, _ = self.sft.generate(inputs, max_summary_length, self.config.generation_temperature, do_sample=False)
                    gpt_gen_ids, _ = self.gpt.generate(inputs, max_summary_length, self.config.generation_temperature, do_sample=False)

                    pdb.set_trace()

                    gpt_text = self.data.tokenizer.decode(gpt_gen_ids[0]).split('TL;DR:')[-1]
                    sft_text = self.data.tokenizer.decode(sft_gen_ids[0]).split('TL;DR:')[-1]

                    print(f"Batch #{_batch_idx}\n")
                    print(f"Prompt: {query_text}\n\n")
                    print(f"Label: {summary}\n")
                    print(f"SFT Response: {sft_text}\n")
                    print(f"GPT Response: {gpt_text}\n")
                    print(f"===================")
            
            
