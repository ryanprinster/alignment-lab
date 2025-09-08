import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from contextlib import nullcontext


from experiments.models import Llama_3p2_1B
from experiments.config import SFTConfig2
from experiments.datasets import TLDRFilteredData
from experiments.logger import Logger
from experiments.checkpointer import Checkpointer
from experiments.profiler import profile
from experiments.monitor import detect_nans


class SFTTrainer():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Llama_3p2_1B(self.config).to(self.device)
        self.data = TLDRFilteredData(tokenizer=self.model.tokenizer, batch_size=self.config.batch_size)
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
    def to_device(self, batch):
        batch['input_ids'] = batch['input_ids'].to(self.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.device)
        batch['labels'] = batch['input_ids']  # reference
        return batch

    @detect_nans
    def loss(self, outputs):
        # This model does CE loss under the hood
        return outputs.loss

    @profile
    def backward(self, loss):
        if self.config.enable_mixed_precision_training:
            # Loss scaling for mixed precision training
            # Note: do this only in backward pass, because otherwise we are logging with a scaled loss 
            loss = self.scaler.scale(loss)
        loss.backward()
    
    @profile
    def update_weights(self):
        if self.config.enable_mixed_precision_training:
            # Unscale gradient, take optimizer step, and update scale factor
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.lr_scheduler.step()
    
    @profile
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    
    @profile
    def train(self):
        print("Starting Training!")
        self.global_step = 0
        self.model.train()

        for epoch in range(self.config.num_epochs):   
                     
            for _batch_idx, batch in enumerate(self.data.train_loader):

                batch = self.to_device(batch)
                
                # FP32 --> FP16 for mixed precision training
                with self.mixed_precision_context: 
                    outputs = self.model.forward(input_ids=batch['input_ids'], 
                                        attention_mask=batch['attention_mask'], 
                                        labels=batch['labels']) 
                    
                    loss = self.loss(outputs)

                self.backward(loss)
                
                if (self.global_step+1) % self.config.accumulation_steps == 0:
                    self.update_weights()
                
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
                        "loss": loss.item()},
                    models=[self.model],
                    epoch=epoch,
                    global_step=self.global_step,
                    lr=self.lr_scheduler.get_last_lr()[0]
                )

                if (self.global_step+1) % self.config.accumulation_steps == 0:
                    self.zero_grad()
                            

