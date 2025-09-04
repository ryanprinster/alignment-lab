import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

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
        self.model = Llama_3p2_1B().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                    lr = self.config.lr)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, 
                                              T_max=self.config.num_epochs, 
                                              eta_min=self.config.lr_final_ratio * self.config.lr)
        self.data = TLDRFilteredData(tokenizer=self.model.tokenizer, batch_size=self.config.batch_size)
        self.checkpointer = Checkpointer(self.config)
        self.logger = Logger(self.config)

    @detect_nans
    def loss(self, outputs):
        # This model does CE loss under the hood
        return outputs.loss

    @profile
    def backward(self, loss):
        loss.backward()
    
    @profile
    def update_weights(self):
        if (self.global_step+1) % self.config.accumulation_steps == 0:
            self.optimizer.step()
        self.global_step += 1
    
    def zero_grad(self):
        if (self.global_step+1) % self.config.accumulation_steps == 0:
            self.optimizer.zero_grad()
    
    def train(self):
        print("Starting Training!")
        self.global_step = 0
        self.model.train()

        for epoch in range(self.config.num_epochs):   
                     
            for _batch_idx, batch in enumerate(self.data.train_loader):
                
                outputs = self.model.forward(input_ids=batch['input_ids'].to(self.device), 
                                       attention_mask=batch['attention_mask'].to(self.device), 
                                       labels=batch['input_ids'].to(self.device)) 
                loss = self.loss(outputs)
                self.backward(loss)
                self.update_weights()

                self.checkpointer.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.global_step,
                    epoch,
                    loss.item()
                )

                self.logger.log(
                    scalars={"loss": loss.item()},
                    models=[self.model],
                    epoch=epoch,
                    global_step=self.global_step,
                    lr=self.lr_scheduler.get_last_lr()[0]
                )

                self.zero_grad()
            
            self.lr_scheduler.step()


