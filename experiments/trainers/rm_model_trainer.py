import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext
import pdb
from datetime import datetime
import json

from experiments.models import HFModel_Reward, HFModel_TokenClassification
from experiments.config import RMConfigBase
from experiments.datasets import OpenAIPreferenceData, TLDRFilteredDataSFT
from experiments.logger import Logger
from experiments.checkpointer import Checkpointer
from experiments.profiler import profile
from experiments.monitor import detect_nans
from experiments.trainers.base_trainer import BaseTrainer


class RMTrainer(BaseTrainer):
    def __init__(self, config: RMConfigBase):
        self.config = config
        self.checkpointer = Checkpointer(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = HFModel_Reward.init_from_hf_pretrained(self.config).to(self.device)
        self.model.set_from_local_state_dict(self.config.sft_model_path) # we

        self.data = OpenAIPreferenceData(tokenizer=self.model.tokenizer, batch_size=self.config.batch_size)
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                    lr = self.config.lr)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, 
                                              T_max=len(self.data.dataset["train"]) / self.config._virtual_batch_size,
                                            #   self.config.num_epochs, # might end up not doing anything, need to do based on global steps
                                              eta_min=self.config.lr_final_ratio * self.config.lr)
        self.logger = Logger(self.config)

        # Mixed precision training
        self.mixed_precision_context = autocast("cuda", dtype=torch.bfloat16) if self.config.enable_mixed_precision_training else nullcontext()
        self.scaler = GradScaler("cuda") 

    def compute_model_bias(self):
        self.model.eval()
        with torch.no_grad():

            sft_data = TLDRFilteredDataSFT(tokenizer=self.model.tokenizer, batch_size=self.config.batch_size)
            
            total_reward = torch.tensor(0.0, device=self.device)
            start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(f"compute_rm_bias{start}.jsonl", "a") as f:
                @profile
                def process_batch(total_reward, _batch_idx, batch):
                    batch['input_ids'] = batch['input_ids'].to(self.device)
                    batch['attention_mask'] = batch['attention_mask'].to(self.device)
                    
                    # Logits are scalar rewards
                    reward_logit = self.model.forward(input_ids=batch['input_ids'], 
                                attention_mask=batch['attention_mask']) 
                    total_reward += torch.sum(reward_logit)
                
                    running_reward_bias = (total_reward / ((_batch_idx + 1) * self.config.batch_size)).cpu().item()
                    
                    log_data = {"batch_idx": _batch_idx,
                                "running_reward_bias": running_reward_bias,
                                "timestamp": datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}
                    f.write(json.dumps(log_data) + "\n")

                    print(f"running_reward_bias {running_reward_bias}, batch_idx {_batch_idx}")

                    return total_reward

                for _batch_idx, batch in enumerate(sft_data.train_loader):
                    total_reward = process_batch(total_reward, _batch_idx, batch)
             
    @profile
    def _to_device(self, batch):
        batch['preferred_input_ids'] = batch['preferred_input_ids'].to(self.device)
        batch['preferred_attention_mask'] = batch['preferred_attention_mask'].to(self.device)
        batch['rejected_input_ids'] = batch['rejected_input_ids'].to(self.device)
        batch['rejected_attention_mask'] = batch['rejected_attention_mask'].to(self.device)
        return batch

    @profile
    def _forward(self, batch):
        r_preferred = self.model.forward(input_ids=batch['preferred_input_ids'], 
                            attention_mask=batch['preferred_attention_mask']) 
        r_rejected = self.model.forward(input_ids=batch['rejected_input_ids'], 
                            attention_mask=batch['rejected_attention_mask'])
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
                
                # FP32 --> FP16 for mixed precision training
                with self.mixed_precision_context: 
                    r_preferred, r_rejected = self._forward(batch)
                    loss = self._loss(r_preferred, r_rejected)

                self._backward(loss)
                
                if (self.global_step+1) % self.config.accumulation_steps == 0:
                    self._update_weights()
                
                self.global_step += 1

                self.checkpointer.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.global_step,
                    epoch,
                    loss.item(),
                    checkpoint_prefix="reward_"
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
    def validation(self):
        from datetime import datetime
        import json
        print("Starting Validation!")

        self.model = HFModel_Reward.init_from_hf_pretrained(self.config).to(self.device)
        self.model.set_from_local_state_dict(self.config.rm_model_path)
        
        self.model.eval()

        total_correct = 0
        total_examples = 0
        with torch.no_grad():                     
            for _batch_idx, batch in enumerate(self.data.validation_loader):

                batch = self._to_device(batch)
                
                # FP32 --> FP16 for mixed precision 
                with self.mixed_precision_context: 
                    outputs = self._forward(batch)
                    
                    # Logits are scalar rewards
                    r_preferred = outputs[0]
                    r_rejected = outputs[1]

                    correct = (r_preferred > r_rejected).float()
                    
                    total_correct += correct.sum().item()
                    total_examples += correct.size(0)

                print(f"\n\nPreferred (reward: {r_preferred[0]})\n ", self.model.tokenizer.decode(batch['preferred_input_ids'][0]), "\n\n")
                
                print(f"\n\nRejected: (reward: {r_rejected[0]})\n ", self.model.tokenizer.decode(batch['rejected_input_ids'][0]), "\n\n")                
                
                print(f"step: {_batch_idx}, cumulative accuracy: {1.0 * total_correct / total_examples}")
            
        now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        log_data = {"step": _batch_idx, "cumulative_accuracy": 1.0 * total_correct / total_examples, "timestamp": now}
        with open(f"rm_validation_{now}.jsonl", "a") as f:
            f.write(json.dumps(log_data) + "\n")


    # def test_reward_model(prompt, with_eos=True):

    #     if with_eos:
    #         prompt += '<|end_of_text|>'
    #     x = self.data.tokenizer.encode(prompt)
    #     x = torch.tensor(x)
    #     x_padded = F.pad(x, (0, self.data.RM_MAX_INPUT_LENGTH - x.size(0)), value=self.data.tokenizer.pad_token_id)
    #     attn_mask = (x_padded != self.data.tokenizer.pad_token_id).long()
    #     y = self.model.forward(input_ids=x_padded.unsqueeze(0).to(self.device), attention_mask=attn_mask.unsqueeze(0).to(self.device))
    #     return y
    
     
