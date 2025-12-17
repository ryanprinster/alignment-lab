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


class RMEval(BaseTrainer):

    def __init__(self, config: RMConfigBase):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = HFModel_Reward.init_from_hf_pretrained(self.config).to(self.device)
        self.model.set_from_local_state_dict(self.config.rm_model_path)

        self.data = OpenAIPreferenceData(tokenizer=self.model.tokenizer, batch_size=self.config.batch_size)

    @profile
    def validation(self):
        print("Starting Validation!")

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