import json
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from experiments.config import RMConfigBase
from experiments.datasets import OpenAIPreferenceData
from experiments.models import HFModel_Reward
from experiments.profiler import profile
from experiments.trainers.base_trainer import BaseTrainer


class RMEval(BaseTrainer):

    def __init__(self, config: RMConfigBase):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.model = HFModel_Reward.init_from_hf_pretrained(self.config).to(self.device)
        # self.model.set_from_local_state_dict(self.config.rm_model_path)

        self.data = OpenAIPreferenceData(
            tokenizer=self.model.tokenizer, batch_size=self.config.batch_size
        )

    def _to_device(self, batch):
        batch["preferred_input_ids"] = batch["preferred_input_ids"].to(self.device)
        batch["preferred_attention_mask"] = batch["preferred_attention_mask"].to(self.device)
        batch["rejected_input_ids"] = batch["rejected_input_ids"].to(self.device)
        batch["rejected_attention_mask"] = batch["rejected_attention_mask"].to(self.device)
        return batch

    @profile
    def _forward(self, batch):
        rewards = self.model.forward(
            input_ids=torch.cat([batch["preferred_input_ids"], batch["rejected_input_ids"]], dim=0),
            attention_mask=torch.cat([batch["preferred_attention_mask"], batch["rejected_attention_mask"]], dim=0),
        )
        batch_size = batch["preferred_input_ids"].shape[0]
        return rewards[:batch_size], rewards[batch_size:]


    @profile
    def validation(self):
        print("Starting Validation!")

        self.model.eval()

        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for _batch_idx, batch in enumerate(self.data.validation_loader):
                batch = self._to_device(batch)

                outputs = self._forward(batch)

                # Logits are scalar rewards
                r_preferred = outputs[0]
                r_rejected = outputs[1]

                correct = (r_preferred > r_rejected).float()

                total_correct += correct.sum().item()
                total_examples += correct.size(0)

                print(
                    f"\n\nPreferred (reward: {r_preferred[0]})\n ",
                    self.model.tokenizer.decode(batch["preferred_input_ids"][0]),
                    "\n\n",
                )
                print(
                    f"\n\nRejected: (reward: {r_rejected[0]})\n ",
                    self.model.tokenizer.decode(batch["rejected_input_ids"][0]),
                    "\n\n",
                )
                print(
                    f"step: {_batch_idx}, cumulative accuracy: {1.0 * total_correct / total_examples}"
                )

        now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        log_data = {
            "step": _batch_idx,
            "cumulative_accuracy": 1.0 * total_correct / total_examples,
            "timestamp": now,
        }
        with open(f"rm_validation_{now}.jsonl", "a") as f:
            f.write(json.dumps(log_data) + "\n")

    def _test_reward_model(self, prompt, with_eos=True, with_prefix=False):
        if with_eos:
            prompt += "<|end_of_text|>"
        if with_prefix:
            prompt = "SUBREDDIT: r/interactive\n\nTITLE: Interactive Test\n\nPOST: " + prompt
        x = self.data.tokenizer.encode(prompt)
        x = torch.tensor(x)
        x_padded = F.pad(
            x,
            (0, self.data.RM_MAX_INPUT_LENGTH - x.size(0)),
            value=self.data.tokenizer.pad_token_id,
        )
        attn_mask = (x_padded != self.data.tokenizer.pad_token_id).long()
        y = self.model.forward(
            input_ids=x_padded.unsqueeze(0).to(self.device),
            attention_mask=attn_mask.unsqueeze(0).to(self.device),
        )
        return y
    
    

    def human_test(self):
        self.model.eval()

        while True:
            prompt_input = input("Prompt> ")
            if prompt_input.lower() in ("quit", "exit"):
                break

            summary_input = input("Summary> ")
            if summary_input.lower() in ("quit", "exit"):
                break
            output = self._test_reward_model(prompt_input + "\n\nTL;DR: " + summary_input)

            print(output)


    def plot_train_curves(self):
        def smooth(values, weight=0.6):
            """exp moving avg"""
            smoothed = []
            last = values[0]
            
            for value in values:
                smoothed_val = last * weight + value * (1 - weight)
                smoothed.append(smoothed_val)
                last = smoothed_val
            
            return smoothed

        file = self.config.rm_training_log_path
        losses = []
        accuracies = []
        steps = []
        r_rejected = []
        r_delta = []

        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data['global_step'] % 5 == 0:
                    steps.append(data['global_step'])
                    losses.append(data['loss']) 
                    accuracies.append(data['accuracy'])
                    r_rejected.append(data['r_rejected'])
                    r_delta.append(data['r_delta'])


        # Plot
        # plt.figure(figsize=(6, 4))
        # plt.plot(steps, losses, alpha=0.15, color='#2ca02c', linewidth=2.5,)
        # plt.plot(steps, smooth(losses, weight=0.9), alpha=1.0, color='#2ca02c', linewidth=2.5, label="SFT")
        # plt.ylabel('RM Loss')
        
        # plt.plot(steps, accuracies, alpha=0.15, color='#2ca02c', linewidth=2.5,)
        # plt.plot(steps, smooth(accuracies, weight=0.9), alpha=1.0, color='#2ca02c', linewidth=2.5, label="SFT")
        # plt.ylabel('Training Accuracy')

        # plt.plot(steps, r_rejected, alpha=0.15, color='#2ca02c', linewidth=2.5,)
        # plt.plot(steps, smooth(r_rejected, weight=0.9), alpha=1.0, color='#2ca02c', linewidth=2.5, label="SFT")
        # plt.ylabel('Rejected Rewards')



        plt.plot(steps, r_delta, alpha=0.15, color='#2ca02c', linewidth=2.5,)
        plt.plot(steps, smooth(r_delta, weight=0.9), alpha=1.0, color='#2ca02c', linewidth=2.5, label="SFT")
        plt.ylabel('Reward Delta (Preferred - Rejected) ')


        plt.xlabel('Step')
        plt.title('Reward Model')  

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # plt.savefig('rm_loss_curve.png', dpi=150)
        plt.show()


    def create_validation_agreement_request(self):
        print("Starting Agreement Calculation!")

        self.model.eval()

        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for _batch_idx, batch in enumerate(self.data.validation_loader):
                batch['pre']



                

        now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        log_data = {
            "step": _batch_idx,
            "cumulative_accuracy": 1.0 * total_correct / total_examples,
            "timestamp": now,
        }
        with open(f"rm_validation_{now}.jsonl", "a") as f:
            f.write(json.dumps(log_data) + "\n")
