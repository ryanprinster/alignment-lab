import torch
import json
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer


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
    
    def _format_batch_prompts_and_summaries(self, batch):
        prompt_ids = []
        reference_summary_ids = []
        for subreddit, title, post, summary in zip(
            batch["subreddit"], batch["title"], batch["post"], batch["summary"]
        ):
            formatted_query = self.data.get_query_text(subreddit, title, post)
            prompt_ids.append(self.data.tokenizer.encode(formatted_query, return_tensors="pt").squeeze(0))
            reference_summary_ids.append(self.data.tokenizer.encode(summary, return_tensors="pt").squeeze(0))

        return prompt_ids, reference_summary_ids

    def evaluate(self):

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_total, rouge_count = 0, 0

        for _batch_idx, batch in enumerate(self.data.test_loader):
            with torch.no_grad():
                batch = self._to_device(batch)

                _full_texts, query_texts, ref_summary_texts = self.data.format_batch(batch)

                inputs = self.data.tokenize_and_pad_left(query_texts, self.data.SFT_MAX_QUERY_LENGTH)
                inputs = self._to_device(inputs)

                sft_gen_ids, _ = self.sft.generate(
                    inputs,
                    self.data.SFT_MAX_INPUT_LENGTH,
                    self.config.generation_temperature,
                    do_sample=False,
                )
                gpt_gen_ids, _ = self.gpt.generate(
                    inputs,
                    self.data.SFT_MAX_INPUT_LENGTH,
                    self.config.generation_temperature,
                    do_sample=False,
                )

                sft_response_ids = sft_gen_ids[:, self.data.SFT_MAX_QUERY_LENGTH:]
                gpt_response_ids = gpt_gen_ids[:, self.data.SFT_MAX_QUERY_LENGTH:]

                sft_response_texts = self.data.tokenizer.batch_decode(sft_response_ids, skip_special_tokens=True)
                gpt_response_texts = self.data.tokenizer.batch_decode(gpt_response_ids, skip_special_tokens=True)
                
                
                # Calculate rouge scores
                for i, ref_text in enumerate(ref_summary_texts):
                    
                    rougeL_score = scorer.score(ref_text, sft_response_texts[i])['rougeL'].fmeasure
                    rouge_total += rougeL_score
                    rouge_count += 1
                
                print(f"Running ROUGE {rouge_total/rouge_count}\n")

                print(f"===================")
                print(f"Batch #{_batch_idx}\n")
                print(f"Prompt: {query_texts[0]}\n\n")
                print(f"Label: {ref_summary_texts[0]}\n")
                print(f"SFT Response: {sft_response_texts[0]}\n")
                print(f"GPT Response: {gpt_response_texts[0]}\n")
                print(f"===================")

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

        file = self.config.sft_training_log_path
        losses = []
        steps = []

        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data['global_step'] % 5 == 0:
                    steps.append(data['global_step'])
                    losses.append(data['loss']) 


        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, alpha=0.15, color='#2ca02c', linewidth=2.5,)
        plt.plot(steps, smooth(losses, weight=0.7), alpha=1.0, color='#2ca02c', linewidth=2.5, label="SFT")
        plt.xlabel('Step')
        plt.ylabel('SFT Loss')
        plt.title('SFT')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('sft_loss_curve.png', dpi=150)
        plt.show()
