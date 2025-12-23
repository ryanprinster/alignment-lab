# Standard library imports
import json
import math
import os

import anthropic

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.checkpointer import Checkpointer
from experiments.config import PPOConfigBase
from experiments.datasets import TLDRFilteredDataPPO
from experiments.models import HFModel_Policy
from experiments.profiler import profile
from experiments.trainers.base_trainer import BaseTrainer

import pdb


class PPORLHFEval(BaseTrainer):

    @profile
    def __init__(self, config: PPOConfigBase):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.checkpointer = Checkpointer(self.config)

        # Trained PPO Model
        self.model = (
            HFModel_Policy.init_from_hf_pretrained(self.config)
            .to(self.device)
            .requires_grad_(False)
        )
        self.model.set_from_local_state_dict(self.config.policy_checkpoint_path)

        # SFT Model
        # self.model = HFModel_SFT.init_from_hf_pretrained(self.config).to(self.device).requires_grad_(False)
        # self.model.set_from_local_state_dict(self.config.sft_model_path)

        # self.model = HFModel_Policy.init_from_hf_pretrained(
        #     config=self.config,
        #     hf_model_name="vwxyzjn/EleutherAI_pythia-1b-deduped__ppo_left_padding_new_nowhiten_reward__tldr",
        #     revision="ppo_left_padding_new_nowhiten_reward__77713__1709671965").to(self.device).requires_grad_(False)

        self.data = TLDRFilteredDataPPO(
            tokenizer=self.model.tokenizer, batch_size=self.config.batch_size
        )

    def human_generate_summary(self):
        """Interactive loop: generates summaries for user-provided prompts."""
        self.model.eval()

        while True:
            prompt_input = input("\nPrompt> ")
            if prompt_input.lower() in ("quit", "exit"):
                break
            print("")

            self.data.get_query_text(subreddit="interactive", title="Interactive Test", post=prompt_input)
            pdb.set_trace()
            batch_inputs, _ = self.format_batch_for_generation(
                {
                    "subreddit": ["interactive"],
                    "title": ["Interactive Test"],
                    "post": [prompt_input],
                    "summary": [""],
                },
                self.data.__class__.SFT_MAX_QUERY_LENGTH,
            )
            del _

            generated = self.generate_summaries(batch_inputs)
            summary_text = self.data.tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"Generated Summary: {summary_text}\n")

    def _to_device(self, batch):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)
        return batch

    @staticmethod
    def _judge_prompt(post, summary_a, summary_b):
        return f"""Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? Judge based on accuracy, coverage, and coherence.

            Post:
            {post}

            Summary A:
            {summary_a}

            Summary B:
            {summary_b}

            FIRST provide a one-sentence comparison of the two summaries, explaining which you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format:
            Comparison: <one-sentence comparison and explanation>
            Preferred: <"A" or "B">"""

    @staticmethod
    def _claude_request_json(idx, post, summary_a, summary_b):
        return {
            "custom_id": f"comparison-{idx}",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 200,
                "messages": [
                    {
                        "role": "user",
                        "content": PPORLHFEval._judge_prompt(post, summary_a, summary_b),
                    }
                ],
            },
        }

    def trim_tensor(self, tensor):
        """
        Trims tokenized tensor of pad, eos, and bos tokens
        """
        trim_ids = [
            self.data.tokenizer.pad_token_id,
            self.data.tokenizer.eos_token_id,
            self.data.tokenizer.bos_token_id,
        ]

        left = 0
        right = len(tensor) - 1

        while left <= right and tensor[left].item() in trim_ids:
            left += 1
        while right >= left and tensor[right].item() in trim_ids:
            right -= 1

        return tensor[left : right + 1]

    def format_batch_for_generation(self, batch, max_query_length):
        input_ids = []
        attention_masks = []
        summary_ids = []
        for subreddit, title, post, summary in zip(
            batch["subreddit"], batch["title"], batch["post"], batch["summary"]
        ):
            query_text = self.data.get_query_text(subreddit, title, post)
            inputs = self.data.tokenizer(query_text, return_tensors="pt")
            inputs["input_ids"] = inputs["input_ids"].squeeze()
            inputs["attention_mask"] = inputs["attention_mask"].squeeze()
            inputs = self._to_device(inputs)
            input_ids.append(
                torch.nn.functional.pad(
                    inputs["input_ids"],
                    (max_query_length - inputs["input_ids"].size(0), 0),
                    value=self.data.tokenizer.pad_token_id,
                )
            )
            attention_masks.append(
                torch.nn.functional.pad(
                    inputs["attention_mask"],
                    (max_query_length - inputs["attention_mask"].size(0), 0),
                    value=0,
                )
            )

            summary_ids.append(
                self.data.tokenizer(summary, return_tensors="pt")["input_ids"].squeeze()
            )
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
        }, summary_ids

    @profile
    def torch_batch_to_request(self, prompts, summary_ids, generated_summaries):
        for i in range(len(summary_ids)):  # enumerate through batch
            prompt_ids = self.trim_tensor(prompts[i])
            gen_sum_ids = self.trim_tensor(generated_summaries[i])
            ref_sum_ids = self.trim_tensor(summary_ids[i])

            prompt_text = self.data.tokenizer.decode(prompt_ids)
            generated_summary_text = self.data.tokenizer.decode(
                gen_sum_ids, skip_special_tokens=True
            )
            reference_summary_text = self.data.tokenizer.decode(ref_sum_ids)
            request = PPORLHFEval._claude_request_json(
                len(self.requests),
                prompt_text,
                generated_summary_text,
                reference_summary_text,
            )
            self.requests.append(request)

            self.summaries.append(
                {
                    "generated": generated_summary_text,
                    "reference": reference_summary_text,
                    "prompt": prompt_text,
                    "log(len(gen)/len(ref))": math.log(gen_sum_ids.size(0) / ref_sum_ids.size(0)),
                }
            )

            print(f"generated_summary_text: {generated_summary_text}\n")
        print(
            "\n\n\n",
            PPORLHFEval._judge_prompt(prompt_text, generated_summary_text, reference_summary_text),
            "\n\n\n",
        )

    def generate_summaries(self, input_batch):
        full_states, _ = self.model.generate(
            input_batch,
            self.data.__class__.SFT_MAX_INPUT_LENGTH,  # ???
            self.config.generation_temperature,
            max_query_length=self.data.__class__.SFT_MAX_QUERY_LENGTH,
        )
        del _
        generated_summaries = full_states[:, self.data.__class__.SFT_MAX_QUERY_LENGTH :]
        del full_states
        return generated_summaries

    def format_batch_prompts_and_summaries(self, batch):
        prompt_ids = []
        reference_summary_ids = []
        for subreddit, title, post, summary in zip(
            batch["subreddit"], batch["title"], batch["post"], batch["summary"]
        ):
            formatted_query = self.data.get_query_text(subreddit, title, post)
            pdb.set_trace()
            prompt_ids.append(self.data.tokenizer.encode(formatted_query)['input_ids'])
            reference_summary_ids.append(self.data.tokenizer.encode(summary))

        return prompt_ids, reference_summary_ids

    def construct_claude_request(self):
        self.model.eval()

        self.requests = []
        self.summaries = []

        for batch_idx, batch in enumerate(self.data.validation_loader):
            print(f"Preparing batch {batch_idx}")
            
            prompt_ids, reference_summary_ids = self.format_batch_prompts_and_summaries(batch)
            generated_summary_ids = self.generate_summaries(batch)

            self.torch_batch_to_request(prompt_ids, reference_summary_ids, generated_summary_ids)

        pdb.set_trace()
        assert(False)
        print("finished creating batched requests")
        batch = self.client.messages.batches.create(requests=self.requests)

        with open(f"summaries_{batch.id}.jsonl", "w") as f:
            for summary in self.summaries:
                f.write(json.dumps(summary) + "\n")
        print(f"Submitted comparisons: {batch.id}")

    def download_batch_results(
        self,
        batch_id="msgbatch_01AhfkjK4M3996M5wbuSepU4",
        summaries_file="summaries_msgbatch_01AhfkjK4M3996M5wbuSepU4.jsonl",
        output_file="batch_results_paper_ppo_v2.jsonl",
    ):
        """Download batch results and save to file"""

        batch_id = self.config.batch_id or batch_id
        summaries_file = self.config.summaries_file or summaries_file
        output_file = self.config.batch_results_file_name or output_file

        # Load summaries from file
        import json

        summaries = []
        with open(summaries_file, "r") as f:
            for line in f:
                summaries.append(json.loads(line))
        print(f"Loaded {len(summaries)} summary pairs from {summaries_file}")

        # Check if batch is complete
        batch = self.client.messages.batches.retrieve(batch_id)
        print(f"Status: {batch.processing_status}")
        print(f"Succeeded: {batch.request_counts.succeeded}")
        print(f"Errored: {batch.request_counts.errored}")

        if batch.processing_status != "ended":
            print(f"Batch not ready yet. Current status: {batch.processing_status}")
            return None

        # Download all results
        results = []
        for result in self.client.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                response_text = result.result.message.content[0].text

                idx = int(result.custom_id.split("-")[1])
                summary_pair = summaries[idx] if idx < len(summaries) else {}

                results.append(
                    {
                        "custom_id": result.custom_id,
                        "response": response_text,
                        "status": "success",
                        "generated_summary": summary_pair.get("generated", ""),
                        "reference_summary": summary_pair.get("reference", ""),
                        "len_control": summary_pair.get("log(len(gen)/len(ref))", ""),
                    }
                )

            else:
                results.append(
                    {
                        "custom_id": result.custom_id,
                        "error": str(result.result.error),
                        "status": "error",
                    }
                )

        # Save to file
        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        print(f"Downloaded {len(results)} results to {output_file}")
        return results

    def parse_preferences(self, results):
        """Extract A/B preferences from results"""
        preferences = []
        for result in results:
            if result["status"] == "success":
                response = result["response"]
                # Extract the "Preferred: A" or "Preferred: B" line
                if "Preferred: A" in response:
                    preference = "A"  # Your model won
                elif "Preferred: B" in response:
                    preference = "B"  # Reference won
                else:
                    preference = "unclear"

                preferences.append(
                    {
                        "comparison_id": result["custom_id"],
                        "preference": preference,
                        "full_response": response,
                    }
                )

        # Calculate win rate
        model_wins = sum(1 for p in preferences if p["preference"] == "A")
        total = len(preferences)
        win_rate = model_wins / total if total > 0 else 0

        print(f"Model win rate: {win_rate:.2%} ({model_wins}/{total})")
        return preferences

    def load_and_bin_results(self, results_file, n_bins=8):
        """Load results and bin by length control"""
        # Load results
        results = []
        with open(results_file, "r") as f:
            for line in f:
                results.append(json.loads(line))

        # Extract length ratios and wins
        length_ratios = []
        wins = []

        for result in results:
            if result["status"] != "success":
                continue

            response = result["response"]

            # Extract preference
            if "Preferred: A" in response:
                wins.append(1)  # Model won
            elif "Preferred: B" in response:
                wins.append(0)  # Reference won
            else:
                continue  # Skip unclear

            # Get length ratio
            len_control = result.get("len_control", None)
            if len_control is not None and len_control != "":
                length_ratios.append(len_control)
            else:
                # Fallback: calculate from summaries if not stored
                gen_tokens = self.data.tokenizer.encode(result["generated_summary"])
                ref_tokens = self.data.tokenizer.encode(result["reference_summary"])
                if len(ref_tokens) > 0:
                    len_control = np.log(len(gen_tokens) / len(ref_tokens))
                    length_ratios.append(len_control)

        length_ratios = np.array(length_ratios)
        wins = np.array(wins)

        # Create bins with equal number of samples (quantile-based)
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(length_ratios, percentiles)

        # Calculate win rate per bin
        bin_centers = []
        win_rates = []
        bin_counts = []

        for i in range(n_bins):
            mask = (length_ratios >= bin_edges[i]) & (length_ratios < bin_edges[i + 1])
            if i == n_bins - 1:  # Include right edge in last bin
                mask = (length_ratios >= bin_edges[i]) & (length_ratios <= bin_edges[i + 1])

            if mask.sum() > 0:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                win_rates.append(wins[mask].mean())
                bin_counts.append(mask.sum())

        return bin_centers, win_rates, bin_counts

    def plot_length_controlled_winrates(
        self,
        ppo_results_file="batch_results_my_ppo.jsonl",
        sft_results_file="batch_results_my_sft.jsonl",
        paper_ppo_results_file="batch_results_paper_ppo_v2.jsonl",
    ):
        # Load and bin both models
        ppo_results_file = self.config.ppo_results_file or ppo_results_file
        sft_results_file = self.config.sft_results_file or sft_results_file
        paper_ppo_results_file = self.config.paper_ppo_results_file or paper_ppo_results_file

        ppo_centers, ppo_rates, ppo_counts = self.load_and_bin_results(ppo_results_file)
        sft_centers, sft_rates, sft_counts = self.load_and_bin_results(sft_results_file)
        ppr_ppo_centers, ppr_ppo_rates, ppr_ppo_counts = self.load_and_bin_results(
            paper_ppo_results_file
        )

        # Plot
        plt.figure(figsize=(12, 7))
        plt.scatter(
            ppo_centers,
            ppo_rates,
            s=80,
            label="PPO",
            color="blue",
            marker="o",
            zorder=3,
        )
        plt.scatter(
            sft_centers,
            sft_rates,
            s=80,
            label="SFT",
            color="green",
            marker="s",
            zorder=3,
        )
        plt.scatter(
            ppr_ppo_centers,
            ppr_ppo_rates,
            s=80,
            label="Huang et. al. PPO 1B 77713",
            color="red",
            marker="o",
            zorder=3,
        )

        # Fit and plot trendlines
        ppo_fit = np.polyfit(ppo_centers, ppo_rates, 1)
        ppo_trendline = np.poly1d(ppo_fit)
        sft_fit = np.polyfit(sft_centers, sft_rates, 1)
        sft_trendline = np.poly1d(sft_fit)
        ppr_ppo_fit = np.polyfit(ppr_ppo_centers, ppr_ppo_rates, 1)
        ppr_ppo_trendline = np.poly1d(ppr_ppo_fit)

        x_range = np.linspace(
            min(min(ppo_centers), min(sft_centers), min(ppr_ppo_centers)),
            max(max(ppo_centers), max(sft_centers), max(ppr_ppo_centers)),
            100,
        )
        plt.plot(
            x_range,
            ppo_trendline(x_range),
            "--",
            linewidth=2,
            color="blue",
            alpha=0.6,
            zorder=2,
        )
        plt.plot(
            x_range,
            sft_trendline(x_range),
            "--",
            linewidth=2,
            color="green",
            alpha=0.6,
            zorder=2,
        )
        plt.plot(
            x_range,
            ppr_ppo_trendline(x_range),
            "--",
            linewidth=2,
            color="red",
            alpha=0.6,
            zorder=2,
        )
        plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")

        plt.xlabel("log(generated_length / reference_length)", fontsize=12)
        plt.ylabel(
            "Winrate against reference summaries (according to claude-sonnet-4-20250514)",
            fontsize=12,
        )
        plt.title("Length-Controlled Win Rate vs Summary Length Ratio", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)

        # Add sample counts as text
        # for x, y, count in zip(ppo_centers, ppo_rates, ppo_counts):
        #     plt.text(x, y + 0.02, f'{count}', ha='center', fontsize=8, alpha=0.6, color='blue')
        # for x, y, count in zip(sft_centers, sft_rates, sft_counts):
        #     plt.text(x, y - 0.02, f'{count}', ha='center', fontsize=8, alpha=0.6, color='orange')
        # for x, y, count in zip(ppr_ppo_centers, ppr_ppo_rates, ppr_ppo_counts):
        #     plt.text(x, y - 0.02, f'{count}', ha='center', fontsize=8, alpha=0.6, color='red')

        plt.tight_layout()
        plt.axis("equal")
        plt.savefig(self.config.plot_name, dpi=300)
        plt.show()

        # Print statistics
        print("\n=== PPO Model ===")
        for i, (center, rate, count) in enumerate(zip(ppo_centers, ppo_rates, ppo_counts)):
            print(f"Bin {i+1}: log_ratio={center:.3f}, win_rate={rate:.3f}, n={count}")

        print("\n=== SFT Model ===")
        for i, (center, rate, count) in enumerate(zip(sft_centers, sft_rates, sft_counts)):
            print(f"Bin {i+1}: log_ratio={center:.3f}, win_rate={rate:.3f}, n={count}")

        print("\n=== Paper PPO Model ===")
        for i, (center, rate, count) in enumerate(
            zip(ppr_ppo_centers, ppr_ppo_rates, ppr_ppo_counts)
        ):
            print(f"Bin {i+1}: log_ratio={center:.3f}, win_rate={rate:.3f}, n={count}")

        # Overall win rates
        ppo_overall = np.average(ppo_rates, weights=ppo_counts)
        sft_overall = np.average(sft_rates, weights=sft_counts)
        ppr_ppo_overall = np.average(ppr_ppo_rates, weights=ppr_ppo_counts)
        print(f"\nOverall PPO win rate: {ppo_overall:.3f}")
        print(f"Overall SFT win rate: {sft_overall:.3f}")
        print(f"Overall Paper PPO win rate: {ppr_ppo_overall:.3f}\n")

        return {
            "ppo": (ppo_centers, ppo_rates, ppo_counts),
            "sft": (sft_centers, sft_rates, sft_counts),
            "ppr_ppo": (ppr_ppo_centers, ppr_ppo_rates, ppr_ppo_counts),
        }
