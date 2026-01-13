# RLHF from Scratch

A from-scratch implementation of Reinforcement Learning from Human Feedback (RLHF) using Proximal Policy Optimization (PPO) for text summarization. Built as a learning and fun project to deeply understand RLHF mechanics.

For replication analysis, see [Evaluation Results](/docs/lessons.md).
For lessons learned, see [r/RLHF_from_Scratch](/docs/eval.md).


## Overview

This project implements the complete RLHF pipeline on the Reddit TL;DR dataset detailed in [Huang et al. 2024](docs/REFERENCES.md#ref1):
1. Supervised Fine-Tuning (SFT)
2. Reward Model Training
3. PPO Optimization

Key deviations from the implentation, chosen primarily for simplicity, are as follows:

| Component | Our Implementation | Huang et al. (2024) | 
|--------|--------|-----------|
|Base Model|Llama-3.2-1B (untuned)|Pythia (Biderman et al., 2023)|
|Tokenizer|HuggingFace Llama Tokenizer|HuggingFace Pythia Tokenizer|
|Hardware Configuration | 1×H200 GPU| 8×H100 GPUs|
|ZeRO Optimization | Not implemented | Stage 2|
|Batch Size |128 (PPO)| 512 (PPO)|
|Learning Rate |1.5e-6 (PPO) | 3e-6 (PPO)|
|Training Episodes |116k (1 epoch) | 1M (~8.56 epochs)|

More details in [evaluation](eval.md).

## Results summary

Overall, results are comparable to and even exceed Huang et al. Any improvements may be statistically significant _but_ are likely attributed to a stronger base model (Llama vs Pythia). For a more detail results and analysis, look [here](eval.md). 

Notable observations include reward hacking via title copying, which also appear in some of the samples shown in Huang et al..

**Key Results:**
- **SFT:** ROUGE-L of 0.2694 vs. ~0.2575 (original)
- **Reward Model:** 69.5% validation accuracy vs. ~63% (original) and 73.4% judge agreement vs. 37.3% (original)
- **PPO:** Length-controlled win rates comparable to original 1B curves, with similar training dynamics but lower policy entropy and early evidence of reward hacking via title copying (TODO: ~X% of outputs) 

<table  align="center">
<tr>
<td><img src="docs/assets/images/ppo_rlhf_reward_2.png" style="max-width: 350px; max-height: 300px;" alt=""/></td>
<td><img src="docs/assets/images/ppo_rlhf_reward_huang.png" style="max-width: 350px; max-height: 300px;" alt=""/></td>
</tr>

<tr>
<td align="center"><i>Reproduced PPO, overall win rate 0.582  (exponential smoothing α = 0.92)</i></td>
<td align="center"><i>Huang et al. (2024)</i></td>
</tr>
</table>


<table  align="center">
<tr>
<td><img src="docs/assets/images/ppo_win_rate.png" style="max-width: 350px; max-height: 300px;" alt=""/></td>
<td><img src="docs/assets/images/ppo_win_rate_huang.png" style="max-width: 350px; max-height: 300px;" alt=""/></td>
</tr>
<tr>
<td align="center"><i>Reproduced PPO overall win rate 0.582</i></td>
<td align="center"><i>Huang et al.. Red and blue are PPO, green is SFT. </i></td>
</tr>
</table>


## Setup
To use on runpod:
```bash
# clone https://github.com/ryanprinster/alignment-lab and cd into /workspace/alignment-lab

startup_runpod.sh
```

To use from other environments 
```bash
# Activate virtual environment with 
source torch_env/bin/activate 

# install stuff with 
python -m pip install <stuff>
hf auth login
export ANTHROPIC_API_KEY="<key_goes_here>"

# Note that some small modifications may be required to use with different GPUs
```

## Usage
```bash
python -m experiments <TrainerName> <method> --config <ConfigName> [options]
# TrainerName is the class name from a python script in /trainers
# method is the name of the method in TrainerName to execute
# ConfigName is the class name from config.py

# For example:
python3 -m experiments PPORLHFTrainer train --config RLFHPPOConfig --batch_size 128 --enable_gradient_checkpointing --no_whiten_rewards --sft_model_path "checkpoints/sft_final_checkpoint.pt"
```

## Project Structure
```
alignment-lab/
├── experiments/        
├───── trainers/     # Training loops for SFT, RM, and PPO, as well as eval entry points
├─────  ...          # Other code
├─── docs/          # Eval, lesson writeup
└─────  ... 
```




