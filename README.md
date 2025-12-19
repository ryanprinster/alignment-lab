# Alignment Lab: RLHF from Scratch

A from-scratch implementation of Reinforcement Learning from Human Feedback (RLHF) using Proximal Policy Optimization (PPO) for text summarization.

## Overview

This project implements the complete RLHF pipeline on the Reddit TL;DR dataset using Llama 3.2 1B models:
1. Supervised Fine-Tuning (SFT)
2. Reward Model Training
3. PPO Optimization

Built as a learning and fun project to deeply understand RLHF mechanics 

## Setup
```bash

# Activate virtual environment with 
source torch_env/bin/activate 

# install stuff with 
python -m pip install <stuff>

```

## Usage
```bash
WIP
```

## Project Structure
```
WIP
alignment-lab/
├── models/          # Model architectures and components
├── training/        # Training loops for SFT, RM, and PPO
├── data/           # Dataset processing and loading
├── configs/        # Configuration files
└── utils/          # Helper functions and utilities
```

## Technical Details
WIP
For an in-depth explanation of the implementation, key design decisions, and lessons learned, see [my writeup](link-to-blog-post).

## References
WIP
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/pdf/2005.12729)
- [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/pdf/2403.17031)





