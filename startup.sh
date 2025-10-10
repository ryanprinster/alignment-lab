set -e

cd /workspace/alignment-lab

git pull

python -m pip install -r requirements.txt

huggingface-cli login

tensorboard --logdir=./runs --host=0.0.0.0 --port=6006

python -m experiments --batch_size 16 --accumulation_steps 4 --no_enable_mixed_precision_training

python3 -m experiments --batch_size 16 --accumulation_steps 4 --load_checkpoint_path /Users/ryanprinster/Projects/trained_models/sft/checkpoint_at_2025-09-08T02:27:15.796408.pt




python -m experiments PPOTrainer train --config PPOConfig --batch_size 8


# SFT Trainer
python -m experiments SFTTrainer train --config RLFHCaseStudyConfig

# RM Trainer
python -m experiments RMTrainer compute_model_bias --config RLFHCaseStudyConfig --batch_size 8 --load_checkpoint_path "./checkpoints/final_checkpoint.pt"
python -m experiments RMTrainer train --config RLFHCaseStudyConfig --load_checkpoint_path checkpoints/sft_final_checkpoint.pt --calculated_sft_bias -8.703847885131836 --save_freq_steps 9999999 --batch_size 32 --accumulation_steps 2

# PPORLHFTrainer
python3 -m experiments PPORLHFTrainer train --config RLFHPPOConfig --batch_size 256 --mini_batch_accumulation_steps 1