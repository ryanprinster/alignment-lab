cd /workspace/alignment-lab && git pull && python -m pip install -r requirements.txt && huggingface-cli login



cd /workspace/alignment-lab

git pull

python -m pip install -r requirements.txt

huggingface-cli login

tensorboard --logdir=./runs --host=0.0.0.0 --port=6006



### Commands ###


# Cartpole Trainer
python -m experiments PPOTrainer train --config PPOConfig --batch_size 8


# SFT Trainer
python -m experiments SFTTrainer train --config RLFHCaseStudyConfig
python -m experiments SFTTrainer evaluate --config RLFHCaseStudyConfig


# RM Trainer
python -m experiments RMTrainer compute_model_bias --config RLFHCaseStudyConfig --batch_size 8 --load_checkpoint_path "./checkpoints/final_checkpoint.pt"
python -m experiments RMTrainer train --config RLFHCaseStudyConfig --load_checkpoint_path checkpoints/sft_final_checkpoint.pt --calculated_sft_bias -8.703847885131836 --save_freq_steps 9999999 --batch_size 32 --accumulation_steps 2

# PPORLHFTrainer
python3 -m experiments PPORLHFTrainer train --config RLFHPPOConfig --batch_size 200 --mini_batch_accumulation_steps 1




# Transfer between pods = source_pod --> local --> dest_pod
# Significantly less efficient
scp -r -P 23898 -i ~/.ssh/id_ed25519 root@103.196.86.188:/path/to/source/directory ./local-directory
scp -r -P 12634 -i ~/.ssh/id_ed25519 ./local-directory root@198.145.108.49:/path/to/destination/


# rsync pod to pod

apt update && apt install -y rsync

# if needed, kill previous tries:
kill %1 %2

# test ssh connection pod to pod first
ssh -p 12634 -i ~/.ssh/id_ed25519 root@198.145.108.49

nohup rsync -avzP -e "ssh -p 12634 -i ~/.ssh/id_ed25519" \
  /workspace/alignment-lab/keep/ \
  root@198.145.108.49:/workspace/keep/ \
  > rsync.log 2>&1 &

tail -f rsync.log
