# cd /workspace/alignment-lab && git pull && python -m pip install -r requirements.txt && huggingface-cli login
cd /workspace/alignment-lab && git pull && python -m pip install -r requirements.txt && hf auth login

tensorboard --logdir=./runs --host=0.0.0.0 --port=6006



### Commands ###


# Cartpole Trainer
python -m experiments PPOTrainer train --config PPOConfig --batch_size 8


# SFT Trainer
python -m experiments SFTTrainer train --config RLFHCaseStudyConfig
python -m experiments SFTTrainer evaluate --config RLFHCaseStudyConfig


# RM Trainer
python -m experiments RMTrainer compute_model_bias --config RLFHCaseStudyConfig --batch_size 128 --load_checkpoint_path "./checkpoints/rm_final_checkpoint_v2.pt"
python -m experiments RMTrainer train --config RLFHCaseStudyConfig --load_checkpoint_path checkpoints/sft_final_checkpoint.pt --calculated_sft_bias 0 --save_freq_steps 9999999 --batch_size 32 --accumulation_steps 2
python -m experiments RMTrainer validation --config RLFHCaseStudyConfig --load_checkpoint_path checkpoints/rm_final_checkpoint_v2.pt  --batch_size 64 --accumulation_steps 1
# PPORLHFTrainer
python3 -m experiments PPORLHFTrainer train --config RLFHPPOConfig --batch_size 128


### Tensorboard
# From runpod
tensorboard --logdir=./runs --bind_all --port=6006
# leave that open. then from a new local terminal:
ssh -L 6006:localhost:6006 root@198.145.108.61 -p 11227 -i ~/.ssh/id_ed25519
# then open in browser:
http://localhost:6006




### WORKING ###
# From YOUR LOCAL MACHINE:
# Step 1: Copy SSH key to source pod
scp -P 38559 -i ~/.ssh/id_ed25519 ~/.ssh/id_ed25519 root@103.196.86.188:~/.ssh/
# Step 2: SSH into source pod
ssh root@103.196.86.188 -p 38559 -i ~/.ssh/id_ed25519

# From INSIDE THE SOURCE POD:
# (After Step 2 connects you to the source pod, run everything below)
# Step 3: Test connection to destination
ssh -p 13751 -i ~/.ssh/id_ed25519 root@198.145.108.61
# (Type yes, then exit)
# Step 4: Check/install rsync
which rsync
# If needed:
apt-get update && apt-get install -y rsync
# Step 5: Start the transfer
nohup rsync -avzP -e "ssh -p 17013 -i ~/.ssh/id_ed25519" \
  /workspace/alignment-lab/checkpoints/rm_final_checkpoint_v2.pt \
  root@198.145.108.61:/workspace/checkpoints/ \
  > rsync.log 2>&1 &
# Step 6: Monitor progress
tail -f rsync.log


ssh root@198.145.108.61 -p 17013 -i ~/.ssh/id_ed25519



# THEN
git clone https://github.com/ryanprinster/alignment-lab
git checkout origin/rlhf_ppo_tweaks
git push -u origin rlhf_ppo_tweaks

#  
mv /workspace/checkpoints /workspace/alignment-lab/checkpoints