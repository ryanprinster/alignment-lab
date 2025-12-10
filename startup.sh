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
nohup python3 -m experiments PPORLHFTrainer train --config RLFHPPOConfig --batch_size 128 --alpha 1.5e-6 > train.log 2>&1 &
tail -f train.log
python3 -m experiments PPORLHFTrainer train --config RLFHPPOConfig --batch_size 128 --rm_model_path "models--vwxyzjn--EleutherAI_pythia-1b-deduped__reward__tldr/snapshots/33b95d01a8f208eba7236e2a3e5277f342b453cf/pytorch_model.bin" --sft_model_path "models--vwxyzjn--EleutherAI_pythia-1b-deduped__sft__tldr/snapshots/997a2257eaaa3bb8d2ecf14e1929789dd3dceab0/pytorch_model.bin"
--hf_sft_model_name "vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr" --hf_sft_revision_name "sft__44413__1708611267" --hf_rm_model_name "vwxyzjn/EleutherAI_pythia-1b-deduped__reward__tldr" --hf_rm_revision_name "reward__44413__1708628552"

# PPORLHFEval
python3 -m experiments PPORLHFEval construct_claude_request --config RLFHPPOConfig --batch_size 128



### Tensorboard
# From runpod
tensorboard --logdir=./runs --bind_all --port=6006
# leave that open. then from a new local terminal:
ssh -L 6006:localhost:6006 root@198.145.108.61 -p 18353 -i ~/.ssh/id_ed25519
ssh -L 6006:localhost:6006 root@103.196.86.188 -p 49850 -i ~/.ssh/id_ed25519
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
nohup rsync -avzP -e "ssh -p 18353 -i ~/.ssh/id_ed25519" \
  /workspace/alignment-lab/checkpoints/rm_final_checkpoint_v2.pt \
  root@198.145.108.49:/workspace/checkpoints/ \
  > rsync.log 2>&1 &
# Step 6: Monitor progress
tail -f rsync.log

ssh root@198.145.108.49 -p 18353 -i ~/.ssh/id_ed25519

ssh root@198.145.108.61 -p 17013 -i ~/.ssh/id_ed25519



# THEN
git clone https://github.com/ryanprinster/alignment-lab
git checkout origin/rlhf_ppo_tweaks
git push -u origin rlhf_ppo_tweaks

#  
mv /workspace/checkpoints /workspace/alignment-lab/checkpoints