set -e

cd /workspace/alignment-lab

git pull

python -m pip install -r requirements.txt

huggingface-cli login

tensorboard --logdir=./runs --host=0.0.0.0 --port=6006

python -m experiments --batch_size 16 --accumulation_steps 4 --no_enable_mixed_precision_training