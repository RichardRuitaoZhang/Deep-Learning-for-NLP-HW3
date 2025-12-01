#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --job-name=hw3_prefix
#SBATCH --output=logs/hw3_prefix_%j.log

# ===== Working Directory =====
cd /projects/b1080/rz/cs461/HW3/

# ===== Environment Setup =====
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate peft

# ===== Paths and Parameters =====
TRAIN_PATH=data/obqa.train.txt
VALID_PATH=data/obqa.valid.txt
SEQLEN=512
BATCH=2
LR=1e-4
EPOCHS=3
LOADNAME=pretrain

# ===== Run Prefix-Tuning Fine-Tuning =====
echo "Starting Prefix Tuning fine-tuning on GPU..."
nvidia-smi

python peft_prefix.py \
  --train_path $TRAIN_PATH \
  --valid_path $VALID_PATH \
  --seqlen $SEQLEN \
  --batch_size $BATCH \
  --lr $LR \
  --epochs $EPOCHS \
  --loadname $LOADNAME

echo "Prefix Tuning fine-tuning completed."
