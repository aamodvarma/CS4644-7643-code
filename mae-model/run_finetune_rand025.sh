#!/bin/bash
#SBATCH --job-name=mae-finetune-rand025
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --partition=ice-gpu
#SBATCH --output=logs/slurm-%x-%j.out

# -------------------------------
# 1. Load modules + activate env
# -------------------------------
module purge
module load anaconda3 cuda/11.8

# Init conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/hice1/jting40/scratch/mae-1/conda_env

# -------------------------------
# 2. Environment variables
# -------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=29507

# -------------------------------
# 3. Run MAE finetuning (random masking, 0.25)
# -------------------------------
cd /home/hice1/jting40/scratch/mae-1

srun python main_finetune_rand.py \
  --batch_size 32 \
  --epochs 50 \
  --accum_iter 1 \
  --model mae_vit_base_patch16 \
  --mask_ratio 0.25 \
  --input_size 224 \
  --data_path ./data \
  --output_dir ./checkpoints_randmask025 \
  --log_dir ./checkpoints_randmask025 \
  --device cuda \
  --num_workers 16 \
  --finetune ./checkpoints/mae_pretrain_vit_base.pth
