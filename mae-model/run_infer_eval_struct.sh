#!/bin/bash
#SBATCH --job-name=infer-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=ice-gpu
#SBATCH --output=logs/slurm-%x-%j.out

# === Load modules ===
module purge
module load anaconda3 cuda/11.8

# Init conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/hice1/jting40/scratch/mae-1/conda_env
pip install torchmetrics facenet_pytorch lpips torch-fidelity

# === Run scripts ===
echo ">>> Running inference..."
python infer_faces_struct.py

echo ">>> Running evaluation..."
python eval_recons_struct.py

echo "✅ Done."
