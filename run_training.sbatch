#!/bin/bash
#SBATCH --job-name=brain_segmentation_training
#SBATCH --output=training_output.log
#SBATCH --error=training_error.log
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=10:00:00

# Load any required modules (if needed)
# module load cuda/11.1  # Example: adjust this based on your environment

# Initialize conda
#source ~/miniconda3/etc/profile.d/conda.sh

# Activate your Python environment
#srun conda activate brain-segm

# Run the training script
srun python -m src.train

