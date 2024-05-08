#!/bin/bash
#SBATCH --job-name=training_job    # Job name
#SBATCH --output=res_%j.txt        # Output and error (e.g., res_12345.txt)
#SBATCH --time=20:00:00            # Time limit hrs:min:sec
#SBATCH --mem=8G                   # Memory total in MB

# Load modules or source software if needed (depends on system configuration)
# e.g., module load python3

# Activate conda environment
source activate brains-segm

# Run the training script
make train

# Or, if you need to run directly in an interactive shell:
# srun --pty --mem=4G --gpus=1 --time=10:00:00 --job-name="interactive_job" bash
# Then you can manually activate the environment and start the script

