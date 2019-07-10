#!/bin/sh
#SBATCH --partition=gpus
#SBATCH --gres=gpu:2
#SBATCH --job-name=gpunet
#SBATCH --output=galnet.out
#SBATCH --error=galnet.err
#SBATCH --time=10-00:00:00
#SBATCH --mail-user=james.casey-clyde@sjsu.edu
#SBATCH --mail-type=ALL

conda activate galnet
PYTHONHASHSEED=0 python galnet/galnet.py --train --model model
