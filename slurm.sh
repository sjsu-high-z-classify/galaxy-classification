#!/bin/sh
#SBATCH --partition=gpus
#SBATCH --nodes=1                                 # Nodes requested
#SBATCH --gres=gpu
#SBATCH --job-name=galnet
#SBATCH --output=galnet.out                    # send stdout to outfile
#SBATCH --error=galnet.err                     # send stderr to errfile
#SBATCH --time=10-00:00:00                            # time requested in day-hour:minute:second
#SBATCH --mail-user=james.casey-clyde@sjsu.edu
#SBATCH --mail-type=ALL

conda activate galnet-gpu
python cnn.py
