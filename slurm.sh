#!/bin/sh
#SBATCH --partition=nodes
#SBATCH --nodes=1                                 # Nodes requested
#SBATCH --ntasks-per-node=16                       # taskes per node
#SBATCH --mem=23000                               # memory per node (MB)
#SBATCH --job-name="train_test"
#SBATCH --output=OUT.out                    # send stdout to outfile
#SBATCH --error=ERR.err                     # send stderr to errfile
#SBATCH --time=4:00:00                            # time requested in day-hour:minute:second
#SBATCH --mail-user=user.email@sjsu.edu
#SBATCH --mail-type=ALL

cd ~/path/to/source/
module load intel-python3
python3 galnet.py -u '***' -p '***' -r 10000
