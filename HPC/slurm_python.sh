#!/bin/sh
#SBATCH --partition=nodes
#SBATCH --nodes=2                                 # Nodes requested
#SBATCH --ntasks-per-node=8                       # taskes per node
#SBATCH --mem=23000                               # memory per node (MB)
#SBATCH --job-name="<filename>"
#SBATCH --output=<filename>.out                    # send stdout to outfile
#SBATCH --error=<filename>.err                     # send stderr to errfile
#SBATCH --time=4:00:00                            # time requested in day-hour:minute:second
#SBATCH --mail-user=hirenkumar.thummar@sjsu.edu
#SBATCH --mail-type=ALL

module load intel-python3
python <filename>.py
