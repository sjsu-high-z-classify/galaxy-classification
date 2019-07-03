#!/bin/sh
#SBATCH --partition=nodes
#SBATCH --nodes=1                                 # Nodes requested
#SBATCH --job-name=galdata
#SBATCH --output=OUT                    # send stdout to outfile
#SBATCH --error=ERROR                     # send stderr to errfile
#SBATCH --time=5-00:00:00                            # time requested in day-hour:minute:second
#SBATCH --mail-user=james.casey-clyde@sjsu.edu
#SBATCH --mail-type=ALL

module load intel-python3
python3 DatasetBuilder.py
