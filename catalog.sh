#!/bin/sh
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --job-name=galdata
#SBATCH --output=galdata.out
#SBATCH --error=galdata.err
#SBATCH --time=5-00:00:00
#SBATCH --mail-user=james.casey-clyde@sjsu.edu
#SBATCH --mail-type=ALL

conda activate galnet
python catalog.py -u username -p passwd
