#!/bin/bash
# Job name:
#SBATCH --job-name=RC_800_1
#
# Partition:
#SBATCH --partition=savio2
#
#SBATCH --account=co_biostat
#
# Wall clock limit ('0' for unlimited):
#SBATCH --time=72:00:00
#
# Number of nodes for use case:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#
# Mail type:
#SBATCH --mail-type=all
#
# Mail user:
#SBATCH --mail-user=yi_li@berkeley.edu
# Load Python module
module load python/3.11.6-gcc-11.4.0
python rc_draft.py > logs/rc_draft_${SLURM_JOB_NAME}.out


