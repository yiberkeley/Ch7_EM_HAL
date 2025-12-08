#!/bin/bash
# Job name:
#SBATCH --job-name=uncensored_NPMLE_EM_500
#
# Partition:
#SBATCH --partition=savio3
#
#SBATCH --qos=biostat_savio3_normal
#SBATCH --account=co_biostat
#
# Wall clock limit ('0' for unlimited):
#SBATCH --time=72:00:00
#
# Number of nodes for use case:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#
# Mail type:
#SBATCH --mail-type=all
#
# Mail user:
#SBATCH --mail-user=yi_li@berkeley.edu
module load python/3.11.6-gcc-11.4.0
python uncensored_NPMLE_EM.py > logs/1_${SLURM_JOB_NAME}.out


