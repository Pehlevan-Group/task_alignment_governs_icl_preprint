#!/bin/bash
# d150_tryagain.sbatch
# 
#SBATCH --job-name=d150_tryagain
#SBATCH -c 1
#SBATCH -t 2:00:00
#SBATCH -p sapphire
#SBATCH --mem=48000
#SBATCH -o /n/netscratch/pehlevan_lab/Lab/ml/ICL-structured-data/Linear_Simulations/test_time_scaling/outputs/d150_tryagain_%a.out
#SBATCH -e /n/netscratch/pehlevan_lab/Lab/ml/ICL-structured-data/Linear_Simulations/test_time_scaling/outputs/d150_tryagain_%a.err
#SBATCH --array=1-10
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4

parentdir="runs"
newdir="$parentdir/${SLURM_JOB_NAME}"
mkdir "$newdir"
python run.py $newdir 150 2 4 $SLURM_ARRAY_TASK_ID 4 