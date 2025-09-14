#!/bin/bash
# isotropic_d10_alpha2_tau4.sbatch
#
#SBATCH --job-name=isotropic_d10_alpha2_tau4
#SBATCH -o /n/netscratch/pehlevan_lab/Lab/ml/ICL-structured-data/NONLINEAR/experiment/remote/alpha_scaling/outputs/isotropic_d10_alpha2_tau4_%a.out
#SBATCH -e /n/netscratch/pehlevan_lab/Lab/ml/ICL-structured-data/NONLINEAR/experiment/remote/alpha_scaling/outputs/isotropic_d10_alpha2_tau4_%a.err
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=2:20:00
#SBATCH --array=1-5
#SBATCH -p kempner
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

module purge
module load python/3.10.12-fasrc01
source activate try4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

parentdir="runs"
newdir="$parentdir/${SLURM_JOB_NAME}"
pkldir="$parentdir/${SLURM_JOB_NAME}/pickles"
mkdir "$newdir"
mkdir "$pkldir"
python alphasweep.py 10 $newdir $SLURM_ARRAY_TASK_ID