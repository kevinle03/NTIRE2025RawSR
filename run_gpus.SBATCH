#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --mem=80GB
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=MambaIRv2_SR_x2_raw_l1=0.9_swt=0.1_L8_B4
#SBATCH --output=MambaIRv2_SR_x2_raw_l1=0.9_swt=0.1_L8_B4%j.txt
#SBATCH --error=MambaIRv2_SR_x2_raw_l1=0.9_swt=0.1_L8_B4%j.err
module purge
module load singularity

SINGULARITY_IMAGE="/scratch/ll5484/lillian/test_singularity/abc.sif"
CONDA_ENV_PATH="/scratch/ll5484/miniconda3/envs/mambavision"
SCRIPT_PATH="basicsr/train.py"


singularity exec --nv "$SINGULARITY_IMAGE" bash -c "
        source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
        conda activate $CONDA_ENV_PATH
        python $SCRIPT_PATH
"
