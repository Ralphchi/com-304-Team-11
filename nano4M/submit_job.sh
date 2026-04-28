#!/bin/bash
#SBATCH --job-name=my_job         # Change as needed
#SBATCH --time=08:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --gres=gpu:2                    # Request 2 GPUs
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # Adjust CPU allocation if needed
#SBATCH --output=logs/train_%x_%j.out   # Per-job output (%x=name, %j=jobid)
#SBATCH --error=logs/train_%x_%j.err    # Per-job error
#SBATCH --partition=l40s

CONFIG_FILE=$1
WANDB=$2
NUM_GPUS=$3

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm
export WANDB_API_KEY=$WANDB && OMP_NUM_THREADS=1 torchrun --nproc_per_node=$NUM_GPUS run_training.py --config $CONFIG_FILE