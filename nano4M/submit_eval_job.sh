#!/bin/bash
#SBATCH --job-name=nano4m_eval
#SBATCH --time=02:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err
#SBATCH --partition=l40s
#
# Submit one job per (variant, seed). Example for the four main variants
# with three seeds each (12 submissions total):
#
#   for v in baseline block ctxblock span; do
#     for seed in 0 1 2; do
#       sbatch submit_eval_job.sh \
#         outputs/nano4M/multiclevr_d6-6w512_${v}/checkpoint-final.safetensors \
#         cfgs/nano4M/multiclevr_d6-6w512_${v}.yaml \
#         results/${v}_seed${seed}.json \
#         ${seed} \
#         A,B,C,D
#     done
#   done

CKPT=$1
CONFIG=$2
OUTPUT=$3
SEED=${4:-0}
PHASES=${5:-A,B,C,D}
NUM_SAMPLES=${6:-500}

if [ -z "$CKPT" ] || [ -z "$CONFIG" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: sbatch submit_eval_job.sh <ckpt> <config> <output_json> [seed] [phases] [num_samples]"
    exit 1
fi

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm

# Use the shared HF cache populated by scripts/prefetch_eval_models.py.
export HF_HOME=/work/com-304/hf_cache
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1

python run_evaluation.py \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --num-samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --phases "$PHASES" \
    --output "$OUTPUT"
