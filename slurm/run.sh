#!/usr/bin/env bash
# The whole experiment as ONE GPU job (~20-30 min for a 1.5B model on half an A100).
# Submit from the repo root; arguments go straight to run.py:
#
#   sbatch slurm/run.sh                                         # SETTINGS in run.py
#   sbatch slurm/run.sh --model meta-llama/Llama-3.2-3B-Instruct
#   sbatch slurm/run.sh --quick                                 # smoke test
#   sbatch slurm/run.sh --pilot --model <id>                    # question-ending pilot
#   sbatch --partition=gpu_a100 --time=04:00:00 slurm/run.sh --model <bigger model>
#
# Partition and --time per model: see MODEL_SETTINGS in run.py. If a job hits its time
# limit, submit the SAME command again: finished steps are reused (runs/<name>/checkpoints).
#
# Compute nodes have no internet: fetch the model first, on the login node:
#   .venv/bin/python run.py --download --model <id>
#
#SBATCH --job-name=hedge
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=slurm-hedge-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

echo "[job] $(hostname) $(date) -- run.py --offline $*"
echo "[job] model cache: ${HF_HOME:-$HOME/.cache/huggingface}  (submit with --export=ALL if you set HF_HOME)"
.venv/bin/python run.py --offline "$@"
