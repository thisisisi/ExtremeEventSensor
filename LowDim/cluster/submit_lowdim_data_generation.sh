#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -J lowdim_data
#SBATCH -o output/%j.lowdim_data_stdout
#SBATCH -e output/%j.lowdim_data_stderr

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-/orcd/pool/006/thieli/conda_envs/emulator/bin/python}"

cd "$PROJECT_DIR"
mkdir -p output

echo "===================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Project directory: $PROJECT_DIR"
echo "Python executable: $PYTHON_BIN"
echo "===================="

"$PYTHON_BIN" -u LowDim/main_create_data.py \
    --duration 510 \
    --time-steps 170000 \
    --tau 8 \
    --sigma 5e-6 \
    --runs 2 \
    --members 16 \
    --seed 0

echo "Job completed at: $(date)"