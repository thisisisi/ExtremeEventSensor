#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=3:00:00
#SBATCH -J lowdim_ml
#SBATCH -o output/%j.lowdim_stdout
#SBATCH -e output/%j.lowdim_stderr
# Author: Isabella Thiel <thieli@mit.edu>

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}"
if [[ -n "${CONDA_ENV:-}" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV/bin/python}"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

cd "$PROJECT_DIR"
mkdir -p output

if [[ -z "${LOWDIM_CONFIG_JSON:-}" ]]; then
    echo "LOWDIM_CONFIG_JSON is not set."
    exit 2
fi

module purge

echo "===================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Project directory: $PROJECT_DIR"
echo "Python executable: $PYTHON_BIN"
echo "Config JSON: $LOWDIM_CONFIG_JSON"
echo "===================="
echo
echo "========== Config =========="
cat "$LOWDIM_CONFIG_JSON"
echo
echo "========== Starting Low-Dimensional Training =========="
"$PYTHON_BIN" -u main.py run
echo "========== Low-Dimensional Training Finished =========="
echo
echo "Job completed at: $(date)"
