#!/bin/bash
#SBATCH --job-name=qg_tables
#SBATCH --partition=mit_normal
#SBATCH --output=logs/tables_%j.out
#SBATCH --error=logs/tables_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=32G
# Author: Isabella Thiel <thieli@mit.edu>

set -euo pipefail

SCRIPT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi

cd "$SCRIPT_DIR"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/qg_matplotlib_${SLURM_JOB_ID:-manual}"
mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

"$PYTHON_BIN" -u generate_all_comparison_tables.py
