#!/bin/bash
# Author: Isabella Thiel <thieli@mit.edu>

#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=3G
#SBATCH --time=2:30:00
#SBATCH -J training_sweep
#SBATCH -o output/sweep_runner_o.%j
#SBATCH -e output/sweep_runner_e.%j

set -euo pipefail

cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
    exit 1
fi

echo "Working directory (logical): $(pwd)"
echo "Working directory (physical): $(pwd -P)"

echo "Starting parameter sweep at: $(date)"
"$PYTHON_BIN" -u ../main.py sweep train --backend slurm --interval 4
echo "Parameter sweep completed at: $(date)"
