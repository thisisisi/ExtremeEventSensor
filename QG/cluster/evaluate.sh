#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=84G
#SBATCH --time=6:00:00
#SBATCH -J ML_test
#SBATCH -o output/%j.teststdout
#SBATCH -e output/%j.teststderr

set -euo pipefail
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source ../cluster/job_runtime.sh
start_job
"$PYTHON_BIN" -u evaluate_qg.py --config_file "$CONFIG_DST"
finish_job
