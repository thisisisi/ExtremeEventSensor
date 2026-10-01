#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH -J MLtrain
#SBATCH -o output/%j.stdout
#SBATCH -e output/%j.stderr

set -euo pipefail
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source ../cluster/job_runtime.sh
start_job
"$PYTHON_BIN" -u train_qg.py --config_file "$CONFIG_DST"
finish_job
