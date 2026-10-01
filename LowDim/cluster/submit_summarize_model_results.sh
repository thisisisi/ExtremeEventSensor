#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=5:00:00
#SBATCH -J lowdim_summary
#SBATCH -o output/%j.lowdim_summary_stdout
#SBATCH -e output/%j.lowdim_summary_stderr
# Author: Isabella Thiel <thieli@mit.edu>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default to workspace path so job doesn't attempt to create directories inside Slurm spool
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"
if [[ ! -d "$PROJECT_DIR" ]]; then
  PROJECT_DIR="$SCRIPT_DIR"
fi
if [[ -n "${CONDA_ENV:-}" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV/bin/python}"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

cd "$PROJECT_DIR"
# Ensure output lives in the workspace, not in Slurm spool dir
mkdir -p "$PROJECT_DIR/output" || echo "Note: output dir creation had issues (may already exist)"

RESULTS_ROOT="${RESULTS_ROOT:-ML_Results}"
OUTPUT_DIR="${OUTPUT_DIR:-ML_Results/summary_tables}"
SIGMAS="${SIGMAS:-1e-4,1e-5,5e-6,1e-6}"
ENSEMBLE_SIZES="${ENSEMBLE_SIZES:-0,1,2,4,8,16}"
ARCHITECTURES="${ARCHITECTURES:-attention}"
RUN_ID_TRAIN="${RUN_ID_TRAIN:-0}"
RUN_ID_TEST="${RUN_ID_TEST:-1}"
STRIDE="${STRIDE:-}"
REPEAT_INDICES="${REPEAT_INDICES:-}"
THRESHOLD_PERCENTILES="${THRESHOLD_PERCENTILES:-90,95,99}"
VARIABLES="${VARIABLES:-0,1,2}"
PDF_GRID_POINTS="${PDF_GRID_POINTS:-1200}"

module purge

echo "===================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Project directory: $PROJECT_DIR"
echo "Python executable: $PYTHON_BIN"
echo "Results root: $RESULTS_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "===================="
echo
echo "========== Starting Summary =========="

SUMMARY_SCRIPT="$PROJECT_DIR/summarize.py"

cmd=(
  "$PYTHON_BIN" -u "$SUMMARY_SCRIPT"
  --results_root "$RESULTS_ROOT"
  --output_dir "$OUTPUT_DIR"
  --sigmas "$SIGMAS"
  --ensemble_sizes "$ENSEMBLE_SIZES"
  --architectures "$ARCHITECTURES"
  --run_id_train "$RUN_ID_TRAIN"
  --run_id_test "$RUN_ID_TEST"
  --threshold_percentiles "$THRESHOLD_PERCENTILES"
  --variables "$VARIABLES"
  --pdf_grid_points "$PDF_GRID_POINTS"
)

if [[ -n "$STRIDE" ]]; then
  cmd+=(--stride "$STRIDE")
fi

if [[ -n "$REPEAT_INDICES" ]]; then
  cmd+=(--repeat_indices "$REPEAT_INDICES")
fi

"${cmd[@]}"

echo "========== Summary Finished =========="
echo
echo "Job completed at: $(date)"