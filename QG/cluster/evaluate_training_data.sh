#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=15G
#SBATCH --time=3:00:00
#SBATCH -J ML_test
#SBATCH -o output/%j.testtrainstdout
#SBATCH -e output/%j.testtrainstderr
# Author: Isabella Thiel <thieli@mit.edu>

# Load required modules
module purge
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"


echo "===================="

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "===================="





echo



echo
echo "========== Starting Evaluation Computation =========="
"$PYTHON_BIN" -u evaluate_training_data.py 
echo "========== Evaluation Computation Finished =========="

# Print completion information
echo
echo "Job completed at: $(date)"
