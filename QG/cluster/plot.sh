#!/bin/bash
#SBATCH --job-name=plot1
#SBATCH --partition=mit_normal
#SBATCH --output=logs/plot%A.out
#SBATCH --error=logs/plot%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=32G
# Author: Isabella Thiel <thieli@mit.edu>


# Activate conda environment
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

#conda install -c conda-forge matplotlib-base



# start time
start_time=$(date +%s)
echo "Job started at: $(date)"

echo "========== Starting plotting =========="
"$PYTHON_BIN" -u compare_all.py

"$PYTHON_BIN" -u plot_data_ens.py Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.05 16.0 --ensemble ens01 ens08 --compare rnudge ens01 --tmin 500 --tmax 1500


echo "========== Plotting Finished =========="

# end time
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Job ended at: $(date)"
echo "Elapsed time: $elapsed seconds"

