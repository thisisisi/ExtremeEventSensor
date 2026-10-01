#!/bin/bash
# Author: Isabella Thiel <thieli@mit.edu>

#SBATCH -p mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH --time=10:10:00
#SBATCH -J sigma
#SBATCH -o output/stdout.%j
#SBATCH -e output/stderr.%j

set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
module purge



echo "=========== Starting Sigma Informativeness Computation ==========="
echo


cd learning/data_diagnostics

"$PYTHON_BIN" sigma_informativeness.py \
  --case sig0.05=../../ML_Data/Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.05 \
  --case sig0.1=../../ML_Data/Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.1 \
  --case sig0.2=../../ML_Data/Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.2 \
  --case sig0.25=../../ML_Data/Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.25 \
  --case sig0.375=../../ML_Data/Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.375 \
  --case sig0.4=../../ML_Data/Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.4 \
  --outdir ../../ML_Data/diagnostics_sigma \
  --tmin 500.0

# python sigma_informativeness.py \
#   --case sig0.05=../../ML_Data/merged_Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization3_24x24_tau16.0_sig0.05 \
#   --case sig0.1=../../ML_Data/merged_Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization3_24x24_tau16.0_sig0.1 \
#   --case sig0.25=../../ML_Data/merged_Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization3_24x24_tau16.0_sig0.25 \
#   --case sig0.375=../../ML_Data/merged_Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization3_24x24_tau16.0_sig0.375 \
#   --outdir ../../ML_Data/diagnostics_sigma


echo
echo "=========== Sigma Informativeness Computation Finished ==========="

  # --case sig0.5=../../ML_Data/merged_Qi_Majda_2018_beta2.0_rdrag0.1_realization1_24x24_tau16.0_sig0.5 \
  # --case sig0.1=../../ML_Data/merged_Qi_Majda_2018_beta2.0_rdrag0.1_realization1_24x24_tau16.0_sig0.1 \
  # --case sig0.025=../../ML_Data/merged_Qi_Majda_2018_beta2.0_rdrag0.1_realization1_24x24_tau16.0_sig0.025 \
  # --case sig0.005=../../ML_Data/merged_Qi_Majda_2018_beta2.0_rdrag0.1_realization1_24x24_tau16.0_sig0.005 \
  # --case sig5e-05=../../ML_Data/merged_Qi_Majda_2018_beta2.0_rdrag0.1_realization1_24x24_tau16.0_sig5e-05 \
