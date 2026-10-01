#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benedikt Barthel Sorensen <bbarthel@mit.edu>
"""


import numpy as np
import scipy.io as sio
import os
import hdf5storage
from typing import Optional
import gc
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

print('Importing Tensorflow... ', end='', flush=True)
import tensorflow as tf
print('Done!')
print('Importing Keras... ', end='', flush=True)
print('Done!')


from utils import autocorr_fft, compute_skip_and_length, load_field, results_save_path
import time

tf.keras.mixed_precision.set_global_policy('float32')


##################### Ensemble Testing on Coarse DNS Data - Start ######################
def evaluate_training_data(
    model_type: str,
    case_name: str,
    dt_ml: float,
    basis_type: str,
    test_time_units: Optional[int] = None,
    neural_nets_dir="../Neural_Nets",
    ml_results_dir="../ML_Results",
    data_dir="../ML_Data",
    t_startup: int = 1000,
    case_name_train: Optional[str] = None,
    **kwargs):

    print('\nDeterministic Testing (Coarse DNS)\n----------------------------------')
    print(f'Case                    : {case_name}')
    print(f'Model type              : {model_type}')
    print(f'Basis                   : {basis_type}')
    T = test_time_units if test_time_units is not None else 1000
    # [0/6] Check if already done
    test_again = bool(kwargs.get("test_again", True))
    save_file = results_save_path(ml_results_dir, case_name_train, model_type, basis_type, 0, dt_ml, T, 0)

    if os.path.isfile(save_file) and not test_again:
        print(f"\nTest results already exist: {save_file}\nUse test_again=True to re-test.\n")
        return

    # -------- [1/6] Import coarse & reference headers --------
    print('[1/6] Importing Reference data for testing... ', end='', flush=True)
    load_file = os.path.join(data_dir, case_name, 'psi1_ref.mat')
    mat_contents = sio.loadmat(load_file)
    Ny = int(np.squeeze(mat_contents["Ny"]))
    Nx = int(np.squeeze(mat_contents["Nx"]))
    delta = mat_contents["delta"]
    beta  = mat_contents["beta"]
    rdrag = mat_contents["rdrag"]
    kd    = mat_contents["kd"]
    dt    = float(np.squeeze(mat_contents["dt"]))
    Lratio = 1.0

    skip_idx, total_times = compute_skip_and_length(mat_contents, t_startup, test_time_units, dt_ml, method='reversed')
    used_features = Ny * Nx * 2
    test_times = total_times

    # slice reference fields to the requested window [skip_idx : skip_idx+final_len]
    psi_ref = np.zeros((Ny, Nx, total_times, 2), dtype=np.float32)
    mat1 = mat_contents["psi1_ref"]
    psi_ref[:, :, :, 0] = mat1[:, :, skip_idx: skip_idx + total_times]
    print(f"psi1_ref shape: {mat1.shape}, using frames {skip_idx} to {skip_idx + total_times - 1}")

    mat2 = load_field(data_dir, case_name, "ref", 2)
    print(f"psi2_ref shape: {mat2.shape}, using frames {skip_idx} to {skip_idx + total_times - 1}")
    psi_ref[:, :, :, 1] = mat2[:, :, skip_idx: skip_idx + total_times]
    del mat_contents, mat1, mat2
    gc.collect()

    print('Done!\n')

    # -------- [2/6] Load coarse fields --------
    print('[2/6] Importing coarse fields for testing... ', end='', flush=True)
    psi1 = load_field(data_dir, case_name, 'rnudge16.0', 1)
    psi_coarse = np.zeros((Ny,Nx,total_times,2),dtype=np.float32)
    print(f"psi1_rnudge shape: {psi1.shape}, using frames {skip_idx} to {skip_idx + total_times - 1}")
    psi_coarse[:,:,:,0] = psi1[:, :, skip_idx: skip_idx + total_times]
    del psi1

    psi2 = load_field(data_dir, case_name, 'rnudge16.0', 2)
    print(f"psi2_rnudge shape: {psi2.shape}, using frames {skip_idx} to {skip_idx + total_times - 1}")
    psi_coarse[:,:,:,1] = psi2[:, :, skip_idx: skip_idx + total_times]
    del psi2
    gc.collect()

    print('Done!\n')


    # -------- [4/6] PDF parameters --------
    print('[4/6] Compute stats... ', end='', flush=True)
    t0 = time.time()
    Np  = int(2001)
    pleft, pright = -4.0, 4.0
    pdf_xaxis = np.linspace(pleft,pright,Np)
    dp = (pright-pleft)/float(Np-1)

    # preallocate pdf variables
    psit_flag = np.zeros((Ny,Nx),dtype=np.float32)
    psi_rpdf_ref = np.zeros((Nx,Ny,Np,2),dtype=np.float32)
    psi_rpdf_coarse = np.zeros((Nx,Ny,Np,2),dtype=np.float32)

    # -------- [5/8] Correlation / spectrum scaffolding --------
    Ni = 100
    dt_i = dt_ml*Ni
    NT = int(T/dt_ml)

    # PDF of area-over-threshold parameters
    Npt  = int(25)
    tleft, tright = 0.0, 4.0
    thresh = np.linspace(tleft,tright,Npt)
    dpt = (tright-tleft)/float(Npt-1)

    Rr = np.zeros((Ny,Nx,test_times,2),dtype=np.float32)
    Rc = np.zeros((Ny,Nx,test_times,2),dtype=np.float32)
    Sr = np.zeros((Ny,Nx,test_times,2),dtype=np.float32)
    Sc = np.zeros((Ny,Nx,test_times,2),dtype=np.float32)


    R_ref = np.zeros((Ny,Nx,int(test_times/Ni),2),dtype=np.float32)
    R_coarse = np.zeros((Ny,Nx,int(test_times/Ni),2),dtype=np.float32)

    S_ref = np.zeros((Ny,Nx,int(test_times/Ni),2),dtype=np.float32)
    S_coarse = np.zeros((Ny,Nx,int(test_times/Ni),2),dtype=np.float32)

    # zonal means
    Psi_ref = np.zeros((Ny,test_times,2),dtype=np.float32)
    Psi_coarse_zm = np.zeros((Ny,test_times,2),dtype=np.float32)
    indEE_ref = np.zeros((test_times,2),dtype=np.float32)

    time_start = 0
    time_end   = test_times - 1
    times = time_end-time_start+1
    flag_coef = float(times*Nx*Ny)

    # account for LSTM stride
    stride = 1
    psi_ref_s = psi_ref[:,:,0:-1:stride,:]
    psi_coarse_s = psi_coarse[:,:,0:-1:stride,:]

    # Update test_times to match actual sliced length
    test_times = psi_ref_s.shape[2]

    # Re-allocate correlation arrays to match actual test_times
    Rr = np.zeros((Ny,Nx,test_times,2),dtype=np.float32)
    Rc = np.zeros((Ny,Nx,test_times,2),dtype=np.float32)
    Sr = np.zeros((Ny,Nx,test_times,2),dtype=np.float32)
    Sc = np.zeros((Ny,Nx,test_times,2),dtype=np.float32)

    R_ref = np.zeros((Ny,Nx,int(test_times/Ni),2),dtype=np.float32)
    R_coarse = np.zeros((Ny,Nx,int(test_times/Ni),2),dtype=np.float32)

    S_ref = np.zeros((Ny,Nx,int(test_times/Ni),2),dtype=np.float32)
    S_coarse = np.zeros((Ny,Nx,int(test_times/Ni),2),dtype=np.float32)

    # zonal means
    Psi_ref = np.zeros((Ny,test_times,2),dtype=np.float32)
    Psi_coarse_zm = np.zeros((Ny,test_times,2),dtype=np.float32)
    indEE_ref = np.zeros((test_times,2),dtype=np.float32)

    aot_ref = np.zeros((Npt,test_times,2),dtype=np.float32)
    aot_coarse = np.zeros((Npt,test_times,2),dtype=np.float32)

    # -------- [4/6] Reference statistics (autocorr, spectra, PDFs) --------
    print()
    idx = np.arange(0, Rr.shape[2]-1, Ni, dtype=int)
    nsave = min(R_ref.shape[2], idx.size)
    print()
    bins = np.linspace(pleft, pright, Np+1)
    for l in range(2):
        print(f"        ↳ layer {l+1}/2: max|ref|={np.max(np.abs(psi_ref_s[:, :, :, l])):.5f}, "
              f"max={np.max(psi_ref_s[:, :, :, l]):.5f}", end='', flush=True)
        Rr[..., l], Sr[..., l] = autocorr_fft(psi_ref_s[..., l], test_times)
        Rc[..., l], Sc[..., l] = autocorr_fft(psi_coarse_s[..., l], test_times)
        for i in range(0,Nx):
            for j in range(0,Ny):
                if nsave > 0:
                    R_ref[j, i, 0:nsave, l] = Rr[j, i, idx[:nsave], l]
                    R_coarse[j, i, 0:nsave, l] = Rc[j, i, idx[:nsave], l]
                    S_ref[j, i, 0:nsave, l] = Sr[j, i, idx[:nsave], l]
                    S_coarse[j, i, 0:nsave, l] = Sc[j, i, idx[:nsave], l]

                # PDFs (reference & coarse)
                vals_ref = psi_ref_s[i, j, :-1, l]
                vals_coarse = psi_coarse_s[i, j, :-1, l]

                hist_ref, _ = np.histogram(vals_ref, bins=bins)
                hist_coarse, _ = np.histogram(vals_coarse, bins=bins)

                # broadcast back into memmaps if needed
                psi_rpdf_ref[i, j, :, l] = hist_ref
                psi_rpdf_coarse[i, j, :, l] = hist_coarse

        print('...Done.', flush=True)

    # global spectra
    spectrum_ref    =  np.mean(Sr,axis=(0,1))
    spectrum_coarse =  np.mean(Sc,axis=(0,1))

    # zonal means & EE indicator
    Psi_ref        = np.sum(psi_ref_s,axis=1)/float(Nx)
    Psi_coarse_zm  = np.sum(psi_coarse_s,axis=1)/float(Nx)
    indEE_ref      = np.sum(np.abs(psi_ref_s),axis=(0,1))/float(Nx*Ny)

    # Pdf of area-over-threshold (moved before slicing)
    for l in range(2):
        print(f"        ↳ AOT layer {l+1}/2 ...", end='', flush=True)
        vals_ref = np.abs(psi_ref_s[..., l])  # (Ny,Nx,T)
        vals_coarse = np.abs(psi_coarse_s[..., l])

        mask_ref = vals_ref[..., None] >= thresh  # (Ny,Nx,T,Npt)
        mask_coarse = vals_coarse[..., None] >= thresh

        aot_ref[..., l] = mask_ref.mean(axis=(0,1)).T    # (Npt,T)
        aot_coarse[..., l] = mask_coarse.mean(axis=(0,1)).T
        print(f" done.", flush=True)

    del psi_ref_s, psi_coarse_s, Rr, Rc, Sr, Sc
    gc.collect()
    print("Done! ({:.2f} seconds)".format(time.time() - t0))
    print()


    os.makedirs(ml_results_dir, exist_ok=True)
    save_file = results_save_path(ml_results_dir, case_name_train, model_type, basis_type, 0, dt_ml, T, 0)

    data_dict = {
        "pdf_xaxis": pdf_xaxis, 'test_times': test_times,
        "psi_rpdf_ref": psi_rpdf_ref, "psi_rpdf_coarse": psi_rpdf_coarse,
        "Psi_ref": Psi_ref, "Psi_coarse": Psi_coarse_zm,
        "spectrum_ref": spectrum_ref, "spectrum_coarse": spectrum_coarse,
        "R_ref": R_ref, "R_coarse": R_coarse,
        "S_ref": S_ref, "S_coarse": S_coarse,
        'dt_ml': dt_ml, 'dt_i': dt_i, 'thresh': thresh,
        'aot_ref': aot_ref, 'aot_coarse': aot_coarse,
        'indEE_ref': indEE_ref
    }

    hdf5storage.savemat(save_file, data_dict, format='7.3', do_compression=True)
    print(f'Ensemble Stats Saved: {save_file} \n')
    return


if __name__ == "__main__":

    model_type = "STORN"
    case_name = "Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.1"
    basis_type = "rnudge16.0"
    dt_ml = 0.1
    test_time_units_list = [1000, 500, 200, 100, 50, 10]

    for test_time_units in test_time_units_list:

        evaluate_training_data(
            model_type=model_type,
            case_name=case_name,
            dt_ml=dt_ml,
            basis_type=basis_type,
            test_time_units=test_time_units,
            neural_nets_dir="../Neural_Nets",
            ml_results_dir="../ML_Results",
            data_dir="../ML_Data",
            t_startup=100,
            case_name_train=case_name
        )
