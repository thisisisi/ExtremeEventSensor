#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benedikt Barthel Sorensen <bbarthel@mit.edu>
"""


import numpy as np
import scipy.io as sio
import os
import fcntl
import hdf5storage
from typing import Optional
import math
import gc
import json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

print('Importing Tensorflow... ', end='', flush=True)
import tensorflow as tf
print('Done!')
print('Importing Keras... ', end='', flush=True)
print('Done!')

from loss_params_qg import loss_params_2D
from networks_qg import get_model, sampling

from utils import (
    autocorr_fft,
    compute_skip_and_length,
    epochs_of,
    load_field,
    model_save_path,
    results_save_path,
    resume_or_initialize_training,
    seed_everything,
)
import time

tf.keras.mixed_precision.set_global_policy('float32')


# ------------------------------- Helpers ------------------------------
#%% Create Model

def _scalar(x):
    try:
        import numpy as _np
        return int(_np.squeeze(x))
    except Exception:
        return x

def make_dataset_det(
    psi_train: np.ndarray,         # (Ny, Nx, T, 2) float32
    psi_ref: np.ndarray,           # (Ny, Nx, T, 2) float32
    Ny: int,
    Nx: int,
    data_times: int,
    lstm_hist_points: int,
    dt_ml: float,
    dt: float,
    sub_sample: int,
    batch_size: int,
    ):
    """
    Deterministic generator dataset (no shuffle), reproducing the exact windowing
    used in the original code:
      - stride s = floor(dt_ml / dt), s >= 1
      - windows of length H = lstm_hist_points
      - step between windows = sub_sample
      - inputs:  [psi_train[...,0], psi_train[...,1]] flattened in Fortran order
      - targets: [psi_ref  [...,0], psi_ref  [...,1]] flattened in Fortran order
      - 90/10 split in sample order (deterministic)
    """
    used_features = Ny * Nx * 2
    lstm_stride   = max(1, int(dt_ml / float(dt)))

    # number of valid starting positions (same as your original)
    train_times = int(np.floor((data_times - lstm_hist_points * lstm_stride) // sub_sample))
    while train_times < 10 and lstm_hist_points > 1:
        # try reducing history length to get at least 10 training samples
        lstm_hist_points -= 1
        train_times = int(np.floor((data_times - lstm_hist_points * lstm_stride) // sub_sample))
    print(f"Adjusted lstm_hist_points: {lstm_hist_points}, resulting in train_times: {train_times}")
    if train_times <= 0:
        raise ValueError("No training windows possible with given T/lstm_hist_points/dt_ml.")

    # Pre-flatten once (T, 2*Ny*Nx), float32, Fortran order to match original reshape
    flat_train = np.concatenate(
        [
            psi_train[..., 0].reshape(Ny * Nx, data_times, order="F"),
            psi_train[..., 1].reshape(Ny * Nx, data_times, order="F"),
        ],
        axis=0,
    ).T.astype(np.float32, copy=False)

    flat_ref = np.concatenate(
        [
            psi_ref[..., 0].reshape(Ny * Nx, data_times, order="F"),
            psi_ref[..., 1].reshape(Ny * Nx, data_times, order="F"),
        ],
        axis=0,
    ).T.astype(np.float32, copy=False)

    # indices inside a window
    idx_vec = (np.arange(lstm_hist_points, dtype=np.int64) * lstm_stride)

    def gen():
        # Deterministic order, exactly like your old nested loops:
        # base_t = sub_sample * tt; t_idx = base_t + lstm_stride * pp
        for tt in range(train_times):
            base_t = sub_sample * tt
            t_idx  = base_t + idx_vec            # (H,)
            X_win  = flat_train[t_idx, :]        # (H, D)
            Y_win  = flat_ref[t_idx, :]          # (H, D)
            # ensure float32 dtypes
            yield (X_win.astype(np.float32, copy=False),
                   Y_win.astype(np.float32, copy=False))

    output_signature = (
        tf.TensorSpec(shape=(lstm_hist_points, used_features), dtype=tf.float32),
        tf.TensorSpec(shape=(lstm_hist_points, used_features), dtype=tf.float32),
    )
    base_ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    # 90/10 split (deterministic, in-order)
    N_total = train_times
    N_train = int(0.9 * N_total)
    N_val   = N_total - N_train

    train_ds = (
        base_ds.take(N_train)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()                       # avoid OutOfRange across many epochs
    )
    val_ds = (
        base_ds.skip(N_train)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
    )
    return train_ds, val_ds, N_train, N_val, used_features


class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("Starting training")

    def on_epoch_end(self, epoch, logs=None):
        print("Epoch training")

    #def on_train_batch_end(self, batch, logs=None):


######################## Deterministic Training ###################################


def train(
    model_type: str,
    case_name: str,
    j_ens: int,
    Nepochs: int,
    sub_sample: int,
    lstm_hist_points: int,
    dt_ml: float,
    basis_type: str,
    T: int,
    batch_size: int = 32,
    lr: Optional[float] = None,
    neural_nets_dir: str = "../Neural_Nets",
    ml_results_dir: str = "../ML_Results",
    data_dir: str = "../ML_Data",
    t_startup: int = 1000,
    **kwargs
    ):
    """
    Deterministic trainer (legacy) with consistent print layout and knobs (batch size, lr).
    """
    print("\nDeterministic Training\n----------------------")
    print(f"Case                    : {case_name}")
    print(f"Model type              : {model_type}")
    print(f"Basis                   : {basis_type}")
    print(f"Ensemble ID             : {j_ens}")

    repeat_index = kwargs.get("repeat_index", None)
    seed = kwargs.get("seed", None)
    if seed is None and repeat_index is not None:
        seed = 1729 + int(repeat_index)
    if seed is not None:
        seed_everything(seed)


    # ---------------- [1/5] Load fields ----------------
    print("\n[1/5] Loading R-nudged and Reference data ... ", end="", flush=True)
    load_file = os.path.join(data_dir, case_name, "psi1_ref.mat")
    mat_contents = sio.loadmat(load_file)
    Ny = int(mat_contents["Ny"])//1
    Nx = int(mat_contents["Nx"])//1
    delta = mat_contents["delta"]; beta = mat_contents["beta"]
    rdrag = mat_contents["rdrag"]; kd = mat_contents["kd"]
    dt = mat_contents["dt"]
    Lratio = 1.0

    print("\n Baseline\n-----------------------------------")
    print(f"Case                    : {case_name}")
    print(f"Model type              : {model_type}")
    print(f"Basis                   : {basis_type}")
    print(f"Ensemble ID (ckpt tag)  : {j_ens}")

    method = "reversed" # "standard" or "reversed" for skip computation
    skip_idx, data_times = compute_skip_and_length(mat_contents, t_startup, T, dt_ml, method=method)


    psi1 = mat_contents["psi1_ref"]
    print(f"  • loaded ref layer 1, shape: {psi1.shape}")
    psi_ref = np.zeros((Ny, Nx, data_times, 2), dtype=np.float32)
    if method == "reversed":
        psi_ref[:, :, :, 0] = psi1[:, :, -data_times:]
    else:
        psi_ref[:, :, :, 0] = psi1[:, :, skip_idx:skip_idx + data_times]
    print(f"    psi1_ref shape: {psi1.shape}, using frames {skip_idx} to {skip_idx + data_times - 1}")
    del mat_contents, psi1

    psi2 = load_field(data_dir, case_name, "ref", 2)
    if method == "reversed":
        psi_ref[:, :, :, 1] = psi2[:, :, -data_times:]
    else:
        psi_ref[:, :, :, 1] = psi2[:, :, skip_idx:skip_idx + data_times]
    print(f"    psi2_ref shape: {psi2.shape}, using frames {skip_idx} to {skip_idx + data_times - 1}")
    del psi2
    gc.collect()

    psi1 = load_field(data_dir, case_name, basis_type, 1)
    print(f"  • loaded r-nudged layer 1, shape: {psi1.shape}")
    psi_train = np.zeros((Ny, Nx, data_times, 2), dtype=np.float32)
    if method == "reversed":
        psi_train[:, :, :, 0] = psi1[:, :, -data_times:]
    else:
        psi_train[:, :, :, 0] = psi1[:, :, skip_idx:skip_idx + data_times]
    print(f"psi1_rnudge shape: {psi1.shape}, using frames {skip_idx} to {skip_idx + data_times - 1}")
    del psi1

    psi2 = load_field(data_dir, case_name, basis_type, 2)
    if method == "reversed":
        psi_train[:, :, :, 1] = psi2[:, :, -data_times:]
    else:
        psi_train[:, :, :, 1] = psi2[:, :, skip_idx:skip_idx + data_times]
    print(f"psi2_rnudge shape: {psi2.shape}, using frames {skip_idx} to {skip_idx + data_times - 1}")
    del psi2
    gc.collect()

    print(f"Ref field zeros   : layer 1: {(psi_ref[:,:,:,0]==0).sum()}, layer 2: {(psi_ref[:,:,:,1]==0).sum()}")
    print(f"Train field zeros : layer 1: {(psi_train[:,:,:,0]==0).sum()}, layer 2: {(psi_train[:,:,:,1]==0).sum()}")

    print("Done.")
    print(f"beta = {beta}, rdrag = {rdrag}, kd = {kd}")
    print(f"Grid Ny×Nx              : {Ny}×{Nx}")
    print(f"data_times              : {data_times}")
    print(f"dt (from data)          : {dt}")

    # ---------------- [2/5] Windowing ------------------
    print("\n[2/5] Building training windows ...", flush=True, end="")
    print(f"Size of Original Data   : {data_times}")
    total_times = data_times  #int(np.floor(T / dt))

    # LSTM stride to achieve desired dt for ML model
    lstm_stride = int((dt_ml / dt))
    time_start = 0
    train_times = total_times - lstm_hist_points * lstm_stride - time_start
    train_times = int(np.floor(train_times // sub_sample))
    train_times = max(0, train_times)

    print(f"Size of Training Data   : {train_times}")
    print(f"ML Time Step            : {dt_ml}")
    print(f"LSTM Stride             : {lstm_stride}")
    print(f"sub_sample (window step): {sub_sample}")
    if train_times <= 0:
        raise ValueError("No training windows possible with given T/lstm_hist_points/dt_ml.")

    # Build generator datasets (no shuffle, exact old windowing, 90/10 split)
    train_ds, val_ds, N_train, N_val, used_features = make_dataset_det(
        psi_train=psi_train,
        psi_ref=psi_ref,
        Ny=Ny, Nx=Nx,
        data_times=data_times,
        lstm_hist_points=lstm_hist_points,
        dt_ml=dt_ml,
        dt=float(dt),
        sub_sample=sub_sample,
        batch_size=int(batch_size),
    )
    del psi_train, psi_ref
    gc.collect()
    print(f"done", flush=True, end="")


    # ---------------- [3/5] Model ----------------------
    print("\n[3/5] Building model ...", flush=True)
    os.makedirs(neural_nets_dir, exist_ok=True)

    loss_data = loss_params_2D(kd, delta, Ny, Nx, beta, rdrag)


    model = get_model(model_type, used_features, loss_data, lr=lr)


    # ---------------- [4/5] Train ----------------------
    print("\n[4/5] Training ...", flush=True)
    Ne = 500
    Nsets = int(np.ceil(Nepochs / Ne))
    save_path_final = model_save_path(
        neural_nets_dir, case_name, model_type, basis_type, Nepochs, dt_ml, T, j_ens,
        repeat_index=repeat_index,
    )
    train_again = bool(kwargs.get("train_again", False))
    if os.path.isfile(save_path_final) and not train_again:
        # check if the file is valid
        try:
            _ = tf.keras.models.load_model(save_path_final, compile=False)
        except Exception as e:
            print(f"Warning: existing model file {save_path_final} is corrupted or incompatible. Overwriting.")
        else:
            print(f"\nModel already exists: {save_path_final}\nSkipping training.\nUse a different Nepochs or set train_again=True to overwrite.\n")
            print("\n[5/5] Done.\n")
            return

    save_path_list = [
        model_save_path(
            neural_nets_dir, case_name, model_type, basis_type, (je+1)*Ne, dt_ml, T, j_ens,
            repeat_index=repeat_index,
        )
        for je in range(Nsets)
    ]
    model, Nsets, latest_ckpt = resume_or_initialize_training(
        model, save_path_list, Nepochs, Ne, train_again=train_again
    )
    done_epochs = epochs_of(latest_ckpt) if latest_ckpt is not None else 0

    steps_per_epoch  = max(1, math.ceil(N_train / int(batch_size)))
    validation_steps = max(1, math.ceil(N_val   / int(batch_size)))
    train_history = None
    for je in range(Nsets):
        hist_tmp = model.fit(
            train_ds,
            epochs=Ne,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            verbose=2,
        )
        if train_history is None:
            train_history = hist_tmp
        else:
            for key in train_history.history.keys():
                train_history.history[key].extend(hist_tmp.history[key])
        del hist_tmp

        Nepochs_j = (je + 1) * Ne + done_epochs
        save_path = model_save_path(
            neural_nets_dir, case_name, model_type, basis_type, Nepochs_j, dt_ml, T, j_ens,
            repeat_index=repeat_index,
        )
        model.save(save_path, save_format="h5")
        print(f"  Saved: {save_path}", flush=True)

    gc.collect()

    hist_path = model_save_path(
        neural_nets_dir, case_name, model_type, basis_type, Nepochs, dt_ml, T, j_ens,
        repeat_index=repeat_index,
    )
    hist_path = hist_path.replace('.h5', '_history.json')

    with open(hist_path, 'w') as f:
        json.dump(train_history.history, f)
        print(f"  Saved training history: {hist_path}", flush=True)

    # ---------------- [5/5] Cleanup --------------------
    # del input_train, output_train
    print("\n[5/5] Done.\n")
    return


##################### Ensemble Testing on Coarse DNS Data - Start ######################
def evaluate(
    model_type: str,
    case_name: str,
    Nens: int,
    Nepochs: int,
    dt_ml: float,
    basis_type: str,
    T: int,
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
    print(f'Nens                    : {Nens}')

    repeat_index = kwargs.get("repeat_index", None)

    # [0/6] Check if already done
    test_again = bool(kwargs.get("test_again", True))
    save_file = results_save_path(
        ml_results_dir, case_name_train, model_type, basis_type, Nepochs, dt_ml, T, Nens,
        repeat_index=repeat_index
    )

    if os.path.isfile(save_file) and not test_again:
        print(f"\nTest results already exist: {save_file}\nUse test_again=True to re-test.\n")
        return

    io_lock_path = "/tmp/qg_test_data_load.lock"
    io_lock = open(io_lock_path, "w")
    print(f"Waiting for node-local test-data I/O slot ({io_lock_path})... ", end="", flush=True)
    fcntl.flock(io_lock, fcntl.LOCK_EX)
    print("acquired.", flush=True)

    # -------- [1/6] Import coarse & reference headers --------
    print('[1/6] Importing Reference data for testing... ', end='', flush=True)
    load_file = os.path.join(data_dir, case_name, 'psi1_ref.mat')
    mat_contents = sio.loadmat(load_file)
    Ny = int(mat_contents["Ny"])//1
    Nx = int(mat_contents["Nx"])//1
    delta = mat_contents["delta"]
    beta  = mat_contents["beta"]
    rdrag = mat_contents["rdrag"]
    kd    = mat_contents["kd"]
    dt    = float(mat_contents["dt"])
    Lratio = 1.0

    skip_idx, total_times = compute_skip_and_length(mat_contents, t_startup, test_time_units, dt_ml)
    used_features = Ny * Nx * 2

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
    psi1 = load_field(data_dir, case_name, 'coarse', 1)
    psi_coarse = np.zeros((Ny,Nx,total_times,2),dtype=np.float32)
    print(f"psi1_coarse shape: {psi1.shape}, using frames {skip_idx} to {skip_idx + total_times - 1}")
    psi_coarse[:,:,:,0] = psi1[:, :, skip_idx: skip_idx + total_times]
    del psi1

    psi2 = load_field(data_dir, case_name, 'coarse', 2)
    print(f"psi2_coarse shape: {psi2.shape}, using frames {skip_idx} to {skip_idx + total_times - 1}")
    psi_coarse[:,:,:,1] = psi2[:, :, skip_idx: skip_idx + total_times]
    del psi2
    gc.collect()

    # count total number of zeros
    print(f"Coarse field zeros: layer 1: {(psi_coarse[:,:,:,0]==0).sum()}, layer 2: {(psi_coarse[:,:,:,1]==0).sum()}")
    print(f"Ref field zeros   : layer 1: {(psi_ref[:,:,:,0]==0).sum()}, layer 2: {(psi_ref[:,:,:,1]==0).sum()}")

    print('Done!\n')

    # -------- [3/8] Build test input (with stride) --------
    print('[3/6] Building test input... ', end='', flush=True)
    input_test = np.zeros((1,total_times,used_features), dtype = np.float32)
    input_test[0,:,0:Nx*Ny]        = np.reshape(np.transpose(psi_coarse[:,:,:,0],(2,0,1)),
                                                (1,total_times,Nx*Ny), order='F')
    input_test[0,:,Nx*Ny:2*Nx*Ny]  = np.reshape(np.transpose(psi_coarse[:,:,:,1],(2,0,1)),
                                                (1,total_times,Nx*Ny), order='F')

    stride = int((dt_ml/float(dt)))
    stride = max(1, stride)
    input_test = input_test[0,0:-1:stride,:]
    input_test = np.reshape(input_test,(1,input_test.shape[0],input_test.shape[1]))
    test_times = input_test.shape[1]
    print('Data Reshape Complete!\n')
    fcntl.flock(io_lock, fcntl.LOCK_UN)
    io_lock.close()
    print("Released shared test-data I/O slot.", flush=True)

    # -------- [4/6] PDF parameters --------
    print('[4/6] Compute stats... ', end='', flush=True)
    t0 = time.time()
    Np  = int(2001)
    pleft, pright = -4.0, 4.0
    pdf_xaxis = np.linspace(pleft,pright,Np)
    dp = (pright-pleft)/float(Np-1)

    # preallocate pdf variables
    psit_flag = np.zeros((Ny,Nx),dtype=np.float32)
    psi_pdf_ML = np.zeros((Nens,Np,2),dtype=np.float32)
    psi_rpdf_ML = np.zeros((Nens,Nx,Ny,Np,2),dtype=np.float32)
    psi_rpdf_ref = np.zeros((Nx,Ny,Np,2),dtype=np.float32)
    psi_rpdf_coarse = np.zeros((Nx,Ny,Np,2),dtype=np.float32)

    # -------- [5/8] Correlation / spectrum scaffolding --------
    Ni = 100
    dt_i = dt_ml*Ni
    NT = int(T/dt_ml)

    Rr = np.zeros((Ny,Nx,test_times,2),dtype=np.float32)
    Rc = np.zeros((Ny,Nx,test_times,2),dtype=np.float32)
    Rt = np.zeros((Ny,Nx,NT,2),dtype=np.float32)
    Sr = np.zeros((Ny,Nx,test_times,2),dtype=np.float32)
    Sc = np.zeros((Ny,Nx,test_times,2),dtype=np.float32)
    St = np.zeros((Ny,Nx,NT,2),dtype=np.float32)

    R_ML = np.zeros((Nens,Ny,Nx,int(test_times/Ni),2),dtype=np.float32)
    R_ref = np.zeros((Ny,Nx,int(test_times/Ni),2),dtype=np.float32)
    R_coarse = np.zeros((Ny,Nx,int(test_times/Ni),2),dtype=np.float32)
    S_ML = np.zeros((Nens,Ny,Nx,int(test_times/Ni),2),dtype=np.float32)
    S_ref = np.zeros((Ny,Nx,int(test_times/Ni),2),dtype=np.float32)
    S_coarse = np.zeros((Ny,Nx,int(test_times/Ni),2),dtype=np.float32)

    spectrum_ML = np.zeros((Nens,test_times,2),dtype=np.float32)

    # zonal means
    Psi_ML = np.zeros((Nens,Ny,test_times,2),dtype=np.float32)
    Psi_ref = np.zeros((Ny,test_times,2),dtype=np.float32)
    Psi_coarse_zm = np.zeros((Ny,test_times,2),dtype=np.float32)
    indEE_ML = np.zeros((Nens,test_times,2),dtype=np.float32)
    indEE_ref = np.zeros((test_times,2),dtype=np.float32)

    time_start = 0
    time_end   = test_times - 1
    times = time_end-time_start+1
    flag_coef = float(times*Nx*Ny)

    # account for LSTM stride
    psi_ref_s = psi_ref[:,:,0:-1:stride,:]
    psi_ref = psi_ref_s
    psi_coarse_s = psi_coarse[:,:,0:-1:stride,:]
    psi_coarse2 = psi_coarse_s


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
        Rt[..., l], St[..., l] = autocorr_fft(psi_ref_s[..., :NT, l], NT)
        for i in range(0,Nx):
            for j in range(0,Ny):
                # full version to compute spectra
                # Rr[j,i,:,l] = scs.correlate(psi_ref[j,i,:,l], psi_ref[j,i,:,l], mode='same', method='auto')
                # Rc[j,i,:,l] = scs.correlate(psi_coarse2[j,i,:,l], psi_coarse2[j,i,:,l], mode='same', method='auto')
                # Rt[j,i,:,l] = scs.correlate(psi_ref[j,i,0:NT,l], psi_ref[j,i,0:NT,l], mode='same', method='auto')

                # Sr[j,i,:,l] = np.abs(np.fft.fftshift(np.fft.fft(Rr[j,i,:,l])))/test_times
                # Sc[j,i,:,l] = np.abs(np.fft.fftshift(np.fft.fft(Rc[j,i,:,l])))/test_times
                # St[j,i,:,l] = np.abs(np.fft.fftshift(np.fft.fft(Rt[j,i,:,l])))/NT

                # interpolated version to save (robust to off-by-one lengths)
                # compute indices analogous to [0:-1:Ni] but ensure assignment fits

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

    # training-time windows stats
    R_train = 1*Rt
    S_train = 1*St

    # global spectra
    spectrum_ref    =  np.mean(Sr,axis=(0,1))
    spectrum_coarse =  np.mean(Sc,axis=(0,1))
    spectrum_train  =  np.mean(St,axis=(0,1))

    # zonal means & EE indicator
    Psi_ref        = np.sum(psi_ref,axis=1)/float(Nx)
    Psi_coarse_zm  = np.sum(psi_coarse_s,axis=1)/float(Nx)
    indEE_ref      = np.sum(np.abs(psi_ref),axis=(0,1))/float(Nx*Ny)

    # Pdf of area-over-threshold
    Npt  = int(25)
    tleft, tright = 0.0, 4.0
    thresh = np.linspace(tleft,tright,Npt)
    dpt = (tright-tleft)/float(Npt-1)

    aot_ref    = np.zeros((Npt,test_times,2),dtype=np.float32)
    aot_coarse = np.zeros((Npt,test_times,2),dtype=np.float32)
    aot_ML     = np.zeros((Nens,Npt,test_times,2),dtype=np.float32)
    for l in range(2):
        print(f"        ↳ AOT layer {l+1}/2 ...", end='', flush=True)
        vals_ref = np.abs(psi_ref_s[..., l])  # (Ny,Nx,T)
        vals_coarse = np.abs(psi_coarse_s[..., l])

        mask_ref = vals_ref[..., None] >= thresh  # (Ny,Nx,T,Npt)
        mask_coarse = vals_coarse[..., None] >= thresh

        aot_ref[..., l] = mask_ref.mean(axis=(0,1)).T    # (Npt,T)
        aot_coarse[..., l] = mask_coarse.mean(axis=(0,1)).T
        print(f" done.", flush=True)
    del psi_ref, psi_coarse2, Rr, Rc, Rt, Sr, Sc, St
    gc.collect()
    print("Done! ({:.2f} seconds)".format(time.time() - t0))
    print()

    # -------- [5/6] ML predictions --------
    print('[5/6] ML predictions\n', end='', flush=True)

    for jens in range(0,Nens):
        print('     ...realization # ' + str(jens+1), end='', flush=True)

        ckpt = model_save_path(
            neural_nets_dir, case_name_train, model_type, basis_type, Nepochs, dt_ml, T, jens + 1,
            repeat_index=repeat_index,
        )
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        if model_type.lower() == 'storn':
            model1 = tf.keras.models.load_model(ckpt, compile=False,
                                    custom_objects={"sampling": sampling})
        else:
            model1 = tf.keras.models.load_model(ckpt, compile=False)

        output_test = np.zeros((1, test_times, used_features), dtype=np.float32)
        pred = model1.predict(input_test, verbose=2)
        output_test[:, :, :] = pred[:, :, 0:used_features]

        # Reshape and Save Results
        psi_ML = np.zeros((Ny,Nx,test_times,2),dtype=np.float32)
        Rm = np.zeros((Ny,Nx,test_times,2),dtype=np.float32)
        Sm = np.zeros((Ny,Nx,test_times,2),dtype=np.float32)

        psi_ML[:,:,:,0] = np.transpose(np.reshape(output_test[0,:,0:Nx*Ny],(test_times,Nx,Ny), order='F'),(1,2,0))
        psi_ML[:,:,:,1] = np.transpose(np.reshape(output_test[0,:,Nx*Ny:2*Nx*Ny],(test_times,Nx,Ny), order='F'),(1,2,0))
        del output_test
        gc.collect()

        # Spectra, PDFs
        idx_m = np.arange(0, Rm.shape[2]-1, Ni, dtype=int)
        nsave_m = min(R_ML.shape[3], idx_m.size)
        for l in range(2):
            print(f"        ↳ layer {l+1}/2: max|ML|={np.max(np.abs(psi_ML[..., l])):.5f}, "
                    f"max={np.max(psi_ML[..., l]):.5f}", flush=True)
            Rm[..., l], Sm[..., l] = autocorr_fft(psi_ML[..., l], test_times)
            for i in range(0,Nx):
                for j in range(0,Ny):
                    if nsave_m > 0:
                        R_ML[jens, j, i, 0:nsave_m, l] = Rm[j, i, idx_m[:nsave_m], l]
                        S_ML[jens, j, i, 0:nsave_m, l] = Sm[j, i, idx_m[:nsave_m], l]

                    vals = psi_ML[j, i, :-1, l]
                    hist, _ = np.histogram(vals, bins=bins)
                    psi_rpdf_ML[jens, i, j, :, l] = hist
                    psi_pdf_ML[jens, :, l] += hist

        # Power Spectra (global)
        spectrum_ML[jens,:,:] =  np.mean(Sm,axis=(0,1))
        # Zonal Mean
        Psi_ML[jens,:,:,:] = np.sum(psi_ML,axis=1)/float(Nx)
        # EE Indicator
        indEE_ML[jens,:] = np.sum(np.abs(psi_ML),axis=(0,1))/float(Nx*Ny)


        # AOT for ML
        for l in range(2):
            print(f"        ↳ AOT layer {l+1}/2", end='', flush=True)
            vals = np.abs(psi_ML[..., l])
            mask = vals[..., None] >= thresh  # (Ny,Nx,T,Npt)
            aot_ML[jens, :, :, l] = mask.mean(axis=(0, 1)).T  # (Npt,T)
        del psi_ML, Rm, Sm
        gc.collect()
        print(' ...done.\n', flush=True)

    del input_test
    gc.collect()

    # ------ Derived ML stats ------
    print('[6/6] Computing derived ML stats + saving ... ', end='', flush=True)

    # Ensemble average AOT
    aot_ML = np.mean(aot_ML,axis=0)

    # Frequency vectors
    f       = np.fft.fftshift(np.fft.fftfreq(test_times,d = stride*dt))
    f_train = np.fft.fftshift(np.fft.fftfreq(NT,d = stride*dt))
    f_i     = np.fft.fftshift(np.fft.fftfreq(int(test_times/Ni),d = Ni*stride*dt))

    # Save Stats
    os.makedirs(ml_results_dir, exist_ok=True)
    save_file = results_save_path(
        ml_results_dir, case_name_train, model_type, basis_type, Nepochs, dt_ml, T, Nens,
        repeat_index=repeat_index,
    )

    data_dict = {
        "pdf_xaxis": pdf_xaxis, 'test_times': test_times,
        "psi_pdf_ML": psi_pdf_ML, "psi_rpdf_ML": psi_rpdf_ML,
        "psi_rpdf_ref": psi_rpdf_ref, "psi_rpdf_coarse": psi_rpdf_coarse,
        "Psi_ML": Psi_ML, "Psi_ref": Psi_ref, "Psi_coarse": Psi_coarse_zm,
        "f": f, "f_train": f_train, "f_i": f_i,
        "spectrum_ref": spectrum_ref, "spectrum_coarse": spectrum_coarse,
        "spectrum_ML": spectrum_ML, "spectrum_train": spectrum_train,
        "R_ML": R_ML, "R_ref": R_ref, "R_coarse": R_coarse, "R_train": R_train,
        "S_ML": S_ML, "S_ref": S_ref, "S_coarse": S_coarse, "S_train": S_train,
        'dt_ml': dt_ml, 'dt_i': dt_i, 'thresh': thresh,
        'aot_ML': aot_ML, 'aot_ref': aot_ref, 'aot_coarse': aot_coarse,
        'indEE_ML': indEE_ML, 'indEE_ref': indEE_ref
    }

    hdf5storage.savemat(save_file, data_dict, format='7.3', do_compression=True)
    print(f'Ensemble Stats Saved: {save_file} \n')
    return
