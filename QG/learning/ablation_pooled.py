"""Pooled-ensemble ablation: train on nudged trajectories without FiLM conditioning."""

import os
import time
import numpy as np
import scipy.io as sio
import hdf5storage
from typing import Optional
import json
import gc

print('Importing Tensorflow... ', end='', flush=True)
import tensorflow as tf
print('Done!')
print('Importing Keras... ', end='', flush=True)
print('Done!')

from loss_params_qg import loss_params_2D
from networks_qg import get_model, sampling
from utils import compute_skip_and_length, load_field, model_save_path2, results_save_path2
from utils import autocorr_fft, epochs_of, load_member_field, resume_or_initialize_training
from math import sqrt

tf.keras.mixed_precision.set_global_policy('float32')


# ------------------------------- Helpers ------------------------------


def make_dataset(
    members,
    psi_ref: np.ndarray,
    Ny: int,
    Nx: int,
    data_times: int,
    lstm_hist_points: int,
    dt_ml: float,
    dt: float,
    sub_sample: int,
    batch_size: int,
    ):
    used_features = Ny * Nx * 2
    lstm_stride = max(1, int(dt_ml / dt))

    train_times = int(np.floor((data_times - lstm_hist_points * lstm_stride) // sub_sample))
    while train_times < 10 and lstm_hist_points > 1:
        # try reducing history length to get at least 10 training samples
        lstm_hist_points -= 1
        train_times = int(np.floor((data_times - lstm_hist_points * lstm_stride) // sub_sample))
    print(f"Adjusted lstm_hist_points: {lstm_hist_points}, resulting in train_times: {train_times}")
    if train_times <= 0:
        raise ValueError("No training windows possible with given T/lstm_hist_points/dt_ml.")

    N = train_times * len(members)  # total windows across all members

    # Pre-flatten REF once
    flat_ref = np.concatenate(
        [
            psi_ref[..., 0].reshape(Ny * Nx, data_times, order="F"),
            psi_ref[..., 1].reshape(Ny * Nx, data_times, order="F"),
        ],
        axis=0,
    ).T.astype(np.float32, copy=False)

    idx_vec = (np.arange(lstm_hist_points, dtype=np.int64) * lstm_stride)

    def gen():
        # deterministic order: member-major, then time-major
        for m in members:
            flat_m = np.concatenate(
                [
                    m[..., 0].reshape(Ny * Nx, data_times, order="F"),
                    m[..., 1].reshape(Ny * Nx, data_times, order="F"),
                ],
                axis=0,
            ).T.astype(np.float32, copy=False)

            for s in range(0, train_times * sub_sample, sub_sample):
                t_idx = s + idx_vec  # (hist,)
                yield flat_m[t_idx, :], flat_ref[t_idx, :]

    output_signature = (
        tf.TensorSpec(shape=(lstm_hist_points, used_features), dtype=tf.float32),
        tf.TensorSpec(shape=(lstm_hist_points, used_features), dtype=tf.float32),
    )

    base_ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    # 90/10 split (deterministic)
    N_train = int(0.9 * N)
    N_val   = N - N_train
    windows_per_member = train_times
    train_count = int(0.9 * windows_per_member)
    member_count = len(members)

    # Enumerate the dataset so each sample has an index
    ds_enum = base_ds.enumerate()

    # Assign samples to train/val based on position within each member’s window block
    train_ds = ds_enum.filter(lambda i, _: (i % windows_per_member) < train_count)
    val_ds   = ds_enum.filter(lambda i, _: (i % windows_per_member) >= train_count)

    # Drop the enumeration index, keep only data tuples
    train_ds = train_ds.map(lambda _, d: d)
    val_ds   = val_ds.map(lambda _, d: d)

    # Final batching + prefetch
    train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE).repeat()
    val_ds   = val_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE).repeat()


    return train_ds, val_ds, N_train, N_val


# ------------------------------ pooled: Train ------------------------------

def train(
    model_type: str,
    case_name: Optional[str] = None,
    j_ens: int = 1,
    Nepochs: int = 2000,
    sub_sample: int = 20,
    lstm_hist_points: int = 100,
    dt_ml: float = 0.1,
    basis_type: str = "rnudge16.0",
    T: int = 1000,
    batch_size: int = 32,
    lr: Optional[float] = None,
    neural_nets_dir: str = "../Neural_Nets",
    ml_results_dir: str = "../ML_Results",
    data_dir: str = "../ML_Data",
    t_startup: int = 1000,
    ensemble_cfg: Optional[dict] = None,
    case_name_train: Optional[str] = None,
    **kwargs
    ):
    case_name = case_name or case_name_train
    if case_name is None:
        raise ValueError("case_name (or case_name_train) must be provided")

    ensemble_cfg = ensemble_cfg or {}
    n_extra = int(ensemble_cfg.get("n_ens", 0))
    include_baseline = bool(ensemble_cfg.get("include_baseline", True))
    n_pool = n_extra + (1 if include_baseline else 0)
    repeat_index = kwargs.get("repeat_index", None)

    print("\nS1 Training (Pooled Nudged Ensemble)\n-----------------------------------")
    print(f"Case                    : {case_name}")
    print(f"Model type              : {model_type}")
    print(f"Basis                   : {basis_type}")
    print(f"Ensemble ID (ckpt tag)  : {j_ens}")
    print(f"Pooled members (k)      : {n_pool}")

    # ---- [1/5] Load headers and reference window ----
    print("\n[1/5] Loading headers & reference window ... ", end="", flush=True)
    head = sio.loadmat(os.path.join(data_dir, case_name, "psi1_ref.mat"))
    Ny = int(head["Ny"])
    Nx = int(head["Nx"])
    delta, beta, rdrag, kd = head["delta"], head["beta"], head["rdrag"], head["kd"]
    dt = float(head["dt"])

    method = "reversed" # "standard" or "reversed" for skip computation
    skip_idx, data_times = compute_skip_and_length(head, t_startup, T, dt_ml, method=method)
    psi_ref = np.zeros((Ny, Nx, data_times, 2), dtype=np.float32)
    print(f"  • skip_idx: {skip_idx}")
    print(f"  • data_times: {data_times}")
    print(f"  • total_times ref (original): {head['psi1_ref'].shape[2]}")
    print(f"  • total_times nudged (original): {load_field(data_dir, case_name, basis_type, 1).shape[2]}")

    if method == "reversed":
        psi_ref[..., 0] = head["psi1_ref"][:, :, -data_times:]
        psi_ref[..., 1] = load_field(data_dir, case_name, "ref", 2)[
            :, :, -data_times:]
    else:
        psi_ref[..., 0] = head["psi1_ref"][:, :, skip_idx : skip_idx + data_times]
        psi_ref[..., 1] = load_field(data_dir, case_name, "ref", 2)[
            :, :, skip_idx : skip_idx + data_times]
    del head
    gc.collect()

    print("Done.")
    print(f"Grid Ny×Nx              : {Ny}×{Nx}")
    print(f"data_times (sliced)     : {data_times}")
    print(f"dt (from data)          : {dt}")

    # ---- [2/5] Load pooled inputs ----
    print("\n[2/5] Loading pooled nudged inputs ...", flush=True)
    sig = ensemble_cfg.get("sig", None)
    if sig is None:
        raise ValueError("ensemble.sig is required for member files")

    members = []

    if include_baseline:
        psi1 = load_field(data_dir, case_name, basis_type, 1)
        psi2 = load_field(data_dir, case_name, basis_type, 2)
        m0 = np.zeros((Ny, Nx, data_times, 2), dtype=np.float32)
        if method == "reversed":
            m0[..., 0] = psi1[:, :, -data_times:]
            m0[..., 1] = psi2[:, :, -data_times:]
        else:
            m0[..., 0] = psi1[:, :, skip_idx : skip_idx + data_times]
            m0[..., 1] = psi2[:, :, skip_idx : skip_idx + data_times]
        members.append(m0)
        print("  • baseline rnudge loaded")

    for k in range(1, n_extra + 1):
        psi1_k = load_member_field(
            data_dir, case_name, nb=1, member_index=k, basis_type=str(basis_type), sig=str(sig)
        )
        psi2_k = load_member_field(
            data_dir, case_name, nb=2, member_index=k, basis_type=str(basis_type), sig=str(sig)
        )
        mk = np.zeros((Ny, Nx, data_times, 2), dtype=np.float32)
        if method == "reversed":
            mk[..., 0] = psi1_k[:, :, -data_times:]
            mk[..., 1] = psi2_k[:, :, -data_times:]
        else:
            mk[..., 0] = psi1_k[:, :, skip_idx : skip_idx + data_times]
            mk[..., 1] = psi2_k[:, :, skip_idx : skip_idx + data_times]
        members.append(mk)
        print(f"  • member {k} loaded")

    print(f"Total pooled members    : {len(members)}")

    # ---- [3/5] Windowing (vectorized) ----
    print("\n[3/5] Building training windows ...", flush=True)
    lstm_stride = max(1, int(dt_ml / dt))
    total_times = data_times
    train_times = total_times - lstm_hist_points * lstm_stride
    train_times = int(np.floor(train_times // sub_sample))
    if train_times <= 0:
        raise ValueError("No training windows possible.")
    print(f"train_times per member  : {train_times}")
    print(f"LSTM Stride             : {lstm_stride}")
    print(f"sub_sample (window step): {sub_sample}")

    train_dataset, val_dataset, N_train, N_val = make_dataset(
        members=members,
        psi_ref=psi_ref,
        Ny=Ny,
        Nx=Nx,
        data_times=data_times,
        lstm_hist_points=lstm_hist_points,
        dt_ml=dt_ml,
        dt=dt,
        sub_sample=sub_sample,
        batch_size=batch_size,
    )


    used_features = Ny * Nx * 2
    output_dim = Ny * Nx * 2
    N = train_times * len(members)

    psi_ref = None
    members = None
    gc.collect()


    # ---- [4/5] Model ----
    print("\n[4/5] Building & training model ...", flush=True)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    loss_data = loss_params_2D(kd, delta, Ny, Nx, beta, rdrag)

    k_eff = max(1.0, np.log(n_pool))
    inter_dim = round(64 * sqrt(k_eff))
    latent_dim = round(60 * sqrt(k_eff))
    model = get_model(model_type, used_features, loss_data, lr=lr, inter_dim=inter_dim, latent_dim=latent_dim)

    Ne = 500
    Nsets = int(np.ceil(Nepochs / Ne))
    os.makedirs(neural_nets_dir, exist_ok=True)

    save_path_final = model_save_path2(
            neural_nets_dir,
            case_name,
            model_type,
            basis_type,
            Nepochs,
            dt_ml,
            T,
            j_ens,
            strategy="pooled",
            n_members=n_pool,
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

    # check what exists already to continue training
    save_path_list = [model_save_path2(
            neural_nets_dir,
            case_name,
            model_type,
            basis_type,
            Nepochs_j,
            dt_ml,
            T,
            j_ens,
            strategy="pooled",
            n_members=n_pool,
            repeat_index=repeat_index,
        ) for Nepochs_j in range(Ne, Nepochs+1, Ne)]

    _, Nsets, latest_ckpt = resume_or_initialize_training(
        model, save_path_list, Nepochs, Ne, train_again=train_again
    )

    done_epochs = epochs_of(latest_ckpt) if latest_ckpt is not None else 0


    steps_per_epoch  = max(1, N_train // batch_size)
    validation_steps = max(1, N_val   // batch_size)

    train_history = None
    for je in range(Nsets):
        hist_tmp = model.fit(
                train_dataset,
                epochs=Ne,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_dataset,
                validation_steps=validation_steps,
                verbose=2,
            )
        if train_history is None:
            train_history = hist_tmp
        else:
            train_history.history['loss'].extend(hist_tmp.history['loss'])
            train_history.history['val_loss'].extend(hist_tmp.history['val_loss'])

        Nepochs_j = (je + 1) * Ne + done_epochs
        ckpt = model_save_path2(
            neural_nets_dir,
            case_name,
            model_type,
            basis_type,
            Nepochs_j,
            dt_ml,
            T,
            j_ens,
            strategy="pooled",
            n_members=n_pool,
            repeat_index=repeat_index,
        )
        model.save(ckpt, save_format="h5")
        print(f"  Saved: {ckpt}", flush=True)

    # del X, Y
    # save history as history_results_save_path2.json

    hist_path = model_save_path2(
        neural_nets_dir,
        case_name,
        model_type,
        basis_type,
        Nepochs_j,
        dt_ml,
        T,
        j_ens,
        strategy="pooled",
        n_members=n_pool,
        repeat_index=repeat_index,
    )
    hist_path = hist_path.replace('.h5', '_history.json')

    merged_history = {'loss': [], 'val_loss': []}
    if isinstance(train_history, list):
        # If train_history is a list of History objects
        for h in train_history:
            merged_history['loss'].extend(h.history.get('loss', []))
            merged_history['val_loss'].extend(h.history.get('val_loss', []))
    else:
        # Single History object
        merged_history['loss'].extend(train_history.history.get('loss', []))
        merged_history['val_loss'].extend(train_history.history.get('val_loss', []))

    with open(hist_path, 'w') as f:
        json.dump(merged_history, f)
        print(f"  Saved training history: {hist_path}", flush=True)

    return


# ------------------------------ pooled: Test ------------------------------
# ---------------- pooled: Ensemble Testing on Coarse DNS (fixed PDFs) ----------------

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
    ensemble_cfg: Optional[dict] = None,
    method: Optional[str] = None,
    **kwargs):
    """
    pooled test: identical to deterministic stats; loads pooled checkpoint.
    (PDF computation fixed: no index-name collisions; consistent shapes & bin clamping.)
    """

    case_name_train = case_name_train or case_name
    ensemble_cfg = ensemble_cfg or {}
    n_extra = int(ensemble_cfg.get('n_ens', 0))
    include_baseline = bool(ensemble_cfg.get('include_baseline', True))
    n_pool = n_extra + (1 if include_baseline else 0)
    repeat_index = kwargs.get("repeat_index", None)

    print('\nS1 Testing (Coarse DNS)\n-----------------------')
    print(f'Case                    : {case_name}')
    print(f'Model type              : {model_type}')
    print(f'Basis                   : {basis_type}')
    print(f'Nens (models to eval)   : {Nens}')
    print(f'Pooled members (k)      : {n_pool}')

    # [0/6] Check if already done
    test_again = bool(kwargs.get("test_again", False))

    save_file = results_save_path2(
        ml_results_dir, case_name_train, model_type, basis_type,
        Nepochs, dt_ml, T, Nens, strategy="pooled", n_members=n_pool,
        repeat_index=repeat_index
    )

    if os.path.isfile(save_file) and not test_again:
        print(f"\nTest results already exist: {save_file}\nUse test_again=True to re-test.\n")
        return

    # [1/6] Reference headers & window
    print('[1/6] Importing Reference data... ', end='', flush=True)
    head = sio.loadmat(os.path.join(data_dir, case_name, 'psi1_ref.mat'))
    Ny = int(head["Ny"])//1; Nx = int(head["Nx"])//1
    delta = head["delta"]; beta = head["beta"]
    rdrag = head["rdrag"]; kd = head["kd"]
    dt = float(head["dt"])
    #skip_idx, total_times = compute_skip_and_length(head, t_startup, test_time_units, dt_ml)
    skip_idx, total_times = compute_skip_and_length(head, t_startup, test_time_units, dt_ml)


    psi_ref = np.zeros((Ny, Nx, total_times, 2), dtype=np.float32)
    print(f"  • Grid Ny×Nx: {Ny}×{Nx}")
    print(f"  • dt (from data): {dt}")
    print(f"  • total_times (sliced): {total_times}")
    print(f"  • skip_idx: {skip_idx}")


    psi_ref[:, :, :, 0] = head["psi1_ref"][:, :, skip_idx: skip_idx + total_times]
    ref2 = load_field(data_dir, case_name, "ref", 2)
    psi_ref[:, :, :, 1] = ref2[:, :, skip_idx: skip_idx + total_times]
    del ref2, head
    print('Done!\n')

    # [2/6] Coarse fields
    print('[2/6] Importing coarse fields... ', end='', flush=True)
    c1 = load_field(data_dir, case_name, 'coarse', 1)
    c2 = load_field(data_dir, case_name, 'coarse', 2)
    psi_coarse = np.zeros((Ny, Nx, total_times, 2), dtype=np.float32)
    psi_coarse[:, :, :, 0] = c1[:, :, skip_idx: skip_idx + total_times]
    psi_coarse[:, :, :, 1] = c2[:, :, skip_idx: skip_idx + total_times]
    del c1, c2
    gc.collect()
    print('Done!\n')

    # [3/6] Build test input (stride)
    print('[3/6] Building test input... ', end='', flush=True)
    used_features = Ny * Nx * 2
    inp = np.zeros((1, total_times, used_features), dtype=np.float32)
    inp[0, :, 0:Nx*Ny]       = np.reshape(np.transpose(psi_coarse[:, :, :, 0], (2,0,1)),
                                          (1, total_times, Nx*Ny), order='F')
    inp[0, :, Nx*Ny:2*Nx*Ny] = np.reshape(np.transpose(psi_coarse[:, :, :, 1], (2,0,1)),
                                          (1, total_times, Nx*Ny), order='F')
    stride = max(1, int((dt_ml/float(dt))))
    inp = inp[0, 0:-1:stride, :]
    inp = np.reshape(inp, (1, inp.shape[0], inp.shape[1]))
    test_times = int(inp.shape[1])
    print('Data Reshape Complete!\n')

    # [4/6] Stats scaffolding (identical to deterministic)
    print('[4/6] Preparing statistics ... ', end='', flush=True)
    t0 = time.time()
    Np = 2001; pleft, pright = -4.0, 4.0
    pdf_xaxis = np.linspace(pleft, pright, Np); dp = (pright - pleft) / float(Np - 1)

    Ni = 100
    dt_i = dt_ml * Ni
    NT = int(T / dt_ml)

    Rr = np.zeros((Ny, Nx, test_times, 2), dtype=np.float32)
    Rc = np.zeros((Ny, Nx, test_times, 2), dtype=np.float32)
    Rt = np.zeros((Ny, Nx, NT, 2), dtype=np.float32)
    Sr = np.zeros((Ny, Nx, test_times, 2), dtype=np.float32)
    Sc = np.zeros((Ny, Nx, test_times, 2), dtype=np.float32)
    St = np.zeros((Ny, Nx, NT, 2), dtype=np.float32)

    R_ML = np.zeros((Nens, Ny, Nx, int(test_times/Ni), 2), dtype=np.float32)
    R_ref = np.zeros((Ny, Nx, int(test_times/Ni), 2), dtype=np.float32)
    R_coarse = np.zeros((Ny, Nx, int(test_times/Ni), 2), dtype=np.float32)
    S_ML = np.zeros((Nens, Ny, Nx, int(test_times/Ni), 2), dtype=np.float32)
    S_ref = np.zeros((Ny, Nx, int(test_times/Ni), 2), dtype=np.float32)
    S_coarse = np.zeros((Ny, Nx, int(test_times/Ni), 2), dtype=np.float32)

    spectrum_ML = np.zeros((Nens, test_times, 2), dtype=np.float32)
    Psi_ML = np.zeros((Nens, Ny, test_times, 2), dtype=np.float32)
    Psi_ref = np.zeros((Ny, test_times, 2), dtype=np.float32)
    Psi_coarse_zm = np.zeros((Ny, test_times, 2), dtype=np.float32)
    indEE_ML = np.zeros((Nens, test_times, 2), dtype=np.float32)
    indEE_ref = np.zeros((test_times, 2), dtype=np.float32)

    psi_ref_s = psi_ref[:, :, 0:-1:stride, :]
    psi_coarse_s = psi_coarse[:, :, 0:-1:stride, :]
    print('Done!\n')

    # reference statistics
    # keep shapes consistent with your deterministic function:
    psi_rpdf_ref    = np.zeros((Nx,Ny,Np,2),dtype=np.float32)   # (Nx,Ny,…) on disk
    psi_rpdf_coarse = np.zeros((Nx,Ny,Np,2),dtype=np.float32)

    print('      Computing reference statistics ...')
    idx = np.arange(0, Rr.shape[2] - 1, Ni, dtype=int)
    nsave = min(R_ref.shape[2], idx.size)
    bins = np.linspace(pleft, pright, Np+1)
    for l in range(2):
        print(f"        ↳ layer {l+1}/2: max|ref|={np.max(np.abs(psi_ref_s[:, :, :, l])):.5f}, "
              f"max={np.max(psi_ref_s[:, :, :, l]):.5f}")
        Rr[..., l], Sr[..., l] = autocorr_fft(psi_ref_s[..., l], test_times)
        Rc[..., l], Sc[..., l] = autocorr_fft(psi_coarse_s[..., l], test_times)
        Rt[..., l], St[..., l] = autocorr_fft(psi_ref_s[..., :NT, l], NT)
        for ix in range(Nx):
            for iy in range(Ny):
                # Rr[iy, ix, :, l] = scs.correlate(psi_ref_s[iy, ix, :, l],  psi_ref_s[iy, ix, :, l],  mode='same', method='auto')
                # Rc[iy, ix, :, l] = scs.correlate(psi_coarse_s[iy, ix, :, l], psi_coarse_s[iy, ix, :, l], mode='same', method='auto')
                # Rt[iy, ix, :, l] = scs.correlate(psi_ref_s[iy, ix, 0:NT, l], psi_ref_s[iy, ix, 0:NT, l], mode='same', method='auto')

                # Sr[iy, ix, :, l] = np.abs(np.fft.fftshift(np.fft.fft(Rr[iy, ix, :, l]))) / test_times
                # Sc[iy, ix, :, l] = np.abs(np.fft.fftshift(np.fft.fft(Rc[iy, ix, :, l]))) / test_times
                # St[iy, ix, :, l] = np.abs(np.fft.fftshift(np.fft.fft(Rt[iy, ix, :, l]))) / NT

                # decimated saves

                if nsave > 0:
                    R_ref[iy, ix, 0:nsave, l]    = Rr[iy, ix, idx[:nsave], l]
                    R_coarse[iy, ix, 0:nsave, l] = Rc[iy, ix, idx[:nsave], l]
                    S_ref[iy, ix, 0:nsave, l]    = Sr[iy, ix, idx[:nsave], l]
                    S_coarse[iy, ix, 0:nsave, l] = Sc[iy, ix, idx[:nsave], l]

                # PDFs (reference & coarse)
                vals_ref = psi_ref_s[iy, ix, :-1, l]
                vals_coarse = psi_coarse_s[iy, ix, :-1, l]

                hist_ref, _ = np.histogram(vals_ref, bins=bins)
                hist_coarse, _ = np.histogram(vals_coarse, bins=bins)

                # broadcast back into memmaps if needed
                psi_rpdf_ref[iy, ix, :, l] = hist_ref
                psi_rpdf_coarse[iy, ix, :, l] = hist_coarse
                # for tt in range(test_times-1):
                #     v_ref = psi_ref_s[iy, ix, tt, l]
                #     pp = int((v_ref-pleft)/dp); pp = 0 if pp < 0 else (Np-1 if pp >= Np else pp)
                #     psi_rpdf_ref[ix, iy, pp, l] += 1  # (Nx,Ny,…) ordering

                #     v_crs = psi_coarse_s[iy, ix, tt, l]
                #     pp = int((v_crs-pleft)/dp); pp = 0 if pp < 0 else (Np-1 if pp >= Np else pp)
                #     psi_rpdf_coarse[ix, iy, pp, l] += 1

    R_train = 1 * Rt
    S_train = 1 * St
    spectrum_ref    = np.mean(Sr, axis=(0, 1))
    spectrum_coarse = np.mean(Sc, axis=(0, 1))
    spectrum_train  = np.mean(St, axis=(0, 1))
    Psi_ref        = np.sum(psi_ref_s,   axis=1) / float(Nx)
    Psi_coarse_zm  = np.sum(psi_coarse_s, axis=1) / float(Nx)
    indEE_ref      = np.sum(np.abs(psi_ref_s), axis=(0, 1)) / float(Nx * Ny)

    # AOT prep
    Npt = 25; thresh = np.linspace(0.0, 4.0, Npt)
    aot_ref    = np.zeros((Npt, test_times, 2), dtype=np.float32)
    aot_coarse = np.zeros((Npt, test_times, 2), dtype=np.float32)
    aot_ML     = np.zeros((Nens, Npt, test_times, 2), dtype=np.float32)
    for l in range(2):
        print(f"        ↳ AOT layer {l+1}/2 ...", end='', flush=True)
        vals_ref = np.abs(psi_ref_s[..., l])  # (Ny,Nx,T)
        vals_coarse = np.abs(psi_coarse_s[..., l])

        mask_ref = vals_ref[..., None] >= thresh  # (Ny,Nx,T,Npt)
        mask_coarse = vals_coarse[..., None] >= thresh

        aot_ref[..., l] = mask_ref.mean(axis=(0,1)).T    # (Npt,T)
        aot_coarse[..., l] = mask_coarse.mean(axis=(0,1)).T
        print(f" done.", flush=True)
    # for l in range(2):
    #     for tt in range(test_times - 1):
    #         for nt in range(Npt):
    #             mask = np.abs(psi_ref_s[:, :, tt, l]) >= thresh[nt]
    #             aot_ref[nt, tt, l] = np.sum(mask) / float(Nx * Ny)
    #             mask = np.abs(psi_coarse_s[:, :, tt, l]) >= thresh[nt]
    #             aot_coarse[nt, tt, l] = np.sum(mask) / float(Nx * Ny)

    del psi_ref_s, psi_coarse_s, psi_ref, psi_coarse, Rr, Rc, Rt, Sr, Sc, St
    gc.collect()
    print("      Reference stats done. ({:.2f}s)\n".format(time.time() - t0))

    # [5/6] Predictions
    print('[5/6] ML predictions', flush=True)
    psi_pdf_ML  = np.zeros((Nens,Np,2),dtype=np.float32)    # (Nens, Np, ...) on disk
    # keep saved shape consistent with deterministic path (Nens, Nx, Ny, ...):
    psi_rpdf_ML = np.zeros((Nens,Nx,Ny,Np,2),dtype=np.float32)

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    _ = loss_params_2D(kd, delta, Ny, Nx, beta, rdrag)  # keeps deps consistent


    for ens in range(Nens):
        print(f'     ...realization #{ens+1}', end='', flush=True)
        ckpt = model_save_path2(
            neural_nets_dir, case_name_train, model_type, basis_type,
            Nepochs, dt_ml, T, ens+1, strategy="pooled", n_members=n_pool,
            repeat_index=repeat_index
        )
        #ckpt_pattern = ckpt.rsplit('_', 1)[0] + '_*.h5'
        files = ckpt #glob.glob(ckpt_pattern)
        if not files:
            raise FileNotFoundError(f"Checkpoint not found: {files} or {ckpt}")

        if model_type.lower() == 'storn':
            model = tf.keras.models.load_model(files, compile=False,
                                   custom_objects={"sampling": sampling})
        else:
            model = tf.keras.models.load_model(files, compile=False)

        out = model.predict(inp, verbose=2)
        if isinstance(out, (list, tuple)):
            out = out[0]
        output_test = np.asarray(out, dtype=np.float32)

        psi_ML = np.zeros((Ny, Nx, test_times, 2), dtype=float)
        psi_ML[:, :, :, 0] = np.transpose(np.reshape(output_test[0, :, 0:Nx*Ny],
                                                     (test_times, Nx, Ny), order='F'), (1,2,0))
        psi_ML[:, :, :, 1] = np.transpose(np.reshape(output_test[0, :, Nx*Ny:2*Nx*Ny],
                                                     (test_times, Nx, Ny), order='F'), (1,2,0))

        Rm = np.zeros((Ny, Nx, test_times, 2), dtype=float)
        Sm = np.zeros((Ny, Nx, test_times, 2), dtype=float)

        idx = np.arange(0, Rm.shape[2]-1, Ni, dtype=int)
        nsave = min(R_ML.shape[3], idx.size)

        print()
        for l in range(2):
            print(f"        ↳ layer {l+1}/2: max|ML|={np.max(np.abs(psi_ML[..., l])):.5f}, "
                    f"max={np.max(psi_ML[..., l]):.5f}", flush=True)
            Rm[..., l], Sm[..., l] = autocorr_fft(psi_ML[..., l], test_times)

        # for l in range(2):
        #     for ix in range(Nx):
        #         for iy in range(Ny):
        #             Rm[iy, ix, :, l] = scs.correlate(psi_ML[iy, ix, :, l], psi_ML[iy, ix, :, l],
        #                                              mode='same', method='auto')
        #         Sm[:, ix, :, l] = np.abs(np.fft.fftshift(np.fft.fft(Rm[:, ix, :, l], axis=1), axes=1)) / test_times

            # decimated saves for R/S

            if nsave > 0:
                R_ML[ens, :, :, 0:nsave, l] = Rm[:, :, idx[:nsave], l]
                S_ML[ens, :, :, 0:nsave, l] = Sm[:, :, idx[:nsave], l]

            # PDFs (ML)
            for ix in range(Nx):
                for iy in range(Ny):
                    vals = psi_ML[iy, ix, :-1, l]
                    hist, _ = np.histogram(vals, bins=bins)
                    psi_rpdf_ML[ens, ix, iy, :, l] = hist
                    psi_pdf_ML[ens, :, l] += hist
                    # for tt in range(test_times-1):
                    #     v = psi_ML[iy, ix, tt, l]
                    #     pp = int((v-pleft)/dp); pp = 0 if pp < 0 else (Np-1 if pp >= Np else pp)
                    #     psi_pdf_ML[ens, pp, l] += 1                # <- ensemble slice = ens
                    #     psi_rpdf_ML[ens, ix, iy, pp, l] += 1        # (Nens, Nx, Ny, ...)

        spectrum_ML[ens, :, :] = np.mean(Sm, axis=(0, 1))
        Psi_ML[ens, :, :, :]   = np.sum(psi_ML, axis=1) / float(Nx)
        indEE_ML[ens, :]       = np.sum(np.abs(psi_ML), axis=(0, 1)) / float(Nx * Ny)

        # AOT for ML
        for l in range(2):
            print(f"        ↳ AOT layer {l+1}/2", end='', flush=True)
            vals = np.abs(psi_ML[..., l])
            mask = vals[..., None] >= thresh  # (Ny,Nx,T,Npt)
            aot_ML[ens, :, :, l] = mask.mean(axis=(0,1)).T  # (Npt,T)
            print("...done.", flush=True)
        # for l in range(2):
        #     for tt in range(test_times - 1):
        #         for nt in range(Npt):
        #             aot_ML[ens, nt, tt, l] = (
        #                 np.sum(np.abs(psi_ML[:, :, tt, l]) >= thresh[nt]) / float(Nx * Ny)
        #             )

        del psi_ML, Rm, Sm
        print(' ...done.')

    del inp
    gc.collect()

    # [6/6] Save
    print('\n[6/6] Saving results ... ', end='', flush=True)
    aot_ML_mean = np.mean(aot_ML, axis=0)
    f       = np.fft.fftshift(np.fft.fftfreq(test_times, d=stride * dt))
    f_train = np.fft.fftshift(np.fft.fftfreq(NT,         d=stride * dt))
    f_i     = np.fft.fftshift(np.fft.fftfreq(int(test_times/Ni), d=Ni * stride * dt))

    os.makedirs(ml_results_dir, exist_ok=True)
    save_file = results_save_path2(
        ml_results_dir, case_name_train, model_type, basis_type,
        Nepochs, dt_ml, T, Nens, strategy="pooled", n_members=n_pool,
        repeat_index=repeat_index
    )
    hdf5storage.savemat(
        save_file,
        {
            "pdf_xaxis": pdf_xaxis, 'test_times': test_times,
            "psi_pdf_ML": psi_pdf_ML,
            "psi_rpdf_ML": psi_rpdf_ML,          # (Nens, Nx, Ny, Np, 2)
            "psi_rpdf_ref": psi_rpdf_ref,        # (Nx, Ny, Np, 2)
            "psi_rpdf_coarse": psi_rpdf_coarse,  # (Nx, Ny, Np, 2)
            "Psi_ML": Psi_ML, "Psi_ref": Psi_ref, "Psi_coarse": Psi_coarse_zm,
            "f": f, "f_train": f_train, "f_i": f_i,
            "spectrum_ref": spectrum_ref, "spectrum_coarse": spectrum_coarse,
            "spectrum_ML": spectrum_ML, "spectrum_train": spectrum_train,
            "R_ML": R_ML, "R_ref": R_ref, "R_coarse": R_coarse, "R_train": R_train,
            "S_ML": S_ML, "S_ref": S_ref, "S_coarse": S_coarse, "S_train": S_train,
            'dt_ml': dt_ml, 'dt_i': dt_i, 'thresh': thresh,
            'aot_ML': aot_ML_mean, 'aot_ref': aot_ref, 'aot_coarse': aot_coarse,
            'indEE_ML': indEE_ML, 'indEE_ref': indEE_ref
        },
        format='7.3', do_compression=True
    )
    print(f"Saved: {save_file}\n", flush=True)
    return
