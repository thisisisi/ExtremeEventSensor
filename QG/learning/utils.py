# Author: Isabella Thiel <thieli@mit.edu>
"""Shared data loading, seeding, checkpoint, and correlation helpers."""
from artifact_paths import model_save_path, results_save_path, model_save_path2, results_save_path2
import sys
import os
import yaml
import re
import random
import scipy.io as sio
import numpy as np
import tensorflow as tf


# ---------- general IO  ----------



def seed_everything(seed: int) -> int:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    return seed


# ---------- master_* files  ----------

def get_arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def load_config():
    cfg_path = get_arg("--config_file")
    if cfg_path is None or not os.path.isfile(cfg_path):
        print("ERROR: --config_file missing or not found")
        sys.exit(1)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


# ---------- train/test files  ----------

def compute_skip_and_length(mat_contents, t_startup, test_time_units, dt_ml, method = "None"):
    tstart_raw = mat_contents.get("tstart", None)
    tvec_raw = mat_contents.get("t", None)     # may be present (time vector)
    orig_data_times = int(mat_contents["data_times"])

    t_startup = t_startup/dt_ml  # convert to data frame units
    skip_idx = 0
    if tvec_raw is not None:
        # if tvec exists, treat t_startup as a time if it's not an integer index
        tvec = np.squeeze(tvec_raw)
        # If t_startup is float/time-like (not an integer frame), find nearest index
        if not float(t_startup).is_integer():
            # find first index where t >= t_startup (robust to non-exact matches)
            skip_idx = int(np.argmax(tvec >= float(t_startup)))
        else:
            # treat both as indices (make them ints)
            tstart_idx = int(np.squeeze(tstart_raw)) if tstart_raw is not None else 0
            skip_idx = max(0, int(t_startup) - tstart_idx)
    else:
        # no time vector — assume tstart is an index offset in the file
        if tstart_raw is None:
            tstart_idx = 0
        else:
            tstart_idx = int(np.squeeze(tstart_raw))
        skip_idx = max(0, int(t_startup) - tstart_idx)

    # clamp and compute available frames
    if skip_idx >= orig_data_times:
        raise ValueError(f"Requested startup index {skip_idx} >= available frames {orig_data_times}")

    available = orig_data_times - skip_idx
    # determine desired test length (in native data frames)
    if test_time_units is None:
        desired_len = available
    else:
        desired_len = int((test_time_units/dt_ml)+1) # in data frame units

    final_len = max(0, min(desired_len, available))
    if final_len < desired_len:
        print(f"Warning: requested test length {desired_len} exceeds available {available} after startup; "
              f"clamping to {final_len}.")
    if final_len == 0:
        raise ValueError(f"No frames available after skip_idx={skip_idx} (available={available})")
    if method == "reversed":
        skip_idx = max(skip_idx, orig_data_times - final_len)
        final_len = orig_data_times - skip_idx
        if final_len <= 0:
            raise ValueError(f"No frames available after reverse skip_idx={skip_idx} (final_len={final_len})")

    return skip_idx, final_len

def load_field(data_dir, case_name, dtype, nb):
    load_file = os.path.join(data_dir, case_name, f'psi{nb}_{dtype}.mat')
    mat_contents = sio.loadmat(load_file)
    if dtype == 'rnudge16.0':
        dtype = 'rnudge'
    data = mat_contents[f'psi{nb}_{dtype}']
    return data


#  ---------- train/test ensemble files  ----------


def epochs_of(path: str) -> int:
    m = re.search(r"_epochs_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def resume_or_initialize_training(model, save_path_list, Nepochs, Ne, train_again=False):
    save_path_list = [p for p in save_path_list if os.path.isfile(p)]
    if not save_path_list or train_again:
        print("No existing checkpoints found (or train_again=True). Starting fresh training.")
        Nsets = int(np.ceil(Nepochs / Ne))
        return model, Nsets, None

    save_path_list = sorted(save_path_list, key=epochs_of, reverse=True)
    best_ckpt, best_epochs, best_model = None, -1, None

    for ckpt in save_path_list:
        try:
            m = tf.keras.models.load_model(ckpt, compile=False)
            e = epochs_of(ckpt)
            print(f"Usable checkpoint: {ckpt} (epochs={e})")
            if e > best_epochs:
                best_ckpt, best_epochs, best_model = ckpt, e, m
        except Exception:
            print(f"Warning: checkpoint {ckpt} is corrupted or incompatible. Skipping.")
    if best_ckpt is None:
        print("All checkpoints corrupted. Starting fresh training.")
        Nsets = int(np.ceil(Nepochs / Ne))
        return model, Nsets, None

    if best_ckpt:
        model.set_weights(best_model.get_weights())
    done_epochs = best_epochs
    if done_epochs >= Nepochs:
        print(f"\nModel already trained for {done_epochs} epochs (>= Nepochs={Nepochs}).")
        print("Skipping training. Use a different Nepochs or set train_again=True to overwrite.\n")
        print("\n[5/5] Done.\n")
        return model, 0, best_ckpt

    Nsets = int(np.ceil((Nepochs - done_epochs) / Ne))
    print(f"Continuing training from {done_epochs} epochs for {Nsets*Ne} more epochs "
          f"to reach {done_epochs + Nsets*Ne} total.")

    return model, Nsets, best_ckpt

def tau_from_basis(basis_type: str) -> str:
    s = str(basis_type)
    return s.split('rnudge')[-1] if s.startswith('rnudge') and len(s) > 6 else ''

def load_member_field(data_dir: str, case_name: str, nb: int, member_index: int, *, basis_type: str, sig: str):
    """Strict path: psi{nb}_ens{mm}_sig{sig}_{tau}.mat"""
    mm = f"{int(member_index):02d}"
    tau = tau_from_basis(basis_type)
    path = os.path.join(data_dir, case_name, f"psi{nb}_{basis_type}_ens{mm}_sig{sig}_{tau}.mat")
    m = sio.loadmat(path)
    for k, v in m.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            return v
    raise KeyError(f"No 3D array found in {path}")


# ---------- memory reducer ----------





# ---------- misc utils ----------
def autocorr_fft(x, T):
        f = np.fft.fft(x, axis=-1)                    # FFT along time
        S = f * np.conj(f)
        c = np.fft.ifft(S, axis=-1).real              # autocorr
        c = np.fft.fftshift(c, axes=-1)               # center
        spec = np.abs(np.fft.fftshift(np.fft.fft(c, axis=-1), axes=-1)) / T
        return c, spec

# ---------- plotting files  ----------
