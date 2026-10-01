#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Isabella Thiel <thieli@mit.edu>
"""
Baseline: covariance-eigenvector FiLM conditioning.

At each time t, concatenate both QG layers into x_t^(m) in R^D,
D = 2*Ny*Nx, subtract the ensemble mean, and form the unbiased centered
covariance

    Sigma_hat_t = 1/(M-1) sum_m q_t^(m) q_t^(m)^T.

The top K <= min(n_eigcov_modes, M-1, D) unit eigenvectors are concatenated
mode-major and supplied to FiLM, so cond_in has shape (B,T,K*D).  At test time
the same dimensional context is estimated strictly causally from the current
and previous coarse states; no reference or test-ensemble values are used.

"""

import os
import json
import time
import fcntl
from typing import Optional
from experiments import experiment_name

import numpy as np
import scipy.io as sio
import hdf5storage
import tensorflow as tf
from tensorflow.keras import layers
import gc

from loss_params_qg import loss_params_2D, loss_params_2DV
from loss_funs_qg import param_loss_2D, vae_loss_2D
from networks_qg import keras
from utils import (
    autocorr_fft,
    compute_skip_and_length,
    epochs_of,
    load_field,
    load_member_field,
    model_save_path2,
    results_save_path2,
    resume_or_initialize_training,
    seed_everything,
)

tf.keras.mixed_precision.set_global_policy('float32')
tf.get_logger().setLevel("ERROR")


# ----------------------------- small helpers -----------------------------
# ---- Serializable micro-layers (replace problematic Lambdas) ----
@tf.keras.utils.register_keras_serializable(package="custom")
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mu, sig = inputs
        eps = tf.random.normal(tf.shape(mu))
        return mu + sig * eps

@tf.keras.utils.register_keras_serializable(package="custom")
class OnePlus(tf.keras.layers.Layer):
    def call(self, x):
        return x + 1.0


def build_film_lstm(used_features: int,
                    out_features: int,
                    cond_dim: int,
                    loss_data,
                    lr: float = 1e-4,
                    conditioning_mode: str = "film") -> keras.Model:
    """
    LSTM trunk with FiLM conditioning. Keras-native ops only (no raw tf.* on KerasTensors).
    Inputs:
      field_in: (B, T, used_features)
      cond_in : (B, T, cond_dim)
    Output head 'field': (B, T, out_features)
    """
    x = layers.Input(shape=(None, used_features), name="field_in")
    z = layers.Input(shape=(None, cond_dim),     name="cond_in")

    trunk_input = (layers.Concatenate(name="context_concat")([x, z])
                   if conditioning_mode == "concat" else x)
    h = layers.Dense(60, activation='tanh', name='pre_lstm')(trunk_input)
    h = layers.LSTM(60, activation='tanh',
                    recurrent_activation='hard_sigmoid',
                    return_sequences=True, name='lstm')(h)

    # FiLM from conditioning stream (per time step)
    u = layers.Dense(32, activation='tanh', name='film_mlp')(z)
    g = layers.Dense(60, activation='tanh', name='gamma')(u)  # scale-1
    b = layers.Dense(60, activation='tanh', name='beta')(u)   # shift

    # Avoid tf.ones_like on a KerasTensor; do (g + 1) via Lambda/Keras ops
    scale = OnePlus(name="one_plus_gamma")(g)
    #scale = layers.Lambda(lambda t: t + 1.0, name="one_plus_gamma")(g)
    if conditioning_mode == "film":
        h = layers.Multiply(name="film_mul")([h, scale])
        h = layers.Add(name="film_add")([h, b])

    y = layers.Dense(out_features, activation='linear', name='field')(h)


    model = tf.keras.Model(inputs={"field_in": x, "cond_in": z}, outputs={"field": y})
    opt   = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss={"field": param_loss_2D(alpha=loss_data)}, optimizer=opt)
    model.summary()
    return model



def _make_window_indices(T_field: int, H: int, stride: int, step: int, time_start: int = 0) -> np.ndarray:
    """Indices for windowed sampling: t = start + p*stride, p=0..H-1; start increments by 'step'."""
    if T_field <= 0 or H <= 0:
        return np.zeros((0, H), dtype=np.int32)

    start_max = (T_field - 1) - (H - 1) * stride
    if start_max < time_start:
        return np.zeros((0, H), dtype=np.int32)

    starts = np.arange(time_start, start_max + 1, max(1, int(step)), dtype=np.int32)

    offs   = np.arange(H, dtype=np.int32) * max(1, int(stride))

    return starts[:, None] + offs[None, :]


def _fix_mode_signs(modes: np.ndarray, previous: Optional[np.ndarray]) -> np.ndarray:
    """Remove the arbitrary eigenvector sign without changing its direction."""
    modes = modes.copy()
    for k in range(modes.shape[1]):
        if previous is not None and np.dot(previous[:, k], modes[:, k]) < 0.0:
            modes[:, k] *= -1.0
        elif previous is None:
            pivot = int(np.argmax(np.abs(modes[:, k])))
            if modes[pivot, k] < 0.0:
                modes[:, k] *= -1.0
    return modes


def _top_modes_from_samples(samples: np.ndarray, n_modes: int,
                            previous: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Top covariance eigenvectors from samples (M,D), via the M-by-M Gram matrix.

    This is algebraically equivalent to eigh(q.T @ q/(M-1)), but avoids a
    D-by-D allocation. Returns (D,K), ordered by decreasing eigenvalue.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"samples must have shape (M,D), got {x.shape}")
    m, d = x.shape
    k = min(int(n_modes), max(0, m - 1), d)
    if k < 1:
        raise ValueError("At least two samples and one requested mode are required")
    q = x - x.mean(axis=0, keepdims=True)
    gram = (q @ q.T) / float(m - 1)
    values, left = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, left = values[order[:k]], left[:, order[:k]]
    modes = q.T @ left
    modes /= np.sqrt(np.maximum((m - 1) * values, 1e-30))[None, :]
    modes /= np.maximum(np.linalg.norm(modes, axis=0, keepdims=True), 1e-14)
    return _fix_mode_signs(modes, previous)


def _build_context_from_members(mem1, mem2, Ny=None, Nx=None, total_times=None,
                                chunk_len=256, n_modes=None) -> np.ndarray:
    """
    Centered full-field covariance eigenvectors.

    Input member arrays have shape (Ny,Nx,T). Output is (T,K*D), where
    D=2*Ny*Nx and K=min(n_modes,M-1,D). Spatial flattening is Fortran-order,
    exactly matching field_in.
    """
    if len(mem1) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # Infer shapes if not given
    if Ny is None or Nx is None or total_times is None:
        Ny, Nx, total_times = mem1[0].shape

    if len(mem1) != len(mem2):
        raise ValueError("Both layers must contain the same number of members")
    m, d = len(mem1), 2 * Ny * Nx
    k = min(m - 1 if n_modes is None else int(n_modes), m - 1, d)
    if k < 1:
        raise ValueError("baseline needs M>=2 ensemble trajectories (covariance rank M-1)")
    Z = np.empty((total_times, k * d), dtype=np.float32)
    previous = None
    for t in range(total_times):
        samples = np.empty((m, d), dtype=np.float64)
        for j, (a1, a2) in enumerate(zip(mem1, mem2)):
            samples[j, :Ny*Nx] = a1[:, :, t].reshape(-1, order="F")
            samples[j, Ny*Nx:] = a2[:, :, t].reshape(-1, order="F")
        modes = _top_modes_from_samples(samples, k, previous)
        Z[t] = modes.T.reshape(-1).astype(np.float32)
        previous = modes
    return Z


def _context_from_coarse_causal(psi_coarse: np.ndarray, n_modes: int,
                                window: int = 32) -> np.ndarray:
    """Leakage-proof proxy using only coarse states at indices <= t."""
    y = np.asarray(psi_coarse)
    if y.ndim != 4 or y.shape[-1] != 2:
        raise ValueError(f"psi_coarse must be (Ny,Nx,T,2), got {y.shape}")
    ny, nx, nt, _ = y.shape
    d = 2 * ny * nx
    k = min(int(n_modes), max(1, int(window) - 1), d)
    states = np.empty((nt, d), dtype=np.float64)
    for t in range(nt):
        states[t, :ny*nx] = y[:, :, t, 0].reshape(-1, order="F")
        states[t, ny*nx:] = y[:, :, t, 1].reshape(-1, order="F")
    Z = np.zeros((nt, k * d), dtype=np.float32)
    previous = None
    for t in range(nt):
        stop = t + 1
        start = max(0, stop - max(2, int(window)))
        block = states[start:stop]
        if block.shape[0] < 2:
            continue
        modes = _top_modes_from_samples(block, k, previous)
        available = modes.shape[1]
        Z[t, :available*d] = modes.T.reshape(-1)
        previous = modes if available == k else None
    return Z


def _rss_gib() -> float:
    """Current resident set size on Linux, for test-memory diagnostics."""
    try:
        with open("/proc/self/status", "r") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return float("nan")


CONTROL_STRATEGIES = {"random_context", "coarse_statistics", "ensemble_mean", "ensemble_variance", "covariance_concat", "mean_spread_concat"}


def _validate_control_design(strategy: str, T: int, sig, n_members: int) -> None:
    """Prevent accidental post-hoc expansion beyond the preregistered control."""
    strategy = experiment_name(strategy)
    if strategy not in CONTROL_STRATEGIES:
        return
    if int(T) != 50 or not np.isclose(float(sig), 0.01) or int(n_members) != 2:
        raise ValueError(
            f"{strategy} is restricted to T_train=50, sigma=0.01, N_add=2; "
            f"received T={T}, sigma={sig}, N_add={n_members}"
        )


def _control_cond_dim(strategy: str, state_dim: int, n_members: int,
                      n_modes_requested: int) -> int:
    strategy = experiment_name(strategy)
    if strategy in {"coarse_statistics", "mean_spread_concat"}:
        return 4
    if strategy in {"random_context", "ensemble_mean", "ensemble_variance"}:
        return state_dim
    if strategy in {"baseline", "covariance_concat"}:
        return min(n_modes_requested, n_members - 1, state_dim) * state_dim
    raise ValueError(f"Unsupported covariance/control strategy: {strategy}")


def _random_unit_context(total_times: int, state_dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    z = rng.normal(size=(total_times, state_dim)).astype(np.float32)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-7)
    return z


def _flatten_two_layers(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    ny, nx, nt = a1.shape
    out = np.empty((nt, 2 * ny * nx), dtype=np.float32)
    for t in range(nt):
        out[t, :ny*nx] = a1[:, :, t].reshape(-1, order="F")
        out[t, ny*nx:] = a2[:, :, t].reshape(-1, order="F")
    return out


def _scalar_context_from_members(mem1, mem2) -> np.ndarray:
    """
    film's four scalar ensemble channels.

    For each layer and time, first take each member's spatial mean, then
    compute the ensemble mean and unbiased ensemble standard deviation. The
    returned column order is [mean_1, std_1, mean_2, std_2].
    """
    if len(mem1) < 2 or len(mem1) != len(mem2):
        raise ValueError("mean_spread_concat needs at least two matched ensemble members")
    nt = mem1[0].shape[2]
    z = np.empty((nt, 4), dtype=np.float32)
    for layer, members in enumerate((mem1, mem2)):
        member_means = np.stack(
            [np.asarray(field).mean(axis=(0, 1), dtype=np.float64) for field in members],
            axis=0,
        )
        z[:, 2 * layer] = member_means.mean(axis=0).astype(np.float32)
        z[:, 2 * layer + 1] = (
            member_means.std(axis=0, ddof=1) + 1e-8
        ).astype(np.float32)
    return z


def _scalar_context_from_coarse(psi_coarse: np.ndarray) -> np.ndarray:
    """
    Leakage-proof film-style proxy from the current coarse state only.

    There is no test ensemble, so each layer's instantaneous spatial mean and
    spatial standard deviation provide the four dimension-compatible proxy
    channels. No reference state, future state, or full-window normalization
    is used.
    """
    y = np.asarray(psi_coarse)
    if y.ndim != 4 or y.shape[-1] != 2:
        raise ValueError(f"psi_coarse must be (Ny,Nx,T,2), got {y.shape}")
    nt = y.shape[2]
    z = np.empty((nt, 4), dtype=np.float32)
    for layer in range(2):
        field = y[..., layer]
        z[:, 2 * layer] = field.mean(axis=(0, 1), dtype=np.float64)
        z[:, 2 * layer + 1] = field.std(axis=(0, 1), dtype=np.float64)
    return z


def _build_control_train_context(strategy: str, mem1, mem2, p1, p2,
                                 n_modes: int, seed: int) -> np.ndarray:
    strategy = experiment_name(strategy)
    ny, nx, nt = p1.shape
    state_dim = 2 * ny * nx
    if strategy == "random_context":
        return _random_unit_context(nt, state_dim, seed + 1010)
    if strategy == "coarse_statistics":
        z = np.empty((nt, 4), dtype=np.float32)
        z[:, 0] = p1.mean(axis=(0, 1))
        z[:, 1] = p1.std(axis=(0, 1))
        z[:, 2] = p2.mean(axis=(0, 1))
        z[:, 3] = p2.std(axis=(0, 1))
        return z
    if strategy in {"ensemble_mean", "ensemble_variance"}:
        stack = np.empty((len(mem1), nt, state_dim), dtype=np.float32)
        for m, (a1, a2) in enumerate(zip(mem1, mem2)):
            stack[m] = _flatten_two_layers(a1, a2)
        if strategy == "ensemble_mean":
            return stack.mean(axis=0, dtype=np.float64).astype(np.float32)
        return stack.var(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
    if strategy == "mean_spread_concat":
        return _scalar_context_from_members(mem1, mem2)
    if strategy in {"baseline", "covariance_concat"}:
        return _build_context_from_members(
            mem1, mem2, Ny=ny, Nx=nx, total_times=nt, n_modes=n_modes
        )
    raise ValueError(f"Unsupported strategy {strategy}")


def _causal_pointwise_variance(psi_coarse: np.ndarray, window: int) -> np.ndarray:
    """Exact trailing-window variance in O(T*D) time and O(window*D) workspace."""
    ny, nx, nt, _ = psi_coarse.shape
    d = 2 * ny * nx
    w = max(2, int(window))
    out = np.zeros((nt, d), dtype=np.float32)
    ring = np.zeros((w, d), dtype=np.float32)
    running = np.zeros(d, dtype=np.float64)
    running2 = np.zeros(d, dtype=np.float64)
    for t in range(nt):
        x = np.empty(d, dtype=np.float32)
        x[:ny*nx] = psi_coarse[:, :, t, 0].reshape(-1, order="F")
        x[ny*nx:] = psi_coarse[:, :, t, 1].reshape(-1, order="F")
        slot = t % w
        if t >= w:
            old = ring[slot]
            running -= old
            running2 -= old.astype(np.float64) ** 2
        ring[slot] = x
        running += x
        running2 += x.astype(np.float64) ** 2
        count = min(t + 1, w)
        if count > 1:
            out[t] = np.maximum(
                (running2 - running * running / count) / (count - 1), 0.0
            )
    return out


def _build_control_test_context(strategy: str, psi_coarse: np.ndarray,
                                cond_dim: int, n_modes: int, window: int,
                                seed: int) -> np.ndarray:
    strategy = experiment_name(strategy)
    ny, nx, nt, _ = psi_coarse.shape
    d = 2 * ny * nx
    if strategy == "random_context":
        return _random_unit_context(nt, d, seed + 2010)
    if strategy == "coarse_statistics":
        z = np.empty((nt, 4), dtype=np.float32)
        for layer in range(2):
            z[:, 2*layer] = psi_coarse[..., layer].mean(axis=(0, 1))
            z[:, 2*layer+1] = psi_coarse[..., layer].std(axis=(0, 1))
        return z
    if strategy == "ensemble_mean":
        return _flatten_two_layers(psi_coarse[..., 0], psi_coarse[..., 1])
    if strategy == "ensemble_variance":
        return _causal_pointwise_variance(psi_coarse, window)
    if strategy == "mean_spread_concat":
        return _scalar_context_from_coarse(psi_coarse)
    if strategy in {"baseline", "covariance_concat"}:
        return _context_from_coarse_causal(psi_coarse, n_modes=n_modes, window=window)
    raise ValueError(f"Unsupported strategy {strategy}")


def make_conditioned_dataset(src_train, src_ref, Z_full,
                    Ny, Nx, used_features,
                    lstm_hist_points, idx, batch_size, train_split=0.9):
    """
    src_train: tuple (p1, p2) arrays (Ny, Nx, T)
    src_ref  : tuple (r1, r2) arrays (Ny, Nx, T)
    """
    p1, p2 = src_train
    r1, r2 = src_ref
    Dz = Z_full.shape[1]

    N = idx.shape[0]
    Ntrain = int(N* train_split)

    def gen():
        for n in range(idx.shape[0]):
            X = np.zeros((lstm_hist_points, used_features), dtype=np.float32)
            Y = np.zeros((lstm_hist_points, used_features), dtype=np.float32)
            Z = np.zeros((lstm_hist_points, Dz), dtype=np.float32)
            for p in range(lstm_hist_points):
                t = int(idx[n, p])
                X[p, 0:Ny*Nx] = p1[:, :, t].reshape(-1, order='F')
                X[p, Ny*Nx:]  = p2[:, :, t].reshape(-1, order='F')
                Y[p, 0:Ny*Nx] = r1[:, :, t].reshape(-1, order='F')
                Y[p, Ny*Nx:]  = r2[:, :, t].reshape(-1, order='F')
                Z[p, :]       = Z_full[t, :]
            yield {"field_in": X, "cond_in": Z}, {"field": Y}

    full_ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                "field_in": tf.TensorSpec(shape=(lstm_hist_points, used_features), dtype=tf.float32),
                "cond_in" : tf.TensorSpec(shape=(lstm_hist_points, Dz), dtype=tf.float32)
            },
            {"field": tf.TensorSpec(shape=(lstm_hist_points, used_features), dtype=tf.float32)}
        )
    )

    train_ds = (full_ds.take(Ntrain)
                     .repeat()
                     .batch(batch_size, drop_remainder=False)
                     .prefetch(tf.data.AUTOTUNE))
    val_ds = (full_ds.skip(Ntrain)
                     .repeat()
                     .batch(batch_size, drop_remainder=False)
                     .prefetch(tf.data.AUTOTUNE))
    return train_ds, val_ds


# --- FiLM-conditioned STORN for baseline (same I/O names as your baseline LSTM) ---
def build_film_storn(used_features: int,
                     out_features: int,
                     cond_dim: int,
                     loss_data,
                     lr: float = 1e-4,
                     inter_dim: int = 60,
                     latent_dim: int = 60,
                     conditioning_mode: str = "film") -> tf.keras.Model:
    """
    Inputs:
      field_in: (B, T, used_features)
      cond_in : (B, T, cond_dim)
    Output:
      concat([field, z_mean, z_sigma]) so vae_loss_2D remains unchanged.
    """
    x = tf.keras.layers.Input(shape=(None, used_features), name='field_in')
    c = tf.keras.layers.Input(shape=(None, cond_dim),     name='cond_in')

    # covariance_concat concatenates context at the encoder/trunk input and does not apply
    # gamma/beta modulation. Other strategies preserve the existing FiLM path.
    encoder_input = (tf.keras.layers.Concatenate(name="context_concat")([x, c])
                     if conditioning_mode == "concat" else x)
    h = tf.keras.layers.Dense(inter_dim, activation='tanh', name='enc_dense')(encoder_input)
    z_mean  = tf.keras.layers.Dense(latent_dim, activation='linear',   name='z_mean')(h)
    z_sigma = tf.keras.layers.Dense(latent_dim, activation='softplus', name='z_var',
                                    kernel_initializer='zeros')(h)

    z = Sampling(name='Sample')([z_mean, z_sigma])

    # def sampling(args):
    #     mu, sig = args
    #     shp = tf.shape(mu)  # (B, T, latent_dim)
    #     eps = kb.random_normal(shape=shp, mean=0., stddev=1.)
    #     return mu + sig * eps

    #z = tf.keras.layers.Lambda(sampling, name='Sample')([z_mean, z_sigma])

    # Trunk LSTM
    phi = tf.keras.layers.Concatenate(axis=-1, name='phi')([h, z])
    trunk = tf.keras.layers.LSTM(
        inter_dim, activation='tanh',
        recurrent_activation='hard_sigmoid',
        return_sequences=True, name='lstm'
    )(phi)

    # FiLM from cond stream (per time step), Keras-native ops only
    u = tf.keras.layers.Dense(32, activation='tanh', name='film_mlp')(c)
    gamma = tf.keras.layers.Dense(inter_dim, activation='tanh', name='gamma')(u)
    beta  = tf.keras.layers.Dense(inter_dim, activation='tanh', name='beta')(u)
    scale = OnePlus(name='one_plus_gamma')(gamma)

    #scale = tf.keras.layers.Lambda(lambda t: t + 1.0, name='one_plus_gamma')(gamma)

    if conditioning_mode == "film":
        h_mod = tf.keras.layers.Multiply(name='film_mul')([trunk, scale])
        h_mod = tf.keras.layers.Add(name='film_add')([h_mod, beta])
    else:
        h_mod = trunk

    field = tf.keras.layers.Dense(out_features, activation='linear', name='field')(h_mod)
    out   = tf.keras.layers.Concatenate(axis=-1, name='concat_out')([field, z_mean, z_sigma])

    # Make output a dict so it matches y={"field": ...} in fit()
    model = tf.keras.Model(inputs={'field_in': x, 'cond_in': c},
                        outputs={'field': out})

    model.film_cond_input = c
    model.film_mlp        = u
    model.film_gamma      = gamma
    model.film_beta       = beta
    model.film_scale      = scale
    model.film_hidden     = trunk


    # Be explicit: map the loss to the "field" output
    model.compile(loss={'field': vae_loss_2D(alpha=loss_data)},
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    #               optimizer=tf.keras.optimizers.Adam(learning_rate=lr))


    model.summary()
    return model


def build_film_model(
    model_type: str,
    used_features: int,
    out_features: int,
    cond_dim: int,
    loss_data,
    lr: float = 1e-4,
    conditioning_mode: str = "film"
    ) -> tf.keras.Model:
    """Factory for FiLM-conditioned models."""
    if model_type.lower() == "lstm":
        return build_film_lstm(
            used_features, out_features, cond_dim, loss_data, lr, conditioning_mode
        )
    elif model_type.lower() == "storn":
        return build_film_storn(
            used_features, out_features, cond_dim, loss_data, lr,
            conditioning_mode=conditioning_mode
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}' for FiLM model.")


# ---------- baseline: FiLM-conditioned training ----------

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
    ensemble_cfg: Optional[dict] = None,
    **kwargs
    ):
    """
    baseline = FiLM conditioning from ensemble context at train time.
    Context is built from K extra nudged members and fed as a per-timestep signal.
    """
    ensemble_cfg = ensemble_cfg or {}
    strategy = experiment_name(ensemble_cfg.get("strategy", "baseline"))
    if strategy not in ({"baseline"} | CONTROL_STRATEGIES):
        raise ValueError(f"Unsupported strategy for this module: {strategy}")
    n_extra = int(ensemble_cfg.get("n_ens", 0))
    if n_extra < 2:
        raise ValueError("baseline requires at least two ensemble members (ensemble.n_ens >= 2).")
    n_modes_requested = int(ensemble_cfg.get("n_eigcov_modes", n_extra - 1))
    _validate_control_design(strategy, T, ensemble_cfg.get("sig", 0.1), n_extra)

    print(f"\n{strategy} Training ({'concatenated' if strategy in {'covariance_concat', 'mean_spread_concat'} else 'FiLM'} context)\n"
          "----------------------------------------")
    print(f"Case                    : {case_name}")
    print(f"Model type              : {model_type}")
    print(f"Basis                   : {basis_type}")
    print(f"Ensemble ID             : {j_ens}")
    print(f"Ensemble members (train): {n_extra}")

    # ---- [1/5] Load fields (slice by startup/length) ----
    print("\n[1/5] Loading fields ... ", end="", flush=True)
    md1 = sio.loadmat(os.path.join(data_dir, case_name, "psi1_ref.mat"))
    Ny = int(md1["Ny"])//1; Nx = int(md1["Nx"])//1
    delta = md1["delta"]; beta = md1["beta"]; rdrag = md1["rdrag"]; kd = md1["kd"]
    dt = float(md1["dt"])

    #skip_idx, data_times = compute_skip_and_length(md1, t_startup, T, dt_ml)
    method = "reversed"
    skip_idx, data_times = compute_skip_and_length(md1, t_startup, T, dt_ml, method=method)
    md2 = sio.loadmat(os.path.join(data_dir, case_name, "psi2_ref.mat"))
    # reference (as sources)
    if method == "reversed":
        r1 = md1["psi1_ref"][:, :, -data_times:]
        r2 = md2["psi2_ref"][:, :, -data_times:]
    else:
        r1 = md1["psi1_ref"][:, :, skip_idx: skip_idx + data_times]
        r2 = md2["psi2_ref"][:, :, skip_idx: skip_idx + data_times]
    del md1, md2
    gc.collect()

    # baseline (as sources)

    if method == "reversed":
        p1 = load_field(data_dir, case_name, basis_type, 1)[:, :, -data_times:]
        p2 = load_field(data_dir, case_name, basis_type, 2)[:, :, -data_times:]
    else:
        p1 = load_field(data_dir, case_name, basis_type, 1)[:, :, skip_idx: skip_idx + data_times]
        p2 = load_field(data_dir, case_name, basis_type, 2)[:, :, skip_idx: skip_idx + data_times]


    used_features = 2 * Ny * Nx
    print("Done.")
    print(f"Grid Ny×Nx              : {Ny}×{Nx}")
    print(f"data_times (sliced)     : {data_times}")
    print(f"dt (from data)          : {dt}")

    # ---- [2/5] Ensemble -> compact per-timestep context ----
    print("\n[2/5] Building per-timestep context from members ... ", end="", flush=True)

    sig = str(ensemble_cfg.get("sig", "0.1"))  # explicit or default
    mem1, mem2 = [], []
    if strategy in {"baseline", "ensemble_mean", "ensemble_variance", "covariance_concat", "mean_spread_concat"}:
        for k in range(1, n_extra + 1):
            m1 = load_member_field(data_dir, case_name, nb=1, member_index=k, basis_type=basis_type, sig=sig)
            m2 = load_member_field(data_dir, case_name, nb=2, member_index=k, basis_type=basis_type, sig=sig)
            if method == "reversed":
                mem1.append(m1[:, :, -data_times:])
                mem2.append(m2[:, :, -data_times:])
            else:
                mem1.append(m1[:, :, skip_idx: skip_idx + data_times])
                mem2.append(m2[:, :, skip_idx: skip_idx + data_times])

    n_modes = min(n_modes_requested, n_extra - 1, used_features)
    context_seed = int(kwargs.get("seed", 1729 + int(kwargs.get("repeat_index", 0) or 0)))
    Z_full = _build_control_train_context(
        strategy, mem1, mem2, p1, p2, n_modes=n_modes, seed=context_seed
    )

    del mem1[:]; del mem2[:]
    gc.collect()


    Dz = int(Z_full.shape[1])
    assert Z_full.shape[0] == data_times, "Z_full length must equal data_times"

    expected_dz = _control_cond_dim(
        strategy, used_features, n_extra, n_modes_requested
    )
    if Dz != expected_dz:
        raise ValueError(f"{strategy} context mismatch: got Dz={Dz}, expected {expected_dz}")
    print(f"Done. strategy={strategy}, M={n_extra}, D={used_features}, Dz={Dz}")

        # ---- [3/5] Windowing (aligned with deterministic) ----
    print("\n[3/5] Windowing ...", flush=True)
    total_times = data_times
    lstm_stride = max(1, int((dt_ml / dt)))
    T_field = min(data_times, total_times)
    idx = _make_window_indices(T_field, lstm_hist_points, lstm_stride, sub_sample, time_start=0)
    train_times = int(idx.shape[0])
    while train_times < 10 and lstm_hist_points > 1:
        print(f"  Warning: only {train_times} training windows with lstm_hist_points={lstm_hist_points}. Decreasing history length.")
        lstm_hist_points -= 1
        idx = _make_window_indices(T_field, lstm_hist_points, lstm_stride, sub_sample, time_start=0)
        train_times = int(idx.shape[0])
    print(f"Adjusted lstm_hist_points: {lstm_hist_points}, resulting in train_times: {train_times}")
    if train_times <= 0:
        raise ValueError("No training windows possible with given T/lstm_hist_points/dt_ml.")
    print(f"Size of Training Data   : {train_times}")

    train_split = 0.9  # keep in one place
    N = idx.shape[0]
    Ntrain = int(N * train_split)
    Nval = max(1, N - Ntrain)


    # Build datasets
    train_ds, val_ds = make_conditioned_dataset(
        src_train=(p1, p2),
        src_ref=(r1, r2),
        Z_full=Z_full,
        Ny=Ny, Nx=Nx, used_features=used_features,
        lstm_hist_points=lstm_hist_points, idx=idx, batch_size=batch_size
    )
    # free big arrays if you want:
    #del p1, p2, r1, r2, Z_full
    gc.collect()


    # ---- [4/5] Model & training ----
    print("\n[4/5] Building FiLM model & training ...", flush=True)
    os.makedirs(neural_nets_dir, exist_ok=True)
    if model_type.lower() == "lstm":
        alpha = loss_params_2D(kd, delta, Ny, Nx, beta, rdrag)
    elif model_type.lower() == "storn":
        alpha = loss_params_2DV(kd, delta, Ny, Nx, beta, rdrag, latent_dim=60)
    else:
        raise ValueError(f"Unknown model_type '{model_type}' for FiLM model.")

    model = build_film_model(
        model_type=model_type,
        used_features=used_features,
        out_features=used_features,
        cond_dim=Dz,
        loss_data=alpha,
        lr=(lr if lr is not None else 1e-3),
        conditioning_mode=("concat" if strategy in {"covariance_concat", "mean_spread_concat"} else "film"),
    )

    Ne = 500
    repeat_index = kwargs.get("repeat_index", None)
    seed = kwargs.get("seed", None)
    if seed is None and repeat_index is not None:
        seed = 1729 + int(repeat_index)
    if seed is not None:
        seed_everything(seed)

    train_again = bool(kwargs.get("train_again", False))
    save_path_final = model_save_path2(
            neural_nets_dir, case_name, model_type, basis_type,
            Nepochs, dt_ml, T, j_ens,
            strategy=strategy, n_members=n_extra,
            extra_tokens={"Dz": Dz},
            repeat_index=repeat_index,
        )

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

    Nsets = int(np.ceil(Nepochs / Ne))


    # check what exists already to continue training
    save_path_list = [model_save_path2(
            neural_nets_dir, case_name, model_type, basis_type,
            Nepochs_j, dt_ml, T, j_ens,
            strategy=strategy, n_members=n_extra,
            extra_tokens={"Dz": Dz},
            repeat_index=repeat_index,
        ) for Nepochs_j in range(Ne, Nepochs+1, Ne)]

    model, Nsets, latest_ckpt = resume_or_initialize_training(
        model, save_path_list, Nepochs, Ne, train_again=train_again
    )
    done_epochs = epochs_of(latest_ckpt) if latest_ckpt is not None else 0

    steps_per_epoch  = max(1, int(np.ceil(Ntrain / batch_size)))
    validation_steps = max(1, int(np.ceil(Nval    / batch_size)))

    hist_all = []
    for je in range(Nsets):
        hist = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=Ne,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=2
        )
        hist_all.append(hist.history)

        Nepochs_j = (je + 1) * Ne + done_epochs
        save_path = model_save_path2(
            neural_nets_dir, case_name, model_type, basis_type,
            Nepochs_j, dt_ml, T, j_ens,
            strategy=strategy, n_members=n_extra,
            extra_tokens={"Dz": Dz},
            repeat_index=repeat_index,
        )
        model.save(save_path, save_format="h5")
        print(f"  Saved: {save_path}", flush=True)

    # save training history per chunk
    hist_path = save_path.replace(".h5", f"_history.json")
    # merge all chunks into one dict of lists
    history_merged = {}
    for hh in hist_all:
        for k, v in hh.items():
            history_merged.setdefault(k, []).extend(v)

    hist_path = save_path.replace(".h5", "_history.json")
    with open(hist_path, "w") as f:
        json.dump(history_merged, f)
    print(f"  History saved: {hist_path}", flush=True)

    del p1, p2, r1, r2, Z_full
    gc.collect()


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
    **kwargs
 ):
    """
    baseline test: identical statistics to deterministic tester (PDFs, rPDFs, AOT, spectra, R/S),
    with a strictly causal covariance-mode proxy derived from coarse input.
    Checkpoints are resolved via model_save_path2 pattern (hash tolerant).
    """

    ensemble_cfg = (ensemble_cfg or {})
    strategy = experiment_name(ensemble_cfg.get("strategy", "baseline"))
    if strategy not in ({"baseline"} | CONTROL_STRATEGIES):
        raise ValueError(f"Unsupported strategy for this module: {strategy}")
    n_extra = int(ensemble_cfg.get("n_ens", 0))
    if n_extra < 2:
        raise ValueError("baseline testing requires ensemble.n_ens >= 2.")
    _validate_control_design(strategy, T, ensemble_cfg.get("sig", 0.1), n_extra)

    print(f"\n{strategy} Testing ({'concatenated' if strategy in {'covariance_concat', 'mean_spread_concat'} else 'FiLM'} context; Coarse DNS)\n"
          "------------------------------------------------")
    print(f'Case                    : {case_name}')
    print(f'Model type              : {model_type}')
    print(f'Basis                   : {basis_type}')
    print(f'Nens                    : {Nens}')

    repeat_index = kwargs.get("repeat_index", None)

    # [0/6] Load reference & coarse fields
    test_again = bool(kwargs.get("test_again", False))
    save_file = results_save_path2(
        ml_results_dir, case_name_train, model_type, basis_type, Nepochs, dt_ml, T, Nens,
        strategy=strategy, n_members=n_extra,
        repeat_index=repeat_index,
    )

    if (os.path.exists(save_file) and (not test_again)):
        print(f"\nResults already exist ({save_file}). Use test_again=True to re-test.\n")
        return

    # The merged test case contains multi-GB MATLAB-v5 arrays. loadmat cannot
    # slice those files in place, and simultaneous sweep jobs can saturate the
    # shared filesystem so badly that every load becomes slower. Serialize only
    # the I/O-heavy load phase across baseline jobs; release before statistics/model
    # work so the expensive computation remains concurrent.
    io_lock_path = "/tmp/qg_test_data_load.lock"
    io_lock = open(io_lock_path, "w")
    print(f"Waiting for node-local test-data I/O slot ({io_lock_path})... ", end="", flush=True)
    fcntl.flock(io_lock, fcntl.LOCK_EX)
    print("acquired.", flush=True)

    # -------- [1/6] Headers --------
    print('[1/6] Importing Reference data for testing... ', end='', flush=True)
    load_t0 = time.time()
    md = sio.loadmat(os.path.join(data_dir, case_name, 'psi1_ref.mat'))
    print(f"psi1_ref={time.time()-load_t0:.1f}s; ", end="", flush=True)
    Ny = int(md["Ny"])//1
    Nx = int(md["Nx"])//1
    delta = md["delta"]; beta = md["beta"]; rdrag = md["rdrag"]; kd = md["kd"]
    dt = float(md["dt"])
    Lratio = 1.0
    skip_idx, total_times = compute_skip_and_length(md, t_startup, test_time_units, dt_ml)
    used_features = Ny * Nx * 2
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    print('Done!\n')

    # -------- [2/6] Load/slice reference & coarse fields --------
    print('[2/6] Importing coarse and Reference data for testing... ', end='', flush=True)
    psi_ref = np.zeros((Ny, Nx, total_times, 2), dtype=np.float32)
    r1 = md["psi1_ref"]; psi_ref[:, :, :, 0] = r1[:, :, skip_idx: skip_idx + total_times]
    load_t0 = time.time()
    r2 = load_field(data_dir, case_name, "ref", 2)
    print(f"psi2_ref={time.time()-load_t0:.1f}s; ", end="", flush=True)
    psi_ref[:, :, :, 1] = r2[:, :, skip_idx: skip_idx + total_times]
    del r1, r2, md
    gc.collect()


    psi_coarse = np.zeros((Ny, Nx, total_times, 2), dtype=np.float32)
    load_t0 = time.time()
    c1 = load_field(data_dir, case_name, "coarse", 1)
    print(f"psi1_coarse={time.time()-load_t0:.1f}s; ", end="", flush=True)
    psi_coarse[:, :, :, 0] = c1[:, :, skip_idx: skip_idx + total_times]; del c1
    load_t0 = time.time()
    c2 = load_field(data_dir, case_name, "coarse", 2)
    print(f"psi2_coarse={time.time()-load_t0:.1f}s; ", end="", flush=True)
    psi_coarse[:, :, :, 1] = c2[:, :, skip_idx: skip_idx + total_times]; del c2
    gc.collect()

    print(f"Coarse field zeros: layer 1: {(psi_coarse[:,:,:,0]==0).sum()}, layer 2: {(psi_coarse[:,:,:,1]==0).sum()}")
    print(f"Ref field zeros   : layer 1: {(psi_ref[:,:,:,0]==0).sum()},   layer 2: {(psi_ref[:,:,:,1]==0).sum()}")
    print('Done!\n')

    # -------- [3/6] Build test input (with stride) --------
    print('[3/6] Building test input... ', end='', flush=True)
    input_test = np.zeros((1, total_times, used_features), dtype=np.float32)
    input_test[0, :, 0:Nx*Ny]       = np.reshape(np.transpose(psi_coarse[:, :, :, 0], (2, 0, 1)),
                                                 (1, total_times, Nx*Ny), order='F')
    input_test[0, :, Nx*Ny:2*Nx*Ny] = np.reshape(np.transpose(psi_coarse[:, :, :, 1], (2, 0, 1)),
                                                 (1, total_times, Nx*Ny), order='F')
    stride = max(1, int((dt_ml / dt)))
    input_test = input_test[0, 0:-1:stride, :]
    input_test = np.reshape(input_test, (1, input_test.shape[0], input_test.shape[1]))
    test_times = input_test.shape[1]
    print('Data Reshape Complete!\n')
    fcntl.flock(io_lock, fcntl.LOCK_UN)
    io_lock.close()
    print("Released shared test-data I/O slot.", flush=True)

    # -------- [4/6] Stats preallocation (parity with deterministic) --------
    print('[4/6] Compute stats... ', end='', flush=True)
    t0 = time.time()
    Np = 2001; pleft, pright = -4.0, 4.0
    pdf_xaxis = np.linspace(pleft, pright, Np); dp = (pright-pleft)/float(Np-1)

    psit_flag = np.zeros((Ny, Nx), dtype=np.float32)
    psi_pdf_ML     = np.zeros((Nens, Np, 2), dtype=np.float32)
    psi_rpdf_ML    = np.zeros((Nens, Nx, Ny, Np, 2), dtype=np.float32)
    psi_rpdf_ref   = np.zeros((Nx, Ny, Np, 2), dtype=np.float32)
    psi_rpdf_coarse= np.zeros((Nx, Ny, Np, 2), dtype=np.float32)

    Ni = 100; dt_i = dt_ml*Ni; NT = int(T/dt_ml)

    Rr = np.zeros((Ny, Nx, test_times, 2), dtype=np.float32); Rc = np.zeros((Ny, Nx, test_times, 2), dtype=np.float32); Rt = np.zeros((Ny, Nx, NT, 2), dtype=np.float32)
    Sr = np.zeros((Ny, Nx, test_times, 2), dtype=np.float32); Sc = np.zeros((Ny, Nx, test_times, 2), dtype=np.float32); St = np.zeros((Ny, Nx, NT, 2), dtype=np.float32)

    R_ML = np.zeros((Nens, Ny, Nx, int(test_times/Ni), 2), dtype=np.float32)
    R_ref = np.zeros((Ny, Nx, int(test_times/Ni), 2), dtype=np.float32)
    R_coarse = np.zeros((Ny, Nx, int(test_times/Ni), 2), dtype=np.float32)
    S_ML = np.zeros((Nens, Ny, Nx, int(test_times/Ni), 2), dtype=np.float32)
    S_ref = np.zeros((Ny, Nx, int(test_times/Ni), 2), dtype=np.float32)
    S_coarse = np.zeros((Ny, Nx, int(test_times/Ni), 2), dtype=np.float32)

    spectrum_ML = np.zeros((Nens, test_times, 2), dtype=np.float32)
    Psi_ML      = np.zeros((Nens, Ny, test_times, 2), dtype=np.float32)
    indEE_ML    = np.zeros((Nens, test_times, 2), dtype=np.float32)
    indEE_ref   = np.zeros((test_times, 2), dtype=np.float32)

    time_start = 0
    time_end   = test_times - 1
    times      = time_end - time_start + 1
    flag_coef  = float(times * Nx * Ny)

    # stride-sliced fields for stats
    psi_ref_s    = psi_ref[:, :, 0:-1:stride, :];    del psi_ref
    psi_coarse_s = psi_coarse[:, :, 0:-1:stride, :]; del psi_coarse
    gc.collect()

    # reference stats + PDFs of reference & coarse
    print()
    bins = np.linspace(pleft, pright, Np+1)
    for l in range(2):
        print(f"        ↳ layer {l+1}/2: max|ref|={np.max(np.abs(psi_ref_s[:, :, :, l])):.5f}, "
              f"max={np.max(psi_ref_s[:, :, :, l]):.5f}")
        Rr[..., l], Sr[..., l] = autocorr_fft(psi_ref_s[..., l], test_times)
        Rc[..., l], Sc[..., l] = autocorr_fft(psi_coarse_s[..., l], test_times)
        Rt[..., l], St[..., l] = autocorr_fft(psi_ref_s[..., :NT, l], NT)
        for i in range(Nx):
            for j in range(Ny):
                # Rr[j, i, :, l] = scs.correlate(psi_ref_s[j, i, :, l],    psi_ref_s[j, i, :, l],    mode='same', method='auto')
                # Rc[j, i, :, l] = scs.correlate(psi_coarse_s[j, i, :, l], psi_coarse_s[j, i, :, l], mode='same', method='auto')
                # Rt[j, i, :, l] = scs.correlate(psi_ref_s[j, i, 0:NT, l],  psi_ref_s[j, i, 0:NT, l],  mode='same', method='auto')

                # Sr[j, i, :, l] = np.abs(np.fft.fftshift(np.fft.fft(Rr[j, i, :, l])))/test_times
                # Sc[j, i, :, l] = np.abs(np.fft.fftshift(np.fft.fft(Rc[j, i, :, l])))/test_times
                # St[j, i, :, l] = np.abs(np.fft.fftshift(np.fft.fft(Rt[j, i, :, l])))/NT

                # reduced save (robust to off-by-one)
                idx = np.arange(0, Rr.shape[2]-1, Ni, dtype=int)
                nsave = min(R_ref.shape[2], idx.size)
                if nsave > 0:
                    R_ref[j, i, 0:nsave, l]    = Rr[j, i, idx[:nsave], l]
                    R_coarse[j, i, 0:nsave, l] = Rc[j, i, idx[:nsave], l]
                    S_ref[j, i, 0:nsave, l]    = Sr[j, i, idx[:nsave], l]
                    S_coarse[j, i, 0:nsave, l] = Sc[j, i, idx[:nsave], l]


                # PDFs (reference & coarse)
                vals_ref = psi_ref_s[i, j, :-1, l]
                vals_coarse = psi_coarse_s[i, j, :-1, l]

                hist_ref, _ = np.histogram(vals_ref, bins=bins)
                hist_coarse, _ = np.histogram(vals_coarse, bins=bins)

                # broadcast back into memmaps if needed
                psi_rpdf_ref[i, j, :, l] = hist_ref
                psi_rpdf_coarse[i, j, :, l] = hist_coarse
                # for tt in range(time_start, time_end):
                #     # ref
                #     flag_pp = int((psi_ref_s[j, i, tt, l] - pleft)/dp); flag_pp = max(0, min(Np-1, flag_pp))
                #     psi_rpdf_ref[j, i, flag_pp, l] += 1
                #     # coarse
                #     flag_pp = int((psi_coarse_s[j, i, tt, l] - pleft)/dp); flag_pp = max(0, min(Np-1, flag_pp))
                #     psi_rpdf_coarse[j, i, flag_pp, l] += 1
        print('...Done.', flush=True)

    R_train = 1*Rt; S_train = 1*St
    spectrum_ref     = np.mean(Sr, axis=(0,1))
    spectrum_coarse  = np.mean(Sc, axis=(0,1))
    spectrum_train   = np.mean(St, axis=(0,1))
    Psi_ref          = np.sum(psi_ref_s,   axis=1)/float(Nx)
    Psi_coarse_zm    = np.sum(psi_coarse_s, axis=1)/float(Nx)
    indEE_ref        = np.sum(np.abs(psi_ref_s), axis=(0,1))/float(Nx*Ny)

    # AOT
    Npt = 25; tleft, tright = 0.0, 4.0
    thresh = np.linspace(tleft, tright, Npt)
    aot_ref    = np.zeros((Npt, test_times, 2), dtype=np.float32)
    aot_coarse = np.zeros((Npt, test_times, 2), dtype=np.float32)
    aot_ML     = np.zeros((Nens, Npt, test_times, 2), dtype=np.float32)
    for l in range(2):
        print(f"        ↳ AOT layer {l+1}/2 ...", end='', flush=True)
        vals_ref = np.abs(psi_ref_s[..., l])  # (Ny,Nx,T)
        vals_coarse = np.abs(psi_coarse_s[..., l])
        # Avoid (Ny,Nx,T,Npt) masks: at T=525k each is ~7.6 GiB.
        for nt, threshold in enumerate(thresh):
            aot_ref[nt, :, l] = (vals_ref >= threshold).mean(axis=(0, 1))
            aot_coarse[nt, :, l] = (vals_coarse >= threshold).mean(axis=(0, 1))
        print(f" done.", flush=True)

    del psi_ref_s, Rr, Rc, Rt, Sr, Sc, St
    gc.collect()
    print("Done! ({:.2f} seconds)".format(time.time() - t0))
    print()

    # -------- [5/6] ML predictions --------
    print('[5/6] ML predictions\n', end='', flush=True)
    for jens in range(Nens):
        print('     ...realization # ' + str(jens+1), end='', flush=True)
        ck_case = (case_name_train or case_name)
        n_modes_requested = int(ensemble_cfg.get("n_eigcov_modes", n_extra - 1))
        expected_dz = _control_cond_dim(
            strategy, used_features, n_extra, n_modes_requested
        )
        # Training always records the exact context width in the checkpoint.
        ckpt_pattern = model_save_path2(
            neural_nets_dir, ck_case, model_type, basis_type,
            Nepochs, dt_ml, T, jens + 1,
            strategy=strategy, n_members=n_extra,
            repeat_index=repeat_index,
            extra_tokens={"Dz": expected_dz},
        )
        if not os.path.exists(ckpt_pattern):
            raise FileNotFoundError(f"No exact {strategy} checkpoint found: {ckpt_pattern}")

        model = tf.keras.models.load_model(ckpt_pattern, compile=False)

        # Causal coarse-only context with the checkpoint's exact Dz.
        need_cond = any(t.name.split(":")[0].endswith("cond_in") for t in model.inputs)
        if need_cond:
            print(f" (with {strategy} context)", end='', flush=True)
            cond_dim = int(model.get_layer("cond_in").output.shape[-1])
            n_modes_requested = int(ensemble_cfg.get("n_eigcov_modes", n_extra - 1))
            expected_cond_dim = _control_cond_dim(
                strategy, used_features, n_extra, n_modes_requested
            )
            if cond_dim != expected_cond_dim:
                raise ValueError(
                    f"{strategy} checkpoint cond_dim={cond_dim}, expected {expected_cond_dim}"
                )
            n_modes = min(n_modes_requested, n_extra - 1, used_features)
            proxy_window = int(ensemble_cfg.get("proxy_window", 32))
            context_seed = int(kwargs.get(
                "seed", 1729 + int(kwargs.get("repeat_index", 0) or 0)
            ))
            Zt = _build_control_test_context(
                strategy, psi_coarse_s, cond_dim=cond_dim, n_modes=n_modes,
                window=proxy_window, seed=context_seed
            )
            if Zt.shape != (test_times, cond_dim):
                raise ValueError(f"Test context is {Zt.shape}, expected {(test_times, cond_dim)}")
            print(f" [context={Zt.nbytes/2**30:.2f} GiB, RSS={_rss_gib():.2f} GiB]",
                  end="", flush=True)

            # Keras predict() keeps the NumPy inputs alive while its data
            # adapter creates additional tensors/buffers. For M=8 and the
            # 525k-step test trajectory, cond_in alone is ~16.9 GiB. Convert
            # explicitly, release the large NumPy/coarse arrays, then call the
            # model directly to avoid the duplicate peak.
            field_tensor = tf.convert_to_tensor(input_test, dtype=tf.float32)
            cond_tensor = tf.convert_to_tensor(Zt[np.newaxis, :, :], dtype=tf.float32)
            del Zt
            if jens == Nens - 1:
                del input_test, psi_coarse_s
            gc.collect()
            print(f" [tensor RSS={_rss_gib():.2f} GiB]", end="", flush=True)
            pred = model({"field_in": field_tensor, "cond_in": cond_tensor}, training=False)
            del field_tensor, cond_tensor
            gc.collect()
        else:
            print("fill zero context", end='', flush=True)
            pred = model.predict({"field_in": input_test.astype(np.float32)}, verbose=0)

        if isinstance(pred, dict):
            output_test = pred.get("field", next(iter(pred.values())))
        elif isinstance(pred, (list, tuple)):
            output_test = pred[0]
        else:
            output_test = pred
        if tf.is_tensor(output_test):
            output_test = output_test.numpy()
        del pred
        gc.collect()

        # reshape to ψ fields
        psi_ML = np.zeros((Ny, Nx, test_times, 2), dtype=np.float32)
        psi_ML[:,:,:,0] = np.transpose(np.reshape(output_test[0,:,0:Nx*Ny], (test_times, Nx, Ny), order='F'), (1,2,0))
        psi_ML[:,:,:,1] = np.transpose(np.reshape(output_test[0,:,Nx*Ny:2*Nx*Ny], (test_times, Nx, Ny), order='F'), (1,2,0))

        # ML stats (R/S, spectra, PDFs, AOT)
        Rm = np.zeros((Ny, Nx, test_times, 2), dtype=np.float32); Sm = np.zeros((Ny, Nx, test_times, 2), dtype=np.float32)
        print()
        for l in range(2):
            print(f"        ↳ layer {l+1}/2: max|ML|={np.max(np.abs(psi_ML[..., l])):.5f}, "
                    f"max={np.max(psi_ML[..., l]):.5f}", flush=True)
            Rm[..., l], Sm[..., l] = autocorr_fft(psi_ML[..., l], test_times)
            for i in range(Nx):
                for j in range(Ny):
                    # Rm[j,i,:,l] = scs.correlate(psi_ML[j,i,:,l], psi_ML[j,i,:,l], mode='same', method='auto')
                    # Sm[j,i,:,l] = np.abs(np.fft.fftshift(np.fft.fft(Rm[j,i,:,l])))/test_times
                    # reduced save
                    idx_m = np.arange(0, Rm.shape[2]-1, Ni, dtype=int)
                    nsave_m = min(R_ML.shape[3], idx_m.size)
                    if nsave_m > 0:
                        R_ML[jens, j, i, 0:nsave_m, l] = Rm[j, i, idx_m[:nsave_m], l]
                        S_ML[jens, j, i, 0:nsave_m, l] = Sm[j, i, idx_m[:nsave_m], l]

                    # PDFs (global + pointwise)
                    vals = psi_ML[j, i, :-1, l]
                    hist, _ = np.histogram(vals, bins=bins)
                    psi_rpdf_ML[jens, i, j, :, l] = hist
                    psi_pdf_ML[jens, :, l] += hist
                    # for tt in range(time_start, time_end):
                    #     flag_pp = int((psi_ML[j,i,tt,l] - pleft)/dp); flag_pp = max(0, min(Np-1, flag_pp))
                    #     psi_pdf_ML[jens, flag_pp, l] += 1
                    #     psi_rpdf_ML[jens, j, i, flag_pp, l] += 1

        spectrum_ML[jens, :, :] = np.mean(Sm, axis=(0,1))
        Psi_ML[jens, :, :, :]   = np.sum(psi_ML, axis=1)/float(Nx)
        indEE_ML[jens, :]       = np.sum(np.abs(psi_ML), axis=(0,1))/float(Nx*Ny)
        del output_test, Rm, Sm
        gc.collect()

        for l in range(2):
            print(f"        ↳ AOT layer {l+1}/2", end='', flush=True)
            vals = np.abs(psi_ML[..., l])
            for nt, threshold in enumerate(thresh):
                aot_ML[jens, nt, :, l] = (vals >= threshold).mean(axis=(0, 1))
            print("...done.", flush=True)
            # for tt in range(time_start, time_end):
            #     # thresholds loop (25)
            #     for nt in range(Npt):
            #         mask = (np.abs(psi_ML[:, :, tt, l]) >= thresh[nt]).astype(float)
            #         aot_ML[jens, nt, tt, l] = np.sum(mask)/float(Nx*Ny)

        del psi_ML
        print(' ...done.\n', flush=True)

    if "input_test" in locals():
        del input_test
    if "psi_coarse_s" in locals():
        del psi_coarse_s
    gc.collect()

    # -------- [6/6] Derived stats + save --------
    print('[6/6] Computing derived ML stats + saving ... ', end='', flush=True)
    aot_ML = np.mean(aot_ML, axis=0)

    f       = np.fft.fftshift(np.fft.fftfreq(test_times, d=stride*dt))
    f_train = np.fft.fftshift(np.fft.fftfreq(NT,         d=stride*dt))
    f_i     = np.fft.fftshift(np.fft.fftfreq(int(test_times/Ni), d=Ni*stride*dt))

    os.makedirs(ml_results_dir, exist_ok=True)
    save_file = results_save_path2(
        ml_results_dir, case_name_train, model_type, basis_type, Nepochs, dt_ml, T, Nens,
        strategy=strategy, n_members=n_extra,
        repeat_index=repeat_index
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
    print(f'Ensemble Stats Saved: {save_file}\n')
    return
