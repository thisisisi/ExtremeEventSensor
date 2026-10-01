#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and evaluate an attention or TCN residual corrector."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from .density import kde_pdf_on_grid
from .config import CorrectorConfig
from .data import (
    load_train_data,
    load_train_coarse_data,
    load_test_coarse_data,
    load_test_reference_data,
    make_windows,
    compute_data_statistics,
    normalize,
    denormalize,
    compute_tail_thresholds,
    context_from_coarse_trajectory,
    apply_sliding_window_inference,
)
from .model import ResidualAttentionCorrector, ResidualTCNCorrector


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )


def plot_three_way_comparison(
    times: np.ndarray,
    coarse_uncorrected: np.ndarray,
    coarse_corrected: np.ndarray,
    reference: np.ndarray,
    save_path: Path,
) -> None:
    """Plot coarse (uncorrected), coarse (corrected via model), and reference."""
    set_plot_style()
    labels = [r"$z_1$", r"$z_2$", r"$z_3$"]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.8), sharex=True, constrained_layout=True)

    for i, ax in enumerate(axes):
        ax.plot(times, reference[:, i], color="#111111", lw=1.8, label="reference", zorder=3)
        ax.plot(times, coarse_corrected[:, i], color="#2ca02c", lw=1.5, ls="--", label="corrected", zorder=2)
        ax.plot(times, coarse_uncorrected[:, i], color="#ff7f0e", lw=1.9, ls="-", alpha=0.85, label="coarse", zorder=4)
        ax.set_ylabel(labels[i])

    axes[0].legend(frameon=False, loc="best")
    axes[-1].set_xlabel("time")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_log_pdfs_3way(
    reference_z: np.ndarray,
    coarse_z: np.ndarray,
    corrected_z: np.ndarray,
    save_dir: Path,
) -> None:
    """Plot log-PDFs showing reference, coarse (uncorrected), and corrected on same plot."""
    set_plot_style()
    for i, label in enumerate([r"$z_1$", r"$z_2$", r"$z_3$"]):
        x_ref = reference_z[:, i]
        x_coarse = coarse_z[:, i]
        x_corr = corrected_z[:, i]

        # Compute grid from all three sequences
        x_lo = min(
            float(np.quantile(x_ref, 0.001)),
            float(np.quantile(x_coarse, 0.001)),
            float(np.quantile(x_corr, 0.001))
        )
        x_hi = max(
            float(np.quantile(x_ref, 0.999)),
            float(np.quantile(x_coarse, 0.999)),
            float(np.quantile(x_corr, 0.999))
        )
        pad = 0.08 * max(x_hi - x_lo, 1e-8)
        grid = np.linspace(x_lo - pad, x_hi + pad, 900)

        p_ref = np.maximum(kde_pdf_on_grid(x_ref, grid), 1e-12)
        p_coarse = np.maximum(kde_pdf_on_grid(x_coarse, grid), 1e-12)
        p_corr = np.maximum(kde_pdf_on_grid(x_corr, grid), 1e-12)

        fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
        ax.plot(grid, p_ref, color="#111111", lw=2.1, label="reference", zorder=3)
        ax.plot(grid, p_corr, color="#2ca02c", lw=1.9, ls="--", label="corrected", zorder=2)
        ax.plot(grid, p_coarse, color="#ff7f0e", lw=2.2, ls="-", alpha=0.9, label="coarse", zorder=4)
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("PDF")
        ax.legend(frameon=False, loc="best")
        fig.savefig(save_dir / f"log_pdf_z{i+1}_3way.png", bbox_inches="tight")
        plt.close(fig)


def plot_multi_threshold_aot(
    times: np.ndarray,
    reference_z: np.ndarray,
    coarse_z: np.ndarray,
    corrected_z: np.ndarray,
    save_dir: Path,
    var_idx: int = 2,
    percentiles: list[float] | None = None,
) -> None:
    """Plot multi-threshold AOT curves for reference, coarse, and corrected.

    Args:
        times: Time array
        reference_z: Reference trajectories (T, 3)
        coarse_z: Coarse trajectories (T, 3)
        corrected_z: Corrected trajectories (T, 3)
        save_dir: Output directory
        var_idx: Which z component to threshold on (default 2 for z3)
        percentiles: List of percentiles, e.g., [50, 75, 90, 95, 99]
    """
    if percentiles is None:
        percentiles = [50, 75, 90, 95, 99]

    set_plot_style()

    # Compute percentile thresholds from reference
    thresholds = [np.percentile(np.abs(reference_z[:, var_idx]), p) for p in percentiles]

    # Cumulative exceedance AOT(t) = (1/t) * int_0^t 1(|z| > tau) dt
    ref_abs = np.abs(reference_z[:, var_idx])
    coarse_abs = np.abs(coarse_z[:, var_idx])
    corr_abs = np.abs(corrected_z[:, var_idx])

    fig, axes = plt.subplots(len(thresholds), 1, figsize=(11.5, 2.2 * len(thresholds)), sharex=True, constrained_layout=True)
    if len(thresholds) == 1:
        axes = [axes]

    for k, (ax, threshold, percentile) in enumerate(zip(axes, thresholds, percentiles)):
        ref_ind = (ref_abs > threshold).astype(np.float64)
        coarse_ind = (coarse_abs > threshold).astype(np.float64)
        corr_ind = (corr_abs > threshold).astype(np.float64)

        denom = np.arange(1, ref_ind.size + 1, dtype=np.float64)
        ref_aot = np.cumsum(ref_ind) / denom
        coarse_aot = np.cumsum(coarse_ind) / denom
        corr_aot = np.cumsum(corr_ind) / denom

        ax.plot(times, ref_aot, color="#111111", lw=2.0, label="reference")
        ax.plot(times, coarse_aot, color="#ff7f0e", lw=1.6, ls=":", label="coarse")
        ax.plot(times, corr_aot, color="#2ca02c", lw=1.6, ls="--", label="corrected")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(f"AOT\n{percentile:.0f}%")
        ax.set_title(f"|z{var_idx+1}| > {threshold:.3f}", fontsize=10)
        if k == 0:
            ax.legend(frameon=False, loc="best")

    axes[-1].set_xlabel("time")

    fig.savefig(save_dir / "multi_threshold_aot_z3.png", bbox_inches="tight")
    plt.close(fig)


def plot_3d_comparison(
    reference_z: np.ndarray,
    coarse_z: np.ndarray,
    corrected_z: np.ndarray,
    save_dir: Path,
) -> None:
    """Plot 3D trajectories for reference, coarse, and corrected on test timescale."""
    set_plot_style()

    # Import mplot3d for 3D plotting
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Plot trajectories
    ax.plot(reference_z[:, 0], reference_z[:, 1], reference_z[:, 2],
            color="#111111", lw=2.0, label="reference", alpha=0.9)
    ax.plot(coarse_z[:, 0], coarse_z[:, 1], coarse_z[:, 2],
            color="#ff7f0e", lw=1.6, ls=":", label="coarse", alpha=0.7)
    ax.plot(corrected_z[:, 0], corrected_z[:, 1], corrected_z[:, 2],
            color="#2ca02c", lw=1.6, ls="--", label="corrected", alpha=0.85)

    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_zlabel(r"$z_3$")
    ax.legend(frameon=False, loc="best")

    fig.savefig(save_dir / "3d_comparison.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def repeat_index_from_output_dir(out_dir: Path) -> int | None:
    match = re.search(r"_rep(\d+)$", out_dir.name)
    return int(match.group(1)) if match else None


def fit_quantile_map(source: np.ndarray, target: np.ndarray, n_quantiles: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit a monotone quantile map from source samples to target samples."""
    q = np.linspace(0.0, 100.0, max(11, int(n_quantiles)))
    source_q = np.percentile(source, q)
    target_q = np.percentile(target, q)
    source_q = np.maximum.accumulate(source_q)
    source_unique, unique_idx = np.unique(source_q, return_index=True)
    target_unique = target_q[unique_idx]
    if source_unique.size < 2:
        eps = 1e-6
        source_unique = np.array([source_q[0] - eps, source_q[0] + eps], dtype=np.float32)
        target_unique = np.array([target_q[0], target_q[-1]], dtype=np.float32)
    return source_unique.astype(np.float32), target_unique.astype(np.float32)


def apply_quantile_map(values: np.ndarray, source_q: np.ndarray, target_q: np.ndarray) -> np.ndarray:
    """Apply a fitted monotone quantile map with linear interpolation."""
    mapped = np.interp(values, source_q, target_q, left=target_q[0], right=target_q[-1])
    return mapped.astype(np.float32)


def train_and_evaluate(cfg: CorrectorConfig, out_dir: Path) -> None:
    """
    Train and test the configured low-dimensional residual corrector.

    There are exactly two modes:
    - n_ensemble_members_used == 0: deterministic baseline
    - n_ensemble_members_used > 0: the selected architecture with FiLM conditioning

    Testing is always nonintrusive. The corrected trajectory is generated from
    the coarse trajectory only. If FiLM is active, test context is derived from
    rolling statistics of the coarse trajectory, following the QG FiLM tester.
    """
    print(f"\n{'='*70}")
    print("LOW-DIMENSIONAL RESIDUAL TCN CORRECTOR")
    print(f"{'='*70}\n")

    # ============================================================
    # [1] Load and normalize training data
    # ============================================================
    print(f"[1/6] Loading training data from run {cfg.run_id_train}...")
    time_train_raw, nudged_train_raw, reference_train_raw, context_train_raw = load_train_data(cfg)
    _, coarse_train_raw, _, _ = load_train_coarse_data(cfg)

    train_stop = min(cfg.train_interval_length, reference_train_raw.shape[0])
    stride = max(int(cfg.ml_stride), 1)
    time_train = time_train_raw[:train_stop:stride]
    nudged_train = nudged_train_raw[:train_stop:stride]
    coarse_train = coarse_train_raw[:train_stop:stride]
    reference_train = reference_train_raw[:train_stop:stride]
    context_train = None if context_train_raw is None else context_train_raw[:train_stop:stride]

    print(f"     train interval: raw={train_stop} points, stride={stride}, ML points={reference_train.shape[0]}")

    # Compute normalization from reference
    data_mean, data_std = compute_data_statistics(reference_train)
    print(f"     mean: {data_mean}")
    print(f"     std: {data_std}")

    # Compute fixed tail thresholds from unnormalized reference
    tail_info = compute_tail_thresholds(reference_train, percentiles=[50, 75, 90, 95, 99],
                                        tail_threshold_std=cfg.tail_threshold_std)
    z3_tail_threshold_unnorm = tail_info["z3_threshold_std"]
    z3_std_unnorm = tail_info["z3_std"]

    # Convert to normalized coordinates for training
    z3_tail_threshold_norm = z3_tail_threshold_unnorm / z3_std_unnorm
    print(f"     z3 tail threshold (unnorm): {z3_tail_threshold_unnorm:.6f}")
    print(f"     z3 tail threshold (norm): {z3_tail_threshold_norm:.6f}")
    print(f"     z3 tail percentiles (unnorm):")
    for pctl in [50, 75, 90, 95, 99]:
        print(f"       q{pctl}: {tail_info[f'z3_q{pctl:.0f}']:.6f}")

    # Normalize sequences
    nudged_train_norm = normalize(nudged_train, data_mean, data_std)
    coarse_train_norm = normalize(coarse_train, data_mean, data_std)
    reference_train_norm = normalize(reference_train, data_mean, data_std)

    context_mean = None
    context_std = None
    context_train_norm = None
    actual_context_dim = 0
    if context_train is not None:
        coarse_context_train = context_from_coarse_trajectory(
            coarse_train,
            eigencov=cfg.eigencov,
            n_eigcov_modes=cfg.n_eigcov_modes,
            window=cfg.coarse_context_window,
        )
        context_mean, context_std = compute_data_statistics(np.concatenate([context_train, coarse_context_train], axis=0))
        context_train_norm = normalize(context_train, context_mean, context_std)
        coarse_context_train_norm = normalize(coarse_context_train, context_mean, context_std)
        actual_context_dim = context_train_norm.shape[1]
        print(f"     FiLM context: dim={actual_context_dim}, dropout={cfg.context_dropout_prob:.2f}")
    else:
        coarse_context_train_norm = None

    # Create windows
    train_obs_w, _ = make_windows(nudged_train_norm, None, cfg.train_sequence_length)
    train_target_w, _ = make_windows(reference_train_norm, None, cfg.train_sequence_length)
    train_context_w = None if context_train_norm is None else make_windows(context_train_norm, None, cfg.train_sequence_length)[0]
    if cfg.include_coarse_training_inputs:
        coarse_obs_w, _ = make_windows(coarse_train_norm, None, cfg.train_sequence_length)
        coarse_target_w, _ = make_windows(reference_train_norm, None, cfg.train_sequence_length)
        train_obs_w = np.concatenate([train_obs_w, coarse_obs_w], axis=0)
        train_target_w = np.concatenate([train_target_w, coarse_target_w], axis=0)
        if train_context_w is not None:
            coarse_context_w = make_windows(coarse_context_train_norm, None, cfg.train_sequence_length)[0]
            train_context_w = np.concatenate([train_context_w, coarse_context_w], axis=0)
        print("     training inputs: nudged and coarse")
    else:
        print("     training inputs: nudged")
    print(f"     train windows: {train_obs_w.shape[0]}")

    # ============================================================
    # [2] Diagnostics: train vs test distribution
    # ============================================================
    print(f"\n[2/6] Distribution diagnostics...")

    nudged_vs_ref_mse_unnorm = float(np.mean(np.square(nudged_train - reference_train)))
    print(f"     MSE(nudged, reference) [train task, unnorm]: {nudged_vs_ref_mse_unnorm:.6f}")

    # Load coarse test data only. The reference is loaded after correction.
    time_test_raw, coarse_test_raw = load_test_coarse_data(cfg)
    test_stop = min(cfg.test_interval_length, coarse_test_raw.shape[0])
    coarse_test = coarse_test_raw[:test_stop:stride]
    print(f"     coarse test points: {coarse_test.shape[0]}")

    # z3 tail statistics
    z3_tail_rate_nudged = float(np.mean(np.abs(nudged_train[:, 2]) > z3_tail_threshold_unnorm))
    z3_tail_rate_coarse = float(np.mean(np.abs(coarse_test[:, 2]) > z3_tail_threshold_unnorm))
    print(f"     z3 tail exceedance rates (|z3| > {z3_tail_threshold_unnorm:.3f}):")
    print(f"       nudged_train: {z3_tail_rate_nudged:.4f}")
    print(f"       coarse_test: {z3_tail_rate_coarse:.4f}")

    # ============================================================
    # [3] Build and train model
    # ============================================================
    print(f"\n[3/6] Building residual {cfg.model_architecture} corrector...")
    if cfg.model_architecture == "attention":
        model = ResidualAttentionCorrector(
            state_dim=cfg.state_dim,
            hidden_dim=cfg.hidden_dim,
            context_dim=actual_context_dim,
            film_hidden_dim=cfg.film_hidden_dim,
            gate_init_bias=cfg.gate_init_bias,
            num_heads=cfg.attention_heads,
            num_layers=cfg.attention_layers,
            ffn_multiplier=cfg.ffn_multiplier,
            dropout_rate=cfg.dropout_rate,
        )
    elif cfg.model_architecture == "tcn":
        model = ResidualTCNCorrector(
            state_dim=cfg.state_dim,
            hidden_dim=cfg.hidden_dim,
            context_dim=actual_context_dim,
            film_hidden_dim=cfg.film_hidden_dim,
            gate_init_bias=cfg.gate_init_bias,
        )
    else:
        raise ValueError("model_architecture must be 'attention' or 'tcn'")
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

    # Train/val split
    n_windows = train_obs_w.shape[0]
    n_val = max(1, int(0.1 * n_windows))
    n_train_act = n_windows - n_val

    train_obs = train_obs_w[:n_train_act]
    train_target = train_target_w[:n_train_act]
    train_context = None if train_context_w is None else train_context_w[:n_train_act]
    val_obs = train_obs_w[n_train_act:]
    val_target = train_target_w[n_train_act:]
    val_context = None if train_context_w is None else train_context_w[n_train_act:]

    if train_context is None:
        train_ds = tf.data.Dataset.from_tensor_slices((train_obs, train_target))
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((train_obs, train_target, train_context))
    train_ds = train_ds.shuffle(min(4000, len(train_obs))).batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"     train split: {n_train_act}, val split: {n_val}")
    print("\n     Training with hybrid loss:")
    print(f"       lambda_delta = {cfg.lambda_delta} (residual regularization)")
    print(f"       lambda_moment = {cfg.lambda_moment} (mean/std matching)")
    print(f"       lambda_tail = {cfg.lambda_tail} (tail exceedance)")

    history = []
    best_val_score = float("inf")
    best_val_mse = float("inf")
    best_val_tail_mse = float("inf")
    best_epoch = 0
    best_weights = None

    for epoch in range(cfg.epochs):
        losses_epoch = []
        loss_dict_agg = {"L_path": 0, "L_delta": 0, "L_moment": 0, "L_tail": 0}

        for batch in train_ds:
            if train_context is None:
                obs_b, target_b = batch
                c_b = None
            else:
                obs_b, target_b, c_b = batch
                if cfg.context_dropout_prob > 0:
                    keep = tf.cast(
                        tf.random.uniform((tf.shape(c_b)[0], 1, 1)) >= cfg.context_dropout_prob,
                        c_b.dtype,
                    )
                    c_b = c_b * keep
            with tf.GradientTape() as tape:
                total_loss, loss_dict = model.sequence_loss(
                    obs_b,
                    target_b,
                    c_b,
                    z3_tail_threshold=z3_tail_threshold_norm,
                    lambda_delta=cfg.lambda_delta,
                    lambda_moment=cfg.lambda_moment,
                    lambda_tail=cfg.lambda_tail,
                    tail_weight_factor=cfg.tail_weight_factor,
                    state_loss_weights=cfg.state_loss_weights,
                    tail_exceedance_percentiles=[50, 75, 90, 95, 99],
                    tail_temperature=cfg.tail_exceedance_temperature,
                )

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            losses_epoch.append(total_loss.numpy())
            for k in loss_dict_agg:
                loss_dict_agg[k] += loss_dict[k]

        train_loss = np.mean(losses_epoch)
        for k in loss_dict_agg:
            loss_dict_agg[k] /= len(losses_epoch)

        # Validation: average MSE plus a z3-tail score for checkpoint selection.
        val_mses = []
        val_tail_mses = []
        if val_context is None:
            val_ds = tf.data.Dataset.from_tensor_slices((val_obs, val_target)).batch(256)
        else:
            val_ds = tf.data.Dataset.from_tensor_slices((val_obs, val_target, val_context)).batch(256)
        for batch in val_ds:
            if val_context is None:
                obs_b, target_b = batch
                c_b = None
            else:
                obs_b, target_b, c_b = batch
            pred = model(obs_b, c_b)
            mse = tf.reduce_mean(tf.square(pred - target_b)).numpy()
            val_mses.append(mse)
            tail_mask = tf.abs(target_b[..., 2]) > z3_tail_threshold_norm
            tail_err = tf.square(pred[..., 2] - target_b[..., 2])
            tail_vals = tf.boolean_mask(tail_err, tail_mask)
            if int(tf.size(tail_vals).numpy()) > 0:
                val_tail_mses.append(float(tf.reduce_mean(tail_vals).numpy()))
        val_mse = np.mean(val_mses)
        val_tail_mse = float(np.mean(val_tail_mses)) if val_tail_mses else float(val_mse)
        val_score = float(val_mse + cfg.tail_loss_weight * val_tail_mse)

        if val_score < best_val_score:
            best_val_score = val_score
            best_val_mse = val_mse
            best_val_tail_mse = val_tail_mse
            best_epoch = epoch + 1
            best_weights = model.get_weights()

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "L_path": loss_dict_agg["L_path"],
            "L_delta": loss_dict_agg["L_delta"],
            "L_moment": loss_dict_agg["L_moment"],
            "L_tail": loss_dict_agg["L_tail"],
            "val_mse": float(val_mse),
            "val_tail_mse": float(val_tail_mse),
            "val_score": float(val_score),
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:03d}/{cfg.epochs} | train={train_loss:.6f} "
                f"L_path={loss_dict_agg['L_path']:.6f} L_delta={loss_dict_agg['L_delta']:.6f} "
                f"L_moment={loss_dict_agg['L_moment']:.6f} | val_mse={val_mse:.6f} "
                f"val_tail_mse={val_tail_mse:.6f} val_score={val_score:.6f}"
            )

    if best_weights is not None:
        model.set_weights(best_weights)
        print(
            f"\n     Best epoch: {best_epoch} "
            f"(val_score={best_val_score:.6f}, val_mse={best_val_mse:.6f}, "
            f"val_tail_mse={best_val_tail_mse:.6f})"
        )

    # ============================================================
    # [4] Correct coarse test trajectory
    # ============================================================
    print(f"\n[4/6] Correcting coarse test trajectory...")

    coarse_test_norm = normalize(coarse_test, data_mean, data_std)
    time_test = time_test_raw[:test_stop:stride]
    context_test_norm = None
    if actual_context_dim > 0:
        context_test = context_from_coarse_trajectory(
            coarse_test,
            eigencov=cfg.eigencov,
            n_eigcov_modes=cfg.n_eigcov_modes,
            window=cfg.coarse_context_window,
        )
        context_test_norm = normalize(context_test, context_mean, context_std)
        print("     Test FiLM context: coarse-derived rolling statistics")

    # Sliding-window inference
    corrected_test_norm = apply_sliding_window_inference(
        model,
        coarse_test_norm,
        window_length=cfg.train_sequence_length,
        context_sequence=context_test_norm,
        overlap_ratio=cfg.inference_window_overlap,
    )
    corrected_test = denormalize(corrected_test_norm, data_mean, data_std)
    corrected_test_before_calibration = corrected_test.copy()

    state_source_q = []
    state_target_q = []
    if cfg.state_quantile_calibration:
        calibration_input_norm = coarse_train_norm if cfg.include_coarse_training_inputs else nudged_train_norm
        calibration_context_norm = None
        if context_train_norm is not None:
            calibration_context_norm = coarse_context_train_norm if cfg.include_coarse_training_inputs else context_train_norm

        train_pred_norm = apply_sliding_window_inference(
            model,
            calibration_input_norm,
            window_length=cfg.train_sequence_length,
            context_sequence=calibration_context_norm,
            overlap_ratio=cfg.inference_window_overlap,
        )
        train_pred = denormalize(train_pred_norm, data_mean, data_std)
        for dim in range(cfg.state_dim):
            source_q, target_q = fit_quantile_map(
                train_pred[:, dim],
                reference_train[:, dim],
                cfg.quantile_grid_size,
            )
            corrected_test[:, dim] = apply_quantile_map(corrected_test[:, dim], source_q, target_q)
            state_source_q.append(source_q)
            state_target_q.append(target_q)
        print("     Applied training-only state quantile calibration")

    print(f"     Corrected test trajectory shape: {corrected_test.shape}")

    # ============================================================
    # [5] Load reference for diagnostics only
    # ============================================================
    print(f"\n[5/6] Loading reference for diagnostics...")
    _, reference_test_raw = load_test_reference_data(cfg)
    reference_test = reference_test_raw[:test_stop:stride]
    coarse_vs_ref_mse_unnorm = float(np.mean(np.square(coarse_test - reference_test)))
    z3_tail_rate_ref = float(np.mean(np.abs(reference_test[:, 2]) > z3_tail_threshold_unnorm))
    print(f"     MSE(coarse, reference) [diagnostic]: {coarse_vs_ref_mse_unnorm:.6f}")
    print(f"     reference_test z3 tail rate: {z3_tail_rate_ref:.4f}")

    # Compute PDF metrics
    def compute_pdf_tail_l1(ref, pred, percentile=90, var_idx=2):
        """Compute log-PDF L1 mismatch in tail region."""
        ref_var = ref[:, var_idx]
        pred_var = pred[:, var_idx]
        threshold = np.percentile(np.abs(ref_var), percentile)

        x_lo = min(float(np.quantile(ref_var, 0.001)), float(np.quantile(pred_var, 0.001)))
        x_hi = max(float(np.quantile(ref_var, 0.999)), float(np.quantile(pred_var, 0.999)))
        pad = 0.1 * max(x_hi - x_lo, 1e-8)
        grid = np.linspace(x_lo - pad, x_hi + pad, 600)

        p_ref = np.maximum(kde_pdf_on_grid(ref_var, grid), 1e-12)
        p_pred = np.maximum(kde_pdf_on_grid(pred_var, grid), 1e-12)

        tail_mask = np.abs(grid) > threshold
        if not np.any(tail_mask):
            return 0.0
        return float(np.mean(np.abs(np.log(p_ref[tail_mask]) - np.log(p_pred[tail_mask]))))

    pdf_tail_coarse_unnorm = compute_pdf_tail_l1(reference_test, coarse_test, percentile=90)
    pdf_tail_corr_unnorm = compute_pdf_tail_l1(reference_test, corrected_test, percentile=90)

    print(f"     PDF tail log-L1 (90th percentile):")
    print(f"       coarse vs ref: {pdf_tail_coarse_unnorm:.6f}")
    print(f"       corrected vs ref: {pdf_tail_corr_unnorm:.6f}")
    corrected_test_final = corrected_test.copy()

    # ============================================================
    # [6] Save results and plots
    # ============================================================
    print(f"\n[6/6] Saving results...")

    # Checkpoints and history
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt.write(str(out_dir / "lsde_film_ckpt"))
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    # Summary
    summary = {
        "dataset": str(Path(cfg.dataset_npz).resolve()),
        "result_dir": str(out_dir),
        "repeat_index": repeat_index_from_output_dir(out_dir),
        "model_kind": cfg.model_kind,
        "architecture": cfg.model_architecture,
        "n_ensemble_members_used": int(cfg.n_ensemble_members_used),
        "eigencov": bool(cfg.eigencov),
        "n_eigcov_modes": int(cfg.n_eigcov_modes),
        "include_coarse_training_inputs": bool(cfg.include_coarse_training_inputs),
        "train_sequence_length": cfg.train_sequence_length,
        "train_time": cfg.train_time,
        "test_time": cfg.test_time,
        "ml_stride": cfg.ml_stride,
        "inference_window_overlap": cfg.inference_window_overlap,
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "best_val_mse": float(best_val_mse),
        "best_val_tail_mse": float(best_val_tail_mse),
        "data_mean": data_mean.tolist(),
        "data_std": data_std.tolist(),
        "z3_tail_threshold_unnorm": float(z3_tail_threshold_unnorm),
        "z3_tail_threshold_norm": float(z3_tail_threshold_norm),
        "lambda_delta": cfg.lambda_delta,
        "lambda_moment": cfg.lambda_moment,
        "lambda_tail": cfg.lambda_tail,
        "state_loss_weights": list(cfg.state_loss_weights),
        "state_quantile_calibration": bool(cfg.state_quantile_calibration),
        "quantile_grid_size": int(cfg.quantile_grid_size),
        "context_dim": int(actual_context_dim),
        "context_dropout_prob": cfg.context_dropout_prob,
        "test_context_source": "coarse_rolling" if actual_context_dim > 0 else "none",
        "coarse_context_window": int(cfg.coarse_context_window),
        "gate_init_bias": cfg.gate_init_bias,
        "metrics": {
            "mse_nudged_vs_ref_train_unnorm": float(nudged_vs_ref_mse_unnorm),
            "mse_coarse_vs_ref_test_unnorm": float(coarse_vs_ref_mse_unnorm),
            "z3_tail_rate_nudged": float(z3_tail_rate_nudged),
            "z3_tail_rate_coarse": float(z3_tail_rate_coarse),
            "z3_tail_rate_ref": float(z3_tail_rate_ref),
            "pdf_tail_l1_coarse": float(pdf_tail_coarse_unnorm),
            "pdf_tail_l1_corrected": float(pdf_tail_corr_unnorm),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plots
    plot_three_way_comparison(time_test, coarse_test, corrected_test_final, reference_test, out_dir / "three_way_comparison.png")
    plot_log_pdfs_3way(reference_test, coarse_test, corrected_test_final, out_dir)
    plot_multi_threshold_aot(time_test, reference_test, coarse_test, corrected_test_final, out_dir, var_idx=2, percentiles=[50, 75, 90, 95, 99])
    plot_3d_comparison(reference_test, coarse_test, corrected_test_final, out_dir)

    if state_source_q:
        max_q_len = max(len(q) for q in state_source_q)
        state_source_q_arr = np.full((len(state_source_q), max_q_len), np.nan, dtype=np.float32)
        state_target_q_arr = np.full((len(state_target_q), max_q_len), np.nan, dtype=np.float32)
        for dim, (source_q, target_q) in enumerate(zip(state_source_q, state_target_q)):
            state_source_q_arr[dim, : len(source_q)] = source_q
            state_target_q_arr[dim, : len(target_q)] = target_q
    else:
        state_source_q_arr = np.empty((0, 0), dtype=np.float32)
        state_target_q_arr = np.empty((0, 0), dtype=np.float32)

    np.savez(
        out_dir / "corrected_test_outputs.npz",
        time=time_test,
        reference=reference_test,
        coarse=coarse_test,
        corrected=corrected_test_final,
        corrected_before_calibration=corrected_test_before_calibration,
        state_source_quantiles=state_source_q_arr,
        state_target_quantiles=state_target_q_arr,
    )

    print(f"\nSaved low-dimensional corrector results to: {out_dir}")
    print(f"  - summary.json")
    print(f"  - training_history.csv (hybrid loss components)")
    print(f"  - Visualizations: 3D, PDF, AOT, time series")
    print(f"  - multi_threshold_aot_z3.png")
    print(f"  - 3d_comparison.png")
    print(f"  - summary.json (includes data_mean, data_std)")
