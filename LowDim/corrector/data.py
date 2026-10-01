from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import CorrectorConfig


def _interpolate_reference_to_coarse(time_ref: np.ndarray, ref: np.ndarray, time_coarse: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(time_coarse, time_ref, ref[:, d]) for d in range(ref.shape[1])])


def _context_from_ensemble_mean_std(ensemble_tm3: np.ndarray) -> np.ndarray:
    """
    Context from mean, std, and top-2 eigenvalues of ensemble.
    ensemble_tm3: (T, M, 3)
    Returns: (T, 8) where columns are [mean_z1, mean_z2, mean_z3, std_z1, std_z2, std_z3, eig1, eig2]
    """
    mean = np.mean(ensemble_tm3, axis=1)  # (T, 3)
    std = np.std(ensemble_tm3, axis=1)    # (T, 3)

    t_len = ensemble_tm3.shape[0]
    eig12 = np.zeros((t_len, 2), dtype=float)
    for t in range(t_len):
        q = ensemble_tm3[t] - mean[t]
        cov = (q.T @ q) / max(q.shape[0] - 1, 1)
        vals = np.linalg.eigvalsh(cov)[::-1]
        eig12[t, 0] = vals[0]
        eig12[t, 1] = vals[1] if vals.size > 1 else vals[0]

    return np.concatenate([mean, std, eig12], axis=1)  # (T, 8)


def _context_from_ensemble_eigenvectors(ensemble_tm3: np.ndarray, n_eigcov_modes: int) -> np.ndarray:
    """
    Context from top-n eigenvectors of ensemble covariance.
    ensemble_tm3: (T, M, 3)
    n_eigcov_modes: number of top eigenvectors to include
    Returns: (T, 3*n_eigcov_modes) where each column is an eigenvector component
    """
    t_len = ensemble_tm3.shape[0]
    state_dim = ensemble_tm3.shape[2]  # 3
    context = np.zeros((t_len, state_dim * n_eigcov_modes), dtype=float)

    previous_modes = None
    for t in range(t_len):
        mean_t = np.mean(ensemble_tm3[t], axis=0)
        q = ensemble_tm3[t] - mean_t
        cov = (q.T @ q) / max(q.shape[0] - 1, 1)
        _, eigvecs = np.linalg.eigh(cov)
        eigvecs = eigvecs[:, ::-1]
        eigvecs = eigvecs / np.maximum(np.linalg.norm(eigvecs, axis=0, keepdims=True), 1e-14)

        n_modes = min(n_eigcov_modes, eigvecs.shape[1])
        if previous_modes is not None:
            for i in range(n_modes):
                if float(np.dot(previous_modes[:, i], eigvecs[:, i])) < 0.0:
                    eigvecs[:, i] *= -1.0

        for i in range(n_modes):
            context[t, i*state_dim:(i+1)*state_dim] = eigvecs[:, i]

        previous_modes = eigvecs[:, :n_modes].copy()

    return context


def context_from_coarse_trajectory(
    coarse_t3: np.ndarray,
    eigencov: bool = False,
    n_eigcov_modes: int = 2,
    window: int = 101,
) -> np.ndarray:
    """
    Build a FiLM context from the coarse trajectory only.

    The FiLM model is trained with ensemble-derived context, but test-time
    conditioning is a surrogate computed from the coarse input. No saved test
    ensemble, nudged trajectory, or reference trajectory is used.
    """
    z = np.asarray(coarse_t3, dtype=np.float32)
    t_len, state_dim = z.shape
    half_window = max(1, int(window) // 2)

    if eigencov:
        context = np.zeros((t_len, state_dim * n_eigcov_modes), dtype=np.float32)
    else:
        context = np.zeros((t_len, 8), dtype=np.float32)

    for t in range(t_len):
        i0 = max(0, t - half_window)
        i1 = min(t_len, t + half_window + 1)
        block = z[i0:i1]
        mean = np.mean(block, axis=0)
        centered = block - mean
        cov = (centered.T @ centered) / max(block.shape[0] - 1, 1)

        if eigencov:
            _, eigvecs = np.linalg.eigh(cov)
            eigvecs = eigvecs[:, ::-1]
            eigvecs = eigvecs / np.maximum(np.linalg.norm(eigvecs, axis=0, keepdims=True), 1e-14)
            n_modes = min(n_eigcov_modes, state_dim)
            if t > 0:
                previous = context[t - 1].reshape(-1, state_dim).T
                for mode in range(n_modes):
                    if float(np.dot(previous[:, mode], eigvecs[:, mode])) < 0.0:
                        eigvecs[:, mode] *= -1.0
            for mode in range(n_modes):
                j0 = mode * state_dim
                context[t, j0:j0 + state_dim] = eigvecs[:, mode]
        else:
            std = np.std(block, axis=0)
            vals = np.linalg.eigvalsh(cov)[::-1]
            eig1 = vals[0] if vals.size else 0.0
            eig2 = vals[1] if vals.size > 1 else eig1
            context[t] = np.concatenate([mean, std, np.array([eig1, eig2], dtype=np.float32)])

    return context.astype(np.float32)


def _load_run_data(
    cfg: CorrectorConfig,
    run_id: int,
    input_source: str,
    target_source: str,
    load_context: bool = True,
 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Load data from a single run.

    Args:
        cfg: Config object
        run_id: Which run to load (0, 1, etc.)
        input_source: 'nudged' or 'coarse'
        target_source: 'reference' (for both train and test)

    Returns:
        (time, input_data, target_data, context)
    """
    data = np.load(cfg.dataset_npz)

    time_coarse = data["time_coarse"]

    # Load ensemble only when conditioning is enabled.
    ensemble_tm3 = None
    if load_context and cfg.n_ensemble_members_used > 0:
        ensemble_raw = data["ensemble"][run_id]  # (M, T, 3)
        m_use = min(cfg.n_ensemble_members_used, ensemble_raw.shape[0])
        ensemble_sel = ensemble_raw[:m_use]
        ensemble_tm3 = np.transpose(ensemble_sel, (1, 0, 2))  # (T, M, 3)

    # Get reference for alignment
    reference = data["reference"][run_id]
    time_reference = data["time_reference"]
    if reference.shape[0] != time_coarse.shape[0] or not np.allclose(time_reference, time_coarse):
        reference = _interpolate_reference_to_coarse(time_reference, reference, time_coarse)

    # Load input
    if input_source == "nudged":
        input_data = data["nudged"][run_id]
    elif input_source == "coarse":
        input_data = data["coarse"][run_id]
    else:
        raise ValueError(f"input_source must be 'nudged' or 'coarse', got {input_source}")

    # Load target
    if target_source == "reference":
        target_data = reference
    else:
        raise ValueError(f"target_source must be 'reference', got {target_source}")

    # Compute context only if requested; baseline returns None.
    if ensemble_tm3 is None:
        context = None
    elif cfg.eigencov:
        context = _context_from_ensemble_eigenvectors(ensemble_tm3, cfg.n_eigcov_modes)
    else:
        context = _context_from_ensemble_mean_std(ensemble_tm3)

    return time_coarse, input_data.astype(np.float32), target_data.astype(np.float32), None if context is None else context.astype(np.float32)


def load_train_data(cfg: CorrectorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training data: input=nudged, target=reference from run_id_train."""
    return _load_run_data(cfg, cfg.run_id_train, input_source="nudged", target_source="reference", load_context=True)


def load_train_coarse_data(cfg: CorrectorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Load training coarse/reference data with no ensemble context."""
    return _load_run_data(cfg, cfg.run_id_train, input_source="coarse", target_source="reference", load_context=False)


def load_test_coarse_data(cfg: CorrectorConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load only the coarse test trajectory."""
    data = np.load(cfg.dataset_npz)
    return data["time_coarse"], data["coarse"][cfg.run_id_test].astype(np.float32)


def load_test_reference_data(cfg: CorrectorConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load the reference test trajectory for diagnostics after correction."""
    data = np.load(cfg.dataset_npz)
    time_coarse = data["time_coarse"]
    time_reference = data["time_reference"]
    reference = data["reference"][cfg.run_id_test]
    if reference.shape[0] != time_coarse.shape[0] or not np.allclose(time_reference, time_coarse):
        reference = _interpolate_reference_to_coarse(time_reference, reference, time_coarse)
    return time_coarse, reference.astype(np.float32)




def make_windows(z: np.ndarray, c: np.ndarray | None, seq_len: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Create sliding windows of sequences."""
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    n = z.shape[0] - seq_len + 1
    if n <= 0:
        raise ValueError("sequence length too long for provided interval")

    z_w = np.stack([z[i : i + seq_len] for i in range(n)], axis=0)
    if c is None:
        return z_w.astype(np.float32), None

    c_w = np.stack([c[i : i + seq_len] for i in range(n)], axis=0)
    return z_w.astype(np.float32), c_w.astype(np.float32)


def compute_data_statistics(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std for normalization.

    Args:
        data: Shape (T, state_dim)

    Returns:
        mean, std: Shape (state_dim,)
    """
    mean = np.mean(data, axis=0, keepdims=False).astype(np.float32)
    std = np.std(data, axis=0, keepdims=False).astype(np.float32)
    std = np.maximum(std, 1e-8)  # Avoid division by zero
    return mean, std


def normalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Normalize data using provided mean and std.

    Args:
        data: Shape (T, state_dim) or (B, T, state_dim)
        mean: Shape (state_dim,)
        std: Shape (state_dim,)

    Returns:
        normalized: Same shape as data
    """
    return ((data - mean) / std).astype(np.float32)


def denormalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Denormalize data using provided mean and std.

    Args:
        data: Shape (T, state_dim) or (B, T, state_dim)
        mean: Shape (state_dim,)
        std: Shape (state_dim,)

    Returns:
        denormalized: Same shape as data
    """
    return (data * std + mean).astype(np.float32)


def compute_tail_thresholds(
    reference_unnormalized: np.ndarray,
    percentiles: list[float] | None = None,
    tail_threshold_std: float = 2.0,
) -> dict[str, float]:
    """
    Compute fixed tail thresholds from unnormalized training reference.

    Args:
        reference_unnormalized: Shape (T, state_dim), unnormalized reference trajectory
        percentiles: List of percentiles [50, 75, 90, 95, 99] for exceedance tracking
        tail_threshold_std: Threshold in units of std of z3

    Returns:
        Dict with:
            'z3_threshold_std': Main threshold = tail_threshold_std * std(z3)
            'z3_q{pctl}': Percentile thresholds of |z3|, e.g., 'z3_q90', 'z3_q95', etc.
    """
    if percentiles is None:
        percentiles = [50, 75, 90, 95, 99]

    z3 = reference_unnormalized[:, 2]
    z3_std = float(np.std(z3))
    z3_threshold = tail_threshold_std * z3_std

    result = {"z3_threshold_std": z3_threshold, "z3_std": z3_std}

    for pctl in percentiles:
        q = np.percentile(np.abs(z3), pctl)
        result[f"z3_q{pctl:.0f}"] = float(q)

    return result


def apply_sliding_window_inference(
    model,
    full_sequence: np.ndarray,
    window_length: int,
    context_sequence: np.ndarray | None = None,
    overlap_ratio: float = 0.5,
) -> np.ndarray:
    """
    Apply model to full sequence using sliding windows with overlap-averaging.

    Args:
        model: Keras model with call() method
        full_sequence: Shape (T, state_dim), full test sequence
        window_length: Length of training windows (e.g., 128)
        overlap_ratio: Overlap between windows, e.g., 0.5 for 50%

    Returns:
        corrected: Shape (T, state_dim), windowed predictions averaged where overlapping
    """
    import tensorflow as tf

    T, D = full_sequence.shape
    stride = max(1, int(window_length * (1.0 - overlap_ratio)))

    if T <= window_length:
        window_batched = tf.constant(full_sequence[None, :, :], dtype=tf.float32)
        if context_sequence is None:
            return model(window_batched).numpy()[0].astype(np.float32)
        context_batched = tf.constant(context_sequence[None, :, :], dtype=tf.float32)
        return model(window_batched, context_batched).numpy()[0].astype(np.float32)

    # Initialize output with counts for averaging
    corrected = np.zeros_like(full_sequence)
    counts = np.zeros(T, dtype=np.float32)

    # Slide windows
    for start_idx in range(0, T - window_length + 1, stride):
        end_idx = start_idx + window_length
        window = full_sequence[start_idx:end_idx]  # (window_length, D)

        # Predict on this window
        window_batched = tf.constant(window[None, :, :], dtype=tf.float32)
        if context_sequence is None:
            pred_window = model(window_batched).numpy()[0]  # (window_length, D)
        else:
            context_window = context_sequence[start_idx:end_idx]
            context_batched = tf.constant(context_window[None, :, :], dtype=tf.float32)
            pred_window = model(window_batched, context_batched).numpy()[0]  # (window_length, D)

        # Accumulate into output
        corrected[start_idx:end_idx] += pred_window
        counts[start_idx:end_idx] += 1.0

    # Ensure the final samples are covered even when T does not align with stride.
    if np.any(counts == 0):
        last_start = T - window_length
        remaining_window = full_sequence[last_start:T]
        remaining_batched = tf.constant(remaining_window[None, :, :], dtype=tf.float32)
        if context_sequence is None:
            pred_remaining = model(remaining_batched).numpy()[0]
        else:
            remaining_context = context_sequence[last_start:T]
            remaining_context_batched = tf.constant(remaining_context[None, :, :], dtype=tf.float32)
            pred_remaining = model(remaining_batched, remaining_context_batched).numpy()[0]
        corrected[last_start:T] += pred_remaining
        counts[last_start:T] += 1.0

    # Average
    corrected = corrected / np.maximum(counts[:, None], 1e-8)
    return corrected.astype(np.float32)


def save_run_metadata(cfg: CorrectorConfig, out_dir: Path) -> None:
    """Save configuration to metadata file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = asdict(cfg)
    cfg_dict["resolved_output_dir"] = str(out_dir)
    repeat_match = re.search(r"_rep(\d+)$", out_dir.name)
    cfg_dict["resolved_repeat_index"] = int(repeat_match.group(1)) if repeat_match else None
    (out_dir / "config.txt").write_text(
        "\n".join([f"{k}: {v}" for k, v in cfg_dict.items()]),
        encoding="utf-8"
    )
