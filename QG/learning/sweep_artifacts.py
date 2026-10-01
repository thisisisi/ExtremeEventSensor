# Author: Isabella Thiel <thieli@mit.edu>
"""Pure-path artifact resolution for sweep preflight checks.

This module deliberately avoids TensorFlow/SciPy imports so parameter sweeps
can skip completed work before allocating a Slurm node.
"""

import os
import re
from experiments import experiment_name, artifact_id
from artifact_paths import _repeat_suffix

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _artifact_root(path):
    path = str(path)
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(_MODULE_DIR, path))


def _grid_state_dim(case_name):
    match = re.search(r"_(\d+)x(\d+)(?:_|$)", str(case_name))
    if not match:
        raise ValueError(f"Cannot infer grid dimensions from case name: {case_name}")
    ny, nx = map(int, match.groups())
    return 2 * ny * nx


def _strategy_signature(config):
    ensemble = config.get("ensemble", {}) or {}
    strategy = experiment_name(ensemble.get("strategy", ""))
    n_add = int(ensemble.get("n_ens", 0))
    include_baseline = bool(ensemble.get("include_baseline", True))
    n_members = n_add + (1 if strategy == "pooled" and include_baseline else 0)

    data = config.get("data", {}) or {}
    state_dim = _grid_state_dim(data.get("case_name_train", ""))
    if strategy in {"baseline", "covariance_concat"}:
        n_modes = min(
            int(ensemble.get("n_eigcov_modes", n_add - 1)),
            n_add - 1,
            state_dim,
        )
        extra = {"Dz": n_modes * state_dim}
    elif strategy in {"random_context", "ensemble_mean", "ensemble_variance"}:
        extra = {"Dz": state_dim}
    elif strategy in {"film", "coarse_statistics", "mean_spread_concat"}:
        extra = {"Dz": 4}
    else:
        extra = {}
    return artifact_id(strategy), n_members, extra


def _extra_parts(extra):
    return [f"{key}{value}" for key, value in sorted((extra or {}).items())]


def model_artifact_path(config):
    data = config.get("data", {}) or {}
    training = config.get("training", {}) or {}
    paths = config.get("paths", {}) or {}
    case = data["case_name_train"]
    model = training.get("model_type", "LSTM")
    basis = data.get("basis_type", "rnudge16.0")
    epochs = int(training.get("Nepochs", 2000))
    dt_ml = float(data.get("dt_ml", 0.1))
    ntime = int(data.get("Ntime_units", 1000))
    j_ens = int(training.get("ensemble_index", 1))
    repeat = training.get("repeat_index", None)
    suffix = _repeat_suffix(repeat)
    base = (
        f"{case}_{model}_{basis}_epochs_{epochs}_dt_{dt_ml}_"
        f"T_{ntime}_j{j_ens}"
    )

    ensemble = config.get("ensemble", {}) or {}
    if ensemble.get("enabled", False):
        strategy, n_members, extra = _strategy_signature(config)
        tokens = [f"S{strategy}", f"k{n_members}", *_extra_parts(extra)]
        base += "_" + "-".join(tokens)
    return os.path.join(
        _artifact_root(paths.get("neural_nets_dir", "../Neural_Nets")),
        base + suffix + ".h5",
    )


def result_artifact_path(config):
    data = config.get("data", {}) or {}
    training = config.get("training", {}) or {}
    paths = config.get("paths", {}) or {}
    case = data["case_name_train"]
    model = training.get("model_type", "LSTM")
    basis = data.get("basis_type", "rnudge16.0")
    epochs = int(training.get("Nepochs", 2000))
    dt_ml = float(data.get("dt_ml", 0.1))
    ntime = int(data.get("Ntime_units", 1000))
    n_models = int(training.get("ensemble_index", 1))
    repeat = training.get("repeat_index", None)
    base = (
        f"{case}_{model}_{basis}_loss_L2_epochs_{epochs}_dt_{dt_ml}_"
        f"T_{ntime}_Nens_{n_models}"
    )

    ensemble = config.get("ensemble", {}) or {}
    if ensemble.get("enabled", False):
        strategy, n_members, extra = _strategy_signature(config)
        tokens = [f"S{strategy}", f"k{n_members}"]
        base += "_" + "-".join(tokens)
    if ensemble.get("enabled", False):
        base += _repeat_suffix(repeat) + "_Ensemble_Stats_Big.mat"
    else:
        base += "_Ensemble_Stats_Big" + _repeat_suffix(repeat) + ".mat"
    return os.path.join(
        _artifact_root(paths.get("ml_results_dir", "../ML_Results")),
        base,
    )
