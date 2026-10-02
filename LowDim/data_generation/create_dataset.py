#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create an ML-ready dataset with exact ref/coarse/nudged numerics + BR/BS ensemble.

Deterministic trajectories follow the same setup as the existing scripts:
- reference from otd.py solution_low_dimensional (odeint, t_span=(0,300), n_ref=100000)
- coarse from coarse.py coarse_solution on temporal coarse grid (factor=12)
- nudged from coupled odeint system on coarse grid with tau fixed to 8

Ensemble is added mathematically as specified:
dv_i = [f(v_i) - (1/tau)(v_i - u_bar)]dt + sigma dB_i
with K independent members, stateless index-based Gaussian noise, and
IC perturbations using beta = 2/sigma^2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.integrate import odeint

from .coarse import coarse_solution, coarse_system
from .otd import low_dimensional_2, solution_low_dimensional


@dataclass(frozen=True)
class DatasetConfig:
    t_span: Tuple[float, float] = (0.0, 510.0)
    initial_state: Tuple[float, float, float] = (0.1, -1.0, 0.1)
    n_ref: int = 170_000
    coarse_factor: int = 1
    save_stride: int = 5
    spinup_time: float = 0.0
    tau: float = 8.0

    # Ensemble configuration: change for different datasets
    n_runs: int = 2
    ensemble_size: int = 32
    sigma: float = 0.000005
    analysis_time: float = 0.0

    # Run initialization: same x0 per run across systems, different x0 across runs.
    init_state_std: Tuple[float, float, float] = (0.02, 0.02, 0.02)

    # Model parameters from otd.py / nudging_sweep.py.
    a1: float = 2.0
    a2: float = 2.0
    epsilon: float = 0.05
    b: float = 20.0

    save_float_dtype: str = "float32"


@dataclass(frozen=True)
class ModelParams:
    a1: float = 2.0
    a2: float = 2.0
    epsilon: float = 0.05
    b: float = 20.0


class StatelessGaussian:
    """Stateless Gaussian generator with disjoint IC/TIME key spaces."""

    def __init__(self, seed: int, dim: int = 3) -> None:
        self.seed = int(seed)
        self.dim = int(dim)

    @staticmethod
    def _u64_to_uniform(key: str) -> float:
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return float((np.frombuffer(digest, dtype=np.uint64)[0] + 0.5) / (2**64))

    def sample(self, run_id: int, member_id: int, global_step: int, ic_mode: bool = False) -> np.ndarray:
        prefix = "IC" if ic_mode else "TIME"
        out = np.zeros(self.dim, dtype=float)

        i = 0
        while i < self.dim:
            u1 = max(
                self._u64_to_uniform(f"{prefix}_{self.seed}_{run_id}_{member_id}_{global_step}_{i}"),
                1e-10,
            )
            u2 = self._u64_to_uniform(f"{prefix}_{self.seed}_{run_id}_{member_id}_{global_step}_{i}_pair")
            r = np.sqrt(-2.0 * np.log(u1))
            theta = 2.0 * np.pi * u2
            out[i] = r * np.cos(theta)
            if i + 1 < self.dim:
                out[i + 1] = r * np.sin(theta)
            i += 2
        return out


def print_progress(current: int, total: int, prefix: str = "", width: int = 32) -> None:
    total = max(total, 1)
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "=" * filled + "." * (width - filled)
    percent = 100.0 * ratio
    
    if current % max(5, total // 100) != 0 and current < total:
        return
    end = "\n" if current >= total else "\r"
    print(f"{prefix:<34} [{bar}] {percent:6.2f}%", end=end, flush=True)


class ProgressTracker:
    def __init__(self, total_work: int) -> None:
        self.total_work = max(int(total_work), 1)
        self.done_work = 0

    def advance(self, work_units: int, prefix: str) -> None:
        self.done_work = min(self.total_work, self.done_work + int(work_units))
        print_progress(self.done_work, self.total_work, prefix=prefix)


def reference_solution(
    initial_state: Iterable[float],
    t_eval: np.ndarray,
    p: ModelParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return solution_low_dimensional(list(initial_state), t_eval, p.a1, p.a2, p.epsilon, p.b)


def to_state_array(solution: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return np.column_stack(solution)


def coarse_time_grid(t_span: Tuple[float, float], n_ref: int, factor: int) -> np.ndarray:
    n_coarse = max(2, n_ref // factor)
    return np.linspace(t_span[0], t_span[1], n_coarse)


def integration_work_units(n_ref: int, n_coarse: int, n_runs: int, ensemble_size: int, analysis_index: int) -> int:
    ref_work = max(n_ref - 1, 1)
    coarse_work = max(n_coarse - 1, 1)
    nudged_work = max(n_coarse - 1, 1)
    ensemble_work = max(n_coarse - 1 - analysis_index, 1) * max(ensemble_size, 1)
    return n_runs * (ref_work + coarse_work + nudged_work + ensemble_work)


def nudged_system(
    full_state: np.ndarray,
    t: float,
    p: Dict[str, float],
    low_dimensional_rhs,
    model_params: ModelParams,
    tau: float,
) -> np.ndarray:
    gain = 1.0 / max(tau, 1e-12)
    x_n = np.asarray(full_state[:3], dtype=float)
    x_r = np.asarray(full_state[3:], dtype=float)
    dx_n = np.asarray(coarse_system(x_n, t, p), dtype=float) + gain * (x_r - x_n)
    dx_r = np.asarray(
        low_dimensional_rhs(x_r, t, model_params.a1, model_params.a2, model_params.epsilon, model_params.b),
        dtype=float,
    )
    return np.concatenate((dx_n, dx_r))


def nudged_solution(
    initial_state: Iterable[float],
    t_eval: np.ndarray,
    p: Dict[str, float],
    low_dimensional_rhs,
    model_params: ModelParams,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray]:
    initial = np.asarray(list(initial_state), dtype=float)
    coupled_initial = np.concatenate((initial, initial))
    sol = odeint(nudged_system, coupled_initial, t_eval, args=(p, low_dimensional_rhs, model_params, tau))
    return sol[:, :3], sol[:, 3:]


def generate_ensemble(
    nudged_states: np.ndarray,
    t_eval: np.ndarray,
    coarse_p: Dict[str, float],
    tau: float,
    sigma: float,
    analysis_index: int,
    run_id: int,
    ensemble_size: int,
    noise_gen: StatelessGaussian,
    progress: ProgressTracker | None = None,
    progress_prefix: str = "",
) -> np.ndarray:
    """Euler-Maruyama co-integrated ensemble around the nudged trajectory."""
    n_steps = nudged_states.shape[0]
    ensemble = np.full((ensemble_size, n_steps, 3), np.nan, dtype=float)
    dt = float(t_eval[1] - t_eval[0])

    if sigma > 0.0:
        beta = 2.0 / (sigma**2)
        beta_inv = 1.0 / beta
        ic_std = np.sqrt(beta_inv)
    else:
        base_rms = float(np.sqrt(np.mean(nudged_states[analysis_index] ** 2)))
        ic_std = 1e-6 * max(base_rms, 1e-6)

    members = np.zeros((ensemble_size, 3), dtype=float)

    for k in range(ensemble_size):
        ic_noise = noise_gen.sample(run_id=run_id, member_id=k + 1, global_step=analysis_index, ic_mode=True)
        members[k] = nudged_states[analysis_index] + ic_std * ic_noise
        ensemble[k, analysis_index] = members[k]

    for n in range(analysis_index, n_steps - 1):
        tt = float(t_eval[n])
        u_bar = nudged_states[n]
        for k in range(ensemble_size):
            drift = np.asarray(coarse_system(members[k], tt, coarse_p), dtype=float) - (members[k] - u_bar) / tau
            if sigma > 0.0:
                zeta = noise_gen.sample(run_id=run_id, member_id=k + 1, global_step=n + 1, ic_mode=False)
            else:
                zeta = np.zeros(3, dtype=float)
            members[k] = members[k] + dt * drift + sigma * np.sqrt(dt) * zeta
            ensemble[k, n + 1] = members[k]
            if progress is not None:
                progress.advance(1, prefix=progress_prefix)

    return ensemble



def truncate_for_save(
    ref_t: np.ndarray,
    coarse_t: np.ndarray,
    reference: np.ndarray,
    coarse: np.ndarray,
    nudged: np.ndarray,
    ensemble: np.ndarray,
    cutoff_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    ref_start = int(np.searchsorted(ref_t, cutoff_time, side="left"))
    coarse_start = int(np.searchsorted(coarse_t, cutoff_time, side="left"))

    ref_start = max(0, min(ref_start, ref_t.size - 1))
    coarse_start = max(0, min(coarse_start, coarse_t.size - 1))

    ref_t_trim = ref_t[ref_start:]
    coarse_t_trim = coarse_t[coarse_start:]
    reference_trim = reference[:, ref_start:, :]
    coarse_trim = coarse[:, coarse_start:, :]
    nudged_trim = nudged[:, coarse_start:, :]
    ensemble_trim = ensemble[:, :, coarse_start:, :]

    return (
        ref_t_trim,
        coarse_t_trim,
        reference_trim,
        coarse_trim,
        nudged_trim,
        ensemble_trim,
        ref_start,
        coarse_start,
    )


def save_dataset(
    output_dir: Path,
    cfg: DatasetConfig,
    master_seed: int,
    ref_t: np.ndarray,
    coarse_t: np.ndarray,
    initial_states: np.ndarray,
    reference: np.ndarray,
    coarse: np.ndarray,
    nudged: np.ndarray,
    ensemble: np.ndarray,
    analysis_index: int,
    cutoff_time: float,
    ref_start: int,
    coarse_start: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.float32 if cfg.save_float_dtype == "float32" else np.float64

    np.savez_compressed(
        output_dir / "trajectories.npz",
        time_reference=ref_t.astype(dtype),
        time_coarse=coarse_t.astype(dtype),
        initial_states=initial_states.astype(dtype),
        reference=reference.astype(dtype),
        coarse=coarse.astype(dtype),
        nudged=nudged.astype(dtype),
        ensemble=ensemble.astype(dtype),
        analysis_index=np.array([analysis_index], dtype=np.int64),
        cutoff_time=np.array([cutoff_time], dtype=dtype),
        ref_start_index=np.array([ref_start], dtype=np.int64),
        coarse_start_index=np.array([coarse_start], dtype=np.int64),
    )

    metadata = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "master_seed": int(master_seed),
        "semantics": {
            "reference": "(n_runs, n_ref, 3)",
            "coarse": "(n_runs, n_coarse, 3)",
            "nudged": "(n_runs, n_coarse, 3)",
            "ensemble": "(n_runs, ensemble_size, n_coarse, 3)",
            "state_order": ["z1", "z2", "z3"],
            "save_stride": int(cfg.save_stride),
            "note": "Within each run, all trajectories share the same initial state; across runs, initial states are re-sampled.",
        },
        "cutoff": {
            "cutoff_time": float(cutoff_time),
            "ref_start_index": int(ref_start),
            "coarse_start_index": int(coarse_start),
            "rule": "saved data starts at max(spinup_time, analysis_time)",
        },
        "shapes": {
            "time_reference": list(ref_t.shape),
            "time_coarse": list(coarse_t.shape),
            "initial_states": list(initial_states.shape),
            "reference": list(reference.shape),
            "coarse": list(coarse.shape),
            "nudged": list(nudged.shape),
            "ensemble": list(ensemble.shape),
        },
        "config": asdict(cfg),
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def stride_for_save(
    ref_t: np.ndarray,
    coarse_t: np.ndarray,
    reference: np.ndarray,
    coarse: np.ndarray,
    nudged: np.ndarray,
    ensemble: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep every ``stride``-th point after integration and truncation."""
    stride = max(int(stride), 1)
    return (
        ref_t[::stride],
        coarse_t[::stride],
        reference[:, ::stride, :],
        coarse[:, ::stride, :],
        nudged[:, ::stride, :],
        ensemble[:, :, ::stride, :],
    )


def generate_dataset(
    cfg: DatasetConfig,
    output_dir: Path | None = None,
    seed: int = 0,
) -> Path:
    """Generate trajectories and return the directory containing the dataset."""
    if cfg.n_ref < 2 or cfg.coarse_factor < 1 or cfg.save_stride < 1:
        raise ValueError("n_ref must be at least 2; coarse_factor and save_stride must be positive")
    if cfg.n_runs < 1 or cfg.ensemble_size < 1:
        raise ValueError("n_runs and ensemble_size must be positive")

    params = ModelParams(a1=cfg.a1, a2=cfg.a2, epsilon=cfg.epsilon, b=cfg.b)

    ref_t = np.linspace(*cfg.t_span, cfg.n_ref)
    coarse_t = coarse_time_grid(cfg.t_span, cfg.n_ref, cfg.coarse_factor)

    coarse_p = {
        "a1": params.a1,
        "a2": params.a2,
        "epsilon": params.epsilon,
        "b": params.b,
        "a1_scale": 1.00,
        "a2_scale": 1.00,
        "epsilon_scale": 1.00,
        "b_scale": 0.95,
        "coupling_scale": 0.90,
        "nonlineafrity_scale": 0.85,
        "z3_damping": 0.03,
    }

    n_coarse = coarse_t.size
    master_seed = int(seed)
    rng = np.random.default_rng(master_seed)
    noise_gen = StatelessGaussian(seed=master_seed, dim=3)

    base_init = np.asarray(cfg.initial_state, dtype=float)
    init_std = np.asarray(cfg.init_state_std, dtype=float)

    initial_states = np.zeros((cfg.n_runs, 3), dtype=float)
    reference = np.zeros((cfg.n_runs, cfg.n_ref, 3), dtype=float)
    coarse = np.zeros((cfg.n_runs, n_coarse, 3), dtype=float)
    nudged = np.zeros((cfg.n_runs, n_coarse, 3), dtype=float)
    ensemble = np.full((cfg.n_runs, cfg.ensemble_size, n_coarse, 3), np.nan, dtype=float)

    analysis_index = int(np.searchsorted(coarse_t, cfg.analysis_time))
    analysis_index = max(0, min(n_coarse - 1, analysis_index))

    total_work = integration_work_units(cfg.n_ref, n_coarse, cfg.n_runs, cfg.ensemble_size, analysis_index)
    progress = ProgressTracker(total_work)
    print_progress(0, total_work, prefix="Integration progress")

    for run_id in range(cfg.n_runs):
        x0 = base_init + init_std * rng.standard_normal(3)
        initial_states[run_id] = x0

        # Reference: exact previous setup (otd.py function + odeint on ref_t).
        reference[run_id] = to_state_array(reference_solution(x0, ref_t, params))
        progress.advance(max(cfg.n_ref - 1, 1), prefix=f"Run {run_id + 1}/{cfg.n_runs} reference")

        # Coarse: exact previous setup (coarse.py coarse_solution on coarse_t).
        coarse[run_id] = to_state_array(coarse_solution(np.array(x0, dtype=float), coarse_t, coarse_p))
        progress.advance(max(n_coarse - 1, 1), prefix=f"Run {run_id + 1}/{cfg.n_runs} coarse   ")

        # Nudged: exact previous setup (coupled odeint on coarse_t).
        nudged_states, _ref_on_coarse = nudged_solution(
            x0,
            coarse_t,
            coarse_p,
            low_dimensional_2,
            params,
            cfg.tau,
        )
        nudged[run_id] = nudged_states
        progress.advance(max(n_coarse - 1, 1), prefix=f"Run {run_id + 1}/{cfg.n_runs} nudged   ")

        # Ensemble: exact EM co-integration form around nudged trajectory.
        ensemble[run_id] = generate_ensemble(
            nudged_states=nudged_states,
            t_eval=coarse_t,
            coarse_p=coarse_p,
            tau=cfg.tau,
            sigma=cfg.sigma,
            analysis_index=analysis_index,
            run_id=run_id,
            ensemble_size=cfg.ensemble_size,
            noise_gen=noise_gen,
            progress=progress,
            progress_prefix=f"Run {run_id + 1}/{cfg.n_runs} ensemble ",
        )

    print_progress(total_work, total_work, prefix="Integration progress")

    cutoff_time = max(float(cfg.spinup_time), float(cfg.analysis_time))
    (
        ref_t_saved,
        coarse_t_saved,
        reference_saved,
        coarse_saved,
        nudged_saved,
        ensemble_saved,
        ref_start,
        coarse_start,
    ) = truncate_for_save(
        ref_t=ref_t,
        coarse_t=coarse_t,
        reference=reference,
        coarse=coarse,
        nudged=nudged,
        ensemble=ensemble,
        cutoff_time=cutoff_time,
    )

    (
        ref_t_saved,
        coarse_t_saved,
        reference_saved,
        coarse_saved,
        nudged_saved,
        ensemble_saved,
    ) = stride_for_save(
        ref_t_saved,
        coarse_t_saved,
        reference_saved,
        coarse_saved,
        nudged_saved,
        ensemble_saved,
        cfg.save_stride,
    )

    # After truncation, the saved analysis index is at the first coarse point.
    analysis_index_saved = 0

    if output_dir is None:
        name = f"dataset_tau{cfg.tau:.2f}_sigma{cfg.sigma:.0e}"
        output_dir = Path(__file__).resolve().parents[1] / name
    else:
        output_dir = Path(output_dir).resolve()

    save_dataset(
        output_dir=output_dir,
        cfg=cfg,
        master_seed=master_seed,
        ref_t=ref_t_saved,
        coarse_t=coarse_t_saved,
        initial_states=initial_states,
        reference=reference_saved,
        coarse=coarse_saved,
        nudged=nudged_saved,
        ensemble=ensemble_saved,
        analysis_index=analysis_index_saved,
        cutoff_time=cutoff_time,
        ref_start=ref_start,
        coarse_start=coarse_start,
    )
    return output_dir
