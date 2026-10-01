#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference-vs-coarse diagnostics reusing the exact otd.py reference code.

Reference quantities are imported directly from otd.py's function block so that
its ODE, parameters, solver, and plotting windows remain unchanged.
The only new ingredient is a temporally under-resolved coarse variant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np

from KDEpy import FFTKDE
from scipy.integrate import odeint
from scipy.stats import gaussian_kde

np.random.seed(0)


# -----------------------------------------------------------------------------
# Load exact reference functions from otd.py 
# -----------------------------------------------------------------------------

def load_otd_namespace(otd_path: Path) -> Dict[str, object]:
    text = otd_path.read_text(encoding="utf-8")
    split_token = "# Parameters of the low-dimensional system"
    if split_token not in text:
        raise RuntimeError("Could not locate the function block in otd.py")
    function_block = text.split(split_token)[0]
    namespace: Dict[str, object] = {}
    exec(function_block, namespace)
    return namespace


# -----------------------------------------------------------------------------
# KDE helper matching the requested workflow
# -----------------------------------------------------------------------------

def custom_KDE(data: np.ndarray, weights=None, bw=None) -> FFTKDE:
    data = data.flatten()
    if bw is None:
        try:
            sc = gaussian_kde(data, weights=weights)
            bw_est = np.sqrt(sc.covariance).flatten()
            if np.size(bw_est) == 1:
                bw = float(bw_est[0])
            else:
                raise ValueError("The bw must be a number.")
        except Exception:
            bw = 1.0
        if bw < 1e-8:
            bw = 1.0
    return FFTKDE(bw=bw).fit(data, weights)


def kde_pdf_on_grid(data: np.ndarray, grid: np.ndarray) -> np.ndarray:
    data = data[np.isfinite(data)]
    if data.size < 2:
        return np.zeros_like(grid)

    mu = float(np.mean(data))
    sigma = max(float(np.std(data)), 1e-8)
    data_n = (data - mu) / sigma
    grid_n = (grid - mu) / sigma

    kde = custom_KDE(data_n)
    try:
        out = kde.evaluate(grid_n)
    except ValueError:
        x_eval, p_eval = kde.evaluate()
        return np.interp(grid_n, x_eval, p_eval, left=0.0, right=0.0) / sigma

    if isinstance(out, tuple):
        x_eval, p_eval = out
        p_n = np.interp(grid_n, x_eval, p_eval, left=0.0, right=0.0)
    else:
        p_n = np.asarray(out, dtype=float)
        if p_n.shape != grid_n.shape:
            x_lin = np.linspace(grid_n.min(), grid_n.max(), p_n.size)
            p_n = np.interp(grid_n, x_lin, p_n)
    return p_n / sigma


# -----------------------------------------------------------------------------
# Coarse grid and utilities
# -----------------------------------------------------------------------------

def coarse_time_grid(t_span: Tuple[float, float], n_ref: int, factor: int) -> np.ndarray:
    n_coarse = max(2, n_ref // factor)
    return np.linspace(t_span[0], t_span[1], n_coarse)

def coarse_system(state: np.ndarray, t: float, p: Dict[str, float]) -> np.ndarray:
    """Coarse system = original nonlinear low-dimensional system, 
    written with parameter dict p."""
    z1, z2, z3 = state

    a1 = p["a1"]
    a2 = p["a2"]
    epsilon = p["epsilon"]
    b = p["b"]
    
    gamma = 15

    dz1_dt = -a1 * z1 + epsilon * z2 + b * z3
    dz2_dt = epsilon**-1 * z1 - a2 * z2
    dz3_dt = b * z3 * (1.0 / np.sqrt(z1**2 + z2**2) - 1.0) - gamma * z3

    return np.array([dz1_dt, dz2_dt, dz3_dt], dtype=float)


def coarse_solution(
    initial_state: np.ndarray,
    t_eval: np.ndarray,
    p: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the original nonlinear system on t_eval."""
    solution = odeint(coarse_system, initial_state, t_eval, args=(p,))
    return solution[:, 0], solution[:, 1], solution[:, 2]


def coarse_jacobian(state: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """Jacobian of the original nonlinear system."""
    z1, z2, z3 = state

    a1 = p["a1"]
    a2 = p["a2"]
    epsilon = p["epsilon"]
    b = p["b"]
    gamma = 15

    r2 = z1**2 + z2**2
    r2 = max(r2, 1e-16)
    r32 = r2 ** 1.5

    return np.array(
        [
            [-a1,              epsilon,   b],
            [epsilon**-1,      -a2,       0.0],
            [-b * z3 * z1 / r32, -b * z3 * z2 / r32,
             b * (1.0 / np.sqrt(r2) - 1.0) - gamma],
        ],
        dtype=float,
    )




# -----------------------------------------------------------------------------
# OTD and eigenvalue helpers using otd.py functions
# -----------------------------------------------------------------------------

def compute_otd_modes(ns: Dict[str, object], time: np.ndarray, z: Tuple[np.ndarray, np.ndarray, np.ndarray], rank: int) -> np.ndarray:
    solution_otd_modes = ns["solution_otd_modes"]
    z1, z2, z3 = z
    ns["n"] = 3
    ns["r"] = rank
    return solution_otd_modes(time, z1, z2, z3)



def compute_eigens(
    ns: Dict[str, object],
    time: np.ndarray,
    z: Tuple[np.ndarray, np.ndarray, np.ndarray],
    otd_modes: np.ndarray,
    use_coarse_jac: bool = False,
    coarse_p: Dict[str, float] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    jacobian = ns["jacobian"]
    a1 = ns["a1"]
    a2 = ns["a2"]
    epsilon = ns["epsilon"]
    b = ns["b"]

    z1, z2, z3 = z
    eig_j = np.zeros((len(time), 3), dtype=float)
    eig_otd = np.zeros(len(time), dtype=float)

    for i in range(len(time)):
        state = np.array([z1[i], z2[i], z3[i]], dtype=float)
        if use_coarse_jac:
            if coarse_p is None:
                raise ValueError("coarse_p must be provided when use_coarse_jac is True")
            j = coarse_jacobian(state, coarse_p)
        else:
            j = jacobian(state, a1, a2, epsilon, b)
        lam = np.real(np.linalg.eigvals(j))
        lam.sort()
        eig_j[i] = lam[::-1]

        lr = otd_modes[i].T @ j @ otd_modes[i]
        eig_otd[i] = float(np.real(np.linalg.eigvals(lr)[0]))

    return eig_j, eig_otd



# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    set_publication_style()

    cfg = {
        "initial_state": (0.1, -1.0, 0.1),
        "t_span": (0.0, 300.0),
        "n_main_ref": 100000,
        "n_eig_ref": 10000,
        "dt_factor_coarse": 1,
        "observable_abs_max": 40.0,
        "otd_rank": 1,
    }

    # Editable coarse-system parameters.
    coarse_params = {
        "a1": 2.0,
        "a2": 2.0,
        "epsilon": 0.05,
        "b": 20.0,
        "a1_scale": 1.00,
        "a2_scale": 1.00,
        "epsilon_scale": 1.00,
        "b_scale": 0.95,
        "coupling_scale": 0.90,
        "nonlinearity_scale": 0.85,
        "z3_damping": 0.03,
    }

    otd_path = Path(__file__).with_name("otd.py")
    output_dir = Path(__file__).resolve().parent
    ns = load_otd_namespace(otd_path)

    # Exact reference values from otd.py.
    a1 = 2.0
    a2 = 2.0
    epsilon = 0.05
    b = 20.0
    initial_state = list(cfg["initial_state"])
    t_span = cfg["t_span"]
    n_main_ref = cfg["n_main_ref"]
    n_eig_ref = cfg["n_eig_ref"]
    dt_factor_coarse = cfg["dt_factor_coarse"]
    observable_abs_max = cfg["observable_abs_max"]
    otd_rank = cfg["otd_rank"]
    


    # Populate the globals expected by otd.py functions.
    ns["a1"] = a1
    ns["a2"] = a2
    ns["epsilon"] = epsilon
    ns["b"] = b
    ns["n"] = 3
    ns["r"] = otd_rank

    solution_low_dimensional = ns["solution_low_dimensional"]
    solution_otd_modes = ns["solution_otd_modes"]

    # Reference system: exact same setup as otd.py.
    t_ref_main = np.linspace(*t_span, n_main_ref)
    z_ref_main = solution_low_dimensional(initial_state, t_ref_main, a1, a2, epsilon, b)

    t_ref_eig = np.linspace(*t_span, n_eig_ref)
    z_ref_eig = solution_low_dimensional(initial_state, t_ref_eig, a1, a2, epsilon, b)
    
    r_ref = np.sqrt(z_ref_main[0]**2 + z_ref_main[1]**2)
    coarse_params["r0"] = float(np.mean(r_ref))
    # Temporal coarse system: same physics, fewer time points.
    t_c_main = coarse_time_grid(t_span, n_main_ref, dt_factor_coarse)
    z_c_main = coarse_solution(np.array(initial_state, dtype=float), t_c_main, coarse_params)

    t_c_eig = coarse_time_grid(t_span, n_eig_ref, dt_factor_coarse)
    z_c_eig = coarse_solution(np.array(initial_state, dtype=float), t_c_eig, coarse_params)

    # OTD modes.
    ns["n"] = 3
    ns["r"] = otd_rank
    otd_ref_main = solution_otd_modes(t_ref_main, *z_ref_main)
    otd_c_main = solution_otd_modes(t_c_main, *z_c_main)
    otd_ref_eig = solution_otd_modes(t_ref_eig, *z_ref_eig)
    otd_c_eig = solution_otd_modes(t_c_eig, *z_c_eig)

    # Eigenvalues and OTD growth.
    eig_ref, sig_ref = compute_eigens(ns, t_ref_eig, z_ref_eig, otd_ref_eig)
    eig_c, sig_c = compute_eigens(ns, t_c_eig, z_c_eig, otd_c_eig, use_coarse_jac=True, coarse_p=coarse_params)


if __name__ == "__main__":
    main()
