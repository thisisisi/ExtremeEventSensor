"""Density estimation for trajectory diagnostics."""
import numpy as np
from KDEpy import FFTKDE
from scipy.stats import gaussian_kde

# Initialize the NumPy stream used by the training pipeline.
np.random.seed(0)


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
