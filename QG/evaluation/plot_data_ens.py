#!/usr/bin/env python3
# Author: Isabella Thiel <thieli@mit.edu>
'''
Plot zonally averaged data from .mat files for different cases and handlers.
Usage:
    python plot_data_ens.py <case_name> <tau> [--intermediate] [--ensemble ens01 ens02 ...] [--compare H1 H2] [--tmax TMAX] [--tmin TMIN]
Options:
    <case_name> : Name of the case, e.g. Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.05
    <tau>       : Relaxation time, e.g. 16.0
    --intermediate : Use intermediate merged data directory, OPTIONAL
    --ensemble ens01 ens02 ... : List of ensemble handlers to include, e.g. ens01 ens02, OPTIONAL
    --compare H1 H2 : Compare two handlers H1 and H2, e.g. --compare ref rnudge, OPTIONAL
    --tmax TMAX : Maximum time for x-axis, OPTIONAL
    --tmin TMIN : Minimum time for x-axis, OPTIONAL
Examples:
    python plot_data_ens.py Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.05 16.0 --intermediate --ensemble ens01 ens08 --compare rnudge ens01
    python plot_data_ens.py Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization2_24x24_sig0.05 16.0 --ensemble ens01 ens08 --compare rnudge ens01 --tmin 250 --tmax 1250
    python plot_data_ens.py <case> 16.0 --intermediate --compare ens01 ens08
    python plot_data_ens.py <case> 16.0 --intermediate --compare rnudge ref


Author: Isabella Thiel, Oct 2025
'''


import os
import argparse
import re
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator


mpl.rcParams.update({

    'font.size': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,


    # Figure
    "figure.figsize": (12, 6),
    "figure.dpi": 350,
    "savefig.dpi": 350,
    "figure.facecolor": "white",

    # Font
    "font.size": 20,
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,

    # Axes
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "0.85",        # light neutral grid
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,    # clean scientific look
    "axes.spines.right": False,

    # Ticks
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,

    # Legend
    "legend.frameon": False,
    "legend.fontsize": 12,

    # Lines
    "lines.linewidth": 1.5
})

HERE = os.path.abspath(os.path.dirname(__file__))

def _find_zonal_file(base, var, h, tau):
    if h in ("ref", "coarse", "rnudge"):
        suffix = tau if h == "rnudge" else ""
        f = os.path.join(base, f"av_{var}_{h}{suffix}.mat")
        if os.path.exists(f): return f
    if h.startswith("ens"):
        m = [fn for fn in os.listdir(base) if fn == f"av_{var}_rnudge_{h}_{tau}.mat"]
        if m: return os.path.join(base, m[0])
        matches = [fn for fn in os.listdir(base)
                   if fn.startswith(f"av_{var}_{h}") and f"tau{tau}" in fn]
        if not matches:
            # fallback: match av_Psi1_ens01_sig..._tau16.0.mat
            matches = [fn for fn in os.listdir(base)
                    if fn.startswith(f"av_{var}_{h}") and f"tau{tau}" in fn and "_sig" in fn]
        if matches:
            return os.path.join(base, matches[0])


        # Look for raw ensemble file files are psi*_ensxxx_sig...6.0.mat
        raw_matches = [fn for fn in os.listdir(base)
                    if fn.startswith(f"{var}_rnudge") and f"_{h}" in fn and f"{tau}" in fn and fn.endswith(".mat")]
        # raw_matches = [fn for fn in os.listdir(base)
        #             if fn.startswith(f"{var.lower()}_{h}") and f"{tau}" in fn and fn.endswith(".mat")]
        if raw_matches:
            raw_file = os.path.join(base, raw_matches[0])
            if os.path.exists(raw_file):
                #mat = sio.loadmat(raw_file)
                return raw_file
        # else its av_{var}_{h}_sig..._{tau}.mat
        final_files = [fn for fn in os.listdir(base)
                       if fn.startswith(f"av_{var}_rnudge_ens{h}") and f"_{tau}" in fn and fn.endswith(".mat")]
        if final_files:
            return os.path.join(base, final_files[0])


        #if HERE conatins scratch
        if "scratch" in HERE:
            m = [fn for fn in os.listdir(base) if fn == f"av_{var}_{h}_sig0.05_{tau}.mat"]
            if m: return os.path.join(base, m[0])

    print(f" files available in {base}: {os.listdir(base)}")

def extract_sigma_from_name(name):
    """
    Extract the sigma (sig) value from a case directory or filename.
    Works for patterns like '_sig0.05', '_sig1', '_sig0.375', etc.
    Returns float value or None if not found.
    """
    m = re.search(r'_sig([0-9]*\.?[0-9]+)', name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

def _mat_get(M, keys, required=True):
    for k in keys:
        if k in M:
            return M[k]
    if required:
        raise KeyError(f"None of the keys found: {keys}")
    return None

def _as_float_scalar(x):
    return float(np.array(x).reshape(-1)[0])

def _resolve_base(case_name, use_intermediate):
    data_dir = os.environ.get("QG_DATA_DIR", os.path.join(HERE, '..', 'ML_Data'))
    if use_intermediate:
        base = os.path.abspath(os.path.join(data_dir, f'merged_{case_name}'))
    else:
        base = os.path.abspath(os.path.join(data_dir, case_name))
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Directory not found: {base}")
    return base


def _load_zonal(base, var, h, tau):
    def _find_var_key_local(M, var, h):
        for c in [f"{var}_{h}", var]:
            if c in M: return c
        prefix = f"{var}_{h}"
        prefix_rnudge = f"{var.lower()}_rnudge"
        for k in M.keys():
            if k.startswith(prefix) or k.startswith(prefix.lower()) or k.startswith(prefix_rnudge):
                return k
        for k in M.keys():
            if k.startswith(var): return k
        raise KeyError(f"No key for var={var}, h={h}. Keys: {list(M.keys())}")
    f = _find_zonal_file(base, var, h, tau)
    if f is None: raise FileNotFoundError(f"{var} {h} tau{tau} not found in {base}")
    M = sio.loadmat(f)
    k = _find_var_key_local(M, var, h)
    A = np.asarray(M[k], dtype=float)
    if A.ndim == 3: A = A.mean(axis=1)
    Ny, Nt = A.shape
    dt = _as_float_scalar(_mat_get(M, ["dt","ml_time_step"]))
    t0 = _as_float_scalar(_mat_get(M, ["tstart"], required=False) or 0.0)
    t = np.arange(Nt)*dt + t0
    y = np.linspace(0, 2*np.pi, Ny)
    return A, t, y


# ------------ PLOTTING FUNCTIONS ------------

def plot_zonal(case_name, tau, use_intermediate=False, ensembles=None, time_max=None, time_min=None):

    base = _resolve_base(case_name, use_intermediate)


    handlers = ['ref','coarse', 'rnudge']
    if ensembles:
        handlers.extend(ensembles)

    def _find_var_key(M, var, h):
        for c in [f"{var}_{h}", var]:
            if c in M: return c
        prefix = f"{var}_{h}"
        prefix_rnudge = f"{var.lower()}_rnudge"
        for k in M.keys():
            if k.startswith(prefix): return k
            elif k.startswith(prefix.lower()):
                return k
            elif k.startswith(prefix_rnudge):
                return k
        for k in M.keys():
            if k.startswith(var): return k
        raise KeyError(f"No key for var={var}, h={h}. Keys: {list(M.keys())}")

    for var in ['Psi1','Psi2']:
        fig,axs = plt.subplots(len(handlers),1,figsize=(12,3*len(handlers)),sharex=True)
        vmax=None
        for i,h in enumerate(handlers):
            zonal_file = _find_zonal_file(base, var, h, tau)
            if zonal_file is None:
                axs[i].axis('off'); axs[i].set_ylabel(h, fontsize=20, fontweight='bold'); continue
            M = sio.loadmat(zonal_file)
            var_key = _find_var_key(M, var, h)
            psi = np.asarray(M[var_key], dtype=float)
            # if ens in zonal file name
            if 'ens' in h and (psi.ndim == 3 or 'av_' not in os.path.basename(zonal_file)):
                psi = np.mean(psi, axis=1) # zonal average: average over x (axis=1)

            Ny,Nt = psi.shape
            dt = _as_float_scalar(_mat_get(M,["dt","ml_time_step"]))
            t0 = _as_float_scalar(_mat_get(M,["tstart"],required=False) or 0.0)
            t = np.arange(Nt)*dt+t0
            y = np.linspace(0,2*np.pi,Ny)
            if vmax is None: vmax=np.max(np.abs(psi))
            im = axs[i].imshow(psi,origin='lower',aspect='auto',
                               extent=[t[0],t[-1],y[0],y[-1]],
                               cmap='bwr',vmin=-vmax,vmax=vmax)
            axs[i].set_ylabel(h, fontsize=20, fontweight='bold')
            tmax = time_max if time_max is not None else t[-1]
            tmin = time_min if time_min is not None else t[0]
            axs[i].set_xlim(tmin, tmax)
            axs[i].xaxis.set_major_locator(MaxNLocator(nbins=4))
            axs[i].set_yticks([0, 2, 4, 6])
            #fig.colorbar(im,ax=axs[i],pad=0.02)
            cbar = fig.colorbar(im, ax=axs[i], pad=0.02)
            cbar.set_ticks([ -1, 0.0, 1 ])
            cbar.set_ticklabels(["-1", "0", "1"])
        axs[-1].set_xlabel('Time')
        fig.suptitle(f'Zonally averaged {var}')
        fig.tight_layout(rect=[0,0,1,0.96])
        outname = os.path.join(base,f"zonal_average_{var}_tau{tau}.png")
        plt.savefig(outname,dpi=300); plt.close(fig)
        print(f"Saved {outname}")


    # quick queck if more than one ensemble is loaded if they are the same
    if ensembles and len(ensembles) > 1:
        for var in ['Psi1','Psi2']:
            data = []
            for h in ensembles:
                zonal_file = _find_zonal_file(base, var, h, tau)
                if zonal_file is None: continue
                M = sio.loadmat(zonal_file)
                var_key = _find_var_key(M, var, h)
                psi = np.asarray(M[var_key], dtype=float)
                if 'ens' in h:
                    psi = np.mean(psi, axis=1) # zonal average: average over x (axis=1)
                data.append(psi)
            all_same = all(np.allclose(data[0], d) for d in data[1:])
            if all_same:
                print(f"WARNING: All ensembles for {var} are identical!")
            else:
                print(f"OK: Ensembles for {var} differ.")
                # check mean and stddev over ensembles
                stacked = np.stack(data, axis=-1) # shape (Ny,Nt,Nens)
                mean_ens = np.mean(stacked, axis=-1)
                std_ens = np.std(stacked, axis=-1)
                print(f"  Mean over ensembles: mean={np.mean(mean_ens):.4f}, std={np.std(mean_ens):.4f}")
                print(f"  Stddev over ensembles: mean={np.mean(std_ens):.4f}, std={np.std(std_ens):.4f}")


def plot_zonal_pair(case_name, tau, h1, h2, use_intermediate=False, time_max=None, time_min=None):
    base = _resolve_base(case_name, use_intermediate)
    for var in ["Psi1","Psi2"]:
        A1, t1, y1 = _load_zonal(base, var, h1, tau)
        A2, t2, y2 = _load_zonal(base, var, h2, tau)
        Ny = min(A1.shape[0], A2.shape[0])
        Nt = min(A1.shape[1], A2.shape[1], len(t1), len(t2))
        A1 = A1[:Ny,:Nt]; A2 = A2[:Ny,:Nt]
        t = t1[:Nt]; y = y1[:Ny]
        D = A1 - A2
        vmax = max(np.abs(A1).max(), np.abs(A2).max())
        vmaxd = np.abs(D).max() if np.abs(D).max() > 0 else 1.0
        fig, axs = plt.subplots(3,1,figsize=(12,9),sharex=True)
        im1 = axs[0].imshow(A1, origin="lower", aspect="auto", extent=[t[0], t[-1], y[0], y[-1]], cmap="bwr", vmin=-vmax, vmax=vmax)
        axs[0].set_ylabel(h1, fontsize=20, fontweight='bold')
        #fig.colorbar(im1, ax=axs[0], pad=0.02)
        im2 = axs[1].imshow(A2, origin="lower", aspect="auto", extent=[t[0], t[-1], y[0], y[-1]], cmap="bwr", vmin=-vmax, vmax=vmax)
        axs[1].set_ylabel(h2, fontsize=20, fontweight='bold')
        #fig.colorbar(im2, ax=axs[1], pad=0.02)
        im3 = axs[2].imshow(D, origin="lower", aspect="auto", extent=[t[0], t[-1], y[0], y[-1]], cmap="bwr", vmin=-vmaxd, vmax=vmaxd)
        axs[2].set_ylabel(f"{h1}-{h2}", fontsize=20, fontweight='bold')
        axs[2].set_xlabel("Time")
        #fig.colorbar(im3, ax=axs[2], pad=0.02)
        tmax = time_max if time_max is not None else t[-1]
        tmin = time_min if time_min is not None else t[0]
        axs[2].set_xlim(tmin, tmax)

        fig.suptitle(f"Zonal {var}: {h1} vs {h2}")
        fig.tight_layout(rect=[0,0,1,0.96])

        for i, ax in enumerate(axs):
            ax.set_yticks([0, 2, 4, 6])
            cbar = fig.colorbar([im1, im2, im3][i], ax=ax, pad=0.02)
            #cbar.set_ticks([ -1, 0.0, 1 ])
            #cbar.set_ticklabels(["-1", "0", "1"])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        safe_h1 = "".join(c for c in h1 if c.isalnum() or c in "_")
        safe_h2 = "".join(c for c in h2 if c.isalnum() or c in "_")
        outname = os.path.join(base, f"zonal_compare_{var}_{safe_h1}_vs_{safe_h2}_tau{tau}.png")
        plt.savefig(outname, dpi=300); plt.close(fig)
        print(f"Saved {outname}")

def plot_sigma_zonal_pair(case_dirs, tau, h1, h2, use_intermediate=False, time_max=None, time_min=None):
    """
    Compare zonally averaged Psi1 and Psi2 between h1 and h2 across multiple directories.
    Each directory corresponds to a column in the figure.
    A single colorbar is shared across columns.
    """
    assert isinstance(case_dirs, (list, tuple)) and len(case_dirs) > 0, "case_dirs must be a list of directories"
    for var in ["Psi1", "Psi2"]:
        data_pairs = []
        vmax_all = 0
        vmaxd_all = 0

        # Load data for all dirs first
        for case_dir in case_dirs:
            base = _resolve_base(case_dir, use_intermediate)
            A1, t1, y1 = _load_zonal(base, var, h1, tau)
            A2, t2, y2 = _load_zonal(base, var, h2, tau)
            Ny = min(A1.shape[0], A2.shape[0])
            Nt = min(A1.shape[1], A2.shape[1], len(t1), len(t2))
            A1, A2 = A1[:Ny, :Nt], A2[:Ny, :Nt]
            t, y = t1[:Nt], y1[:Ny]
            D = A1 - A2
            vmax = max(np.abs(A1).max(), np.abs(A2).max())
            vmaxd = np.abs(D).max() if np.abs(D).max() > 0 else 1.0
            vmax_all = max(vmax_all, vmax)
            vmaxd_all = max(vmaxd_all, vmaxd)

            sigma = extract_sigma_from_name(os.path.basename(case_dir))
            label = f"σ = {sigma:.3f}" if sigma is not None else os.path.basename(case_dir)

            data_pairs.append((A1, A2, D, t, y, label))

        ncols = len(case_dirs)
        fig, axs = plt.subplots(3, ncols, figsize=(5*ncols, 9), sharex='col', sharey='row')
        if ncols == 1:
            axs = axs.reshape(3, 1)

        for j, (A1, A2, D, t, y, label) in enumerate(data_pairs):
            im1 = axs[0, j].imshow(A1, origin="lower", aspect="auto",
                                   extent=[t[0], t[-1], y[0], y[-1]],
                                   cmap="bwr", vmin=-vmax_all, vmax=vmax_all)
            im2 = axs[1, j].imshow(A2, origin="lower", aspect="auto",
                                   extent=[t[0], t[-1], y[0], y[-1]],
                                   cmap="bwr", vmin=-vmax_all, vmax=vmax_all)
            im3 = axs[2, j].imshow(D, origin="lower", aspect="auto",
                                   extent=[t[0], t[-1], y[0], y[-1]],
                                   cmap="bwr", vmin=-vmaxd_all, vmax=vmaxd_all)
            axs[0, j].set_title(label, fontweight='bold')
            axs[2, j].set_xlabel("Time")
            for i in range(3):
                axs[i, j].set_yticks([0, 2, 4, 6])
                axs[i, j].tick_params(axis='both', which='major')
                axs[i, j].set_xlim(time_min or t[0], time_max or t[-1])
                axs[i, j].xaxis.set_major_locator(MaxNLocator(nbins=3))
                if j== len(data_pairs)-1: # only last column
                    divider = make_axes_locatable(axs[i, -1])
                    cax = divider.append_axes("right", size="5%", pad=0.2)
                    cbar = fig.colorbar([im1, im2, im3][i], cax=cax)
                    cbar.ax.tick_params(labelsize=14)
                    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))

        axs[0, 0].set_ylabel(h1, fontweight='bold')
        axs[1, 0].set_ylabel(h2,  fontweight='bold')
        axs[2, 0].set_ylabel(f"{h1}-{h2}", fontweight='bold')

        fig.suptitle(f"Zonal {var}: {h1} vs {h2}", fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # # Shared colorbar at the right
        # cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        # fig.colorbar(im3, cax=cbar_ax)
        # grid off
        for ax in axs.flatten():
            ax.grid(False)

        safe_h1 = "".join(c for c in h1 if c.isalnum() or c in "_")
        safe_h2 = "".join(c for c in h2 if c.isalnum() or c in "_")
        outname = f"figures/zonal_sigma_{var}_{safe_h1}_vs_{safe_h2}_tau{tau}.png"
        plt.savefig(outname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {outname}")

def plot_figure_final(case_dirs, tau, h1, h2, use_intermediate=False,
                      time_max=None, time_min=None):

    assert isinstance(case_dirs, (list, tuple)) and len(case_dirs) > 0

    ncols = len(case_dirs)
    odd_cols = (ncols % 2 == 1)
    mid_col = ncols // 2

    for var in ["Psi1", "Psi2"]:

        # --- Load reference, coarse, rnudge (from the FIRST directory only) ---
        base0 = _resolve_base(case_dirs[0], use_intermediate)
        R_raw, tR, yR = _load_zonal(base0, var, 'ref', tau)
        C_raw, tC, yC = _load_zonal(base0, var, 'coarse', tau)
        N_raw, tN, yN = _load_zonal(base0, var, h1, tau)

        Ny = min(R_raw.shape[0], C_raw.shape[0], N_raw.shape[0])
        Nt = min(R_raw.shape[1], C_raw.shape[1], N_raw.shape[1],
                 len(tR), len(tC), len(tN))

        # --- Load all ensemble members first, determine global Ny,Nt ---
        ens_raw = []
        labels = []

        for case_dir in case_dirs:
            base = _resolve_base(case_dir, use_intermediate)
            E_raw, tE, yE = _load_zonal(base, var, h2, tau)

            Ny = min(Ny, E_raw.shape[0])
            Nt = min(Nt, E_raw.shape[1], len(tE))

            sigma = extract_sigma_from_name(os.path.basename(case_dir))
            label = f"σ = {sigma:.3f}" if sigma is not None else os.path.basename(case_dir)

            ens_raw.append(E_raw)
            labels.append(label)

        # --- Now truncate EVERYTHING to uniform shape ---
        R = R_raw[:Ny, :Nt]
        C = C_raw[:Ny, :Nt]
        N = N_raw[:Ny, :Nt]
        t = tR[:Nt]
        y = yR[:Ny]

        ens = [E_raw[:Ny, :Nt] for E_raw in ens_raw]
        diff = [N - E for E in ens]

        # --- Compute scales ---
        vmax_main = max(np.abs(A).max() for A in [R, C, N] + ens)
        print('max R,C,N,ens:', np.abs(R).max(), np.abs(C).max(), np.abs(N).max(),
              [np.abs(E).max() for E in ens])
        if vmax_main == 0: vmax_main = 1.0
        vmin_main = -vmax_main

        vmax_diff = max(np.abs(D).max() for D in diff)
        if vmax_diff == 0: vmax_diff = 1.0
        vmin_diff = -vmax_diff

        # --- Build figure ---
        nrows = 5
        fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 3*nrows),
                                sharex="col", sharey="row")
        fig.tight_layout(rect=[0,0,1,0.96])
        if ncols == 1:
            axs = axs.reshape(nrows, 1)

        im_rows = [[None]*ncols for _ in range(nrows)]

        def plot_field(ax, A, vmin, vmax):
            return ax.imshow(A, origin="lower", aspect="auto",
                             extent=[t[0], t[-1], y[0], y[-1]],
                             cmap="bwr", vmin=vmin, vmax=vmax)

        # ============================
        # Plot rows
        # ============================

        for col in range(ncols):
            # Row 0: ref
            ax = axs[0, col]
            if odd_cols and col != mid_col:
                ax.axis("off")
            else:
                im_rows[0][col] = plot_field(ax, R, vmin_main, vmax_main)

            # Row 1: coarse
            ax = axs[1, col]
            if odd_cols and col != mid_col:
                ax.axis("off")
            else:
                im_rows[1][col] = plot_field(ax, C, vmin_main, vmax_main)

            # Row 2: rnudge
            ax = axs[2, col]
            if odd_cols and col != mid_col:
                ax.axis("off")
            else:
                im_rows[2][col] = plot_field(ax, N, vmin_main, vmax_main)

            # Row 3: ensemble for THIS sigma
            ax = axs[3, col]
            im_rows[3][col] = plot_field(ax, ens[col], vmin_main, vmax_main)
            ax.set_title(labels[col], fontweight="bold")

            # Row 4: diff = rnudge - ens
            ax = axs[4, col]
            im_rows[4][col] = plot_field(ax, diff[col], vmin_diff, vmax_diff)

            for row in range(nrows):
                ax = axs[row, col]
                if ax.get_visible():
                    ax.set_yticks([0,2,4,6])
                    ax.set_xlim(time_min or t[0], time_max or t[-1])
                    if row == nrows - 1:
                        ax.set_xlabel("Time")
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                    ax.grid(False)

        # ============================
        # Labels
        # ============================

        label_col = mid_col if odd_cols else 0

        axs[0, label_col].set_ylabel(r"$u(t)$", fontweight="bold", fontsize=28)
        axs[1, label_col].set_ylabel(r"$v(t)$", fontweight="bold", fontsize=28)
        axs[2, label_col].set_ylabel(r"$v_{\tau}(t)$", fontweight="bold", fontsize=28)
        axs[3, 0].set_ylabel(r"$v_{\tau}^{ens}(t)$", fontweight="bold", fontsize=28)
        axs[4, 0].set_ylabel(r"$v_{\tau}(t) - v_{\tau}^{ens}(t)$", fontweight="bold", fontsize=28)

        # ============================
        # Colorbars — one per row
        # ============================
        for row in range(5):

            # -----------------------------
            # SPECIAL CASE: ref / coarse / rnudge with odd columns
            # -----------------------------
            if row < 3 and odd_cols:
                data_col = mid_col
                im = im_rows[row][mid_col]
                if im is None:
                    continue

                # The Bbox of the subplot in figure coordinates
                bbox = axs[row, mid_col].get_position()

                # Same height as the axes (exactly like make_axes_locatable)
                height = bbox.height
                y0 = bbox.y0

                # Compute width identical to size="5%"
                subplot_width = bbox.width
                cbar_width = 0.05 * subplot_width

                # Compute pad identical to pad=0.2
                pad = 0.05 * subplot_width

                # Left boundary of the colorbar
                x0 = bbox.x1 + pad

                # ADD DUMMY AXES FOR COLORBAR
                dummy = fig.add_axes([x0, y0, cbar_width, height])

                cbar = fig.colorbar(im, cax=dummy)
                cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))
                continue

            # -----------------------------
            # NORMAL CASE (ens & diff rows)
            # -----------------------------
            data_col = ncols - 1
            im = im_rows[row][data_col]
            if im is None:
                continue

            divider = make_axes_locatable(axs[row, data_col])
            cax = divider.append_axes("right", size="5%", pad=0.2)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))


        # ============================
        # Save
        # ============================
        safe_h1 = "".join(c for c in h1 if c.isalnum() or c in "_")
        safe_h2 = "".join(c for c in h2 if c.isalnum() or c in "_")

        fig.suptitle(f"Zonal {var}: {h1} vs {h2}", fontweight="bold")


        outname = f"figures/zonal_final_{var}_{safe_h1}_vs_{safe_h2}_tau{tau}.png"
        plt.savefig(outname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Saved", outname)


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case_name")
    ap.add_argument("tau")
    ap.add_argument("--intermediate",action="store_true")
    ap.add_argument("--ensemble",nargs="+",help="e.g. ens01 ens02") # optional
    ap.add_argument("--compare",nargs=2,metavar=('H1','H2'),
                    help="Compare two handlers, e.g. --compare ref rnudge")
    ap.add_argument("--tmax", type=float, default=None, help="Max time for x-axis")
    ap.add_argument("--tmin", type=float, default=None, help="Min time for x-axis")


    args = ap.parse_args()


    # if args.compare:
    #     plot_zonal_pair(args.case_name, args.tau, args.compare[0], args.compare[1],
    #                     use_intermediate=args.intermediate, time_max=args.tmax, time_min=args.tmin)

    # plot_zonal(args.case_name,args.tau,
    #            use_intermediate=args.intermediate,
    #            ensembles=args.ensemble, time_max=args.tmax, time_min=args.tmin)

    plot_sigma_zonal_pair(
    case_dirs=[
        "Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.001",
        "Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.01",
        "Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.05",
        "Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.1",
        "Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.2",
    ],
    tau=16.0,
    h1=args.compare[0] if args.compare else "rnudge",
    h2=args.compare[1] if args.compare else "ens01",
    use_intermediate = args.intermediate,
    time_max = args.tmax,
    time_min = args.tmin
    )

    plot_figure_final(
    case_dirs=[
        "Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.001",
        "Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.01",
        "Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.05",
        "Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.1",
        "Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig0.2",
    ],
    tau=16.0,
    h1=args.compare[0] if args.compare else "rnudge",
    h2=args.compare[1] if args.compare else "ens01",
    use_intermediate = args.intermediate,
    time_max = args.tmax,
    time_min = args.tmin
    )
