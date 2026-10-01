# Author: Isabella Thiel <thieli@mit.edu>
"""Comparison metrics and plots for saved QG histogram statistics."""

import os
import gc
import json
from collections import OrderedDict
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import settings
from artifact_paths import results_save_path, results_save_path2
from plot_style import _add_external_legend

_EPS = 1e-12  # numerical floor for logs/divisions

from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter

import matplotlib as mpl


mpl.rcParams.update({

    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,


    # Figure
    "figure.figsize": (12, 6),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.facecolor": "white",

    # Font
    "font.size": 16,
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
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,

    # Legend
    "legend.frameon": False,
    "legend.fontsize": 18,

    # Lines
    "lines.linewidth": 1.5,

    #"text.usetex": True
    #"mathtext.default": "regular",


    "pdf.fonttype": 42,
    "ps.fonttype": 42

})


# ----------------------------- Basic I/O -------------------------------- #

def _load_result_file(path):
    if not os.path.isfile(path):
        print(f"    [WARN] File not found: {path}")
        return None
    try:
        return hdf5storage.loadmat(path)
    except Exception as e:
        print(f"    [ERROR] Could not load {path}: {e}")
        return None


def _resolve_results_file(results_dir, cfg):
    sig, n_ens, ntime, strategy, model_type, Nepochs, dt_ml, basis_type = cfg
    files_found = []

    if strategy is None:
        # Deterministic results do not depend on the ensemble-noise sigma.
        # Prefer the current case label, then fall back to the legacy label
        # under which the available T=50/T=1000 baselines were saved.
        case_stem = "Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24"
        for det_sig in ("0.01", "0.001"):
            case_name_det = f"{case_stem}_sig{det_sig}"
            candidates = [
                results_save_path(
                    results_dir, case_name_det, model_type, basis_type,
                    Nepochs, dt_ml, ntime, 1,
                )
            ]
            candidates.extend(
                results_save_path(
                    results_dir, case_name_det, model_type, basis_type,
                    Nepochs, dt_ml, ntime, 1, repeat_index=rep,
                )
                for rep in range(1, 20)
            )
            files_found.extend(path for path in candidates if os.path.exists(path))
            if files_found:
                break

    else:
        case_name = f"Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig{sig}"
        extra_tokens = None
        # The mechanism-control comparison is preregistered for exactly three
        # repeats. Ignore unnumbered legacy files and historical repeats 4+.
        for rep in range(1, 4):
            p_rep = results_save_path2(
                results_dir, case_name, model_type, basis_type, Nepochs, dt_ml,
                ntime, 1, strategy=strategy, n_members=n_ens,
                extra_tokens=extra_tokens, repeat_index=rep,
            )
            if os.path.exists(p_rep) and p_rep not in files_found:
                files_found.append(p_rep)

    return files_found


def _choose_result_path(paths):
    """Normalize _resolve_results_file output to a single chosen path string.
    Prefer a file without '_rep' in the basename; otherwise pick lowest rep number; fall back to first.
    """
    if paths is None:
        return None
    if isinstance(paths, str):
        return paths
    if not isinstance(paths, (list, tuple)) or len(paths) == 0:
        return None
    # prefer canonical file without _rep
    no_rep = [p for p in paths if "_rep" not in os.path.basename(p)]
    if no_rep:
        return no_rep[0]
    # parse rep numbers
    import re
    reps = []
    for p in paths:
        m = re.search(r"_rep(\d{3})", os.path.basename(p))
        if m:
            reps.append((int(m.group(1)), p))
    if reps:
        reps.sort()
        return reps[0][1]
    return paths[0]

# ------------------------ Histogram → PDF utilities --------------------- #

def _label_for_cfg(cfg):
    sig, n_ens, ntime, strategy, model_type, Nepochs, dt_ml, basis_type = cfg
    if strategy is None:
        label = r"ML(CR$_{\mathrm{test}}$ - baseline$_{T" + str(ntime) + "}$)"
        return label
    base = r"ML(CR$_{\mathrm{test}}$ - "
    ensemble_part = fr"ens$_{{T{ntime}}}^{{\sigma={sig}, N_{{add}}={n_ens}}})$"
    label = base + ensemble_part
    return label


def _counts_to_pdf(counts, dp, clip_eps=True):
   
    pdf = _counts_to_pdf_plot(counts, dp)
    return pdf

def _counts_to_pdf_plot(counts, dp):
    """
    Your exact plotting style:
      pdf = counts / (sum(counts) * dp)
    No clipping/renormalization to avoid tail artifacts on log-y plots.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum(axis=0)
    return counts / (total * dp)

def _global_ref_pdf_counts(data):
    # ref counts: (Nx, Ny, Np, 2) → sum over space
    C = data["psi_rpdf_ref"]  # (Nx,Ny,Np,2)
    return C.sum(axis=(0, 1))  # (Np,2)

def _global_coarse_pdf_counts(data):
    # coarse counts: (Nx, Ny, Np, 2) → sum over space
    C = data["psi_rpdf_coarse"]  # (Nx,Ny,Np,2)
    return C.sum(axis=(0, 1))  # (Np,2)


def _global_ml_pdf_counts(data):
    # Prefer pre-collapsed ensemble counts if present: (Nens,Np,2)
    if "psi_pdf_ML" in data:
        return np.mean(data["psi_pdf_ML"], axis=0)  # (Np,2) counts
    # Fallback: sum cell hist per ensemble then mean: (Nens,Nx,Ny,Np,2)
    C = data["psi_rpdf_ML"]
    return np.mean(C.sum(axis=(1, 2)), axis=0)  # (Np,2)


def _percell_ref_pdf(data):  # → (Nx,Ny,Np,2) counts
    return data["psi_rpdf_ref"]


def _percell_ml_counts_mean(data):  # → (Nx,Ny,Np,2) counts (mean over Nens)
    # psi_rpdf_ML: (Nens,Nx,Ny,Np,2)
    C = data["psi_rpdf_ML"]
    return np.mean(C, axis=0)

# --------------------------- Scalar metrics ---------------------------- #
def _kl_divergence(p_pdf, q_pdf, dp):
    eps = 1e-12
    eps = eps * max(p_pdf.max(), q_pdf.max())
    p = np.clip(p_pdf, eps, None)
    q = np.clip(q_pdf, eps, None)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1) * dp

    #return np.sum(p_pdf * (np.log(p_pdf + eps) - np.log(q_pdf + eps)), axis=-1) * dp

def _l1_log_pdf_error(p_pdf, q_pdf, dp):
    eps = 1e-12
    eps = eps * max(p_pdf.max(), q_pdf.max())
    p = np.clip(p_pdf, eps, None)
    q = np.clip(q_pdf, eps, None)
    return np.sum(np.abs(np.log(p) - np.log(q))) * dp
    #return np.sum(np.abs(np.log(p_pdf + eps) - np.log(q_pdf + eps)), axis=-1) * dp


def _wasserstein1_1d(p_pdf, q_pdf, dx):
    """
    W1(P,Q) = ∫ |F_P - F_Q| dx ≈ ∑ |cdf_p - cdf_q| * dx
    p_pdf, q_pdf: (..., Np)
    """
    cdf_p = np.cumsum(p_pdf, axis=-1) * dx
    cdf_q = np.cumsum(q_pdf, axis=-1) * dx
    return np.sum(np.abs(cdf_p - cdf_q), axis=-1) * dx

def compute_global_kl(data, target="ml"):
    """
    Global KL averaged over ψ1+ψ2. target ∈ {'ml','coarse'}.
    Returns dict with per-layer and average.
    """
    x = np.squeeze(data["pdf_xaxis"])
    dp = float(x[1] - x[0])

    Pref = _counts_to_pdf(_global_ref_pdf_counts(data), dp)  # (Np,2)
    if target == "ml":
        Pmod = _counts_to_pdf(_global_ml_pdf_counts(data), dp)
    elif target == "coarse":
        Pmod = _counts_to_pdf(_global_coarse_pdf_counts(data), dp)
    else:
        raise ValueError("target must be 'ml' or 'coarse'")

    kl1 = float(_kl_divergence(Pref[:, 0], Pmod[:, 0], dp))
    kl2 = float(_kl_divergence(Pref[:, 1], Pmod[:, 1], dp))
    return {"psi1": kl1, "psi2": kl2, "avg12": 0.5 * (kl1 + kl2)}


def compute_l1_logpdf(data, target="ml"):
    """
    Global L1 log-PDF error averaged over ψ1+ψ2.
    """
    x = np.squeeze(data["pdf_xaxis"])
    dp = float(x[1] - x[0])

    Pref = _counts_to_pdf(_global_ref_pdf_counts(data), dp)
    if target == "ml":
        Pmod = _counts_to_pdf(_global_ml_pdf_counts(data), dp)
    elif target == "coarse":
        Pmod = _counts_to_pdf(_global_coarse_pdf_counts(data), dp)
    else:
        raise ValueError("target must be 'ml' or 'coarse'")

    e1 = float(_l1_log_pdf_error(Pref[:, 0], Pmod[:, 0], dp))
    e2 = float(_l1_log_pdf_error(Pref[:, 1], Pmod[:, 1], dp))
    return {"psi1": e1, "psi2": e2, "avg12": 0.5 * (e1 + e2)}


def compute_wasserstein1(data, target="ml"):
    """
    Global Wasserstein-1 distance averaged over ψ1+ψ2.
    """
    x = np.squeeze(data["pdf_xaxis"])
    dx = float(x[1] - x[0])

    Pref = _counts_to_pdf(_global_ref_pdf_counts(data), dx)
    if target == "ml":
        Pmod = _counts_to_pdf(_global_ml_pdf_counts(data), dx)
    elif target == "coarse":
        Pmod = _counts_to_pdf(_global_coarse_pdf_counts(data), dx)
    else:
        raise ValueError("target must be 'ml' or 'coarse'")

    w1_1 = float(_wasserstein1_1d(Pref[:, 0], Pmod[:, 0], dx))
    w1_2 = float(_wasserstein1_1d(Pref[:, 1], Pmod[:, 1], dx))
    return {"psi1": w1_1, "psi2": w1_2, "avg12": 0.5 * (w1_1 + w1_2)}

# ---------------------- Threshold exceedance |ψ1|/|ψ2| ------------------ #






from scipy.stats import gaussian_kde




# --------------------------- Regional (ψ2) plots ------------------------ #



def plot_regional_statistics(data, out_png_prefix, target="ml"):
    """
    Regional statistics focusing on ψ2 (lower layer).
    Saves two files:
      - f"{out_png_prefix}_pdf_psi2.png" (PDF overlay, log-y)
      - f"{out_png_prefix}_spectrum_psi2.png" (spectrum overlay, log-log)
    """
    x = np.squeeze(data["pdf_xaxis"])
    dp = float(x[1] - x[0])

    # Use plotting normalization to match your visual style exactly
    Pref_counts = _global_ref_pdf_counts(data)
    Pref = _counts_to_pdf_plot(Pref_counts, dp)

    Pcrs = None
    if "psi_rpdf_coarse" in data:
        Pcrs_counts = _global_coarse_pdf_counts(data)
        Pcrs = _counts_to_pdf_plot(Pcrs_counts, dp)

    if target == "ml":
        Pmod_counts = _global_ml_pdf_counts(data)
        Pmod = _counts_to_pdf_plot(Pmod_counts, dp)
    elif target == "coarse":
        Pmod = Pcrs
    else:
        raise ValueError("target must be 'ml' or 'coarse'")

    # --- PDF overlay (ψ2) ---
    plt.figure(figsize=(7,5))
    plt.yscale("log")
    plt.plot(x, Pref[:,1], label="RD (test)")
    if Pcrs is not None:
        plt.plot(x, Pcrs[:,1], label="CR (test)", linestyle="--")
    lbl = "ML" if target == "ml" else "CR"
    plt.plot(x, Pmod[:,1], label=f"{lbl}", linewidth=2)
    plt.xlabel(fr"$\psi_2$")
    plt.ylabel(fr"PDF($\psi_2$)")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    out_pdf = f"{out_png_prefix}_pdf_psi2.png"
    plt.savefig(out_pdf); plt.close()
    print(f"    [INFO] Saved: {out_pdf}")

    # --- Spectrum overlay (ψ2) ---
    f = np.squeeze(data["f"])
    spec_ref = data["spectrum_ref"][:,1]
    spec_crs = data["spectrum_coarse"][:,1] if "spectrum_coarse" in data else None
    spec_ml  = np.mean(data["spectrum_ML"], axis=0)[:,1] if "spectrum_ML" in data else None
    plt.figure(figsize=(7,5))
    plt.loglog(f, spec_ref, label="RD (test)")
    if spec_crs is not None:
        plt.loglog(f, spec_crs, label="CR (test)", linestyle="--")
    if target == "ml" and spec_ml is not None:
        plt.loglog(f, spec_ml, label="ML", linewidth=2)
    elif target == "coarse" and spec_crs is not None:
        plt.loglog(f, spec_crs, label="CR", linewidth=2)
    plt.xlabel("frequency")
    plt.ylabel(fr"spectrum($\psi_2$)")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    out_spec = f"{out_png_prefix}_spectrum_psi2.png"
    plt.savefig(out_spec); plt.close()
    print(f"    [INFO] Saved: {out_spec}")

    return {"pdf_png": out_pdf, "spectrum_png": out_spec}
# ===================== 1) OVERLAY: Threshold exceedance ==================== #





def overlay_exceedance_panel(
    configs,
    results_dir="../ML_Results",
    out_dir=None,
    c_values=(0.5, 1.0, 1.5, 2.0),
    nbins=50,
    layer=2,
    legend_position="right",     # "right" or "top"
    formats=("png", "pdf"),      # save vector PDF too
):
    """
    Publication-style overlay plot for PDF of exceedance area fraction A_c/A.

    RD: solid black
    CR: dashed black
    ML: one curve per config, color-coded, sparse markers if not too many configs
    """

    if int(layer) not in (1, 2):
        raise ValueError("layer must be 1 or 2")

    if nbins < 10:
        raise ValueError("nbins should be at least 10 for a meaningful PDF estimate")

    out_dir = out_dir or results_dir
    os.makedirs(out_dir, exist_ok=True)

    loaded = []
    for cfg in configs:
        path = _resolve_results_file(results_dir, cfg)
        if not path:
            continue

        chosen = _choose_result_path(path)
        if not chosen:
            continue
        data = _load_result_file(chosen)
        if data is None:
            continue

        loaded.append((cfg, data, _label_for_cfg(cfg)))

    if not loaded:
        print("[WARN] No data found for overlay_exceedance_panel")
        return []

    edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dp = edges[1] - edges[0]

    layer_idx = 0 if int(layer) == 1 else 1

    def pdf_from_values(vals):
        vals = np.asarray(vals, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.full_like(centers, np.nan, dtype=float)

        vals = np.clip(vals, 0.0, 1.0)
        counts, _ = np.histogram(vals, bins=edges)
        denom = counts.sum() * dp

        if denom <= 0:
            return np.full_like(centers, np.nan, dtype=float)

        pdf = counts / denom
        return np.where(pdf > 0, pdf, np.nan)  # log-safe

    def extract_series(data, c, kind):
        thresh = np.squeeze(np.asarray(data["thresh"], dtype=float))
        k = int(np.nanargmin(np.abs(thresh - c)))

        if kind == "ref":
            arr = np.asarray(data["aot_ref"])
        elif kind == "coarse":
            arr = np.asarray(data["aot_coarse"])
        elif kind == "ml":
            arr = np.asarray(data["aot_ML"])
            if arr.ndim == 4:
                arr = np.mean(arr, axis=0)
        else:
            raise ValueError(f"Unknown series kind: {kind}")

        return arr[k, :, layer_idx].ravel(), float(thresh[k])

    n_panels = len(c_values)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))

    use_markers = len(loaded) <= 7
    markers = ("o", "s", "^", "D", "v", "P", "X")
    markevery = max(1, nbins // 10)

    rc = {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.9,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 350,
    }

    with mpl.rc_context(rc):
        fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=(11.2, 7.6) if legend_position == "right" else (10.2, 8.0),
            sharex=True,
            sharey=True,
        )

        axs = np.atleast_1d(axs).ravel()
        panel_axes = axs[:n_panels]

        for ax in axs[n_panels:]:
            ax.set_visible(False)

        base_data = loaded[0][1]

        for ax, c in zip(panel_axes, c_values):
            ref_vals, actual_c = extract_series(base_data, c, "ref")
            crs_vals, _ = extract_series(base_data, c, "coarse")

            ax.plot(
                centers,
                pdf_from_values(ref_vals),
                color="k",
                lw=2.5,
                ls="-",
                label=r"RD$_{\mathrm{test}}$",
                zorder=10,
            )

            ax.plot(
                centers,
                pdf_from_values(crs_vals),
                color="k",
                lw=2.2,
                ls=(0, (5, 2.5)),
                label=r"CR$_{\mathrm{test}}$",
                zorder=9,
            )

            for j, (_, data, label) in enumerate(loaded):
                ml_vals, _ = extract_series(data, c, "ml")

                style = {
                    "lw": 1.9,
                    "ls": "-",
                    "alpha": 0.95,
                    "label": label,
                    "zorder": 5,
                }

                if use_markers:
                    style.update(
                        marker=markers[j % len(markers)],
                        markersize=4.6,
                        markerfacecolor="white",
                        markeredgewidth=1.0,
                        markevery=markevery,
                    )

                ax.plot(centers, pdf_from_values(ml_vals), **style)

            title = rf"$c={c:g}$"
            if not np.isclose(actual_c, c):
                title += rf"  $(\mathrm{{used}}\ {actual_c:g})$"

            ax.set_title(title, pad=6)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(centers[0], 1.0)

            ax.grid(True, which="major", ls="-", lw=0.55, alpha=0.28)
            ax.grid(True, which="minor", ls=":", lw=0.40, alpha=0.20)

            ax.tick_params(which="both", direction="in", top=True, right=True)

            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())

        fig.supxlabel(r"exceedance area fraction $A_c/A$", y=0.045)
        fig.supylabel("PDF", x=0.045)

        legend_items = OrderedDict()
        for ax in panel_axes:
            handles, labels = ax.get_legend_handles_labels()
            for h, lab in zip(handles, labels):
                if lab and lab not in legend_items:
                    legend_items[lab] = h

        if legend_position == "right":
            fig.legend(
                legend_items.values(),
                legend_items.keys(),
                loc="center left",
                bbox_to_anchor=(0.815, 0.5),
                frameon=False,
                handlelength=3.0,
                handletextpad=0.8,
                labelspacing=0.7,
                borderaxespad=0.0,
            )
            fig.tight_layout(rect=(0.06, 0.06, 0.80, 0.98))

        elif legend_position == "top":
            ncols_legend = min(4, max(1, len(legend_items)))
            fig.legend(
                legend_items.values(),
                legend_items.keys(),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=ncols_legend,
                frameon=False,
                handlelength=3.0,
                columnspacing=1.4,
                handletextpad=0.7,
            )
            fig.tight_layout(rect=(0.06, 0.06, 0.98, 0.91))

        else:
            raise ValueError("legend_position must be either 'right' or 'top'")

        saved = []
        if isinstance(formats, str):
            formats = (formats,)

        for fmt in formats:
            fmt = fmt.lower().lstrip(".")
            out = os.path.join(out_dir, f"overlay_exceedance_panel_psi{layer}.{fmt}")
            fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
            saved.append(out)

        plt.close(fig)

    print("[INFO] Saved: " + ", ".join(saved))
    return saved

# ===================== 2) OVERLAY: Regional ψ₂ PDFs ======================= #

def overlay_regional_pdf_logpdf_grid(configs, results_dir="../ML_Results", out_dir=None,
                                     grid_shape=(3,3), target="ml", layer = 2):
    """
    PAPER-EXACT (Fig. 7a,b): 3x3 grid of regional pdf and log-pdf for ψ2.
      - Uses per-cell histograms (psi_rpdf_ref, psi_rpdf_coarse, psi_rpdf_ML).
      - RD/CR drawn once; ML one curve per config in each panel.
      - Two figures saved: regional_pdf_psi2.png (linear y), regional_logpdf_psi2.png (log y).
    """


    out_dir = out_dir or results_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load one representative file to get geometry and x-axis
    first_path = None
    first_data = None
    for cfg in configs:
        p = _resolve_results_file(results_dir, cfg)
        if p:
            chosen = _choose_result_path(p)
            if not chosen:
                continue
            first_path = chosen
            first_data = _load_result_file(chosen)
            if first_data is not None: break
    if first_data is None:
        print("[WARN] No data found for regional grid.")
        return
    layer_idx = 0 if int(layer)==1 else 1

    x = np.squeeze(first_data["pdf_xaxis"])
    dp = float(x[1] - x[0])
    Nx, Ny = first_data["psi_rpdf_ref"].shape[:2]  # (Nx,Ny,Np,2)
    gx, gy = grid_shape

    # Compute region index slices
    x_edges = np.linspace(0, Nx, gx+1, dtype=int)
    y_edges = np.linspace(0, Ny, gy+1, dtype=int)
    regions = []
    for ix in range(gx):
        for iy in range(gy):
            regions.append((slice(x_edges[ix], x_edges[ix+1]),
                            slice(y_edges[iy], y_edges[iy+1])))

    # Prepare figure canvases
    fig_lin, axs_lin = plt.subplots(gy, gx, figsize=(6*gx, 4*gy), sharex=True, sharey=True)
    fig_log, axs_log = plt.subplots(gy, gx, figsize=(6*gx, 4*gy), sharex=True, sharey=True)
    axs_lin = np.atleast_2d(axs_lin); axs_log = np.atleast_2d(axs_log)

    def _sum_counts_region(C, xs, ys):
        # C: (Nx,Ny,Np,2); return region-summed counts (Np,2)
        return np.sum(C[xs, ys, :, :], axis=(0,1))

    # Draw RD/CR once (from first_data)
    Cref = first_data["psi_rpdf_ref"]
    Ccrs = first_data["psi_rpdf_coarse"] if "psi_rpdf_coarse" in first_data else None

    for r,(xs,ys) in enumerate(regions):
        j = r % gx; i = r // gx
        ax1 = axs_lin[i,j]; ax2 = axs_log[i,j]; ax2.set_yscale("log")

        ref_counts = _sum_counts_region(Cref, xs, ys)
        P_ref = _counts_to_pdf_plot(ref_counts, dp)
        lab_ref = "RD (test)" if (i == 0 and j == 0) else None
        ax1.plot(x, P_ref[:,layer_idx], color="black", linewidth=2.6, label=lab_ref, zorder=30)
        ax2.plot(x, P_ref[:,layer_idx], color="black", linewidth=2.6, label=lab_ref, zorder=30)

        if Ccrs is not None:
            cr_counts = _sum_counts_region(Ccrs, xs, ys)
            P_crs = _counts_to_pdf_plot(cr_counts, dp)
            lab_crs = "CR (test)" if (i == 0 and j == 0) else None
            ax1.plot(x, P_crs[:,layer_idx], color="black", ls="--", linewidth=2.6, label=lab_crs, zorder=29)
            ax2.plot(x, P_crs[:,layer_idx], color="black", ls="--", linewidth=2.6, label=lab_crs, zorder=29)

        #ax1.set_title(f"Region ({i+1},{j+1})")

    # For each config: overlay ML in all panels
    for cfg in configs:
        label = _label_for_cfg(cfg)
        path  = _resolve_results_file(results_dir, cfg)
        chosen = _choose_result_path(path)
        if not chosen:
            continue
        data  = _load_result_file(chosen)
        if data is None:
            print(f"[WARN] Missing regional data for {label}")
            continue

        # Prefer ensemble-mean regional counts: sum per-cell over region then mean over ens
        if "psi_rpdf_ML" not in data:
            del data; gc.collect(); continue
        Cml = data["psi_rpdf_ML"]  # (Nens,Nx,Ny,Np,2)

        for r,(xs,ys) in enumerate(regions):
            j = r % gx; i = r // gx
            ax1 = axs_lin[i,j]; ax2 = axs_log[i,j]

            # Sum over region then mean over ensemble
            region_counts = np.sum(Cml[:, xs, ys, :, :], axis=(1,2))  # (Nens,Np,2)
            ml_counts = np.mean(region_counts, axis=0)                 # (Np,2)
            P_ml = _counts_to_pdf_plot(ml_counts, dp)

            plot_kwargs = dict(linewidth=2.4, label=label)
            if "ens" in label.lower():
                plot_kwargs.update(marker='o', markersize=5, mfc='none', markevery=55)
            ax1.plot(x, P_ml[:,layer_idx], **plot_kwargs)
            ax2.plot(x, P_ml[:,layer_idx], **plot_kwargs)

        del data; gc.collect()


    # Style & save
    for fig, axs, suffix in [(fig_lin, axs_lin, f"regional_pdf_psi{layer_idx}.png"),
                             (fig_log, axs_log, f"regional_logpdf_psi{layer_idx}.png")]:
        axs = np.atleast_2d(axs)
        rows, cols = axs.shape
        xlabel = rf"$\psi_{{{layer}}}$"
        ylabel = r"$p$"

        for i in range(rows):
            for j in range(cols):
                ax = axs[i, j]
                ax.set_xlim([-3.5, 3.5])
                ax.grid(True, which="both", ls="--")

                # ---- LEFT COLUMN: Y LABEL + TICKS ----
                if j == 0:
                    ax.set_ylabel(ylabel, fontsize=30)
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                    if 'log' in suffix:
                        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
                    ax.tick_params(labelleft=True)
                    #ax.set_yticklabels([])
                    #ax.tick_params(labelleft=False)

                # ---- BOTTOM ROW: X LABEL + TICKS ----
                if i == rows - 1:
                    #make axis label nicer with mathtext and subscript
                    ax.set_xlabel(xlabel, fontsize=30)
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                    ax.tick_params(labelbottom=True)
                    #ax.set_xticklabels([])
                    #ax.tick_params(labelbottom=False)

        _add_external_legend(fig, axs, position="top")
        out = os.path.join(out_dir, suffix)
        fig.savefig(out); plt.close(fig)
        print(f"    [INFO] Saved: {out}")

# ===================== 3) OVERLAY: Regional ψ₂ Spectra ==================== #

# ------------------------ Spatial error distributions ------------------- #

def _cell_metric_map(data, metric="kl"):
    """
    Per-cell metric between REF and mean-ensemble ML, for each layer.
    Returns dict: {"psi1": (Nx,Ny), "psi2": (Nx,Ny)}
    """
    x = np.squeeze(data["pdf_xaxis"])
    dp = float(x[1] - x[0])

    Cref = _percell_ref_pdf(data)     # (Nx,Ny,Np,2) counts
    Cml  = _percell_ml_counts_mean(data) # (Nx,Ny,Np,2) counts

    Pref = _counts_to_pdf(Cref, dp)      # (Nx,Ny,Np,2)
    Pml  = _counts_to_pdf(Cml,  dp)      # (Nx,Ny,Np,2)

    if metric == "kl":
        m1 = _kl_divergence(Pref[...,0], Pml[...,0], dp)  # (Nx,Ny)
        m2 = _kl_divergence(Pref[...,1], Pml[...,1], dp)
    elif metric == "l1":
        m1 = _l1_log_pdf_error(Pref[...,0], Pml[...,0], dp)
        m2 = _l1_log_pdf_error(Pref[...,1], Pml[...,1], dp)
    else:
        raise ValueError("metric must be 'kl' or 'l1'")

    return {"psi1": m1, "psi2": m2}


def plot_spatial_error_maps(data, out_png_prefix, metric="kl"):
    """
    Plot 2D heatmaps for per-cell metric (KL or L1 log-PDF) for ψ1 and ψ2.
    Saves:
      f"{out_png_prefix}_{metric}_psi1.png"
      f"{out_png_prefix}_{metric}_psi2.png"
    """
    maps = _cell_metric_map(data, metric=metric)  # (Nx,Ny)
    for key in ["psi1", "psi2"]:
        M = maps[key]
        plt.figure(figsize=(6,5))
        im = plt.imshow(M.T, origin="lower", aspect="equal")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"{metric.upper()} per-cell: {key}")
        plt.tight_layout()
        out_png = f"{out_png_prefix}_{metric}_{key}.png"
        plt.savefig(out_png); plt.close()
        print(f"    INFO] Saved: {out_png}")
    return {
        "psi1_png": f"{out_png_prefix}_{metric}_psi1.png",
        "psi2_png": f"{out_png_prefix}_{metric}_psi2.png",
    }

# --------------------------- Config overlays ---------------------------- #

def make_overlays_for_configs(configs, results_dir="../ML_Results", out_dir=None):
    """
    Build overlay figures (PDF, spectrum, zonal) across configs.
    Uses your plotting style for PDFs.
    """
    out_dir = out_dir or results_dir
    os.makedirs(out_dir, exist_ok=True)

    loaded = []
    for cfg in configs:
        label = _label_for_cfg(cfg)
        paths = _resolve_results_file(results_dir, cfg)
        if not paths:
            print(f"[WARN] No results for {label}")
            continue
        # _resolve_results_file may return a single path or a list of paths
        if isinstance(paths, str):
            chosen_path = paths
        else:
            # prefer file without '_rep' in name
            no_rep = [p for p in paths if "_rep" not in os.path.basename(p)]
            if len(no_rep) > 0:
                chosen_path = no_rep[0]
            else:
                # parse rep numbers and pick lowest
                import re
                reps = []
                for p in paths:
                    m = re.search(r"_rep(\d{3})", os.path.basename(p))
                    if m:
                        reps.append((int(m.group(1)), p))
                if reps:
                    reps.sort()
                    chosen_path = reps[0][1]
                else:
                    chosen_path = paths[0]

        data = _load_result_file(chosen_path)
        if data is None:
            continue
        loaded.append((label, data))

    if not loaded:
        print("[WARN] No data loaded for overlays.")
        return

    # free memory
    for _, d in loaded:
        del d
    gc.collect()

# ------------------------- Orchestrator / Driver ------------------------ #

def run_all_metrics_for_configs(configs, results_dir="../ML_Results", out_dir=None,
                                target="ml", plot_per_config=False, make_overlays=True):
    """
    For each config:
      - load results file
      - compute & print global metrics (KL, L1 log-PDF, W1) averaged over ψ1+ψ2
      - (optional) save per-config figures:
          * threshold-exceedance curve
          * ψ2 regional overlays (PDF + spectrum)
          * spatial error maps (KL and L1)
      - (optional) build overlays across all configs (PDF, spectrum, zonal)

    Args:
      target: 'ml' (default) or 'coarse' (compare CR to RD as "model")
      plot_per_config: default False (toggle ON to emit per-config plots)
      make_overlays: default True (emit comparison_*.png across configs)
    """
    out_dir = out_dir or results_dir
    os.makedirs(out_dir, exist_ok=True)

    do_summary = True
    if do_summary:
        summary = []
        print(f"Number of configs: {len(configs)}")
        for cfg in configs:
            sig_val, n_ens_val, t_val, strat_val = cfg[0], cfg[1], cfg[2], cfg[3]
            json_label = f"T{t_val}-sig{sig_val}-ens{n_ens_val}-S{strat_val}" if strat_val else f"T{t_val}-sig0.0-ens1-det"
            print_label = _label_for_cfg(cfg)

            paths = _resolve_results_file(results_dir, cfg)
            if not paths:
                print(f"[WARN] No results for {print_label}")
                continue

            runs_kl = []
            runs_l1 = []
            runs_w1 = []

            for path in paths:
                data = _load_result_file(path)
                if data is None: continue
                runs_kl.append(compute_global_kl(data, target=target))
                runs_l1.append(compute_l1_logpdf(data, target=target))
                runs_w1.append(compute_wasserstein1(data, target=target))

            if not runs_kl:
                continue

            def agg_metric(dicts):
                return {k: {"mean": float(np.mean([d[k] for d in dicts])), "std": float(np.std([d[k] for d in dicts])), "avg12": float(np.mean([d[k] for d in dicts]))} for k in dicts[0].keys()}

            KL = agg_metric(runs_kl)
            L1L = agg_metric(runs_l1)
            W1 = agg_metric(runs_w1)

            print(f"[METRICS] {print_label} (N={len(paths)} runs)")
            print(f"  KL(P_ref || P_{target}):  psi1={KL['psi1']['mean']:.6g}±{KL['psi1']['std']:.6g}, psi2={KL['psi2']['mean']:.6g}±{KL['psi2']['std']:.6g}, avg={KL['avg12']['mean']:.6g}±{KL['avg12']['std']:.6g}")
            print(f"  L1 log-PDF error:         psi1={L1L['psi1']['mean']:.6g}±{L1L['psi1']['std']:.6g}, psi2={L1L['psi2']['mean']:.6g}±{L1L['psi2']['std']:.6g}, avg={L1L['avg12']['mean']:.6g}±{L1L['avg12']['std']:.6g}")
            print(f"  Wasserstein-1 (PDF):      psi1={W1['psi1']['mean']:.6g}±{W1['psi1']['std']:.6g}, psi2={W1['psi2']['mean']:.6g}±{W1['psi2']['std']:.6g}, avg={W1['avg12']['mean']:.6g}±{W1['avg12']['std']:.6g}")

            summary.append({
                "label": json_label,
                "KL": KL, "L1log": L1L, "W1": W1,
            })

            if plot_per_config:
                # Threshold exceedance |ψ1|/|ψ2| ≥ c
                out_ratio = os.path.join(out_dir, f"ratio_exceed_{label}.png")
                _ = plot_threshold_exceedance(data, out_ratio, target=target)

                # Regional ψ2 plots
                out_reg_prefix = os.path.join(out_dir, f"regional_{label}")
                _ = plot_regional_statistics(data, out_reg_prefix, target=target)

                # Spatial maps
                out_sp_prefix = os.path.join(out_dir, f"spatial_{label}")
                _ = plot_spatial_error_maps(data, out_sp_prefix, metric="kl")
                _ = plot_spatial_error_maps(data, out_sp_prefix, metric="l1")

            del data
            gc.collect()

        # write consolidated summary
        sum_path = os.path.join(out_dir, "metrics_summary.json")
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Saved metrics summary: {sum_path}")

    if make_overlays and len(summary) > 0:
        make_overlays_for_configs(configs, results_dir=results_dir, out_dir=out_dir)
        for layer_nb in range(1,3):
            overlay_exceedance_panel(configs, results_dir=results_dir, out_dir=out_dir,
                            c_values=(0.5,1.0,1.5,1.75), nbins=100, layer= layer_nb)
            overlay_regional_pdf_logpdf_grid(configs, results_dir=results_dir, out_dir=out_dir, target=target, layer = layer_nb)
