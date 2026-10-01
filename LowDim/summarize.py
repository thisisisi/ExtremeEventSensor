#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize low-dimensional corrector results across sigmas and ensembles.

The script expects each completed training run to contain:

    corrected_test_outputs.npz
    summary.json

inside the corresponding ML_Results directory. It computes PDF metrics and
AOT-distribution metrics for reference, coarse, and corrected trajectories.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


STATE_LABELS = (r"$z_1$", r"$z_2$", r"$z_3$")


@dataclass(frozen=True)
class SummaryConfig:
    results_root: Path = Path("ML_Results")
    output_dir: Path = Path("ML_Results/summary_tables")
    sigmas: tuple[float, ...] = (1e-4, 1e-5, 5e-6, 1e-6)
    ensemble_sizes: tuple[int, ...] = (0, 1, 2, 4, 8, 12, 16)
    architectures: tuple[str, ...] = ("attention")
    run_id_train: int = 0
    run_id_test: int = 1
    stride: int | None = None
    repeat_indices: tuple[int, ...] | None = None
    threshold_percentiles: tuple[float, ...] = (90.0, 95.0, 99.0)
    variables: tuple[int, ...] = (0, 1, 2)
    pdf_grid_points: int = 1200
    pdf_epsilon: float = 1e-12


def parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def parse_str_list(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def sigma_token(sigma: float) -> str:
    return f"{sigma:.2e}".replace("+", "")


def set_plot_style() -> None:
    if plt is None:
        return
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "lines.linewidth": 1.9,
            "lines.markersize": 5.0,
        }
    )


def result_candidates(cfg: SummaryConfig, sigma: float, n_ens: int) -> list[Path]:
    sig = sigma_token(sigma)
    ens_pattern = f"ens{n_ens}_"
    candidates: list[Path] = []

    for path in cfg.results_root.glob(f"sigma{sig}_*"):
        if ens_pattern not in path.name:
            continue
        if f"run{cfg.run_id_train}_{cfg.run_id_test}" not in path.name:
            continue
        if cfg.stride is not None and f"stride{cfg.stride}_" not in path.name:
            continue
        repeat_index = infer_repeat_index(path)
        if cfg.repeat_indices is not None and repeat_index not in cfg.repeat_indices:
            continue
        if cfg.architectures and not any(f"_{arch}_" in path.name for arch in cfg.architectures):
            # Also allow legacy directories without an architecture token.
            if any(arch in path.name for arch in cfg.architectures):
                pass
            elif "_att" in path.name or "_tcn" in path.name:
                continue
        if (path / "corrected_test_outputs.npz").exists():
            candidates.append(path)

    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def infer_repeat_index(result_dir: Path) -> int | None:
    match = re.search(r"_rep(\d+)$", result_dir.name)
    return int(match.group(1)) if match else None


def infer_ml_stride(result_dir: Path, summary: dict) -> int | None:
    match = re.search(r"_stride(\d+)_T", result_dir.name)
    if match:
        return int(match.group(1))
    value = summary.get("ml_stride")
    return int(value) if value is not None else None


def infer_train_time(result_dir: Path, summary: dict) -> int | None:
    match = re.search(r"_T(\d+)_run", result_dir.name)
    if match:
        return int(match.group(1))
    value = summary.get("train_time")
    return int(value) if value is not None else None


def infer_architecture(result_dir: Path, summary: dict) -> str:
    arch = summary.get("architecture")
    if arch:
        return str(arch)
    match = re.search(r"_det_([^_]+)_ens", result_dir.name)
    return match.group(1) if match else "unknown"


def infer_conditioning(result_dir: Path, summary: dict) -> tuple[str, bool, int | None]:
    n_ens = int(summary.get("n_ensemble_members_used", -1))
    eigencov = bool(summary.get("eigencov", False))
    n_modes = summary.get("n_eigcov_modes", None)

    if n_ens == 0 or "_nofilm_" in result_dir.name:
        return "nofilm", False, None
    if "_meanstd_" in result_dir.name:
        return "meanstd", False, None

    match = re.search(r"_vec(\d+)_", result_dir.name)
    if match:
        return f"vec{match.group(1)}", True, int(match.group(1))
    if eigencov:
        n_mode_int = int(n_modes) if n_modes is not None else None
        return f"vec{n_mode_int}" if n_mode_int is not None else "vec", True, n_mode_int

    return "unknown", eigencov, int(n_modes) if n_modes is not None else None


def load_summary(result_dir: Path) -> dict:
    path = result_dir / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pdf_metrics(
    reference: np.ndarray,
    model: np.ndarray,
    var_idx: int,
    grid_points: int,
    eps: float,
) -> dict[str, float]:
    x_ref = reference[:, var_idx]
    x_model = model[:, var_idx]
    x_lo = min(float(np.quantile(x_ref, 0.001)), float(np.quantile(x_model, 0.001)))
    x_hi = max(float(np.quantile(x_ref, 0.999)), float(np.quantile(x_model, 0.999)))
    pad = 0.08 * max(x_hi - x_lo, 1e-8)
    bins = np.linspace(x_lo - pad, x_hi + pad, grid_points + 1)
    p_ref, _ = np.histogram(x_ref, bins=bins, density=True)
    p_model, _ = np.histogram(x_model, bins=bins, density=True)
    dx = float(np.mean(np.diff(bins)))

    p_ref = np.maximum(p_ref, eps)
    p_model = np.maximum(p_model, eps)
    p_ref = p_ref / max(float(np.sum(p_ref) * dx), eps)
    p_model = p_model / max(float(np.sum(p_model) * dx), eps)

    kl = float(np.sum(p_ref * np.log(p_ref / p_model)) * dx)
    log_l1 = float(np.mean(np.abs(np.log(p_ref) - np.log(p_model))))
    l1 = float(np.sum(np.abs(p_ref - p_model)) * dx)
    return {"kl": kl, "log_l1": log_l1, "pdf_l1": l1}


def aot_series(states: np.ndarray, threshold: float, var_idx: int) -> np.ndarray:
    """Low-dimensional AOT analogue: binary exceedance fraction per time."""
    return (np.abs(states[:, var_idx]) > threshold).astype(np.float64)


def wasserstein_1_1d(x: np.ndarray, y: np.ndarray) -> float:
    x_sorted = np.sort(np.asarray(x, dtype=np.float64))
    y_sorted = np.sort(np.asarray(y, dtype=np.float64))
    n = max(x_sorted.size, y_sorted.size)
    if n == 0:
        return 0.0
    q = (np.arange(n, dtype=np.float64) + 0.5) / n
    xq = np.quantile(x_sorted, q)
    yq = np.quantile(y_sorted, q)
    return float(np.mean(np.abs(xq - yq)))


def aot_metrics(reference: np.ndarray, model: np.ndarray, threshold: float, var_idx: int) -> dict[str, float]:
    a_ref = aot_series(reference, threshold, var_idx)
    a_model = aot_series(model, threshold, var_idx)
    return {
        "aot_w1": wasserstein_1_1d(a_ref, a_model),
        "event_frequency_error": float(abs(np.mean(a_ref > 0.0) - np.mean(a_model > 0.0))),
        "ref_event_frequency": float(np.mean(a_ref > 0.0)),
        "model_event_frequency": float(np.mean(a_model > 0.0)),
    }


def rows_for_result(cfg: SummaryConfig, sigma: float, n_ens: int, result_dir: Path) -> tuple[list[dict], list[dict]]:
    data = np.load(result_dir / "corrected_test_outputs.npz")
    reference = np.asarray(data["reference"], dtype=np.float64)
    coarse = np.asarray(data["coarse"], dtype=np.float64)
    corrected = np.asarray(data["corrected"], dtype=np.float64)

    summary = load_summary(result_dir)
    architecture = infer_architecture(result_dir, summary)
    conditioning, eigencov, n_eigcov_modes = infer_conditioning(result_dir, summary)
    train_time = infer_train_time(result_dir, summary)
    ml_stride = infer_ml_stride(result_dir, summary)
    common = {
        "sigma": sigma,
        "ensemble_size": n_ens,
        "architecture": architecture,
        "conditioning": conditioning,
        "eigencov": eigencov,
        "n_eigcov_modes": n_eigcov_modes,
        "train_time": train_time,
        "ml_stride": ml_stride,
        "repeat_index": infer_repeat_index(result_dir),
        "result_dir": str(result_dir),
    }

    pdf_rows: list[dict] = []
    aot_rows: list[dict] = []

    for var_idx in cfg.variables:
        for model_name, states in (("coarse", coarse), ("corrected", corrected)):
            metrics = pdf_metrics(reference, states, var_idx, cfg.pdf_grid_points, cfg.pdf_epsilon)
            pdf_rows.append(
                {
                    **common,
                    "model": model_name,
                    "variable": f"z{var_idx + 1}",
                    **metrics,
                }
            )

        thresholds = np.percentile(np.abs(reference[:, var_idx]), cfg.threshold_percentiles)
        for percentile, threshold in zip(cfg.threshold_percentiles, thresholds):
            for model_name, states in (("coarse", coarse), ("corrected", corrected)):
                metrics = aot_metrics(reference, states, float(threshold), var_idx)
                aot_rows.append(
                    {
                        **common,
                        "model": model_name,
                        "variable": f"z{var_idx + 1}",
                        "threshold_percentile": float(percentile),
                        "threshold": float(threshold),
                        **metrics,
                    }
                )

    return pdf_rows, aot_rows


def summarize(cfg: SummaryConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pdf_rows: list[dict] = []
    aot_rows: list[dict] = []
    run_rows: list[dict] = []

    for sigma in cfg.sigmas:
        for n_ens in cfg.ensemble_sizes:
            candidates = result_candidates(cfg, sigma, n_ens)
            if not candidates:
                run_rows.append(
                    {
                        "sigma": sigma,
                        "ensemble_size": n_ens,
                        "status": "missing",
                        "result_dir": "",
                    }
                )
                continue

            for result_dir in candidates:
                summary = load_summary(result_dir)
                conditioning, eigencov, n_eigcov_modes = infer_conditioning(result_dir, summary)
                run_rows.append(
                    {
                        "sigma": sigma,
                        "ensemble_size": n_ens,
                        "repeat_index": infer_repeat_index(result_dir),
                        "status": "found",
                        "architecture": infer_architecture(result_dir, summary),
                        "conditioning": conditioning,
                        "eigencov": eigencov,
                        "n_eigcov_modes": n_eigcov_modes,
                        "train_time": infer_train_time(result_dir, summary),
                        "ml_stride": infer_ml_stride(result_dir, summary),
                        "result_dir": str(result_dir),
                        "best_epoch": summary.get("best_epoch", np.nan),
                        "best_val_score": summary.get("best_val_score", np.nan),
                    }
                )
                pdf_part, aot_part = rows_for_result(cfg, sigma, n_ens, result_dir)
                pdf_rows.extend(pdf_part)
                aot_rows.extend(aot_part)

    return pd.DataFrame(run_rows), pd.DataFrame(pdf_rows), pd.DataFrame(aot_rows)


def _format_group_value(values: pd.Series, prefix: str) -> str:
    unique_values = sorted({int(value) for value in values.dropna().unique()})
    if not unique_values:
        return f"{prefix}unknown"
    if len(unique_values) == 1:
        return f"{prefix}{unique_values[0]}"
    return f"{prefix}{'-'.join(str(value) for value in unique_values)}"


def summary_group_name(runs: pd.DataFrame) -> str:
    train_time_part = _format_group_value(runs["train_time"], "T") if "train_time" in runs else "Tunknown"
    stride_part = _format_group_value(runs["ml_stride"], "stride") if "ml_stride" in runs else "strideunknown"
    return f"{train_time_part}_{stride_part}"


def filter_group_frame(frame: pd.DataFrame, train_time: int, ml_stride: int) -> pd.DataFrame:
    if frame.empty or "train_time" not in frame or "ml_stride" not in frame:
        return frame.iloc[0:0].copy()
    mask = (frame["train_time"] == train_time) & (frame["ml_stride"] == ml_stride)
    return frame[mask.fillna(False)].copy()


def write_tables(cfg: SummaryConfig, output_dir: Path, runs: pd.DataFrame, pdf: pd.DataFrame, aot: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / "runs.csv", index=False)
    pdf.to_csv(output_dir / "pdf_metrics.csv", index=False)
    aot.to_csv(output_dir / "aot_metrics.csv", index=False)

    pdf_agg, aot_agg = aggregate_repeats(pdf, aot)
    pdf_agg.to_csv(output_dir / "pdf_metrics_repeat_aggregate.csv", index=False)
    aot_agg.to_csv(output_dir / "aot_metrics_repeat_aggregate.csv", index=False)

    payload = {
        "config": {
            **asdict(cfg),
            "results_root": str(cfg.results_root),
            "output_dir": str(output_dir),
        }
    }
    (output_dir / "summary_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not pdf.empty:
        pdf_pivot = pdf[pdf["model"] == "corrected"].pivot_table(
            index=["sigma", "ensemble_size", "architecture", "conditioning", "repeat_index"],
            columns="variable",
            values=["kl", "log_l1"],
            aggfunc="first",
        )
        (output_dir / "pdf_metrics_corrected.md").write_text(table_to_markdown(pdf_pivot), encoding="utf-8")

    if not aot.empty:
        aot_pivot = aot[aot["model"] == "corrected"].pivot_table(
            index=["sigma", "ensemble_size", "architecture", "conditioning", "repeat_index", "threshold_percentile"],
            columns="variable",
            values=["aot_w1", "event_frequency_error"],
            aggfunc="first",
        )
        (output_dir / "aot_metrics_corrected.md").write_text(table_to_markdown(aot_pivot), encoding="utf-8")


def aggregate_repeats(pdf: pd.DataFrame, aot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pdf_group_cols = ["sigma", "ensemble_size", "architecture", "conditioning", "model", "variable"]
    aot_group_cols = pdf_group_cols + ["threshold_percentile"]

    if pdf.empty:
        pdf_agg = pd.DataFrame()
    else:
        pdf_agg = (
            pdf.groupby(pdf_group_cols, dropna=False)[["kl", "log_l1", "pdf_l1"]]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        pdf_agg.columns = ["_".join(str(part) for part in col if str(part)) for col in pdf_agg.columns]

    if aot.empty:
        aot_agg = pd.DataFrame()
    else:
        aot_agg = (
            aot.groupby(aot_group_cols, dropna=False)[["aot_w1", "event_frequency_error"]]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        aot_agg.columns = ["_".join(str(part) for part in col if str(part)) for col in aot_agg.columns]

    return pdf_agg, aot_agg


def table_to_markdown(table: pd.DataFrame) -> str:
    try:
        return table.to_markdown()
    except ImportError:
        flat = table.copy()
        flat.columns = [" / ".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col) for col in flat.columns]
        return flat.to_csv(index=True)


def plot_metric_vs_ensemble(
    output_dir: Path,
    cfg: SummaryConfig,
    table: pd.DataFrame,
    metric: str,
    model: str,
    filename: str,
    ylabel: str,
    threshold_percentile: float | None = None,
) -> None:
    if table.empty or plt is None:
        return

    set_plot_style()
    fig, axes = plt.subplots(1, len(cfg.variables), figsize=(4.3 * len(cfg.variables), 3.4), constrained_layout=True)
    if len(cfg.variables) == 1:
        axes = [axes]

    markers = ("o", "s", "D", "^", "v", "P")
    conditions = sorted(table["conditioning"].dropna().unique()) if "conditioning" in table else [""]
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(cfg.sigmas) * len(conditions))))

    for ax, var_idx in zip(axes, cfg.variables):
        var_name = f"z{var_idx + 1}"
        subset = table[(table["model"] == model) & (table["variable"] == var_name)]
        if threshold_percentile is not None:
            subset = subset[np.isclose(subset["threshold_percentile"], threshold_percentile)]

        color_idx = 0
        for sigma in cfg.sigmas:
            for condition in conditions:
                s = subset[(subset["sigma"] == sigma) & (subset["conditioning"] == condition)].sort_values("ensemble_size")
                if s.empty:
                    continue
                label = rf"$\sigma={sigma:.0e}$, {condition}" if condition else rf"$\sigma={sigma:.0e}$"
                ax.plot(
                    s["ensemble_size"],
                    s[metric],
                    marker=markers[color_idx % len(markers)],
                    color=colors[color_idx],
                    label=label,
                )
                color_idx += 1

        ax.set_title(STATE_LABELS[var_idx])
        ax.set_xlabel("ensemble members used in training")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(cfg.ensemble_sizes))
        ax.legend(frameon=False)

    fig.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(fig)


def write_plots(output_dir: Path, cfg: SummaryConfig, pdf: pd.DataFrame, aot: pd.DataFrame) -> None:
    if plt is None:
        print("Matplotlib is not available; tables were written but plots were skipped.")
        return

    plot_metric_vs_ensemble(
        output_dir,
        cfg,
        pdf,
        metric="kl",
        model="corrected",
        filename="corrected_pdf_kl_vs_ensemble.png",
        ylabel=r"$D_{\mathrm{KL}}(p_{\mathrm{ref}}\Vert p_{\mathrm{model}})$",
    )
    plot_metric_vs_ensemble(
        output_dir,
        cfg,
        pdf,
        metric="log_l1",
        model="corrected",
        filename="corrected_pdf_logl1_vs_ensemble.png",
        ylabel=r"mean $|\log p_{\mathrm{ref}}-\log p_{\mathrm{model}}|$",
    )
    if not aot.empty:
        max_percentile = max(cfg.threshold_percentiles)
        plot_metric_vs_ensemble(
            output_dir,
            cfg,
            aot,
            metric="aot_w1",
            model="corrected",
            filename=f"corrected_aot_w1_q{max_percentile:.0f}_vs_ensemble.png",
            ylabel=r"$\mathcal{E}_{\mathrm{AOT}}$",
            threshold_percentile=max_percentile,
        )
        plot_metric_vs_ensemble(
            output_dir,
            cfg,
            aot,
            metric="event_frequency_error",
            model="corrected",
            filename=f"corrected_aot_frequency_q{max_percentile:.0f}_vs_ensemble.png",
            ylabel=r"$\mathcal{E}_{\mathrm{freq}}$",
            threshold_percentile=max_percentile,
        )


def parse_args() -> SummaryConfig:
    parser = argparse.ArgumentParser(description="Summarize low-dimensional corrector results.")
    parser.add_argument("--results_root", type=Path, default=SummaryConfig.results_root)
    parser.add_argument("--output_dir", type=Path, default=SummaryConfig.output_dir)
    parser.add_argument("--sigmas", type=parse_float_list, default=SummaryConfig.sigmas)
    parser.add_argument("--ensemble_sizes", type=parse_int_list, default=SummaryConfig.ensemble_sizes)
    parser.add_argument("--architectures", type=parse_str_list, default=SummaryConfig.architectures)
    parser.add_argument("--run_id_train", type=int, default=SummaryConfig.run_id_train)
    parser.add_argument("--run_id_test", type=int, default=SummaryConfig.run_id_test)
    parser.add_argument("--stride", type=int, default=SummaryConfig.stride)
    parser.add_argument("--repeat_indices", type=parse_int_list, default=SummaryConfig.repeat_indices)
    parser.add_argument("--threshold_percentiles", type=parse_float_list, default=SummaryConfig.threshold_percentiles)
    parser.add_argument("--variables", type=parse_int_list, default=SummaryConfig.variables)
    parser.add_argument("--pdf_grid_points", type=int, default=SummaryConfig.pdf_grid_points)
    args = parser.parse_args()

    return SummaryConfig(
        results_root=args.results_root,
        output_dir=args.output_dir,
        sigmas=args.sigmas,
        ensemble_sizes=args.ensemble_sizes,
        architectures=args.architectures,
        run_id_train=args.run_id_train,
        run_id_test=args.run_id_test,
        stride=args.stride,
        repeat_indices=args.repeat_indices,
        threshold_percentiles=args.threshold_percentiles,
        variables=args.variables,
        pdf_grid_points=args.pdf_grid_points,
    )


def main() -> None:
    cfg = parse_args()
    runs, pdf, aot = summarize(cfg)
    found_runs = runs[runs["status"] == "found"].copy()

    if found_runs.empty or "train_time" not in found_runs or "ml_stride" not in found_runs:
        output_dir = cfg.output_dir / "Tunknown_strideunknown"
        write_tables(cfg, output_dir, runs, pdf, aot)
        write_plots(output_dir, cfg, pdf, aot)
        print(f"Summary written to {output_dir}")
    else:
        groups = found_runs[["train_time", "ml_stride"]].dropna().drop_duplicates().sort_values(["train_time", "ml_stride"])
        for _, group in groups.iterrows():
            train_time = int(group["train_time"])
            ml_stride = int(group["ml_stride"])
            group_runs = filter_group_frame(runs, train_time, ml_stride)
            group_pdf = filter_group_frame(pdf, train_time, ml_stride)
            group_aot = filter_group_frame(aot, train_time, ml_stride)

            output_dir = cfg.output_dir / summary_group_name(group_runs)
            write_tables(cfg, output_dir, group_runs, group_pdf, group_aot)
            write_plots(output_dir, cfg, group_pdf, group_aot)
            print(f"Summary written to {output_dir}")

    print(f"Runs found: {(runs['status'] == 'found').sum()} / {len(runs)}")
    if not pdf.empty:
        corrected = pdf[pdf["model"] == "corrected"]
        print(f"Mean corrected KL: {corrected['kl'].mean():.6f}")
        print(f"Mean corrected logL1: {corrected['log_l1'].mean():.6f}")
    if not aot.empty:
        corrected_aot = aot[aot["model"] == "corrected"]
        print(f"Mean corrected AOT W1: {corrected_aot['aot_w1'].mean():.6f}")
        print(f"Mean corrected event-frequency error: {corrected_aot['event_frequency_error'].mean():.6f}")


if __name__ == "__main__":
    main()
