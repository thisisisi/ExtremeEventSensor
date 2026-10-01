#!/usr/bin/env python3
# Author: Isabella Thiel <thieli@mit.edu>
"""Exact repeat-1 report for fixed controls plus deterministic baselines."""

import csv
import gc
import glob
import json
import os

import numpy as np

from settings import CONFIG, EXPERIMENT_IDS

from metrics import _load_result_file, compute_global_kl, compute_l1_logpdf, compute_wasserstein1
from compare_all_aot import (
    _results_save_path,
    _results_save_path2,
    _slice_at_threshold,
    _wasserstein_distance_empirical,
)


STRATEGIES = EXPERIMENT_IDS
THRESHOLDS = (0.5, 1.0, 1.5, 1.75)
LAYERS = ("psi1", "psi2")
DET_T_TRAINS = (50, 1000)
SIGMA = 0.01
N_ADD = 2
T_TRAIN = 50
REPEAT = 1
MODEL_TYPE = CONFIG.model_type
BASIS_TYPE = CONFIG.basis_type
N_EPOCHS = CONFIG.epochs
DT_ML = CONFIG.dt_ml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
RESULTS_DIR = os.environ.get("QG_RESULTS_DIR", os.path.join(PROJECT_ROOT, "ML_Results"))
OUT_DIR = os.path.join(os.environ.get("QG_PLOTS_DIR", os.path.join(PROJECT_ROOT, "ML_Plots")), "fixed_case_controls")


def result_path(strategy):
    case = (
        "Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_"
        "realization4_24x24_sig0.01"
    )
    n_members = N_ADD + 1 if strategy == "S1" else N_ADD
    return _results_save_path2(
        RESULTS_DIR, case, MODEL_TYPE, BASIS_TYPE, N_EPOCHS, DT_ML,
        T_TRAIN, 1, strategy, n_members, repeat_index=REPEAT,
    )


def deterministic_result_path(t_train):
    case_stem = (
        "Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_"
        "realization4_24x24"
    )
    for det_sig in ("0.01", "0.001"):
        case = f"{case_stem}_sig{det_sig}"
        candidates = (
            _results_save_path(
                RESULTS_DIR, case, MODEL_TYPE, BASIS_TYPE, N_EPOCHS,
                DT_ML, int(t_train), 1,
            ),
            _results_save_path(
                RESULTS_DIR, case, MODEL_TYPE, BASIS_TYPE, N_EPOCHS,
                DT_ML, int(t_train), 1, repeat_index=REPEAT,
            ),
        )
        for path in candidates:
            if os.path.exists(path):
                return path
    return candidates[0]


def aot_rows(data, strategy, t_train):
    """Return one scalar error per threshold/layer, plus the layer average."""
    threshold_grid = np.squeeze(np.asarray(data["thresh"], dtype=np.float64))
    ref = np.asarray(data["aot_ref"], dtype=np.float64)
    model = np.asarray(data["aot_ML"], dtype=np.float64)
    if model.ndim == 4:
        model = model.mean(axis=0)

    order = np.argsort(threshold_grid)
    threshold_grid = threshold_grid[order]
    ref, model = ref[order], model[order]
    rows = []
    for threshold in THRESHOLDS:
        ref_at = _slice_at_threshold(ref, threshold_grid, threshold)
        model_at = _slice_at_threshold(model, threshold_grid, threshold)
        layer_rows = []
        for layer_idx, layer in enumerate(LAYERS):
            x = np.asarray(ref_at[:, layer_idx]).ravel()
            y = np.asarray(model_at[:, layer_idx]).ravel()
            layer_rows.append({
                "strategy": strategy,
                "T_train": int(t_train),
                "threshold": float(threshold),
                "layer": layer,
                "epsilon_AOT_W1": _wasserstein_distance_empirical(x, y),
                "epsilon_freq": abs(float(np.mean(x > 0.0)) - float(np.mean(y > 0.0))),
            })
        rows.extend(layer_rows)
        rows.append({
            "strategy": strategy,
            "T_train": int(t_train),
            "threshold": float(threshold),
            "layer": "avg12",
            "epsilon_AOT_W1": 0.5 * sum(r["epsilon_AOT_W1"] for r in layer_rows),
            "epsilon_freq": 0.5 * sum(r["epsilon_freq"] for r in layer_rows),
        })
    return rows


def _all_numbered_result_paths(strategy, t_train):
    """All independent numbered results for one method; legacy base only as fallback."""
    if strategy == "det":
        base = deterministic_result_path(t_train)
        marker = "_rep"
        if marker in base:
            base = base[:base.rfind(marker)] + ".mat"
        pattern = base[:-4] + "_rep*.mat"
    else:
        base = result_path(strategy)
        marker = "_rep"
        base = base[:base.rfind(marker)] + "_Ensemble_Stats_Big.mat"
        pattern = base.replace(
            "_Ensemble_Stats_Big.mat",
            "_rep*_Ensemble_Stats_Big.mat",
        )
    numbered = sorted(glob.glob(pattern))
    if numbered:
        return numbered
    return [base] if os.path.exists(base) else []


def aggregate_all_available_rows():
    """Aggregate every available numbered run independently for each method."""
    specs = [("det", t_train) for t_train in DET_T_TRAINS]
    compact_strategies = (
        "S4", "S0", "S10", "S11", "S12", "S13", "S14", "S15",
    )
    specs.extend((strategy, T_TRAIN) for strategy in compact_strategies)
    normal_runs = {}
    exceedance_runs = {}
    run_counts = {}

    for strategy, t_train in specs:
        paths = _all_numbered_result_paths(strategy, t_train)
        run_counts[f"{strategy}_T{t_train}"] = len(paths)
        for path in paths:
            data = _load_result_file(path)
            if data is None:
                continue
            kl = compute_global_kl(data, target="ml")
            l1 = compute_l1_logpdf(data, target="ml")
            key = (strategy, int(t_train))
            normal_runs.setdefault(key, []).append({
                "KL_psi1": float(kl["psi1"]),
                "KL_psi2": float(kl["psi2"]),
                "KL_avg12": float(kl["avg12"]),
                "L1log_psi1": float(l1["psi1"]),
                "L1log_psi2": float(l1["psi2"]),
                "L1log_avg12": float(l1["avg12"]),
            })
            for row in aot_rows(data, strategy, t_train):
                row_key = (
                    strategy, int(t_train), float(row["threshold"]), row["layer"]
                )
                exceedance_runs.setdefault(row_key, []).append(row)
            del data
            gc.collect()

    normal_rows = []
    for (strategy, t_train), rows in normal_runs.items():
        averaged = {
            field: float(np.mean([row[field] for row in rows]))
            for field in rows[0]
        }
        normal_rows.append({
            "strategy": strategy,
            "T_train": t_train,
            **averaged,
        })

    exceedance_rows = []
    for (strategy, t_train, threshold, layer), rows in exceedance_runs.items():
        exceedance_rows.append({
            "strategy": strategy,
            "T_train": t_train,
            "threshold": threshold,
            "layer": layer,
            "epsilon_AOT_W1": float(np.mean([
                row["epsilon_AOT_W1"] for row in rows
            ])),
            "epsilon_freq": float(np.mean([
                row["epsilon_freq"] for row in rows
            ])),
        })
    return normal_rows, exceedance_rows, run_counts


def markdown_table(normal_rows, exceedance_rows):
    lines = [
        "# Fixed-case mechanism-control metrics",
        "",
        "Controls: T_train=50, sigma=0.01, N_add=2, repeat=1. "
        "Deterministic baselines: T_train=50 and 1000.",
        "",
        "## PDF metrics",
        "",
        "| Strategy | T_train | KL psi1 | KL psi2 | KL avg12 | log-L1 psi1 | log-L1 psi2 | log-L1 avg12 | PDF-W1 psi1 | PDF-W1 psi2 | PDF-W1 avg12 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in normal_rows:
        lines.append(
            "| {strategy} | {T_train} | {KL_psi1:.6g} | {KL_psi2:.6g} | {KL_avg12:.6g} | "
            "{L1log_psi1:.6g} | {L1log_psi2:.6g} | {L1log_avg12:.6g} | "
            "{PDF_W1_psi1:.6g} | {PDF_W1_psi2:.6g} | {PDF_W1_avg12:.6g} |".format(**row)
        )
    lines.extend([
        "",
        "## Area-over-threshold metrics",
        "",
        "| Strategy | T_train | c | Layer | epsilon_AOT (W1) | epsilon_freq |",
        "|---|---:|---:|---|---:|---:|",
    ])
    for row in exceedance_rows:
        lines.append(
            "| {strategy} | {T_train} | {threshold:g} | {layer} | "
            "{epsilon_AOT_W1:.6g} | {epsilon_freq:.6g} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def compact_markdown_tables(normal_rows, exceedance_rows):
    """Return the three requested paper-style compact Markdown tables."""
    normal = {
        (row["strategy"], int(row["T_train"])): row for row in normal_rows
    }
    exceedance = {}
    for row in exceedance_rows:
        key = (row["strategy"], int(row["T_train"]))
        exceedance.setdefault(key, {}).setdefault(row["layer"], []).append(row)

    def values(strategy, t_train=50):
        key = (strategy, int(t_train))
        pdf = normal.get(key)
        aot = exceedance.get(key, {})
        if pdf is None or not all(layer in aot for layer in LAYERS + ("avg12",)):
            return None
        base = (
            float(pdf["KL_avg12"]),
            float(pdf["L1log_avg12"]),
            float(np.mean([row["epsilon_AOT_W1"] for row in aot["avg12"]])),
            float(np.mean([row["epsilon_freq"] for row in aot["avg12"]])),
        )
        layer_extremes = tuple(
            value
            for layer in LAYERS
            for value in (
                float(np.mean([row["epsilon_AOT_W1"] for row in aot[layer]])),
                float(np.mean([row["epsilon_freq"] for row in aot[layer]])),
            )
        )
        return base + layer_extremes

    main_rows = [
        ("STORN (baseline), $T_{\\mathrm{train}}=1000$", "det", 1000),
        ("STORN (baseline), $T_{\\mathrm{train}}=50$", "det", 50),
        ("*FiLM-STORN, $T_{\\mathrm{train}}=50$ (ours)*", "S4", 50),
    ]
    control_rows = [
        ("Covariance direction + FiLM", "S0", 50),
        ("Random direction + FiLM", "S10", 50),
        ("Coarse statistics + FiLM", "S11", 50),
        ("Ensemble mean only + FiLM", "S12", 50),
        ("Ensemble variance only + FiLM", "S13", 50),
        ("Covariance direction + concat.", "S14", 50),
        ("Mean/spread + concat.", "S15", 50),
    ]

    def make_table(extra_layers=(), percentages=False):
        indices = [0, 1, 2, 3]
        headers = [
            "$D_{\\mathrm{KL}}\\!\\downarrow$",
            "$L_{\\log}\\!\\downarrow$",
            "$\\overline{E}_{\\mathrm{AOT}}\\!\\downarrow$",
            "$\\overline{E}_{\\mathrm{freq}}\\!\\downarrow$",
        ]
        for layer in extra_layers:
            offset = 4 if layer == "psi1" else 6
            indices.extend((offset, offset + 1))
            subscript = "1" if layer == "psi1" else "2"
            headers.extend((
                f"$\\overline{{E}}_{{\\mathrm{{AOT}},\\psi_{subscript}}}\\!\\downarrow$",
                f"$\\overline{{E}}_{{\\mathrm{{freq}},\\psi_{subscript}}}\\!\\downarrow$",
            ))

        row_values = {
            (strategy, t_train): values(strategy, t_train)
            for _, strategy, t_train in main_rows + control_rows
        }
        ours = row_values[("S4", 50)]

        def section_minima(section):
            available = [
                row_values[(strategy, t_train)]
                for _, strategy, t_train in section
                if row_values[(strategy, t_train)] is not None
            ]
            return {
                index: min(row[index] for row in available) for index in indices
            }

        def render(section, minima):
            output = []
            for description, strategy, t_train in section:
                metric_values = row_values[(strategy, t_train)]
                if metric_values is None:
                    cells = ["—"] * len(indices)
                else:
                    cells = []
                    for index in indices:
                        digits = 4 if index == 1 else 5
                        cell = f"{metric_values[index]:.{digits}f}"
                        if np.isclose(metric_values[index], minima[index]):
                            cell = f"**{cell}**"
                        if percentages:
                            change = 100.0 * (
                                metric_values[index] / ours[index] - 1.0
                            )
                            comparison = (
                                "reference" if strategy == "S4"
                                else f"{change:+.1f}%"
                            )
                            cell += f" *({comparison})*"
                        cells.append(cell)
                output.append(f"| {description} | " + " | ".join(cells) + " |")
            return output

        lines = [
            "| Method | " + " | ".join(headers) + " |",
            "|---|" + "|".join("---:" for _ in headers) + "|",
            "| **Main comparison** |" + " |" * len(headers),
        ]
        lines.extend(render(main_rows, section_minima(main_rows)))
        lines.append("| **Mechanism controls** |" + " |" * len(headers))
        lines.extend(render(control_rows, section_minima(control_rows)))
        return "\n".join(lines) + "\n"

    return {
        "compact_extremes_both_layers.md": make_table(("psi1", "psi2")),
        "compact_extremes_psi2.md": make_table(("psi2",)),
        "compact_percent_vs_ours.md": make_table(
            ("psi2",), percentages=True
        ),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    normal_rows, exceedance_rows, used_paths = [], [], {}
    result_specs = [
        ("det", t_train, deterministic_result_path(t_train))
        for t_train in DET_T_TRAINS
    ]
    result_specs.extend(
        (strategy, T_TRAIN, result_path(strategy)) for strategy in STRATEGIES
    )
    for strategy, t_train, path in result_specs:
        data = _load_result_file(path)
        if data is None:
            raise FileNotFoundError("Missing or unreadable repeat-1 result: {}".format(path))
        required = ("psi_rpdf_ref", "psi_rpdf_ML", "thresh", "aot_ref", "aot_ML")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError("{} lacks {}".format(path, missing))

        kl = compute_global_kl(data, target="ml")
        l1 = compute_l1_logpdf(data, target="ml")
        pdf_w1 = compute_wasserstein1(data, target="ml")
        normal_rows.append({
            "strategy": strategy,
            "T_train": int(t_train),
            "KL_psi1": float(kl["psi1"]),
            "KL_psi2": float(kl["psi2"]),
            "KL_avg12": float(kl["avg12"]),
            "L1log_psi1": float(l1["psi1"]),
            "L1log_psi2": float(l1["psi2"]),
            "L1log_avg12": float(l1["avg12"]),
            "PDF_W1_psi1": float(pdf_w1["psi1"]),
            "PDF_W1_psi2": float(pdf_w1["psi2"]),
            "PDF_W1_avg12": float(pdf_w1["avg12"]),
        })
        exceedance_rows.extend(aot_rows(data, strategy, t_train))
        used_paths[f"{strategy}_T{t_train}"] = path

    payload = {
        "configuration": {
            "control_T_train": T_TRAIN,
            "deterministic_T_train": list(DET_T_TRAINS),
            "sigma": SIGMA, "N_add": N_ADD,
            "repeat": REPEAT, "thresholds": list(THRESHOLDS),
        },
        "paths": used_paths,
        "normal_metrics": normal_rows,
        "exceedance_metrics": exceedance_rows,
    }
    with open(os.path.join(OUT_DIR, "fixed_case_metrics.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    with open(os.path.join(OUT_DIR, "normal_metrics.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normal_rows[0]))
        writer.writeheader()
        writer.writerows(normal_rows)
    with open(os.path.join(OUT_DIR, "exceedance_metrics.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exceedance_rows[0]))
        writer.writeheader()
        writer.writerows(exceedance_rows)
    report = markdown_table(normal_rows, exceedance_rows)
    report_path = os.path.join(OUT_DIR, "fixed_case_metrics.md")
    with open(report_path, "w") as handle:
        handle.write(report)
    compact_reports = compact_markdown_tables(normal_rows, exceedance_rows)
    compact_paths = []
    for filename, compact_report in compact_reports.items():
        compact_path = os.path.join(OUT_DIR, filename)
        with open(compact_path, "w") as handle:
            handle.write(compact_report)
        compact_paths.append(compact_path)

    max_normal, max_exceedance, max_run_counts = aggregate_all_available_rows()
    max_reports = compact_markdown_tables(max_normal, max_exceedance)
    for filename, compact_report in max_reports.items():
        stem, extension = os.path.splitext(filename)
        compact_path = os.path.join(OUT_DIR, f"{stem}_max{extension}")
        with open(compact_path, "w") as handle:
            handle.write(compact_report)
        compact_paths.append(compact_path)
    with open(os.path.join(OUT_DIR, "compact_max_run_counts.json"), "w") as handle:
        json.dump(max_run_counts, handle, indent=2, sort_keys=True)
    print(report)
    print("Saved report:", report_path)
    for compact_path in compact_paths:
        print("Saved compact report:", compact_path)


if __name__ == "__main__":
    main()
