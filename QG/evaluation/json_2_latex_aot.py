from settings import EXPERIMENT_IDS, DISPLAY_NAMES
import json
import numpy as np
import pandas as pd


def _to_float_threshold_keys(threshold_dict):
    out = {}
    for k, v in threshold_dict.items():
        # compare_all_aot aggregates repeated runs as {"mean", "std"}.
        # Preserve the historical scalar table format by rendering the mean.
        if isinstance(v, dict):
            v = v.get("mean", np.nan)
        out[float(k)] = float(v)
    return out


def _flatten_aot_json(json_file, metric_key="AOT_W1", layer="avg12", variant="corrected"):
    with open(json_file, "r") as f:
        data = json.load(f)

    rows = []
    all_thresholds = set()

    for entry in data:
        cfg = entry.get("config", {})
        metrics = entry.get("aot_metrics", {})
        variant_metrics = metrics.get(variant, {})
        layer_metrics = variant_metrics.get(layer, {})

        by_threshold = _to_float_threshold_keys(layer_metrics.get(f"{metric_key}_by_threshold", {}))
        for threshold, value in by_threshold.items():
            rows.append(
                {
                    "T": int(cfg.get("T")),
                    "sig": float(cfg.get("sig", 0.0)),
                    "n_ens": int(cfg.get("n_ens", 1)),
                    "n_add": int(cfg.get("n_add", 0)),
                    "strategy": str(cfg.get("strategy", "det")),
                    "threshold": float(threshold),
                    "value": float(value),
                }
            )
            all_thresholds.add(float(threshold))

    if not rows:
        return pd.DataFrame(columns=["T", "sig", "n_ens", "n_add", "strategy", "threshold", "value"]), []

    df = pd.DataFrame(rows)
    thresholds = sorted(all_thresholds)
    return df, thresholds


def _metric_latex(metric_key):
    if metric_key == "AOT_W1":
        return r"$\mathcal{E}_{\mathrm{AOT}}$"
    if metric_key == "freq_error":
        return r"$\mathcal{E}_{\mathrm{freq}}$"
    return metric_key


def _layer_latex(layer):
    if layer == "psi1":
        return r"$\psi_1$"
    if layer == "psi2":
        return r"$\psi_2$"
    return r"avg$(\psi_1,\psi_2)$"


def build_aot_master_table(
    json_file,
    out_prefix="master_aot",
    metric_key="AOT_W1",
    layer="avg12",
    variant="corrected",
    strategies=None,
    T_list=None,
    sig_list=None,
    thresholds=None,
    caption_bottom=True,
):
    if strategies is None:
        strategies = EXPERIMENT_IDS

    df, thresholds_in_data = _flatten_aot_json(
        json_file,
        metric_key=metric_key,
        layer=layer,
        variant=variant,
    )
    if df.empty:
        print(f"[WARN] No rows in {json_file} for metric={metric_key}, layer={layer}")
        return None

    if thresholds is None:
        thresholds = thresholds_in_data
    thresholds = [float(t) for t in thresholds]

    df_det = df[df["strategy"] == "det"].copy()
    df_sig = df[df["strategy"] != "det"].copy()

    if sig_list is None:
        sig_list = sorted([float(s) for s in df_sig["sig"].unique() if s != 0.0])
    if T_list is None:
        T_list = sorted(int(t) for t in df["T"].unique())

    n_add_vals = sorted(int(n) for n in df_sig["n_add"].unique())

    tex = []
    tex.append("\\begin{table}[htbp]")
    tex.append("\\centering")

    metric_ltx = _metric_latex(metric_key)
    layer_ltx = _layer_latex(layer)
    variant_ltx = "coarse" if variant == "coarse" else "corrected"
    threshold_text = ", ".join(f"{t:.3g}" for t in thresholds)
    caption_text = (
        r"\\caption{Threshold-resolved %s on %s (%s) over long test trajectory. "
        r"Each cell reports values at thresholds $c\\in\\{%s\\}$. "
        r"Deterministic baseline is in the left column; stochastic columns are grouped by $\\sigma$ with rows $(N_{\\mathrm{add}},\\mathrm{strategy})$.}"
        % (metric_ltx, layer_ltx, variant_ltx, threshold_text)
    )
    label_text = f"\\label{{tab:{metric_key}_{layer}_{variant}_abs}}"

    if not caption_bottom:
        tex.append(caption_text)
        tex.append(label_text)

    tex.append("\\small")
    tex.append("\\setlength{\\tabcolsep}{3pt}")
    tex.append("\\renewcommand{\\arraystretch}{1.1}")
    tex.append("\\begin{tabular}{l" + "c" * (1 + len(sig_list)) + "}")
    tex.append("\\toprule")
    header = "det & " + " & ".join([f"${s}$" for s in sig_list])
    tex.append("$T$/$\\sigma$ & " + header + " \\\\ \\midrule")

    thr_header = " & ".join([f"$c={t:g}$" for t in thresholds])

    for T in T_list:
        tex.append("\\textbf{T=%d}" % T)

        det_vals = []
        for t in thresholds:
            sub = df_det[(df_det["T"] == T) & (np.isclose(df_det["threshold"], t))]
            if sub.empty:
                det_vals.append("--")
            else:
                item = sub["value"].iloc[0]
                val = float(item["mean"]) if isinstance(item, dict) else float(item)
                std = float(item.get("std", 0.0)) if isinstance(item, dict) else 0.0
                det_vals.append(f"{val:.3f} $\\pm$ {std:.3f}" if std > 0 else f"{val:.3f}")

        det_block = ["\\begin{tabular}{" + "c" * len(thresholds) + "}"]
        det_block.append(thr_header + " \\\\ \\cmidrule(lr){1-" + str(len(thresholds)) + "}")
        det_block.append(" & ".join(det_vals))
        det_block.append("\\end{tabular}")

        row_cells = [" & " + "".join(det_block)]

        for j, sig in enumerate(sig_list):
            sub_sig = df_sig[(df_sig["T"] == T) & np.isclose(df_sig["sig"], sig)]
            if sub_sig.empty:
                row_cells.append(" & -- ")
                continue

            if j == 0:
                block = ["\\begin{tabular}{r l|" + "c" * len(thresholds) + "}"]
                block.append(
                    "\\textbf{N$_{\\mathrm{add}}$} & \\textbf{strat} & "
                    + thr_header
                    + " \\\\ \\cmidrule(lr){1-"
                    + str(len(thresholds) + 2)
                    + "}"
                )
            else:
                block = ["\\begin{tabular}{r l|" + "c" * len(thresholds) + "}"]
                block.append(
                    "\\textbf{N$_{\\mathrm{add}}$} & \\textbf{strat} & "
                    + thr_header
                    + " \\\\ \\cmidrule(lr){1-"
                    + str(len(thresholds) + 2)
                    + "}"
                )

            for n_add in n_add_vals:
                for strat in strategies:
                    sub_row = sub_sig[(sub_sig["n_add"] == n_add) & (sub_sig["strategy"] == strat)]
                    if sub_row.empty:
                        continue
                    vals = []
                    for t in thresholds:
                        vv = sub_row[np.isclose(sub_row["threshold"], t)]["value"]
                        vals.append("%.3f" % float(vv.iloc[0]) if len(vv) > 0 else "--")
                    block.append(f"\\textbf{{{n_add}}} & {DISPLAY_NAMES.get(strat, strat)} & " + " & ".join(vals) + " \\\\")

            block.append("\\end{tabular}")
            row_cells.append(" & " + "".join(block))

        tex.append("".join(row_cells))
        tex.append(" \\\\ \\midrule")

    tex.append("\\bottomrule\\end{tabular}")

    if caption_bottom:
        tex.append(caption_text)
        tex.append(label_text)

    tex.append("\\end{table}")

    out = f"{out_prefix}_{metric_key}_{layer}.tex"
    with open(out, "w") as f:
        f.write("\n".join(tex))
    print(f"[SAVED] {out}")
    return out


def build_aot_all_tables(
    json_file,
    out_prefix="master_aot",
    variant="corrected",
    strategies=None,
    T_list=None,
    sig_list=None,
    thresholds=None,
    layers=("psi1", "psi2", "avg12"),
):
    outputs = []
    for metric_key in ("AOT_W1", "freq_error"):
        for layer in layers:
            out = build_aot_master_table(
                json_file=json_file,
                out_prefix=out_prefix,
                metric_key=metric_key,
                layer=layer,
                variant=variant,
                strategies=strategies,
                T_list=T_list,
                sig_list=sig_list,
                thresholds=thresholds,
                caption_bottom=True,
            )
            if out is not None:
                outputs.append(out)
    return outputs
