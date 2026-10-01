from settings import EXPERIMENT_IDS, DISPLAY_NAMES
import json
import pandas as pd
import numpy as np

def _get_val(x):
    if isinstance(x, dict):
        return x.get("mean", float(x.get("mean", 0.0))), x.get("std", 0.0)
    return float(x), 0.0

def _get_mean(x):
    if isinstance(x, dict):
        return float(x.get("mean", 0.0))
    return float(x)

def build_master_table(json_file, out_prefix="master", strategies=EXPERIMENT_IDS,
                       T_list=None, sig_list=None, use_bold=True, use_color=True, caption_bottom=True, strategy_display_names=None):
    with open(json_file, "r") as f:
        data = json.load(f)

    if strategy_display_names is None:
        strategy_display_names = DISPLAY_NAMES

    # Flatten JSON -> rows
    rows = []
    for e in data:
        parts = e["label"].split("-")
        sig   = next((float(p[3:]) for p in parts if p.startswith("sig")), 0.0)
        n_en  = next((int(p[3:])   for p in parts if p.startswith("ens")), 1)
        T     = next((int(p[1:])   for p in parts if p.startswith("T")),   None)
        strat = next((p            for p in parts if p.startswith("S")),   "det")
        for m in ["KL", "L1log", "W1"]:
            rows.append([T, sig, n_en, strat, e[m]["avg12"], m])

    df = pd.DataFrame(rows, columns=["T","sig","n_ens","strategy","value","metric"])
    df_det = df[df["strategy"]=="det"].copy()
    df_sig = df[df["strategy"]!="det"].copy()

    # Ordering
    sig_list = sorted([s for s in df_sig["sig"].unique() if s != 0.0]) if sig_list is None else sig_list
    T_list   = sorted(df_sig["T"].unique()) if T_list is None else T_list

    # Build N_add rows (invert S1 +1 saving convention)
    n_add_vals = []
    for _, r in df_sig[["n_ens","strategy"]].drop_duplicates().iterrows():
        n_add = r["n_ens"] if r["strategy"] != "S1" else r["n_ens"] - 1
        if n_add > 0: n_add_vals.append(n_add)
    NADD = sorted(set(n_add_vals))

    # Global minima over all σ, per (T, N_add)
    def global_min_for_T(metric, T):
        gmin = {}
        for n_add in NADD:
            vals = []
            for sig in sig_list:
                sub = df_sig[(df_sig["T"]==T) & (df_sig["sig"]==sig) & (df_sig["metric"]==metric)]
                if sub.empty:
                    continue
                for s in strategies:
                    n_query = n_add if s != "S1" else n_add + 1
                    vq = sub.query(f"n_ens=={n_query} and strategy=='{s}'").value
                    if len(vq)>0:
                        vals.append(_get_mean(vq.values[0]))
            gmin[n_add] = (min(vals) if len(vals)>0 else np.inf)
        return gmin

    for metric in ["KL","L1log","W1"]:
        tex = []
        tex.append("\\begin{table}[htbp]")
        tex.append("\\centering")

        # Custom label for each metric: tab:KL_abs, tab:L1log_abs, tab:W1_abs
        if metric == "L1log":
            label = "tab:logL1_abs"
            metric_latex = "$\\log(L_1)$"
        elif metric == "KL":
            label = f"tab:{metric}_abs"
            metric_latex = "$D_{\\mathrm{KL}}$"
        else:
            label = f"tab:{metric}_abs"
            metric_latex = f"${metric}$"

        # Caption at top or bottom
        caption_text = "\\caption{Global %s vs $T$, deterministic baseline (left), and stochastic strategies across $\\sigma$ with rows = additional ensemble members $N_{\\text{add}}$.}" % metric_latex
        label_text = f"\\label{{{label}}}"

        if not caption_bottom:
            tex.append(caption_text)
            tex.append(label_text)

        tex.append("\\small")
        tex.append("\\setlength{\\tabcolsep}{3pt}")
        tex.append("\\renewcommand{\\arraystretch}{1.1}")
        # det + |sig| columns
        tex.append("\\begin{tabular}{l" + "c"*(1+len(sig_list)) + "}")
        tex.append("\\toprule")
        header = "det & " + " & ".join([f"${s}$" for s in sig_list])
        # Header cell: $T$/$\sigma$
        tex.append("$T$/$\\sigma$ & " + header + " \\\\ \\midrule")

        for T in T_list:
            gmin = global_min_for_T(metric, T)

            tex.append("\\textbf{T=%d}" % T)

            # deterministic cell
            sub_det = df_det.query(f"T=={T} and metric=='{metric}'")
            if sub_det.empty:
                det_cell = "--"
                v_det_mean = np.inf
            else:
                v_det_mean, v_det_std = _get_val(sub_det["value"].iloc[0])
                if v_det_std > 0:
                    det_cell = "\\begin{tabular}{c}%.3f $\\pm$ %.3f\\end{tabular}" % (v_det_mean, v_det_std)
                else:
                    det_cell = "\\begin{tabular}{c}%.3f\\end{tabular}" % v_det_mean
            row_cells = [" & " + det_cell]

            # σ subtables
            for j, sig in enumerate(sig_list):
                sub = df_sig.query(f"T=={T} and sig=={sig} and metric=='{metric}'")
                if sub.empty:
                    row_cells.append(" & -- ")
                    continue

                if j == 0:
                    # First σ: include N_add column + strategy header
                    block = ["\\begin{tabular}{r|" + "c"*len(strategies) + "}"]
                    block.append("\\textbf{N$_{\\text{add}}$} & " + " & ".join(strategy_display_names[s] for s in strategies) +
                                     " \\\\ \\cmidrule(lr){1-" + str(len(strategies)+1) + "}")
                else:
                    # Other σ: ONLY strategy columns, but still show the header row
                    block = ["\\begin{tabular}{" + "c"*len(strategies) + "}"]
                    block.append(" " + " & ".join(strategy_display_names[s] for s in strategies) + " \\\\ \\cmidrule(lr){1-" + str(len(strategies)) + "}")

                for n_add in NADD:
                    # gather row values (strategy) for this σ
                    row_vals = []
                    for s in strategies:
                        n_query = n_add if s != "S1" else n_add + 1
                        vq = sub.query(f"n_ens=={n_query} and strategy=='{s}'").value
                        row_vals.append(_get_val(vq.values[0]) if len(vq)>0 else (np.inf, 0.0))

                    # compute local (per-σ) row min and global (across all σ) row min
                    means = [x[0] for x in row_vals]
                    local_min = np.min(means) if len(means)>0 else np.inf
                    global_min = gmin.get(n_add, np.inf)

                    # render values:
                    # - global best → bold + Blue
                    # - else local best → bold
                    # - else plain
                    vals_rendered = []
                    for (v, std) in row_vals:
                        if not np.isfinite(v):
                            vals_rendered.append("--")
                            continue
                        formatted = "%.3f $\pm$ %.3f" % (v, std) if std > 0 else "%.3f" % v
                        if np.isclose(v, global_min):
                            if use_bold and use_color:
                                vals_rendered.append("\\textbf{\\textcolor{Blue}{%s}}" % formatted)
                            elif use_bold:
                                vals_rendered.append("\\textbf{%s}" % formatted)
                            elif use_color:
                                vals_rendered.append("\\textcolor{Blue}{%s}" % formatted)
                            else:
                                vals_rendered.append(formatted)
                        elif np.isclose(v, local_min):
                            if use_bold:
                                vals_rendered.append("\\textbf{%s}" % formatted)
                            else:
                                vals_rendered.append(formatted)
                        else:
                            vals_rendered.append(formatted)

                    if j == 0:
                        block.append("\\textbf{%d} & " % n_add + " & ".join(vals_rendered) + " \\\\")
                    else:
                        block.append(" " + " & ".join(vals_rendered) + " \\\\")

                block.append("\\end{tabular}")
                row_cells.append(" & " + "".join(block))

            tex.append("".join(row_cells))
            tex.append(" \\\\ \\midrule")

        tex.append("\\bottomrule\\end{tabular}")

        # Caption at bottom if requested
        if caption_bottom:
            tex.append(caption_text)
            tex.append(label_text)

        tex.append("\\end{table}")
        out = f"{out_prefix}_{metric}.tex"
        with open(out, "w") as f:
            f.write("\n".join(tex))
        print("[SAVED] %s" % out)


def build_master_table_relative(json_file, out_prefix="master_rel", strategies=EXPERIMENT_IDS,
                                 T_list=None, sig_list=None, use_bold=True, use_color=True, caption_bottom=True, strategy_display_names=None):
    with open(json_file, "r") as f:
        data = json.load(f)

    if strategy_display_names is None:
        strategy_display_names = DISPLAY_NAMES
    # Flatten JSON -> rows
    rows = []
    for e in data:
        parts = e["label"].split("-")
        sig   = next((float(p[3:]) for p in parts if p.startswith("sig")), 0.0)
        n_en  = next((int(p[3:])   for p in parts if p.startswith("ens")), 1)
        T     = next((int(p[1:])   for p in parts if p.startswith("T")),   None)
        strat = next((p            for p in parts if p.startswith("S")),   "det")
        for m in ["KL", "L1log", "W1"]:
            rows.append([T, sig, n_en, strat, e[m]["avg12"], m])

    df = pd.DataFrame(rows, columns=["T","sig","n_ens","strategy","value","metric"])
    df_det = df[df["strategy"]=="det"].copy()
    df_sig = df[df["strategy"]!="det"].copy()

    sig_list = sorted([s for s in df_sig["sig"].unique() if s != 0.0]) if sig_list is None else sig_list
    T_list   = sorted(df_sig["T"].unique()) if T_list is None else T_list


    # Build N_add (invert S1 +1 saving convention)
    n_add_vals = []
    for _, r in df_sig[["n_ens","strategy"]].drop_duplicates().iterrows():
        n_add = r["n_ens"] if r["strategy"] != "S1" else r["n_ens"] - 1
        if n_add > 0: n_add_vals.append(n_add)
    NADD = sorted(set(n_add_vals))

    # Deterministic baselines per (metric, T)
    det_map = {}
    for (m, T), grp in df_det.groupby(["metric","T"]):
        if not grp.empty:
            det_map[(m, T)] = _get_mean(grp["value"].iloc[0])

    # Global best (max improvement %) across ALL σ, per (metric, T, N_add)
    def global_best_impr_for_T(metric, T):
        gbest = {}
        v_det = det_map.get((metric, T), np.nan)
        for n_add in NADD:
            best = -np.inf
            if np.isfinite(v_det) and v_det > 0:
                for sig in sig_list:
                    sub = df_sig[(df_sig["T"]==T) & (df_sig["sig"]==sig) & (df_sig["metric"]==metric)]
                    if sub.empty:
                        continue
                    for s in strategies:
                        n_query = n_add if s != "S1" else n_add + 1
                        vq = sub.query("n_ens==@n_query and strategy==@s").value
                        if len(vq)>0:
                            v = _get_mean(vq.values[0])
                            impr = (v_det - v) / v_det * 100.0
                            if np.isfinite(impr):
                                if impr > best: best = impr
            gbest[n_add] = best
        return gbest

    for metric in ["KL","L1log","W1"]:
        tex = []
        tex.append("\\begin{table}[htbp]")
        tex.append("\\centering")

        # Custom label for each metric: tab:KL_rel, tab:logL1_rel, tab:W1_rel
        if metric == "L1log":
            label = "tab:logL1_rel"
            metric_latex = "$\\log(L_1)$"
        elif metric == "KL":
            label = f"tab:{metric}_rel"
            metric_latex = "$D_{\\mathrm{KL}}$"
        else:
            label = f"tab:{metric}_rel"
            metric_latex = f"${metric}$"

        # Caption at top or bottom
        caption_text = "\\caption{Percent change of %s vs $T$ relative to deterministic baseline (left). Positive values mean improvement (smaller is better). Stochastic strategies across $\\sigma$ with rows = additional ensemble members $N_{\\text{add}}$.}" % metric_latex
        label_text = f"\\label{{{label}}}"

        if not caption_bottom:
            tex.append(caption_text)
            tex.append(label_text)

        tex.append("\\small")
        tex.append("\\setlength{\\tabcolsep}{3pt}")
        tex.append("\\renewcommand{\\arraystretch}{1.1}")
        tex.append("\\begin{tabular}{l" + "c"*(1+len(sig_list)) + "}")
        tex.append("\\toprule")
        header = "det & " + " & ".join(["$%s$" % s for s in sig_list])
        tex.append("$T$/$\\sigma$ & " + header + " \\\\ \\midrule")

        for T in T_list:
            v_det = det_map.get((metric, T), np.nan)
            gbest = global_best_impr_for_T(metric, T)

            tex.append("\\textbf{T=%d}" % T)

            # det cell -> 0.00%
            det_cell = "--"
            if np.isfinite(v_det) and v_det > 0:
                det_cell = "\\begin{tabular}{c}0.00\\%\\end{tabular}"
            row_cells = [" & " + det_cell]

            for j, sig in enumerate(sig_list):
                sub = df_sig.query("T==@T and sig==@sig and metric==@metric")
                if sub.empty or not (np.isfinite(v_det) and v_det > 0):
                    row_cells.append(" & -- ")
                    continue

                if j == 0:
                    # Include N_add + strategy header
                    block = ["\\begin{tabular}{r|" + "c"*len(strategies) + "}"]
                    block.append("\\textbf{N$_{\\text{add}}$} & " +
                    " & ".join(strategy_display_names[s] for s in strategies) +
                    " \\\\ \\cmidrule(lr){1-" + str(len(strategies)+1) + "}")
                else:
                    # Only strategy columns but keep header
                    block = ["\\begin{tabular}{" + "c"*len(strategies) + "}"]
                    block.append(" " + " & ".join(strategy_display_names[s] for s in strategies) +
                                 " \\\\ \\cmidrule(lr){1-" + str(len(strategies)) + "}")

                for n_add in NADD:
                    # gather improvements for the row across strategies
                    row_impr = []
                    for s in strategies:
                        n_query = n_add if s != "S1" else n_add + 1
                        vq = sub.query("n_ens==@n_query and strategy==@s").value
                        if len(vq)>0:
                            v = _get_mean(vq.values[0])
                            impr = (v_det - v) / v_det * 100.0
                        else:
                            impr = -np.inf  # missing
                        row_impr.append(impr)

                    # local best (per-σ row) is max improvement
                    local_best = np.max(row_impr) if len(row_impr)>0 else -np.inf
                    global_best = gbest.get(n_add, -np.inf)

                    # render: global best → bold + Blue, local best → bold, else plain
                    vals_rendered = []
                    for impr in row_impr:
                        if not np.isfinite(impr):
                            vals_rendered.append("--")
                            continue
                        txt = "%.2f\\%%" % impr
                        if np.isclose(impr, global_best):
                            if use_bold and use_color:
                                vals_rendered.append("\\textbf{\\textcolor{Blue}{%s}}" % txt)
                            elif use_bold:
                                vals_rendered.append("\\textbf{%s}" % txt)
                            elif use_color:
                                vals_rendered.append("\\textcolor{Blue}{%s}" % txt)
                            else:
                                vals_rendered.append("%s" % txt)
                        elif np.isclose(impr, local_best):
                            if use_bold:
                                vals_rendered.append("\\textbf{%s}" % txt)
                            else:
                                vals_rendered.append("%s" % txt)
                        else:
                            vals_rendered.append("%s" % txt)

                    if j == 0:
                        block.append("\\textbf{%d} & " % n_add + " & ".join(vals_rendered) + " \\\\")
                    else:
                        block.append(" " + " & ".join(vals_rendered) + " \\\\")

                block.append("\\end{tabular}")
                row_cells.append(" & " + "".join(block))

            tex.append("".join(row_cells))
            tex.append(" \\\\ \\midrule")

        tex.append("\\bottomrule\\end{tabular}")

        # Caption at bottom if requested
        if caption_bottom:
            tex.append(caption_text)
            tex.append(label_text)

        tex.append("\\end{table}")
        out = "%s_%s.tex" % (out_prefix, metric)
        with open(out, "w") as f:
            f.write("\n".join(tex))
        print("[SAVED] %s" % out)
