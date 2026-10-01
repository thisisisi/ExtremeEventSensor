# Author: Isabella Thiel <thieli@mit.edu>
from settings import CONFIG, EXPERIMENT_IDS
import json
import os
import numpy as np
import hdf5storage

from json_2_latex_aot import build_aot_all_tables


# === CONFIGURATION ===
SIG_LIST = CONFIG.sigmas
STRATEGIES = EXPERIMENT_IDS
N_ENS_LIST = CONFIG.ensemble_sizes
NTIME_UNITS = CONFIG.training_durations
DET_NTIME_UNITS = CONFIG.unconditioned_durations
TABLE_NTIME_UNITS = sorted(set(NTIME_UNITS + DET_NTIME_UNITS))

MODEL_TYPE = CONFIG.model_type
BASIS_TYPE = CONFIG.basis_type
N_EPOCHS = CONFIG.epochs
DT_ML = CONFIG.dt_ml
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
RESULTS_DIR = os.environ.get("QG_RESULTS_DIR", os.path.join(PROJECT_ROOT, "ML_Results"))
OUT_DIR = os.environ.get("QG_PLOTS_DIR", os.path.join(PROJECT_ROOT, "ML_Plots"))

MAIN_THRESHOLDS = (1.5, 1.75)
APPENDIX_THRESHOLDS = (0.5, 1.0, 1.5, 1.75) #AoT thresholds 
MODEL_VARIANTS = (
    ("coarse", "aot_coarse"),
    ("corrected", "aot_ML"),
)


def _load_result_file(path):
    if not os.path.isfile(path):
        print(f"    [WARN] File not found: {path}")
        return None
    try:
        return hdf5storage.loadmat(path)
    except Exception as e:
        print(f"    [ERROR] Could not load {path}: {e}")
        return None


def _results_save_path(results_dir, case_name, model_type, basis_type, n_epochs, dt_ml, ntime, n_ens, repeat_index=None):
    fname = (
        f"{case_name}_{model_type}_{basis_type}_loss_L2_epochs_{n_epochs}_"
        f"dt_{dt_ml}_T_{ntime}_Nens_{n_ens}_Ensemble_Stats_Big"
    )
    if repeat_index is not None:
        fname += f"_rep{repeat_index:03d}"
    fname += ".mat"
    return os.path.join(results_dir, fname)


def _results_save_path2(
    results_dir,
    case_name,
    model_type,
    basis_type,
    n_epochs,
    dt_ml,
    ntime,
    n_ens,
    strategy,
    n_members,
    repeat_index=None,
):
    base = (
        f"{case_name}_{model_type}_{basis_type}_loss_L2_epochs_{n_epochs}_"
        f"dt_{dt_ml}_T_{ntime}_Nens_{n_ens}"
    )
    strategy = str(strategy).upper()
    toks = [f"S{strategy}", f"k{int(n_members)}"]
    fname = f"{base}_{'-'.join(toks)}"
    if repeat_index is not None:
        fname += f"_rep{repeat_index:03d}"
    fname += "_Ensemble_Stats_Big.mat"
    return os.path.join(results_dir, fname)


def _resolve_results_file(results_dir, cfg):
    sig, n_ens, ntime, strategy, model_type, n_epochs, dt_ml, basis_type = cfg
    files_found = []

    if strategy is None:
        case_stem = "Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24"
        for det_sig in ("0.01", "0.001"):
            case_name_det = f"{case_stem}_sig{det_sig}"
            candidates = [
                _results_save_path(
                    results_dir, case_name_det, model_type, basis_type,
                    n_epochs, dt_ml, ntime, 1,
                )
            ]
            candidates.extend(
                _results_save_path(
                    results_dir, case_name_det, model_type, basis_type,
                    n_epochs, dt_ml, ntime, 1, repeat_index=rep,
                )
                for rep in range(1, 20)
            )
            files_found.extend(path for path in candidates if os.path.exists(path))
            if files_found:
                break

    else:
        case_name = f"Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig{sig}"
        # Use only the three preregistered repeats; exclude unnumbered legacy
        # outputs and any historical rep004/rep005 files.
        for rep in range(1, 4):
            p_rep = _results_save_path2(results_dir, case_name, model_type, basis_type, n_epochs, dt_ml, ntime, 1, strategy=strategy, n_members=n_ens, repeat_index=rep)
            if os.path.exists(p_rep) and p_rep not in files_found:
                files_found.append(p_rep)

    return files_found


def _wasserstein_distance_empirical(x, y):
    """NumPy-only empirical 1D Wasserstein-1 distance for sample arrays."""
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    y = np.sort(np.asarray(y, dtype=np.float64).ravel())

    if x.size == 0 or y.size == 0:
        return np.nan

    if x.size == y.size:
        return float(np.mean(np.abs(x - y)))

    q = np.linspace(0.0, 1.0, max(x.size, y.size), endpoint=True)
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)
    return float(np.mean(np.abs(xq - yq)))


def _slice_at_threshold(arr, thresh, c):
    """Linearly interpolate arr[threshold, ...] to an arbitrary threshold c."""
    c = float(c)

    if c <= float(thresh[0]):
        return arr[0, ...]
    if c >= float(thresh[-1]):
        return arr[-1, ...]

    k1 = int(np.searchsorted(thresh, c, side="left"))
    if np.isclose(float(thresh[k1]), c):
        return arr[k1, ...]

    k0 = k1 - 1
    t0 = float(thresh[k0])
    t1 = float(thresh[k1])
    w = (c - t0) / (t1 - t0)
    return (1.0 - w) * arr[k0, ...] + w * arr[k1, ...]


def _cfg_to_meta(cfg):
    sig, n_ens, ntime, strategy, model_type, n_epochs, dt_ml, basis_type = cfg
    strategy_name = "det" if strategy is None else str(strategy)
    if strategy_name == "S1":
        n_add = int(n_ens) - 1
    elif strategy_name == "det":
        n_add = 0
    else:
        n_add = int(n_ens)

    return {
        "sig": float(sig),
        "n_ens": int(n_ens),
        "n_add": int(n_add),
        "T": int(ntime),
        "strategy": strategy_name,
        "model_type": str(model_type),
        "n_epochs": int(n_epochs),
        "dt_ml": float(dt_ml),
        "basis_type": str(basis_type),
    }


def _build_all_configs():
    all_configs = []
    for T in DET_NTIME_UNITS:
        det_cfg = (0.0, 1, T, None, MODEL_TYPE, N_EPOCHS, DT_ML, BASIS_TYPE)
        all_configs.append(det_cfg)
    for T in NTIME_UNITS:
        for sig in SIG_LIST:
            for strat in STRATEGIES:
                for n_ens in N_ENS_LIST:
                    n_run = n_ens + 1 if strat == "S1" else n_ens
                    cfg = (sig, n_run, T, strat, MODEL_TYPE, N_EPOCHS, DT_ML, BASIS_TYPE)
                    all_configs.append(cfg)
    return all_configs


def compute_aot_metrics(data, model_key, tail_thresholds, collapse_realizations=True):
    thresh = np.squeeze(np.asarray(data["thresh"], dtype=np.float64))
    ref = np.asarray(data["aot_ref"], dtype=np.float64)
    model = np.asarray(data[model_key], dtype=np.float64)

    # Ensure threshold axis is monotone increasing for interpolation.
    order = np.argsort(thresh)
    thresh = thresh[order]
    ref = ref[order, ...]
    model = model[order, ...]

    aggregation_mode = "direct"
    if collapse_realizations and model.ndim == 4:
        model = np.mean(model, axis=0)
        aggregation_mode = "ensemble_mean"

    out = {}
    layer_names = ["psi1", "psi2"]

    for layer_idx, layer_name in enumerate(layer_names):
        w1_vals = []
        freq_vals = []
        w1_by_thr = {}
        freq_by_thr = {}

        for c in tail_thresholds:
            ref_slice = _slice_at_threshold(ref, thresh, c)
            model_slice = _slice_at_threshold(model, thresh, c)
            x = ref_slice[:, layer_idx].ravel()
            y = model_slice[:, layer_idx].ravel()

            w1 = _wasserstein_distance_empirical(x, y)
            freq_err = float(abs(np.mean(x > 0.0) - np.mean(y > 0.0)))

            w1_vals.append(w1)
            freq_vals.append(freq_err)
            w1_by_thr[float(c)] = w1
            freq_by_thr[float(c)] = freq_err

        out[layer_name] = {
            "AOT_W1_mean": float(np.mean(w1_vals)),
            "freq_error_mean": float(np.mean(freq_vals)),
            "AOT_W1_by_threshold": w1_by_thr,
            "freq_error_by_threshold": freq_by_thr,
        }

    out["avg12"] = {
        "AOT_W1_mean": 0.5 * (out["psi1"]["AOT_W1_mean"] + out["psi2"]["AOT_W1_mean"]),
        "freq_error_mean": 0.5 * (out["psi1"]["freq_error_mean"] + out["psi2"]["freq_error_mean"]),
        "AOT_W1_by_threshold": {
            float(c): 0.5 * (
                out["psi1"]["AOT_W1_by_threshold"][float(c)] + out["psi2"]["AOT_W1_by_threshold"][float(c)]
            )
            for c in tail_thresholds
        },
        "freq_error_by_threshold": {
            float(c): 0.5 * (
                out["psi1"]["freq_error_by_threshold"][float(c)]
                + out["psi2"]["freq_error_by_threshold"][float(c)]
            )
            for c in tail_thresholds
        },
    }

    out["__meta__"] = {
        "model_key": str(model_key),
        "aggregation_mode": aggregation_mode,
    }

    return out


def run_all_aot_metrics_for_configs(configs, results_dir=RESULTS_DIR, out_dir=OUT_DIR, tail_thresholds=MAIN_THRESHOLDS):
    os.makedirs(out_dir, exist_ok=True)

    summary = []
    for cfg in configs:
        paths = _resolve_results_file(results_dir, cfg)
        if not paths:
            print(f"[WARN] Could not resolve file for config={cfg}")
            continue

        runs_aot = []
        for path in paths:
            data = _load_result_file(path)
            if data is None:
                continue

            required_keys = ("thresh", "aot_ref") + tuple(model_key for _, model_key in MODEL_VARIANTS)
            if not all(k in data for k in required_keys):
                print(f"[WARN] Missing AOT keys in {path}; skipping")
                continue

            aot_metrics = {}
            for variant_name, model_key in MODEL_VARIANTS:
                aot_metrics[variant_name] = compute_aot_metrics(
                    data,
                    model_key=model_key,
                    tail_thresholds=tail_thresholds,
                )
            runs_aot.append((path, aot_metrics))

        if not runs_aot:
            continue

        meta = _cfg_to_meta(cfg)

        def agg_recursive(dicts):
            if isinstance(dicts[0], dict):
                return {k: agg_recursive([d[k] for d in dicts]) for k in dicts[0].keys()}
            try:
                mean_val = float(np.mean(dicts))
                std_val = float(np.std(dicts))
                return {"mean": mean_val, "std": std_val}
            except Exception:
                return dicts[0]

        aot_metrics = agg_recursive([r[1] for r in runs_aot])

        summary.append(
            {
                "config": meta,
                "paths": paths,
                "aot_metrics": aot_metrics,
            }
        )

        corrected = aot_metrics["corrected"]["avg12"]
        coarse = aot_metrics["coarse"]["avg12"]
        print(
            "[AOT] T={T}, sig={sig}, strat={strategy}, n_ens={n_ens} (N={N} runs): "
            "CR W1(avg12)={cw1:.6g}±{cw1s:.6g}, ML W1(avg12)={mw1:.6g}±{mw1s:.6g}, "
            "CR E_freq(avg12)={cfe:.6g}±{cfes:.6g}, ML E_freq(avg12)={mfe:.6g}±{mfes:.6g}".format(
                T=meta["T"],
                sig=meta["sig"],
                strategy=meta["strategy"],
                n_ens=meta["n_ens"],
                N=len(runs_aot),
                cw1=coarse["AOT_W1_mean"]["mean"], cw1s=coarse["AOT_W1_mean"]["std"],
                mw1=corrected["AOT_W1_mean"]["mean"], mw1s=corrected["AOT_W1_mean"]["std"],
                cfe=coarse["freq_error_mean"]["mean"], cfes=coarse["freq_error_mean"]["std"],
                mfe=corrected["freq_error_mean"]["mean"], mfes=corrected["freq_error_mean"]["std"],
            )
        )

    out_json = os.path.join(out_dir, "aot_metrics_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved AOT metrics summary: {out_json}")
    return out_json


def main():
    print("=== Running AOT exceedance metrics for all configurations ===")
    all_configs = _build_all_configs()

    json_path = run_all_aot_metrics_for_configs(
        all_configs,
        results_dir=RESULTS_DIR,
        out_dir=OUT_DIR,
        tail_thresholds=APPENDIX_THRESHOLDS,
    )

    print("=== Building AOT LaTeX tables ===")
    outputs = []
    for threshold_name, thresholds in (("main", MAIN_THRESHOLDS), ("appendix", APPENDIX_THRESHOLDS)):
        for variant_name, _ in MODEL_VARIANTS:
            out_prefix = os.path.join(OUT_DIR, f"master_aot_{threshold_name}_{variant_name}")
            outputs.extend(
                build_aot_all_tables(
                    json_file=json_path,
                    out_prefix=out_prefix,
                    strategies=STRATEGIES,
                    T_list=TABLE_NTIME_UNITS,
                    sig_list=SIG_LIST,
                    thresholds=thresholds,
                    layers=("psi1", "psi2", "avg12"),
                    variant=variant_name,
                )
            )

    print("[INFO] Generated AOT tables:")
    for p in outputs:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
