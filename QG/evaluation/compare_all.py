# Author: Isabella Thiel <thieli@mit.edu>
from settings import CONFIG, EXPERIMENT_IDS, DISPLAY_NAMES
from metrics import run_all_metrics_for_configs
from json_2_latex_structured import build_master_table, build_master_table_relative
import os

# === CONFIGURATION ===
SIG_LIST = CONFIG.sigmas
STRATEGIES = EXPERIMENT_IDS
N_ENS_LIST = CONFIG.ensemble_sizes
NTIME_UNITS = CONFIG.training_durations
DET_NTIME_UNITS = CONFIG.unconditioned_durations
TABLE_NTIME_UNITS = sorted(set(NTIME_UNITS + DET_NTIME_UNITS))

model_type = CONFIG.model_type
basis_type = CONFIG.basis_type
n_epochs = CONFIG.epochs
dt_ml = CONFIG.dt_ml
results_dir = os.environ.get("QG_RESULTS_DIR", "../ML_Results")
out_dir = os.environ.get("QG_PLOTS_DIR", "../ML_Plots")

# === BUILD ALL CONFIGS ===
all_configs = []
for T in NTIME_UNITS:
    for sig in SIG_LIST:
        for strat in STRATEGIES:
            for n_ens in N_ENS_LIST:
                n_run = n_ens + 1 if strat == "S1" else n_ens
                cfg = (sig, n_run, T, strat, model_type, n_epochs, dt_ml, basis_type)
                all_configs.append(cfg)
for T in DET_NTIME_UNITS:
    det_cfg = (0.0, 1, T, None, model_type, n_epochs, dt_ml, basis_type)
    all_configs.insert(0, det_cfg)

def main():
    """Compute configured PDF metrics and write comparison tables."""
    print("\n=== Running all metrics for all configurations ===")
    os.makedirs(out_dir, exist_ok=True)
    run_all_metrics_for_configs(
        all_configs,
        results_dir=results_dir,
        out_dir=out_dir,
        target="ml",
        plot_per_config=False,
        make_overlays=False
    )
    print("\n=== Metrics computation finished ===")

    # === STEP 2: GENERATE LATEX TABLES ===
    json_path = os.path.join(out_dir, "metrics_summary.json")
    print(f"\n=== Generating LaTeX tables from {json_path} ===")

    # Configuration for table formatting
    USE_BOLD = False          # Bold for best values
    USE_COLOR = False         # Blue color for global best values
    CAPTION_BOTTOM = True    # Caption at bottom of table

    build_master_table(json_path, out_prefix=os.path.join(out_dir, "master"),
                       strategies=STRATEGIES, T_list=TABLE_NTIME_UNITS, sig_list=SIG_LIST,
                       use_bold=USE_BOLD, use_color=USE_COLOR, caption_bottom=CAPTION_BOTTOM, strategy_display_names=DISPLAY_NAMES)
    build_master_table_relative(json_path, out_prefix=os.path.join(out_dir, "master_relative"),
                                strategies=STRATEGIES, T_list=TABLE_NTIME_UNITS, sig_list=SIG_LIST,
                                use_bold=USE_BOLD, use_color=USE_COLOR, caption_bottom=CAPTION_BOTTOM, strategy_display_names=DISPLAY_NAMES)
    print(f"\n=== Done. Tables saved to {out_dir}/master_*.tex ===")


if __name__ == "__main__":
    main()
