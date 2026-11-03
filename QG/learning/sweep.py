"""Shared QG sweep design and Slurm submission helpers."""
import os
import yaml
import copy
import subprocess
from pathlib import Path
from datetime import datetime
from sweep_artifacts import model_artifact_path

TRAIN_TEMPLATE = "Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization4_24x24_sig{}"
TEST_TEMPLATE  = "merged_Qi_Majda_2018_Gaussian_Bumps_03_beta2.0_rdrag0.1_realization2_24x24_tau16.0_sig0"

SIG_LIST       = [0.01]
STRATEGIES     = ["baseline", "pooled", "film", "random_context", "coarse_statistics", "ensemble_mean", "ensemble_variance", "covariance_concat", "mean_spread_concat"]
# Preregistered mechanism-control design: T_train=50, sigma=0.01, N_add=2.
N_ENS_LIST     = [2]
NTIME_UNITS    = [50]
# Deterministic comparators are trained at both the short- and long-data limits.
DET_NTIME_UNITS = [50, 1000]
RUN_DET = True
DEFAULT_REPEATS = 5 # number of repeats for each configuration (for determining variability in performance)
DEFAULT_REPEAT_OFFSET = 1 # starting index for repeat numbering (for continuing a sweep after a crash or interruption + setting the random seed schedule to match the original sweep)

BASE_SEED = 1

SWEEP_LOG_HEADER = (
    "submitted_at,slurm_job_id,mode,repeat_index,seed,sig,n_ens,Ntime,strategy,"
    "started_at,ended_at,elapsed_seconds,"
    "sacct_state,sacct_elapsed,sacct_req_mem,max_rss\n"
)


def load_base_config(path="ml_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def write_config(config, job_id, repeat_index):
    sweep_dir = Path("sweep_configs")
    sweep_dir.mkdir(exist_ok=True)
    p = sweep_dir / f"ml_config_{job_id}_rep{int(repeat_index):03d}.yaml"
    with open(p, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    return p

def init_sweep_log():
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"sweep_runtime_{ts}.csv"
    if not log_path.exists():
        log_path.write_text(SWEEP_LOG_HEADER)
    return log_path.resolve()


def submit_job(config_path, run_meta, log_csv_path, stage="train", dependency=None):
    env = os.environ.copy()
    env["CONFIG_SRC"] = str(config_path)
    env["SWEEP_LOG_CSV"] = str(log_csv_path)
    env["SWEEP_REPEAT_INDEX"] = str(run_meta.get("repeat_index", ""))
    env["SWEEP_SEED"] = str(run_meta.get("seed", ""))
    env["SWEEP_SIG"] = str(run_meta.get("sig", ""))
    env["SWEEP_N_ENS"] = str(run_meta.get("n_ens", ""))
    env["SWEEP_NTIME"] = str(run_meta.get("Ntime", ""))
    env["SWEEP_STRATEGY"] = str(run_meta.get("strategy", ""))
    env["SWEEP_MODE"] = str(run_meta.get("mode", ""))
    env["SWEEP_SUBMITTED_AT"] = str(run_meta.get("submitted_at", ""))
    command = ["sbatch", "--parsable"]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(Path(__file__).resolve().parents[1] / "cluster" / f"{stage}.sh"))
    r = subprocess.run(command, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"sbatch failed: {r.stderr}")
    return r.stdout.strip().split(";")[0]

def base_overrides(cfg, train_case, test_case, ntime_units):
    cfg["data"]["case_name_train"] = train_case
    cfg["data"]["case_name_test"]  = test_case
    cfg["data"]["Ntime_units"]     = ntime_units
    return cfg

def add_repeat_metadata(cfg, repeat_index):
    cfg.setdefault("training", {})
    cfg["training"]["repeat_index"] = int(repeat_index)
    # Sweep seeds are controlled by BASE_SEED, even if the base YAML has a seed.
    cfg["training"]["seed"] = int(BASE_SEED + repeat_index)
    return cfg

def generate_sweep_configs(base_config, run_det=RUN_DET, repeats=DEFAULT_REPEATS, repeat_offset=DEFAULT_REPEAT_OFFSET):
    configs = []

    for repeat in range(repeat_offset, repeat_offset + max(1, int(repeats))):
        # Deterministic baselines use the preregistered sigma=0.01 training case
        # and are repeated with the same seed schedule as the ensemble models.
        if run_det:
            for ntime_units in DET_NTIME_UNITS:
                cfg = copy.deepcopy(base_config)
                train_case = TRAIN_TEMPLATE.format(SIG_LIST[0])
                test_case  = TEST_TEMPLATE.format(SIG_LIST[0])
                cfg = base_overrides(cfg, train_case, test_case, ntime_units)
                cfg["ensemble"] = {"enabled": False}
                cfg = add_repeat_metadata(cfg, repeat)
                configs.append(cfg)

        # Preregistered mechanism controls: T=50, sigma=0.01, N_add=2.
        for ntime_units in NTIME_UNITS:
            for sig in SIG_LIST:
                train_case = TRAIN_TEMPLATE.format(sig)
                test_case  = TEST_TEMPLATE.format(sig)
                for strategy in STRATEGIES:
                    for n_ens in N_ENS_LIST:
                        cfg = copy.deepcopy(base_config)
                        cfg = base_overrides(cfg, train_case, test_case, ntime_units)
                        cfg["ensemble"] = {
                            "enabled": True,
                            "strategy": strategy,
                            "n_ens": n_ens,
                            # baseline defaults to every nonzero covariance mode.
                            "n_eigcov_modes": n_ens - 1,
                            # Test-time proxy is causal: current plus 31 past states.
                            "proxy_window": 32,
                            "sig": sig
                        }
                        if strategy == "pooled":
                            # n_ens is N_add for pooling: baseline + two additions.
                            cfg["ensemble"]["include_baseline"] = True
                        cfg = add_repeat_metadata(cfg, repeat)
                        configs.append(cfg)

    return configs

