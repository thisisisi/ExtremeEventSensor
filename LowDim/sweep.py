#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run or submit the shared LowDim ensemble/context/repeat grid."""

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
import time
from dataclasses import fields
from pathlib import Path

from corrector.config import CorrectorConfig


DEFAULT_ENSEMBLE_SIZES = (0, 1, 2, 4, 8, 16)
DEFAULT_EIGENCOV_VALUES = (False, True)


def parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def parse_bool_list(text: str) -> tuple[bool, ...]:
    values = []
    for part in text.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in {"true", "1", "yes", "y"}:
            values.append(True)
        elif token in {"false", "0", "no", "n"}:
            values.append(False)
        else:
            raise ValueError(f"Could not parse boolean token: {part}")
    return tuple(values)


def condition_label(n_ens: int, eigencov: bool, n_eigcov_modes: int) -> str:
    if n_ens == 0:
        return "nofilm"
    return f"vec{n_eigcov_modes}" if eigencov else "meanstd"


def config_payload(base_cfg: CorrectorConfig, n_ens: int, eigencov: bool, repeat: int) -> dict:
    allowed = {field.name for field in fields(CorrectorConfig)}
    payload = {
        "dataset_npz": base_cfg.dataset_npz,
        "run_id_train": base_cfg.run_id_train,
        "run_id_test": base_cfg.run_id_test,
        "n_ensemble_members_used": int(n_ens),
        "eigencov": bool(eigencov),
        "n_eigcov_modes": int(base_cfg.n_eigcov_modes),
        "repeat_index": int(repeat),
        "auto_increment_repeats": False,
    }

    for key in [
        "ml_stride",
        "model_kind",
        "train_time",
        "test_time",
        "train_sequence_length",
        "test_sequence_length",
        "state_dim",
        "model_architecture",
        "hidden_dim",
        "film_hidden_dim",
        "attention_heads",
        "attention_layers",
        "ffn_multiplier",
        "dropout_rate",
        "epochs",
        "batch_size",
        "learning_rate",
        "tail_loss_weight",
        "tail_threshold_std",
        "tail_loss_interval",
        "tail_weight_factor",
        "lambda_delta",
        "lambda_moment",
        "lambda_tail",
        "state_loss_weights",
        "include_coarse_training_inputs",
        "gate_init_bias",
        "tail_exceedance_temperature",
        "context_dropout_prob",
        "state_quantile_calibration",
        "quantile_grid_size",
        "inference_window_overlap",
        "coarse_context_window",
    ]:
        if key in allowed:
            payload[key] = getattr(base_cfg, key)

    return payload


def write_config(config_dir: Path, payload: dict, n_ens: int, eigencov: bool, repeat: int) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    label = condition_label(n_ens, eigencov, int(payload["n_eigcov_modes"]))
    path = config_dir / f"ens{n_ens}_{label}_rep{repeat:03d}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def result_exists(cfg: CorrectorConfig, payload: dict) -> bool:
    candidate = CorrectorConfig(**payload).resolve_output_dir()
    return (candidate / "corrected_test_outputs.npz").exists() or (candidate / "summary.json").exists()


def submit_job(
    project_dir: Path,
    sbatch_script: Path,
    config_path: Path,
    dry_run: bool,
    extra_export: str,
) -> None:
    export_items = [
        "ALL",
        f"PROJECT_DIR={project_dir}",
        f"LOWDIM_CONFIG_JSON={config_path}",
    ]
    if extra_export:
        export_items.append(extra_export)

    cmd = [
        "sbatch",
        f"--export={','.join(export_items)}",
        str(sbatch_script),
    ]
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated LowDim experiments locally or on Slurm.")
    parser.add_argument("--config", type=Path, help="JSON overrides for the base configuration.")
    parser.add_argument("--project_dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--sbatch_script", type=Path, default=Path(__file__).resolve().parent / "cluster" / "submit_lowdim_job.sh")
    parser.add_argument("--config_dir", type=Path, default=Path("sweep_configs/lowdim"))
    parser.add_argument("--ensemble-sizes", "--ensemble_sizes", type=parse_int_list, default=DEFAULT_ENSEMBLE_SIZES)
    parser.add_argument("--eigencov-values", "--eigencov_values", type=parse_bool_list, default=DEFAULT_EIGENCOV_VALUES)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--repeat-offset", "--repeat_offset", type=int, default=0)
    parser.add_argument("--interval", "--submit_interval_seconds", dest="submit_interval_seconds", type=float, default=3.0)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--resubmit_existing", action="store_true", default=False)
    parser.add_argument("--backend", choices=("local", "slurm"), default="local")
    parser.add_argument("--dry-run", "--dry_run", action="store_true", default=False)
    parser.add_argument("--extra_export", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    config_dir = (project_dir / args.config_dir).resolve() if not args.config_dir.is_absolute() else args.config_dir
    sbatch_script = args.sbatch_script.resolve()
    if args.repeats < 1 or args.submit_interval_seconds < 0:
        raise ValueError("repeats must be positive and interval nonnegative")
    base_cfg = CorrectorConfig()
    if args.config:
        from main import load_config
        os.environ["LOWDIM_CONFIG_JSON"] = str(args.config.resolve())
        base_cfg = load_config()
    config_dir = config_dir / str(time.time_ns())

    jobs_submitted = 0
    jobs_skipped = 0
    seen_jobs: set[tuple[int, str, int]] = set()

    for n_ens in args.ensemble_sizes:
        for eigencov in args.eigencov_values:
            for rep_local in range(args.repeats):
                repeat = args.repeat_offset + rep_local
                label = condition_label(n_ens, eigencov, base_cfg.n_eigcov_modes)
                job_key = (n_ens, label, repeat)
                if job_key in seen_jobs:
                    print(f"Skipping duplicate no-op configuration: ens{n_ens}_{label}_rep{repeat:03d}")
                    jobs_skipped += 1
                    continue
                seen_jobs.add(job_key)

                payload = config_payload(base_cfg, n_ens, eigencov, repeat)
                config_path = config_dir / f"ens{n_ens}_{label}_rep{repeat:03d}.json"
                if not args.dry_run:
                    config_path = write_config(config_dir, payload, n_ens, eigencov, repeat)

                if args.skip_existing and not args.resubmit_existing and result_exists(base_cfg, payload):
                    print(f"Skipping existing result for {config_path.name}")
                    jobs_skipped += 1
                    continue

                if args.backend == "local":
                    command = [sys.executable, str(project_dir / "main.py"), "run", "--config", str(config_path)]
                    print(" ".join(command))
                    if not args.dry_run:
                        subprocess.run(command, cwd=project_dir, check=True)
                else:
                    os.environ.setdefault("PYTHON_BIN", sys.executable)
                    if not args.dry_run:
                        (project_dir / "output").mkdir(exist_ok=True)
                    submit_job(
                        project_dir=project_dir,
                        sbatch_script=sbatch_script,
                        config_path=config_path,
                        dry_run=args.dry_run,
                        extra_export=args.extra_export,
                    )
                jobs_submitted += 1

                if args.submit_interval_seconds > 0 and not args.dry_run:
                    time.sleep(args.submit_interval_seconds)

    print("Sweep submission complete.")
    print(f"Configurations {'planned' if args.dry_run else 'launched'}: {jobs_submitted}")
    print(f"Jobs skipped: {jobs_skipped}")


if __name__ == "__main__":
    main()
