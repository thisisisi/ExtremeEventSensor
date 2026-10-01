#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Isabella Thiel <thieli@mit.edu>
"""Run a residual corrector using corrector/config.py and optional LOWDIM_CONFIG_JSON overrides."""

from __future__ import annotations

import json
import os
from dataclasses import fields, replace
from pathlib import Path

from corrector.config import CorrectorConfig


def load_config() -> CorrectorConfig:
    cfg = CorrectorConfig()
    override_path = os.environ.get("LOWDIM_CONFIG_JSON")
    if not override_path:
        return cfg

    path = Path(override_path)
    if not path.exists():
        raise FileNotFoundError(f"LOWDIM_CONFIG_JSON does not exist: {path}")

    allowed = {field.name for field in fields(CorrectorConfig)}
    overrides = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"Unknown config fields in {path}: {unknown}")

    values = {field.name: getattr(cfg, field.name) for field in fields(CorrectorConfig)}
    values.update(overrides)
    return CorrectorConfig(**values)


def run_from_config(cfg: CorrectorConfig | None = None) -> None:
    from corrector.data import save_run_metadata
    from corrector.training import train_and_evaluate

    if cfg is None:
        cfg = load_config()
    out_dir = cfg.resolve_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(cfg, out_dir)

    if cfg.model_kind not in {"deterministic", "deterministic_residual"}:
        raise ValueError(
            "The clean low-dimensional runner supports only deterministic_residual. "
            "Set n_ensemble_members_used = 0 for the baseline or > 0 for FiLM."
        )
    train_and_evaluate(cfg, out_dir)



def main():
    import argparse
    import subprocess
    import sys

    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="LowDim training, correction, summaries, and sweeps.")
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="Train, correct the test trajectory, and save metrics/plots.")
    run.add_argument("--backend", choices=("local", "slurm"), default="local")
    run.add_argument("--dry-run", action="store_true", help="Validate configuration without training or submitting.")
    run.add_argument("--config", type=Path, help="JSON overrides; dataset paths are relative to LowDim.")
    run.add_argument("--dataset", type=Path, help="NPZ archive, relative to your shell directory; overrides JSON.")
    run.add_argument("--output-dir", type=Path, help="Results root, relative to your shell directory; overrides JSON.")
    sub.add_parser("summary", help="Summarize completed runs; use summary --help for filters.", add_help=False)
    sub.add_parser("sweep", help="Preview/run repeated experiments; use sweep --help for grid options.", add_help=False)
    args, remaining = parser.parse_known_args()
    if args.action == "run":
        if remaining:
            parser.error("unrecognized arguments: " + " ".join(remaining))
        config = args.config or os.environ.get("LOWDIM_CONFIG_JSON")
        if config:
            os.environ["LOWDIM_CONFIG_JSON"] = str(Path(config).resolve())
        overrides = {}
        if args.dataset is not None:
            overrides["dataset_npz"] = str(args.dataset.resolve())
        if args.output_dir is not None:
            overrides["results_dir"] = str(args.output_dir.resolve())
        os.chdir(root)
        cfg = replace(load_config(), **overrides)
        if args.dry_run:
            print(f"{args.backend}: train -> correct -> evaluate | dataset={cfg.dataset_npz}")
            print(f"results root={cfg.output_root()}")
        elif args.backend == "slurm":
            from dataclasses import asdict
            import time
            from sweep import submit_job

            snapshot = root / "runs" / f"config_{time.time_ns()}.json"
            snapshot.parent.mkdir(exist_ok=True)
            snapshot.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
            (root / "output").mkdir(exist_ok=True)
            os.environ.setdefault("PYTHON_BIN", sys.executable)
            submit_job(root, root / "cluster/submit_lowdim_job.sh", snapshot, False, "")
        else:
            run_from_config(cfg)
    else:
        script = "summarize.py" if args.action == "summary" else "sweep.py"
        subprocess.run([sys.executable, str(root / script), *remaining], cwd=root, check=True)


if __name__ == "__main__":
    main()
