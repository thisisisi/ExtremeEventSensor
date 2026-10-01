#!/usr/bin/env python3
# Author: Isabella Thiel <thieli@mit.edu>
"""Generate every fixed-case comparison table from completed ML test results."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def run(script_name):
    script = SCRIPT_DIR / script_name
    print(f"\n=== Running {script.name} ===", flush=True)
    subprocess.run([sys.executable, "-u", str(script)], cwd=SCRIPT_DIR, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, help='Saved results directory.')
    parser.add_argument('--output-dir', type=Path, help='Destination for all comparison outputs.')
    parser.add_argument(
        "--skip-fixed-report",
        action="store_true",
        help="Skip the strict repeat-1 CSV/Markdown report.",
    )
    args = parser.parse_args()
    for value, variable in ((args.results_dir, 'QG_RESULTS_DIR'), (args.output_dir, 'QG_PLOTS_DIR')):
        if value is not None:
            os.environ[variable] = str(value.resolve())

    # Global PDF metrics, including deterministic T=50/T=1000 and avg12.
    run("compare_all.py")

    # Scalar threshold-resolved epsilon_AOT/epsilon_freq tables, including
    # deterministic T=50/T=1000 and the arithmetic average over both layers.
    run("compare_all_aot.py")

    # Strict repeat-1 human-readable report. This intentionally fails if one
    # of the requested fixed-case result files is absent.
    if not args.skip_fixed_report:
        run("fixed_case_control_report.py")

    print(
        "\nAll comparison outputs are in: "
        f"{os.environ.get('QG_PLOTS_DIR', PROJECT_ROOT / 'ML_Plots')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
