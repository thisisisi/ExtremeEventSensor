#!/usr/bin/env python3
# Author: Isabella Thiel <thieli@mit.edu>
"""Run QG or LowDim workflows using the active Python environment."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run_script(script, arguments):
    """Keep each workflow's original process and path behavior."""
    return subprocess.call([sys.executable, str(ROOT / script), *arguments])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    workflows = parser.add_subparsers(dest="workflow", required=True)
    for name in ("qg", "lowdim"):
        child = workflows.add_parser(name, add_help=False, help=f"Run {name}; use --help for options.")
        child.add_argument("arguments", nargs=argparse.REMAINDER)
    args, remaining = parser.parse_known_args(argv)
    project = {"qg": "QG", "lowdim": "LowDim"}[args.workflow]
    return run_script(f"{project}/main.py", remaining + args.arguments)


if __name__ == "__main__":
    sys.exit(main())
