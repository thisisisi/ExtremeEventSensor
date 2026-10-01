#!/usr/bin/env bash
# Submit one run per ensemble/context combination using independent JSON configs.
# Usage: bash sweep_submit.sh [--dry-run] [interval_seconds]
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
args=()
if [[ "${1:-}" == "--dry-run" ]]; then
    args+=(--dry_run)
    shift
fi
interval="${1:-60}"
cd "$SCRIPT_DIR/.."
"$PYTHON_BIN" "$SCRIPT_DIR/../sweep.py" \
    --backend slurm --ensemble_sizes 0,1,2,4,8,16 --eigencov_values true,false \
    --repeats 1 --submit_interval_seconds "$interval" "${args[@]}"
