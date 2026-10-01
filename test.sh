#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export MPLBACKEND=Agg
python -m pytest -q -p no:cacheprovider tests
