#!/usr/bin/env bash
# Author: Isabella Thiel <thieli@mit.edu>
# Install the pinned requirements into the active Python environment.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 is required (see README)."'
python -m pip install -r "$SCRIPT_DIR/requirements.txt" "$@"
