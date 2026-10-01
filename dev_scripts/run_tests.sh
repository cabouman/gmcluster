#!/bin/bash
# Run the test suite in the environment.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate "$NAME"
# Use `python -m pytest` so the active environment's pytest is used, not one
# earlier on PATH.
python -m pytest -ra "$REPO_ROOT/tests"
