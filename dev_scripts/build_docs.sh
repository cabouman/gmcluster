#!/bin/bash
# Build the HTML documentation.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate "$NAME"
rm -rf "$REPO_ROOT/docs/build"
make -C "$REPO_ROOT/docs" clean html
echo "*** HTML docs at $REPO_ROOT/docs/build/html/index.html ***"
