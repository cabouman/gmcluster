#!/bin/bash
# Remove build artifacts and uninstall the package.  Safe to run any time.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/config.sh"

rm -rf "$REPO_ROOT/docs/build" "$REPO_ROOT/dist" \
       "$REPO_ROOT/$NAME.egg-info" "$REPO_ROOT/build"

pip uninstall -y "$NAME" 2>/dev/null || true

# Package-specific cleanup, if config.sh defines it.
declare -f extra_clean >/dev/null && extra_clean
