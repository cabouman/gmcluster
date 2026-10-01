#!/bin/bash
# Delete and recreate the package's conda environment (empty).
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"

# Leave any active env so the target env can be removed (conda will not remove
# the environment it is currently in).
conda activate base

# Package-specific setup before the env is created, if config.sh defines it.
declare -f before_env_create >/dev/null && before_env_create

# Remove the env, and any leftover directory a failed run may have left behind.
conda env remove -y -n "$NAME" 2>/dev/null || true
rm -rf "$(conda info --base)/envs/$NAME"

conda create -y -n "$NAME" python="$PYTHON_VERSION"
conda activate "$NAME"
