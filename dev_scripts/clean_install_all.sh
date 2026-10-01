#!/bin/bash
# Full clean install: remove, recreate the env, install, build docs.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

bash "$SCRIPT_DIR/remove_package.sh"
bash "$SCRIPT_DIR/install_empty_conda_environment.sh"
bash "$SCRIPT_DIR/install_package.sh"
bash "$SCRIPT_DIR/build_docs.sh"

red=$(tput setaf 1); reset=$(tput sgr0)
echo
echo "Use  ${red}conda activate $NAME${reset}  to activate the environment."
