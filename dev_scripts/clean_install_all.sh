#!/bin/bash
# This script installs everything from scratch

source remove_package.sh
source install_conda_environment.sh

source install_package.sh
source build_docs.sh

red=`tput setaf 1`
reset=`tput sgr0`

echo " "
echo "Use"
echo "${red}   conda activate gmcluster   ${reset}"
echo "to activate the conda environment."
echo " "
