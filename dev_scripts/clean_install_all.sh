#!/bin/bash
# This script installs everything from scratch

yes | source remove_package.sh
yes | source install_conda_environment.sh

yes | source install_package.sh
yes | source build_docs.sh

red=`tput setaf 1`
reset=`tput sgr0`

echo " "
echo "Use"
echo "${red}   conda activate gmcluster   ${reset}"
echo "to activate the conda environment."
echo " "
