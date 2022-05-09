#!/bin/bash
# This script destroys the conda environment named "pycluster".
# It then creates an "pycluster" environment and installs pycluster.

# Destroy conda environement named "pycluster" and reinstall it
source install_conda_environment.sh

# Install pycluster
source install_pycluster.sh


