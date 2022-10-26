#!/bin/bash
# This script destroys the conda environment named "gmcluster".
# It then creates an "gmcluster" environment and installs gmcluster in it.

# Destroy conda environement named "gmcluster" and reinstall it
source install_conda_environment.sh

# Install gmcluster
source install_gmcluster.sh


