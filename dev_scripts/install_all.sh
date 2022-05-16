#!/bin/bash
# This script destroys the conda environment named "pygmcluster".
# It then creates an "pygmcluster" environment and installs PyGMCluster in it.

# Destroy conda environement named "pygmcluster" and reinstall it
source install_conda_environment.sh

# Install PyGMCluster
source install_pygmcluster.sh


