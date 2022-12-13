#!/bin/bash
# This script destroys the conda environment named "gmcluster" and reinstalls it.
# It then installs gmcluster package.
# It also installs the documentation.

# Destroy conda environement named "gmcluster" and reinstall it
source install_conda_environment.sh

# Install gmcluster
source install_gmcluster.sh

# Install documentation
source install_docs.sh
<<<<<<< HEAD
=======

>>>>>>> ce4eadfdc21a041b7a4078051a20d33487ab94f6

