#!/bin/bash
# This script destroys the conda environment named "pygmcluster" and reinstalls it.

# Create and activate new conda environment
cd ..
conda remove env --name pygmcluster --all
conda create --name pygmcluster python=3.8
cd dev_scripts

