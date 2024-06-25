#!/bin/bash
# This script destroys the conda environment named "gmcluster" and reinstalls it.

# Create and activate new conda environment
cd ..
conda deactivate
conda deactivate
conda remove env --name gmcluster --all
conda create --name gmcluster python=3.9
conda activate gmcluster
cd dev_scripts

