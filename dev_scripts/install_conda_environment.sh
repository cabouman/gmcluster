#!/bin/bash
# This script destroys the conda environment named "pycluster" and reinstalls it.

# Create and activate new conda environment
cd ..
conda deactivate
conda remove env --name pycluster --all
conda create --name pycluster python=3.8
conda activate pycluster
cd dev_scripts

