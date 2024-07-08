#!/bin/bash
# This script destroys the conda environment for this pacakge and reinstalls it.

# Create and activate new conda environment
# First check if the target environment is active and deactivate if so

if [ "$CONDA_DEFAULT_ENV"==gmcluster ]; then
    conda deactivate
fi

conda remove env --name gmcluster --all
conda create --name gmcluster python=3.9
conda activate gmcluster

