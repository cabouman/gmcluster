#!/bin/bash
# This script just installs gmcluster along with all requirements
# for the package, demos, and documentation.
# However, it does not remove the existing installation of gmcluster.

conda activate gmcluster
cd ..
pip install -r requirements.txt
pip install -e .
pip install -r docs/requirements.txt 
cd dev_scripts

