#!/bin/bash
# This script just installs PyGMCluster along with requirements of PyGMCluster.

cd ..
pip install -r requirements.txt
pip install .
cd dev_scripts

