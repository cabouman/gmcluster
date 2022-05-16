#!/bin/bash
# This script just installs PyGMCluster along with requirements of PyGMCluster.

cd ..
conda run -n pygmcluster pip install -r requirements.txt
conda run -n pygmcluster pip install .
cd dev_scripts

