#!/bin/bash
# This script just installs gmcluster along with requirements of gmcluster.

cd ..
pip install -r requirements.txt
pip install .
cd dev_scripts

