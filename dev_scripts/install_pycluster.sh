#!/bin/bash
# This script just installs pycluster along with requirements of pycluster.

cd ..
pip install -r requirements.txt
pip install .
cd dev_scripts

