#!/bin/bash
# This script installs the documentation.
# You can view documentation pages from hyper_neutron_tomography/docs/build/index.html .

# Build documentation
cd ../docs
pip install -r requirements.txt
make clean html
cd ../dev_scripts

