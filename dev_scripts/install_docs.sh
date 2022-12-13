#!/bin/bash
# This script installs the documentation.
<<<<<<< HEAD
# You can view documentation pages from hyper_neutron_tomography/docs/build/index.html .
=======
# You can view documentation pages from gmcluster/docs/build/index.html .
>>>>>>> ce4eadfdc21a041b7a4078051a20d33487ab94f6

# Build documentation
cd ../docs
pip install -r requirements.txt
make clean html
cd ../dev_scripts

