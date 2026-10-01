#!/bin/bash
# This script purges the docs and rebuilds them

cd ../docs
/bin/rm -r build

conda activate gmcluster
make clean html

echo ""
echo "*** The html documentation is at gmcluster/docs/build/html/index.html ***"
echo ""

cd ../dev_scripts
