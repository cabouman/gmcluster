#!/bin/bash
# This script purges the docs and environment

cd ..
/bin/rm -r docs/build
/bin/rm -r dist
/bin/rm -r gmcluster.egg-info
/bin/rm -r build

pip uninstall gmcluster

cd dev_scripts
