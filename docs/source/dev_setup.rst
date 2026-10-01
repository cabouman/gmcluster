====================================
Setting up a development environment
====================================

GMCluster's developer scripts live in ``dev_scripts/``.  They create an
isolated conda environment, install the package into it, and build the
documentation.  The scripts find their own location, so you can run them from
any directory.

Full install
============

Run::

    dev_scripts/clean_install_all.sh

This removes old build files, deletes and recreates the ``gmcluster`` conda
environment with Python 3.11, installs the package in editable mode with the
``test`` and ``docs`` extras, and builds the documentation.  When it finishes,
activate the environment::

    conda activate gmcluster

Individual steps
================

Each script can be run on its own, which is useful while developing:

- ``dev_scripts/remove_package.sh`` -- remove build files and uninstall the package.
- ``dev_scripts/install_empty_conda_environment.sh`` -- delete and recreate the empty environment.
- ``dev_scripts/install_package.sh`` -- install the package and its extras into the environment.
- ``dev_scripts/build_docs.sh`` -- build the HTML documentation (at ``docs/build/html/index.html``).
- ``dev_scripts/run_tests.sh`` -- run the test suite.

The install is editable, so edits to the source take effect without
reinstalling.

Per-package settings
====================

``dev_scripts/config.sh`` holds the settings specific to this package: the
environment name, the Python version, and the pip extras.  It is the only
developer script that differs between packages.
