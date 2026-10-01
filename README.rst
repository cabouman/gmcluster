GMCluster
=========

GMCluster fits a Gaussian mixture model to data by EM and selects the number of clusters automatically using the minimum description length (MDL) criterion.

It is a Python rewrite of the C package `Cluster <https://engineering.purdue.edu/~bouman/software/cluster/>`_.  Full documentation is at https://gmcluster.readthedocs.io/ .

Installing
----------

Install the latest release from PyPI::

    pip install gmcluster

To install from source (for development), clone the repository and do an editable install::

    git clone https://github.com/cabouman/gmcluster.git
    cd gmcluster
    pip install -e .

Quick Start
-----------

The package provides one class, ``GMModel``.  Estimate a model from your data, read
its parameters, then classify points or draw new samples.

.. code-block:: python

    import numpy as np
    from gmcluster import GMModel

    X = np.random.default_rng(0).standard_normal((500, 2))

    # Estimate the mixture; "auto" selects the number of clusters by MDL.
    gm = GMModel.estimate(X, num_clusters="auto")

    print(gm.num_components)   # number of clusters found
    print(gm.weights)          # shape (K,)
    print(gm.means)            # shape (K, M)
    print(gm.covariances)      # shape (K, M, M)

    labels = gm.classify(X)    # most-likely cluster per point, shape (N,)
    new_points = gm.sample(100)  # draw 100 samples from the mixture

Running the demos
-----------------

Validate the installation by running a demo::

    cd demo
    python demo_1.py

Citation
--------

Please cite this software when you use it.  The BibTeX entry is in
``docs/source/credits.rst`` and in the online documentation at
https://gmcluster.readthedocs.io/ .
