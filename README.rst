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

The package provides one class, ``GaussianMixture``.  Fit it to your data, read the
estimated parameters, then classify points or draw new samples.

.. code-block:: python

    import numpy as np
    from gmcluster import GaussianMixture

    X = np.random.default_rng(0).standard_normal((500, 2))

    # Fit the mixture; "auto" selects the number of clusters by MDL.
    gm = GaussianMixture(num_clusters="auto").fit(X)

    print(gm.estimated_num_clusters)   # number of clusters found
    print(gm.estimated_weights)        # shape (K,)
    print(gm.estimated_means)          # shape (K, M)
    print(gm.estimated_covariances)    # shape (K, M, M)

    labels = gm.classify(X)            # most-likely cluster per point, shape (N,)
    new_points = gm.sample(100)        # draw 100 samples from the fitted mixture

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
