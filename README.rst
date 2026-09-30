GMCluster
=========

GMCluster fits a Gaussian mixture model to data by EM and selects the number of clusters automatically using the minimum description length (MDL) criterion.

*This is an EM-based clustering package for python that is based on the following C package:*
>https://engineering.purdue.edu/~bouman/software/cluster/

*The documentation for this package is available here:*
>https://gmcluster.readthedocs.io/

Installing
----------

1. *Clone or download the repository and get inside:*

.. code-block::

	git clone https://github.com/cabouman/gmcluster.git
	cd gmcluster

2. Install the conda environment and package

    a. Option 1: Clean install from dev_scripts

        *******You can skip all other steps if you do a clean install.******

        To do a clean install, use the command:

		.. code-block::

			cd dev_scripts
			source clean_install_all.sh
			cd ..

    b. Option 2: Manual install

        1. *Create conda environment:*

            Create a new conda environment named ``gmcluster`` using the following commands:

			.. code-block::
	
				conda create --name gmcluster python=3.9
				conda activate gmcluster

            Anytime you want to use this package, this ``gmcluster`` environment should be activated with the following:

			.. code-block::
	
				conda activate gmcluster

	2. *Install the dependencies:*

	   To install the packages, use the following command.
	                	
			.. code-block::
	
	                	pip install -r requirements.txt

        3. *Install gmcluster package:*

            Use the following command to install the package.

			.. code-block::
	
	                	pip install .

            To allow editing of the package source while using the package, use

			.. code-block::
	                	
				pip install -e .

	4. *Build the documentation:*
	
	   Use the following command to build the documentation.

			.. code-block::
			
				cd docs
				pip install -r requirements.txt
				make clean html
				cd ..

Running Demo(s)
---------------

You can validate the installation by running demo scripts.

.. code-block::

	cd demo
	python demo_1.py

Quick Start
-----------

The package provides one class, ``GaussianMixture``. Fit it to your data, read the
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

Citation
--------

Please cite this software when you use it. The BibTeX entry is in
``docs/source/credits.rst`` and in the online documentation at
https://gmcluster.readthedocs.io/ .
