============
Installation 
============

The ``gmcluster`` package currently is only available to download and install from source through GitHub.

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
				pip install -r requirements.txt

            Anytime you want to use this package, this ``gmcluster`` environment should be activated with the following:

			.. code-block::
	
				conda activate gmcluster

        2. *Install gmcluster package:*

            Use the following command to install the package.

			.. code-block::
	
	                	pip install .

            To allow editing of the package source while using the package, use

			.. code-block::
	                	
				pip install -e .
				

3. *Validate the installation by running a demo script:*

.. code-block::

	cd demo
	python demo_1.py

