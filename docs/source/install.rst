============
Installation 
============

The ``PyGMCluster`` package currently is only available to download and install from source through GitHub.


Downloading and installing from source
-----------------------------------------

1. 1. Download the source code:

  In order to download the python code, move to a directory of your choice and run the following two commands.

	| ``git clone https://github.com/cabouman/pygmcluster.git``
	| ``cd pygmcluster``


2. Create a Virtual Environment:

  It is recommended that you install to a virtual environment.
  If you have Anaconda installed, you can run the following.

	| ``conda create --name pygmcluster python=3.8``
	| ``conda activate pygmcluster``

3. Install the dependencies:

  In order to install the dependencies, use the following command.

	``pip install -r requirements.txt``

4. Install the PyGMCluster package:

  In order to install the package, use the following command.

	``pip install .``

  Now you can use the package. The ``pygmcluster`` environment needs to be activated.


5. Validate installation:

  You can validate the installation by running a demo script.
  
	| ``cd demo``
	| ``python demo_1.py``

