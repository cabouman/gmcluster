# GMCluster 
GMCluster: An Unsupervised Algorithm for Modeling Gaussian Mixtures

*This is an EM-based clustering package for python that is based on the following C package:*
>https://engineering.purdue.edu/~bouman/software/cluster/

*The documentation for this package is available here:*
>https://gmcluster.readthedocs.io/

## Installation Instructions:
1. Download the source code:

  Move to a directory of your choice and run the following two commands.

```
git clone https://github.com/cabouman/gmcluster.git
cd gmcluster
```
	
  Alternatively, you can directly clone from GitHub and then enter the repository.

2. Installation:

  Follow any of the two methods.

* 2.1. Easy installation:

  If you have Anaconda installed, run the following commands.
	  
	```
	cd dev_scripts
	source ./install_all.sh
	cd ..
	```
		
* 2.2. Manual installation:

	- 2.2.1 Create a Virtual Environment:

	  It is recommended that you install the package to a virtual environment.
	  If you have Anaconda installed, you can run the following.

		```
		conda create --name gmcluster python=3.8
		conda activate gmcluster
		```

	- 2.2.2 Install the dependencies:

	  In order to install the dependencies, use the following command.

		```
		pip install -r requirements.txt
		```

	- 2.2.3 Install the gmcluster package:

	  Use the following command to install the package.

		```
		pip install .
		```

	- 2.2.4 Install the documentation:

	  Use the following command to install the documentation.

		```
		cd docs
		pip install -r requirements.txt
		make clean html
		cd ..
		```

  The installation is done. The ``gmcluster`` environment needs to be activated every time you use the package.

3. Validate installation:

  You can validate the installation by running a demo script.
  
```
cd demo
python demo_1.py
```






