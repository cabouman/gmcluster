============
Installation
============

Install the latest release from PyPI::

    pip install gmcluster

That is all most users need.

Install from source
--------------------

For development, or to use the current development branch, install from the
`GitHub repository <https://github.com/cabouman/gmcluster>`_.

1. Clone the repository::

       git clone https://github.com/cabouman/gmcluster.git
       cd gmcluster

2. (Optional) create and activate a conda environment::

       conda create --name gmcluster python=3.11
       conda activate gmcluster

3. Install the package.  Use ``-e`` for an editable install that reflects
   source edits::

       pip install -e .

Validate the installation by running a demo::

    cd demo
    python demo_1.py
