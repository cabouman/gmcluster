from setuptools import setup, find_packages
import os

NAME = "gmcluster"
VERSION = "0.0.1"
DESCR = "Python code for EM algorithm-based clustering"
REQUIRES = ['numpy', 'matplotlib']
LICENSE = "BSD-3-Clause"

AUTHOR = 'GMCluster development team'
EMAIL = "bouman@purdue.edu"
PACKAGE_DIR = "gmcluster"

setup(install_requires=REQUIRES,
      zip_safe=False,
      name=NAME,
      version=VERSION,
      description=DESCR,
      author=AUTHOR,
      author_email=EMAIL,
      license=LICENSE,
      packages=find_packages(include=['gmcluster']),
      )

