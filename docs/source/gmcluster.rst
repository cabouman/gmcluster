gmcluster
-----------
.. currentmodule:: gmcluster

.. autoclass:: GMModel
   :no-members:

Estimate from data
~~~~~~~~~~~~~~~~~~~
Build a model by estimating its parameters from sample data.

.. automethod:: GMModel.estimate

Read and change the parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The parameters are read as read-only attributes of a model, and changed with
``set_parameters``.

====================  ===========  ===============================
Attribute             Shape        Meaning
====================  ===========  ===============================
model.weights         (K,)         component weights, summing to 1
model.means           (K, M)       component means
model.covariances     (K, M, M)    component covariances
model.num_components   int         number of components, K
model.num_features     int         number of features, M
====================  ===========  ===============================

.. automethod:: GMModel.set_parameters

Use the model
~~~~~~~~~~~~~~
Call these on a model.  X has shape (N, M).

.. automethod:: GMModel.sample
.. automethod:: GMModel.classify
.. automethod:: GMModel.posterior
.. automethod:: GMModel.log_density
.. automethod:: GMModel.split

EstimationInfo
~~~~~~~~~~~~~~
.. autoclass:: EstimationInfo
   :members:
