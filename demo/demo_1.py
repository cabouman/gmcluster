import pygmcluster

"""
This file demonstrates a demo of the EM algorithm to estimate the order and parameters of a Gaussian Mixture model 
using "PyGMCluster" library.
"""

# Generate demo data
pixels = pygmcluster.gen_demo_dataset_1()

# Estimate optimal order and clustering data
[mtrs, omtr] = pygmcluster.gaussian_mixture(pixels, 20, 0, True, 'full', 1e5)

print('\noptimal order: ', omtr.K)
for i in range(omtr.K):
    cluster_obj = omtr.cluster[i]
    print('\nCluster: ', i)
    print('pi: ', cluster_obj.pb)
    print('mean: \n', cluster_obj.mu)
    print('covar: \n', cluster_obj.R, '\n')


# Estimate clustering data assuming optimal order of 5
[mtrs, omtr] = pygmcluster.gaussian_mixture(pixels, 20, 5, True, 'full', 1e5)

for i in range(omtr.K):
    cluster_obj = omtr.cluster[i]
    print('\nCluster: ', i)
    print('pi: ', cluster_obj.pb)
    print('mean: \n', cluster_obj.mu)
    print('covar: \n', cluster_obj.R, '\n')
