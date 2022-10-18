import pygmcluster

"""
This file demonstrates a demo of the EM algorithm to estimate the order and parameters of a Gaussian Mixture model 
using "PyGMCluster" library.
"""

# Generate demo data
pixels = pygmcluster.gen_demo_dataset_1()

# Estimate optimal order and clustering data
omtr = pygmcluster.estimate_gaussian_mixture(pixels)

print('\noptimal order: ', omtr.K)
for i in range(omtr.K):
    cluster_obj = omtr.cluster[i]
    print('\nCluster: ', i)
    print('pi: ', cluster_obj.pb)
    print('mean: \n', cluster_obj.mu)
    print('covar: \n', cluster_obj.R, '\n')


# Estimate clustering data assuming optimal order of 5
optimal_order = 5

omtr = pygmcluster.estimate_gaussian_mixture(pixels, final_K=optimal_order)

for i in range(omtr.K):
    cluster_obj = omtr.cluster[i]
    print('\nCluster: ', i)
    print('pi: ', cluster_obj.pb)
    print('mean: \n', cluster_obj.mu)
    print('covar: \n', cluster_obj.R, '\n')
