import gmcluster

"""
This file demonstrates a demo of the EM algorithm to estimate the order and parameters of a Gaussian Mixture model 
using "gmcluster" library.
"""

# Generate demo data
pixels = gmcluster.gen_demo_dataset_1()

# Estimate optimal order and clustering data
omtr = gmcluster.estimate_gm_params(pixels)

print('\noptimal order: ', omtr.K)
for i in range(omtr.K):
    cluster_obj = omtr.cluster[i]
    print('\nCluster: ', i)
    print('pi: ', cluster_obj.pb)
    print('mean: \n', cluster_obj.mu)
    print('covar: \n', cluster_obj.R, '\n')


# Estimate clustering data assuming optimal order of 5
optimal_order = 5

omtr = gmcluster.estimate_gm_params(pixels, final_K=optimal_order)

for i in range(omtr.K):
    cluster_obj = omtr.cluster[i]
    print('\nCluster: ', i)
    print('pi: ', cluster_obj.pb)
    print('mean: \n', cluster_obj.mu)
    print('covar: \n', cluster_obj.R, '\n')
