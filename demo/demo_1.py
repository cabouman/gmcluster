import gmcluster
import matplotlib.pyplot as plt
"""
This file demonstrates a demo of the EM algorithm to estimate the order and parameters of a Gaussian Mixture model 
using "gmcluster" library.
"""


class MixtureObj:
    """Class to store the parameters for mixture object."""

    def __init__(self):
        """Function to initialize the parameters of the class MixtureObj."""
        self.K = None
        self.M = None
        self.cluster = None


class ClusterObj:
    """Class to store the parameters for cluster object."""

    def __init__(self):
        """Function to initialize the parameters of the class ClusterObj."""
        self.pb = None
        self.mu = None
        self.R = None


# Data parameters
N = 500  # number of observations/sample
pb = [0.4, 0.4, 0.2]  # cluster probabilities
R1 = [[1, 0.1], [0.1, 1]]
R2 = [[1, -0.1], [-0.1, 1]]
R3 = [[1, 0.2], [0.2, 0.5]]
R = [R1, R2, R3]  # cluster covariance matrices
mu1 = [[2], [2]]
mu2 = [[-2], [-2]]
mu3 = [[5.5], [2]]
mu = [mu1, mu2, mu3]  # cluster means

# Create the mixture with GM parameters
cluster = [None]*3
for i in range(3):
    cluster_obj = ClusterObj()
    cluster_obj.pb = pb[i]
    cluster_obj.R = R[i]
    cluster_obj.mu = mu[i]
    cluster[i] = cluster_obj

mixture = MixtureObj()
mixture.K = 3
mixture.M = 2
mixture.cluster = cluster

# Generate demo data
pixels = gmcluster.generate_gm_samples(mixture, N)

# Plot the generated samples
plt.plot(pixels[:, 0], pixels[:, 1], 'o')
plt.title('Scatter Plot of Multimodal Data')
plt.xlabel('first component')
plt.ylabel('second component')
plt.show()

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
