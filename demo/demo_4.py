import numpy as np
import matplotlib.pyplot as plt
import gmcluster

"""
This file demonstrates a demo of the EM algorithm to estimate the order, and parameters of a Gaussian Mixture model 
and perform unsupervised cluster classification with and without decorrelated coordinates using "gmcluster" library.
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

# Estimate optimal order and clustering data using original coordinates
omtr = gmcluster.estimate_gm_params(pixels)

print('\noptimal order: ', omtr.K)
for i in range(omtr.K):
    cluster_obj = omtr.cluster[i]
    print('\nCluster: ', i)
    print('pi: ', cluster_obj.pb)
    print('mean: \n', cluster_obj.mu)
    print('covar: \n', cluster_obj.R, '\n')

# Split classes
mtrs = gmcluster.split_classes(omtr)

# Perform classification
likelihood = np.zeros((np.shape(pixels)[0], len(mtrs)))
for k in range(len(mtrs)):
    likelihood[:, k] = gmcluster.compute_class_likelihood(mtrs[k], pixels)[:, 0]
class_list = np.argmax(likelihood, axis=1)
for n in range(np.shape(pixels)[0]):
    print(pixels[n, :], ' Log-likelihood: ', likelihood[n, :], ' class: ', class_list[n])

# Plot classification results using original coordinates
plt.plot(pixels[np.argwhere(class_list == 0), 0], pixels[np.argwhere(class_list == 0), 1], 'o', label='class 0')
plt.plot(pixels[np.argwhere(class_list == 1), 0], pixels[np.argwhere(class_list == 1), 1], 'x', label='class 1')
plt.plot(pixels[np.argwhere(class_list == 2), 0], pixels[np.argwhere(class_list == 2), 1], '*', label='class 2')
plt.title('Gaussian mixture classification using original coordinates')
plt.xlabel('first component')
plt.ylabel('second component')
plt.legend()
plt.show()

# Estimate optimal order and clustering data using decorrelated coordinates
omtr = gmcluster.estimate_gm_params(pixels, decorrelate_coordinates=True)

print('\noptimal order: ', omtr.K)
for i in range(omtr.K):
    cluster_obj = omtr.cluster[i]
    print('\nCluster: ', i)
    print('pi: ', cluster_obj.pb)
    print('mean: \n', cluster_obj.mu)
    print('covar: \n', cluster_obj.R, '\n')

# Split classes
mtrs = gmcluster.split_classes(omtr)

# Perform classification
likelihood = np.zeros((np.shape(pixels)[0], len(mtrs)))
for k in range(len(mtrs)):
    likelihood[:, k] = gmcluster.compute_class_likelihood(mtrs[k], pixels)[:, 0]
class_list = np.argmax(likelihood, axis=1)
for n in range(np.shape(pixels)[0]):
    print(pixels[n, :], ' Log-likelihood: ', likelihood[n, :], ' class: ', class_list[n])

# Plot classification results using decorrelated coordinates
plt.plot(pixels[np.argwhere(class_list == 0), 0], pixels[np.argwhere(class_list == 0), 1], 'o', label='class 0')
plt.plot(pixels[np.argwhere(class_list == 1), 0], pixels[np.argwhere(class_list == 1), 1], 'x', label='class 1')
plt.plot(pixels[np.argwhere(class_list == 2), 0], pixels[np.argwhere(class_list == 2), 1], '*', label='class 2')
plt.title('Gaussian mixture classification using decorrelated coordinates')
plt.xlabel('first component')
plt.ylabel('second component')
plt.legend()
plt.show()
