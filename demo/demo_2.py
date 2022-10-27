import numpy as np
import matplotlib.pyplot as plt
import gmcluster

"""
This file demonstrates a demo of the EM algorithm to estimate the orders and parameters of 2 different Gaussian mixture 
models and perform binary maximum likelihood classification of test observations based on the estimated parameters for 
the 2 mixture models using "gmcluster" library.
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
N = 500  # number of observations/sample (same for class 0 and 1)
pb = [0.4, 0.4, 0.2]  # cluster probabilities (same for class 0 and 1)
R_0 = [[1, 0.1], [0.1, 1]]
R_1 = [[1, -0.1], [-0.1, 1]]
R_2 = [[1, 0.2], [0.2, 0.5]]
R = [R_0, R_1, R_2]  # cluster covariance matrices (same for class 0 and 1)
mu_00 = [[2], [2]]
mu_01 = [[-2], [-2]]
mu_02 = [[5.5], [2]]
mu_0 = [mu_00, mu_01, mu_02]  # cluster means for class 0
mu_10 = [[-2], [2]]
mu_11 = [[2], [-2]]
mu_12 = [[-5.5], [2]]
mu_1 = [mu_10, mu_11, mu_12]  # cluster means for class 1
mu_all = [mu_0, mu_1]  # cluster means for class 0 and class 1

# Create the 2 mixtures with the above GM parameters
mixtures = [None]*2
for i in range(2):
    mu = mu_all[i]

    cluster = [None]*3
    for j in range(3):
        cluster_obj = ClusterObj()
        cluster_obj.pb = pb[j]
        cluster_obj.R = R[j]
        cluster_obj.mu = mu[j]
        cluster[j] = cluster_obj

    mixture = MixtureObj()
    mixture.K = 3
    mixture.M = 2
    mixture.cluster = cluster
    mixtures[i] = mixture

# Generate demo data
train_data_0 = gmcluster.generate_gm_samples(mixtures[0], N)
train_data_1 = gmcluster.generate_gm_samples(mixtures[1], N)
test_data_0 = gmcluster.generate_gm_samples(mixtures[0], N//5)
test_data_1 = gmcluster.generate_gm_samples(mixtures[1], N//5)
test_data = np.concatenate((test_data_0, test_data_1), axis=0)

# Plot the generated training data for class 0 and class 1
plt.plot(train_data_0[:, 0], train_data_0[:, 1], 'o', label='class 0')
plt.plot(train_data_1[:, 0], train_data_1[:, 1], 'x', label='class 1')
plt.title('Scatter Plot of Multimodal Training Data for Class 0 and Class 1')
plt.xlabel('first component')
plt.ylabel('second component')
plt.legend()
plt.show()

# Plot the generated testing data
plt.plot(test_data[:, 0], test_data[:, 1], 'x')
plt.title('Scatter Plot of Multimodal Testing Data')
plt.xlabel('first component')
plt.ylabel('second component')
plt.show()

# Estimate optimal order and clustering data for class 0
class_0 = gmcluster.estimate_gm_params(train_data_0)

print('\noptimal order: ', class_0.K)
for i in range(class_0.K):
    cluster_obj = class_0.cluster[i]
    print('\nCluster: ', i)
    print('pi: ', cluster_obj.pb)
    print('mean: \n', cluster_obj.mu)
    print('covar: \n', cluster_obj.R, '\n')


# Estimate optimal order and clustering data for class 1
class_1 = gmcluster.estimate_gm_params(train_data_1)

print('\noptimal order: ', class_1.K)
for i in range(class_1.K):
    cluster_obj = class_1.cluster[i]
    print('\nCluster: ', i)
    print('pi: ', cluster_obj.pb)
    print('mean: \n', cluster_obj.mu)
    print('covar: \n', cluster_obj.R, '\n')

# Perform classification
likelihood = np.zeros((np.shape(test_data)[0], 2))
likelihood[:, 0] = gmcluster.compute_class_likelihood(class_0, test_data)[:, 0]
likelihood[:, 1] = gmcluster.compute_class_likelihood(class_1, test_data)[:, 0]
class_list = np.argmax(likelihood, axis=1)
for n in range(np.shape(test_data)[0]):
    print(test_data[n, :], ' Log-likelihood: ', likelihood[n, :], ' class: ', class_list[n])

# Plot the classification results
plt.plot(test_data[np.argwhere(class_list == 0), 0], test_data[np.argwhere(class_list == 0), 1], 'o', label='class 0')
plt.plot(test_data[np.argwhere(class_list == 1), 0], test_data[np.argwhere(class_list == 1), 1], 'x', label='class 1')
plt.title('Scatter Plot of Multimodal Testing Data After Classification')
plt.xlabel('first component')
plt.ylabel('second component')
plt.legend()
plt.show()
