import numpy as np
import matplotlib.pyplot as plt
import pygmcluster

"""
This file demonstrates a demo of the EM algorithm to estimate the orders and parameters of 2 different Gaussian Mixture 
models and perform binary maximum likelihood classification of test observations based on the estimated parameters for 
the 2 mixture models using "PyGMCluster" library.
"""

# Generate demo data
train_data_0, train_data_1, test_data = pygmcluster.gen_demo_dataset_2()

# Estimate optimal order and clustering data for class 0
[mtrs, class_0] = pygmcluster.gaussian_mixture(train_data_0, 20, 0, True, 'full', 1e5)

print('\noptimal order: ', class_0.K)
for i in range(class_0.K):
    cluster_obj = class_0.cluster[i]
    print('\nCluster: ', i)
    print('pi: ', cluster_obj.pb)
    print('mean: \n', cluster_obj.mu)
    print('covar: \n', cluster_obj.R, '\n')


# Estimate optimal order and clustering data for class 1
[mtrs, class_1] = pygmcluster.gaussian_mixture(train_data_1, 20, 0, True, 'full', 1e5)

print('\noptimal order: ', class_1.K)
for i in range(class_1.K):
    cluster_obj = class_1.cluster[i]
    print('\nCluster: ', i)
    print('pi: ', cluster_obj.pb)
    print('mean: \n', cluster_obj.mu)
    print('covar: \n', cluster_obj.R, '\n')

# Perform classification
likelihood = np.zeros((np.shape(test_data)[0], 2))
likelihood[:, 0] = pygmcluster.GM_class_likelihood(class_0, test_data)[:, 0]
likelihood[:, 1] = pygmcluster.GM_class_likelihood(class_1, test_data)[:, 0]
class_list = np.argmax(likelihood, axis=1)
for n in range(np.shape(test_data)[0]):
    print(test_data[n, :], ' Log-likelihood: ', likelihood[n, :], ' class: ', class_list[n])

# Plot the classification results
plt.plot(test_data[np.argwhere(class_list == 0), 0], test_data[np.argwhere(class_list == 0), 1], 'o', label='class 0')
plt.plot(test_data[np.argwhere(class_list == 1), 0], test_data[np.argwhere(class_list == 1), 1], 'x', label='class 1')
plt.title('Gaussian mixture classification')
plt.xlabel('first component')
plt.ylabel('second component')
plt.legend()
plt.show()
