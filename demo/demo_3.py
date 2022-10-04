import numpy as np
import matplotlib.pyplot as plt
import pygmcluster

"""
This file demonstrates a demo of the EM algorithm to estimate the order and parameters of a Gaussian Mixture model 
and perform unsupervised classification of clusters within the mixture using "PyGMCluster" library.
"""

# Generate demo data
pixels = pygmcluster.sim.gen_demo_dataset_1()

# Estimate optimal order and clustering data
[mtrs, omtr] = pygmcluster.gaussian_mixture(pixels)

print('\noptimal order: ', omtr.K)
for i in range(omtr.K):
    cluster_obj = omtr.cluster[i]
    print('\nCluster: ', i)
    print('pi: ', cluster_obj.pb)
    print('mean: \n', cluster_obj.mu)
    print('covar: \n', cluster_obj.R, '\n')

# Split classes
mtrs = pygmcluster.split_classes(omtr)
likelihood = np.zeros((np.shape(pixels)[0], len(mtrs)))
for k in range(len(mtrs)):
    likelihood[:, k] = pygmcluster.GM_class_likelihood(mtrs[k], pixels)[:, 0]

# Perform classification
class_list = np.argmax(likelihood, axis=1)
for n in range(np.shape(pixels)[0]):
    print(pixels[n, :], ' Log-likelihood: ', likelihood[n, :], ' class: ', class_list[n])

# Plot the classification results
plt.plot(pixels[np.argwhere(class_list == 0), 0], pixels[np.argwhere(class_list == 0), 1], 'o', label='class 0')
plt.plot(pixels[np.argwhere(class_list == 1), 0], pixels[np.argwhere(class_list == 1), 1], 'x', label='class 1')
plt.plot(pixels[np.argwhere(class_list == 2), 0], pixels[np.argwhere(class_list == 2), 1], '*', label='class 2')
plt.title('Gaussian mixture classification')
plt.xlabel('first component')
plt.ylabel('second component')
plt.legend()
plt.show()
