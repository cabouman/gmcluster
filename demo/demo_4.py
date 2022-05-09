import numpy as np
import matplotlib.pyplot as plt
import pycluster

"""
This file demonstrates a demo of the EM algorithm to estimate the order, and parameters of a Gaussian Mixture model 
and perform unsupervised cluster classification with and without decorrelated coordinates using "pycluster" library.
"""

# Generate demo data
pixels = pycluster.sim.gen_demo_dataset_1(draw_eigen_vecs=True)

# Estimate optimal order and clustering data using original coordinates
[mtrs, omtr] = pycluster.gaussian_mixture(pixels, 20, 0, True, 'full', 1e5)

print('\noptimal order: ', omtr.K)
for i in range(omtr.K):
    cluster_obj = omtr.cluster[i]
    print('\nCluster: ', i)
    print('pi: ', cluster_obj.pb)
    print('mean: \n', cluster_obj.mu)
    print('covar: \n', cluster_obj.R, '\n')

# Split classes
mtrs = pycluster.split_classes(omtr)

# Perform classification
likelihood = np.zeros((np.shape(pixels)[0], len(mtrs)))
for k in range(len(mtrs)):
    likelihood[:, k] = pycluster.GM_class_likelihood(mtrs[k], pixels)[:, 0]
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
[mtrs, omtr] = pycluster.gaussian_mixture_with_decorrelation(pixels, 20, 0, True, 'full', 1e5)

print('\noptimal order: ', omtr.K)
for i in range(omtr.K):
    cluster_obj = omtr.cluster[i]
    print('\nCluster: ', i)
    print('pi: ', cluster_obj.pb)
    print('mean: \n', cluster_obj.mu)
    print('covar: \n', cluster_obj.R, '\n')

# Split classes
mtrs = pycluster.split_classes(omtr)

# Perform classification
likelihood = np.zeros((np.shape(pixels)[0], len(mtrs)))
for k in range(len(mtrs)):
    likelihood[:, k] = pycluster.GM_class_likelihood(mtrs[k], pixels)[:, 0]
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
