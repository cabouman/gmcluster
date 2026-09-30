import numpy as np
import matplotlib.pyplot as plt
from gmcluster import GaussianMixture

"""
Demo of GaussianMixture: estimate the order and parameters of a Gaussian
mixture and classify the points by their most-likely cluster.
"""

# Ground-truth mixture used to generate the demo data.
N = 500  # number of samples
pb = np.array([0.4, 0.4, 0.2])  # cluster probabilities
R = np.array([
    [[1.0, 0.1], [0.1, 1.0]],
    [[1.0, -0.1], [-0.1, 1.0]],
    [[1.0, 0.2], [0.2, 0.5]],
])  # cluster covariance matrices
mu = np.array([[2.0, 2.0], [-2.0, -2.0], [5.5, 2.0]])  # cluster means

# Generate demo data by drawing a component per sample, then a point from it.
rng = np.random.default_rng(0)
labels = rng.choice(len(pb), size=N, p=pb)
pixels = np.empty((N, 2))
for k in range(len(pb)):
    idx = np.nonzero(labels == k)[0]
    L = np.linalg.cholesky(R[k])
    pixels[idx] = mu[k] + rng.standard_normal((idx.size, 2)) @ L.T

# Plot the generated samples
plt.plot(pixels[:, 0], pixels[:, 1], 'o')
plt.title('Scatter Plot of Multimodal Data')
plt.xlabel('first component')
plt.ylabel('second component')
plt.show()

# Estimate the order and cluster parameters.
gm = GaussianMixture(num_clusters="auto").fit(pixels)

print('\nestimated order: ', gm.estimated_num_clusters)
for i in range(gm.estimated_num_clusters):
    print('\nCluster: ', i)
    print('pi: ', gm.estimated_weights[i])
    print('mean: \n', gm.estimated_means[i])
    print('covar: \n', gm.estimated_covariances[i], '\n')

# Classify each point by its most-likely cluster.
class_list = gm.classify(pixels)

# Plot the classification results
markers = ['o', 'x', '*', 's', 'd', 'v', '^', '<', '>', 'p']
for k in range(gm.estimated_num_clusters):
    pts = pixels[class_list == k]
    plt.plot(pts[:, 0], pts[:, 1], markers[k % len(markers)], label='class %d' % k)
plt.title('Gaussian mixture classification')
plt.xlabel('first component')
plt.ylabel('second component')
plt.legend()
plt.show()
