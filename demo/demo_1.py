import numpy as np
import matplotlib.pyplot as plt
from gmcluster import GMModel

"""
Demo of GMModel: estimate the order and parameters of a Gaussian
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

# Ground-truth model used to generate the demo data.
truth = GMModel(pb, mu, R)
pixels = truth.sample(N, rng=0)

# Plot the generated samples
plt.plot(pixels[:, 0], pixels[:, 1], 'o')
plt.title('Scatter Plot of Multimodal Data')
plt.xlabel('first component')
plt.ylabel('second component')
plt.show()

# Estimate the order and cluster parameters.
gm = GMModel.estimate(pixels, num_clusters="auto")

print('\nestimated order: ', gm.num_components)
for i in range(gm.num_components):
    print('\nCluster: ', i)
    print('pi: ', gm.weights[i])
    print('mean: \n', gm.means[i])
    print('covar: \n', gm.covariances[i], '\n')

# Classify each point by its most-likely cluster.
class_list = gm.classify(pixels)

# Plot the classification results
markers = ['o', 'x', '*', 's', 'd', 'v', '^', '<', '>', 'p']
for k in range(gm.num_components):
    pts = pixels[class_list == k]
    plt.plot(pts[:, 0], pts[:, 1], markers[k % len(markers)], label='class %d' % k)
plt.title('Gaussian mixture classification')
plt.xlabel('first component')
plt.ylabel('second component')
plt.legend()
plt.show()
