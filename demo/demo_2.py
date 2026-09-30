import numpy as np
import matplotlib.pyplot as plt
from gmcluster import GaussianMixture

"""
Demo of GaussianMixture for binary classification: fit one mixture per class,
then label test points by the class with the higher log-likelihood.
"""


def sample_mixture(mu, R, pb, N, rng):
    """Draw N points from a Gaussian mixture with the given means, covariances, and weights."""
    labels = rng.choice(len(pb), size=N, p=pb)
    X = np.empty((N, mu.shape[1]))
    for k in range(len(pb)):
        idx = np.nonzero(labels == k)[0]
        L = np.linalg.cholesky(R[k])
        X[idx] = mu[k] + rng.standard_normal((idx.size, mu.shape[1])) @ L.T
    return X


# Ground-truth mixtures used to generate the demo data (same weights/covariances for both classes).
N = 500  # samples per class
pb = np.array([0.4, 0.4, 0.2])
R = np.array([
    [[1.0, 0.1], [0.1, 1.0]],
    [[1.0, -0.1], [-0.1, 1.0]],
    [[1.0, 0.2], [0.2, 0.5]],
])
mu_class0 = np.array([[2.0, 2.0], [-2.0, -2.0], [5.5, 2.0]])
mu_class1 = np.array([[-2.0, 2.0], [2.0, -2.0], [-5.5, 2.0]])

rng = np.random.default_rng(0)
train_data_0 = sample_mixture(mu_class0, R, pb, N, rng)
train_data_1 = sample_mixture(mu_class1, R, pb, N, rng)
test_data_0 = sample_mixture(mu_class0, R, pb, N // 5, rng)
test_data_1 = sample_mixture(mu_class1, R, pb, N // 5, rng)
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

# Fit one mixture per class.
class_0 = GaussianMixture(num_clusters="auto").fit(train_data_0)
class_1 = GaussianMixture(num_clusters="auto").fit(train_data_1)

for name, gm in [('class 0', class_0), ('class 1', class_1)]:
    print('\n%s estimated order: %d' % (name, gm.estimated_num_clusters))
    for i in range(gm.estimated_num_clusters):
        print('\nCluster: ', i)
        print('pi: ', gm.estimated_weights[i])
        print('mean: \n', gm.estimated_means[i])
        print('covar: \n', gm.estimated_covariances[i], '\n')

# Label each test point by the class with the higher log-likelihood.
likelihood = np.zeros((test_data.shape[0], 2))
likelihood[:, 0] = class_0.log_likelihood(test_data)
likelihood[:, 1] = class_1.log_likelihood(test_data)
class_list = np.argmax(likelihood, axis=1)

# Plot the classification results
plt.plot(test_data[class_list == 0, 0], test_data[class_list == 0, 1], 'o', label='class 0')
plt.plot(test_data[class_list == 1, 0], test_data[class_list == 1, 1], 'x', label='class 1')
plt.title('Scatter Plot of Multimodal Testing Data After Classification')
plt.xlabel('first component')
plt.ylabel('second component')
plt.legend()
plt.show()
