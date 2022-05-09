import matplotlib.pyplot as plt
import numpy as np


def GMM(N, Pi, mu1, mu2, mu3, R1, R2, R3):
    """Function to generate Gaussian Mixture model with 3 clusters for a given number of observation.

    Args:
        N: number of observation
        Pi: Pi[k] = Prob(Xn=k|K, theta*)
        mu1: mean vector of cluster 1
        mu2: mean vector of cluster 2
        mu3: mean vector of cluster 3
        R1: covariance matrix of cluster 1
        R2: covariance matrix of cluster 2
        R3: covariance matrix of cluster 3

    return:
        x: a M x N matrix of observation vectors with each column being an M-dimensional observation vector,
            totally N observations
        """
    Pi_0 = Pi[0]
    Pi_1 = Pi[1]

    [D, V] = np.linalg.eig(R1)
    A1 = np.matmul(V, np.diagflat(np.sqrt(D)))

    [D, V] = np.linalg.eig(R2)
    A2 = np.matmul(V, np.diagflat(np.sqrt(D)))

    [D, V] = np.linalg.eig(R3)
    A3 = np.matmul(V, np.diagflat(np.sqrt(D)))

    # Generate data with different distributions
    x_0 = np.matmul(A1, np.random.randn(2, N)) + np.matmul(mu1, np.ones((1, N)))
    x_1 = np.matmul(A2, np.random.randn(2, N)) + np.matmul(mu2, np.ones((1, N)))
    x3 = np.matmul(A3, np.random.randn(2, N)) + np.matmul(mu3, np.ones((1, N)))

    SwitchVar = np.matmul(np.ones((2, 1)), np.random.rand(1, N))
    SwitchVar1 = np.zeros(np.shape(SwitchVar))
    SwitchVar2 = np.zeros(np.shape(SwitchVar))
    SwitchVar3 = np.zeros(np.shape(SwitchVar))
    SwitchVar1[SwitchVar < Pi_0] = 1
    SwitchVar2[(SwitchVar >= Pi_0) & (SwitchVar < (Pi_0+Pi_1))] = 1
    SwitchVar3[SwitchVar >= (Pi_0+Pi_1)] = 1

    # Combine data from all distributions
    x = SwitchVar1*x_0 + SwitchVar2*x_1 + SwitchVar3*x3

    return x


def draw_eigen_vecs_for_cov(pixels):
    """Function to draw eigen vectors.

    Args:
        pixels: a N x M matrix of observation vectors with each row being an M-dimensional observation vector,
            totally N observations.
        """

    smean = np.mean(pixels, axis=0)
    R = np.cov(pixels, rowvar=False)
    E, D, _ = np.linalg.svd(R)

    # Plot the eigen vectors
    x = [smean[0], smean[0]+np.sqrt(D[0])*E[0, 0]]
    y = [smean[1], smean[1]+np.sqrt(D[0])*E[1, 0]]
    plt.plot(x, y)
    x = [smean[0], smean[0]+np.sqrt(D[1])*E[0, 1]]
    y = [smean[1], smean[1]+np.sqrt(D[1])*E[1, 1]]
    plt.plot(x, y)

    return


def gen_demo_dataset_1(draw_eigen_vecs=False):
    """Function to generate demo Gaussian Mixture model with 3 clusters for a fixed set of observation.

    Args:
        draw_eigen_vecs: draw eigen vectors on top of the data if set to True

    return:
        x: a N x M matrix of observation vectors with each row being an M-dimensional observation vector,
            totally N observations
        """
    N = 500

    # Generate data
    R1 = [[1, 0.1], [0.1, 1]]
    mu1 = [[2], [2]]

    R2 = [[1, -0.1], [-0.1, 1]]
    mu2 = [[-2], [-2]]

    R3 = [[1, 0.2], [0.2, 0.5]]
    mu3 = [[5.5], [2]]

    Pi = [0.4, 0.4, 0.2]

    x = GMM(N, Pi, mu1, mu2, mu3, R1, R2, R3)

    # Plot the generated data
    plt.plot(x[0, :], x[1, :], 'o')
    if draw_eigen_vecs is True:
        draw_eigen_vecs_for_cov(np.transpose(x))
    plt.title('Scatter Plot of Multimodal Data')
    plt.xlabel('first component')
    plt.ylabel('second component')
    plt.show()

    return np.transpose(x)


def gen_demo_dataset_2():
    """Function to generate demo Gaussian Mixture model with 3 clusters for a fixed set of observation.

    return:
        x_0: a N x M matrix of observation vectors for class 0 training with each row being an M-dimensional observation vector,
            totally N observations
        x_1: a N x M matrix of observation vectors for class 1 training with each row being an M-dimensional observation vector,
            totally N observations
        y: a (N/10) x M matrix of observation vectors for testing with each row being an M-dimensional observation vector,
            totally N observations
        """
    N = 500

    # Generate data for class 0
    R_00 = [[1, 0.1], [0.1, 1]]
    mu_00 = [[2], [2]]

    R_10 = [[1, -0.1], [-0.1, 1]]
    mu_10 = [[-2], [-2]]

    R_20 = [[1, 0.2], [0.2, 0.5]]
    mu_20 = [[5.5], [2]]

    Pi_0 = [0.4, 0.4, 0.2]

    x_0 = GMM(N, Pi_0, mu_00, mu_10, mu_20, R_00, R_10, R_20)
    y_0 = GMM(N//20, Pi_0, mu_00, mu_10, mu_20, R_00, R_10, R_20)

    # Generate data for class 1
    R_01 = [[1, 0.1], [0.1, 1]]
    mu_01 = [[-2], [2]]

    R_11 = [[1, -0.1], [-0.1, 1]]
    mu_11 = [[2], [-2]]

    R_21 = [[1, 0.2], [0.2, 0.5]]
    mu_21 = [[-5.5], [2]]

    Pi_1 = [0.4, 0.4, 0.2]

    x_1 = GMM(N, Pi_1, mu_01, mu_11, mu_21, R_01, R_11, R_21)
    y_1 = GMM(N//20, Pi_1, mu_01, mu_11, mu_21, R_01, R_11, R_21)

    # Combine testing data from both classes
    y = np.concatenate((y_0, y_1), axis=1)

    # Plot the generated training data for class 0 and class 1
    plt.plot(x_0[0, :], x_0[1, :], 'o', label='class 0')
    plt.plot(x_1[0, :], x_1[1, :], 'x', label='class 1')
    plt.title('Scatter Plot of Multimodal Training Data for Class 0 and Class 1')
    plt.xlabel('first component')
    plt.ylabel('second component')
    plt.legend()
    plt.show()

    # Plot the generated testing data
    plt.plot(y[0, :], y[1, :], 'x')
    plt.title('Scatter Plot of Multimodal Testing Data')
    plt.xlabel('first component')
    plt.ylabel('second component')
    plt.show()

    return np.transpose(x_0), np.transpose(x_1), np.transpose(y)
