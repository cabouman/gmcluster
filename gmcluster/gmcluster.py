# EM Clustering Library
# Copyright (C) 2022, Charles A Bouman.
# All rights reserved.

import copy
import numpy as np


class MixtureObj:
    """Class to store the parameters for mixture object."""

    def __init__(self):
        """Function to initialize the parameters of the class MixtureObj."""
        self.K = None
        self.M = None
        self.cluster = None
        self.rissanen = None
        self.loglikelihood = None
        self.pnk = None
        self.D_reg = None


class ClusterObj:
    """Class to store the parameters for cluster object."""

    def __init__(self):
        """Function to initialize the parameters of the class ClusterObj."""
        self.N = None
        self.pb = None
        self.mu = None
        self.R = None
        self.invR = None
        self.const = None


def estimate_gm_params(data, init_K=20, final_K=0, verbose=True, est_kind='full', decorrelate_coordinates=False, alpha=0.1):
    """Function to perform the EM algorithm to estimate the order, and parameters of a Gaussian mixture model for a
    given set of observations.

    Args:
        data(ndarray): an N x M 2D array of observation vectors with each row being an M-dimensional observation vector,
            totally N observations
        init_K(int,optional): the initial number of clusters to start with and will be reduced to find the optimal order
            or the desired order based on MDL
        final_K(int,optional): the final number of clusters for the model. Estimate the optimal order if final_K == 0
        verbose(bool,optional): true/false, return clustering information if true
        est_kind(string,optional):
            - est_kind = 'diag' constrains the class covariance matrices to be diagonal
            - est_kind = 'full' allows the class covariance matrices to be full matrices
        decorrelate_coordinates(bool,optional): true/false, decorrelate the coordinates to better condition the problem
            if true
        alpha(float,optional): a constant (0 < alpha <= 1) that controls the shape of the cluster by regularizing the
            covariance matrices. alpha = 1 gives the cluster a spherical shape and alpha = 0 gives the cluster an
            elliptical shape. The default value is 0.1

    Returns:
        class object: a structure with optimum Gaussian mixture parameters, where
            - opt_mixture.K: order of the mixture
            - opt_mixture.M: dimension of observation vectors
            - opt_mixture.cluster: an array of cluster structures with each containing the converged cluster parameters
            - opt_mixture.rissanen: converged MDL(K)
            - opt_mixture.loglikelihood: ln( Prob{Y=y|K, theta*} )
            - opt_mixture.pnk: Prob(Xn=k|Yn=yn, theta)
            - opt_mixture.D_reg: a diagonal matrix used for regularizing class covariance matrices
        """
    if (isinstance(init_K, int) is False) or init_K <= 0:
        print('estimate_gm_params: initial number of clusters init_K must be a positive integer')
        return
    if (isinstance(final_K, int) is False) or final_K < 0:
        print('estimate_gm_params: final number of clusters final_K must be a positive integer or zero')
        return
    if final_K > init_K:
        print('estimate_gm_params: final_K cannot be greater than init_K')
        return
    if data.dtype != float:
        data = data.astype(float)
    if (est_kind != 'full') and (est_kind != 'diag'):
        print('estimate_gm_params: estimator kind can only be diag or full')
        return
    if (alpha <= 0) or (alpha > 1):
        print('estimate_gm_params: alpha must be greater than 0 and less than or equal to 1')
        return

    if decorrelate_coordinates:
        data, T, smean = decorrelate_and_normalize(data)

    [N, M] = np.shape(data)

    # Calculate the no. of parameters per cluster
    if est_kind == 'full':
        nparams_clust = 1 + M + 0.5*M*(M + 1)
    else:
        nparams_clust = 1 + M + M

    # Calculate the total no. of data points
    ndata_points = np.size(data)

    # Calculate the maximum no.of allowed parameters to be estimated
    max_params = (ndata_points + 1)/nparams_clust - 1
    if init_K > (max_params/2):
        print('Too many clusters for the given amount of data')
        init_K = int(max_params/2)
        print('No. of clusters initialized to: ', str(init_K))

    mtr = init_mixture(data, init_K, est_kind, alpha)
    mtr = EM_iterate(mtr, data, est_kind, alpha)

    if verbose:
        print('K: ', mtr.K, 'rissanen: ', mtr.rissanen)

    mixture = [None]*(mtr.K - max(1, final_K)+1)
    mixture[mtr.K - max(1, final_K)] = copy.deepcopy(mtr)
    while mtr.K > max(1, final_K):
        mtr = MDL_reduce_order(mtr, verbose)
        mtr = EM_iterate(mtr, data, est_kind, alpha)
        if verbose:
            print('K: ', mtr.K, 'rissanen: ', mtr.rissanen)
        mixture[mtr.K - max(1, final_K)] = copy.deepcopy(mtr)

    if final_K > 0:
        opt_mixture = mixture[0]
    else:
        min_riss = mixture[-1].rissanen
        opt_l = len(mixture)-1
        for l in range(len(mixture)-2, -1, -1):
            if mixture[l].rissanen < min_riss:
                min_riss = mixture[l].rissanen
                opt_l = l
        opt_mixture = copy.deepcopy(mixture[opt_l])

    if decorrelate_coordinates:
        opt_mixture = transform_back_to_original_coordinates(opt_mixture, T, smean)

    return opt_mixture


def split_classes(mixture):
    """ Function to splits the Gaussian mixture with K subclasses into K Gaussian mixtures, each of order 1 containing
    each of the subclasses.

    Args:
        mixture(class): a structure representing the parameters for a Gaussian mixture of order K (K subclasses)

    Returns:
        list: a list of K structures, each representing the parameters for a Gaussian mixture of order 1 (one of the K
        original subclasses)

        """
    classes = [None]*mixture.K

    for k in range(mixture.K):
        classes[k] = copy.deepcopy(mixture)
        classes[k].K = 1
        classes[k].cluster = [mixture.cluster[k]]
        classes[k].rissanen = None
        classes[k].loglikelihood = None

    return classes


def compute_class_likelihood(mixture, data):
    """Function to calculate the log-likelihood of data vectors assuming they are generated by a given Gaussian mixture.

    args:
        mixture(class): a structure representing the parameters for a Gaussian mixture of order 1
        data(ndarray): an N x M 2D array of observation vectors with each row being an M-dimensional observation vector,
            totally N observations

    Returns:
        ndarray: an N x 1 array with the n-th entry returning the log-likelihood of the n-th observation for the given
        Gaussian mixture of order 1
        """
    [N, M] = np.shape(data)
    pnk = np.zeros((N, mixture.K))
    pb_mat = np.zeros((1, mixture.K))

    for k in range(mixture.K):
        cluster_obj = mixture.cluster[k]
        Y1 = data-np.ones((N, 1))@cluster_obj.mu.T
        Y2 = -0.5*Y1@cluster_obj.invR
        pnk[:, k] = np.sum(Y1*Y2, axis=1) + mixture.cluster[k].const
        pb_mat[0, k] = cluster_obj.pb

    llmax = np.expand_dims(np.max(pnk, axis=1), axis=1)
    pnk = np.exp(pnk - llmax@np.ones((1, mixture.K)))
    pnk = pnk*np.ones((N, 1))@pb_mat
    ss = np.expand_dims(np.sum(pnk, axis=1), axis=1)
    ll = np.log(ss)+llmax

    return ll


def generate_gm_samples(mixture, N=500):
    """Function to generate Gaussian mixture model with K clusters for a given set of parameters and number of
    observations.

    Args:
        mixture(class): a structure representing the parameters for a Gaussian mixture of a given order
        N(int,optional): number of observation

    Returns:
        ndarray: an N x M 2D array of observation vectors with each row being an M-dimensional observation vector,
        totally N observations
        """
    gm_samples = np.zeros((mixture.M, N))
    switch_var = np.ones((mixture.M, 1))@np.random.rand(1, N)
    pb_low = 0

    for k in range(mixture.K):
        cluster_obj = mixture.cluster[k]
        pb = cluster_obj.pb
        mu = cluster_obj.mu
        R = cluster_obj.R

        # Eigen decomposition
        [D, V] = np.linalg.eig(R)
        A = V@np.diagflat(np.sqrt(D))

        # Generate data with the given distributions
        x = A@np.random.randn(mixture.M, N) + mu@np.ones((1, N))

        # Limit number of samples from the distribution using the given probability value
        switch_var_k = np.zeros(np.shape(switch_var))
        pb_high = pb_low + pb
        switch_var_k[(switch_var >= pb_low) & (switch_var < pb_high)] = 1
        pb_low = pb_high

        # Combine data from all distributions
        gm_samples = gm_samples + switch_var_k*x

    return gm_samples.T


def cluster_normalize(mixture):
    """Function to normalize cluster.

    Args:
        mixture(class): a structure representing the parameters for a Gaussian mixture of a given order

    Returns:
        class object: a structure containing the Gaussian mixture parameters of the same order with normalized cluster
        parameters
        """
    cluster = mixture.cluster

    s = 0
    for k in range(mixture.K):
        cluster_obj = cluster[k]
        s = s + np.sum(cluster_obj.pb)

    for k in range(mixture.K):
        cluster_obj = cluster[k]
        cluster_obj.pb = cluster_obj.pb/s
        cluster_obj.invR = np.linalg.inv(cluster_obj.R)
        cluster_obj.const = -(mixture.M*np.log(2*np.pi) + np.log(np.linalg.det(cluster_obj.R)))/2
        cluster[k] = cluster_obj
    mixture.cluster = cluster

    return mixture


def ridge_regression(R, est_kind, alpha, D_reg=None):
    """Function to regularize and constrain class covariance matrix.

    Args:
        R(ndarray): the initial class covariance matrix
        est_kind(string):
            - est_kind = 'diag' constrains the class covariance matrices to be diagonal
            - est_kind = 'full' allows the class covariance matrices to be full matrices
        alpha(float): a constant (0 < alpha <= 1) that controls the shape of the cluster by regularizing the covariance
            matrices. alpha = 1 gives the cluster a spherical shape and alpha = 0 gives the cluster an elliptical shape.
            The default value is 0.1
        D_reg(ndarray,optional): a diagonal matrix used as the regularization term in the class covariance matrix update
            equation. The function will compute it from the given R if set to default

    Returns:
        ndarray: the regularized and constrained class covariance matrix
        tuple/ndarray: (R, D_reg) or just R (if return_D_reg is false), where
            - R(ndarray): the regularized and constrained class covariance matrix
            - D_reg(ndarray): diagonal matrix used as the regularization term
        """
    if est_kind == 'diag':
        R = np.diag(np.diag(R))

    if D_reg is None:
        return_D_reg = True
        D_reg = np.mean(np.diag(R))*np.eye(R.shape[0])
    else:
        return_D_reg = False

    # Ensure that the alpha of R is <= alpha
    R = (1.0 - (alpha**2))*R + (alpha**2)*D_reg

    if return_D_reg:
        return R, D_reg
    else:
        return R


def init_mixture(data, K, est_kind, alpha):
    """Function to initialize the structure containing the Gaussian mixture.

    Args:
        data(ndarray): an N x M 2D array of observation vectors with each row being an M-dimensional observation vector,
            totally N observations
        K(int): order of the mixture
        est_kind(string):
            - est_kind = 'diag' constrains the class covariance matrices to be diagonal
            - est_kind = 'full' allows the class covariance matrices to be full matrices
        alpha(float): a constant (0 < alpha <= 1) that controls the shape of the cluster by regularizing the covariance
            matrices. alpha = 1 gives the cluster a spherical shape and alpha = 0 gives the cluster an elliptical shape.
            The default value is 0.1

    Returns:
        class object: a structure containing the initial parameter values for the Gaussian mixture of a given order
        """
    [N, M] = np.shape(data)

    mixture = MixtureObj()
    mixture.K = K
    mixture.M = M

    # Compute sample covariance for entire data set
    R = (N - 1)*np.cov(data, rowvar=False)/N

    # Regularize the covariance matrix and impose constrains
    R, D_reg = ridge_regression(R, est_kind, alpha)

    # Allocate and array of K clusters
    cluster = [None]*K

    # Initalize first element of cluster
    cluster_obj = ClusterObj()
    cluster_obj.N = 0
    cluster_obj.pb = 1/K
    cluster_obj.mu = np.expand_dims(data[0, :], 1)
    cluster_obj.R = R
    cluster[0] = cluster_obj

    # Initialize remaining clusters in array
    if K > 1:
        period = (N - 1)/(K - 1)
        for k in range(1, K):
            cluster_obj = ClusterObj()
            cluster_obj.N = 0
            cluster_obj.pb = 1/K
            cluster_obj.mu = np.expand_dims(data[int((k - 1)*period + 1), :], 1)
            cluster_obj.R = R
            cluster[k] = cluster_obj

    mixture.cluster = cluster
    mixture.D_reg = D_reg
    mixture = cluster_normalize(mixture)

    return mixture


def E_step(mixture, data):
    """Function to perform the E-step of the EM algorithm
    	1) calculate pnk = Prob(Xn=k|Yn=yn, theta)
    	2) calculate likelihood = log ( prob(Y=y|theta) )

    Args:
        mixture(class): a structure representing the parameters for a Gaussian mixture of a given order
        data(ndarray): an N x M 2D array of observation vectors with each row being an M-dimensional observation vector,
            totally N observations
    Returns:
        tuple: (mixture, likelihood), where
            - mixture(class): a structure containing the Gaussian mixture parameters for the same order with updated pnk
            - likelihood(float): log ( prob(Y=y|theta) )
        """
    [N, M] = np.shape(data)
    pnk = np.zeros((N, mixture.K))
    pb_mat = np.zeros((1, mixture.K))

    for k in range(mixture.K):
        cluster_obj = mixture.cluster[k]
        Y1 = data - np.ones((N, 1))@cluster_obj.mu.T
        Y2 = -0.5*Y1@cluster_obj.invR
        pnk[:, k] = np.sum(Y1*Y2, axis=1) + cluster_obj.const
        pb_mat[0, k] = cluster_obj.pb

    llmax = np.expand_dims(np.max(pnk, axis=1), axis=1)
    pnk = np.exp(pnk - llmax@np.ones((1, mixture.K)))
    pnk = pnk*np.ones((N, 1))@pb_mat
    ss = np.expand_dims(np.sum(pnk, axis=1), axis=1)
    likelihood = np.sum(np.log(ss) + llmax)
    pnk = pnk/(ss@np.ones((1, mixture.K)))
    mixture.pnk = pnk

    return mixture, likelihood


def M_step(mixture, data, est_kind, alpha):
    """Function to perform the M-step of the EM algorithm. From the pnk calculated in the E-step, it updates parameters
    of each cluster.

    Args:
        mixture(class): a structure representing the parameters for a Gaussian mixture of a given order
        data(ndarray): an N x M 2D array of observation vectors with each row being an M-dimensional observation vector,
            totally N observations
        est_kind(string):
            - est_kind = 'diag' constrains the class covariance matrices to be diagonal
            - est_kind = 'full' allows the class covariance matrices to be full matrices
        alpha(float): a constant (0 < alpha <= 1) that controls the shape of the cluster by regularizing the covariance
            matrices. alpha = 1 gives the cluster a spherical shape and alpha = 0 gives the cluster an elliptical shape.
            The default value is 0.1

    Returns:
        class object: a structure containing the parameters for a Gaussian mixture of the same order with updated
        cluster parameters
        """
    for k in range(mixture.K):
        cluster_obj = mixture.cluster[k]
        cluster_obj.N = np.sum(mixture.pnk[:, k])
        cluster_obj.pb = cluster_obj.N
        cluster_obj.mu = np.expand_dims(data.T@mixture.pnk[:, k]/cluster_obj.N, axis=1)

        R = cluster_obj.R
        for r in range(mixture.M):
            for s in range(r, mixture.M):
                R[r, s] = ((data[:, r] - cluster_obj.mu[r]).T@((data[:, s] - cluster_obj.mu[s])*mixture.pnk[:, k]))\
                          /cluster_obj.N
                if r != s:
                    R[s, r] = R[r, s]

        # Regularize the covariance matrix and impose constrains
        R = ridge_regression(R, est_kind, alpha, mixture.D_reg)

        cluster_obj.R = R
        mixture.cluster[k] = cluster_obj

    mixture = cluster_normalize(mixture)

    return mixture


def EM_iterate(mixture, data, est_kind, alpha):
    """Function to perform the EM algorithm with a preassigned fixed order K.

    Args:
        mixture(class): a structure representing the parameters for a Gaussian mixture of a given order
        data(ndarray): an N x M 2D array of observation vectors with each row being an M-dimensional observation vector,
            totally N observations
        est_kind(string):
            - est_kind = 'diag' constrains the class covariance matrices to be diagonal
            - est_kind = 'full' allows the class covariance matrices to be full matrices
        alpha(float): a constant (0 < alpha <= 1) that controls the shape of the cluster by regularizing the covariance
            matrices. alpha = 1 gives the cluster a spherical shape and alpha = 0 gives the cluster an elliptical shape.
            The default value is 0.1

    Returns:
        class object: a structure containing the parameters for the converged Gaussian mixture of order K
        """
    [N, M] = np.shape(data)

    if est_kind == 'full':
        Lc = 1 + M + 0.5*M*(M + 1)
    else:
        Lc = 1 + M + M

    epsilon = 0.01*Lc*np.log(N*M)
    [mixture, ll_new] = E_step(mixture, data)

    while True:
        ll_old = ll_new
        mixture = M_step(mixture, data, est_kind, alpha)
        [mixture, ll_new] = E_step(mixture, data)
        if (ll_new - ll_old) <= epsilon:
            break

    mixture.rissanen = -ll_new + 0.5*(mixture.K*Lc - 1)*np.log(N*M)
    mixture.loglikelihood = ll_new

    return mixture


def add_cluster(cluster1, cluster2):
    """Function to combine two clusters.

    Args:
        cluster1(class): the first cluster
        cluster2(class): the second cluster

    Returns:
        class object: the combined cluster
        """
    wt1 = cluster1.N/(cluster1.N + cluster2.N)
    wt2 = 1 - wt1
    M = np.shape(cluster1.mu)[0]

    cluster3 = ClusterObj()
    cluster3.mu = wt1*cluster1.mu + wt2*cluster2.mu
    cluster3.R = wt1*(cluster1.R + (cluster3.mu - cluster1.mu)@(cluster3.mu - cluster1.mu).T)\
                 + wt2*(cluster2.R + (cluster3.mu-cluster2.mu)@(cluster3.mu - cluster2.mu).T)
    cluster3.invR = np.linalg.inv(cluster3.R)
    cluster3.pb = cluster1.pb + cluster2.pb
    cluster3.N = cluster1.N + cluster2.N
    cluster3.const = -(M*np.log(2*np.pi) + np.log(np.linalg.det(cluster3.R)))/2

    return cluster3


def distance(cluster1, cluster2):
    """Function to calculate the distance between two clusters.

    Args:
        cluster1(class): the first cluster
        cluster2(class): the second cluster

    Returns:
        float: distance between the two clusters
        """
    cluster3 = add_cluster(cluster1, cluster2)
    dist = cluster1.N*cluster1.const + cluster2.N*cluster2.const - cluster3.N*cluster3.const

    return dist


def MDL_reduce_order(mixture, verbose):
    """Function to reduce the order of a given mixture by 1 by combining the two closest clusters.

    Args:
        mixture(class): a structure containing the parameters for the converged Gaussian mixture of a given order K
        verbose(bool): true/false, return clustering information if true

    Returns:
        class object: a structure containing the parameters for the converged Gaussian mixture of order (K-1)
        """
    K = mixture.K
    
    min_dist = np.inf
    for k1 in range(K):
        for k2 in range(k1+1, K):
            dist = distance(mixture.cluster[k1], mixture.cluster[k2])
            if (k1 == 0 and k2 == 1) or (dist < min_dist):
                mink1 = k1
                mink2 = k2
                min_dist = dist
    if verbose:
        print('combining cluster: ', str(mink1), ' and ', str(mink2))

    mixture.cluster[mink1] = add_cluster(mixture.cluster[mink1], mixture.cluster[mink2])
    mixture.cluster[mink2: (K - 1)] = mixture.cluster[(mink2 + 1): K]
    mixture.cluster = mixture.cluster[:(K - 1)]
    mixture.K = K - 1
    mixture = cluster_normalize(mixture)

    return mixture


def decorrelate_and_normalize(data):
    """ Function to decorrelate and normalize data

    Args:
        data(ndarray): an N x M 2D array of observation vectors with each row being an M-dimensional observation vector,
            totally N observations

    Returns:
        tuple: (data, T, smean), where
            - data(ndarray): decorrelated and normalized observation vectors
            - T(ndarray): transformation 2D array
            - smean(ndarray): mean values
        """
    # Decorrelate and normalize the data
    smean = np.mean(data, axis=0)
    scov = np.cov(data, rowvar=False)
    D, E = np.linalg.eig(scov)
    D = np.diag(D)
    T = E@np.linalg.inv(np.sqrt(D))
    data = (data - (np.diag(smean)@np.ones((np.shape(data)[1], np.shape(data)[0]))).T)@T

    return data, T, smean


def transform_back_to_original_coordinates(opt_mixture, T, smean):
    """ Function to transform the optimum mixture parameters to correspond to original coordinates

    Args:
        opt_mixture(class): a structure representing the optimum Gaussian mixture parameters corresponding to
            decorrelated coordinates
        T(ndarray): transformation 2D array
        smean(ndarray): mean values

    Returns:
        class object: a structure representing the optimum Gaussian mixture parameters corresponding to original
            coordinates
        """
    invT = np.linalg.inv(T)
    # Transform the parameters back to original coordinates
    for k in range(opt_mixture.K):
        opt_mixture.cluster[k].mu = (opt_mixture.cluster[k].mu.T@invT + smean).T
        opt_mixture.cluster[k].R = invT.T@opt_mixture.cluster[k].R@invT
        opt_mixture.cluster[k].invR = T@opt_mixture.cluster[k].invR@T.T
        opt_mixture.cluster[k].const = opt_mixture.cluster[k].const - np.log(
            np.linalg.det(invT.T@invT))/2

    return opt_mixture

