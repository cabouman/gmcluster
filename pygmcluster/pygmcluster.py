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
        self.Rmin = None
        self.pnk = None


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


def cluster_normalize(mixture):
    """Function to normalize cluster.

    Args:
        mixture: structure containing the Gaussian mixture at a given order

    returns:
        mixture: structure containing the Gaussian mixture of same order with normalized cluster parameters
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
        cluster_obj.const = -(mixture.M*np.log(2 * np.pi) + np.log(np.linalg.det(cluster_obj.R))) / 2
        cluster[k] = cluster_obj
    mixture.cluster = cluster

    return mixture


def init_mixture(pixels, K, est_kind, condition_number):
    """Function to initialize the structure containing the Gaussian mixture.

    Args:
        pixels: a N x M matrix of observation vectors with each row being an M-dimensional observation vector,
            totally N observations
        K: order of the mixture
        est_kind: est_kind = 'diag' constrains the class covariance matrices to be diagonal
            est_kind = 'full' allows the the class covariance matrices to be full matrices
        condition_number: a constant scaling that controls the ratio of  mean to minimum diagonal element of the
            estimated covariance matrices. The default value is 1e5.

    returns:
        mixture: structure containing the initial parameter values for the Gaussian mixture of a given order
        """
    [N, M] = np.shape(pixels)

    mixture = MixtureObj()
    mixture.K = K
    mixture.M = M

    R = (N - 1) * np.cov(pixels, rowvar=False) / N
    if est_kind == 'diag':
        R = np.diag(np.diag(R))

    mixture.Rmin = np.mean(np.diag(R)) / condition_number

    cluster = [None]*K
    cluster_obj = ClusterObj()
    cluster_obj.N = 0
    cluster_obj.pb = 1/K
    cluster_obj.mu = np.expand_dims(pixels[0, :], 1)
    cluster_obj.R = R + mixture.Rmin * np.eye(mixture.M)
    cluster[0] = cluster_obj
    if K > 1:
        period = (N - 1) / (K - 1)
        for k in range(1, K):
            cluster_obj = ClusterObj()
            cluster_obj.N = 0
            cluster_obj.pb = 1/K
            cluster_obj.mu = np.expand_dims(pixels[int((k - 1) * period + 1), :], 1)
            cluster_obj.R = R + mixture.Rmin * np.eye(mixture.M)
            cluster[k] = cluster_obj

    mixture.cluster = cluster
    mixture = cluster_normalize(mixture)

    return mixture


def E_step(mixture, pixels):
    """Function to perform the E-step of the EM algorithm
        1) calculate pnk = Prob(Xn=k|Yn=yn, theta)
        2) calculate likelihood = log ( prob(Y=y|theta) )

    Args:
        mixture: structure containing the Gaussian mixture at a given order
        pixels: a N x M matrix of observation vectors with each row being an M-dimensional observation vector,
            totally N observations

    returns:
        mixture: structure containing the Gaussian mixture  of same order with updated pnk
        likelihood: log ( prob(Y=y|theta) )
        """
    [N, M] = np.shape(pixels)
    pnk = np.zeros((N, mixture.K))
    pb_mat = np.zeros((1, mixture.K))

    for k in range(mixture.K):
        cluster_obj = mixture.cluster[k]
        Y1 = pixels - np.matmul(np.ones((N, 1)), np.transpose(cluster_obj.mu))
        Y2 = -0.5 * np.matmul(Y1, cluster_obj.invR)
        pnk[:, k] = np.sum(Y1 * Y2, axis=1) + cluster_obj.const
        pb_mat[0, k] = cluster_obj.pb

    llmax = np.expand_dims(np.max(pnk, axis=1), axis=1)
    pnk = np.exp(pnk - np.matmul(llmax, np.ones((1, mixture.K))))
    pnk = pnk * np.matmul(np.ones((N, 1)), pb_mat)
    ss = np.expand_dims(np.sum(pnk, axis=1), axis=1)
    likelihood = np.sum(np.log(ss) + llmax)
    pnk = pnk / np.matmul(ss, np.ones((1, mixture.K)))
    mixture.pnk = pnk

    return mixture, likelihood


def M_step(mixture, pixels, est_kind):
    """Function to perform the M-step of the EM algorithm. From the pnk calculated in the E-step, it updates parameters
    of each cluster.

    Args:
        mixture: structure containing the Gaussian mixture at a given order
        pixels: a N x M matrix of observation vectors with each row being an M-dimensional observation vector,
            totally N observations
        est_kind: est_kind = 'diag' constrains the class covariance matrices to be diagonal
            est_kind = 'full' allows the the class covariance matrices to be full matrices

    returns:
        mixture: structure containing the Gaussian mixture  of same order with updated cluster parameters
        """
    for k in range(mixture.K):
        cluster_obj = mixture.cluster[k]
        cluster_obj.N = np.sum(mixture.pnk[:, k])
        cluster_obj.pb = cluster_obj.N
        cluster_obj.mu = np.expand_dims(np.matmul(np.transpose(pixels), mixture.pnk[:, k]) / cluster_obj.N, axis=1)

        R = cluster_obj.R
        for r in range(mixture.M):
            for s in range(r, mixture.M):
                R[r, s] = np.matmul(np.transpose(pixels[:, r] - cluster_obj.mu[r]),
                                    ((pixels[:, s] - cluster_obj.mu[s]) * mixture.pnk[:, k])) / cluster_obj.N
                if r != s:
                    R[s, r] = R[r, s]

        R = R + mixture.Rmin * np.eye(mixture.M)
        if est_kind == 'diag':
            R = np.diag(np.diag(R))
        cluster_obj.R = R
        mixture.cluster[k] = cluster_obj

    mixture = cluster_normalize(mixture)

    return mixture


def EM_iterate(mixture, pixels, est_kind):
    """Function to perform the EM algorithm with a preassigned fixed order K.

    Args:
        mixture: pre-initialized structure for the Gaussian mixture at a given order
        pixels: a N x M matrix of observation vectors with each row being an M-dimensional observation vector,
            totally N observations
        est_kind: est_kind = 'diag' constrains the class covariance matrices to be diagonal
            est_kind = 'full' allows the the class covariance matrices to be full matrices

    returns:
        mixture: structure containing the converged Gaussian mixture with updated parameters
        """
    [N, M] = np.shape(pixels)

    if est_kind == 'full':
        Lc = 1 + M + 0.5 * M * (M + 1)
    else:
        Lc = 1 + M + M

    epsilon = 0.01 * Lc * np.log(N * M)
    [mixture, ll_new] = E_step(mixture, pixels)

    while True:
        ll_old = ll_new
        mixture = M_step(mixture, pixels, est_kind)
        [mixture, ll_new] = E_step(mixture, pixels)
        if (ll_new - ll_old) <= epsilon:
            break

    mixture.rissanen = -ll_new + 0.5 * (mixture.K * Lc - 1) * np.log(N * M)
    mixture.loglikelihood = ll_new

    return mixture


def add_cluster(cluster1, cluster2):
    """Function to combine two clusters.

    Args:
        cluster1: the first cluster
        cluster2: the second cluster

    return:
        cluster3: the combined cluster
        """
    wt1 = cluster1.N / (cluster1.N + cluster2.N)
    wt2 = 1 - wt1
    M = np.shape(cluster1.mu)[0]

    cluster3 = ClusterObj()
    cluster3.mu = wt1 * cluster1.mu + wt2 * cluster2.mu
    cluster3.R = wt1 * (cluster1.R + np.matmul(cluster3.mu - cluster1.mu, np.transpose(cluster3.mu - cluster1.mu)))\
                 + wt2 * (cluster2.R + np.matmul(cluster3.mu-cluster2.mu, np.transpose(cluster3.mu - cluster2.mu)))
    cluster3.invR = np.linalg.inv(cluster3.R)
    cluster3.pb = cluster1.pb + cluster2.pb
    cluster3.N = cluster1.N + cluster2.N
    cluster3.const = -(M * np.log(2 * np.pi) + np.log(np.linalg.det(cluster3.R))) / 2

    return cluster3


def distance(cluster1, cluster2):
    """Function to calculate the distance between two clusters.

    Args:
        cluster1: the first cluster
        cluster2: the second cluster

    return:
        dist: distance between the two clusters
        """
    cluster3 = add_cluster(cluster1, cluster2)
    dist = cluster1.N * cluster1.const + cluster2.N * cluster2.const - cluster3.N * cluster3.const

    return dist


def MDL_reduce_order(mixture, verbose):
    """Function to reduce the order of the mixture by 1 by combining the  two closest distance clusters.

    Args:
        mixture: structure containing the converged Gaussian mixture at a given order
        verbose: true/false, return clustering information if true.

    returns:
        mixture: structure containing the converged Gaussian mixture at an order reduced by one
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


def GM_class_likelihood(mixture, Y):
    """Function to calculate the log-likelihood of data vectors assuming they are generated by the given Gaussian Mixture.

    args:
        mixture: a structure representing a Gaussian mixture
        Y: a NxM matrix, each row is an observation vector of dimension M

    returns:
        ll: a Nx1 matrix with the n-th entry returning the log-likelihood of the n-th observation
        """
    [N, M] = np.shape(Y)
    pnk = np.zeros((N, mixture.K))
    pb_mat = np.zeros((1, mixture.K))

    for k in range(mixture.K):
        cluster_obj = mixture.cluster[k]
        Y1 = Y-np.matmul(np.ones((N, 1)), np.transpose(cluster_obj.mu))
        Y2 = -0.5*np.matmul(Y1, cluster_obj.invR)
        pnk[:, k] = np.sum(Y1 * Y2, axis=1) + mixture.cluster[k].const
        pb_mat[0, k] = cluster_obj.pb

    llmax = np.expand_dims(np.max(pnk, axis=1), axis=1)
    pnk = np.exp(pnk - np.matmul(llmax, np.ones((1, mixture.K))))
    pnk = pnk * np.matmul(np.ones((N, 1)), pb_mat)
    ss = np.expand_dims(np.sum(pnk, axis=1), axis=1)
    ll = np.log(ss)+llmax

    return ll


def split_classes(mixture):
    """ Function to splits the Gaussian mixture with K subclasses into K Gaussian mixtures, each of order 1 containing
    each of the subclasses.

    Args:
        mixture: a structure representing a Gaussian mixture of order mixture.K

    Returns:
        classes: an array of structures, with each representing a Gaussian mixture of order 1 consisting
            of one of the original subclasses
        """
    classes = [None]*mixture.K

    for k in range(mixture.K):
        classes[k] = copy.deepcopy(mixture)
        classes[k].K = 1
        classes[k].cluster = [mixture.cluster[k]]
        classes[k].Rmin = None
        classes[k].rissanen = None
        classes[k].loglikelihood = None

    return classes


def gaussian_mixture(pixels, init_K=20, final_K=0, verbose=True, est_kind='full', condition_number=1e5):
    """Function to perform the EM algorithm to estimate the order, and parameters of a Gaussian Mixture model for a
    given set of observation.

    Args:
        pixels: a N x M matrix of observation vectors with each row being an M-dimensional observation vector,
            totally N observations
        init_K: the initial number of clusters to start with and will be reduced to find the optimal order or the
            desired order based on MDL
        final_K: the desired final number of clusters for the model. Estimate the optimal order if final_K == 0.
        verbose: true/false, return clustering information if true.
        est_kind: est_kind = 'diag' constrains the class covariance matrices to be diagonal
            est_kind = 'full' allows the the class covariance matrices to be full matrices
        condition_number: a constant scaling that controls the ratio of  mean to minimum diagonal element of the
            estimated covariance matrices. The default value is 1e5.

    returns:
        mixture: a list of mixture structures with each containing the converged Gaussian mixture at a given order
            mixture[l].K: order of the mixture
            mixture[l].M: dimension of observation vectors
            mixture[l].cluster: an array of cluster structures with each containing the converged cluster parameters
            mixture[l].rissanen: converged MDL(K)
            mixture[l].loglikelihood: ln( Prob{Y=y|K, theta*} )
            mixture[l].Rmin: intermediate parameter dependent on condition number
            mixture[l].pnk: Prob(Xn=k|Yn=yn, theta)
        opt_mixture: one of the elements in the mixture list.
            If final_K > 0, opt_mixture = mixture[0] and is the mixture with order final_K.
            If final_K == 0, opt_mixture is the one in mixture with minimum MDL
        """
    if (isinstance(init_K, int) is False) or init_K <= 0:
        print('GaussianMixture: initial number of clusters init_K must be a positive integer')
        return
    if (isinstance(final_K, int) is False) or final_K < 0:
        print('GaussianMixture: final number of clusters final_K must be a positive integer or zero')
        return
    if final_K > init_K:
        print('GaussianMixture: final_K cannot be greater than init_K')
        return
    if pixels.dtype != float:
        pixels = pixels.astype(float)
    if (est_kind != 'full') and (est_kind != 'diag'):
        print('GaussianMixture: estimator kind can only be diag or full')
        return

    [N, M] = np.shape(pixels)

    # Calculate the no. of parameters per cluster
    if est_kind == 'full':
        nparams_clust = 1 + M + 0.5 * M * (M + 1)
    else:
        nparams_clust = 1 + M + M

    # Calculate the total no. of data points
    ndata_points = np.size(pixels)

    # Calculate the maximum no.of allowed parameters to be estimated
    max_params = (ndata_points + 1) / nparams_clust - 1
    if init_K > (max_params / 2):
        print('Too many clusters for the given amount of data')
        init_K = int(max_params / 2)
        print('No. of clusters initialized to: ', str(init_K))

    mtr = init_mixture(pixels, init_K, est_kind, condition_number)
    mtr = EM_iterate(mtr, pixels, est_kind)

    if verbose:
        print('K: ', mtr.K, 'rissanen: ', mtr.rissanen)

    mixture = [None]*(mtr.K - max(1, final_K)+1)
    mixture[mtr.K - max(1, final_K)] = copy.deepcopy(mtr)
    while mtr.K > max(1, final_K):
        mtr = MDL_reduce_order(mtr, verbose)
        mtr = EM_iterate(mtr, pixels, est_kind)
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

    return mixture, opt_mixture


def gaussian_mixture_with_decorrelation(pixels, init_K=20, final_K=0, verbose=True, est_kind='full', condition_number=1e5):
    """Function to perform the EM algorithm to estimate the order, and parameters of a Gaussian Mixture model for a
    given set of observation. The parameters are estimated from decorrelated coordinates to condition the problem better.

    Args:
        pixels: a N x M matrix of observation vectors with each row being an M-dimensional observation vector,
            totally N observations
        init_K: the initial number of clusters to start with and will be reduced to find the optimal order or the
            desired order based on MDL
        final_K: the desired final number of clusters for the model. Estimate the optimal order if final_K == 0.
        verbose: true/false, return clustering information if true.
        est_kind: est_kind = 'diag' constrains the class covariance matrices to be diagonal
            est_kind = 'full' allows the the class covariance matrices to be full matrices
        condition_number: a constant scaling that controls the ratio of  mean to minimum diagonal element of the
            estimated covariance matrices. The default value is 1e5.

    returns:
        mixture: a list of mixture structures with each containing the converged Gaussian mixture at a given order
            mixture[l].K: order of the mixture
            mixture[l].M: dimension of observation vectors
            mixture[l].cluster: an array of cluster structures with each containing the converged cluster parameters
            mixture[l].rissanen: converged MDL(K)
            mixture[l].loglikelihood: ln( Prob{Y=y|K, theta*} )
            mixture[l].Rmin: intermediate parameter dependent on condition number
            mixture[l].pnk: Prob(Xn=k|Yn=yn, theta)
        opt_mixture: one of the elements in the mixture list.
            If final_K > 0, opt_mixture = mixture[0] and is the mixture with order final_K.
            If final_K == 0, opt_mixture is the one in mixture with minimum MDL
        """
    if (isinstance(init_K, int) is False) or init_K <= 0:
        print('GaussianMixture: initial number of clusters init_K must be a positive integer')
        return
    if (isinstance(final_K, int) is False) or final_K < 0:
        print('GaussianMixture: final number of clusters final_K must be a positive integer or zero')
        return
    if final_K > init_K:
        print('GaussianMixture: final_K cannot be greater than init_K')
        return
    if pixels.dtype != float:
        pixels = pixels.astype(float)
    if (est_kind != 'full') and (est_kind != 'diag'):
        print('GaussianMixture: estimator kind can only be diag or full')
        return

    # Decorrelate and normalize the data
    smean = np.mean(pixels, axis=0)
    scov = np.cov(pixels, rowvar=False)
    D, E = np.linalg.eig(scov)
    D = np.diag(D)
    T = np.matmul(E, np.linalg.inv(np.sqrt(D)))
    invT = np.linalg.inv(T)
    pixels = np.matmul(pixels - np.transpose(np.matmul(np.diag(smean), np.ones((np.shape(pixels)[1], np.shape(pixels)[0])))), T)

    # Estimate parameters in decorrelated coordinates
    mixture, opt_mixture = gaussian_mixture(pixels, init_K, final_K, verbose, est_kind, condition_number)

    # Transform the parameters back to original coordinates
    if final_K > 0:
        for k in range(mixture[0].K):
            mixture[0].cluster[k].mu = np.transpose(np.matmul(np.transpose(mixture[0].cluster[k].mu), invT) + smean)
            mixture[0].cluster[k].R = np.matmul(np.transpose(invT), np.matmul(mixture[0].cluster[k].R, invT))
            mixture[0].cluster[k].invR = np.matmul(T, np.matmul(mixture[0].cluster[k].invR, np.transpose(T)))
            mixture[0].cluster[k].const = mixture[0].cluster[k].const - np.log(
                np.linalg.det(np.matmul(np.transpose(invT), invT))) / 2
        opt_mixture = mixture[0]
    else:
        for j in range(init_K):
            for k in range(j):
                mixture[j].cluster[k].mu = np.transpose(np.matmul(np.transpose(mixture[j].cluster[k].mu), invT) + smean)
                mixture[j].cluster[k].R = np.matmul(np.transpose(invT), np.matmul(mixture[j].cluster[k].R, invT))
                mixture[j].cluster[k].invR = np.matmul(T, np.matmul(mixture[j].cluster[k].invR, np.transpose(T)))
                mixture[j].cluster[k].const = mixture[j].cluster[k].const - np.log(
                    np.linalg.det(np.matmul(np.transpose(invT), invT))) / 2
        for k in range(opt_mixture.K):
            opt_mixture.cluster[k].mu = np.transpose(np.matmul(np.transpose(opt_mixture.cluster[k].mu), invT) + smean)
            opt_mixture.cluster[k].R = np.matmul(np.transpose(invT), np.matmul(opt_mixture.cluster[k].R, invT))
            opt_mixture.cluster[k].invR = np.matmul(T, np.matmul(opt_mixture.cluster[k].invR, np.transpose(T)))
            opt_mixture.cluster[k].const = opt_mixture.cluster[k].const - np.log(
                np.linalg.det(np.matmul(np.transpose(invT), invT))) / 2

    return mixture, opt_mixture




