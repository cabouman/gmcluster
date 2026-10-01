# EM Clustering Library
# Copyright (C) 2022, Charles A Bouman.
# All rights reserved.

import copy
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("gmcluster")


@dataclass
class EstimationInfo:
    """Record describing an estimation run returned by GMModel.estimate.

    Fields:
        num_clusters: the order chosen, K.
        mdl: the description length at the chosen order.
        mdl_path: list of (K, MDL) pairs, one per order visited, ascending K.
        converged: whether EM converged.
        num_iterations: number of EM iterations at the chosen order.
    """

    num_clusters: int
    mdl: float
    mdl_path: list
    converged: bool
    num_iterations: int


class GMModel:
    """Gaussian mixture distribution defined by its weights, means, and covariances.

    A model holds the parameters and provides the operations on the
    distribution: sample, classify, posterior, log_density, and split. Build a
    model from chosen parameters with the constructor, or from data with the
    GMModel.estimate classmethod.
    """

    def __init__(self, weights, means, covariances):
        """Build a model from parameters, validating shapes and conditions.

        Args:
            weights: shape (K,), each positive, summing to 1.
            means: shape (K, M).
            covariances: shape (K, M, M), each symmetric and invertible.

        Raises:
            ValueError: on any shape or condition violation.
        """
        self._set(weights, means, covariances)

    def _set(self, weights, means, covariances):
        """Validate parameters, build the internal mixture, store the arrays."""
        weights, means, covariances = _check_parameters(weights, means, covariances)
        self._mixture = _mixture_from_parameters(weights, means, covariances)
        self._weights = weights
        self._means = means
        self._covariances = covariances

    @property
    def weights(self):
        """Return the component weights, shape (K,)."""
        return self._weights

    @property
    def means(self):
        """Return the component means, shape (K, M)."""
        return self._means

    @property
    def covariances(self):
        """Return the component covariances, shape (K, M, M)."""
        return self._covariances

    @property
    def num_components(self):
        """Return the number of components, K."""
        return self._mixture.K

    @property
    def num_features(self):
        """Return the number of features, M."""
        return self._mixture.M

    def set_parameters(self, weights, means, covariances):
        """Replace the parameters in place, with the constructor's validation."""
        self._set(weights, means, covariances)

    def sample(self, num_samples=1, rng=None, with_labels=False):
        """Draw samples from the mixture.

        Args:
            num_samples: number of samples to draw.
            rng: a numpy Generator or seed for reproducibility.
            with_labels: also return the component index of each sample.

        Returns:
            X of shape (num_samples, M), or (X, labels) with labels of shape
            (num_samples,) if with_labels is True.
        """
        rng = np.random.default_rng(rng)
        weights = self.weights
        means = self.means
        covs = self.covariances
        M = means.shape[1]

        labels = rng.choice(len(weights), size=num_samples, p=weights)
        samples = np.empty((num_samples, M))

        # Draw each component's points from its Gaussian via a symmetric-eigen factor.
        for k in range(len(weights)):
            idx = np.nonzero(labels == k)[0]
            if idx.size == 0:
                continue
            eigvals, eigvecs = np.linalg.eigh(covs[k])
            eigvals = np.clip(eigvals, 0.0, None)
            factor = eigvecs * np.sqrt(eigvals)
            z = rng.standard_normal((idx.size, M))
            samples[idx] = means[k] + z @ factor.T

        if with_labels:
            return samples, labels
        return samples

    def posterior(self, X):
        """Return p(component | x), shape (N, K), rows summing to 1."""
        X = _check_data(X, n_features=self._mixture.M)
        _, _ = E_step(self._mixture, X)
        return np.array(self._mixture.pnk)

    def classify(self, X):
        """Return the most-probable component index per point, shape (N,)."""
        return np.argmax(self.posterior(X), axis=1)

    def log_density(self, X):
        """Return the per-point log density log p(x), shape (N,)."""
        X = _check_data(X, n_features=self._mixture.M)
        return _class_log_likelihood(self._mixture, X).ravel()

    def split(self):
        """Return a list of K single-component GMModel objects, each with weight 1."""
        K = self._mixture.K
        return [GMModel(np.array([1.0]), self.means[k:k + 1], self.covariances[k:k + 1])
                for k in range(K)]

    @classmethod
    def estimate(cls, X, num_clusters="auto", max_clusters=20, covariance_type="full",
                 alpha=0.1, whiten=False, verbose=False, return_info=False):
        """Estimate a model from data by EM with MDL order selection.

        Args:
            X: (num_points, num_features) 2D float array of observations.
            num_clusters: "auto" to select the order by MDL, or a positive int to fix it.
            max_clusters: positive int, the ceiling for the "auto" search.
            covariance_type: "full" or "diagonal".
            alpha: covariance regularization, 0 < alpha <= 1 (1 spherical, ->0 elliptical).
            whiten: decorrelate coordinates before estimation.
            verbose: report progress through the logging module.
            return_info: if True, return (model, info) with an EstimationInfo.

        Returns:
            A GMModel, or (GMModel, EstimationInfo) if return_info is True.
        """
        # num_clusters: "auto" or a positive int (bool is not a count).
        if isinstance(num_clusters, str):
            if num_clusters != "auto":
                raise ValueError('num_clusters must be "auto" or a positive int')
        elif isinstance(num_clusters, bool) or not isinstance(num_clusters, (int, np.integer)):
            raise TypeError('num_clusters must be "auto" or a positive int')
        elif num_clusters <= 0:
            raise ValueError("num_clusters must be a positive int")

        # max_clusters: positive int.
        if isinstance(max_clusters, bool) or not isinstance(max_clusters, (int, np.integer)):
            raise TypeError("max_clusters must be a positive int")
        if max_clusters <= 0:
            raise ValueError("max_clusters must be a positive int")

        # covariance_type: "full" or "diagonal" (mapped to the internal "diag").
        if covariance_type == "full":
            est_kind = "full"
        elif covariance_type == "diagonal":
            est_kind = "diag"
        else:
            raise ValueError('covariance_type must be "full" or "diagonal"')

        # alpha: 0 < alpha <= 1.
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.integer, np.floating)):
            raise TypeError("alpha must be a number in (0, 1]")
        if not (0 < alpha <= 1):
            raise ValueError("alpha must satisfy 0 < alpha <= 1")

        if not isinstance(whiten, bool):
            raise TypeError("whiten must be a bool")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a bool")
        if not isinstance(return_info, bool):
            raise TypeError("return_info must be a bool")

        X = _check_data(X)

        if num_clusters == "auto":
            init_K = int(max_clusters)
            final_K = 0
        else:
            final_K = int(num_clusters)
            init_K = max(int(max_clusters), final_K)

        mixture, mdl_path = _fit_mixture(X, init_K, final_K, est_kind,
                                         float(alpha), whiten, verbose)

        clusters = mixture.cluster
        weights = np.array([float(c.pb) for c in clusters])
        means = np.array([c.mu.ravel() for c in clusters])
        covariances = np.array([np.asarray(c.R) for c in clusters])
        model = cls(weights, means, covariances)

        if return_info:
            info = EstimationInfo(
                num_clusters=int(mixture.K),
                mdl=mixture.rissanen,
                mdl_path=mdl_path,
                converged=True,
                num_iterations=getattr(mixture, "num_iterations", None),
            )
            return model, info
        return model

    def __repr__(self):
        return "GMModel(num_components={}, num_features={})".format(
            self._mixture.K, self._mixture.M)


def _check_parameters(weights, means, covariances):
    """Validate mixture parameters and return them as float ndarrays.

    Checks weights shape (K,) with positive entries summing to 1 within 1e-6,
    means shape (K, M), and covariances shape (K, M, M) with each matrix
    symmetric and invertible. Raises ValueError on any violation.
    """
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)

    if weights.ndim != 1:
        raise ValueError("weights must have shape (K,)")
    K = weights.shape[0]
    if K < 1:
        raise ValueError("weights must have at least one component")
    if not np.all(weights > 0):
        raise ValueError("weights must all be positive")
    if abs(np.sum(weights) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1 within 1e-6")

    if means.ndim != 2 or means.shape[0] != K:
        raise ValueError("means must have shape (K, M)")
    M = means.shape[1]

    if covariances.shape != (K, M, M):
        raise ValueError("covariances must have shape (K, M, M)")
    for k in range(K):
        R = covariances[k]
        if not np.allclose(R, R.T):
            raise ValueError("each covariance must be symmetric")
        try:
            np.linalg.inv(R)
        except np.linalg.LinAlgError:
            raise ValueError("each covariance must be invertible")

    return weights, means, covariances


def _mixture_from_parameters(weights, means, covariances):
    """Build an engine MixtureObj from validated parameters.

    Fills each cluster's weight, mean (as an (M, 1) column), and covariance,
    then calls cluster_normalize to fill invR and const.
    """
    K = weights.shape[0]
    M = means.shape[1]

    mixture = MixtureObj()
    mixture.K = K
    mixture.M = M
    mixture.cluster = []
    for k in range(K):
        cluster_obj = ClusterObj()
        cluster_obj.pb = float(weights[k])
        cluster_obj.mu = means[k].reshape(M, 1)
        cluster_obj.R = covariances[k]
        mixture.cluster.append(cluster_obj)
    mixture.D_reg = None
    mixture = cluster_normalize(mixture)

    return mixture


def _check_data(X, n_features=None):
    """Validate X as a 2D float array and return it. Raise on bad input."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (num_points, num_features)")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("X must be a numeric array")
    if X.dtype != float:
        X = X.astype(float)
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("X has {} features; the model was fit on {}".format(
            X.shape[1], n_features))
    return X


class MixtureObj:
    """Bag of parameters for a Gaussian mixture (the engine's mixture record)."""

    def __init__(self):
        """Initialize the fields to None."""
        self.K = None
        self.M = None
        self.cluster = None
        self.rissanen = None
        self.loglikelihood = None
        self.pnk = None
        self.D_reg = None
        self.num_iterations = None


class ClusterObj:
    """Bag of parameters for one cluster (the engine's cluster record)."""

    def __init__(self):
        """Initialize the fields to None."""
        self.N = None
        self.pb = None
        self.mu = None
        self.R = None
        self.invR = None
        self.const = None


def _fit_mixture(data, init_K, final_K, est_kind, alpha, whiten, verbose):
    """Run EM order selection and return (opt_mixture, mdl_path).

    Starts at init_K clusters, merges down one order at a time, and either
    returns the model at final_K (fixed order) or the minimum-MDL model
    (final_K == 0). mdl_path lists (K, MDL) for every order visited.

    Args:
        data: (N, M) 2D float array of observations.
        init_K: number of clusters to start from.
        final_K: fixed final order, or 0 to select by MDL.
        est_kind: "full" or "diag".
        alpha: covariance regularization in (0, 1].
        whiten: decorrelate coordinates before clustering.
        verbose: log progress if True.

    Returns:
        (opt_mixture, mdl_path) where opt_mixture is a MixtureObj and mdl_path
        is a list of (K, MDL) tuples in ascending K.
    """
    if whiten:
        data, T, smean = decorrelate_and_normalize(data)

    [N, M] = np.shape(data)

    # Number of parameters per cluster.
    if est_kind == 'full':
        nparams_clust = 1 + M + 0.5 * M * (M + 1)
    else:
        nparams_clust = 1 + M + M

    ndata_points = np.size(data)

    # Cap the starting order to the amount of data.
    max_params = (ndata_points + 1) / nparams_clust - 1
    if init_K > (max_params / 2):
        init_K = int(max_params / 2)
        if verbose:
            logger.warning("Too many clusters for the given data; init_K set to %d", init_K)

    mtr = init_mixture(data, init_K, est_kind, alpha)
    mtr = EM_iterate(mtr, data, est_kind, alpha)
    if verbose:
        logger.info("K: %d  MDL: %g", mtr.K, mtr.rissanen)

    mixture = [None] * (mtr.K - max(1, final_K) + 1)
    mixture[mtr.K - max(1, final_K)] = copy.deepcopy(mtr)
    while mtr.K > max(1, final_K):
        mtr = MDL_reduce_order(mtr, False)
        mtr = EM_iterate(mtr, data, est_kind, alpha)
        if verbose:
            logger.info("K: %d  MDL: %g", mtr.K, mtr.rissanen)
        mixture[mtr.K - max(1, final_K)] = copy.deepcopy(mtr)

    if final_K > 0:
        opt_mixture = mixture[0]
    else:
        min_riss = mixture[-1].rissanen
        opt_l = len(mixture) - 1
        for l in range(len(mixture) - 2, -1, -1):
            if mixture[l].rissanen < min_riss:
                min_riss = mixture[l].rissanen
                opt_l = l
        opt_mixture = copy.deepcopy(mixture[opt_l])

    # MDL for every order visited, in ascending K.
    mdl_path = [(m.K, m.rissanen) for m in mixture if m is not None]

    if whiten:
        opt_mixture = transform_back_to_original_coordinates(opt_mixture, T, smean)

    return opt_mixture, mdl_path


def _class_log_likelihood(mixture, data):
    """Return per-point log density log p(x) as an (N, 1) array."""
    [N, M] = np.shape(data)
    pnk = np.zeros((N, mixture.K))
    pb_mat = np.zeros((1, mixture.K))

    for k in range(mixture.K):
        cluster_obj = mixture.cluster[k]
        Y1 = data - cluster_obj.mu.T
        Y2 = -0.5 * Y1 @ cluster_obj.invR
        pnk[:, k] = np.sum(Y1 * Y2, axis=1) + mixture.cluster[k].const
        pb_mat[0, k] = cluster_obj.pb

    llmax = np.expand_dims(np.max(pnk, axis=1), axis=1)
    pnk = np.exp(pnk - llmax)
    pnk = pnk * pb_mat
    ss = np.expand_dims(np.sum(pnk, axis=1), axis=1)
    ll = np.log(ss) + llmax

    return ll


def cluster_normalize(mixture):
    """Normalize cluster weights to sum to 1 and refresh invR and const.

    Args:
        mixture(class): a Gaussian mixture record.

    Returns:
        class object: the mixture with normalized weights and updated invR/const.
        """
    cluster = mixture.cluster

    s = 0
    for k in range(mixture.K):
        cluster_obj = cluster[k]
        s = s + np.sum(cluster_obj.pb)

    for k in range(mixture.K):
        cluster_obj = cluster[k]
        cluster_obj.pb = cluster_obj.pb / s
        cluster_obj.invR = np.linalg.inv(cluster_obj.R)
        cluster_obj.const = -(mixture.M * np.log(2 * np.pi) + np.log(np.linalg.det(cluster_obj.R))) / 2
        cluster[k] = cluster_obj
    mixture.cluster = cluster

    return mixture


def ridge_regression(R, est_kind, alpha, D_reg=None):
    """Regularize and constrain a class covariance matrix.

    Args:
        R(ndarray): the initial class covariance matrix
        est_kind(str):
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
        D_reg = np.mean(np.diag(R)) * np.eye(R.shape[0])
    else:
        return_D_reg = False

    # Ensure that the alpha of R is <= alpha
    R = (1.0 - (alpha ** 2)) * R + (alpha ** 2) * D_reg

    if return_D_reg:
        return R, D_reg
    else:
        return R


def init_mixture(data, K, est_kind, alpha):
    """Initialize a Gaussian mixture record of a given order.

    Args:
        data(ndarray): an N x M 2D array of observation vectors with each row being an M-dimensional observation vector,
            totally N observations
        K(int): order of the mixture
        est_kind(str):
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
    R = (N - 1) * np.cov(data, rowvar=False) / N

    # Regularize the covariance matrix and impose constrains
    R, D_reg = ridge_regression(R, est_kind, alpha)

    # Allocate and array of K clusters
    cluster = [None] * K

    # Initalize first element of cluster
    cluster_obj = ClusterObj()
    cluster_obj.N = 0
    cluster_obj.pb = 1 / K
    cluster_obj.mu = np.expand_dims(data[0, :], 1)
    cluster_obj.R = R
    cluster[0] = cluster_obj

    # Initialize remaining clusters in array
    if K > 1:
        period = (N - 1) / (K - 1)
        for k in range(1, K):
            cluster_obj = ClusterObj()
            cluster_obj.N = 0
            cluster_obj.pb = 1 / K
            cluster_obj.mu = np.expand_dims(data[int((k - 1) * period + 1), :], 1)
            cluster_obj.R = R
            cluster[k] = cluster_obj

    mixture.cluster = cluster
    mixture.D_reg = D_reg
    mixture = cluster_normalize(mixture)

    return mixture


def E_step(mixture, data):
    """Perform the E-step: compute responsibilities pnk and the log-likelihood.

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
        Y1 = data - cluster_obj.mu.T
        Y2 = -0.5 * Y1 @ cluster_obj.invR
        pnk[:, k] = np.sum(Y1 * Y2, axis=1) + cluster_obj.const
        pb_mat[0, k] = cluster_obj.pb

    llmax = np.expand_dims(np.max(pnk, axis=1), axis=1)
    pnk = np.exp(pnk - llmax)
    pnk = pnk * pb_mat
    ss = np.expand_dims(np.sum(pnk, axis=1), axis=1)
    likelihood = np.sum(np.log(ss) + llmax)
    pnk = pnk / ss
    mixture.pnk = pnk

    return mixture, likelihood


def M_step(mixture, data, est_kind, alpha):
    """Perform the M-step: update each cluster's weight, mean, and covariance.

    Args:
        mixture(class): a structure representing the parameters for a Gaussian mixture of a given order
        data(ndarray): an N x M 2D array of observation vectors with each row being an M-dimensional observation vector,
            totally N observations
        est_kind(str):
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
        cluster_obj.mu = np.expand_dims((data.T @ mixture.pnk[:, k]) / cluster_obj.N, axis=1)

        # Weighted covariance about the cluster mean
        w = mixture.pnk[:, k]
        Xc = data - cluster_obj.mu.T
        R = (Xc.T * w) @ Xc / cluster_obj.N

        # Regularize the covariance matrix and impose constrains
        R = ridge_regression(R, est_kind, alpha, mixture.D_reg)

        cluster_obj.R = R
        mixture.cluster[k] = cluster_obj

    mixture = cluster_normalize(mixture)

    return mixture


def EM_iterate(mixture, data, est_kind, alpha):
    """Run EM to convergence at a fixed order K.

    Records the number of EM iterations on mixture.num_iterations. The counter
    is bookkeeping only; the update equations and the convergence test are
    unchanged.

    Args:
        mixture(class): a structure representing the parameters for a Gaussian mixture of a given order
        data(ndarray): an N x M 2D array of observation vectors with each row being an M-dimensional observation vector,
            totally N observations
        est_kind(str):
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
        Lc = 1 + M + 0.5 * M * (M + 1)
    else:
        Lc = 1 + M + M

    epsilon = 0.01 * Lc * np.log(N * M)
    [mixture, ll_new] = E_step(mixture, data)

    n_iter = 0
    while True:
        ll_old = ll_new
        mixture = M_step(mixture, data, est_kind, alpha)
        [mixture, ll_new] = E_step(mixture, data)
        n_iter += 1
        if (ll_new - ll_old) <= epsilon:
            break

    mixture.rissanen = -ll_new + 0.5 * (mixture.K * Lc - 1) * np.log(N * M)
    mixture.loglikelihood = ll_new
    mixture.num_iterations = n_iter

    return mixture


def add_cluster(cluster1, cluster2):
    """Combine two clusters into one.

    Args:
        cluster1(class): the first cluster
        cluster2(class): the second cluster

    Returns:
        class object: the combined cluster
        """
    wt1 = cluster1.N / (cluster1.N + cluster2.N)
    wt2 = 1 - wt1
    M = np.shape(cluster1.mu)[0]

    cluster3 = ClusterObj()
    cluster3.mu = wt1 * cluster1.mu + wt2 * cluster2.mu
    cluster3.R = wt1 * (cluster1.R + (cluster3.mu - cluster1.mu) @ (cluster3.mu - cluster1.mu).T) \
                 + wt2 * (cluster2.R + (cluster3.mu - cluster2.mu) @ (cluster3.mu - cluster2.mu).T)
    cluster3.invR = np.linalg.inv(cluster3.R)
    cluster3.pb = cluster1.pb + cluster2.pb
    cluster3.N = cluster1.N + cluster2.N
    cluster3.const = -(M * np.log(2 * np.pi) + np.log(np.linalg.det(cluster3.R))) / 2

    return cluster3


def distance(cluster1, cluster2):
    """Return the merge distance between two clusters.

    Args:
        cluster1(class): the first cluster
        cluster2(class): the second cluster

    Returns:
        float: distance between the two clusters
        """
    cluster3 = add_cluster(cluster1, cluster2)
    dist = cluster1.N * cluster1.const + cluster2.N * cluster2.const - cluster3.N * cluster3.const

    return dist


def MDL_reduce_order(mixture, verbose):
    """Reduce the order by one by merging the two closest clusters.

    Args:
        mixture(class): a structure containing the parameters for the converged Gaussian mixture of a given order K
        verbose(bool): true/false, return clustering information if true

    Returns:
        class object: a structure containing the parameters for the converged Gaussian mixture of order (K-1)
        """
    K = mixture.K

    min_dist = np.inf
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            dist = distance(mixture.cluster[k1], mixture.cluster[k2])
            if (k1 == 0 and k2 == 1) or (dist < min_dist):
                mink1 = k1
                mink2 = k2
                min_dist = dist
    if verbose:
        logger.info("combining cluster %d and %d", mink1, mink2)

    mixture.cluster[mink1] = add_cluster(mixture.cluster[mink1], mixture.cluster[mink2])
    mixture.cluster[mink2: (K - 1)] = mixture.cluster[(mink2 + 1): K]
    mixture.cluster = mixture.cluster[:(K - 1)]
    mixture.K = K - 1
    mixture = cluster_normalize(mixture)

    return mixture


def decorrelate_and_normalize(data):
    """Decorrelate and normalize data, returning the transform for inversion.

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
    T = E @ np.linalg.inv(np.sqrt(D))
    data = (data - (np.diag(smean) @ np.ones((np.shape(data)[1], np.shape(data)[0]))).T) @ T

    return data, T, smean


def transform_back_to_original_coordinates(opt_mixture, T, smean):
    """Map mixture parameters from whitened coordinates back to the original ones.

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
        opt_mixture.cluster[k].mu = (opt_mixture.cluster[k].mu.T @ invT + smean).T
        opt_mixture.cluster[k].R = invT.T @ opt_mixture.cluster[k].R @ invT
        opt_mixture.cluster[k].invR = T @ opt_mixture.cluster[k].invR @ T.T
        opt_mixture.cluster[k].const = opt_mixture.cluster[k].const - np.log(
            np.linalg.det(invT.T @ invT)) / 2

    return opt_mixture
