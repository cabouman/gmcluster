# In-process equivalence: GMModel.estimate reproduces the private EM engine
# exactly. This proves the model layer does not alter the math.

import numpy as np
import pytest

from gmcluster import GMModel
from gmcluster.gmcluster import _fit_mixture

ATOL = 1e-12


def make_data(seed=0, n=300):
    """Seeded three-cluster 2D data."""
    rng = np.random.default_rng(seed)
    centers = np.array([[4.0, 4.0], [-4.0, -4.0], [4.0, -4.0]])
    parts = [rng.standard_normal((n, 2)) + c for c in centers]
    return np.vstack(parts)


def engine_params(mixture):
    """Pull (K, weights, means, covariances, mdl) out of an engine mixture."""
    K = int(mixture.K)
    weights = np.array([float(c.pb) for c in mixture.cluster])
    means = np.array([c.mu.ravel() for c in mixture.cluster])
    covs = np.array([np.asarray(c.R) for c in mixture.cluster])
    return K, weights, means, covs, mixture.rissanen


@pytest.mark.parametrize("covariance_type,est_kind", [("full", "full"), ("diagonal", "diag")])
@pytest.mark.parametrize("num_clusters", ["auto", 2])
@pytest.mark.parametrize("whiten", [False, True])
def test_model_matches_engine(covariance_type, est_kind, num_clusters, whiten):
    X = make_data()
    max_clusters = 6
    alpha = 0.1

    # Model layer.
    gm, info = GMModel.estimate(X, num_clusters=num_clusters, max_clusters=max_clusters,
                                covariance_type=covariance_type, alpha=alpha,
                                whiten=whiten, verbose=False, return_info=True)

    # Private engine with the same init_K / final_K mapping estimate() uses.
    if num_clusters == "auto":
        init_K, final_K = max_clusters, 0
    else:
        final_K = int(num_clusters)
        init_K = max(max_clusters, final_K)
    mixture, mdl_path = _fit_mixture(X, init_K, final_K, est_kind, alpha, whiten, False)

    K, weights, means, covs, mdl = engine_params(mixture)

    assert gm.num_components == K
    assert np.allclose(gm.weights, weights, atol=ATOL)
    assert np.allclose(gm.means, means, atol=ATOL)
    assert np.allclose(gm.covariances, covs, atol=ATOL)
    assert np.allclose(info.mdl, mdl, atol=ATOL)
    assert info.mdl_path == mdl_path
