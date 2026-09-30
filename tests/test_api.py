# Behavior of the GaussianMixture public surface. Small and fast.

import numpy as np
import pytest

from gmcluster import GaussianMixture


def make_data(seed=0, n=200):
    rng = np.random.default_rng(seed)
    centers = np.array([[4.0, 4.0], [-4.0, -4.0], [4.0, -4.0]])
    return np.vstack([rng.standard_normal((n, 2)) + c for c in centers])


def fitted_model():
    return GaussianMixture(num_clusters="auto", max_clusters=6).fit(make_data())


def test_posterior_rows_sum_to_one():
    gm = fitted_model()
    X = make_data(seed=1, n=50)
    P = gm.posterior(X)
    assert P.shape == (X.shape[0], gm.estimated_num_clusters)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_classify_is_argmax_of_posterior():
    gm = fitted_model()
    X = make_data(seed=1, n=50)
    assert np.array_equal(gm.classify(X), np.argmax(gm.posterior(X), axis=1))


def test_log_likelihood_shape():
    gm = fitted_model()
    X = make_data(seed=1, n=50)
    ll = gm.log_likelihood(X)
    assert ll.shape == (X.shape[0],)


def test_sample_shapes_and_reproducibility():
    gm = fitted_model()
    M = gm.estimated_means.shape[1]

    X = gm.sample(30, rng=np.random.default_rng(7))
    assert X.shape == (30, M)

    X2, labels = gm.sample(30, rng=np.random.default_rng(7), with_labels=True)
    assert X2.shape == (30, M)
    assert labels.shape == (30,)
    # Same seed reproduces the same draw.
    assert np.array_equal(X, X2)
    assert np.all((labels >= 0) & (labels < gm.estimated_num_clusters))


def test_split_clusters():
    gm = fitted_model()
    parts = gm.split_clusters()
    assert len(parts) == gm.estimated_num_clusters
    X = make_data(seed=2, n=20)
    for part in parts:
        assert part.estimated_num_clusters == 1
        assert np.isclose(part.estimated_weights.sum(), 1.0)
        # Each split model is usable as its own density.
        assert part.log_likelihood(X).shape == (X.shape[0],)
        assert part.classify(X).shape == (X.shape[0],)


def test_estimate_shapes_and_weights():
    gm = fitted_model()
    K = gm.estimated_num_clusters
    M = gm.estimated_means.shape[1]
    assert gm.estimated_means.shape == (K, M)
    assert gm.estimated_covariances.shape == (K, M, M)
    assert gm.estimated_weights.shape == (K,)
    assert np.isclose(gm.estimated_weights.sum(), 1.0)


def test_access_before_fit_raises():
    gm = GaussianMixture()
    for name in ["estimated_num_clusters", "estimated_weights", "estimated_means",
                 "estimated_covariances", "mdl", "mdl_path", "converged", "num_iterations"]:
        with pytest.raises(RuntimeError):
            getattr(gm, name)


def test_query_before_fit_raises():
    gm = GaussianMixture()
    with pytest.raises(RuntimeError):
        gm.posterior(make_data(n=5))


@pytest.mark.parametrize("kwargs", [
    {"num_clusters": 0},
    {"num_clusters": -3},
    {"alpha": 2},
    {"alpha": 0},
    {"covariance_type": "bogus"},
    {"max_clusters": 0},
])
def test_bad_constructor_args_raise(kwargs):
    with pytest.raises((ValueError, TypeError)):
        GaussianMixture(**kwargs)


def test_non_2d_input_raises():
    gm = GaussianMixture(max_clusters=4)
    with pytest.raises(ValueError):
        gm.fit(np.zeros(10))


def test_repr():
    gm = GaussianMixture()
    assert "unfitted" in repr(gm)
    gm = fitted_model()
    assert "clusters=" in repr(gm) and "dims=" in repr(gm)
