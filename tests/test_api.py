# Behavior of the GMModel public surface. Small and fast.

import numpy as np
import pytest

from gmcluster import GMModel


def make_data(seed=0, n=200):
    rng = np.random.default_rng(seed)
    centers = np.array([[4.0, 4.0], [-4.0, -4.0], [4.0, -4.0]])
    return np.vstack([rng.standard_normal((n, 2)) + c for c in centers])


def estimated_model():
    return GMModel.estimate(make_data(), num_clusters="auto", max_clusters=6)


def hand_model():
    """A two-component model built from chosen parameters."""
    weights = np.array([0.6, 0.4])
    means = np.array([[0.0, 0.0], [5.0, 5.0]])
    covariances = np.array([np.eye(2), 0.5 * np.eye(2)])
    return GMModel(weights, means, covariances)


def test_posterior_rows_sum_to_one():
    gm = estimated_model()
    X = make_data(seed=1, n=50)
    P = gm.posterior(X)
    assert P.shape == (X.shape[0], gm.num_components)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_classify_is_argmax_of_posterior():
    gm = estimated_model()
    X = make_data(seed=1, n=50)
    assert np.array_equal(gm.classify(X), np.argmax(gm.posterior(X), axis=1))


def test_log_density_shape():
    gm = estimated_model()
    X = make_data(seed=1, n=50)
    ld = gm.log_density(X)
    assert ld.shape == (X.shape[0],)


def test_sample_shapes_and_reproducibility():
    gm = estimated_model()
    M = gm.num_features

    X = gm.sample(30, rng=7)
    assert X.shape == (30, M)

    X2, labels = gm.sample(30, rng=7, with_labels=True)
    assert X2.shape == (30, M)
    assert labels.shape == (30,)
    # The same seed reproduces the same draw.
    assert np.array_equal(X, X2)
    assert np.all((labels >= 0) & (labels < gm.num_components))


def test_split_returns_single_component_models():
    gm = estimated_model()
    parts = gm.split()
    assert len(parts) == gm.num_components
    X = make_data(seed=2, n=20)
    for part in parts:
        assert part.num_components == 1
        assert np.isclose(part.weights.sum(), 1.0)
        # Each split model is usable as its own density.
        assert part.log_density(X).shape == (X.shape[0],)
        assert part.classify(X).shape == (X.shape[0],)


def test_estimated_shapes_and_weights():
    gm = estimated_model()
    K = gm.num_components
    M = gm.num_features
    assert gm.means.shape == (K, M)
    assert gm.covariances.shape == (K, M, M)
    assert gm.weights.shape == (K,)
    assert np.isclose(gm.weights.sum(), 1.0)


def test_estimate_return_info():
    gm, info = GMModel.estimate(make_data(), num_clusters="auto", max_clusters=6,
                                return_info=True)
    assert info.num_clusters == gm.num_components
    assert info.num_iterations >= 1
    assert info.converged
    assert len(info.mdl_path) >= 1


def test_model_from_parameters():
    gm = hand_model()
    assert gm.num_components == 2
    assert gm.num_features == 2
    assert np.allclose(gm.weights, [0.6, 0.4])
    assert gm.means.shape == (2, 2)
    assert gm.covariances.shape == (2, 2, 2)
    X = make_data(seed=3, n=25)
    assert gm.posterior(X).shape == (X.shape[0], 2)
    assert gm.log_density(X).shape == (X.shape[0],)
    assert gm.sample(10, rng=0).shape == (10, 2)


def test_set_parameters():
    gm = hand_model()
    new_means = np.array([[1.0, 1.0], [-1.0, -1.0]])
    gm.set_parameters(np.array([0.5, 0.5]), new_means, np.array([np.eye(2), np.eye(2)]))
    assert np.allclose(gm.weights, [0.5, 0.5])
    assert np.allclose(gm.means, new_means)


@pytest.mark.parametrize("weights,means,covariances", [
    (np.array([0.6, -0.1, 0.5]), np.zeros((3, 2)), np.array([np.eye(2)] * 3)),   # negative weight
    (np.array([0.6, 0.6]), np.zeros((2, 2)), np.array([np.eye(2)] * 2)),         # weights not summing to 1
    (np.array([0.5, 0.5]), np.zeros((2, 2)), np.array([[[1.0, 0.9], [0.0, 1.0]], np.eye(2)])),  # non-symmetric
    (np.array([0.5, 0.5]), np.zeros((2, 2)), np.array([np.zeros((2, 2)), np.eye(2)])),          # singular
    (np.array([0.5, 0.5]), np.zeros((3, 2)), np.array([np.eye(2)] * 2)),         # shape mismatch
])
def test_invalid_parameters_raise(weights, means, covariances):
    with pytest.raises(ValueError):
        GMModel(weights, means, covariances)


@pytest.mark.parametrize("kwargs", [
    {"num_clusters": 0},
    {"num_clusters": -3},
    {"alpha": 2},
    {"alpha": 0},
    {"covariance_type": "bogus"},
    {"max_clusters": 0},
])
def test_bad_estimate_args_raise(kwargs):
    with pytest.raises((ValueError, TypeError)):
        GMModel.estimate(make_data(), **kwargs)


def test_non_2d_input_raises():
    with pytest.raises(ValueError):
        GMModel.estimate(np.zeros(10), max_clusters=4)


def test_repr():
    assert "GMModel" in repr(hand_model())
    assert "GMModel" in repr(estimated_model())
