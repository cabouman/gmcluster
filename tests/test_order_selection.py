# MDL order selection finds 3 clusters; a fixed order returns exactly that many.

import numpy as np

from gmcluster import GaussianMixture


def make_data(seed=0, n_per=300):
    """Three well-separated 2-D Gaussians."""
    rng = np.random.default_rng(seed)
    centers = np.array([[6.0, 6.0], [-6.0, -6.0], [6.0, -6.0]])
    return np.vstack([rng.standard_normal((n_per, 2)) + c for c in centers])


def test_auto_selects_three():
    data = make_data(seed=0)
    gm = GaussianMixture(num_clusters="auto", max_clusters=6).fit(data)
    assert gm.estimated_num_clusters == 3


def test_fixed_order_returns_two():
    data = make_data(seed=0)
    gm = GaussianMixture(num_clusters=2).fit(data)
    assert gm.estimated_num_clusters == 2
    assert gm.estimated_means.shape[0] == 2
    assert gm.estimated_weights.shape[0] == 2
