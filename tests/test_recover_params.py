# The estimate recovers the true cluster means on well-separated data.

import numpy as np

from gmcluster import GMModel


def make_mixture(seed=0, n_per=300):
    """Three well-separated 2-D Gaussians. Returns (data, true_means)."""
    rng = np.random.default_rng(seed)
    true_means = np.array([[6.0, 6.0], [-6.0, -6.0], [6.0, -6.0]])
    data = np.vstack([rng.standard_normal((n_per, 2)) + m for m in true_means])
    return data, true_means


def match_nearest(true_means, means):
    """For each true mean, the distance to its nearest estimated mean."""
    dists = []
    for t in true_means:
        d = np.linalg.norm(means - t, axis=1)
        dists.append(d.min())
    return np.array(dists)


def test_recover_means():
    data, true_means = make_mixture(seed=0)
    gm = GMModel.estimate(data, num_clusters="auto", max_clusters=6)

    assert gm.num_components == 3
    # Every true center has an estimated center close to it.
    assert np.all(match_nearest(true_means, gm.means) < 0.5)
