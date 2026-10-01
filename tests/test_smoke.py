# Smoke test: a tiny estimate returns a well-formed model.

import numpy as np
from gmcluster import GMModel


def test_smoke_tiny_estimate():
    np.random.seed(0)
    a = np.random.randn(200, 2) + np.array([5.0, 5.0])
    b = np.random.randn(200, 2) + np.array([-5.0, -5.0])
    data = np.vstack([a, b])

    gm = GMModel.estimate(data, num_clusters="auto", max_clusters=5, verbose=False)

    assert gm.num_components >= 1
    assert gm.weights.shape == (gm.num_components,)
