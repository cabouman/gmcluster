# Smoke test: a tiny fit returns a well-formed model.

import numpy as np
from gmcluster import GaussianMixture


def test_smoke_tiny_fit():
    np.random.seed(0)
    a = np.random.randn(200, 2) + np.array([5.0, 5.0])
    b = np.random.randn(200, 2) + np.array([-5.0, -5.0])
    data = np.vstack([a, b])

    gm = GaussianMixture(num_clusters="auto", max_clusters=5, verbose=False).fit(data)

    assert isinstance(gm.estimated_num_clusters, int)
    assert gm.estimated_num_clusters >= 1
    assert gm.estimated_weights.shape == (gm.estimated_num_clusters,)
