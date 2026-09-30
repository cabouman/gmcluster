# Smoke test: a tiny fit returns a well-formed mixture.

import numpy as np
import gmcluster


def test_smoke_tiny_fit():
    np.random.seed(0)
    a = np.random.randn(200, 2) + np.array([5.0, 5.0])
    b = np.random.randn(200, 2) + np.array([-5.0, -5.0])
    data = np.vstack([a, b])

    opt = gmcluster.estimate_gm_params(data, init_K=5, final_K=0, verbose=False)

    assert isinstance(opt.K, int)
    assert opt.K >= 1
    assert len(opt.cluster) == opt.K
