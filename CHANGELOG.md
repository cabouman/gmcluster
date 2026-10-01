# Changelog

## v0.4.0 — 2026-10-01

- New `GMModel` class as the single public API. A model holds a Gaussian
  mixture's parameters and is specified exactly by them.
- Build a model from parameters you choose with
  `GMModel(weights, means, covariances)`, or from data with the classmethod
  `GMModel.estimate(X, num_clusters="auto", max_clusters=20, ...)`, which
  returns a model. Pass `return_info=True` to also get an `EstimationInfo`
  record with `num_clusters`, `mdl`, `mdl_path`, `converged`, and
  `num_iterations`.
- Read the parameters from `weights`, `means`, `covariances`,
  `num_components`, and `num_features`. Change them with `set_parameters`.
- Methods: `sample`, `classify`, `posterior`, `log_density`, and `split`.
- `classify` returns the most-probable component, the one that maximizes the
  posterior p(component | x).

## v0.3.0 — 2026-09-30

- New `GaussianMixture` class as the public API. The constructor holds the
  settings (`num_clusters`, `max_clusters`, `covariance_type`, `alpha`,
  `whiten`, `verbose`) and `fit(X)` runs the EM algorithm on a
  `(num_points, num_features)` array.
- Set `num_clusters="auto"` to choose the number of clusters by the minimum
  description length (MDL) criterion, or pass a positive integer to fix it.
- After `fit`, the results are read from `estimated_num_clusters`,
  `estimated_weights`, `estimated_means`, and `estimated_covariances`, with
  the diagnostics `mdl`, `mdl_path`, `converged`, and `num_iterations`.
  Reading any of these before `fit` raises a clear error.
- Methods on a fitted model: `posterior(X)`, `classify(X)`,
  `log_likelihood(X)`, `sample(n)`, and `split_clusters()`.
- Packaging: the build uses `pyproject.toml` with setuptools; the version is
  single-sourced from `gmcluster.__version__`. Automated tests run on GitHub
  Actions. Citation metadata (`CITATION.cff`) is included.
