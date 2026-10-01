===================
Releasing a version
===================

GMCluster is published to PyPI by the GitHub Actions workflow in
``.github/workflows/release.yml``.  You drive it from your machine with
``dev_scripts/release.sh``, which stamps the version, tags the release, and lets
GitHub Actions do the upload.  Uploads use Trusted Publishing, so no API token
is ever stored or typed.

The examples below release version ``0.3.0``.  Replace ``0.3.0`` with the
version you are releasing.

Dry run on TestPyPI (optional, recommended the first time)
==========================================================

This proves the whole pipeline on a throwaway upload before the real one.  A
PyPI version number can never be reused, so it is worth doing once.

1. Publish a release candidate::

       dev_scripts/release.sh 0.3.0rc1

   This stamps the version, pushes ``prerelease``, and creates a GitHub
   pre-release tagged ``v0.3.0rc1``.  GitHub Actions builds the package and
   uploads it to TestPyPI.  No approval is needed.

2. Check the upload::

       pip install -i https://test.pypi.org/simple/ gmcluster

   If something is wrong, fix it and repeat with ``0.3.0rc2``.

Release to PyPI
===============

1. Publish the release::

       dev_scripts/release.sh 0.3.0

   This stamps the version on ``prerelease``, fast-forwards ``main`` to the same
   commit, and creates the GitHub release tagged ``v0.3.0``.  Afterwards
   ``main``, ``prerelease``, and the ``v0.3.0`` tag are all on the same commit.
   GitHub Actions builds the package and then pauses for your approval.

2. Approve the upload on GitHub: open the **Actions** tab, click the running
   **Release** workflow, click **Review deployments**, check the **pypi** box,
   and click **Approve and deploy**.

3. Confirm it is live::

       pip install gmcluster

Notes
=====

- The tag is always ``v`` followed by the version; the workflow fails the build
  if the tag does not match ``__version__``.
- At a release, ``main`` only fast-forwards to ``prerelease``, so ``main``,
  ``prerelease``, and the release tag are always on the same commit afterwards.
- The version is single-sourced from ``gmcluster/__init__.py``; ``pyproject.toml``
  and the docs read it from there, so bump it in that one place.
