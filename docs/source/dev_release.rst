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

Do these steps in order.

1. Open the release pull request::

       dev_scripts/release.sh 0.3.0

   This stamps the version, pushes ``prerelease``, and opens the pull request
   from ``prerelease`` to ``main``.  Nothing is uploaded yet.

2. On GitHub, wait for the checks to pass, then **merge the pull request** into
   ``main``.

3. Publish the release::

       dev_scripts/release.sh 0.3.0 --publish

   This confirms that ``main`` carries ``__version__ = '0.3.0'`` (it stops if
   the pull request is not merged yet), then creates the GitHub release tagged
   ``v0.3.0`` on ``main``.  GitHub Actions builds the package and then pauses
   for your approval.

   Approve the upload on GitHub: open the **Actions** tab, click the running
   **Release** workflow, click **Review deployments**, check the **pypi** box,
   and click **Approve and deploy**.

4. Confirm it is live::

       pip install gmcluster

Notes
=====

- The tag is always ``v`` followed by the version; the workflow fails the build
  if the tag does not match ``__version__``.
- ``main`` changes only through the pull request above, never a direct push.
- The version is single-sourced from ``gmcluster/__init__.py``; ``pyproject.toml``
  and the docs read it from there, so bump it in that one place.
