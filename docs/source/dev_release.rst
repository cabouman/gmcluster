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

One-time setup
==============

This setup is done once per package, when it is first published, and never
again for that package.  For gmcluster the GitHub CLI is already logged in and
the two environments already exist, so only the pending publishers remain.

- **GitHub CLI login** (already done on this host): ``gh auth status`` shows you
  logged in.  Run ``gh auth login`` only if it does not.

- **GitHub environments** (already created for gmcluster): in the repository
  settings under **Environments** there is a ``pypi`` environment with you set
  as a required reviewer, so the upload waits for your approval, and a
  ``testpypi`` environment with no reviewer.

- **Pending publishers** (already added for gmcluster; needed once per new
  package): on **PyPI** and on **TestPyPI**, a pending publisher for the project
  with these values:

  - Project name: ``gmcluster``
  - Owner: ``cabouman``
  - Repository: ``gmcluster``
  - Workflow: ``release.yml``
  - Environment: ``pypi`` on PyPI, ``testpypi`` on TestPyPI

Trusted Publishing means no API tokens are stored anywhere.

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

Do these four steps in order.  Steps 1 and 3 are commands on your machine;
steps 2 and 4 are clicks on GitHub.

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

4. Approve the upload on GitHub: open the **Actions** tab, click the running
   **Release** workflow, click **Review deployments**, check the **pypi** box,
   and click **Approve and deploy**.

5. Confirm it is live::

       pip install gmcluster

Notes
=====

- The tag is always ``v`` followed by the version; the workflow fails the build
  if the tag does not match ``__version__``.
- ``main`` changes only through the pull request above, never a direct push.
- The version is single-sourced from ``gmcluster/__init__.py``; ``pyproject.toml``
  and the docs read it from there, so bump it in that one place.
