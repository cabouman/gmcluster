===================
Releasing a version
===================

GMCluster is published to PyPI by the release workflow in
``.github/workflows/release.yml``, driven one stage at a time by
``dev_scripts/release.sh``.  This is for maintainers.  It needs the ``gh`` CLI
logged in, and a one-time setup of Trusted Publishing.  The examples release
version ``0.X.Y``.

This setup is done once per package, when it is first published, and never
again for that package.  For gmcluster the GitHub CLI is already logged in and
the two environments already exist, so only the pending publishers remain.

- **GitHub CLI login** (already done on this host): ``gh auth status`` shows you
  logged in.  Run ``gh auth login`` only if it does not.

- **GitHub environments** (already created for gmcluster): in the repository
  settings under **Environments** there is a ``pypi`` environment with you set
  as a required reviewer, so the upload waits for your approval, and a
  ``testpypi`` environment with no reviewer.

- **Pending publishers** (the remaining step; do it once on each site): log in
  to **PyPI** and to **TestPyPI** and add a pending publisher for the project
  with these values:

  - Project name: ``gmcluster``
  - Owner: ``cabouman``
  - Repository: ``gmcluster``
  - Workflow: ``release.yml``
  - Environment: ``pypi`` on PyPI, ``testpypi`` on TestPyPI

Trusted Publishing means no API tokens are stored anywhere.

Dry run on TestPyPI (optional)
==============================

1. Publish a release candidate::

       dev_scripts/release.sh 0.X.Yrc1

   This sets ``__version__`` to ``0.X.Yrc1``, stamps ``CITATION.cff``, commits
   and pushes ``prerelease``, and creates a GitHub pre-release tagged
   ``v0.X.Yrc1``.  CI builds the package and uploads it to TestPyPI; no approval
   is needed.

2. Check the upload::

       pip install -i https://test.pypi.org/simple/ gmcluster

   If something is wrong, fix it and repeat with ``0.X.Yrc2``.

Release to PyPI
===============

1. Open the release pull request::

       dev_scripts/release.sh 0.X.Y

   This sets ``__version__`` to ``0.X.Y``, stamps ``CITATION.cff``, commits and
   pushes ``prerelease``, and opens the pull request from ``prerelease`` to
   ``main``.  Nothing is uploaded.  When the checks pass on GitHub, merge the
   pull request into ``main``.

2. Publish the release::

       dev_scripts/release.sh 0.X.Y --publish

   This checks that ``main`` carries ``__version__ = '0.X.Y'`` (it stops if the
   pull request is not merged yet), then creates a GitHub release tagged
   ``v0.X.Y`` on ``main``.  CI builds the package, then pauses for your approval.

   Approve the deployment: on GitHub, open the **Actions** tab, click the
   running **Release** workflow, click **Review deployments**, check the
   **pypi** box, and click **Approve and deploy**.

3. Check the upload::

       pip install gmcluster

Notes
=====

- The tag is always ``v`` followed by the version; the workflow fails the build
  if the tag does not match ``__version__``.
- ``main`` changes only through the pull request above, never a direct push.
- The version is single-sourced from ``gmcluster/__init__.py``; ``pyproject.toml``
  and the docs read it from there, so bump it in that one place.
