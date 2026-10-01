===================
Releasing a version
===================

GMCluster is published to PyPI by the GitHub Actions workflow in
``.github/workflows/release.yml``.  You drive it from your machine with
``dev_scripts/release.sh``: one command opens a pull request for you to review,
and after you merge it a second command tags the release and publishes it.
Uploads use Trusted Publishing, so no API token is ever stored or typed.

The examples below release version ``0.3.1``.  Replace ``0.3.1`` with the
version you are releasing.

Dry run on TestPyPI (optional, recommended the first time)
==========================================================

This proves the whole pipeline on a throwaway upload before the real one.  A
PyPI version number can never be reused, so it is worth doing once.

1. Publish a release candidate::

       dev_scripts/release.sh 0.3.1rc1

   This stamps the version, pushes ``prerelease``, and creates a GitHub
   pre-release tagged ``v0.3.1rc1``.  GitHub Actions builds the package and
   uploads it to TestPyPI.  No approval is needed.

2. Check the upload::

       pip install -i https://test.pypi.org/simple/ gmcluster

   If something is wrong, fix it and repeat with ``0.3.1rc2``.

Release to PyPI
===============

1. Open the release pull request::

       dev_scripts/release.sh 0.3.1

   This sets the version to ``0.3.1`` on ``prerelease``, pushes it, and opens a
   pull request from ``prerelease`` to ``main``.  Nothing is published yet.

2. Review the code and accept.  On GitHub, open the pull request and look at the
   diff.  The tests run on the pull request automatically, so their result is
   shown next to the **Merge** button.  If you are happy, click **Merge pull
   request**.  (If you are not, close it and keep working on ``prerelease``.)

3. Publish the release::

       dev_scripts/release.sh 0.3.1 --publish

   This fast-forwards ``prerelease`` up to ``main`` so the two branches are
   identical, updates your local ``main`` to match, and tags the shared commit
   ``v0.3.1``.  GitHub Actions then builds the package and publishes it to PyPI.

4. Confirm it is live::

       pip install gmcluster

Notes
=====

- The tag is always ``v`` followed by the version; the workflow fails the build
  if the tag does not match ``__version__``.
- After step 3, ``main``, ``prerelease``, and the ``v0.3.1`` tag are all on the
  same commit.  Tagging adds a label to that commit; it moves no branch, so the
  branches stay in sync and you can go on adding to ``prerelease``.
- The version is single-sourced from ``gmcluster/__init__.py``; ``pyproject.toml``
  and the docs read it from there, so bump it in that one place.
