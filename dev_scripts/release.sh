#!/bin/bash
# Run one stage of the release procedure.
#
#   dev_scripts/release.sh 0.3.1rc1        # dry run: publish a pre-release to TestPyPI
#   dev_scripts/release.sh 0.3.1           # open the pull request from prerelease to main
#   dev_scripts/release.sh 0.3.1 --publish # after you merge the PR: sync, tag, publish to PyPI
#
# Requires the gh CLI, logged in.  GitHub Actions publishes to PyPI with no
# manual approval step.
set -euo pipefail

cd "$(dirname "$0")/.."
VERSION="${1:?usage: release.sh X.Y.Z[rcN] [--publish]}"
PUBLISH="${2:-}"
INIT=gmcluster/__init__.py

case "$VERSION" in
  *rc*) STAGE=rc ;;
  *)    STAGE=final ;;
esac
if [[ "$PUBLISH" == "--publish" && "$STAGE" == "rc" ]]; then
  echo "--publish is for a final version; an rc publishes on its own" >&2
  exit 2
fi

# --publish: the PR is merged, so main carries this version.  Bring prerelease
# up to main, then tag the shared commit and let CI publish.
if [[ "$PUBLISH" == "--publish" ]]; then
  git fetch -q origin main
  if ! git show origin/main:$INIT | grep -q "__version__ = '$VERSION'"; then
    echo "main does not have __version__ = '$VERSION'; merge the pull request first" >&2
    exit 1
  fi
  # Fast-forward prerelease to main so the two branches are identical.
  git checkout -q prerelease
  git pull -q origin prerelease
  if ! git merge -q --ff-only origin/main; then
    echo "could not fast-forward prerelease to main; merge the PR with the" >&2
    echo "default 'Create a merge commit' option, then run --publish again" >&2
    exit 1
  fi
  git push -q origin prerelease
  git branch -f main origin/main        # local main to the same commit
  # Tag the shared commit.  A tag is a label; it moves no branch.
  gh release create "v$VERSION" --target main --title "GMCluster v$VERSION" \
    --generate-notes
  echo "Release v$VERSION created.  main, prerelease, and the v$VERSION tag are"
  echo "all on the same commit.  GitHub Actions is publishing to PyPI; check in a"
  echo "minute with:  pip install gmcluster"
  exit 0
fi

# rc or final: stamp the version on prerelease, commit, and push.
git checkout -q prerelease
git pull -q origin prerelease
sed -i '' "s/^__version__ = '.*'/__version__ = '$VERSION'/" $INIT
grep -q "__version__ = '$VERSION'" $INIT
sed -i '' "s/^version:.*/version: $VERSION/" CITATION.cff
sed -i '' "s/^date-released:.*/date-released: $(date +%F)/" CITATION.cff
git add $INIT CITATION.cff
if git diff --cached --quiet; then
  echo "version is already $VERSION; nothing to commit"
else
  git commit -q -m "Set version to $VERSION"
fi
git push -q origin prerelease

if [[ "$STAGE" == "rc" ]]; then
  # A pre-release tag on prerelease; CI uploads it to TestPyPI, no approval.
  gh release create "v$VERSION" --target prerelease --prerelease \
    --title "GMCluster v$VERSION" --generate-notes
  echo "Pre-release v$VERSION created; the TestPyPI upload is running."
  echo "Check with:  pip install -i https://test.pypi.org/simple/ gmcluster"
else
  # Open the pull request from prerelease to main; you review it and merge it.
  if gh pr list --base main --head prerelease --state open --json number -q '.[0].number' | grep -q .; then
    echo "The pull request from prerelease to main is already open and now carries $VERSION."
  else
    gh pr create --base main --head prerelease --title "GMCluster v$VERSION" \
      --body "Release v$VERSION."
  fi
  echo "Review the pull request on GitHub (the tests run on it automatically)."
  echo "When you are happy, merge it.  Then run:"
  echo "  dev_scripts/release.sh $VERSION --publish"
fi
