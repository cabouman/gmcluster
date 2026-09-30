#!/bin/bash
# Run one stage of the release procedure.
#
#   dev_scripts/release.sh 0.3.0rc1           # rc:   publish a pre-release to TestPyPI
#   dev_scripts/release.sh 0.3.0              # final: open the pull request to main
#   dev_scripts/release.sh 0.3.0 --publish    # after main advances: publish to PyPI
#
# Requires the gh CLI, logged in.  The PyPI upload still needs approval of the
# pypi environment on the workflow run page.
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

# --publish: tag main (which must already carry this version) and let CI publish.
if [[ "$PUBLISH" == "--publish" ]]; then
  git fetch -q origin main
  if ! git show origin/main:$INIT | grep -q "__version__ = '$VERSION'"; then
    echo "main does not have __version__ = '$VERSION'; merge the pull request first" >&2
    exit 1
  fi
  gh release create "v$VERSION" --target main --title "GMCluster v$VERSION" \
    --generate-notes
  echo "Release v$VERSION created.  Approve the pypi environment on the"
  echo "workflow run page (Actions -> the Release run -> Review deployments),"
  echo "then check with:  pip install gmcluster"
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
  # main changes only through a pull request, merged on GitHub.
  if gh pr list --base main --head prerelease --state open --json number -q '.[0].number' | grep -q .; then
    echo "The pull request from prerelease to main is already open and now carries $VERSION."
  else
    gh pr create --base main --head prerelease --title "GMCluster v$VERSION" \
      --body "Release v$VERSION."
  fi
  echo "When the checks pass, merge the pull request on GitHub.  Then run:"
  echo "  dev_scripts/release.sh $VERSION --publish"
fi
