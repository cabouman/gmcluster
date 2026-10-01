#!/bin/bash
# Run the release procedure.
#
#   dev_scripts/release.sh 0.3.0rc1     # dry run: publish a pre-release to TestPyPI
#   dev_scripts/release.sh 0.3.0        # release: fast-forward main, publish to PyPI
#
# Requires the gh CLI, logged in.  GitHub Actions publishes to PyPI with no
# manual approval step.
set -euo pipefail

cd "$(dirname "$0")/.."
VERSION="${1:?usage: release.sh X.Y.Z[rcN]}"
INIT=gmcluster/__init__.py

case "$VERSION" in
  *rc*) STAGE=rc ;;
  *)    STAGE=final ;;
esac

# Stamp the version on prerelease, commit, and push.
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
  exit 0
fi

# Final release: fast-forward main to prerelease (no merge commit), then tag.
git push -q origin prerelease:main
git branch -f main prerelease        # local main to the same commit
gh release create "v$VERSION" --target main --title "GMCluster v$VERSION" \
  --generate-notes
echo "Release v$VERSION created.  local and remote main, prerelease, and the"
echo "v$VERSION tag are all on the same commit.  GitHub Actions is publishing"
echo "to PyPI; check in a minute with:  pip install gmcluster"
