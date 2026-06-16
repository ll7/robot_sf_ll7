#!/usr/bin/env bash
set -euo pipefail

if [[ "${GITHUB_ACTIONS:-}" != "true" ]]; then
  echo "Not running in GitHub Actions; skipping hosted-runner disk cleanup."
  exit 0
fi

echo "Disk usage before cleanup:"
df -h /

sudo rm -rf \
  /opt/ghc \
  /opt/hostedtoolcache/CodeQL \
  /usr/local/.ghcup \
  /usr/local/julia* \
  /usr/local/lib/android \
  /usr/local/share/boost \
  /usr/local/share/chromium \
  /usr/share/dotnet \
  /usr/share/swift || true

docker image prune -af || true

echo "Disk usage after cleanup:"
df -h /
