#!/usr/bin/env bash
# Install an exact uv version through the Python package index with bounded
# retry/backoff.  This is the transport fallback for setup-uv action failures
# (issue #7368), not a way to accept an unpinned or partially installed uv.
#
# The primary CI path remains the pinned astral-sh/setup-uv action because it
# owns the pruned uv cache.  This helper is used when that action cannot be
# downloaded from GitHub's codeload service, and by the small workflows that do
# not need the shared composite action.  PyPI publishes platform wheels for the
# supported GitHub-hosted Linux and macOS runners; --only-binary keeps a missing
# wheel fail-closed instead of silently building an alternate artifact.
#
# Configuration (env):
#   UV_INSTALL_MAX_ATTEMPTS  total pip attempts (default 3)
#   UV_INSTALL_BACKOFF_BASE  initial retry delay in seconds (default 5)
#   UV_INSTALL_BACKOFF_CAP   maximum retry delay in seconds (default 30)
#   UV_INSTALL_BIN_DIR       optional user-bin override, primarily for tests
#
# Usage:
#   scripts/dev/ci_install_uv_retry.sh <exact uv version>

set -euo pipefail

uv_version="${1:-${UV_VERSION:-}}"
if [[ -z "$uv_version" ]]; then
  echo "ci_install_uv_retry: an exact uv version is required" >&2
  exit 2
fi

if ! [[ "$uv_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; then
  echo "ci_install_uv_retry: invalid uv version: ${uv_version}" >&2
  exit 2
fi

max_attempts="${UV_INSTALL_MAX_ATTEMPTS:-3}"
backoff_base="${UV_INSTALL_BACKOFF_BASE:-5}"
backoff_cap="${UV_INSTALL_BACKOFF_CAP:-30}"

if ! [[ "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "ci_install_uv_retry: UV_INSTALL_MAX_ATTEMPTS='${max_attempts}' is not a positive integer; defaulting to 3" >&2
  max_attempts=3
fi

if ! [[ "$backoff_base" =~ ^[0-9]+$ ]]; then
  echo "ci_install_uv_retry: UV_INSTALL_BACKOFF_BASE='${backoff_base}' is not a non-negative integer; defaulting to 5" >&2
  backoff_base=5
fi

if ! [[ "$backoff_cap" =~ ^[0-9]+$ ]]; then
  echo "ci_install_uv_retry: UV_INSTALL_BACKOFF_CAP='${backoff_cap}' is not a non-negative integer; defaulting to 30" >&2
  backoff_cap=30
fi

if ! command -v python >/dev/null 2>&1; then
  echo "::error::ci_install_uv_retry: python is not on PATH; cannot install pinned uv" >&2
  exit 127
fi

if [[ -n "${UV_INSTALL_BIN_DIR:-}" ]]; then
  install_bin_dir="$UV_INSTALL_BIN_DIR"
else
  user_base="$(python -c 'import site; print(site.USER_BASE)')"
  install_bin_dir="${user_base}/bin"
fi

if [[ -z "$install_bin_dir" ]]; then
  echo "::error::ci_install_uv_retry: could not determine the user script directory" >&2
  exit 1
fi

mkdir -p "$install_bin_dir"
export PATH="$install_bin_dir:$PATH"
if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$install_bin_dir" >>"$GITHUB_PATH"
fi

expected="uv ${uv_version}"
uv_version_matches() {
  local actual="$1"
  [[ "$actual" == uv\ * ]] || return 1
  local actual_version="${actual#uv }"
  actual_version="${actual_version%% *}"
  [[ "$actual_version" == "$uv_version" ]]
}

if command -v uv >/dev/null 2>&1 && uv_version_matches "$(uv --version 2>/dev/null || true)"; then
  echo "ci_install_uv_retry reuse source=path version=${uv_version}"
  exit 0
fi

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "::group::Install pinned uv (attempt ${attempt}/${max_attempts})"
  set +e
  python -m pip install \
    --disable-pip-version-check \
    --no-input \
    --no-deps \
    --only-binary=:all: \
    --retries 1 \
    --timeout 60 \
    --user \
    --upgrade \
    "uv==${uv_version}"
  status=$?
  set -e
  echo "::endgroup::"

  if [[ "$status" -eq 0 ]]; then
    hash -r
    actual="$(uv --version 2>/dev/null || true)"
    if uv_version_matches "$actual"; then
      echo "ci_install_uv_retry success source=pypi version=${uv_version} attempt=${attempt}/${max_attempts}"
      exit 0
    fi
    echo "::error::ci_install_uv_retry installed '${actual:-missing uv}', expected '${expected}'" >&2
    exit 1
  fi

  if [[ "$attempt" -ge "$max_attempts" ]]; then
    echo "::error::ci_install_uv_retry failed after ${attempt} attempt(s) (last status=${status})" >&2
    exit "$status"
  fi

  delay=$((backoff_base * (2 ** (attempt - 1))))
  if [[ "$delay" -gt "$backoff_cap" ]]; then
    delay="$backoff_cap"
  fi
  echo "ci_install_uv_retry transient status=${status} retry_in=${delay}s next_attempt=$((attempt + 1))/${max_attempts}"
  sleep "$delay"
done
