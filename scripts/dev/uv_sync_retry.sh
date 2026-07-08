#!/usr/bin/env bash
# Run ``uv sync`` with bounded retry + exponential backoff to absorb transient
# PyPI / index failures (issue #4889: wheel-smoke-install died mid-run with a
# connection error downloading ``nvidia-cufft-cu12==11.3.3.83``).
#
# ``uv sync --frozen`` resolves against a fixed lock, so the only realistic
# non-zero exits are network/download failures, which are transient. uv caches
# every wheel it fully downloads, so a retry resumes from the last good wheel
# rather than re-fetching the whole dependency set. A genuine (non-transient)
# failure reproduces on every attempt and still fails the step once the retry
# budget is exhausted. Status 127 (``uv`` not on PATH) is treated as
# non-transient and fails immediately without burning retries.
#
# Configuration (env):
#   UV_SYNC_MAX_ATTEMPTS  total attempts before giving up (default 3)
#   UV_SYNC_BACKOFF_BASE  initial backoff in seconds (default 5)
#   UV_SYNC_BACKOFF_CAP   maximum backoff in seconds (default 30)
#
# Usage:
#   scripts/dev/uv_sync_retry.sh -- <uv sync args...>
#   scripts/dev/uv_sync_retry.sh            # defaults to --all-extras --frozen
#
# The leading ``--`` separator is optional; all args are forwarded to ``uv sync``.

set -euo pipefail

max_attempts="${UV_SYNC_MAX_ATTEMPTS:-3}"
backoff_base="${UV_SYNC_BACKOFF_BASE:-5}"
backoff_cap="${UV_SYNC_BACKOFF_CAP:-30}"

if ! [[ "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "uv_sync_retry: UV_SYNC_MAX_ATTEMPTS='${max_attempts}' is not a positive integer; defaulting to 3" >&2
  max_attempts=3
fi

# Validate the backoff knobs the same way: a non-numeric value (e.g. ``5s``)
# would otherwise be treated as an unbound variable in the arithmetic
# expansion below and abort the wrapper under ``set -u`` with a confusing
# error. Fall back to the documented defaults instead.
if ! [[ "$backoff_base" =~ ^[0-9]+$ ]]; then
  echo "uv_sync_retry: UV_SYNC_BACKOFF_BASE='${backoff_base}' is not a non-negative integer; defaulting to 5" >&2
  backoff_base=5
fi

if ! [[ "$backoff_cap" =~ ^[0-9]+$ ]]; then
  echo "uv_sync_retry: UV_SYNC_BACKOFF_CAP='${backoff_cap}' is not a non-negative integer; defaulting to 30" >&2
  backoff_cap=30
fi

# Drop an optional leading ``--`` separator so callers can write the wrapper as
# ``uv_sync_retry.sh -- --all-extras --frozen`` for unambiguous arg forwarding.
if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ $# -eq 0 ]]; then
  set -- --all-extras --frozen
fi

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "::group::uv sync (attempt ${attempt}/${max_attempts})"
  set +e
  uv sync "$@"
  status=$?
  set -e
  echo "::endgroup::"

  if [[ "$status" -eq 0 ]]; then
    echo "uv_sync_retry success attempt=${attempt}/${max_attempts}"
    exit 0
  fi

  if [[ "$status" -eq 127 ]]; then
    echo "::error::uv not found on PATH (status 127); not retryable" >&2
    exit 127
  fi

  if [[ "$attempt" -ge "$max_attempts" ]]; then
    echo "::error::uv sync failed after ${attempt} attempt(s) (last status=${status})" >&2
    exit "$status"
  fi

  # Exponential backoff capped at UV_SYNC_BACKOFF_CAP.
  delay=$(( backoff_base * (2 ** (attempt - 1)) ))
  if [[ "$delay" -gt "$backoff_cap" ]]; then
    delay="$backoff_cap"
  fi
  echo "uv_sync_retry transient status=${status} retry_in=${delay}s next_attempt=$((attempt + 1))/${max_attempts}"
  sleep "$delay"
done
