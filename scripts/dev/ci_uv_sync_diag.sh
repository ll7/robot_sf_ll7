#!/usr/bin/env bash
# Diagnostic probe for uv sync steps in CI.
#
# Prints runner, disk, uv cache, and virtual-environment state in a compact,
# GitHub-Actions-friendly grouped log. Safe to run before and after sync.
# All errors are advisory: the script exits 0 so a diagnostic step never
# masks a real CI failure.
#
# Usage: scripts/dev/ci_uv_sync_diag.sh [<label>]

set -uo pipefail

label="${1:-uv-sync-diag}"

# Bounded tree sizing (issue #8249): `du` walks scale with cache/venv size and
# shared-host I/O contention. With GNU timeout(1), each tree-size probe below
# is capped so this advisory diagnostic stays fast; on timeout it reports
# `unavailable-timed-out` instead of hanging the caller. Override the cap with
# ROBOT_SF_DIAG_DU_TIMEOUT_SECONDS (default 10). Hosts without GNU timeout(1)
# (e.g. macOS) keep the previous direct-`du` fallback behavior.
diag_du_timeout="${ROBOT_SF_DIAG_DU_TIMEOUT_SECONDS:-10}"
if ! [[ "$diag_du_timeout" =~ ^[1-9][0-9]*$ ]]; then
  diag_du_timeout=10
fi
# GNU timeout(1) is absent on some hosts (e.g. stock macOS bash 3.2 runners);
# without it keep the previous direct-`du` behavior. Detect GNU explicitly so
# a non-GNU command with the same name does not receive incompatible options.
# The function form (rather than an arg array) stays safe under `set -u` on old
# bash versions.
du_timeout_bin=""
if command -v timeout >/dev/null 2>&1 &&
  timeout --version 2>/dev/null | grep -q "GNU coreutils"; then
  du_timeout_bin="$(command -v timeout)"
fi
du_timeout_kill_after_secs=2
bounded_du() {
  if [[ -n "${du_timeout_bin:-}" ]]; then
    "$du_timeout_bin" --kill-after="${du_timeout_kill_after_secs}s" "$diag_du_timeout" "$@"
  else
    "$@"
  fi
}

du_timed_out() {
  local rc="$1"
  [[ -n "${du_timeout_bin:-}" && ( "$rc" -eq 124 || "$rc" -eq 137 ) ]]
}

report_du_failure() {
  local prefix="$1"
  local rc="$2"
  if du_timed_out "$rc"; then
    echo "  ${prefix}_sizing_status=timed-out"
    echo "  ${prefix}_sizing_timeout_seconds=${diag_du_timeout}"
  else
    echo "  ${prefix}_sizing_status=error"
  fi
  echo "  ${prefix}_sizing_exit_code=${rc}"
}

echo "::group::${label}"

echo "uv_sync_diag runner_info"
echo "  timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  nproc=$(nproc 2>/dev/null || echo unknown)"
echo "  uptime=$(uptime 2>/dev/null || echo unknown)"
if [[ -r /proc/meminfo ]]; then
    echo "  mem_available_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo unknown)"
fi

echo "uv_sync_diag uv_info"
if command -v uv >/dev/null 2>&1; then
    echo "  uv_version=$(uv --version 2>/dev/null || echo unknown)"
    echo "  uv_cache_dir=$(uv cache dir 2>/dev/null || echo unknown)"
else
    echo "  uv_version=not_installed"
fi

# Respect UV_CACHE_DIR if set; otherwise fall back to what uv reports, then ~/.cache/uv.
cache_dir=""
if [[ -n "${UV_CACHE_DIR:-}" ]]; then
    cache_dir="$UV_CACHE_DIR"
elif command -v uv >/dev/null 2>&1; then
    cache_dir="$(uv cache dir 2>/dev/null || true)"
fi
if [[ -z "${cache_dir:-}" ]]; then
    cache_dir="${HOME:-}/.cache/uv"
fi

echo "uv_sync_diag disk_info"
disk_target="."
if [[ -e "$cache_dir" ]]; then
    disk_target="$cache_dir"
fi
disk_info="$(df -Pk "$disk_target" 2>/dev/null | awk '
    NR == 2 {
        print "  filesystem_size_kb=" $2
        print "  filesystem_used_kb=" $3
        print "  filesystem_available_kb=" $4
        print "  filesystem_used_percent=" $5
    }
' || true)"
if [[ -n "$disk_info" ]]; then
    printf '%s\n' "$disk_info"
else
    echo "  filesystem_probe=unavailable"
fi

echo "uv_sync_diag cache_size"
if [[ -d "$cache_dir" ]]; then
    echo "  cache_dir=${cache_dir}"
    # Single-pass cache sizing (issue #3703): `du -h -d 1` walks the cache tree
    # once and emits the size of every immediate subdirectory plus the cache
    # total. The previous loop re-ran `du` per subdirectory, re-traversing the
    # tree up to a dozen times and risking preflight timeouts on large caches.
    # The captured output is then parsed in a single awk pass (pure in-memory,
    # no further disk I/O), preserving the curated key names and ordering.
    # The walk itself is capped by bounded_du (issue #8249). It is the sole
    # cache-size traversal; `uv cache size` is intentionally not called because
    # that full-cache operation cannot be deadline controlled here.
    cache_du=""
    cache_du_rc=0
    cache_du="$(bounded_du du -h -d 1 "$cache_dir" 2>/dev/null)" || cache_du_rc=$?
    if [[ "$cache_du_rc" -eq 0 ]]; then
        echo "  cache_sizing_status=ok"
        printf '%s\n' "$cache_du" | awk -F'\t' -v dir="$cache_dir" '
            { size[$2] = $1 }
            END {
                if (dir in size) print "  cache_total_size=" size[dir]
                n = split("archive-v0 wheels-v6 wheels-v5 sdists-v9 sdists-v8 simple-v21 simple-v20 builds-v0 environments-v2 environments-v1 interpreter-v4 git-v0", subs, " ")
                for (i = 1; i <= n; i++) {
                    p = dir "/" subs[i]
                    if (p in size) print "  cache_" subs[i] "_size=" size[p]
                }
            }
        '
    else
        report_du_failure "cache" "$cache_du_rc"
        if du_timed_out "$cache_du_rc"; then
            echo "  cache_total_size=unavailable-timed-out"
        else
            echo "  cache_total_size=unavailable-error"
        fi
    fi
else
    echo "  cache_dir=${cache_dir} (does not exist)"
fi

echo "uv_sync_diag venv_info"
if [[ -d .venv ]]; then
    venv_du_rc=0
    venv_du="$(bounded_du du -sh .venv 2>/dev/null)" || venv_du_rc=$?
    if [[ "$venv_du_rc" -eq 0 ]]; then
        echo "  venv_sizing_status=ok"
        printf '%s\n' "$venv_du" | awk '{print "  venv_size="$1}' || true
    else
        report_du_failure "venv" "$venv_du_rc"
        if du_timed_out "$venv_du_rc"; then
            echo "  venv_size=unavailable-timed-out"
        else
            echo "  venv_size=unavailable-error"
        fi
    fi
    if [[ -x .venv/bin/python ]]; then
        echo "  python_version=$(.venv/bin/python --version 2>&1 || true)"
    else
        echo "  python_version=binary_missing"
    fi
else
    echo "  venv_present=false"
fi

echo "::endgroup::"

exit 0
