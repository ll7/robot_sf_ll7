#!/usr/bin/env bash
# Diagnostic probe for uv sync steps in CI.
#
# Prints runner, uv cache, and virtual-environment state in a compact,
# GitHub-Actions-friendly grouped log. Safe to run before and after sync.
# All errors are advisory: the script exits 0 so a diagnostic step never
# masks a real CI failure.
#
# Usage: scripts/dev/ci_uv_sync_diag.sh [<label>]

set -uo pipefail

label="${1:-uv-sync-diag}"

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
    # Best-effort cache size report; older uv versions may not have 'uv cache size'.
    uv_cache_size="$(uv cache size 2>/dev/null || true)"
    if [[ -n "$uv_cache_size" ]]; then
        echo "  uv_cache_size=${uv_cache_size}"
    fi
else
    echo "  uv_version=not_installed"
fi

echo "uv_sync_diag cache_size"
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

if [[ -d "$cache_dir" ]]; then
    echo "  cache_dir=${cache_dir}"
    du -sh "$cache_dir" 2>/dev/null | awk '{print "  cache_total_size="$1}' || true
    for sub in archive-v0 wheels-v6 wheels-v5 sdists-v9 sdists-v8 simple-v21 simple-v20 builds-v0 environments-v2 environments-v1 interpreter-v4 git-v0; do
        if [[ -d "$cache_dir/$sub" ]]; then
            du -sh "$cache_dir/$sub" 2>/dev/null | awk -v s="$sub" '{print "  cache_"s"_size="$1}' || true
        fi
    done
else
    echo "  cache_dir=${cache_dir} (does not exist)"
fi

echo "uv_sync_diag venv_info"
if [[ -d .venv ]]; then
    du -sh .venv 2>/dev/null | awk '{print "  venv_size="$1}' || true
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
