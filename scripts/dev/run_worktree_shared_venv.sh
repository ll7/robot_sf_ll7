#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: scripts/dev/run_worktree_shared_venv.sh [options] -- <uv-run-command> [args...]

Run a targeted validation command from the current checkout while reusing a shared virtualenv.
The helper pins imports to this worktree by prepending PYTHONPATH=$PWD:$PWD/fast-pysf and sets UV_NO_SYNC=1 so
`uv run` does not silently resync or rewrite the shared environment.
For linked worktrees, the helper also derives a per-worktree COVERAGE_FILE unless one is already
set, preventing parallel focused pytest runs from sharing output/coverage/.coverage state.

Because the shared env is reused without resync (UV_NO_SYNC=1), a stale owning-checkout .venv can
lag the current worktree source. The vendored `pysocialforce` package is shadowed by
PYTHONPATH=$PWD:$PWD/fast-pysf, so an initialized checkout source is authoritative and must not be
rejected because the reused installed copy differs. If the source package is unavailable, the
helper leaves the installed-environment decision to the command that imports it.

Standalone commands with a verified boundary that does not import project packages can use
--standalone. That mode skips the project-source freshness check and does not add the worktree root
to PYTHONPATH, while still reusing the shared environment for third-party dependencies.

Options:
  --venv PATH            Shared virtualenv path exported as UV_PROJECT_ENVIRONMENT. Defaults to an
                         initialized current-worktree .venv, otherwise the main checkout .venv.
  --profile NAME         Dependency import profile checked before the command. Defaults to core;
                         use all-extras (or a named pyproject extra) when the command needs it.
  --scratch-dir PATH     Use PATH for temporary files and default uv/XDG caches for this run.
  --standalone           Run a command that is verified not to import project packages. This skips
                         the dependency-profile and project-source checks and does not prepend the
                         worktree root to PYTHONPATH.
  --no-freshness-check   Retained for compatibility; checkout-local fast-pysf source already takes
                         precedence over any reused installed copy. Also accepted via
                         ROBOT_SF_VENV_FRESHNESS_CHECK=skip.
  -h, --help             Show this help message.

Environment:
  ROBOT_SF_CI_MIN_FREE_BYTES
                         Minimum free bytes required for the effective temporary directory
                         (default: 1073741824).

Examples:
  scripts/dev/run_worktree_shared_venv.sh -- pytest tests/test_ci_script_contract.py -q
  scripts/dev/run_worktree_shared_venv.sh --venv ../robot_sf_ll7/.venv -- ruff check scripts/dev
  scripts/dev/run_worktree_shared_venv.sh --standalone -- \
    python scripts/dev/check_docs_evidence_integrity.py --files docs/dev_guide.md

Use a full local .venv plus PR_READY_MODE=final for final PR proof; this helper is for quick,
targeted validation in sibling worktrees.
EOF
}

check_scratch_capacity() {
  local space_path="$1"
  local minimum_bytes="${ROBOT_SF_CI_MIN_FREE_BYTES:-1073741824}"
  local minimum_kib
  local df_output
  local available_kib

  if ! [[ "$minimum_bytes" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ROBOT_SF_CI_MIN_FREE_BYTES must be a non-negative integer (got '$minimum_bytes')." >&2
    return 2
  fi
  minimum_kib=$(( (10#$minimum_bytes + 1023) / 1024 ))

  if [[ ! -d "$space_path" || ! -w "$space_path" ]]; then
    echo "ERROR: shared-venv scratch path is missing or not writable: $space_path" >&2
    echo "Use --scratch-dir /path/on/disk or set TMPDIR to a writable disk-backed directory." >&2
    return 2
  fi
  if ! command -v df >/dev/null 2>&1 || ! command -v awk >/dev/null 2>&1; then
    echo "ERROR: shared-venv scratch preflight requires both 'df' and 'awk'." >&2
    return 2
  fi
  if ! df_output="$(df -Pk "$space_path" 2>/dev/null)"; then
    echo "ERROR: shared-venv scratch preflight could not inspect filesystem capacity: $space_path" >&2
    echo "Use --scratch-dir /path/on/disk and retry." >&2
    return 2
  fi
  available_kib="$(awk 'NR > 1 && $4 ~ /^[0-9]+$/ { print $4; exit }' <<<"$df_output")"
  if ! [[ "$available_kib" =~ ^[0-9]+$ ]]; then
    echo "ERROR: shared-venv scratch preflight could not parse available space for: $space_path" >&2
    echo "Use --scratch-dir /path/on/disk and retry." >&2
    return 2
  fi
  if (( 10#$available_kib < minimum_kib )); then
    echo "ERROR: shared-venv scratch preflight failed: ${available_kib} KiB available at $space_path; ${minimum_kib} KiB required." >&2
    echo "The uv command was not started. Free space or retry with --scratch-dir /path/on/disk." >&2
    echo "For a deliberately bounded run only, lower ROBOT_SF_CI_MIN_FREE_BYTES explicitly." >&2
    return 2
  fi
  echo "Shared-venv scratch preflight passed: path=$space_path available=${available_kib}KiB required=${minimum_kib}KiB" >&2
}

configure_scratch_dir() {
  local requested_path="$1"
  local scratch_root

  if ! mkdir -p "$requested_path"; then
    echo "ERROR: could not create shared-venv scratch directory: $requested_path" >&2
    echo "Choose a writable disk-backed path and retry." >&2
    return 2
  fi
  if ! scratch_root="$(cd "$requested_path" 2>/dev/null && pwd -P)"; then
    echo "ERROR: could not resolve shared-venv scratch directory: $requested_path" >&2
    return 2
  fi
  if ! mkdir -p "$scratch_root/tmp" "$scratch_root/uv-cache" "$scratch_root/xdg-cache" "$scratch_root/mplconfig"; then
    echo "ERROR: could not create shared-venv scratch subdirectories under: $scratch_root" >&2
    echo "Choose a writable disk-backed path and retry." >&2
    return 2
  fi

  export ROBOT_SF_CI_SCRATCH_DIR="$scratch_root"
  export TMPDIR="$scratch_root/tmp"
  export UV_CACHE_DIR="$scratch_root/uv-cache"
  export XDG_CACHE_HOME="$scratch_root/xdg-cache"
  export MPLCONFIGDIR="$scratch_root/mplconfig"
  echo "Using shared-venv scratch directory: $scratch_root" >&2
}

venv_override=""
dependency_profile="core"
skip_freshness=""
standalone=""
scratch_dir="${ROBOT_SF_CI_SCRATCH_DIR:-}"
cmd=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      if [[ $# -lt 2 ]]; then
        echo "--venv requires a path." >&2
        exit 2
      fi
      venv_override="$2"
      shift 2
      ;;
    --profile)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "--profile requires a dependency profile name." >&2
        exit 2
      fi
      dependency_profile="$2"
      shift 2
      ;;
    --scratch-dir)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "--scratch-dir requires a path." >&2
        exit 2
      fi
      scratch_dir="$2"
      shift 2
      ;;
    --standalone)
      standalone=1
      shift
      ;;
    --no-freshness-check)
      skip_freshness=1
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    --)
      shift
      cmd=("$@")
      break
      ;;
    *)
      cmd=("$@")
      break
      ;;
  esac
done

if [[ ${#cmd[@]} -eq 0 ]]; then
  show_help >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

git_common_dir="$(git rev-parse --git-common-dir)"
if [[ "$git_common_dir" != /* ]]; then
  git_common_dir="$(cd "$repo_root/$git_common_dir" && pwd)"
fi
main_repo_root="$(cd "$git_common_dir/.." && pwd)"

if [[ -n "$scratch_dir" ]]; then
  configure_scratch_dir "$scratch_dir"
fi
check_scratch_capacity "${TMPDIR:-/tmp}"

if [[ -n "$venv_override" ]]; then
  venv_path="$venv_override"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  venv_path="$repo_root/.venv"
else
  venv_path="$main_repo_root/.venv"
fi
if [[ "$venv_path" != /* ]]; then
  venv_path="$repo_root/$venv_path"
fi

if [[ ! -x "$venv_path/bin/python" ]]; then
  echo "Shared virtualenv not found or incomplete: $venv_path" >&2
  echo "Create it with 'uv sync --all-extras' in the owning checkout, or use a local .venv." >&2
  exit 2
fi

check_dependency_profile() {
  local report
  if ! report="$("$venv_path/bin/python" \
    "$repo_root/scripts/dev/check_worktree_optional_deps.py" \
    --profile "$dependency_profile" 2>&1)"; then
    echo "ERROR: shared-venv dependency profile '$dependency_profile' is incomplete in $venv_path." >&2
    printf '%s\n' "$report" >&2
    echo "Run 'cd $repo_root && scripts/dev/bootstrap_worktree.sh', then rerun this command." >&2
    return 2
  fi
}

if [[ -z "$standalone" ]]; then
  check_dependency_profile
fi

check_shared_venv_freshness() {
  local src_pkg="$repo_root/fast-pysf/pysocialforce"

  # PYTHONPATH makes the checkout source authoritative; an owning checkout's
  # installed copy is intentionally not a freshness boundary for this helper.
  return 0
}

if [[ -z "$standalone" && -z "$skip_freshness" && "${ROBOT_SF_VENV_FRESHNESS_CHECK:-}" != "skip" ]]; then
  if ! check_shared_venv_freshness "$venv_path"; then
    exit 2
  fi
fi

export UV_PROJECT_ENVIRONMENT="$venv_path"
export UV_NO_SYNC=1
if [[ -z "$standalone" ]]; then
  export PYTHONPATH="$repo_root:$repo_root/fast-pysf${PYTHONPATH:+:$PYTHONPATH}"
fi

if [[ -z "${COVERAGE_FILE:-}" && "$git_common_dir" != "$repo_root/.git" ]]; then
  worktree_id="$(printf '%s' "$repo_root" | git hash-object --stdin | cut -c1-12)"
  export COVERAGE_FILE="$repo_root/output/coverage/.coverage.${worktree_id}"
fi

exec uv run "${cmd[@]}"
