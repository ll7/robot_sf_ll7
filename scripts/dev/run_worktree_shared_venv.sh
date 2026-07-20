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
  --standalone           Run a command that is verified not to import project packages. This skips
                         the project-source freshness check and does not prepend the worktree root
                         to PYTHONPATH.
  --no-freshness-check   Retained for compatibility; checkout-local fast-pysf source already takes
                         precedence over any reused installed copy. Also accepted via
                         ROBOT_SF_VENV_FRESHNESS_CHECK=skip.
  -h, --help             Show this help message.

Examples:
  scripts/dev/run_worktree_shared_venv.sh -- pytest tests/test_ci_script_contract.py -q
  scripts/dev/run_worktree_shared_venv.sh --venv ../robot_sf_ll7/.venv -- ruff check scripts/dev
  scripts/dev/run_worktree_shared_venv.sh --standalone -- \
    python scripts/dev/check_docs_evidence_integrity.py --files docs/dev_guide.md

Use a full local .venv plus PR_READY_MODE=final for final PR proof; this helper is for quick,
targeted validation in sibling worktrees.
EOF
}

venv_override=""
skip_freshness=""
standalone=""
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

check_shared_venv_freshness() {
  # A checkout-local fast-pysf source package is authoritative because it is placed first on
  # PYTHONPATH below. Do not compare it with the reused installed copy: that copy may belong to the
  # owning checkout and can legitimately differ from this linked worktree.
  local venv="$1"
  local src_pkg="$repo_root/fast-pysf/pysocialforce"

  if [[ ! -d "$src_pkg" ]]; then
    return 0
  fi

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
