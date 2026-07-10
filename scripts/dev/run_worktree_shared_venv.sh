#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: scripts/dev/run_worktree_shared_venv.sh [options] -- <uv-run-command> [args...]

Run a targeted validation command from the current checkout while reusing a shared virtualenv.
The helper pins imports to this worktree by prepending PYTHONPATH=$PWD and sets UV_NO_SYNC=1 so
`uv run` does not silently resync or rewrite the shared environment.
For linked worktrees, the helper also derives a per-worktree COVERAGE_FILE unless one is already
set, preventing parallel focused pytest runs from sharing output/coverage/.coverage state.

Because the shared env is reused without resync (UV_NO_SYNC=1), a stale owning-checkout .venv can
lag the current worktree source. The vendored `pysocialforce` package (force-included from
fast-pysf/pysocialforce and NOT shadowed by PYTHONPATH=$PWD) is especially drift-prone: importing
a newer API from a stale install fails mid-collection with a confusing ImportError. The helper
therefore runs a cheap freshness check comparing the installed `pysocialforce` package against this
checkout's fast-pysf/pysocialforce source and fails early with an actionable message when they
diverge. Refresh the owning checkout with
`uv sync --all-extras --reinstall-package robot-sf`; a plain sync does not rebuild this
force-included source.

Standalone commands with a verified boundary that does not import project packages can use
--standalone. That mode skips the project-source freshness check and does not add the worktree root
to PYTHONPATH, while still reusing the shared environment for third-party dependencies.

Options:
  --venv PATH            Shared virtualenv path exported as UV_PROJECT_ENVIRONMENT. Defaults to the
                         main checkout .venv for linked worktrees.
  --standalone           Run a command that is verified not to import project packages. This skips
                         the project-source freshness check and does not prepend the worktree root
                         to PYTHONPATH.
  --no-freshness-check   Skip the shared-venv freshness check. Use only when you have confirmed the
                         reused env matches this checkout (e.g. an editable install that already
                         points at this source). Also skippable via
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

venv_path="${venv_override:-$main_repo_root/.venv}"
if [[ "$venv_path" != /* ]]; then
  venv_path="$repo_root/$venv_path"
fi

if [[ ! -x "$venv_path/bin/python" ]]; then
  echo "Shared virtualenv not found or incomplete: $venv_path" >&2
  echo "Create it with 'uv sync --all-extras' in the owning checkout, or use a local .venv." >&2
  exit 2
fi

check_shared_venv_freshness() {
  # Guard the shared-venv fast path against a stale owning-checkout environment.
  #
  # The vendored `pysocialforce` package is force-included from fast-pysf/pysocialforce and is NOT
  # shadowed by PYTHONPATH=$PWD (the package lives at fast-pysf/pysocialforce, not at the repo
  # root). With UV_NO_SYNC=1 a reused env can lag the current worktree source, so importing a newer
  # API from a stale install fails mid-collection with a confusing ImportError. Compare the
  # installed package's .py files against this checkout's source and fail early on divergence.
  local venv="$1"
  local src_pkg="$repo_root/fast-pysf/pysocialforce"

  if [[ ! -d "$src_pkg" ]]; then
    return 0
  fi

  local installed_pkg=""
  local candidate
  for candidate in "$venv"/lib/python*/site-packages/pysocialforce; do
    if [[ -d "$candidate" ]]; then
      installed_pkg="$candidate"
      break
    fi
  done

  if [[ -z "$installed_pkg" ]]; then
    return 0
  fi

  local src_file rel_path installed_file
  while IFS= read -r -d '' src_file; do
    rel_path="${src_file#"$src_pkg"/}"
    installed_file="$installed_pkg/$rel_path"
    if [[ ! -f "$installed_file" ]] || ! cmp -s "$src_file" "$installed_file"; then
      cat >&2 <<EOF
Shared virtualenv is stale relative to this checkout: $venv
  diverging module: pysocialforce/$rel_path
The reused env (UV_NO_SYNC=1) lacks source present in fast-pysf/pysocialforce, so imports can
fail mid-run with a confusing ImportError. Refresh the owning checkout with
'uv sync --all-extras --reinstall-package robot-sf'; a plain sync does not rebuild this
force-included source; use --standalone for a command verified not to import project packages, or rerun with
--no-freshness-check (or ROBOT_SF_VENV_FRESHNESS_CHECK=skip) once you have confirmed the env matches
this checkout.
EOF
      return 1
    fi
  done < <(find "$src_pkg" -type f -name '*.py' -print0)
}

if [[ -z "$standalone" && -z "$skip_freshness" && "${ROBOT_SF_VENV_FRESHNESS_CHECK:-}" != "skip" ]]; then
  if ! check_shared_venv_freshness "$venv_path"; then
    exit 2
  fi
fi

export UV_PROJECT_ENVIRONMENT="$venv_path"
export UV_NO_SYNC=1
if [[ -z "$standalone" ]]; then
  export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
fi

if [[ -z "${COVERAGE_FILE:-}" && "$git_common_dir" != "$repo_root/.git" ]]; then
  worktree_id="$(printf '%s' "$repo_root" | git hash-object --stdin | cut -c1-12)"
  export COVERAGE_FILE="$repo_root/output/coverage/.coverage.${worktree_id}"
fi

exec uv run "${cmd[@]}"
