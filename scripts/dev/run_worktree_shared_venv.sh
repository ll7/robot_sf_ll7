#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: scripts/dev/run_worktree_shared_venv.sh [options] -- <uv-run-command> [args...]

Run a targeted validation command from the current checkout while reusing a shared virtualenv.
The helper pins imports to this worktree by prepending PYTHONPATH=$PWD and sets UV_NO_SYNC=1 so
`uv run` does not silently resync or rewrite the shared environment.

Options:
  --venv PATH   Shared virtualenv path exported as UV_PROJECT_ENVIRONMENT. Defaults to the main
                checkout .venv for linked worktrees.
  -h, --help    Show this help message.

Examples:
  scripts/dev/run_worktree_shared_venv.sh -- pytest tests/test_ci_script_contract.py -q
  scripts/dev/run_worktree_shared_venv.sh --venv ../robot_sf_ll7/.venv -- ruff check scripts/dev

Use a full local .venv plus PR_READY_MODE=final for final PR proof; this helper is for quick,
targeted validation in sibling worktrees.
EOF
}

venv_override=""
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

export UV_PROJECT_ENVIRONMENT="$venv_path"
export UV_NO_SYNC=1
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

exec uv run "${cmd[@]}"
