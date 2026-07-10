#!/usr/bin/env bash
# Bootstrap a fresh linked Git worktree so that `source .venv/bin/activate` works.
#
# Problem: `uv sync --all-extras` in a fresh linked worktree may detect and
# reuse the main checkout's .venv instead of creating one locally, leaving the
# worktree without a .venv/bin/activate to source (issue #5091).
#
# Fix: explicitly create a local virtual environment with `uv venv .venv` first,
# then sync packages into it, then verify the environment exists before returning.
# This makes the bootstrap fail closed with an actionable message rather than
# silently succeeding and leaving the caller without a working .venv.
#
# Usage:
#   scripts/dev/bootstrap_worktree.sh [--no-symlink-machine] [--help|-h]
#
# Options:
#   --no-symlink-machine  Skip symlinking local.machine.md from the main checkout.
#   -h, --help            Show this help and exit.
#
# The script must be run from the root of the worktree to bootstrap.
# Example:
#   cd /path/to/robot_sf_ll7.worktrees/my-branch
#   scripts/dev/bootstrap_worktree.sh
#   source .venv/bin/activate

set -euo pipefail

show_help() {
    cat <<'EOF'
Usage: scripts/dev/bootstrap_worktree.sh [--no-symlink-machine] [--help|-h]

Bootstrap a fresh linked Git worktree: run `uv venv .venv && uv sync --all-extras`,
then verify .venv/bin/python exists before returning. Fails closed with an
actionable error message if the environment is not usable after sync.

The explicit `uv venv .venv` step is required: `uv sync --all-extras` alone may
silently reuse the main checkout's .venv without creating one in the worktree,
leaving .venv/bin/activate missing (issue #5091).

Options:
  --no-symlink-machine  Skip symlinking local.machine.md from the main checkout.
  -h, --help            Show this help and exit.

Run from the worktree root you want to bootstrap. Example:
  cd ../robot_sf_ll7.worktrees/issue-1234-my-branch
  scripts/dev/bootstrap_worktree.sh
  source .venv/bin/activate
EOF
}

symlink_machine=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-symlink-machine)
            symlink_machine=0
            shift
            ;;
        -h | --help)
            show_help
            exit 0
            ;;
        *)
            echo "bootstrap_worktree: unknown argument: $1" >&2
            show_help >&2
            exit 2
            ;;
    esac
done

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

# Derive the main checkout path from the git common dir.
git_common_dir="$(git rev-parse --git-common-dir)"
if [[ "$git_common_dir" != /* ]]; then
    git_common_dir="$(cd "$repo_root/$git_common_dir" && pwd)"
fi
main_repo_root="$(cd "$git_common_dir/.." && pwd)"

is_linked=0
if [[ "$git_common_dir" != "$repo_root/.git" ]]; then
    is_linked=1
fi

# Symlink local.machine.md from the main checkout when requested and not already present.
if [[ "$symlink_machine" -eq 1 && "$is_linked" -eq 1 ]]; then
    main_machine="$main_repo_root/local.machine.md"
    if [[ -f "$main_machine" && ! -e "$repo_root/local.machine.md" ]]; then
        ln -s "$main_machine" "$repo_root/local.machine.md"
        echo "bootstrap_worktree: symlinked local.machine.md from $main_machine"
    fi
fi

# Create a local virtual environment explicitly.
# `uv sync --all-extras` alone may silently reuse the main checkout's .venv
# (detectable because it prints no "Creating virtual environment" line and runs
# in ~1ms). Creating .venv explicitly first guarantees packages land here.
if [[ ! -d "$repo_root/.venv" ]]; then
    echo "bootstrap_worktree: creating local .venv ..."
    uv venv .venv
fi

echo "bootstrap_worktree: syncing dependencies (uv sync --all-extras) ..."
uv sync --all-extras

# Fail closed: verify the environment is actually usable.
if [[ ! -x "$repo_root/.venv/bin/python" ]]; then
    cat >&2 <<'EOF'
bootstrap_worktree: ERROR — .venv/bin/python not found after uv sync --all-extras.

Suggested recovery:
  rm -rf .venv
  uv venv .venv
  uv sync --all-extras
  source .venv/bin/activate

If uv is not on PATH, ensure the main checkout's .venv/bin is in PATH or
install uv via your system package manager.
EOF
    exit 1
fi

echo "bootstrap_worktree: .venv/bin/python is ready."
echo "bootstrap_worktree: run 'source .venv/bin/activate' to activate."
