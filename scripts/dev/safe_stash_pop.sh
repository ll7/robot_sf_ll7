#!/usr/bin/env bash
set -euo pipefail

# Fail-closed stash pop for multi-worktree repositories (issue #7700).
#
# The stash stack (refs/stash) lives in the common Git dir, so every linked
# worktree shares one stash namespace. A bare `git stash pop` in one worktree
# can apply another session's WIP into the wrong checkout. This wrapper pops
# only when the top stash entry's message names the current branch; otherwise
# it exits 2 listing the top entries so the caller can identify the intended
# stash explicitly (e.g. `git stash pop stash@{n}`) after verifying the message.
#
# Workflow guidance:
#   - Never use bare `git stash pop` in a linked worktree.
#   - Prefer temp commits (`git commit -m "WIP <branch>"`) on long-lived lanes.
#   - When stashing is required, use `git stash push -m "<branch> <purpose>"`
#     and restore with this wrapper or an explicit `git stash pop stash@{n}`.

usage() {
  cat <<'EOF'
Usage: scripts/dev/safe_stash_pop.sh [--help]

Pops the top stash entry only when its message names the current branch.
Refuses (exit 2) when the top entry is not branch-named or no stash exists.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

# A linked worktree has `.git` as a file and a different git-dir from the common dir.
git_dir="$(git rev-parse --git-dir 2>/dev/null || true)"
common_dir="$(git rev-parse --git-common-dir 2>/dev/null || true)"
if [[ -n "$git_dir" && "$git_dir" != "$common_dir" ]]; then
  printf 'Linked worktree detected (git-dir %s != common-dir %s).\n' "$git_dir" "$common_dir" >&2
fi

current_branch="$(git branch --show-current 2>/dev/null || true)"
if [[ -z "$current_branch" ]]; then
  printf 'ERROR: not on a branch (detached HEAD); refusing to pop a shared stash.\n' >&2
  exit 2
fi

stash_list="$(git stash list --format='%gd %gs' 2>/dev/null || true)"
if [[ -z "$stash_list" ]]; then
  printf 'No stash entries present; nothing to pop.\n' >&2
  exit 0
fi

top_message="$(printf '%s\n' "$stash_list" | head -n 1)"
top_ref="${top_message%% *}"

if [[ "$top_message" == *"$current_branch"* ]]; then
  printf 'Top stash entry names current branch %s; popping %s.\n' "$current_branch" "$top_ref" >&2
  exec git stash pop
fi

printf 'ERROR: top stash entry does not name current branch %q.\n' "$current_branch" >&2
printf 'The stash namespace is shared across all linked worktrees; popping it would apply\n' >&2
printf "%s\n" "another session's WIP into this checkout." >&2
printf '\nTop stash entries:\n' >&2
printf '%s\n' "$stash_list" | head -n 5 >&2
printf '\nPop the intended entry explicitly after verifying its message, for example:\n' >&2
printf '  git stash list\n  git stash pop stash@{n}\n' >&2
exit 2