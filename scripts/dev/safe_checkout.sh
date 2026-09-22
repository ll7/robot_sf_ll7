#!/usr/bin/env bash
# Fail-closed guard for destructive `git checkout` restores (issue #9445).
#
# Usage:
#   scripts/dev/safe_checkout.sh [--backup <patch-file>] [--path <path> ...] -- <checkout-args...>
#
# Refuses (exit 2) when the guard scope has tracked modifications unless
# --backup captured them to a patch file first, then runs
# `git checkout <checkout-args...>` verbatim. Untracked files are never
# touched by a path restore, so only tracked dirt is guarded.
#
# Examples:
#   scripts/dev/safe_checkout.sh -- README.md -- README.md
#   scripts/dev/safe_checkout.sh --backup /tmp/opencode/work.patch -- docs/ -- docs/
set -euo pipefail

backup_file=""
declare -a guard_paths=()
declare -a checkout_args=()

usage() {
  sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --backup)
      [[ $# -ge 2 ]] || { echo "safe_checkout: --backup requires a patch file path" >&2; exit 2; }
      backup_file="$2"
      shift 2
      ;;
    --path)
      [[ $# -ge 2 ]] || { echo "safe_checkout: --path requires a path argument" >&2; exit 2; }
      guard_paths+=("$2")
      shift 2
      ;;
    --)
      shift
      checkout_args=("$@")
      break
      ;;
    *)
      echo "safe_checkout: unexpected argument '$1' (pass checkout args after --)" >&2
      exit 2
      ;;
  esac
done

[[ ${#checkout_args[@]} -gt 0 ]] || { echo "safe_checkout: no checkout args after --" >&2; exit 2; }
git rev-parse --git-dir >/dev/null 2>&1 || { echo "safe_checkout: not inside a git repository" >&2; exit 2; }

if [[ ${#guard_paths[@]} -eq 0 ]]; then
  dirty_list="$(git status --short --untracked-files=no || true)"
else
  dirty_list="$(git status --short --untracked-files=no -- "${guard_paths[@]}" || true)"
fi

if [[ -z "$dirty_list" ]]; then
  exec git checkout "${checkout_args[@]}"
fi

dirty_count="$(printf '%s\n' "$dirty_list" | wc -l)"
if [[ -z "$backup_file" ]]; then
  echo "safe_checkout: refusing checkout with $dirty_count dirty tracked file(s):" >&2
  printf '%s\n' "$dirty_list" | head -20 >&2
  if [[ "$dirty_count" -gt 20 ]]; then
    echo "safe_checkout: ... and $((dirty_count - 20)) more (see git status)" >&2
  fi
  echo "safe_checkout: rerun with --backup <patch-file> to snapshot first, or clean the tree" >&2
  exit 2
fi

backup_dir="$(dirname "$backup_file")"
[[ -d "$backup_dir" ]] || { echo "safe_checkout: backup directory missing: $backup_dir" >&2; exit 2; }
if [[ ${#guard_paths[@]} -eq 0 ]]; then
  git diff HEAD -- >"$backup_file"
else
  git diff HEAD -- "${guard_paths[@]}" >"$backup_file"
fi
[[ -s "$backup_file" ]] || {
  echo "safe_checkout: backup is empty but the tree is dirty; refusing" >&2
  exit 2
}
echo "safe_checkout: dirty tree snapshotted to $backup_file ($dirty_count file(s))" >&2
exec git checkout "${checkout_args[@]}"
