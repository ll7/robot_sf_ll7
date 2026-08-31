#!/usr/bin/env bash
set -euo pipefail

# Guarded PR merge wrapper with a REST fallback for the worktree-base conflict
# (issue #7733).
#
# `gh pr merge --squash --delete-branch --match-head-commit <sha>` switches the
# current checkout to the base branch after merging. In multi-worktree
# repositories the base (usually `main`) is often checked out in another linked
# worktree, so the local switch fails with
# `fatal: '<base>' is already used by worktree at ...` and the merge never
# reaches GitHub. This wrapper detects that exact signature and retries the
# squash merge through the REST API with the same exact-head binding — no local
# checkout switch needed. Every other failure stays fail-closed.
#
# Usage:
#   scripts/dev/gh_pr_merge.sh <pr-number> --match-head-commit <sha> [--repo owner/name]
#
# The exact-head binding is mandatory: without `--match-head-commit` the REST
# fallback is refused and the wrapper exits 2.

usage() {
  cat <<'EOF'
Usage: scripts/dev/gh_pr_merge.sh <pr-number> --match-head-commit <sha> [--repo owner/name]

Runs the standard `gh pr merge --squash --delete-branch --match-head-commit <sha>`
and falls back to the REST squash merge (same exact-head binding) only when the
base branch is checked out in another worktree.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "$#" -lt 3 ]]; then
  usage
  exit 2
fi

pr_number="$1"
shift
expected_head_sha=""
repo=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --match-head-commit)
      expected_head_sha="$2"
      shift 2
      ;;
    --repo)
      repo="$2"
      shift 2
      ;;
    *)
      printf 'ERROR: unexpected argument %q\n' "$1" >&2
      usage
      exit 2
      ;;
  esac
done

repo_args=()
if [[ -n "$repo" ]]; then
  repo_args=(--repo "$repo")
fi

if ! [[ "$expected_head_sha" =~ ^[0-9a-fA-F]{40}$ ]]; then
  printf 'ERROR: --match-head-commit requires a full 40-char SHA; got %q\n' "$expected_head_sha" >&2
  exit 2
fi

if ! python3 "$SCRIPT_DIR/github_transport_policy.py" check \
  --helper "$SCRIPT_DIR/gh_pr_merge.sh" --root "$SCRIPT_DIR/../.." --json >/dev/null; then
  printf 'ERROR: gh_pr_merge.sh is not admitted by the GitHub transport policy.\n' >&2
  exit 2
fi

# First try the native gh path (handles branch deletion, queue interplay, etc.).
_gh_pr_merge_err="$(mktemp "${TMPDIR:-/tmp}/gh-pr-merge.XXXXXX")"
trap 'rm -f "$_gh_pr_merge_err"' EXIT
if gh pr merge "$pr_number" --squash --delete-branch --match-head-commit "$expected_head_sha" "${repo_args[@]}" 2>"$_gh_pr_merge_err"; then
  exit 0
fi
merge_error="$(cat "$_gh_pr_merge_err")"
rm -f "$_gh_pr_merge_err"

# Only a policy-approved worktree-conflict signature triggers the REST fallback.
if ! python3 "$SCRIPT_DIR/github_transport_policy.py" classify \
  --helper "$SCRIPT_DIR/gh_pr_merge.sh" --error "$merge_error" >/dev/null 2>&1; then
  printf 'ERROR: gh pr merge failed:\n%s\n' "$merge_error" >&2
  exit 1
fi

printf 'gh pr merge blocked by a worktree base checkout; retrying through REST (issue #7733).\n' >&2
printf '%s\n' "$merge_error" >&2

if [[ -z "$repo" ]]; then
  repo="$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || true)"
fi
if [[ -z "$repo" ]]; then
  printf 'ERROR: cannot resolve owner/name for the REST merge fallback.\n' >&2
  exit 2
fi

# Re-read the live head before merging so the REST sha binding is current.
live_head="$(gh pr view "$pr_number" "${repo_args[@]}" --json headRefOid --jq .headRefOid 2>/dev/null || true)"
if [[ -z "$live_head" || "$live_head" != "$expected_head_sha" ]]; then
  printf 'ERROR: REST fallback refuses stale head (expected %s, live %s).\n' \
    "$expected_head_sha" "${live_head:-<unknown>}" >&2
  exit 2
fi

merge_json="$(gh api -X PUT "repos/${repo}/pulls/${pr_number}/merge" \
  -f merge_method=squash -f sha="$expected_head_sha" --jq '{merged, message, sha}' 2>/dev/null || true)"
case "$merge_json" in
  *'"merged": true'*)
    merged_sha="$(printf '%s\n' "$merge_json" | sed -n 's/.*"sha": "\([0-9a-f]\{40\}\)".*/\1/p')"
    printf 'Merged via REST fallback; merge SHA: %s\n' "${merged_sha:-(see API response)}" >&2
    ;;
  *)
    printf 'ERROR: REST merge failed or returned an unexpected payload.\n%s\n' "$merge_json" >&2
    exit 1
    ;;
esac

# Best-effort remote branch deletion after the squash (mirrors --delete-branch).
head_ref="$(gh pr view "$pr_number" "${repo_args[@]}" --json headRefName --jq .headRefName 2>/dev/null || true)"
if [[ -n "$head_ref" ]]; then
  if gh api -X DELETE "repos/${repo}/git/refs/heads/${head_ref}" >/dev/null 2>&1; then
    printf 'Deleted remote branch %s.\n' "$head_ref" >&2
  else
    printf 'WARNING: could not delete remote branch %s (best-effort).\n' "$head_ref" >&2
  fi
fi
exit 0
