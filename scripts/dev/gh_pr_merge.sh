#!/usr/bin/env bash
set -euo pipefail

# Guarded PR merge wrapper with narrow REST fallbacks for transport-only
# failures: the worktree-base conflict from issue #7733 and GraphQL quota
# exhaustion from issue #8132.
#
# `gh pr merge --squash --delete-branch --match-head-commit <sha>` switches the
# current checkout to the base branch after merging. In multi-worktree
# repositories the base (usually `main`) is often checked out in another linked
# worktree, so the local switch fails with
# `fatal: '<base>' is already used by worktree at ...` and the merge never
# reaches GitHub. This wrapper detects that exact signature and retries the
# squash merge through the REST API with the same exact-head binding — no local
# checkout switch needed.
#
# The native path can also fail before merging when the GraphQL quota is
# exhausted even though REST remains available. That fallback is eligible only
# for a GraphQL rate-limit/quota diagnostic. It re-verifies the live head,
# open/non-draft state, clean mergeability, and `merge-ready` label through REST
# before the exact-head REST merge. Authentication, repository-resolution, and
# every other failure stay fail-closed.
#
# Usage:
#   scripts/dev/gh_pr_merge.sh <pr-number> --match-head-commit <sha> [--repo owner/name]
#
# The exact-head binding is mandatory: without `--match-head-commit` the REST
# fallback is refused and the wrapper exits 2.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REST_LABEL_PAGE_SIZE=100
REST_LABEL_PAGE_CEILING=10

usage() {
  cat <<'EOF_USAGE'
Usage: scripts/dev/gh_pr_merge.sh <pr-number> --match-head-commit <sha> [--repo owner/name]

Runs the standard `gh pr merge --squash --delete-branch --match-head-commit <sha>`
and falls back to the REST squash merge with the same exact-head binding only
when the base branch is checked out in another worktree or the GraphQL quota is
exhausted. The quota fallback first re-verifies the live head, open/non-draft
state, clean mergeability, and `merge-ready` label through REST. All other
failures remain fail-closed.
EOF_USAGE
}

is_graphql_quota_failure() {
  local normalized
  normalized="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"

  # Auth, authorization, and repository lookup failures may also be prefixed
  # with `GraphQL:`. They take precedence over the quota marker.
  case "$normalized" in
    *"bad credentials"* | *"http 401"* | *"requires authentication"* | \
      *"authentication required"* | *"authentication failed"* | *"invalid token"* | \
      *"resource not accessible"* | *"forbidden"* | *"permission denied"* | \
      *"could not resolve to a repository"* | *"could not resolve to a pull request"* | \
      *"could not resolve to a pullrequest"* | *"repository not found"*)
      return 1
      ;;
  esac

  [[ "$normalized" == *"graphql:"* ]] || return 1
  [[ "$normalized" == *"rate limit"* ||
    ( "$normalized" == *"quota"* &&
      ( "$normalized" == *"exhausted"* || "$normalized" == *"exceeded"* ) ) ]]
}

repo_from_git_remote() {
  local remote_url host_path host candidate
  local expected_host="${GH_HOST:-github.com}"
  remote_url="$(git config --get remote.origin.url 2>/dev/null || true)"
  [[ -n "$remote_url" ]] || return 1
  remote_url="${remote_url%.git}"

  case "$remote_url" in
    *://*/*/*)
      host_path="${remote_url#*://}"
      host_path="${host_path#*@}"
      host="${host_path%%/*}"
      host="${host%%:*}"
      candidate="${host_path#*/}"
      ;;
    *:*/*)
      host="${remote_url%%:*}"
      host="${host#*@}"
      candidate="${remote_url#*:}"
      ;;
    *)
      return 1
      ;;
  esac

  [[ "$host" == "$expected_host" ]] || return 1
  if [[ "$candidate" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  return 1
}

read_merge_ready_label() {
  local repo_name="$1"
  local number="$2"
  local error_file="$3"
  local page page_result status count found
  local has_merge_ready="false"

  for ((page = 1; page <= REST_LABEL_PAGE_CEILING; page++)); do
    : >"$error_file"
    if ! page_result="$(gh api \
      "repos/${repo_name}/issues/${number}/labels?per_page=${REST_LABEL_PAGE_SIZE}&page=${page}" \
      --jq 'if type != "array" then
              ["error", "expected-array", ""]
            elif (all(.[]; if type == "object" then
              ((.name | type) == "string" and (.name | length) > 0)
            else false end) | not) then
              ["error", "malformed-label-row", ""]
            else
              ["ok", (length | tostring),
               ((any(.[]; .name == "merge-ready")) | tostring)]
            end | @tsv' \
      2>"$error_file")"; then
      printf 'ERROR: REST merge-ready label read failed on page %s:\n%s\n' \
        "$page" "$(cat "$error_file")" >&2
      return 1
    fi

    status=""
    count=""
    found=""
    IFS=$'\t' read -r status count found <<<"$page_result"
    if [[ "$status" != "ok" ]]; then
      printf 'ERROR: REST merge-ready label page %s was malformed (%s).\n' \
        "$page" "${count:-unknown-error}" >&2
      return 1
    fi
    if ! [[ "$count" =~ ^[0-9]+$ ]]; then
      printf 'ERROR: REST merge-ready label page %s returned an invalid count %q.\n' \
        "$page" "$count" >&2
      return 1
    fi
    if ((count > REST_LABEL_PAGE_SIZE)); then
      printf 'ERROR: REST merge-ready label page %s exceeded page size %s.\n' \
        "$page" "$REST_LABEL_PAGE_SIZE" >&2
      return 1
    fi
    if [[ "$found" != "true" && "$found" != "false" ]]; then
      printf 'ERROR: REST merge-ready label page %s returned an invalid match flag %q.\n' \
        "$page" "$found" >&2
      return 1
    fi
    if [[ "$found" == "true" ]]; then
      has_merge_ready="true"
    fi
    if ((count < REST_LABEL_PAGE_SIZE)); then
      printf '%s\n' "$has_merge_ready"
      return 0
    fi
  done

  printf 'ERROR: REST merge-ready label pagination exceeded the page ceiling of %s.\n' \
    "$REST_LABEL_PAGE_CEILING" >&2
  return 1
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

fallback_mode=""
if [[ "$merge_error" == *"already used by worktree"* ]]; then
  fallback_mode="worktree"
elif is_graphql_quota_failure "$merge_error"; then
  fallback_mode="graphql_quota"
else
  printf 'ERROR: gh pr merge failed:\n%s\n' "$merge_error" >&2
  exit 1
fi

# Only a policy-approved transport signature may trigger the REST fallback.
# The quota predicate above remains the semantic guard for GraphQL failures;
# the shared policy supplies the canonical fail-closed marker precedence.
if ! python3 "$SCRIPT_DIR/github_transport_policy.py" classify \
  --helper "$SCRIPT_DIR/gh_pr_merge.sh" --error "$merge_error" >/dev/null 2>&1; then
  printf 'ERROR: gh pr merge failure is not admitted by the GitHub transport policy:\n%s\n' \
    "$merge_error" >&2
  exit 1
fi

if [[ "$fallback_mode" == "worktree" ]]; then
  printf 'gh pr merge blocked by a worktree base checkout; retrying through REST (issue #7733).\n' >&2
else
  printf 'gh pr merge hit GraphQL quota exhaustion; re-verifying guarded state through REST (issue #8132).\n' >&2
fi
printf '%s\n' "$merge_error" >&2

if [[ -z "$repo" ]]; then
  repo="${GH_REPO:-}"
fi
if [[ -z "$repo" ]]; then
  repo="$(repo_from_git_remote || true)"
fi
if [[ -z "$repo" ]]; then
  # Compatibility fallback for callers outside a Git checkout. This may use
  # GraphQL, so quota-path reliability comes from --repo, GH_REPO, or origin.
  repo="$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || true)"
fi
if [[ -z "$repo" ]]; then
  printf 'ERROR: cannot resolve owner/name for the REST merge fallback.\n' >&2
  exit 2
fi
if ! [[ "$repo" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  printf 'ERROR: invalid owner/name for the REST merge fallback: %q\n' "$repo" >&2
  exit 2
fi
repo_args=(--repo "$repo")

live_head=""
head_ref=""
if [[ "$fallback_mode" == "graphql_quota" ]]; then
  : >"$_gh_pr_merge_err"
  rest_preflight=""
  if ! rest_preflight="$(gh api "repos/${repo}/pulls/${pr_number}" \
    --jq '[.head.sha // "", .head.ref // "", .state // "", (.draft | tostring), (.mergeable | tostring), (.mergeable_state // "")] | @tsv' \
    2>"$_gh_pr_merge_err")"; then
    printf 'ERROR: REST merge preflight failed:\n%s\n' "$(cat "$_gh_pr_merge_err")" >&2
    exit 1
  fi
  pr_state=""
  draft=""
  mergeable=""
  mergeable_state=""
  IFS=$'\t' read -r live_head head_ref pr_state draft mergeable mergeable_state \
    <<<"$rest_preflight"

  if [[ -z "$live_head" || "$live_head" != "$expected_head_sha" ]]; then
    printf 'ERROR: REST fallback refuses stale head (expected %s, live %s).\n' \
      "$expected_head_sha" "${live_head:-<unknown>}" >&2
    exit 2
  fi
  if [[ "$pr_state" != "open" || "$draft" != "false" ]]; then
    printf 'ERROR: REST fallback refuses PR state (state=%s, draft=%s).\n' \
      "${pr_state:-<unknown>}" "${draft:-<unknown>}" >&2
    exit 2
  fi
  if [[ "$mergeable" != "true" || "$mergeable_state" != "clean" ]]; then
    printf 'ERROR: REST fallback refuses non-clean mergeability (mergeable=%s, state=%s).\n' \
      "${mergeable:-<unknown>}" "${mergeable_state:-<unknown>}" >&2
    exit 2
  fi
  if ! has_merge_ready="$(read_merge_ready_label "$repo" "$pr_number" "$_gh_pr_merge_err")"; then
    exit 1
  fi
  if [[ "$has_merge_ready" != "true" ]]; then
    printf 'ERROR: REST fallback refuses PR without the merge-ready label.\n' >&2
    exit 2
  fi
else
  # Preserve the issue #7733 contract: re-read the live head before merging so
  # the REST sha binding is current.
  live_head="$(gh pr view "$pr_number" "${repo_args[@]}" --json headRefOid --jq .headRefOid 2>/dev/null || true)"
  if [[ -z "$live_head" || "$live_head" != "$expected_head_sha" ]]; then
    printf 'ERROR: REST fallback refuses stale head (expected %s, live %s).\n' \
      "$expected_head_sha" "${live_head:-<unknown>}" >&2
    exit 2
  fi
fi

merge_json="$(gh api -X PUT "repos/${repo}/pulls/${pr_number}/merge" \
  -f merge_method=squash -f sha="$expected_head_sha" --jq '{merged, message, sha}' 2>/dev/null || true)"
case "$merge_json" in
  *'"merged": true'* | *'"merged":true'*)
    merged_sha="$(printf '%s\n' "$merge_json" | sed -n 's/.*"sha":[[:space:]]*"\([0-9a-f]\{40\}\)".*/\1/p')"
    printf 'Merged via REST fallback; merge SHA: %s\n' "${merged_sha:-(see API response)}" >&2
    ;;
  *)
    printf 'ERROR: REST merge failed or returned an unexpected payload.\n%s\n' "$merge_json" >&2
    exit 1
    ;;
esac

# Best-effort remote branch deletion after the squash (mirrors --delete-branch).
if [[ -z "$head_ref" ]]; then
  head_ref="$(gh pr view "$pr_number" "${repo_args[@]}" --json headRefName --jq .headRefName 2>/dev/null || true)"
fi
if [[ -n "$head_ref" ]]; then
  if gh api -X DELETE "repos/${repo}/git/refs/heads/${head_ref}" >/dev/null 2>&1; then
    printf 'Deleted remote branch %s.\n' "$head_ref" >&2
  else
    printf 'WARNING: could not delete remote branch %s (best-effort).\n' "$head_ref" >&2
  fi
fi
exit 0
