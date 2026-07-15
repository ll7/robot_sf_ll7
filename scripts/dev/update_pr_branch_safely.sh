#!/usr/bin/env bash
# shellcheck shell=bash
#
# Guarded PR branch updater with a lease-protected local fallback.
#
# This is the safe gate helper referenced in docs/dev_guide.md.  The installed
# GitHub CLI in some gate environments does not support `gh pr update-branch`
# (unknown command/flag) and the REST `/pulls/{n}/update-branch` endpoint may
# return 404.  When the supported remote branch-update path is unavailable this
# script falls back to a local rebase onto the base branch followed by a
# force-with-lease push.  Every mutating step is guarded by the caller-recorded
# expected head SHA and a PR-gate worktree lease so the operation cannot
# silently retarget a different commit or be reaped mid-flight.
#
# Usage:
#     scripts/dev/update_pr_branch_safely.sh <pr> \
#         --expected-head-sha <sha> [--repo OWNER/REPO] [options]
#
# Options:
#     --pr <n>                  PR number (positional also accepted)
#     --repo OWNER/REPO         owner/repo (default: detect from gh)
#     --expected-head-sha <sha> required guard; no mutation if the live head moved
#     --base <branch>           base branch to rebase onto (default: PR base ref)
#     --remote <name>           remote to fetch/push (default: origin)
#     --no-local-fallback       fail instead of falling back to local rebase/push
#     --dry-run                 verify and print the plan without mutating
#     --json                    emit machine-readable JSON (default behavior)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEASE_HELPER="${SCRIPT_DIR}/pr_gate_lease.py"

REPO=""
PR=""
EXPECTED=""
BASE_REF_OVERRIDE=""
BASE_REF=""
LIVE_HEAD=""
REMOTE="origin"
LOCAL_FALLBACK=1
DRY_RUN=0

usage() {
  echo "Usage: $0 <pr> --expected-head-sha <sha> [--repo OWNER/REPO] [options]" >&2
  exit 2
}

# --- argument parsing ---------------------------------------------------------
POS_PR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      [[ $# -ge 2 ]] || usage
      REPO="$2"
      shift 2
      ;;
    --expected-head-sha)
      [[ $# -ge 2 ]] || usage
      EXPECTED="$2"
      shift 2
      ;;
    --base)
      [[ $# -ge 2 ]] || usage
      BASE_REF_OVERRIDE="$2"
      shift 2
      ;;
    --remote)
      [[ $# -ge 2 ]] || usage
      REMOTE="$2"
      shift 2
      ;;
    --no-local-fallback)
      LOCAL_FALLBACK=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --json)
      shift
      ;;
    -*)
      echo "Unexpected option: $1" >&2
      usage
      ;;
    *)
      [[ -z "$POS_PR" ]] || usage
      POS_PR="$1"
      shift
      ;;
  esac
done

PR="${PR:-$POS_PR}"
[[ -n "$PR" ]] || usage
[[ -n "$EXPECTED" ]] || {
  echo "error: --expected-head-sha is required (record the current PR head SHA first)" >&2
  exit 2
}

sanitize() {
  # Strip double quotes and newlines so values stay valid in JSON output.
  local v="$1"
  v="${v//\"/}"
  v="${v//$'\n'/ }"
  printf '%s' "$v"
}

emit_result() {
  # $1 status, $2 updated(bool), $3 error(string), $4 method(string)
  local status="$1" updated="$2" error="${3:-}" method="${4:-}"
  local err_json="null"
  if [[ -n "$error" ]]; then
    err_json="\"$(sanitize "$error")\""
  fi
  printf '{"status":"%s","pr":"%s","repo":"%s","expected_head_sha":"%s","live_head_sha":"%s","base":"%s","method":"%s","updated":%s,"error":%s}\n' \
    "$status" "$PR" "$REPO" "$EXPECTED" "$LIVE_HEAD" "$BASE_REF" "$method" "$updated" "$err_json"
}

LEASE_CREATED=0
release_lease() {
  if [[ "$LEASE_CREATED" -eq 1 ]] && [[ -f "$LEASE_HELPER" ]]; then
    python3 "$LEASE_HELPER" release >/dev/null 2>&1 || true
    LEASE_CREATED=0
  fi
}
trap release_lease EXIT

resolve_repo() {
  if [[ -n "$REPO" ]]; then
    printf '%s' "$REPO"
    return 0
  fi
  gh repo view --json nameWithOwner --jq '.nameWithOwner' 2>/dev/null || true
}

# --- metadata + guard ---------------------------------------------------------
REPO="$(resolve_repo)"
if [[ -z "$REPO" ]]; then
  emit_result "error" "false" "could not resolve repository (pass --repo OWNER/REPO)" ""
  exit 2
fi

set +e
META_ERR=""
LIVE_HEAD="$(gh api "repos/${REPO}/pulls/${PR}" --jq '.head.sha' 2>/dev/null)"
RC_HEAD=$?
HEAD_REF="$(gh api "repos/${REPO}/pulls/${PR}" --jq '.head.ref' 2>/dev/null)"
BASE_REF_RESOLVED="$(gh api "repos/${REPO}/pulls/${PR}" --jq '.base.ref' 2>/dev/null)"
set -e

if [[ $RC_HEAD -ne 0 ]] || [[ -z "$LIVE_HEAD" ]]; then
  emit_result "error" "false" "could not fetch PR head SHA from REST" ""
  exit 2
fi

BASE_REF="${BASE_REF_OVERRIDE:-$BASE_REF_RESOLVED}"
BASE_REF="${BASE_REF:-main}"

if [[ "$LIVE_HEAD" != "$EXPECTED" ]]; then
  emit_result "head_mismatch" "false" "PR head changed since expected SHA was recorded" ""
  exit 1
fi

# --- attempt the supported remote branch-update path --------------------------
REST_ERR_FILE="$(mktemp)"
set +e
REST_MSG="$(gh api "repos/${REPO}/pulls/${PR}/update-branch" \
  --method PUT -f "expected_head_sha=${EXPECTED}" --jq '.message' 2>"${REST_ERR_FILE}")"
REST_RC=$?
REST_STDERR="$(cat "${REST_ERR_FILE}" 2>/dev/null || true)"
rm -f "${REST_ERR_FILE}"
set -e

if [[ $REST_RC -eq 0 ]]; then
  emit_result "update_requested" "true" "" "gh_rest_update_branch"
  exit 0
fi

# --- fallback to local lease-protected rebase/push ----------------------------
if [[ "$LOCAL_FALLBACK" -eq 0 ]]; then
  emit_result "error" "false" "gh update-branch unavailable and --no-local-fallback set (${REST_STDERR})" "gh_rest_update_branch"
  exit 2
fi

echo "info: gh update-branch unavailable (rc=${REST_RC}); falling back to local rebase/push" >&2

if [[ "$HEAD_REF" == "$BASE_REF" ]]; then
  emit_result "error" "false" "refusing to rebase/push the base branch itself (${HEAD_REF})" "local_fallback"
  exit 2
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
if [[ "$CURRENT_BRANCH" != "$HEAD_REF" ]]; then
  emit_result "error" "false" "current branch '${CURRENT_BRANCH}' is not the PR head branch '${HEAD_REF}'" "local_fallback"
  exit 2
fi

LOCAL_HEAD="$(git rev-parse HEAD 2>/dev/null || true)"
if [[ "$LOCAL_HEAD" != "$EXPECTED" ]]; then
  emit_result "head_mismatch" "false" "local HEAD (${LOCAL_HEAD}) differs from expected SHA" "local_fallback"
  exit 1
fi

if [[ -f "$LEASE_HELPER" ]]; then
  python3 "$LEASE_HELPER" create --pr "$PR" >/dev/null 2>&1 || true
  LEASE_CREATED=1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "dry-run: would fetch ${REMOTE} ${BASE_REF} ${HEAD_REF}, rebase onto origin/${BASE_REF}, then push --force-with-lease to ${REMOTE}/${HEAD_REF}" >&2
  emit_result "dry_run" "false" "" "local_fallback"
  exit 0
fi

set +e
git fetch "${REMOTE}" "${BASE_REF}" "${HEAD_REF}" >/dev/null 2>&1
FETCH_RC=$?
set -e
if [[ $FETCH_RC -ne 0 ]]; then
  emit_result "error" "false" "git fetch of base/head refs failed" "local_fallback"
  exit 2
fi

set +e
git rebase "origin/${BASE_REF}" >/dev/null 2>&1
REBASE_RC=$?
set -e
if [[ $REBASE_RC -ne 0 ]]; then
  git rebase --abort >/dev/null 2>&1 || true
  emit_result "error" "false" "git rebase onto origin/${BASE_REF} failed (resolve conflicts manually)" "local_fallback"
  exit 2
fi

NEW_HEAD="$(git rev-parse HEAD 2>/dev/null || true)"
PUSH_REF="refs/heads/${HEAD_REF}"

set +e
git push --force-with-lease="${REMOTE}/${HEAD_REF}:${EXPECTED}" \
  "${REMOTE}" "HEAD:${PUSH_REF}" >/dev/null 2>&1
PUSH_RC=$?
set -e
if [[ $PUSH_RC -ne 0 ]]; then
  emit_result "error" "false" "git push --force-with-lease was rejected (remote head moved or divergence)" "local_fallback"
  exit 2
fi

REMOTE_SHA="$(git ls-remote --heads "${REMOTE}" "${PUSH_REF}" 2>/dev/null | awk '{print $1}' | head -n1 || true)"
if [[ -n "$REMOTE_SHA" && "$REMOTE_SHA" != "$NEW_HEAD" ]]; then
  emit_result "error" "false" "post-push verification failed: remote ${REMOTE_SHA} != local ${NEW_HEAD}" "local_fallback"
  exit 2
fi

release_lease
emit_result "fallback_local_rebase" "true" "" "local_fallback"
exit 0
