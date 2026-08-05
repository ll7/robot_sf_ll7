#!/usr/bin/env bash
# shellcheck shell=bash
#
# Guarded PR branch updater with deleted-source-ref restore and a
# lease-protected local fallback.
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
# When the PR source branch has been deleted on the remote, update-branch fails
# with "Could not resolve head ref" (issue #6689).  After the metadata read and
# the expected-head guard pass, this script detects the missing
# refs/heads/<head-ref> and restores it with a plain (non-force) push of the
# immutable PR head SHA, which was already verified equal to the expected head;
# the restore is reported in the JSON result and the normal update paths run
# afterwards.  Cross-fork PRs with a deleted head branch fail closed instead.
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
#     --gate-worktree-path      registered gate worktree path; a vanished worktree
#                               fails closed before any branch-switch/conflict op
#     --json                    emit machine-readable JSON (default behavior)
#     -h, --help                print this help and exit 0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEASE_HELPER="${SCRIPT_DIR}/pr_gate_lease.py"
GUARD_HELPER="${SCRIPT_DIR}/gate_worktree_guard.py"

REPO=""
PR=""
EXPECTED=""
BASE_REF_OVERRIDE=""
BASE_REF=""
LIVE_HEAD=""
HEAD_REF=""
HEAD_REPO_FULL=""
BASE_REPO_FULL=""
REMOTE="origin"
LOCAL_FALLBACK=1
DRY_RUN=0
GATE_WORKTREE_PATH=""
SOURCE_REF_RESTORED=0

usage() {
  echo "Usage: $0 <pr> --expected-head-sha <sha> [--repo OWNER/REPO] [options]" >&2
  exit 2
}

print_help() {
  cat <<'HELP'
Usage: scripts/dev/update_pr_branch_safely.sh <pr> \
    --expected-head-sha <sha> [--repo OWNER/REPO] [options]

Guarded PR branch updater with deleted-source-ref restore and a
lease-protected local fallback. A deleted PR source branch is restored
with a plain (non-force) push of the immutable PR head SHA before the
update path runs; cross-fork PRs with a deleted head branch fail closed.

Options:
    --pr <n>                  PR number (positional also accepted)
    --repo OWNER/REPO         owner/repo (default: detect from gh)
    --expected-head-sha <sha> required guard; no mutation if the live head moved
    --base <branch>           base branch to rebase onto (default: PR base ref)
    --remote <name>           remote to fetch/push (default: origin)
    --no-local-fallback       fail instead of falling back to local rebase/push
    --dry-run                 verify and print the plan without mutating
    --gate-worktree-path <p>  registered gate worktree path; a vanished worktree
                              fails closed before any branch-switch/conflict op
    --json                    emit machine-readable JSON (default behavior)
    -h, --help                print this help and exit 0
HELP
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
    --pr)
      [[ $# -ge 2 ]] || usage
      PR="$2"
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
    --gate-worktree-path)
      [[ $# -ge 2 ]] || usage
      GATE_WORKTREE_PATH="$2"
      shift 2
      ;;
    --json)
      shift
      ;;
    -h|--help)
      print_help
      exit 0
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

if [[ -n "$PR" && -n "$POS_PR" && "$PR" != "$POS_PR" ]]; then
  echo "error: conflicting PR numbers: pass either positional or --pr, not both" >&2
  exit 2
fi
PR="${PR:-$POS_PR}"
[[ -n "$PR" ]] || usage
[[ "$PR" =~ ^[0-9]+$ ]] || {
  echo "error: PR number must be numeric" >&2
  exit 2
}
[[ -n "$EXPECTED" ]] || {
  echo "error: --expected-head-sha is required (record the current PR head SHA first)" >&2
  exit 2
}

emit_result() {
  # $1 status, $2 updated(bool), $3 error(string), $4 method(string)
  local status="$1" updated="$2" error="${3:-}" method="${4:-}"
  python3 - "$status" "$PR" "$REPO" "$EXPECTED" "$LIVE_HEAD" "$BASE_REF" "$REMOTE" "$method" "$updated" "$error" "$SOURCE_REF_RESTORED" <<'PY'
import json
import sys

status, pr, repo, expected, live, base, remote, method, updated, error, restored = sys.argv[1:]
print(
    json.dumps(
        {
            "status": status,
            "pr": pr,
            "repo": repo,
            "expected_head_sha": expected,
            "live_head_sha": live,
            "base": base,
            "remote": remote,
            "method": method,
            "updated": updated == "true",
            "source_ref_restored": restored == "1",
            "error": error or None,
        },
        separators=(",", ":"),
    )
)
PY
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
META_OUTPUT="$(gh api "repos/${REPO}/pulls/${PR}" \
  --jq '[.head.sha, .head.ref, .base.ref, (.head.repo.full_name // ""), (.base.repo.full_name // "")] | @tsv' 2>/dev/null)"
META_RC=$?
set -e

if [[ $META_RC -ne 0 ]] || [[ -z "$META_OUTPUT" ]]; then
  emit_result "error" "false" "could not fetch PR metadata from REST" ""
  exit 2
fi

IFS=$'\t' read -r LIVE_HEAD HEAD_REF BASE_REF_RESOLVED HEAD_REPO_FULL BASE_REPO_FULL <<< "$META_OUTPUT"
if [[ -z "$LIVE_HEAD" || -z "$HEAD_REF" || -z "$BASE_REF_RESOLVED" ]]; then
  emit_result "error" "false" "PR metadata is missing head or base ref" ""
  exit 2
fi

BASE_REF="${BASE_REF_OVERRIDE:-$BASE_REF_RESOLVED}"
BASE_REF="${BASE_REF:-main}"

if [[ "$LIVE_HEAD" != "$EXPECTED" ]]; then
  emit_result "head_mismatch" "false" "PR head changed since expected SHA was recorded" ""
  exit 1
fi

# The update-branch endpoint is itself a write, so dry-run must exit before
# probing it. The metadata read and exact-head guard above remain intentional.
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'dry-run: would request GitHub update-branch for %s#%s at %s; ' \
    "$REPO" "$PR" "$EXPECTED" >&2
  printf 'if unavailable, would use the guarded local fallback via %s\n' \
    "$REMOTE" >&2
  emit_result "dry_run" "false" "" "gh_rest_update_branch"
  exit 0
fi

# --- restore a deleted PR source ref (issue #6689) ----------------------------
# A deleted head branch makes update-branch fail with "Could not resolve head
# ref". The exact-head guard above already proved that the immutable PR head
# SHA equals the expected head, so a plain (non-force) push of that SHA can
# only create the absent ref; it can never overwrite or retarget an existing
# branch. If the ref reappears concurrently, re-detect it and continue with
# the normal update path instead of pushing. A probe failure is treated as
# "ref present" so environments without a usable git keep the historical path.
set +e
PRE_RESTORE_SHA="$(git ls-remote --heads "$REMOTE" "$HEAD_REF" 2>/dev/null | awk '{print $1}' | head -n1)"
PRE_RESTORE_RC=$?
set -e
if [[ $PRE_RESTORE_RC -eq 0 ]] && [[ -z "$PRE_RESTORE_SHA" ]]; then
  if [[ "$HEAD_REF" == "$BASE_REF" ]]; then
    emit_result "source_ref_restore_failed" "false" \
      "refusing to restore the base branch itself (${HEAD_REF})" "source_ref_restore"
    exit 2
  fi
  if [[ -z "$HEAD_REPO_FULL" ]]; then
    emit_result "source_ref_restore_failed" "false" \
      "PR head repository is unknown or deleted; cannot restore refs/heads/${HEAD_REF} through ${REMOTE}" "source_ref_restore"
    exit 2
  fi
  BASE_REPO_COMPARE="${BASE_REPO_FULL:-$REPO}"
  if [[ "$HEAD_REPO_FULL" != "$BASE_REPO_COMPARE" ]]; then
    emit_result "source_ref_restore_failed" "false" \
      "cross-fork PR: head ref lives in ${HEAD_REPO_FULL}, cannot restore through ${REMOTE} (${BASE_REPO_COMPARE})" "source_ref_restore"
    exit 2
  fi
  echo "info: PR source ref refs/heads/${HEAD_REF} is missing on ${REMOTE}; restoring it at immutable PR head ${EXPECTED}" >&2
  set +e
  git fetch "$REMOTE" "$EXPECTED" >/dev/null 2>&1
  RESTORE_FETCH_RC=$?
  set -e
  if [[ $RESTORE_FETCH_RC -ne 0 ]]; then
    emit_result "source_ref_restore_failed" "false" \
      "could not fetch immutable PR head SHA ${EXPECTED} from ${REMOTE}; refusing to restore refs/heads/${HEAD_REF}" "source_ref_restore"
    exit 2
  fi
  FETCHED_RESTORE_SHA="$(git rev-parse FETCH_HEAD 2>/dev/null || true)"
  if [[ "$FETCHED_RESTORE_SHA" != "$EXPECTED" ]]; then
    emit_result "source_ref_restore_failed" "false" \
      "fetched FETCH_HEAD (${FETCHED_RESTORE_SHA:-empty}) differs from immutable PR head ${EXPECTED}; refusing to restore" "source_ref_restore"
    exit 2
  fi
  set +e
  git push "$REMOTE" "${EXPECTED}:refs/heads/${HEAD_REF}" >/dev/null 2>&1
  RESTORE_PUSH_RC=$?
  set -e
  if [[ $RESTORE_PUSH_RC -ne 0 ]]; then
    RECHECK_SHA="$(git ls-remote --heads "$REMOTE" "$HEAD_REF" 2>/dev/null | awk '{print $1}' | head -n1 || true)"
    if [[ -n "$RECHECK_SHA" ]]; then
      echo "info: refs/heads/${HEAD_REF} reappeared concurrently at ${RECHECK_SHA}; continuing with the normal update path" >&2
    else
      emit_result "source_ref_restore_failed" "false" \
        "plain push restore of refs/heads/${HEAD_REF} was rejected and the ref is still missing" "source_ref_restore"
      exit 2
    fi
  else
    RESTORED_SHA="$(git ls-remote --heads "$REMOTE" "refs/heads/${HEAD_REF}" 2>/dev/null | awk '{print $1}' | head -n1 || true)"
    if [[ "$RESTORED_SHA" != "$EXPECTED" ]]; then
      emit_result "source_ref_restore_failed" "false" \
        "post-restore verification failed: remote refs/heads/${HEAD_REF} is ${RESTORED_SHA:-empty}, expected ${EXPECTED}" "source_ref_restore"
      exit 2
    fi
    SOURCE_REF_RESTORED=1
    echo "info: restored refs/heads/${HEAD_REF} at ${EXPECTED}" >&2
  fi
fi

# --- attempt the supported remote branch-update path --------------------------
set +e
REST_STDERR="$(gh api "repos/${REPO}/pulls/${PR}/update-branch" \
  --method PUT -f "expected_head_sha=${EXPECTED}" --jq '.message' 2>&1 >/dev/null)"
REST_RC=$?
set -e

if [[ $REST_RC -eq 0 ]]; then
  emit_result "update_requested" "true" "" "gh_rest_update_branch"
  exit 0
fi

# --- gate worktree health check before any local branch-switch ----------------
# The local fallback performs a git rebase and force-with-lease push inside the
# registered gate worktree. If that worktree has vanished, fail closed and report
# the lease cleanup owner rather than dying opaquely with
# "CreateProcess ... No such file or directory" mid-rebase.
if [[ -n "$GATE_WORKTREE_PATH" ]]; then
  if [[ ! -f "$GUARD_HELPER" ]]; then
    emit_result "error" "false" "gate worktree guard helper is missing; refusing unguarded local fallback" "local_fallback"
    exit 2
  fi
  set +e
  GUARD_JSON="$(python3 "$GUARD_HELPER" verify --path "$GATE_WORKTREE_PATH" --json 2>/dev/null)"
  GUARD_RC=$?
  set -e
  if [[ -z "$GUARD_JSON" ]]; then
    emit_result "error" "false" "could not verify gate worktree before local fallback" "local_fallback"
    exit 2
  fi
  set +e
  GUARD_RESULT="$(python3 -c '
import json
import sys

try:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("guard output must be a JSON object")
    if payload.get("exists"):
        print("ok")
    else:
        print("missing:" + str(payload.get("cleanup_owner") or "unknown"))
except Exception:
    print("error")
    raise SystemExit(2)
' <<<"$GUARD_JSON")"
  PARSE_RC=$?
  set -e
  if [[ $PARSE_RC -ne 0 ]] || [[ "$GUARD_RESULT" == "error" ]]; then
    emit_result "error" "false" "could not parse gate worktree guard output before local fallback" "local_fallback"
    exit 2
  fi
  if [[ "$GUARD_RC" -ne 0 && "$GUARD_RESULT" == "ok" ]]; then
    emit_result "error" "false" "gate worktree guard failed despite reporting an existing path" "local_fallback"
    exit 2
  fi
  if [[ "$GUARD_RESULT" == missing:* ]]; then
    CLEANUP_OWNER="${GUARD_RESULT#missing:}"
    emit_result "gate_worktree_missing" "false" "registered gate worktree vanished before local branch-switch; cleanup owner: ${CLEANUP_OWNER}" "local_fallback"
    exit 1
  fi
  if [[ "$GUARD_RESULT" != "ok" ]]; then
    emit_result "error" "false" "gate worktree guard returned an unrecognized result" "local_fallback"
    exit 2
  fi
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

if [[ ! -f "$LEASE_HELPER" ]]; then
  emit_result "error" "false" "lease helper is missing; refusing unleased local fallback" "local_fallback"
  exit 2
fi
if ! python3 "$LEASE_HELPER" create --pr "$PR" >/dev/null 2>&1; then
  emit_result "error" "false" "could not acquire PR-gate lease; refusing local fallback" "local_fallback"
  exit 2
fi
LEASE_CREATED=1

set +e
git fetch "${REMOTE}" "${BASE_REF}" "${HEAD_REF}" >/dev/null 2>&1
FETCH_RC=$?
set -e
if [[ $FETCH_RC -ne 0 ]]; then
  emit_result "error" "false" "git fetch of base/head refs failed" "local_fallback"
  exit 2
fi

set +e
git rebase "${REMOTE}/${BASE_REF}" >/dev/null 2>&1
REBASE_RC=$?
set -e
if [[ $REBASE_RC -ne 0 ]]; then
  git rebase --abort >/dev/null 2>&1 || true
  emit_result "error" "false" "git rebase onto ${REMOTE}/${BASE_REF} failed (resolve conflicts manually)" "local_fallback"
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
if [[ -z "$REMOTE_SHA" ]]; then
  emit_result "error" "false" "post-push verification failed: remote SHA was empty" "local_fallback"
  exit 2
fi
if [[ "$REMOTE_SHA" != "$NEW_HEAD" ]]; then
  emit_result "error" "false" "post-push verification failed: remote ${REMOTE_SHA} != local ${NEW_HEAD}" "local_fallback"
  exit 2
fi

release_lease
emit_result "fallback_local_rebase" "true" "" "local_fallback"
exit 0
