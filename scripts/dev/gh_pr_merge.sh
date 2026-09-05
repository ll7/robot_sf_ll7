#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for callers that still use the historical merge helper.
# The receipt owner is the only component allowed to perform the merge write;
# this shell entry point only binds the requested PR head and delegates the
# report/apply lifecycle to that owner.
# Source-branch cleanup is intentionally outside this compatibility wrapper.
# The guarded merger may perform that separate post-merge action only after
# verifying that the branch contains no unique unpreserved work.
#
# Usage:
#   scripts/dev/gh_pr_merge.sh <pr-number> --match-head-commit <sha> [--repo owner/name]
#
# A full exact-head SHA is mandatory.  Repository discovery is deliberately
# limited to an explicit --repo, GH_REPO, or a GitHub origin; an ambiguous
# repository fails closed before the receipt owner is invoked.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$SCRIPT_DIR/../.."
# Delegated owner: scripts/dev/single_account_merge_receipt.py.
RECEIPT_OWNER="$SCRIPT_DIR/single_account_merge_receipt.py"
RECEIPT_OWNER_MODULE="scripts.dev.single_account_merge_receipt"

usage() {
  cat <<'EOF_USAGE'
Usage: scripts/dev/gh_pr_merge.sh <pr-number> --match-head-commit <sha> [--repo owner/name]

Checks the exact-head receipt contract and delegates report/apply to the
canonical single-account receipt owner.  This compatibility wrapper performs
no GitHub merge operation itself.  It fails closed when the receipt is not
ready, the live head differs, or the owner cannot complete its final recheck.
EOF_USAGE
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

run_receipt_owner() (
  cd "$REPOSITORY_ROOT"
  python3 -m "$RECEIPT_OWNER_MODULE" "$@"
)

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
      if [[ "$#" -lt 2 ]]; then
        printf 'ERROR: --match-head-commit requires a full 40-char SHA.\n' >&2
        exit 2
      fi
      if [[ -n "$expected_head_sha" ]]; then
        printf 'ERROR: --match-head-commit may be supplied only once.\n' >&2
        exit 2
      fi
      expected_head_sha="$2"
      shift 2
      ;;
    --repo)
      if [[ "$#" -lt 2 ]]; then
        printf 'ERROR: --repo requires owner/name.\n' >&2
        exit 2
      fi
      if [[ -n "$repo" ]]; then
        printf 'ERROR: --repo may be supplied only once.\n' >&2
        exit 2
      fi
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

if ! [[ "$pr_number" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ERROR: PR number must be a positive integer; got %q\n' "$pr_number" >&2
  exit 2
fi
if ! [[ "$expected_head_sha" =~ ^[0-9a-fA-F]{40}$ ]]; then
  printf 'ERROR: --match-head-commit requires a full 40-char SHA; got %q\n' \
    "$expected_head_sha" >&2
  exit 2
fi

if ! python3 "$SCRIPT_DIR/github_transport_policy.py" check \
  --helper "$SCRIPT_DIR/gh_pr_merge.sh" --root "$REPOSITORY_ROOT" --json >/dev/null; then
  printf 'ERROR: gh_pr_merge.sh is not admitted by the GitHub transport policy.\n' >&2
  exit 2
fi

if [[ -z "$repo" ]]; then
  repo="${GH_REPO:-}"
fi
if [[ -z "$repo" ]]; then
  repo="$(repo_from_git_remote || true)"
fi
if [[ -z "$repo" ]]; then
  printf 'ERROR: cannot resolve owner/name for the receipt-owner merge request.\n' >&2
  exit 2
fi
if ! [[ "$repo" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  printf 'ERROR: invalid owner/name for the receipt-owner merge request: %q\n' "$repo" >&2
  exit 2
fi

receipt_file="$(mktemp "${TMPDIR:-/tmp}/gh-pr-merge-receipt.XXXXXX.json")"
trap 'rm -f "$receipt_file"' EXIT

if run_receipt_owner \
  --pr "$pr_number" \
  --repo "$repo" \
  --mode report-only \
  --expected-head "$expected_head_sha" \
  --output "$receipt_file"; then
  :
else
  report_status=$?
  printf 'ERROR: canonical receipt report was not ready; merge was not attempted.\n' >&2
  exit "$report_status"
fi

if [[ ! -s "$receipt_file" ]]; then
  printf 'ERROR: canonical receipt report produced no receipt; merge was not attempted.\n' >&2
  exit 1
fi

if run_receipt_owner \
  --pr "$pr_number" \
  --repo "$repo" \
  --mode apply \
  --expected-head "$expected_head_sha" \
  --receipt-file "$receipt_file"; then
  exit 0
else
  apply_status=$?
  printf 'ERROR: canonical receipt owner refused or failed the merge.\n' >&2
  exit "$apply_status"
fi
