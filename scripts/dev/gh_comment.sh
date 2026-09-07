#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/dev/gh_comment.sh pr <number> [--repo <owner/repo>] [--body-file <path>]
  scripts/dev/gh_comment.sh issue <number> [--repo <owner/repo>] [--body-file <path>]
  scripts/dev/gh_comment.sh pr --current [--repo <owner/repo>] [--body-file <path>]

Notes:
  - If --body-file is omitted, comment body is read from stdin.
  - Prefer heredoc stdin for multiline comments to avoid literal "\n" escapes.
  - Both PR and issue comments use the REST issue-comments endpoint
    (POST repos/<owner>/<repo>/issues/<number>/comments) with REST target
    validation, so publication is independent of GraphQL comment quotas.
  - GitHub CLI calls use GH_COMMENT_TIMEOUT_SECONDS (default: 30) and fail
    closed when the finite timeout is reached.
EOF
}

DEFAULT_GH_COMMENT_TIMEOUT_SECONDS="30"
GH_COMMENT_TIMEOUT_SECONDS="${GH_COMMENT_TIMEOUT_SECONDS:-$DEFAULT_GH_COMMENT_TIMEOUT_SECONDS}"

gh_api() {
  python3 - "$GH_COMMENT_TIMEOUT_SECONDS" gh "$@" <<'PY'
import math
import os
import signal
import subprocess
import sys
import time

TIMEOUT_EXIT_CODE = 124
BACKEND_ERROR_EXIT_CODE = 125
PROCESS_GROUP_CLEANUP_GRACE_SECONDS = 5.0
PROCESS_GROUP_POLL_INTERVAL_SECONDS = 0.01


def process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_process_group_exit(process: subprocess.Popen[bytes]) -> bool:
    deadline = time.monotonic() + PROCESS_GROUP_CLEANUP_GRACE_SECONDS
    while process_group_exists(process.pid):
        process.poll()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(PROCESS_GROUP_POLL_INTERVAL_SECONDS, remaining))
    process.wait()
    return True


def terminate_process_group(process: subprocess.Popen[bytes]) -> bool:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return True
    except OSError:
        process.terminate()
        try:
            process.wait(timeout=PROCESS_GROUP_CLEANUP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return False
        return False

    if wait_for_process_group_exit(process):
        return True

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        process.kill()
        process.wait()
        return False
    return wait_for_process_group_exit(process)


def write_output(stream: object, payload: bytes | None) -> None:
    if payload:
        stream.buffer.write(payload)  # type: ignore[attr-defined]
        stream.buffer.flush()  # type: ignore[attr-defined]


def main() -> int:
    try:
        timeout_seconds = float(sys.argv[1])
    except (IndexError, ValueError):
        print(
            "gh_comment.sh: GH_COMMENT_TIMEOUT_SECONDS must be a finite positive number",
            file=sys.stderr,
        )
        return BACKEND_ERROR_EXIT_CODE
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        print(
            "gh_comment.sh: GH_COMMENT_TIMEOUT_SECONDS must be a finite positive number",
            file=sys.stderr,
        )
        return BACKEND_ERROR_EXIT_CODE

    command = sys.argv[2:]
    if not command:
        print("gh_comment.sh: bounded GitHub CLI command is missing", file=sys.stderr)
        return BACKEND_ERROR_EXIT_CODE

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError:
        print(f"gh_comment.sh: command not found: {command[0]}", file=sys.stderr)
        return 127
    except PermissionError:
        print(f"gh_comment.sh: command is not executable: {command[0]}", file=sys.stderr)
        return 126

    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        cleaned = terminate_process_group(process)
        try:
            stdout, stderr = process.communicate(timeout=PROCESS_GROUP_CLEANUP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""
            cleaned = False
        write_output(sys.stdout, stdout)
        write_output(sys.stderr, stderr)
        print(
            f"gh_comment.sh: GitHub CLI command timed out after {timeout_seconds:g} seconds",
            file=sys.stderr,
        )
        return TIMEOUT_EXIT_CODE if cleaned else BACKEND_ERROR_EXIT_CODE

    write_output(sys.stdout, stdout)
    write_output(sys.stderr, stderr)
    returncode = process.returncode
    return 128 - returncode if returncode < 0 else returncode


raise SystemExit(main())
PY
}

if [ "$#" -gt 0 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; then
  usage
  exit 0
fi

if [ "$#" -lt 1 ]; then
  usage
  exit 2
fi

target_type="$1"
shift

if [ "$target_type" != "pr" ] && [ "$target_type" != "issue" ]; then
  echo "Error: target must be 'pr' or 'issue'." >&2
  usage
  exit 2
fi

target_id=""
use_current_pr=false
repo_arg=""
body_file=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo)
      if [ "$#" -lt 2 ]; then
        echo "Error: --repo requires a value." >&2
        exit 2
      fi
      repo_arg="$2"
      shift 2
      ;;
    --body-file)
      if [ "$#" -lt 2 ]; then
        echo "Error: --body-file requires a path." >&2
        exit 2
      fi
      body_file="$2"
      shift 2
      ;;
    --current)
      use_current_pr=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    -*)
      echo "Error: unknown option '$1'." >&2
      usage
      exit 2
      ;;
    *)
      if [ -n "$target_id" ]; then
        echo "Error: unexpected extra argument '$1'." >&2
        usage
        exit 2
      fi
      target_id="$1"
      shift
      ;;
  esac
done

if [ "$target_type" = "issue" ] && [ "$use_current_pr" = true ]; then
  echo "Error: --current is only supported for 'pr' target type." >&2
  exit 2
fi

if [ "$use_current_pr" = true ]; then
  branch_name="$(git branch --show-current)"
  if [ -z "$branch_name" ]; then
    # actions/checkout intentionally leaves CI jobs detached.  Prefer the PR
    # source ref, then the checked-out event ref, so REST lookup remains
    # deterministic without manufacturing a branch in the worktree.
    branch_name="${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-}}"
  fi
  if [ -z "$branch_name" ]; then
    echo "Error: could not resolve the current branch for --current." >&2
    exit 1
  fi
  # Review leases use a local branch name distinct from the PR source branch
  # while tracking that source branch. Query the tracked branch when available
  # so --current still finds the PR from an isolated review worktree.
  if upstream_ref="$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null)"; then
    branch_name="${upstream_ref#*/}"
  fi
  api_repo="{owner}/{repo}"
  head_owner="{owner}"
  if [ -n "$repo_arg" ]; then
    api_repo="$repo_arg"
    head_owner="${repo_arg%%/*}"
  fi
  if target_id="$(gh_api api "repos/$api_repo/pulls?state=open&head=$head_owner:$branch_name&per_page=100" --jq '.[0].number // empty')"; then
    :
  else
    gh_api_rc=$?
    if [[ "$gh_api_rc" -eq 124 || "$gh_api_rc" -eq 125 ]]; then
      exit "$gh_api_rc"
    fi
    echo "Error: could not resolve an open PR for branch '$branch_name'." >&2
    exit 1
  fi
fi

if [ -z "$target_id" ]; then
  echo "Error: missing target number. Provide <number> or use --current for PRs." >&2
  usage
  exit 2
fi

if [ -z "$body_file" ]; then
  body_file="$(mktemp)"
  trap 'rm -f "$body_file"' EXIT
  cat >"$body_file"
fi

if [ ! -f "$body_file" ]; then
  echo "Error: body file '$body_file' does not exist." >&2
  exit 2
fi

if [ ! -s "$body_file" ]; then
  echo "Error: comment body is empty." >&2
  exit 2
fi

if ! python3 "$SCRIPT_DIR/github_transport_policy.py" check \
  --helper "$SCRIPT_DIR/gh_comment.sh" --root "$SCRIPT_DIR/../.." --json >/dev/null; then
  echo "Error: gh_comment.sh is not admitted by the GitHub transport policy." >&2
  exit 2
fi

if [ "$target_type" = "pr" ]; then
  api_repo="{owner}/{repo}"
  if [ -n "$repo_arg" ]; then
    api_repo="$repo_arg"
  fi
  gh_api_rc=0
  gh_api api "repos/$api_repo/pulls/$target_id" --silent || gh_api_rc=$?
  if [[ "$gh_api_rc" -ne 0 ]]; then
    if [[ "$gh_api_rc" -eq 124 || "$gh_api_rc" -eq 125 ]]; then
      exit "$gh_api_rc"
    fi
    echo "Error: PR '$target_id' could not be resolved through the REST API." >&2
    exit 1
  fi
  # Use gh's silent mode so a successful POST with an empty/malformed response
  # body cannot surface as a client-side JSON parse failure (issue #6891).
  gh_api_rc=0
  gh_api api --method POST "repos/$api_repo/issues/$target_id/comments" \
    --silent -F "body=@$body_file" || gh_api_rc=$?
  exit "$gh_api_rc"
else
  api_repo="{owner}/{repo}"
  if [ -n "$repo_arg" ]; then
    api_repo="$repo_arg"
  fi
  # Mirror the PR path: validate the target through REST before publication,
  # then post through the REST issue-comments endpoint. This keeps the issue
  # path independent of the GraphQL-backed ``gh issue comment`` command, which
  # fails under exhausted GraphQL comment quota even when REST is available.
  gh_api_rc=0
  gh_api api "repos/$api_repo/issues/$target_id" --silent || gh_api_rc=$?
  if [[ "$gh_api_rc" -ne 0 ]]; then
    if [[ "$gh_api_rc" -eq 124 || "$gh_api_rc" -eq 125 ]]; then
      exit "$gh_api_rc"
    fi
    echo "Error: Issue '$target_id' could not be resolved through the REST API." >&2
    exit 1
  fi
  # Use gh's silent mode so a successful POST with an empty/malformed response
  # body cannot surface as a client-side JSON parse failure (issue #6891).
  gh_api_rc=0
  gh_api api --method POST "repos/$api_repo/issues/$target_id/comments" \
    --silent -F "body=@$body_file" || gh_api_rc=$?
  exit "$gh_api_rc"
fi
