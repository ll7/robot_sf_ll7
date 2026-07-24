#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/dev/pr_ready_check.sh [--help | -h]

Runs PR readiness gates: ruff format/fix, pytest, coverage checks,
docstring-todo diff/ratchet checks, and freshness stamp write.

Environment variables:
  BASE_REF            Base ref to compare against for coverage and freshness
                      (default: origin/main)
  PR_READY_MODE       Set to "final" for proof-on-committed-HEAD mode, or
                      "interim" for work-in-progress feedback.
                      Overrides PR_READY_FINAL when set.
  PR_READY_FINAL      Legacy compatibility flag: "1", "true", "yes", or "on"
                      for final mode; "0", "false", "no", "off" for interim.
                      Ignored when PR_READY_MODE is set.
  PR_READY_SKIP_PREFLIGHT  Set to "1" to skip cheap preflight checks for
                      test-collection dependencies and the bundled fast-pysf API.
  PR_READY_PR_BODY_FILE  Optional markdown PR body from an existing readable
                      regular file.
                      Process-substitution paths such as /dev/fd/63 are rejected
                      because their descriptors do not survive into downstream
                      readiness processes. When set, readiness checks that
                      deferred work has a linked issue or explicit NA.
  PR_READY_REQUIRE_OPEN_FOLLOWUP_ISSUES
                      Set to "0" to skip open-state verification for linked
                      follow-up issues when PR_READY_PR_BODY_FILE is set.
  PR_READY_SERIAL_FALLBACK
                      Set to "1" to auto-rerun serially when parallel pytest
                      workers crash (issue #5633), separating an env crash from
                      real failures. Disabled by default; the gate stays
                      fail-closed and only reports the crash diagnostic.
EOF
}

if [ "$#" -gt 0 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

export BASE_REF="${BASE_REF:-origin/main}"
export MIN_COVERAGE="${MIN_COVERAGE:-80}"
export GOAL_COVERAGE="${GOAL_COVERAGE:-100}"
export PR_READY_FINAL="${PR_READY_FINAL:-0}"
export PR_READY_MODE="${PR_READY_MODE:-}"
export PR_READY_SKIP_PREFLIGHT="${PR_READY_SKIP_PREFLIGHT:-0}"
export PR_READY_PR_BODY_FILE="${PR_READY_PR_BODY_FILE:-}"
export PR_READY_REQUIRE_OPEN_FOLLOWUP_ISSUES="${PR_READY_REQUIRE_OPEN_FOLLOWUP_ISSUES:-1}"

validate_pr_body_file() {
  local body_file="$PR_READY_PR_BODY_FILE"
  [[ -z "$body_file" ]] && return

  if [[ -f "$body_file" && -r "$body_file" ]]; then
    return
  fi

  printf 'PR_READY_PR_BODY_FILE must name an existing readable regular file; received %q.\n' \
    "$body_file" >&2
  case "$body_file" in
    /dev/fd/*|/proc/*/fd/*)
      printf 'Process-substitution paths are not supported because their file descriptors are not inherited by readiness subprocesses.\n' >&2
      printf 'Write the body to a persistent file first, for example:\n' >&2
      printf '  body_file="%s"; gh pr view ... --json body --jq .body >"%s"; PR_READY_PR_BODY_FILE="%s" %q\n' \
        '$(mktemp)' '$body_file' '$body_file' "$0" >&2
      ;;
    *)
      printf 'Create the file first, then rerun with PR_READY_PR_BODY_FILE=/path/to/body.md.\n' >&2
      ;;
  esac
  exit 2
}

validate_pr_body_file

worktree_state() {
  if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    printf 'dirty\n'
  else
    printf 'clean\n'
  fi
}

print_compact_worktree_status() {
  local max_lines=30
  local tracked_status
  tracked_status=$(git status --short --untracked-files=no)
  if [[ -n "$tracked_status" ]]; then
    printf 'Tracked or staged changes:\n' >&2
    printf '%s\n' "$tracked_status" | head -n "$max_lines" >&2
    local total_lines
    total_lines=$(printf '%s\n' "$tracked_status" | wc -l)
    if (( total_lines > max_lines )); then
      printf '... truncated (%d more tracked lines)\n' "$((total_lines - max_lines))" >&2
    fi
  fi
  local generated_path
  local generated_present=()
  for generated_path in .venv .opencode node_modules output .pytest_cache __pycache__; do
    if [[ -e "$generated_path" ]]; then
      generated_present+=("$generated_path")
    fi
  done
  if [[ ${#generated_present[@]} -gt 0 ]]; then
    printf 'Generated paths present (contents not enumerated):\n' >&2
    printf '%s\n' "${generated_present[@]}" >&2
  fi
  local porcelain_count
  porcelain_count=$(git status --porcelain --untracked-files=normal | wc -l)
  if (( porcelain_count > 0 )); then
    printf 'Full untracked inventory omitted from this message (%d porcelain lines detected).\n' "$porcelain_count" >&2
    printf 'Run `git status --short --untracked-files=normal` only when a full inventory is needed.\n' >&2
  elif [[ -z "$tracked_status" && ${#generated_present[@]} -eq 0 ]]; then
    printf 'No tracked changes or known generated paths detected.\n' >&2
  fi
}

resolve_base_ref() {
  # Ensure BASE_REF resolves to a real commit before any `git diff
  # "$BASE_REF...HEAD"` comparison or downstream consumer (perf/coverage/
  # freshness checks) uses it. On fresh checkouts where the default
  # origin/main has not been fetched, the unresolved ref otherwise leaks a
  # raw `fatal: ambiguous argument` from git: the comparison sits inside a
  # process substitution, so `set -e` silently swallows the failure and the
  # script proceeds with an empty changed-file set and an invalid base ref.
  if git rev-parse --verify --quiet "${BASE_REF}^{commit}" >/dev/null 2>&1; then
    return 0
  fi

  printf 'BASE_REF %q does not resolve to a local commit.\n' "$BASE_REF" >&2

  # Best-effort fetch for remote-tracking refs (e.g. origin/main) so a network
  # checkout can still compare against the intended base before falling back.
  if [[ "$BASE_REF" == */* ]]; then
    local remote="${BASE_REF%%/*}"
    local branch="${BASE_REF#*/}"
    if git remote get-url "$remote" >/dev/null 2>&1; then
      printf 'Attempting git fetch --quiet %q %q to resolve BASE_REF.\n' "$remote" "$branch" >&2
      if git fetch --quiet "$remote" "$branch" >/dev/null 2>&1 \
        && git rev-parse --verify --quiet "${BASE_REF}^{commit}" >/dev/null 2>&1; then
        printf 'Resolved BASE_REF %q after fetch.\n' "$BASE_REF" >&2
        return 0
      fi
    fi
  fi

  # Fall back to HEAD so readiness gates still run instead of crashing. On a
  # fresh checkout this yields an empty changed-file set rather than a fatal
  # git error, and keeps a valid ref flowing to the coverage/perf/freshness
  # checks that also consume BASE_REF.
  printf 'Falling back to BASE_REF=HEAD; changed-file gates will compare against HEAD.\n' >&2
  BASE_REF="HEAD"
  export BASE_REF
}

is_optional_readiness_path() {
  # Code paths and test directory patterns that require optional extras
  case "$1" in
    robot_sf/benchmark/*|\
    robot_sf/baselines/drl_vo.py|\
    robot_sf/feature_extractor.py|\
    robot_sf/feature_extractors/*|\
    robot_sf/planner/*|\
    robot_sf/training/*|\
    robot_sf_carla_bridge/*|\
    scripts/multi_extractor_training.py|\
    scripts/tools/benchmark_feature_extractors.py|\
    scripts/tools/probe_social_navigation_pyenvs_socialforce_runtime.py|\
    scripts/tools/probe_sonic_model_inference.py|\
    scripts/training/*|\
    tests/benchmark/*|\
    tests/benchmark_full/*|\
    tests/carla_bridge/*|\
    tests/integration/*|\
    tests/planner/*|\
    tests/render/*|\
    tests/training/*|\
    tests/visuals/*)
      return 0
      ;;
  esac

  # Test paths from the single source of truth
  local allowlist_file="${SCRIPT_DIR}/../../tests/support/optional_test_allowlist.txt"
  if [[ ! -f "$allowlist_file" ]]; then
    printf 'Error: optional test allowlist file not found: %s\n' "$allowlist_file" >&2
    exit 1
  fi

  local test_path="$1"
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    # Remove trailing slash for pattern matching
    line="${line%/}"
    # Check if test_path matches the pattern
    if [[ "$test_path" == "$line" || "$test_path" == "$line"/* ]]; then
      return 0
    fi
  done < "$allowlist_file"

  return 1
}

pr_ready_mode_lower=$(printf '%s' "$PR_READY_MODE" | tr '[:upper:]' '[:lower:]')
case "$pr_ready_mode_lower" in
  "")
    pr_ready_final_lower=$(printf '%s' "$PR_READY_FINAL" | tr '[:upper:]' '[:lower:]')
    case "$pr_ready_final_lower" in
      1|true|yes|on) pr_ready_final=1 ;;
      0|false|no|off) pr_ready_final=0 ;;
      *)
        printf 'Invalid PR_READY_FINAL value %q (expected 1|0|true|false|yes|no|on|off).\n' "$PR_READY_FINAL" >&2
        exit 2
        ;;
    esac
    ;;
  final) pr_ready_final=1 ;;
  interim) pr_ready_final=0 ;;
  *)
    printf 'Invalid PR_READY_MODE value %q (expected final|interim).\n' "$PR_READY_MODE" >&2
    exit 2
    ;;
esac

# Friction guard for issue #5533: untracked new files are invisible to the
# committed-HEAD diff gates (changed-file coverage, docstring TODO diff). They
# previously produced a misleading "No changed files vs BASE_REF" while silently
# omitting the only changed code. Fail clearly so the wrapper never reports
# changed-file proof it cannot produce. Staging/committing the new files lets the
# diff-scoped gates see them; this is the intended workaround from the issue.
# This runs before the final clean-tree guard so both modes fail with the same
# explicit, actionable message rather than a generic "clean tree required".
untracked_new_files=()
while IFS= read -r untracked_path; do
  [[ -z "$untracked_path" ]] && continue
  untracked_new_files+=("$untracked_path")
done < <(git ls-files --others --exclude-standard)

if [[ ${#untracked_new_files[@]} -gt 0 ]]; then
  max_untracked_paths=30
  printf 'Changed-file proof cannot see the following untracked new files:\n' >&2
  printf '%s\n' "${untracked_new_files[@]:0:max_untracked_paths}" >&2
  if (( ${#untracked_new_files[@]} > max_untracked_paths )); then
    printf '... truncated (%d more untracked files)\n' "$(( ${#untracked_new_files[@]} - max_untracked_paths ))" >&2
  fi
  printf 'Changed-file gates (diff vs %s) omit untracked files, so readiness cannot prove them.\n' "$BASE_REF" >&2
  printf 'Stage/commit the new files, then rerun for changed-file proof (issue #5533).\n' >&2
  printf 'Recorded interim readiness from a non-ignored worktree with untracked new files is not proof.\n' >&2
  exit 2
fi

if [[ "$pr_ready_final" == "1" && "$(worktree_state)" != "clean" ]]; then
  printf 'Final PR readiness requires a clean non-ignored worktree before validation.\n' >&2
  printf 'Commit or remove local changes, or run without PR_READY_MODE=final for interim feedback.\n' >&2
  print_compact_worktree_status
  exit 2
fi

if [[ "$pr_ready_final" == "1" ]]; then
  preflight_check_test_deps
  preflight_check_fast_pysf
fi

resolve_base_ref

if [[ "$pr_ready_final" != "1" && "$(worktree_state)" != "clean" ]]; then
  dirty_paths=()
  while IFS= read -r dirty_path; do
    [[ -z "$dirty_path" ]] && continue
    dirty_paths+=("$dirty_path")
  done < <(
    {
      git diff --name-only HEAD
      git ls-files --others --exclude-standard
    } | LC_ALL=C sort -u
  )

  printf 'Interim changed-file scope is committed HEAD vs %s.\n' "$BASE_REF" >&2
  if [[ ${#dirty_paths[@]} -gt 0 ]]; then
    printf 'Dirty paths excluded from diff-scoped gates:\n' >&2
    max_dirty_paths=30
    printf '%s\n' "${dirty_paths[@]:0:max_dirty_paths}" >&2
    if (( ${#dirty_paths[@]} > max_dirty_paths )); then
      printf '... truncated (%d more dirty paths)\n' "$(( ${#dirty_paths[@]} - max_dirty_paths ))" >&2
    fi
  fi
fi

changed_files=()
core_changed_files=()
optional_changed_files=()
while IFS= read -r changed_file; do
  [[ -z "$changed_file" ]] && continue
  changed_files+=("$changed_file")
  if is_optional_readiness_path "$changed_file"; then
    optional_changed_files+=("$changed_file")
  else
    core_changed_files+=("$changed_file")
  fi
done < <(git diff --name-only --diff-filter=ACMRT "$BASE_REF...HEAD")

# Validate that every changed test file under tests/ or fast-pysf/tests/ is classified
# and covered by the optional test allowlist if classified as optional.
for changed_file in "${changed_files[@]}"; do
  if [[ "$changed_file" == tests/*.py || "$changed_file" == fast-pysf/tests/*.py ]]; then
    test_basename="$(basename "$changed_file")"
    if [[ "$test_basename" == test_*.py || "$test_basename" == *_test.py ]]; then
      if is_optional_readiness_path "$changed_file"; then
        allowlist_file="${SCRIPT_DIR}/../../tests/support/optional_test_allowlist.txt"
        is_in_allowlist=0
        if [[ -f "$allowlist_file" ]]; then
          while IFS= read -r line || [[ -n "$line" ]]; do
            [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
            line="${line%/}"
            if [[ "$changed_file" == "$line" || "$changed_file" == "$line"/* ]]; then
              is_in_allowlist=1
              break
            fi
          done < "$allowlist_file"
        fi
        if [[ "$is_in_allowlist" -eq 0 ]]; then
          printf 'Error: Changed test path %q is classified for lane %q but is omitted from optional test allowlist.\n' \
            "$changed_file" "optional" >&2
          printf 'Expected lane: optional\n' >&2
          printf 'Actual collection decision: omitted\n' >&2
          printf 'Remediation: Add %q to tests/support/optional_test_allowlist.txt.\n' "$changed_file" >&2
          exit 2
        fi
      fi
    fi
  fi
done

# The existing readiness lanes intentionally exclude deleted paths from their
# optional-test classification.  The base-drift guard has a stronger contract:
# every path changed by the PR must participate in the intersection, including a
# path deleted by the PR, because main can modify that path during the run.
drift_changed_files=()
while IFS= read -r changed_file; do
  [[ -z "$changed_file" ]] && continue
  drift_changed_files+=("$changed_file")
done < <(git diff --name-only --diff-filter=ACDMRT "$BASE_REF...HEAD")

if [[ ${#changed_files[@]} -gt 0 ]]; then
  if [[ "$pr_ready_final" == "1" ]]; then
    printf 'Changed files vs %s:\n' "$BASE_REF" >&2
  else
    printf 'Committed changed files vs %s:\n' "$BASE_REF" >&2
  fi
  printf '%s\n' "${changed_files[@]}" >&2
fi
if [[ ${#core_changed_files[@]} -gt 0 ]]; then
  printf 'Core-ready changed files:\n' >&2
  printf '%s\n' "${core_changed_files[@]}" >&2
fi
if [[ ${#optional_changed_files[@]} -gt 0 ]]; then
  printf 'Optional-extra changed files requiring the predictive lane:\n' >&2
  printf '%s\n' "${optional_changed_files[@]}" >&2
fi

# Issue #5782: capture the concrete base SHA this readiness run validates against
# BEFORE the expensive lanes start, so a later base-drift recheck can compare the
# moving base to the SHA we actually proved the PR against.  When BASE_REF does not
# resolve locally (handled by resolve_base_ref's HEAD fallback), leave VALIDATED_BASE_SHA
# empty so the drift check and freshness stamp fall back to the base_ref string.
VALIDATED_BASE_SHA="$(git rev-parse --verify --quiet "${BASE_REF}^{commit}" 2>/dev/null || true)"
if [[ -n "$VALIDATED_BASE_SHA" ]]; then
  printf 'Validated base SHA for this run: %s\n' "${VALIDATED_BASE_SHA:0:8}" >&2
fi

followup_args=()
if [[ "$pr_ready_final" == "1" ]]; then
  followup_args+=(--require-body)
fi
uv run python "$SCRIPT_DIR/check_pr_followups.py" "${followup_args[@]}"
perf_args=(--base-ref "$BASE_REF")
if [[ "$pr_ready_final" == "1" ]]; then
  perf_args+=(--require-body)
else
  # Interim runs only advise: a WIP perf branch may not yet carry its evidence section.
  perf_args+=(--advisory)
fi
uv run python "$SCRIPT_DIR/check_perf_evidence.py" "${perf_args[@]}"
uv run python "$SCRIPT_DIR/check_fast_results_claim_map.py"
"$SCRIPT_DIR/ruff_fix_format.sh"
printf 'Running core readiness lane.\n' >&2
ROBOT_SF_PYTEST_COVERAGE=1 ROBOT_SF_TEST_LANE=core "$SCRIPT_DIR/run_tests_parallel.sh" --lane core
if [[ ${#optional_changed_files[@]} -gt 0 ]]; then
  printf 'Running optional-extra lane for predictive/optional changed files.\n' >&2
  optional_pytest_addopts="${PYTEST_ADDOPTS:-}"
  if [[ " $optional_pytest_addopts " != *" --cov-append "* ]]; then
    optional_pytest_addopts="${optional_pytest_addopts:+$optional_pytest_addopts }--cov-append"
  fi
  PYTEST_ADDOPTS="$optional_pytest_addopts" ROBOT_SF_PYTEST_COVERAGE=1 ROBOT_SF_TEST_LANE=optional "$SCRIPT_DIR/run_tests_parallel.sh" --lane optional
else
  if [[ "$pr_ready_final" == "1" ]]; then
    printf 'No changed files require the optional-extra lane.\n' >&2
  else
    printf 'No committed changed files require the optional-extra lane.\n' >&2
  fi
fi
"$SCRIPT_DIR/check_changed_coverage.sh"
"$SCRIPT_DIR/check_docstring_todos_diff.sh"
"$SCRIPT_DIR/check_docstring_todos_ratchet.sh"
"$SCRIPT_DIR/check_docstring_todos_baseline_freshness.sh"
uv run python "$SCRIPT_DIR/check_optional_import_pr_freshness.py" --base-ref "$BASE_REF"
uv run python "$SCRIPT_DIR/../validation/check_broad_exceptions.py"

# Orphan guard for issue #5594: the readiness command must leave no worktree-owned
# pytest worker behind. A leaked planner-step worker (or any pytest descendant) that
# outlives the controller and its parents becomes reparented to PID 1 while still
# holding host resources; that orphans the shared host. We scan for reparented
# (parent PID 1) pytest processes whose cwd/cmdline ties them to this worktree.
check_no_orphaned_pytest_workers() {
  local worktree_root
  worktree_root="$(git rev-parse --show-toplevel 2>/dev/null)" || return 0
  local orphaned=0
  local pid ppid cmdline cwd_link
  for proc_dir in /proc/[0-9]*; do
    pid="${proc_dir#/proc/}"
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    ppid=$(awk '{print $4}' "/proc/$pid/stat" 2>/dev/null) || continue
    [[ "$ppid" == "1" ]] || continue
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
    [[ "$cmdline" == *"pytest"* ]] || continue
    cwd_link=$(readlink "/proc/$pid/cwd" 2>/dev/null) || continue
    if [[ "$cwd_link" == "$worktree_root"* || "$cmdline" == *"$worktree_root"* ]]; then
      printf 'Orphaned pytest worker still attached to this worktree: PID %s (cwd=%s cmdline=%s)\n' \
        "$pid" "$cwd_link" "$cmdline" >&2
      orphaned=1
    fi
  done
  if (( orphaned )); then
    printf 'Readiness left an orphaned pytest worker on the shared host (issue #5594).\n' >&2
    printf 'Terminate the listed PID(s) and fix the leaking test/worker before reporting success.\n' >&2
    return 1
  fi
  return 0
}

freshness_args=(write --base-ref "$BASE_REF")
if [[ -n "$VALIDATED_BASE_SHA" ]]; then
  freshness_args+=(--base-sha "$VALIDATED_BASE_SHA")
fi
if [[ "$pr_ready_final" == "1" ]]; then
  freshness_args+=(--require-clean-tree)
elif [[ "$(worktree_state)" != "clean" ]]; then
  printf 'Warning: recording interim PR readiness from a dirty non-ignored worktree.\n' >&2
  printf 'Interim readiness is not changed-file proof for the dirty paths listed above.\n' >&2
  printf 'Use PR_READY_MODE=final BASE_REF=%q %q for final committed-HEAD PR proof.\n' "$BASE_REF" "$0" >&2
fi

# Issue #5782: final base-drift recheck immediately before recording the stamp.
# The base may have advanced during the long lanes; if the drift intersects the
# PR's changed files, fail closed and name the exact base SHA to revalidate.  If
# drift is unrelated to the changed paths, the check recommends reuse (exit 0) so
# a passing run is not needlessly re-run; the reuse message is surfaced so the
# decision is reviewable.  Either way the resolved SHA is recorded in the stamp so
# a later freshness status check still catches any residual drift.
if [[ -n "$VALIDATED_BASE_SHA" ]]; then
  changed_files_tmp="$(mktemp)"
  if [[ ${#drift_changed_files[@]} -gt 0 ]]; then
    printf '%s\n' "${drift_changed_files[@]}" > "$changed_files_tmp"
  fi
  # ``set -e`` would exit on the failing command substitution before ``drift_rc`` is
  # captured, so append ``|| drift_rc=$?`` to record the exit code without aborting.
  drift_rc=0
  drift_msg="$(uv run python "$SCRIPT_DIR/check_base_drift.py" \
    --base-ref "$BASE_REF" \
    --validated-base-sha "$VALIDATED_BASE_SHA" \
    --changed-files "$changed_files_tmp")" || drift_rc=$?
  rm -f "$changed_files_tmp"
  case "$drift_rc" in
    0)
      # rc 0 covers both "current" (no drift) and "reuse_recommended" (drift
      # unrelated to the PR's changed paths).  Surface the reuse message so the
      # reviewable reuse path is visible, then record the stamp as-is.
      if [[ -n "$drift_msg" ]]; then
        printf '%s\n' "$drift_msg" >&2
      fi
      ;;
    1)
      # Drift intersects the PR's changed files: the validated base no longer
      # matches reality, so the stamp would be misleading.  Fail closed and name
      # the exact base to revalidate.
      printf '%s\n' "$drift_msg" >&2
      printf 'Base drifted during readiness lanes and touches PR-changed files; ' >&2
      printf 'revalidate against %s before recording the stamp (issue #5782).\n' "$BASE_REF" >&2
      exit 1
      ;;
    *)
      # Indeterminate (rc 2): the base ref could not be resolved/compared (e.g.
      # offline checkout).  Do not block on a check that cannot run; record the
      # stamp and report the indeterminate result so it is not silently ignored.
      printf 'Final base-drift recheck returned rc=%d; recording stamp regardless (issue #5782).\n' \
        "$drift_rc" >&2
      ;;
  esac
fi

uv run python "$SCRIPT_DIR/pr_ready_freshness.py" "${freshness_args[@]}"

# Final orphan guard (issue #5594): after all lanes exit, assert no worktree-owned
# pytest worker survived the controller. A leaked planner-step worker reparented to
# PID 1 would otherwise sit orphaned on the shared host.
check_no_orphaned_pytest_workers
