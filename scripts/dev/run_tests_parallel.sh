#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: scripts/dev/run_tests_parallel.sh [wrapper-options] [pytest-args...]

Runs pytest in parallel (`-n auto`) with fail-fast defaults for local triage.

Wrapper options:
  --fast-fail      Stop on first failure (`-x`) [default]
  --no-fast-fail   Disable fail-fast
  --failed-first   Run previously failed tests first [default]
  --new-first      Run tests from new files first
  --no-ordering    Disable ordering hints
  --lane core|optional|all
                   Select the pytest collection lane. Core defaults to a
                   dependency-minimal path set; optional defaults to
                   optional-extra paths; all preserves the existing behavior.
  -h, --help       Show this help message

Environment overrides:
  ROBOT_SF_PYTEST_COVERAGE=1
    Explicit opt-in for coverage output when running this wrapper. Sharded
    coverage also requires a unique COVERAGE_FILE per shard for later combine.
  ROBOT_SF_SHARD_INCLUDE_SLOW=1
    Include slow tests in a pytest-split shard. The default sharded lane keeps
    excluding slow tests for pull-request and local fast feedback.
  ROBOT_SF_TEST_LANE=core|optional|all
  COVERAGE_FILE=<path>
  PYTEST_FAST_FAIL=1|0
  PYTEST_XDIST_DIST=load|worksteal|loadscope|loadfile|loadgroup
  PYTEST_ORDER_MODE=failed-first|new-first|none
  PYTEST_NUM_WORKERS=<int>|auto
  PYTEST_SHARD_COUNT=<int>
    pytest-split shard count (default 1). When >1, runs a disjoint subset via
    `--splits N --group G` and disables coverage (partial shard data cannot be
    combined in the single-job CI topology).
  PYTEST_SHARD_INDEX=<int>
    pytest-split shard index in 1..PYTEST_SHARD_COUNT (default 1).
  PR_READY_SERIAL_FALLBACK=1|0
    Opt-in fallback for issue #5633: when the parallel run crashes (e.g.
    native-extension segfaults under xdist), rerun serially with a single
    worker to separate an environment crash from real test failures. Disabled
    by default so the gate stays fail-closed: an environment crash is neither
    success nor a silently skipped validation.

Examples:
  scripts/dev/run_tests_parallel.sh
  PYTEST_XDIST_DIST=worksteal scripts/dev/run_tests_parallel.sh
  scripts/dev/run_tests_parallel.sh --new-first tests/benchmark
  scripts/dev/run_tests_parallel.sh --no-fast-fail tests
EOF
}

if [[ "$#" -gt 0 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  show_help
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

fast_fail="${PYTEST_FAST_FAIL:-1}"
dist_mode="${PYTEST_XDIST_DIST:-load}"
order_mode="${PYTEST_ORDER_MODE:-failed-first}"
worker_override="${PYTEST_NUM_WORKERS:-}"
lane_mode="${ROBOT_SF_TEST_LANE:-all}"
core_test_paths=(
  tests/adversarial
  tests/analysis_workbench
  tests/common
  tests/contract
  tests/factories
  tests/guard
  tests/ped_npc
  tests/prediction
  tests/scenario_certification
  tests/sensor
  tests/sim
  tests/unit
  tests/dev
  tests/test_action_adapters.py
  tests/test_config_validation.py
  tests/test_environment_factory_signatures.py
  tests/test_error_policy.py
  tests/test_planner.py
  tests/test_range_sensor.py
  tests/test_seed_utils.py
  tests/test_socnav_observation.py
  tests/test_types.py
  tests/test_ci_script_contract.py
  tests/map_test.py
  tests/navigation_test.py
  tests/ped_grouping_test.py
  tests/sim_config_test.py
  tests/unicycle_drive_test.py
  tests/zone_sampling_test.py
)
# Optional test allowlist is loaded from the single source of truth
# tests/support/optional_test_allowlist.txt
allowlist_file="${SCRIPT_DIR}/../../tests/support/optional_test_allowlist.txt"
if [[ ! -f "$allowlist_file" ]]; then
  echo "Error: optional test allowlist file not found: $allowlist_file" >&2
  exit 1
fi
optional_test_paths=()
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    # Remove trailing slash for shell pattern matching
    line="${line%/}"
    optional_test_paths+=("$line")
done < "$allowlist_file"
normalize_pytest_target_path() {
  local path="${1#./}"
  path="${path%%::*}"
  printf '%s\n' "$path"
}

is_optional_test_path() {
  local path
  path="$(normalize_pytest_target_path "$1")"

  # Check against the loaded optional_test_paths array
  for pattern in "${optional_test_paths[@]}"; do
    # Directory pattern: check if path is the directory or under it
    # Exact file match: path equals pattern
    if [[ "$path" == "$pattern" || "$path" == "$pattern"/* ]]; then
      return 0
    fi
  done

  return 1
}

pytest_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast-fail)
      fast_fail="1"
      shift
      ;;
    --no-fast-fail)
      fast_fail="0"
      shift
      ;;
    --failed-first)
      order_mode="failed-first"
      shift
      ;;
    --new-first)
      order_mode="new-first"
      shift
      ;;
    --no-ordering)
      order_mode="none"
      shift
      ;;
    --lane)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "--lane requires a non-empty value." >&2
        exit 2
      fi
      lane_mode="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      pytest_args+=("$1")
      shift
      ;;
  esac
done

case "$dist_mode" in
  load|worksteal|loadscope|loadfile|loadgroup)
    ;;
  *)
    echo "Invalid PYTEST_XDIST_DIST value '$dist_mode' (expected load|worksteal|loadscope|loadfile|loadgroup)." >&2
    exit 2
    ;;
esac

case "$lane_mode" in
  all|core|optional)
    ;;
  *)
    echo "Invalid ROBOT_SF_TEST_LANE value '$lane_mode' (expected all|core|optional)." >&2
    exit 2
    ;;
esac

echo "Resolved pytest-xdist distribution mode: $dist_mode" >&2
echo "Resolved pytest lane: $lane_mode" >&2

requested_args=()
if [[ -n "$worker_override" ]]; then
  requested_args=(--requested "$worker_override")
fi

worker_spec="$(
  uv run python "$SCRIPT_DIR/resolve_pytest_workers.py" ${requested_args[@]+"${requested_args[@]}"} \
    --show-reason 2> >(cat >&2)
)"
if [[ -z "$worker_spec" ]]; then
  echo "Failed to resolve pytest worker count." >&2
  exit 2
fi

echo "Resolved pytest-xdist workers: $worker_spec" >&2

# Fallback for issue #5633: when parallel pytest workers crash on a
# native-extension/segfault combination (not an ordinary assertion failure),
# a serial rerun separates the environment crash from real test failures.
# Default is fail-closed: only an explicit opt-in reruns serially. The
# diagnostic itself (issue #5633) captures the crash signature and runtime
# fingerprint so the crash is never misread as a passed or skipped gate.
serial_fallback="${PR_READY_SERIAL_FALLBACK:-0}"
case "$serial_fallback" in
  1|true|yes|on) serial_fallback=1 ;;
  0|false|no|off) serial_fallback=0 ;;
  *)
    echo "Invalid PR_READY_SERIAL_FALLBACK value '$serial_fallback' (expected 1|0|true|false|yes|no|on|off)." >&2
    exit 2
    ;;
esac

cmd=(uv run pytest -n "$worker_spec" --dist "$dist_mode")

# pytest-split sharding: when CI provisions multiple shards, run a disjoint
# subset per shard so the suite parallelizes across runners. Main CI may opt in
# to complete cross-shard coverage by assigning one unique COVERAGE_FILE to
# every shard and combining the data in a downstream job.
shard_count="${PYTEST_SHARD_COUNT:-1}"
shard_index="${PYTEST_SHARD_INDEX:-1}"
sharding_active="0"
if [[ "$shard_count" =~ ^[0-9]+$ ]] && [[ "$shard_count" -gt 1 ]]; then
  if ! [[ "$shard_index" =~ ^[0-9]+$ ]] || [[ "$shard_index" -lt 1 ]] || [[ "$shard_index" -gt "$shard_count" ]]; then
    echo "PYTEST_SHARD_INDEX must be an integer in 1..$shard_count when PYTEST_SHARD_COUNT>1 (got '$shard_index')." >&2
    exit 2
  fi
  sharding_active="1"
  cmd+=("--splits" "$shard_count" "--group" "$shard_index")
  echo "Resolved pytest-split shard: group $shard_index of $shard_count" >&2
fi

# Fast PR/local lane: sharding excludes slow tests unless the caller explicitly
# opts into the complete suite. Main CI uses that opt-in for its four shards.
include_slow="${ROBOT_SF_SHARD_INCLUDE_SLOW:-0}"
case "$include_slow" in
  1|true|yes|on) include_slow=1 ;;
  0|false|no|off) include_slow=0 ;;
  *)
    echo "Invalid ROBOT_SF_SHARD_INCLUDE_SLOW value '$include_slow' (expected 1|0|true|false|yes|no|on|off)." >&2
    exit 2
    ;;
esac
has_marker="0"
for pytest_arg in "${pytest_args[@]}"; do
  if [[ "$pytest_arg" == "-m" || "$pytest_arg" == --markexpr=* ]]; then
    has_marker="1"
    break
  fi
done
if [[ "$sharding_active" == "1" && "$has_marker" == "0" && "$include_slow" != "1" ]]; then
  cmd+=("-m" "not slow")
fi

coverage_requested="${ROBOT_SF_PYTEST_COVERAGE:-}"
coverage_enabled="0"
case "$coverage_requested" in
  1|true|TRUE|yes|YES|on|ON) coverage_enabled=1 ;;
  ""|0|false|FALSE|no|NO|off|OFF) ;;
  *)
    echo "Invalid ROBOT_SF_PYTEST_COVERAGE value '$coverage_requested' (expected 1|0|true|false|yes|no|on|off)." >&2
    exit 2
    ;;
esac
if [[ "$sharding_active" != "1" && "${CI:-}" == "true" ]]; then
  coverage_enabled=1
fi
if [[ "$coverage_enabled" == "1" ]]; then
  if [[ "$sharding_active" == "1" ]]; then
    if [[ -z "${COVERAGE_FILE:-}" ]]; then
      echo "Sharded coverage requires a unique COVERAGE_FILE per shard." >&2
      exit 2
    fi
    cmd+=("--cov=robot_sf" "--cov-report=")
  else
    cmd+=("--cov=robot_sf" "--cov-report=html" "--cov-report=json")
  fi
fi

if [[ "$lane_mode" != "all" ]]; then
  export ROBOT_SF_TEST_LANE="$lane_mode"
fi

explicit_test_targets=()
for pytest_arg in "${pytest_args[@]}"; do
  [[ "$pytest_arg" == -* ]] && continue
  normalized_pytest_arg="$(normalize_pytest_target_path "$pytest_arg")"
  if [[ -e "$normalized_pytest_arg" || "$normalized_pytest_arg" == tests* ]]; then
    explicit_test_targets+=("$pytest_arg")
  fi
done

# The fixed core list keeps routine readiness fast, but must not hide a newly
# changed top-level core test that is absent from that list.
changed_top_level_core_test_paths=()
if [[ "$lane_mode" == "core" ]]; then
  changed_test_base_ref="${BASE_REF:-origin/main}"
  if git rev-parse --verify --quiet "${changed_test_base_ref}^{commit}" >/dev/null 2>&1; then
    while IFS= read -r changed_test_path; do
      if [[ -f "$changed_test_path" ]] && ! is_optional_test_path "$changed_test_path"; then
        changed_top_level_core_test_paths+=("$changed_test_path")
      fi
    done < <(
      git diff --name-only --diff-filter=ACMR "${changed_test_base_ref}...HEAD" -- \
        ':(top,glob)tests/test_*.py'
    )
  fi
fi

append_unique_pytest_arg() {
  local candidate="$1"
  local existing
  for existing in "${pytest_args[@]}"; do
    if [[ "$existing" == "$candidate" ]]; then
      return
    fi
  done
  pytest_args+=("$candidate")
}

if [[ "$lane_mode" == "core" ]]; then
  for pytest_arg in "${explicit_test_targets[@]}"; do
    if is_optional_test_path "$pytest_arg"; then
      echo "Core pytest lane cannot run optional-extra path '$pytest_arg'." >&2
      echo "Use scripts/dev/run_tests_parallel.sh --lane optional for torch/rvo2/CARLA/predictive tests." >&2
      exit 2
    fi
  done
  if [[ ${#explicit_test_targets[@]} -eq 0 ]]; then
    for core_test_path in "${core_test_paths[@]}"; do
      if [[ -e "$core_test_path" ]]; then
        append_unique_pytest_arg "$core_test_path"
      fi
    done
    for changed_test_path in "${changed_top_level_core_test_paths[@]}"; do
      if [[ -e "$changed_test_path" ]]; then
        append_unique_pytest_arg "$changed_test_path"
      fi
    done
  else
    for optional_test_path in "${optional_test_paths[@]}"; do
      if [[ -e "$optional_test_path" ]]; then
        cmd+=("--ignore=$optional_test_path")
      fi
    done
  fi
elif [[ "$lane_mode" == "optional" && ${#explicit_test_targets[@]} -eq 0 ]]; then
  for optional_test_path in "${optional_test_paths[@]}"; do
    if [[ -e "$optional_test_path" ]]; then
      pytest_args+=("$optional_test_path")
    fi
  done
fi

if [[ "$fast_fail" == "1" ]]; then
  # When sharding, each shard should report all failures, not just the first,
  # unless the caller explicitly requested fast-fail for local debugging.
  if [[ "$sharding_active" != "1" || "${PYTEST_FAST_FAIL:-}" == "1" ]]; then
    cmd+=("-x")
  fi
elif [[ "$fast_fail" != "0" ]]; then
  echo "Invalid PYTEST_FAST_FAIL value '$fast_fail' (expected 0 or 1)." >&2
  exit 2
fi

case "$order_mode" in
  failed-first)
    cmd+=("--failed-first")
    ;;
  new-first)
    cmd+=("--new-first")
    ;;
  none)
    ;;
  *)
    echo "Invalid PYTEST_ORDER_MODE value '$order_mode' (expected failed-first|new-first|none)." >&2
    exit 2
    ;;
esac

if [[ ${#pytest_args[@]} -gt 0 ]]; then
  cmd+=("${pytest_args[@]}")
fi

# Run pytest, capturing output so parallel-worker crashes can be classified
# (issue #5633). We capture to a log and print it on failure so the crash
# signature is never silently swallowed by `set -e`.
pytest_log="$(mktemp "${TMPDIR:-/tmp}/pytest_run.XXXXXX.log")"
set +e
"${cmd[@]}" >"$pytest_log" 2>&1
pytest_exit=$?
set -e
if [[ "$pytest_exit" -ne 0 ]]; then
  cat "$pytest_log" >&2
  # Classify the captured failure. The diagnostic is fail-closed: it only
  # reports, it never converts an incomplete run into success evidence.
  uv run python "$SCRIPT_DIR/diagnose_xdist_crash.py" \
    --log-file "$pytest_log" \
    --requested-workers "$worker_spec" \
    --dist-mode "$dist_mode" >&2 || true
  # Opt-in serial fallback: rerun with a single worker to separate an
  # environment crash from real failures. Disabled by default so the gate
  # stays fail-closed (an env crash is not success and not silently skipped).
  if [[ "$serial_fallback" == "1" ]]; then
    printf '\n[pr_ready_check] PR_READY_SERIAL_FALLBACK=1: rerunning serially to separate env crash from real failures.\n' >&2
    serial_log="$(mktemp "${TMPDIR:-/tmp}/pytest_serial.XXXXXX.log")"
    # ``cmd`` starts with ``uv run pytest -n <workers> --dist <mode>``. Keep
    # the wrapper-added pytest options, but do not copy the original worker
    # flags after ``-n 1``; pytest would otherwise use the later worker value.
    serial_cmd=(uv run pytest -n 1 --dist "$dist_mode")
    if [[ ${#cmd[@]} -gt 7 ]]; then
      serial_cmd+=("${cmd[@]:7}")
    fi
    set +e
    PYTEST_NUM_WORKERS=1 "${serial_cmd[@]}" >"$serial_log" 2>&1
    serial_exit=$?
    set -e
    cat "$serial_log" >&2
    uv run python "$SCRIPT_DIR/diagnose_xdist_crash.py" \
      --log-file "$serial_log" \
      --requested-workers 1 \
      --dist-mode "$dist_mode" \
      --serialized-ok "$([[ "$serial_exit" -eq 0 ]] && echo true || echo false)" >&2 || true
    rm -f "$serial_log"
  fi
  rm -f "$pytest_log"
  exit "$pytest_exit"
fi
rm -f "$pytest_log"
true
