#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

readonly PHASES=("lint" "typecheck" "test" "smoke" "artifact-policy")

show_help() {
  cat <<'EOF'
Usage: scripts/dev/ci_driver.sh [--list-phases] <phase> [<phase> ...]

Run one or more canonical CI validation phases.

Phases:
  lint             Ruff lint + format check
  typecheck        Ty type check (advisory, matches CI)
  test             Main pytest suite
  smoke            Validation and benchmark smoke checks
  artifact-policy  Canonical artifact root policy guard

Environment overrides:
  CI_DRIVER_EVENT_NAME=<event>    Override event context (defaults to GITHUB_EVENT_NAME or local)
  CI_DRIVER_GITHUB_REF=<ref>      Override ref context (defaults to GITHUB_REF or current branch)

Examples:
  scripts/dev/ci_driver.sh lint test
  CI_DRIVER_EVENT_NAME=workflow_dispatch scripts/dev/ci_driver.sh smoke
  scripts/dev/ci_driver.sh --list-phases
EOF
}

list_phases() {
  printf "%s\n" "${PHASES[@]}"
}

require_known_phase() {
  local phase="$1"
  local candidate
  for candidate in "${PHASES[@]}"; do
    if [[ "$candidate" == "$phase" ]]; then
      return 0
    fi
  done
  echo "Unknown CI phase: $phase" >&2
  echo "Known phases:" >&2
  list_phases >&2
  exit 2
}

resolve_event_name() {
  if [[ -n "${CI_DRIVER_EVENT_NAME:-}" ]]; then
    printf "%s\n" "$CI_DRIVER_EVENT_NAME"
    return
  fi
  if [[ -n "${GITHUB_EVENT_NAME:-}" ]]; then
    printf "%s\n" "$GITHUB_EVENT_NAME"
    return
  fi
  printf "local\n"
}

resolve_github_ref() {
  if [[ -n "${CI_DRIVER_GITHUB_REF:-}" ]]; then
    printf "%s\n" "$CI_DRIVER_GITHUB_REF"
    return
  fi
  if [[ -n "${GITHUB_REF:-}" ]]; then
    printf "%s\n" "$GITHUB_REF"
    return
  fi

  local branch
  branch="$(git branch --show-current 2>/dev/null || true)"
  if [[ -n "$branch" ]]; then
    printf "refs/heads/%s\n" "$branch"
    return
  fi

  printf "local\n"
}

is_strict_smoke_mode() {
  local event_name="$1"
  local github_ref="$2"
  [[ "$github_ref" == "refs/heads/main" || "$event_name" == "workflow_dispatch" ]]
}

run_smoke_phase() {
  local event_name="$1"
  local github_ref="$2"

  echo "Running map verification in CI mode (scope=ci)"
  uv run python scripts/validation/verify_maps.py \
    --scope ci \
    --mode ci \
    --output output/validation/map_verification.json

  echo "Running environment validation tests..."
  ./scripts/validation/test_basic_environment.sh
  ./scripts/validation/test_model_prediction.sh

  if is_strict_smoke_mode "$event_name" "$github_ref"; then
    echo "Running full performance smoke test..."
    uv run python scripts/validation/performance_smoke_test.py
  else
    echo "Skipping full performance smoke test outside main/workflow_dispatch context"
  fi

  local baseline="configs/benchmarks/perf_baseline_classic_cold_warm_v1.json"
  local max_slowdown="0.60"
  local max_throughput_drop="0.50"
  local min_seconds_delta="0.15"
  local min_throughput_delta="0.75"

  if [[ "$event_name" == "pull_request" ]]; then
    echo "Using conservative pull_request perf gate profile"
    max_slowdown="0.75"
    max_throughput_drop="0.60"
    min_seconds_delta="0.20"
    min_throughput_delta="1.00"
  fi

  local perf_cmd=(
    uv run python -m robot_sf.benchmark.perf_cold_warm
    --scenario-config configs/scenarios/archetypes/classic_crossing.yaml
    --scenario-name classic_crossing_low
    --episode-steps 48
    --cold-runs 1
    --warm-runs 2
    --baseline "$baseline"
    --output-json output/benchmarks/perf/cold_warm_pr_smoke.json
    --output-markdown output/benchmarks/perf/cold_warm_pr_smoke.md
    --max-slowdown-pct "$max_slowdown"
    --max-throughput-drop-pct "$max_throughput_drop"
    --min-seconds-delta "$min_seconds_delta"
    --min-throughput-delta "$min_throughput_delta"
    --require-baseline
  )

  if is_strict_smoke_mode "$event_name" "$github_ref"; then
    echo "Enforcing cold/warm regression gate"
    "${perf_cmd[@]}" --fail-on-regression
  else
    echo "Running cold/warm smoke in advisory mode"
    "${perf_cmd[@]}"
  fi

  echo "Running telemetry tracker smoke + perf wrapper"
  uv run python scripts/validation/run_examples_smoke.py --perf-tests-only

  set -euo pipefail
  echo "Preflight: checking schema file, uv availability, and optional jq"
  local schema_path="robot_sf/benchmark/schemas/episode.schema.v1.json"
  if [[ ! -r "$schema_path" ]]; then
    echo "ERROR: Schema file not found or not readable: $schema_path" >&2
    ls -la "$(dirname "$schema_path")" || true
    exit 1
  fi
  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' CLI not found in PATH" >&2
    exit 1
  fi
  if [[ ! -x "$(command -v uv)" ]]; then
    echo "ERROR: 'uv' CLI is not executable" >&2
    exit 1
  fi
  if command -v jq >/dev/null 2>&1; then
    echo "jq found; JSONL validation will be performed after smoke run"
  else
    echo "jq not found; JSONL validation will be skipped (episodes.jsonl size will still be checked)"
  fi

  mkdir -p output/benchmarks/ci_smoke
  local matrix_path="output/benchmarks/ci_smoke/matrix.yaml"
  cat > "$matrix_path" <<'YAML'
- id: ci-smoke-uni-low-open
  density: low
  flow: uni
  obstacle: open
  groups: 0.0
  speed_var: low
  goal_topology: point
  robot_context: embedded
  repeats: 1
YAML

  uv run robot_sf_bench run \
    --matrix "$matrix_path" \
    --out output/benchmarks/ci_smoke/episodes.jsonl \
    --schema "$schema_path" \
    --horizon 3 \
    --dt 0.1 \
    --base-seed 0
  test -s output/benchmarks/ci_smoke/episodes.jsonl

  if command -v jq >/dev/null 2>&1; then
    echo "Validating first JSONL record with jq..."
    head -n1 output/benchmarks/ci_smoke/episodes.jsonl | jq type >/dev/null
  else
    echo "jq not installed; skipping JSONL validation"
  fi
}

run_phase() {
  local phase="$1"
  local event_name="$2"
  local github_ref="$3"

  case "$phase" in
    lint)
      uv run ruff check .
      uv run ruff format --check .
      ;;
    typecheck)
      uvx ty check . --exit-zero
      ;;
    test)
      "$SCRIPT_DIR/run_tests_parallel.sh" tests
      ;;
    smoke)
      run_smoke_phase "$event_name" "$github_ref"
      ;;
    artifact-policy)
      uv run python scripts/tools/check_artifact_root.py
      ;;
    *)
      require_known_phase "$phase"
      ;;
  esac
}

if [[ $# -eq 0 ]]; then
  show_help >&2
  exit 2
fi

if [[ "$1" == "--list-phases" ]]; then
  list_phases
  exit 0
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  show_help
  exit 0
fi

event_name="$(resolve_event_name)"
github_ref="$(resolve_github_ref)"

for phase in "$@"; do
  require_known_phase "$phase"
  echo "==> CI phase: $phase"
  run_phase "$phase" "$event_name" "$github_ref"
done
