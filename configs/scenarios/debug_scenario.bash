#!/usr/bin/env bash
# Debug scenario runner for the Social Navigation Benchmark.
# Purpose: Run a minimal scenario set to validate the pipeline end-to-end.
# Fixes previous issues:
#   1) Misspelled YAML filename (debug_senario.yaml -> debug_scenario.yaml)
#   2) Trailing spaces after a line-continuation backslash causing the next line
#      to be executed as a separate command ("--workers: command not found").
# Usage: ./configs/scenarios/debug_scenario.bash
# Can be executed from any directory; it resolves repo root automatically.

set -euo pipefail

# Resolve repo root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENARIO_FILE="${REPO_ROOT}/configs/scenarios/debug_scenario.yaml"
OUT_FILE="${REPO_ROOT}/results/debug_scenario.jsonl"

if [[ ! -f "${SCENARIO_FILE}" ]]; then
  echo "Error: Scenario file not found: ${SCENARIO_FILE}" >&2
  exit 1
fi

echo "Running debug benchmark scenario..."
echo "Scenario: ${SCENARIO_FILE}"
echo "Output  : ${OUT_FILE}"

uv run python -m robot_sf.benchmark.cli run \
  --scenarios "${SCENARIO_FILE}" \
  --output "${OUT_FILE}" \
  --workers 4 \
  --resume

echo "✓ Done. Inspect first lines with: head -n 3 ${OUT_FILE}"