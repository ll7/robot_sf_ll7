#!/usr/bin/env bash
# Quickstart validator for the imitation research reporting pipeline.
# Runs the end-to-end pipeline with tracker enabled, generates the research report,
# and validates the emitted artifacts (mirrors specs/270-imitation-report/quickstart.md).

set -euo pipefail

RUN_ID="${1:-research_quickstart_$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${2:-output/research_reports/${RUN_ID}}"
TRACKER_DIR="output/run-tracker/${RUN_ID}"

echo "[Quickstart] Using run_id=${RUN_ID}"

if [ -e "${TRACKER_DIR}" ]; then
  echo "[Quickstart] Run directory already exists: ${TRACKER_DIR}"
  echo "[Quickstart] Choose a fresh run_id (pass as first argument) or remove the old directory."
  exit 1
fi

echo "[Quickstart] Step 1/3: running imitation pipeline with tracker..."
uv run python examples/advanced/16_imitation_learning_pipeline.py \
  --enable-tracker \
  --tracker-output "${RUN_ID}"

if [ ! -f "${TRACKER_DIR}/manifest.jsonl" ] && [ ! -f "${TRACKER_DIR}/manifest.json" ]; then
  echo "[Quickstart] Tracker manifest missing at ${TRACKER_DIR}/manifest.jsonl (or .json)"
  exit 1
fi

echo "[Quickstart] Step 2/3: generating research report..."
uv run python scripts/research/generate_report.py \
  --tracker-run "${RUN_ID}" \
  --experiment-name "Quickstart ${RUN_ID}" \
  --output "${REPORT_DIR}"

echo "[Quickstart] Step 3/3: validating report artifacts..."
uv run python scripts/tools/validate_report.py "${REPORT_DIR}"

echo "[Quickstart] Success!"
echo "[Quickstart] Tracker artifacts: ${TRACKER_DIR}"
echo "[Quickstart] Report directory:  ${REPORT_DIR}"
