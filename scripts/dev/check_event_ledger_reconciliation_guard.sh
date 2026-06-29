#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

fixture="${1:-tests/benchmark/fixtures/event_ledger_reconciliation/reconciled_episodes.jsonl}"
out="${2:-output/validation/event_ledger_reconciliation_guard.jsonl}"

mkdir -p "$(dirname "$out")"

echo "Running EpisodeEventLedger exact/derived reconciliation guard on $fixture"
uv run python scripts/benchmark/export_event_ledger_reconciliation.py \
  --episodes "$fixture" \
  --out "$out" \
  --fail-on-violations
