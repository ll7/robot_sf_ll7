#!/usr/bin/env bash
# Combined contract check for repository agent instructions.
#
# Runs the instruction-reference checker, the provider-adapter scope check, and the focused
# instruction-contract test files. Use this as the single local entry point when changing
# AGENTS.md, .agents/*, docs/ai/*, provider adapters, or instruction tests.
#
# Usage:
#   scripts/dev/check_agent_instructions.sh [--json]
#
# The optional --json flag is forwarded to the instruction-reference checker.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

checker_args=()
if [[ "${1:-}" == "--json" ]]; then
    checker_args+=(--json)
fi

echo "[1/3] instruction references and contracts"
uv run python scripts/dev/check_instruction_references.py "${checker_args[@]}"

echo "[2/3] provider adapter scope"
uv run python scripts/tools/sync_ai_config.py --check

echo "[3/3] instruction behavior tests"
uv run pytest -q \
    tests/dev/test_instruction_references.py \
    tests/dev/test_instruction_precedence.py \
    tests/dev/test_task_scope_manifest.py \
    tests/dev/test_maintainer_values.py \
    tests/dev/test_delivery_and_friction.py \
    tests/dev/test_compendium_index.py \
    tests/dev/test_slurm_agent_contract.py \
    tests/dev/test_instruction_task_fixtures.py \
    tests/dev/test_compact_boot_router.py
