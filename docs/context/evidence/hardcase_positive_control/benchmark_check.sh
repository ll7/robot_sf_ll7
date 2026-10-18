#!/usr/bin/env bash
# Reproduce the task's static/mocked benchmark audit without simulator entry.
set -u
hardcase_root="$(git rev-parse --show-toplevel)" || exit 2
hardcase_output="${1:?provide an empty output directory outside the checkout}"
mkdir -p "$hardcase_output" || exit 2
hardcase_output="$(cd "$hardcase_output" && pwd)"
cd "$hardcase_root" || exit 2
export HARDCASE_NO_TEST_SIMULATION=1
export ROBOT_SF_PYTEST_SEED_GUARD=1
export ROBOT_SF_PYTEST_SEED_AUDIT="$hardcase_output/test_seed_audit.jsonl"
export PYTHONPATH="$hardcase_root/docs/context/evidence/hardcase_positive_control/test_guard:$hardcase_root/tests/support:$hardcase_root"
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/benchmark -q \
  --tb=short --continue-on-collection-errors \
  --junitxml="$hardcase_output/benchmark.xml" > "$hardcase_output/benchmark.log" 2>&1
hardcase_rc=$?
printf '%s\n' "$hardcase_rc" > "$hardcase_output/benchmark.exit"
exit "$hardcase_rc"
