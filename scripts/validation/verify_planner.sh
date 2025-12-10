#!/usr/bin/env bash
set -euo pipefail

echo "Running planner test suite..."
uv run pytest tests/test_planner
