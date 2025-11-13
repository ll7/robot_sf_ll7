#!/bin/bash
# Complete Simulation Test
# Tests end-to-end functionality with a short simulation run

set -e  # Exit on any error

echo "Testing complete simulation..."

export DISPLAY=
export MPLBACKEND=Agg 
export SDL_VIDEODRIVER=dummy

# Run a short simulation to validate end-to-end functionality
timeout 30 uv run python examples/advanced/09_defensive_policy.py

echo "✅ Complete simulation test passed"

echo "Running manifest-driven example smoke tests..."
uv run python scripts/validation/run_examples_smoke.py

echo "✅ Example smoke tests passed"