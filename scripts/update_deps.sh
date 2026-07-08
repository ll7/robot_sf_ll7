#!/bin/bash
set -e

echo "🔄 Updating dependencies with uv..."

# Update the lock file
echo "📦 Updating uv.lock..."
uv lock --upgrade

# Sync the environment
echo "🔄 Syncing environment..."
uv sync

echo "✅ Dependencies updated successfully!"
echo ""
echo "To install specific extras, use:"
echo "  uv sync --group dev    # Install with development tools"
echo "  uv sync --extra training  # Install training and experiment tools"
echo "  uv sync --extra gpu    # Install with GPU support"
echo "  uv sync --group docs   # Install with documentation tools"
echo "  uv sync --extra rllib --extra training  # Install RLlib/DreamerV3 training tools"
echo "  uv sync --all-extras   # Install all extras"
echo ""
echo "To install optional dependency groups, use:"
echo "  uv sync --group imitation  # Install imitation-learning tools"
echo "  uv sync --all-extras --group carla  # Add pinned CARLA Python client explicitly"
