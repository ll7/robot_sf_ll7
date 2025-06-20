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
echo "  uv sync --extra dev    # Install with development tools"
echo "  uv sync --extra gpu    # Install with GPU support"
echo "  uv sync --extra docs   # Install with documentation tools"
