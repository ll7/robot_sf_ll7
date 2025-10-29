#!/bin/sh
set -e

git config --global --add safe.directory /workspaces/robot_sf_ll7

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "Failed to install uv"; exit 1; }
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Use uv for all dependency management
uv sync --all-extras || { echo "Failed to sync dependencies with uv"; exit 1; }

# Set the display environment variable for GUI applications based on the host OS
if [ $HOST_OS == *"Windows"* ]; then
    # We are in Windows
    echo "export DISPLAY=host.docker.internal:0.0" >> ~/.bashrc || { echo "Failed to set DISPLAY environment variable for Windows"; exit 1; }
else
    # We are in Linux
    echo "export DISPLAY=:0" >> ~/.bashrc || { echo "Failed to set DISPLAY environment variable for Linux"; exit 1; }
fi