#!/bin/sh
set -e

git config --global --add safe.directory /workspaces/robot_sf_ll7

## This script is run after the devcontainer is created. It installs the necessary dependencies for the project.
pwd
git submodule update --init --recursive || { echo "Failed to update git submodules"; exit 1; }
git config --global --add safe.directory /workspaces/robot_sf_ll7/fast-pysf

pip install -r ./requirements.txt || { echo "Failed to install requirements from ./requirements.txt"; exit 1; }
pip install -r ./fast-pysf/requirements.txt || { echo "Failed to install requirements from ./fast-pysf/requirements.txt"; exit 1; }
pip install -e . || { echo "Failed to install the current directory as a package"; exit 1; }
pip install -e ./fast-pysf || { echo "Failed to install ./fast-pysf as a package"; exit 1; }

# Set the display environment variable for GUI applications based on the host OS
if [ $HOST_OS == *"Windows"* ]; then
    # We are in Windows
    echo "export DISPLAY=host.docker.internal:0.0" >> ~/.bashrc || { echo "Failed to set DISPLAY environment variable for Windows"; exit 1; }
else
    # We are in Linux
    echo "export DISPLAY=:0" >> ~/.bashrc || { echo "Failed to set DISPLAY environment variable for Linux"; exit 1; }
fi