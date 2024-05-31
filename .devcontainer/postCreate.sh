#!/bin/bash

## This script is run after the devcontainer is created. It installs the necessary dependencies for the project.
git submodule update --init --recursive
pip install -r ./requirements.txt
pip install -r ./fast-pysf/requirements.txt
pip install -e .
pip install -e ./fast-pysf

# Set the display environment variable for GUI applications based on the host OS
if [[ $HOST_OS == *"Windows"* ]]; then
    # We are in Windows
    echo "export DISPLAY=host.docker.internal:0.0" >> ~/.bashrc
else
    # We are in Linux
    echo "export DISPLAY=:0" >> ~/.bashrc
fi
