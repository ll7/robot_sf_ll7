#!/bin/bash

# Check if Graphviz is installed
if ! command -v dot &> /dev/null; then
    echo "Error: Graphviz is not installed. Please install it using:"
    echo "sudo apt-get install graphviz"
    exit 1
fi

# Check if pylint is installed
if ! command -v pylint &> /dev/null; then
    echo "Error: pylint is not installed. Please install it using:"
    echo "pip install pylint"
    exit 1
fi

pyreverse -o svg -p robot_sf --colorized --max-color-depth 8 --show-ancestors 1 --show-associated 1 -my --show-stdlib --output-dir class_diagram/ robot_sf/
