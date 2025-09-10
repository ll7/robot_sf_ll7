#!/bin/bash
# Script to run all tests for robot-sf and fast-pysf

set -e

echo "Running robot-sf tests..."
python -m pytest tests/ -v

echo ""
echo "Running fast-pysf tests..."
cd fast-pysf
python -m pytest tests/ -v
cd ..

echo ""
echo "All tests completed successfully!"