#! /bin/bash

# Script to query jobs in a specified SLURM partition

# date and time
echo "Partion queue at $(date)"

# Default partition name
PARTITION="epyc-gpu-test"

# Allow overriding the partition name via command line argument
if [ $# -eq 1 ]; then
    PARTITION="$1"
fi

# Check if squeue command exists
if ! command -v squeue &> /dev/null; then
    echo "Error: squeue command not found. Is SLURM installed?"
    exit 1
fi

# Run squeue command and capture its exit status
squeue -p "$PARTITION" -l
exit_status=$?

# Check if squeue command was successful
if [ $exit_status -ne 0 ]; then
    echo "Error: squeue command failed with exit status $exit_status"
    exit $exit_status
fi
