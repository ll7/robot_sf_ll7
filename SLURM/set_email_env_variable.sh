#!/bin/bash
# Set the email as an environment variable in .bashrc if not already set

# Check for email argument
if [ -z "$1" ]; then
    echo "Usage: $0 <email>"
    exit 1
fi

# Check if SLURM_EMAIL is already set in the environment
if [ -z "$SLURM_EMAIL" ]; then
    # Check if SLURM_EMAIL is already in .bashrc
    if ! grep -q "export SLURM_EMAIL" ~/.bashrc; then
        # Add SLURM_EMAIL to .bashrc
        echo "export SLURM_EMAIL=$1" >> ~/.bashrc
        echo "SLURM_EMAIL has been added to ~/.bashrc"
        source ~/.bashrc
        echo "We also sourced ~/.bashrc"
    else
        echo "SLURM_EMAIL is already in ~/.bashrc"
    fi
    
    # Set SLURM_EMAIL for the current session
    export SLURM_EMAIL="$1"
    echo "SLURM_EMAIL has been set to $SLURM_EMAIL for the current session"
else
    echo "SLURM_EMAIL is already set to $SLURM_EMAIL"
fi
