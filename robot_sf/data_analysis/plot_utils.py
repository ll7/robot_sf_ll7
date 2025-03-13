"""
Utility functions for plotting and saving visualization files.

This module provides centralized utilities for:
- Creating plot directories
- Saving plots with consistent error handling
- Other common plot-related functionality
"""

import os

import matplotlib.pyplot as plt


def ensure_plot_dir_exists(plot_path):
    """
    Ensure the plot directory exists, creating it if necessary.

    Args:
        plot_path (str): Path to the plot file (including filename)

    Returns:
        str: The validated plot path
    """
    # Extract directory from the full path
    directory = os.path.dirname(plot_path)

    # Create directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {os.path.abspath(directory)}")

    return plot_path


def save_plot(filename, title=None, interactive=False):
    """
    Save a plot to file, ensuring the directory exists.

    Args:
        filename (str): Path where the plot should be saved
        title (str, optional): Title for the plot
        interactive (bool, optional): Whether to display the plot interactively
    """
    if title:
        plt.title(title)

    plt.tight_layout()

    # Ensure the directory exists and get the validated path
    validated_path = ensure_plot_dir_exists(filename)

    # Save the plot
    try:
        plt.savefig(validated_path)
        print(f"Plot saved to: {os.path.abspath(validated_path)}")
    except Exception as e:
        print(f"Error saving plot to {validated_path}: {e}")

    # Show the plot if interactive
    if interactive:
        plt.show()

    # Close the plot to free memory
    plt.close()
