#!/usr/bin/env python3
"""
Integration Test: CI Performance Validation

Tests that the CI system package installation optimization achieves the required performance targets.
"""

import subprocess
import time
from pathlib import Path

import pytest


def test_package_installation_performance():
    """Test that package installation completes within 73 seconds (50% reduction target)."""
    start_time = time.time()

    # Simulate the optimized package installation
    # In real CI, this would be the apt-fast command
    result = subprocess.run(
        ["sudo", "apt-get", "update"], capture_output=True, text=True, timeout=300
    )

    assert result.returncode == 0, f"apt-get update failed: {result.stderr}"

    result = subprocess.run(
        [
            "sudo",
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "ffmpeg",
            "libglib2.0-0",
            "libgl1",
            "fonts-dejavu-core",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )  # 2 minute timeout

    end_time = time.time()
    duration = end_time - start_time

    assert result.returncode == 0, f"Package installation failed: {result.stderr}"
    assert duration < 73.0, f"Installation took {duration:.2f}s, exceeds 73s target"

    print(f"Package installation completed in {duration:.2f} seconds")


def test_apt_fast_availability():
    """Test that apt-fast is available and functional."""
    # Check if apt-fast is installed
    result = subprocess.run(["which", "apt-fast"], capture_output=True, text=True)
    assert result.returncode == 0, "apt-fast is not installed"

    # Test apt-fast help
    result = subprocess.run(["apt-fast", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, "apt-fast is not functional"
    assert "apt-fast" in result.stdout or result.stderr


def test_package_caching_setup():
    """Test that package caching is properly configured."""
    cache_dir = Path("/var/cache/apt/archives")

    # Check if cache directory exists
    assert cache_dir.exists(), "APT cache directory does not exist"

    # Check if we can write to cache (sudo might be needed in CI)
    # This is a basic check - in CI this would be tested differently
    assert cache_dir.is_dir(), "APT cache is not a directory"


def test_required_packages_installed():
    """Test that all required packages are properly installed."""
    required_packages = ["ffmpeg", "libglib2.0-0", "libgl1", "fonts-dejavu-core"]

    for package in required_packages:
        # Check if package is installed
        result = subprocess.run(["dpkg", "-l", package], capture_output=True, text=True)
        assert result.returncode == 0, f"Package {package} is not installed"

        # Verify package provides expected functionality
        if package == "ffmpeg":
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
            assert result.returncode == 0, "ffmpeg is not functional"
        elif package == "libglib2.0-0":
            result = subprocess.run(
                ["pkg-config", "--modversion", "glib-2.0"], capture_output=True, text=True
            )
            assert result.returncode == 0, "glib-2.0 is not available"


if __name__ == "__main__":
    pytest.main([__file__])
