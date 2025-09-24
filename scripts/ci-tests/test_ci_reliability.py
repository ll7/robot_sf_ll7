#!/usr/bin/env python3
"""
Integration Test: CI Reliability Validation

Tests that the CI system package installation optimization maintains reliability
with zero installation failures.
"""

import os
import subprocess

import pytest


def test_package_installation_reliability():
    """Test that package installation succeeds consistently."""
    # Attempt package installation multiple times to test reliability
    for attempt in range(3):
        result = subprocess.run(
            ["sudo", "apt-get", "update"], capture_output=True, text=True, check=False, timeout=300
        )

        assert result.returncode == 0, (
            f"apt-get update failed on attempt {attempt + 1}: {result.stderr}"
        )

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
            check=False,
            timeout=120,
        )

        assert result.returncode == 0, (
            f"Package installation failed on attempt {attempt + 1}: {result.stderr}"
        )


def test_apt_fast_reliability():
    """Test that apt-fast provides reliable package installation."""
    # Test apt-fast installation
    result = subprocess.run(
        [
            "sudo",
            "apt-fast",
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
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, f"apt-fast installation failed: {result.stderr}"


def test_cache_reliability():
    """Test that package caching doesn't break installation."""
    # Clear any existing cache issues
    result = subprocess.run(
        ["sudo", "apt-get", "clean"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, "apt-get clean failed"

    # Test installation with fresh cache
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
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, "Installation with cache failed"


def test_concurrent_installation_reliability():
    """Test that installation works reliably in CI environment."""
    # Simulate CI environment conditions
    env = {
        "DEBIAN_FRONTEND": "noninteractive",
        "APT_LISTCHANGES_FRONTEND": "none",
        "APT_OPTS": "-o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold",
    }

    result = subprocess.run(
        [
            "sudo",
            "-E",
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
        check=False,
        timeout=120,
        env={**env, **os.environ},
    )

    assert result.returncode == 0, f"CI-style installation failed: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__])
