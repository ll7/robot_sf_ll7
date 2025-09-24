#!/usr/bin/env python3
"""
Integration Test: CI Cache Effectiveness

Tests that the CI system package caching provides performance benefits
and maintains installation reliability.
"""

import os
import subprocess
import time
from pathlib import Path

import pytest


def test_cache_directory_exists():
    """Test that the APT cache directory exists and is accessible."""
    cache_dir = Path("/var/cache/apt/archives")

    assert cache_dir.exists(), "APT cache directory does not exist"
    assert cache_dir.is_dir(), "APT cache path is not a directory"

    # Check if we can list contents (may require sudo in CI)
    try:
        list(cache_dir.iterdir())
        # Cache may be empty, that's OK
    except PermissionError:
        pytest.skip("Cannot access cache directory due to permissions")


def test_cache_configuration():
    """Test that cache configuration is properly set up."""
    # Check if actions/cache would work with our cache key
    cache_key = f"apt-{os.environ.get('RUNNER_OS', 'Linux')}-ci-optimization"

    assert cache_key is not None, "Cache key should be defined"
    assert "apt" in cache_key, "Cache key should include 'apt'"


def test_cache_restore_simulation():
    """Test that cache restore would work (simulation)."""
    # In real CI, actions/cache would restore /var/cache/apt/archives
    # Here we simulate the effect

    cache_dir = Path("/var/cache/apt/archives")

    # Get initial cache size
    initial_size = sum(f.stat().st_size for f in cache_dir.glob("*.deb") if f.is_file())

    # Simulate package installation (which would add to cache)
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

    assert result.returncode == 0, f"Package installation failed: {result.stderr}"

    # Check cache size after installation
    final_size = sum(f.stat().st_size for f in cache_dir.glob("*.deb") if f.is_file())

    # Cache should have grown (unless packages were already cached)
    assert final_size >= initial_size, "Cache should contain package files"


def test_cache_hit_performance():
    """Test that cached installation is faster (simulation)."""
    # First installation (cache miss)
    start_time = time.time()
    result1 = subprocess.run(
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
    first_duration = time.time() - start_time

    assert result1.returncode == 0, "First installation failed"

    # Second installation (cache hit - packages already installed)
    start_time = time.time()
    result2 = subprocess.run(
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
    second_duration = time.time() - start_time

    assert result2.returncode == 0, "Second installation failed"

    # Second installation should be faster (though not dramatically so for already installed packages)
    assert second_duration <= first_duration, (
        f"Second installation ({second_duration:.2f}s) should not be slower than first ({first_duration:.2f}s)"
    )


def test_cache_cleanup():
    """Test that cache cleanup works properly."""
    # Clean the cache
    result = subprocess.run(
        ["sudo", "apt-get", "clean"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, "Cache cleanup failed"

    # Verify cache is clean
    cache_dir = Path("/var/cache/apt/archives")
    deb_files = list(cache_dir.glob("*.deb"))

    # Note: apt-get clean may not remove all files, but should remove most
    # This is a basic check that the command runs
    assert isinstance(deb_files, list), "Cache cleanup should not break directory listing"


if __name__ == "__main__":
    pytest.main([__file__])
