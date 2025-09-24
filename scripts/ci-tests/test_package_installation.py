#!/usr/bin/env python3
"""
CI Package Installation Performance Test

Simulates the CI package installation process locally to test performance improvements.
This provides a more controlled environment than running full act simulations.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

REQUIRED_PACKAGES = ["ffmpeg", "libglib2.0-0", "libgl1", "fonts-dejavu-core"]


def time_package_installation(use_apt_fast: bool = False) -> Dict[str, Any]:
    """Time package installation using apt or apt-fast.

    Args:
        use_apt_fast: Whether to use apt-fast instead of apt

    Returns:
        Dictionary with timing results
    """
    print(f"Testing package installation with {'apt-fast' if use_apt_fast else 'apt'}...")

    # Update package list first
    print("Updating package list...")
    start_time = time.time()
    update_result = subprocess.run(
        ["sudo", "apt-get", "update"], capture_output=True, text=True, check=False
    )
    update_time = time.time() - start_time

    if update_result.returncode != 0:
        return {
            "success": False,
            "error": f"apt-get update failed: {update_result.stderr}",
            "update_time": update_time,
        }

    # Install packages
    cmd = ["sudo", "apt-fast" if use_apt_fast else "apt-get", "install", "-y"] + REQUIRED_PACKAGES
    print(f"Installing packages: {' '.join(REQUIRED_PACKAGES)}")

    start_time = time.time()
    install_result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    install_time = time.time() - start_time

    return {
        "success": install_result.returncode == 0,
        "method": "apt-fast" if use_apt_fast else "apt",
        "update_time": update_time,
        "install_time": install_time,
        "total_time": update_time + install_time,
        "error": install_result.stderr if install_result.returncode != 0 else None,
    }


def simulate_ci_workflow() -> Dict[str, Any]:
    """Simulate the full CI workflow package installation steps."""
    print("Simulating CI workflow package installation...")

    results = {}

    # Test regular apt
    print("\n=== Testing regular apt-get ===")
    apt_result = time_package_installation(use_apt_fast=False)
    results["apt"] = apt_result

    if apt_result["success"]:
        print(".1f")
        print(".1f")
    else:
        print(f"✗ Failed: {apt_result.get('error', 'Unknown error')}")

    # Clean up for next test
    print("\nCleaning up for apt-fast test...")
    subprocess.run(
        ["sudo", "apt-get", "remove", "-y"] + REQUIRED_PACKAGES,
        capture_output=True,
        text=True,
        check=False,
    )

    # Test apt-fast
    print("\n=== Testing apt-fast ===")
    apt_fast_result = time_package_installation(use_apt_fast=True)
    results["apt_fast"] = apt_fast_result

    if apt_fast_result["success"]:
        print(".1f")
        print(".1f")
    else:
        print(f"✗ Failed: {apt_fast_result.get('error', 'Unknown error')}")

    # Calculate improvement
    if apt_result["success"] and apt_fast_result["success"]:
        improvement = apt_result["total_time"] - apt_fast_result["total_time"]
        improvement_pct = (improvement / apt_result["total_time"]) * 100
        results["improvement_seconds"] = improvement
        results["improvement_percentage"] = improvement_pct

        print("\n=== Performance Comparison ===")
        print(".1f")
        print(".1f")
        print(".1f")
        print(".1f")
        print(".1f")
        print(".1f")
    return results


def main() -> int:
    """Main function."""
    print("CI Package Installation Performance Test")
    print("=" * 50)

    # Check if running on Ubuntu/Debian
    try:
        result = subprocess.run(["lsb_release", "-i"], capture_output=True, text=True, check=True)
        if "Ubuntu" not in result.stdout and "Debian" not in result.stdout:
            print("⚠️  Warning: This test is designed for Ubuntu/Debian systems")
            print("   Current system:", result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Warning: Could not detect Linux distribution")
        print("   This test may not work on non-Ubuntu/Debian systems")

    # Check if apt-fast is available
    try:
        subprocess.run(["which", "apt-fast"], capture_output=True, check=True)
        apt_fast_available = True
    except subprocess.CalledProcessError:
        apt_fast_available = False
        print("⚠️  apt-fast not found. Install it first:")
        print("   sudo apt-get update && sudo apt-get install -y aria2")
        print("   wget https://raw.githubusercontent.com/ilikenwf/apt-fast/master/apt-fast")
        print(
            "   sudo mv apt-fast /usr/local/bin/apt-fast && sudo chmod +x /usr/local/bin/apt-fast"
        )

    if not apt_fast_available:
        print("\nCannot run full test without apt-fast. Exiting.")
        return 1

    # Run the simulation
    results = simulate_ci_workflow()

    # Save results
    output_file = Path("ci_package_test_results.json")
    import json

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
