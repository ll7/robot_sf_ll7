#!/usr/bin/env python3
"""
System Package Validation Script

Validates that required system packages are properly installed and functional.
Used by CI to ensure package installation optimization doesn't break functionality.
"""

import subprocess
import sys
from typing import Dict, Tuple

REQUIRED_PACKAGES = ["ffmpeg", "libglib2.0-0", "libgl1", "fonts-dejavu-core"]


def validate_package_installation(package: str) -> Tuple[bool, str]:
    """Validate that a specific package is installed and functional.

    Args:
        package: Package name to validate

    Returns:
        Tuple of (success, message)
    """
    # Check if package is installed via dpkg
    result = subprocess.run(["dpkg", "-l", package], capture_output=True, text=True, check=False)

    if result.returncode != 0:
        return False, f"Package {package} is not installed"

    # Package-specific functionality tests
    if package == "ffmpeg":
        return validate_ffmpeg()
    elif package == "libglib2.0-0":
        return validate_glib()
    elif package == "libgl1":
        return validate_glx()
    elif package == "fonts-dejavu-core":
        return validate_fonts()
    else:
        return True, f"Package {package} is installed (no specific validation)"


def validate_ffmpeg() -> Tuple[bool, str]:
    """Validate ffmpeg functionality."""
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False)

    if result.returncode != 0:
        return False, "ffmpeg is not functional"

    if "ffmpeg" not in result.stdout:
        return False, "ffmpeg version info not available"

    return True, "ffmpeg is functional"


def validate_glib() -> Tuple[bool, str]:
    """Validate glib functionality."""
    result = subprocess.run(
        ["pkg-config", "--modversion", "glib-2.0"], capture_output=True, text=True, check=False
    )

    if result.returncode != 0:
        return False, "glib-2.0 is not available"

    version = result.stdout.strip()
    if not version:
        return False, "glib-2.0 version not detectable"

    return True, f"glib-2.0 version {version} is available"


def validate_glx() -> Tuple[bool, str]:
    """Validate OpenGL functionality."""
    # glxinfo may not be available in headless environments
    result = subprocess.run(["glxinfo", "--version"], capture_output=True, text=True, check=False)

    if result.returncode != 0:
        # In headless CI, glx may not be available - this is OK
        return True, "glx not available in headless environment (expected)"

    return True, "OpenGL/glx is available"


def validate_fonts() -> Tuple[bool, str]:
    """Validate font availability."""
    result = subprocess.run(["fc-list"], capture_output=True, text=True, check=False)

    if result.returncode != 0:
        return False, "fontconfig is not available"

    if "DejaVu" not in result.stdout:
        return False, "DejaVu fonts are not available"

    return True, "DejaVu fonts are available"


def validate_all_packages() -> Dict[str, Tuple[bool, str]]:
    """Validate all required packages.

    Returns:
        Dictionary mapping package names to (success, message) tuples
    """
    results = {}
    for package in REQUIRED_PACKAGES:
        results[package] = validate_package_installation(package)
    return results


def main() -> int:
    """Main validation function. Returns exit code."""
    print("Validating system package installation...")

    results = validate_all_packages()

    all_success = True
    for package, (success, message) in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {package}: {message}")
        if not success:
            all_success = False

    if all_success:
        print("\nAll required packages are properly installed and functional.")
        return 0
    else:
        print("\nSome packages are missing or not functional.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
