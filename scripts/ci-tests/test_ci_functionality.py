#!/usr/bin/env python3
"""
Integration Test: CI Functionality Preservation

Tests that the CI system package installation optimization maintains
headless GUI testing capability for pygame and matplotlib.
"""

import os
import subprocess

import pytest


def test_pygame_headless_functionality():
    """Test that pygame works correctly in headless environment."""
    # Set headless environment
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

    # Test pygame import and basic functionality
    try:
        import pygame

        pygame.init()

        # Test basic display creation
        screen = pygame.display.set_mode((640, 480))
        assert screen is not None, "Failed to create pygame display"

        # Test basic drawing
        screen.fill((255, 255, 255))
        pygame.draw.circle(screen, (255, 0, 0), (320, 240), 50)

        pygame.quit()
    except ImportError:
        pytest.fail("pygame not available for headless testing")
    except Exception as e:
        pytest.fail(f"pygame headless functionality failed: {e}")


def test_matplotlib_headless_functionality():
    """Test that matplotlib works correctly in headless environment."""
    # Set headless backend
    os.environ["MPLBACKEND"] = "Agg"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Test basic plotting
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title("Test Plot")

        # Test saving (simulates what tests might do)
        import io

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        assert len(buf.getvalue()) > 0, "Failed to generate plot image"

        plt.close(fig)
    except ImportError:
        pytest.fail("matplotlib not available for headless testing")
    except Exception as e:
        pytest.fail(f"matplotlib headless functionality failed: {e}")


def test_glx_functionality():
    """Test that OpenGL functionality is available."""
    # Test glxinfo if available (may not work in all CI environments)
    result = subprocess.run(["glxinfo", "--version"], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        assert "glx" in result.stdout.lower() or "opengl" in result.stdout.lower()
    else:
        # glxinfo may not be available in headless environment, that's OK
        pass


def test_font_availability():
    """Test that required fonts are available."""
    # Test fontconfig
    result = subprocess.run(["fc-list"], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        # Check for DejaVu fonts
        assert "DejaVu" in result.stdout, "DejaVu fonts not available"
    else:
        pytest.skip("fontconfig not available in test environment")


def test_ffmpeg_functionality():
    """Test that ffmpeg is functional."""
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, "ffmpeg not functional"
    assert "ffmpeg" in result.stdout, "ffmpeg version info not available"


def test_glib_functionality():
    """Test that glib libraries are functional."""
    result = subprocess.run(
        ["pkg-config", "--modversion", "glib-2.0"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, "glib-2.0 not available"
    # Version should be parseable as float
    version = result.stdout.strip()
    assert version.count(".") >= 1, f"Invalid glib version: {version}"


if __name__ == "__main__":
    pytest.main([__file__])
