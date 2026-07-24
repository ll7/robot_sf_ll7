"""Tests for plot utility functions (ensure_plot_dir_exists, save_plot)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

from robot_sf.data_analysis.plot_utils import ensure_plot_dir_exists, save_plot

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _close_all_figures() -> Iterator[None]:
    """Close every Matplotlib figure after each plot utility test."""
    yield
    plt.close("all")


# --- ensure_plot_dir_exists ---


def test_ensure_plot_dir_exists_creates_directory(tmp_path: Path) -> None:
    """Create the parent directory and preserve the requested plot path."""
    plot_path = str(tmp_path / "new_dir" / "plot.png")
    result = ensure_plot_dir_exists(plot_path)
    assert result == plot_path
    assert (tmp_path / "new_dir").is_dir()


def test_ensure_plot_dir_exists_existing_directory(tmp_path: Path) -> None:
    """Leave an existing parent directory intact while returning the path."""
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    plot_path = str(existing_dir / "plot.png")
    result = ensure_plot_dir_exists(plot_path)
    assert result == plot_path
    assert existing_dir.is_dir()


def test_ensure_plot_dir_exists_filename_only(tmp_path: Path) -> None:
    """Handle a filename without a directory component."""
    original = Path.cwd()
    os.chdir(str(tmp_path))
    try:
        result = ensure_plot_dir_exists("plot.png")
        assert result == "plot.png"
    finally:
        os.chdir(str(original))


# --- save_plot ---


def test_save_plot_creates_directory_and_file(tmp_path: Path) -> None:
    """Save a plot and create its missing parent directory."""
    plt.figure()
    plt.plot([0, 1], [0, 1])
    filepath = str(tmp_path / "sub" / "plot.png")
    save_plot(filepath)
    assert os.path.isfile(filepath)
    assert os.path.getsize(filepath) > 0


def test_save_plot_with_title(tmp_path: Path) -> None:
    """Apply the supplied title before saving and closing the figure."""
    plt.figure()
    plt.plot([0, 1], [0, 1])
    fig = plt.gcf()
    filepath = str(tmp_path / "titled.png")
    save_plot(filepath, title="Test Title")
    assert fig.axes[0].get_title() == "Test Title"
    assert os.path.isfile(filepath)
    assert os.path.getsize(filepath) > 0


def test_save_plot_closes_figure(tmp_path: Path) -> None:
    """Close the active figure after the plot has been saved."""
    fig = plt.figure()
    fig_num = fig.number
    filepath = str(tmp_path / "closed.png")
    save_plot(filepath)
    assert fig_num not in plt.get_fignums()


def test_save_plot_suppresses_value_error(tmp_path: Path) -> None:
    """Suppress invalid-path errors and still close the active figure."""
    plt.figure()
    plt.plot([0, 1], [0, 1])
    save_plot("\0")
    assert plt.get_fignums() == []


def test_save_plot_suppresses_permission_error(tmp_path: Path) -> None:
    """Suppress save-time operating-system errors and still close the figure."""
    plt.figure()
    plt.plot([0, 1], [0, 1])
    with patch(
        "robot_sf.data_analysis.plot_utils.plt.savefig",
        side_effect=OSError("permission denied"),
    ) as savefig:
        save_plot(str(tmp_path / "plot.png"))
    savefig.assert_called_once()
    assert plt.get_fignums() == []


def test_save_plot_existing_directory_no_op(tmp_path: Path) -> None:
    """Save normally when the target directory already exists."""
    (tmp_path / "ready").mkdir()
    plt.figure()
    plt.plot([0, 1], [0, 1])
    filepath = str(tmp_path / "ready" / "plot.png")
    save_plot(filepath)
    assert os.path.isfile(filepath)
    assert os.path.getsize(filepath) > 0


def test_save_plot_default_interactive(tmp_path: Path) -> None:
    """Keep the default non-interactive mode from showing the plot."""
    plt.figure()
    plt.plot([0, 1], [0, 1])
    filepath = str(tmp_path / "default_interactive.png")
    with patch("robot_sf.data_analysis.plot_utils.plt.show") as show:
        save_plot(filepath)
    show.assert_not_called()
    assert os.path.isfile(filepath)
    assert plt.get_fignums() == []
