"""Tests for plot utility functions (ensure_plot_dir_exists, save_plot)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from robot_sf.data_analysis.plot_utils import ensure_plot_dir_exists, save_plot


@pytest.fixture(autouse=True)
def _close_all_figures() -> None:
    yield
    plt.close("all")


# --- ensure_plot_dir_exists ---


def test_ensure_plot_dir_exists_creates_directory(tmp_path: Path) -> None:
    plot_path = str(tmp_path / "new_dir" / "plot.png")
    result = ensure_plot_dir_exists(plot_path)
    assert result == plot_path
    assert (tmp_path / "new_dir").is_dir()


def test_ensure_plot_dir_exists_existing_directory(tmp_path: Path) -> None:
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    plot_path = str(existing_dir / "plot.png")
    result = ensure_plot_dir_exists(plot_path)
    assert result == plot_path
    assert existing_dir.is_dir()


def test_ensure_plot_dir_exists_filename_only(tmp_path: Path) -> None:
    original = Path.cwd()
    os.chdir(str(tmp_path))
    try:
        result = ensure_plot_dir_exists("plot.png")
        assert result == "plot.png"
    finally:
        os.chdir(str(original))


# --- save_plot ---


def test_save_plot_creates_directory_and_file(tmp_path: Path) -> None:
    plt.figure()
    plt.plot([0, 1], [0, 1])
    filepath = str(tmp_path / "sub" / "plot.png")
    save_plot(filepath)
    assert os.path.isfile(filepath)
    assert os.path.getsize(filepath) > 0


def test_save_plot_with_title(tmp_path: Path) -> None:
    plt.figure()
    plt.plot([0, 1], [0, 1])
    filepath = str(tmp_path / "titled.png")
    save_plot(filepath, title="Test Title")
    assert os.path.isfile(filepath)
    assert os.path.getsize(filepath) > 0


def test_save_plot_closes_figure(tmp_path: Path) -> None:
    fig = plt.figure()
    fig_num = fig.number
    filepath = str(tmp_path / "closed.png")
    save_plot(filepath)
    assert fig_num not in plt.get_fignums()


def test_save_plot_suppresses_value_error(tmp_path: Path) -> None:
    plt.figure()
    plt.plot([0, 1], [0, 1])
    save_plot("\0")
    assert plt.get_fignums() == []


def test_save_plot_suppresses_permission_error(tmp_path: Path) -> None:
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    readonly_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        plt.figure()
        plt.plot([0, 1], [0, 1])
        save_plot(str(readonly_dir / "plot.png"))
        assert plt.get_fignums() == []
    finally:
        readonly_dir.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)


def test_save_plot_existing_directory_no_op(tmp_path: Path) -> None:
    (tmp_path / "ready").mkdir()
    plt.figure()
    plt.plot([0, 1], [0, 1])
    filepath = str(tmp_path / "ready" / "plot.png")
    save_plot(filepath)
    assert os.path.isfile(filepath)
    assert os.path.getsize(filepath) > 0


def test_save_plot_default_interactive(tmp_path: Path) -> None:
    plt.figure()
    plt.plot([0, 1], [0, 1])
    filepath = str(tmp_path / "default_interactive.png")
    save_plot(filepath)
    assert os.path.isfile(filepath)
    assert plt.get_fignums() == []
