"""Unit tests for figure generation module."""

import matplotlib
import matplotlib.pyplot as plt
import pytest

from robot_sf.research.figures import configure_matplotlib_backend, save_figure


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory."""
    output_path = tmp_path / "figures"
    output_path.mkdir(exist_ok=True)
    return output_path


def test_configure_matplotlib_backend():
    """Test matplotlib backend configuration."""
    configure_matplotlib_backend()
    assert matplotlib.get_backend().lower() == "agg"


def test_save_figure_dual_export(output_dir):
    """Test save_figure creates both PDF and PNG."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])

    paths = save_figure(fig, output_dir, "test_figure")

    assert "pdf" in paths
    assert "png" in paths
    assert paths["pdf"].exists()
    assert paths["png"].exists()
    assert paths["pdf"].name == "test_figure.pdf"
    assert paths["png"].name == "test_figure.png"

    plt.close(fig)
