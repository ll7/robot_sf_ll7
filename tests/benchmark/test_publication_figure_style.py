"""Tests for publication figure style pack.

This module tests the publication-style context, planner palette,
vector export, and provenance sidecar functionality.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from robot_sf.benchmark.figures.export import save_publication_figure
from robot_sf.benchmark.figures.provenance import (
    build_caption_fragment,
    build_provenance,
    write_caption_fragment,
    write_provenance,
)
from robot_sf.benchmark.figures.style import (
    figure_size,
    planner_color,
    planner_palette,
    publication_style,
)


class TestPlannerPalette:
    """Tests for planner color palette."""

    def test_planner_palette_returns_known_planners(self):
        """Verify known planners are in the palette."""
        palette = planner_palette()
        assert "goal" in palette
        assert "orca" in palette
        assert "social_force" in palette
        assert "ppo" in palette

    def test_planner_color_known_planner(self):
        """Verify known planners return fixed colors."""
        assert planner_color("goal") == "#E69F00"
        assert planner_color("orca") == "#56B4E9"

    def test_planner_color_unknown_planner_deterministic(self):
        """Verify unknown planners return deterministic fallback colors."""
        color1 = planner_color("unknown_planner_xyz")
        color2 = planner_color("unknown_planner_xyz")
        assert color1 == color2

    def test_planner_color_unknown_planner_stable_across_processes(self):
        """Pin fallback colors so an unknown planner keeps its color everywhere.

        The fallback must use a stable hash (not Python's per-process salted
        ``hash()``); otherwise the same planner would get a different color in
        each figure-generation run. These pinned values encode that contract:
        if the fallback hashing changes, this test fails on purpose.
        """
        assert planner_color("planner_a") == "#66A61E"
        assert planner_color("planner_b") == "#CC79A7"
        assert planner_color("planner_c") == "#0072B2"

    def test_planner_color_unknown_planner_hex_format(self):
        """Verify unknown planner colors are valid hex."""
        color = planner_color("unknown_planner")
        assert color.startswith("#")
        assert len(color) == 7


class TestFigureSize:
    """Tests for figure size presets."""

    def test_single_column_size(self):
        """Verify single-column size is approximately 3.4 inches."""
        w, h = figure_size("single")
        assert abs(w - 3.4) < 0.1
        assert h > 0

    def test_double_column_size(self):
        """Verify double-column size is approximately 7 inches."""
        w, h = figure_size("double")
        assert abs(w - 7.0) < 0.1
        assert h > 0

    def test_invalid_size_raises(self):
        """Verify invalid size raises ValueError."""
        with pytest.raises(ValueError, match="Invalid figure size"):
            figure_size("invalid")


class TestPublicationStyle:
    """Tests for publication style context manager."""

    def test_restores_rcparams_after_exit(self):
        """Verify rcParams are restored after context exit."""
        plt = pytest.importorskip("matplotlib.pyplot")
        original_font = plt.rcParams["font.family"]

        with publication_style():
            # Inside context, font should be serif
            assert "serif" in plt.rcParams["font.family"] or "serif" in str(
                plt.rcParams["font.serif"]
            )

        # After exit, should be restored
        assert plt.rcParams["font.family"] == original_font

    def test_single_size_sets_figure_size(self):
        """Verify single size sets correct figure dimensions."""
        plt = pytest.importorskip("matplotlib.pyplot")

        with publication_style(size="single"):
            assert abs(plt.rcParams["figure.figsize"][0] - 3.4) < 0.1

    def test_double_size_sets_figure_size(self):
        """Verify double size sets correct figure dimensions."""
        plt = pytest.importorskip("matplotlib.pyplot")

        with publication_style(size="double"):
            assert abs(plt.rcParams["figure.figsize"][0] - 7.0) < 0.1


class TestProvenance:
    """Tests for provenance sidecar builder."""

    def test_build_provenance_minimal(self):
        """Verify minimal provenance has required fields."""
        provenance = build_provenance()
        assert "generated_at" in provenance
        assert "repo_commit" in provenance
        assert "generator_command" in provenance

    def test_build_provenance_with_source_artifacts(self):
        """Verify provenance includes source artifacts."""
        artifacts = [{"path": "data.jsonl", "hash": "abc123"}]
        provenance = build_provenance(source_artifacts=artifacts)
        assert provenance["source_artifacts"] == artifacts

    def test_write_provenance_creates_file(self, tmp_path):
        """Verify write_provenance creates a JSON file."""
        output_base = tmp_path / "figure"
        output_base.mkdir()
        provenance = build_provenance()

        path = write_provenance(output_base / "test", provenance)
        assert path.exists()
        assert path.suffix == ".json"

    def test_write_provenance_is_deterministic(self, tmp_path):
        """Verify same inputs produce identical provenance content."""
        output_base = tmp_path / "figure"
        output_base.mkdir()
        provenance = build_provenance()

        # Write twice with fixed timestamp
        fixed_time = datetime(2026, 1, 1, tzinfo=UTC)
        path1 = write_provenance(output_base / "test", provenance, timestamp=fixed_time)
        path2 = write_provenance(output_base / "test2", provenance, timestamp=fixed_time)

        content1 = json.loads(path1.read_text())
        content2 = json.loads(path2.read_text())

        # Compare ignoring output_hashes (different filenames)
        content1.pop("output_hashes", None)
        content2.pop("output_hashes", None)
        assert content1 == content2

    def test_build_caption_fragment(self):
        """Verify caption fragment includes campaign and episodes."""
        caption = build_caption_fragment(
            campaign_name="test_campaign",
            episode_ids=["ep1", "ep2", "ep3"],
        )
        assert "test_campaign" in caption
        assert "ep1" in caption

    def test_write_caption_fragment_creates_file(self, tmp_path):
        """Verify write_caption_fragment creates a .caption.tex file."""
        output_base = tmp_path / "figure"
        output_base.mkdir()
        caption = "Test caption"

        path = write_caption_fragment(output_base / "test", caption)
        assert path.exists()
        assert path.suffix == ".tex"
        assert path.read_text() == caption


class TestExport:
    """Tests for figure export helper."""

    def test_save_publication_figure_pdf(self, tmp_path):
        """Verify PDF export creates file and sidecar."""
        plt = pytest.importorskip("matplotlib.pyplot")

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        paths = save_publication_figure(
            fig,
            tmp_path / "test_figure",
            formats=("pdf",),
            provenance=build_provenance(generator_command="test"),
        )

        plt.close(fig)

        # Should have PDF and provenance
        assert any(p.suffix == ".pdf" for p in paths)
        assert any(".provenance.json" in p.name for p in paths)

    def test_save_publication_figure_png(self, tmp_path):
        """Verify PNG export creates file."""
        plt = pytest.importorskip("matplotlib.pyplot")

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        paths = save_publication_figure(
            fig,
            tmp_path / "test_figure",
            formats=("png",),
            provenance=build_provenance(generator_command="test"),
        )

        plt.close(fig)

        assert any(p.suffix == ".png" for p in paths)

    def test_save_publication_figure_svg(self, tmp_path):
        """Verify SVG export creates file."""
        plt = pytest.importorskip("matplotlib.pyplot")

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        paths = save_publication_figure(
            fig,
            tmp_path / "test_figure",
            formats=("svg",),
            provenance=build_provenance(generator_command="test"),
        )

        plt.close(fig)

        assert any(p.suffix == ".svg" for p in paths)

    def test_save_publication_figure_with_caption(self, tmp_path):
        """Verify caption fragment is written when provided."""
        plt = pytest.importorskip("matplotlib.pyplot")

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        paths = save_publication_figure(
            fig,
            tmp_path / "test_figure",
            formats=("pdf",),
            caption_fragment="Test caption",
        )

        plt.close(fig)

        assert any(".caption.tex" in p.name for p in paths)

    def test_save_publication_figure_provenance_content(self, tmp_path):
        """Verify provenance includes source hashes and command."""
        plt = pytest.importorskip("matplotlib.pyplot")

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        provenance = build_provenance(
            source_artifacts=[{"path": "data.jsonl", "hash": "abc123"}],
            generator_command="test command",
        )

        paths = save_publication_figure(
            fig,
            tmp_path / "test_figure",
            formats=("pdf",),
            provenance=provenance,
        )

        plt.close(fig)

        # Find provenance file
        prov_path = next(p for p in paths if ".provenance.json" in p.name)
        content = json.loads(prov_path.read_text())

        assert content["generator_command"] == "test command"
        assert content["source_artifacts"][0]["path"] == "data.jsonl"

    def test_save_publication_figure_empty_formats_raises(self, tmp_path):
        """Verify empty formats raises ValueError."""
        plt = pytest.importorskip("matplotlib.pyplot")

        fig, _ax = plt.subplots()

        with pytest.raises(ValueError, match="At least one format"):
            save_publication_figure(fig, tmp_path / "test", formats=())

        plt.close(fig)


class TestStyleDoesNotAlterData:
    """Tests verifying style context does not alter plotted data."""

    def test_plot_values_unchanged_by_style(self):
        """Verify style context does not modify plot data."""
        plt = pytest.importorskip("matplotlib.pyplot")

        # Create a figure with known data
        fig, ax = plt.subplots()
        x_data = [1.0, 2.0, 3.0]
        y_data = [4.0, 5.0, 6.0]
        ax.plot(x_data, y_data)

        # Get line data before style
        line_before = ax.lines[0]
        x_before = list(line_before.get_xdata())
        y_before = list(line_before.get_ydata())

        # Apply style
        with publication_style():
            # Get line data during style
            line_during = ax.lines[0]
            x_during = list(line_during.get_xdata())
            y_during = list(line_during.get_ydata())

        # Data should be unchanged
        assert x_before == x_during
        assert y_before == y_during

        plt.close(fig)
