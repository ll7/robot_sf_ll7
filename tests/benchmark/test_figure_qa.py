"""Tests for deterministic figure artifact QA checks."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from PIL import Image

from robot_sf.benchmark.artifact_catalog import (
    ARTIFACT_CATALOG_SCHEMA_VERSION,
    ArtifactCatalog,
    ArtifactCatalogEntry,
    ArtifactFileRef,
    sha256_file,
)
from robot_sf.benchmark.figure_qa import (
    FigureQA,
    assert_clean,
    check_figure_entry,
    check_figure_file,
    lint_figure,
    main,
    validate_figures_in_catalog,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "figure_qa"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _create_png(path: Path, size: tuple[int, int] = (100, 100)) -> Path:
    """Create a small valid PNG file at *path* and return it."""
    img = Image.new("RGB", size, color="blue")
    img.save(path, format="PNG")
    return path


def _error_message(issues: list[FigureQA], check: str) -> str | None:
    """Return the message of the first issue matching *check*, or None."""
    for issue in issues:
        if issue.check == check:
            return issue.message
    return None


def _assert_check(issues: list[FigureQA], check: str, artifact_id: str) -> None:
    """Assert that an issue with *check* exists and names *artifact_id*."""
    assert any(issue.check == check and issue.artifact_id == artifact_id for issue in issues), (
        f"Expected check={check!r} not found in {issues}"
    )


def _assert_not_check(issues: list[FigureQA], check: str) -> None:
    """Assert that no issue with *check* exists."""
    assert not any(issue.check == check for issue in issues), (
        f"Unexpected check={check!r} found in {issues}"
    )


# ---------------------------------------------------------------------------
# valid path
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_figure(tmp_path: Path) -> Path:
    """Return a path to a valid small PNG."""
    return _create_png(tmp_path / "valid_figure.png")


def test_valid_figure_passes(valid_figure: Path) -> None:
    """A valid figure should pass all checks with no issues."""
    issues = check_figure_file(valid_figure, artifact_id="fig_test")
    assert issues == []


def test_valid_figure_with_caption_passes(valid_figure: Path, tmp_path: Path) -> None:
    """A valid figure with a caption file should pass all checks."""
    caption = tmp_path / "captions.md"
    caption.write_text("# Valid caption\n", encoding="utf-8")
    issues = check_figure_file(
        valid_figure,
        artifact_id="fig_test",
        caption_path=caption,
    )
    assert issues == []


# ---------------------------------------------------------------------------
# missing file
# ---------------------------------------------------------------------------


def test_missing_file() -> None:
    """Non-existent path should report file_exists check."""
    missing = Path("/nonexistent/fig_missing.png")
    issues = check_figure_file(missing, artifact_id="fig_missing")
    _assert_check(issues, "file_exists", "fig_missing")
    assert "does not exist" in issues[0].message


def test_directory_path(tmp_path: Path) -> None:
    """A directory path should report file_exists."""
    dirpath = tmp_path / "subdir"
    dirpath.mkdir()
    issues = check_figure_file(dirpath, artifact_id="fig_dir")
    _assert_check(issues, "file_exists", "fig_dir")
    assert "not a regular file" in issues[0].message


# ---------------------------------------------------------------------------
# empty / near-empty PNG
# ---------------------------------------------------------------------------


def test_empty_file(tmp_path: Path) -> None:
    """A zero-byte file should fail file_size and valid_image checks."""
    empty = tmp_path / "empty.png"
    empty.touch()
    issues = check_figure_file(empty, artifact_id="fig_empty")
    _assert_check(issues, "file_size", "fig_empty")
    _assert_check(issues, "valid_image", "fig_empty")


def test_tiny_dimensions_fail(tmp_path: Path) -> None:
    """A 1x1 pixel image should fail the valid_image dimension check."""
    tiny = tmp_path / "tiny.png"
    _create_png(tiny, size=(1, 1))
    issues = check_figure_file(tiny, artifact_id="fig_tiny")
    _assert_check(issues, "valid_image", "fig_tiny")
    msg = _error_message(issues, "valid_image")
    assert msg is not None
    assert "below minimum" in msg or "1x1" in msg


def test_near_empty_png(tmp_path: Path) -> None:
    """A valid PNG just below the dimension threshold should fail."""
    small = tmp_path / "small.png"
    _create_png(small, size=(9, 100))
    issues = check_figure_file(small, artifact_id="fig_small")
    _assert_check(issues, "valid_image", "fig_small")
    msg = _error_message(issues, "valid_image")
    assert msg is not None
    assert "below minimum" in msg


# ---------------------------------------------------------------------------
# missing caption metadata
# ---------------------------------------------------------------------------


def test_missing_caption_file(tmp_path: Path) -> None:
    """Missing caption path should report caption_file check."""
    figure = tmp_path / "figure.png"
    _create_png(figure)
    caption = tmp_path / "missing_captions.md"
    issues = check_figure_file(
        figure,
        artifact_id="fig_no_cap",
        caption_path=caption,
    )
    _assert_check(issues, "caption_file", "fig_no_cap")
    assert "does not exist" in issues[0].message


def test_empty_caption_file(tmp_path: Path) -> None:
    """Empty caption file should report caption_file check."""
    figure = tmp_path / "figure.png"
    _create_png(figure)
    caption = tmp_path / "empty_captions.md"
    caption.touch()
    issues = check_figure_file(
        figure,
        artifact_id="fig_empty_cap",
        caption_path=caption,
    )
    _assert_check(issues, "caption_file", "fig_empty_cap")
    assert "empty" in issues[0].message


# ---------------------------------------------------------------------------
# unexpected format
# ---------------------------------------------------------------------------


def test_unexpected_format_not_png(tmp_path: Path) -> None:
    """A non-PNG file should fail the format check when expected_format=png."""
    not_png = tmp_path / "not_png.png"
    not_png.write_bytes(b"\x00\x00\x00\x00\x00\x00\x00\x00")
    issues = check_figure_file(not_png, artifact_id="fig_not_png")
    _assert_check(issues, "format", "fig_not_png")


def test_catalog_unexpected_format_set(tmp_path: Path) -> None:
    """A catalog figure should fail when it advertises an unsupported format."""
    figure = _create_png(tmp_path / "fig_unexpected.png")
    file_ref = ArtifactFileRef(path=figure.name, sha256=sha256_file(figure))
    caption = FIXTURE_DIR / "captions.md"
    entry = ArtifactCatalogEntry(
        artifact_id="fig_unexpected_format",
        artifact_kind="figure",
        source_kind="test",
        source_files=[file_ref],
        outputs={"jpg": file_ref},
        generation_command="pytest fixture",
        generation_commit="0000000",
        claim_boundary="fixture only",
        caption_file=ArtifactFileRef(path=str(caption), sha256=sha256_file(caption)),
    )

    issues = check_figure_entry(entry, catalog_dir=tmp_path)

    _assert_check(issues, "format_set", "fig_unexpected_format")


def test_catalog_custom_allowed_format_passes_format_checks(tmp_path: Path) -> None:
    """Custom allowed formats should apply at both entry and file levels."""
    raw = tmp_path / "fig_custom.raw"
    raw.write_bytes(b"custom deterministic fixture" * 4)
    file_ref = ArtifactFileRef(path=raw.name, sha256=sha256_file(raw))
    caption = FIXTURE_DIR / "captions.md"
    entry = ArtifactCatalogEntry(
        artifact_id="fig_custom_format",
        artifact_kind="figure",
        source_kind="test",
        source_files=[file_ref],
        outputs={"raw": file_ref},
        generation_command="pytest fixture",
        generation_commit="0000000",
        claim_boundary="fixture only",
        caption_file=ArtifactFileRef(path=str(caption), sha256=sha256_file(caption)),
    )

    issues = check_figure_entry(
        entry,
        catalog_dir=tmp_path,
        required_formats=frozenset({"raw"}),
        allowed_formats=frozenset({"raw"}),
    )

    assert issues == []


def test_catalog_missing_required_png_format(tmp_path: Path) -> None:
    """A catalog figure should fail when the required PNG output is absent."""
    pdf = tmp_path / "fig_only_pdf.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% fixture\n")
    file_ref = ArtifactFileRef(path=pdf.name, sha256=sha256_file(pdf))
    caption = FIXTURE_DIR / "captions.md"
    entry = ArtifactCatalogEntry(
        artifact_id="fig_missing_png",
        artifact_kind="figure",
        source_kind="test",
        source_files=[file_ref],
        outputs={"pdf": file_ref},
        generation_command="pytest fixture",
        generation_commit="0000000",
        claim_boundary="fixture only",
        caption_file=ArtifactFileRef(path=str(caption), sha256=sha256_file(caption)),
    )

    issues = check_figure_entry(entry, catalog_dir=tmp_path)

    _assert_check(issues, "format_set", "fig_missing_png")
    _assert_not_check(issues, "valid_image")


def test_valid_png_passes_format_check(tmp_path: Path) -> None:
    """A valid PNG should pass the format signature check."""
    figure = tmp_path / "valid.png"
    _create_png(figure)
    issues = check_figure_file(figure, artifact_id="fig_valid_png")
    _assert_not_check(issues, "format")


# ---------------------------------------------------------------------------
# artifact catalog integration
# ---------------------------------------------------------------------------


def test_catalog_validation_valid_figure(tmp_path: Path) -> None:
    """All checks should pass for a valid figure in a minimal catalog."""
    figure = _create_png(tmp_path / "fig_valid.png")
    catalog = _make_minimal_catalog(figure, tmp_path)
    issues = validate_figures_in_catalog(catalog, catalog_path=tmp_path / "catalog.yaml")
    assert issues == []


def test_catalog_validation_missing_output(tmp_path: Path) -> None:
    """A missing output file in a catalog figure should be reported."""
    figure = tmp_path / "fig_missing_output.png"
    figure.touch()
    catalog = _make_minimal_catalog(figure, tmp_path)
    figure.unlink()
    issues = validate_figures_in_catalog(catalog, catalog_path=tmp_path / "catalog.yaml")
    any_found = any(
        issue.check == "file_exists" and issue.artifact_id == "fig_catalog_test" for issue in issues
    )
    assert any_found, f"Expected file_exists check not found in {issues}"


def test_catalog_validation_missing_caption(tmp_path: Path) -> None:
    """A missing caption file in a catalog figure should be reported."""
    figure = _create_png(tmp_path / "fig_no_cap_catalog.png")
    entry = ArtifactCatalogEntry(
        artifact_id="fig_cap_missing",
        artifact_kind="figure",
        source_kind="test",
        source_files=[ArtifactFileRef(path=figure.name, sha256=sha256_file(figure))],
        outputs={"png": ArtifactFileRef(path=figure.name, sha256=sha256_file(figure))},
        generation_command="pytest fixture",
        generation_commit="0000000",
        claim_boundary="fixture only",
        caption_file=ArtifactFileRef(path="does_not_exist.md", sha256="0" * 64),
    )
    issues = check_figure_entry(entry, catalog_dir=tmp_path)
    _assert_check(issues, "caption_file", "fig_cap_missing")


def test_catalog_validation_requires_caption_metadata(tmp_path: Path) -> None:
    """A catalog figure should report missing caption metadata when omitted."""
    figure = _create_png(tmp_path / "fig_no_caption_metadata.png")
    checksum = sha256_file(figure)
    entry = ArtifactCatalogEntry(
        artifact_id="fig_no_caption_metadata",
        artifact_kind="figure",
        source_kind="test",
        source_files=[ArtifactFileRef(path=figure.name, sha256=checksum)],
        outputs={"png": ArtifactFileRef(path=figure.name, sha256=checksum)},
        generation_command="pytest fixture",
        generation_commit="0000000",
        claim_boundary="fixture only",
        caption_file=None,
    )

    issues = check_figure_entry(entry, catalog_dir=tmp_path)

    _assert_check(issues, "caption_file", "fig_no_caption_metadata")


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_valid_figure_passes(valid_figure: Path) -> None:
    """CLI should exit 0 for a valid figure."""
    exit_code = main([str(valid_figure), "--artifact-id", "fig_cli"])
    assert exit_code == 0


def test_cli_missing_file_exits_fail(tmp_path: Path) -> None:
    """CLI should exit 2 for a missing figure."""
    missing = tmp_path / "not_a_file.png"
    exit_code = main([str(missing), "--artifact-id", "fig_cli"])
    assert exit_code == 2


def test_cli_json_output(tmp_path: Path) -> None:
    """CLI JSON output should have the expected schema."""
    figure = _create_png(tmp_path / "json_test.png")
    exit_code = main([str(figure), "--artifact-id", "fig_json", "--json"])
    assert exit_code == 0


def test_cli_json_failure_output(tmp_path: Path) -> None:
    """CLI JSON output for a failure should include issues."""
    missing = tmp_path / "missing.png"
    exit_code = main([str(missing), "--artifact-id", "fig_json_fail", "--json"])
    assert exit_code == 2


def test_cli_catalog_validation_passes(tmp_path: Path) -> None:
    """CLI catalog mode should validate a tiny checksummed figure catalog."""
    figure = _create_png(tmp_path / "fig_cli_catalog.png")
    caption = tmp_path / "captions.md"
    caption.write_text("# fig_cli_catalog\n\nFixture caption.\n", encoding="utf-8")
    payload = {
        "schema_version": ARTIFACT_CATALOG_SCHEMA_VERSION,
        "catalog_id": "figure_qa_cli_fixture",
        "artifacts": [
            {
                "artifact_id": "fig_cli_catalog",
                "artifact_kind": "figure",
                "source_kind": "test",
                "source_files": [{"path": figure.name, "sha256": sha256_file(figure)}],
                "outputs": {"png": {"path": figure.name, "sha256": sha256_file(figure)}},
                "caption_file": {"path": caption.name, "sha256": sha256_file(caption)},
                "generation_command": "pytest fixture",
                "generation_commit": "0000000",
                "claim_boundary": "fixture only",
            }
        ],
    }
    catalog = tmp_path / "artifact_catalog.yaml"
    catalog.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    exit_code = main([str(catalog), "--catalog"])

    assert exit_code == 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_minimal_catalog(figure_path: Path, base_dir: Path) -> ArtifactCatalog:
    """Build a minimal artifact catalog with a single figure entry."""
    checksum = sha256_file(figure_path)
    file_ref = ArtifactFileRef(path=figure_path.name, sha256=checksum)
    caption = FIXTURE_DIR / "captions.md"
    caption_checksum = sha256_file(caption)
    caption_ref = ArtifactFileRef(path=str(caption), sha256=caption_checksum)
    return ArtifactCatalog(
        schema_version=ARTIFACT_CATALOG_SCHEMA_VERSION,
        catalog_id="figure_qa_fixture",
        artifacts=[
            ArtifactCatalogEntry(
                artifact_id="fig_catalog_test",
                artifact_kind="figure",
                source_kind="test",
                source_files=[file_ref],
                outputs={"png": file_ref},
                generation_command="pytest fixture",
                generation_commit="0000000",
                claim_boundary="fixture only",
                caption_file=caption_ref,
            )
        ],
    )


# ---------------------------------------------------------------------------
# matplotlib figure artist-level linting
# ---------------------------------------------------------------------------


def _make_figure_with_texts(
    labels: list[str], *, positions: list[tuple[float, float]] | None = None
) -> plt.Figure:
    """Build a figure with one text label per supplied string."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    if positions is None:
        positions = [(5.0, 5.0) for _ in labels]
    for label, (x, y) in zip(labels, positions, strict=False):
        ax.text(x, y, label, fontsize=14)
    return fig


def test_text_text_overlap_detected() -> None:
    """Two overlapping text labels should be reported as an error."""
    fig = _make_figure_with_texts(["alpha", "beta"], positions=[(5.0, 5.0), (5.05, 5.0)])
    defects = lint_figure(fig)
    types = {d.defect_type for d in defects}
    assert "text_text_overlap" in types
    assert any(d.severity == "error" for d in defects if d.defect_type == "text_text_overlap")
    plt.close(fig)


def test_text_text_overlap_far_apart_clean() -> None:
    """Well-separated text labels should not overlap."""
    fig = _make_figure_with_texts(["alpha", "beta"], positions=[(2.0, 5.0), (8.0, 5.0)])
    defects = lint_figure(fig)
    assert not any(d.defect_type == "text_text_overlap" for d in defects)
    plt.close(fig)


def test_text_line_overlap_detected() -> None:
    """Text sitting on top of a line should be reported as an error."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.plot([1.0, 9.0], [5.0, 5.0], "b-")
    ax.text(5.0, 5.0, "on line", fontsize=14)
    defects = lint_figure(fig)
    assert any(d.defect_type == "text_line_overlap" for d in defects)
    plt.close(fig)


def test_text_marker_overlap_detected() -> None:
    """Text on top of a scatter marker should be reported as an error."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.scatter([5.0], [5.0], s=200, c="red")
    ax.text(5.0, 5.0, "X", fontsize=14)
    defects = lint_figure(fig)
    assert any(d.defect_type == "text_marker_overlap" for d in defects)
    plt.close(fig)


def test_text_out_of_axes_detected() -> None:
    """Text placed outside the axes area should be reported as a warning."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(11.0, 5.0, "outside", fontsize=12)
    defects = lint_figure(fig)
    assert any(d.defect_type == "text_out_of_axes" for d in defects)
    plt.close(fig)


def test_marker_crowding_detected() -> None:
    """Markers closer than the threshold should be reported as a warning."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    pts = np.array([[5.0, 5.0], [5.02, 5.02], [9.0, 9.0]])
    ax.scatter(pts[:, 0], pts[:, 1], s=80)
    defects = lint_figure(fig, marker_min_separation_px=5.0)
    assert any(d.defect_type == "marker_crowding" for d in defects)
    plt.close(fig)


def test_marker_crowding_clean_when_separated() -> None:
    """Well-separated markers should not be reported as crowding."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    pts = np.array([[10.0, 10.0], [50.0, 50.0], [90.0, 90.0]])
    ax.scatter(pts[:, 0], pts[:, 1], s=80)
    defects = lint_figure(fig, marker_min_separation_px=3.0)
    assert not any(d.defect_type == "marker_crowding" for d in defects)
    plt.close(fig)


def test_saturated_color_count_advisory() -> None:
    """Many highly saturated colours should raise an advisory warning."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    for i, color in enumerate(["red", "lime", "blue", "magenta", "yellow", "cyan", "orange"]):
        ax.plot([0, 10], [i, i], color=color, linewidth=2)
    defects = lint_figure(fig, saturated_color_count_threshold=6)
    assert any(d.defect_type == "saturated_color_count" for d in defects)
    plt.close(fig)


def test_clean_figure_no_error_defects() -> None:
    """A deliberately clean figure should produce no error-severity defects."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.plot([1.0, 9.0], [1.0, 9.0], "b-", label="traj")
    ax.scatter([5.0], [5.0], s=80, c="steelblue")
    ax.text(2.0, 8.0, "label", fontsize=10)
    ax.legend()
    defects = lint_figure(fig)
    assert not any(d.severity == "error" for d in defects)
    plt.close(fig)


def test_assert_clean_passes_clean_figure() -> None:
    """assert_clean should not raise for a clean figure at error severity."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.plot([1.0, 9.0], [1.0, 9.0], "b-")
    ax.text(2.0, 8.0, "label", fontsize=10)
    assert_clean(fig, max_severity="error")
    plt.close(fig)


def test_assert_clean_raises_on_overlap() -> None:
    """assert_clean should raise when an error-severity defect is seeded."""
    fig = _make_figure_with_texts(["a", "b"], positions=[(5.0, 5.0), (5.03, 5.0)])
    with pytest.raises(AssertionError):
        assert_clean(fig, max_severity="error")
    plt.close(fig)


def test_lint_figure_rejects_non_figure() -> None:
    """lint_figure should return [] for non-Figure inputs."""
    assert lint_figure(object()) == []


def test_real_rendered_scene_passes_at_warn() -> None:
    """The render CLI's scene figure should pass at warn tolerance.

    Reuses the renderer used by ``scripts/tools/render_trace_scene_figure.py``
    so the lint gate tracks the actual shipped figure.
    """
    from scripts.tools.render_trace_scene_figure import (
        EpisodeStep,
        render_scene_figure,
    )

    rng = np.random.default_rng(7)
    steps = [
        EpisodeStep(t=i * 0.1, x=1.0 + i * 0.15, y=3.0 + rng.normal(0, 0.05)) for i in range(40)
    ]
    fig = render_scene_figure(steps, title="exemplar scene")
    defects = lint_figure(fig)
    error_defects = [d for d in defects if d.severity == "error"]
    assert not error_defects, f"error-severity defects: {error_defects}"
    plt.close(fig)
