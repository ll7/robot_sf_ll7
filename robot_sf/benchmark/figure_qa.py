"""Deterministic QA checks for generated figure artifacts.

Validates figure PNG files for common issues: missing file, empty/near-empty
image, wrong format, and missing caption metadata.  Designed for CI integration
and test fixtures; does not require benchmark inputs or pixel-perfect golden
comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from robot_sf.benchmark.artifact_catalog import (
    ArtifactCatalog,
    ArtifactCatalogEntry,
    load_artifact_catalog,
)

_MIN_PNG_DIMENSION = 10
_MIN_PNG_FILE_SIZE = 67
_DEFAULT_REQUIRED_FORMATS = frozenset({"png"})
_DEFAULT_ALLOWED_FORMATS = frozenset({"png", "pdf", "svg"})
_PDF_SIGNATURE = b"%PDF-"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True, slots=True)
class FigureQA:
    """One deterministic QA check result for a figure artifact.

    Attributes:
        artifact_id: Stable identifier from the artifact catalog or caller.
        check:       Short check name (e.g. ``file_exists``, ``valid_image``).
        message:     Human-readable description of the issue.
    """

    artifact_id: str
    check: str
    message: str


def check_figure_file(
    path: Path,
    *,
    artifact_id: str = "<unknown>",
    expected_format: str = "png",
    caption_path: Path | None = None,
    allowed_formats: frozenset[str] = _DEFAULT_ALLOWED_FORMATS,
) -> list[FigureQA]:
    """Run deterministic QA checks on a single figure file.

    Checks performed (in order):

    1. File exists and is a regular file.
    2. File starts with the expected format signature (PNG by default).
    3. File is a valid image with dimensions >= ``_MIN_PNG_DIMENSION``.
    4. File size meets the minimum threshold.
    5. Optional caption file exists and is non-empty.

    Args:
        path:             Absolute path to the figure file.
        artifact_id:      Stable identifier used in failure messages.
        expected_format:  Expected image format (``"png"``, ``"pdf"``, etc.).
        caption_path:     Optional path to a caption / description file.
        allowed_formats:  Output formats permitted for figure artifacts.

    Returns:
        List of ``FigureQA`` results.  Empty means all checks passed.
    """
    issues: list[FigureQA] = []

    # --- check 1: file exists and is a regular file ---
    if not path.exists():
        issues.append(FigureQA(artifact_id, "file_exists", f"file does not exist: {path}"))
        return issues
    if not path.is_file():
        issues.append(FigureQA(artifact_id, "file_exists", f"path is not a regular file: {path}"))
        return issues

    normalized_format = expected_format.lower()
    if normalized_format not in allowed_formats:
        issues.append(
            FigureQA(
                artifact_id,
                "format",
                f"unsupported figure format '{expected_format}'",
            )
        )
        return issues

    # --- check 2: format signature ---
    if normalized_format == "png":
        _check_png_signature(path, artifact_id, issues)
    elif normalized_format == "pdf":
        _check_pdf_signature(path, artifact_id, issues)

    # --- check 3: valid raster image with reasonable content ---
    if normalized_format == "png":
        _check_image_content(path, artifact_id, issues)

    # --- check 4: file size ---
    _check_file_size(path, artifact_id, issues)

    # --- check 5: caption file ---
    if caption_path is not None:
        _check_caption_file(caption_path, artifact_id, issues)

    return issues


def check_figure_entry(
    entry: ArtifactCatalogEntry,
    *,
    catalog_dir: Path,
    required_formats: frozenset[str] = _DEFAULT_REQUIRED_FORMATS,
    allowed_formats: frozenset[str] = _DEFAULT_ALLOWED_FORMATS,
) -> list[FigureQA]:
    """Run QA checks on a single catalog figure entry.

    Args:
        entry:       Typed artifact catalog entry (``artifact_kind`` must be
                     ``"figure"``).
        catalog_dir: Directory used to resolve relative file paths.
        required_formats: Output formats that every figure entry must include.
        allowed_formats:  Output formats permitted for figure entries.

    Returns:
        List of ``FigureQA`` results.  Empty means all checks passed.
    """
    if entry.artifact_kind != "figure":
        return []

    issues: list[FigureQA] = []
    output_formats = {output_key.lower() for output_key in entry.outputs}
    for required_format in sorted(required_formats):
        if required_format not in output_formats:
            issues.append(
                FigureQA(
                    entry.artifact_id,
                    "format_set",
                    f"required output format '{required_format}' is missing",
                )
            )
    unexpected_formats = output_formats - allowed_formats
    if unexpected_formats:
        issues.append(
            FigureQA(
                entry.artifact_id,
                "format_set",
                "unexpected output formats: " + ", ".join(sorted(unexpected_formats)),
            )
        )

    caption_path: Path | None = None
    if entry.caption_file is None:
        issues.append(
            FigureQA(
                entry.artifact_id,
                "caption_file",
                "figure artifact is missing caption metadata",
            )
        )
    else:
        caption_path = (catalog_dir / entry.caption_file.path).resolve()

    for output_key, file_ref in entry.outputs.items():
        figure_path = (catalog_dir / file_ref.path).resolve()
        issues.extend(
            check_figure_file(
                figure_path,
                artifact_id=entry.artifact_id,
                expected_format=output_key,
                caption_path=caption_path,
                allowed_formats=allowed_formats,
            )
        )
    return issues


def validate_figures_in_catalog(
    catalog: ArtifactCatalog,
    *,
    catalog_path: Path | None = None,
    required_formats: frozenset[str] = _DEFAULT_REQUIRED_FORMATS,
    allowed_formats: frozenset[str] = _DEFAULT_ALLOWED_FORMATS,
) -> list[FigureQA]:
    """Run QA checks on all figure entries in an artifact catalog.

    Args:
        catalog:      Typed artifact catalog metadata.
        catalog_path: Path to the catalog file (used to resolve relative file
                      paths).  Falls back to ``Path.cwd()`` when ``None``.
        required_formats: Output formats that every figure entry must include.
        allowed_formats:  Output formats permitted for figure entries.

    Returns:
        List of ``FigureQA`` results.  Empty means all checks passed.
    """
    catalog_dir = catalog_path.parent.resolve() if catalog_path else Path.cwd()
    issues: list[FigureQA] = []
    for entry in catalog.artifacts:
        issues.extend(
            check_figure_entry(
                entry,
                catalog_dir=catalog_dir,
                required_formats=required_formats,
                allowed_formats=allowed_formats,
            )
        )
    return issues


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_png_signature(path: Path, artifact_id: str, issues: list[FigureQA]) -> None:
    """Verify the file starts with a valid 8-byte PNG signature."""
    _check_file_signature(
        path,
        artifact_id,
        _PNG_SIGNATURE,
        "file does not have a valid PNG signature",
        issues,
    )


def _check_pdf_signature(path: Path, artifact_id: str, issues: list[FigureQA]) -> None:
    """Verify the file starts with the PDF magic bytes."""
    _check_file_signature(
        path,
        artifact_id,
        _PDF_SIGNATURE,
        "file does not have a valid PDF signature",
        issues,
    )


def _check_file_signature(
    path: Path,
    artifact_id: str,
    signature: bytes,
    mismatch_message: str,
    issues: list[FigureQA],
) -> None:
    """Verify the file starts with the expected signature bytes."""
    try:
        with path.open("rb") as handle:
            header = handle.read(len(signature))
    except OSError as exc:
        issues.append(FigureQA(artifact_id, "format", f"cannot read file for format check: {exc}"))
        return
    if header != signature:
        issues.append(FigureQA(artifact_id, "format", mismatch_message))


def _check_image_content(path: Path, artifact_id: str, issues: list[FigureQA]) -> None:
    """Verify the image opens correctly and has reasonable dimensions."""
    if not path.is_file():
        issues.append(FigureQA(artifact_id, "valid_image", "path is not a file"))
        return
    try:
        with Image.open(path) as img:
            width, height = img.size
            img.verify()
    except (OSError, ValueError) as exc:
        issues.append(FigureQA(artifact_id, "valid_image", f"cannot verify image: {exc}"))
        return

    if width < _MIN_PNG_DIMENSION or height < _MIN_PNG_DIMENSION:
        issues.append(
            FigureQA(
                artifact_id,
                "valid_image",
                f"image dimensions ({width}x{height}) are below minimum "
                f"({_MIN_PNG_DIMENSION}x{_MIN_PNG_DIMENSION})",
            )
        )


def _check_file_size(path: Path, artifact_id: str, issues: list[FigureQA]) -> None:
    """Verify the file size meets the minimum threshold."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        issues.append(FigureQA(artifact_id, "file_size", f"cannot stat file: {exc}"))
        return
    if size < _MIN_PNG_FILE_SIZE:
        issues.append(
            FigureQA(
                artifact_id,
                "file_size",
                f"file size ({size} bytes) is below minimum ({_MIN_PNG_FILE_SIZE} bytes)",
            )
        )


def _check_caption_file(path: Path, artifact_id: str, issues: list[FigureQA]) -> None:
    """Verify the caption file exists and contains non-whitespace content."""
    if not path.exists():
        issues.append(FigureQA(artifact_id, "caption_file", f"caption file does not exist: {path}"))
        return
    if not path.is_file():
        issues.append(
            FigureQA(
                artifact_id,
                "caption_file",
                f"caption path is not a regular file: {path}",
            )
        )
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        issues.append(FigureQA(artifact_id, "caption_file", f"cannot read caption file: {exc}"))
        return
    if not text.strip():
        issues.append(FigureQA(artifact_id, "caption_file", f"caption file is empty: {path}"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the figure artifact QA argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Validate figure artifacts with deterministic QA checks."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a figure file or artifact catalog (use --catalog).",
    )
    parser.add_argument(
        "--catalog",
        action="store_true",
        help="Treat ``path`` as an artifact catalog YAML/JSON file.",
    )
    parser.add_argument(
        "--artifact-id",
        default=None,
        help="Artifact identifier for single-file validation (defaults to the filename stem).",
    )
    parser.add_argument(
        "--caption",
        type=Path,
        default=None,
        help="Caption file path for single-file validation.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON validation report.",
    )
    parser.add_argument(
        "--require-format",
        action="append",
        default=None,
        help="Required catalog output format; may be repeated. Defaults to png.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate figure artifacts and return a shell-friendly exit code.

    Returns:
        ``0`` when all checks pass, ``2`` when any check fails.
    """
    args = build_arg_parser().parse_args(argv)

    if args.catalog:
        catalog = load_artifact_catalog(args.path)
        required_formats = (
            frozenset(item.lower() for item in args.require_format)
            if args.require_format
            else _DEFAULT_REQUIRED_FORMATS
        )
        issues = validate_figures_in_catalog(
            catalog,
            catalog_path=args.path,
            required_formats=required_formats,
        )
    else:
        artifact_id = args.artifact_id or args.path.stem
        issues = check_figure_file(
            args.path,
            artifact_id=artifact_id,
            caption_path=args.caption,
        )

    if args.json:
        sys.stdout.write(
            json.dumps(
                {
                    "schema": "figure_qa.v1",
                    "target": str(args.path),
                    "ok": not issues,
                    "issues": [asdict(issue) for issue in issues],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    elif issues:
        for issue in issues:
            sys.stdout.write(f"[{issue.artifact_id}] {issue.check}: {issue.message}\n")
    else:
        sys.stdout.write(f"All figure QA checks passed: {args.path}\n")

    return 0 if not issues else 2


# ---------------------------------------------------------------------------
# Matplotlib Figure Artist-Level Linting
# ---------------------------------------------------------------------------


_DEFECT_TYPE_TEXT_TEXT_OVERLAP = "text_text_overlap"
_DEFECT_TYPE_TEXT_LINE_OVERLAP = "text_line_overlap"
_DEFECT_TYPE_TEXT_MARKER_OVERLAP = "text_marker_overlap"
_DEFECT_TYPE_TEXT_OUT_OF_AXES = "text_out_of_axes"
_DEFECT_TYPE_MARKER_CROWDING = "marker_crowding"
_DEFECT_TYPE_SATURATED_COLOR_COUNT = "saturated_color_count"

_SEVERITY_ERROR = "error"
_SEVERITY_WARN = "warn"

_TEXT_OVERLAP_TOLERANCE_PX = 2.0
_TEXT_LINE_OVERLAP_TOLERANCE_PX = 1.0
_TEXT_MARKER_OVERLAP_TOLERANCE_PX = 1.0
_MARKER_MIN_SEPARATION_PX = 3.0
_SATURATED_COLOR_SATURATION_THRESHOLD = 0.85
_SATURATED_COLOR_COUNT_THRESHOLD = 6


@dataclass(frozen=True, slots=True)
class FigureDefect:
    """One legibility defect detected in a matplotlib figure.

    Attributes:
        defect_type: Short defect identifier (e.g. ``text_text_overlap``).
        severity:    ``"error"`` or ``"warn"``.
        message:     Human-readable description.
        location:    Display-space ``(x, y)`` center of the offending artist,
                     or ``None`` when not tied to a single point.
    """

    defect_type: str
    severity: str
    message: str
    location: tuple[float, float] | None = None


def lint_figure(
    fig: object,
    *,
    text_overlap_tolerance_px: float = _TEXT_OVERLAP_TOLERANCE_PX,
    text_line_overlap_tolerance_px: float = _TEXT_LINE_OVERLAP_TOLERANCE_PX,
    text_marker_overlap_tolerance_px: float = _TEXT_MARKER_OVERLAP_TOLERANCE_PX,
    marker_min_separation_px: float = _MARKER_MIN_SEPARATION_PX,
    saturated_color_saturation_threshold: float = _SATURATED_COLOR_SATURATION_THRESHOLD,
    saturated_color_count_threshold: int = _SATURATED_COLOR_COUNT_THRESHOLD,
) -> list[FigureDefect]:
    """Inspect a live matplotlib ``Figure`` for legibility defects.

    Operates on the renderer's ``get_window_extent`` bounding boxes so
    detection is exact in display space rather than pixel-heuristic.

    Args:
        fig: A matplotlib ``Figure`` instance.
        text_overlap_tolerance_px: Minimum gap in pixels before two text
            artists are considered overlapping.
        text_line_overlap_tolerance_px: Gap tolerance for text-line overlaps.
        text_marker_overlap_tolerance_px: Gap tolerance for text-marker overlaps.
        marker_min_separation_px: Minimum allowed separation between scatter
            marker centres in pixels.
        saturated_color_saturation_threshold: HSV saturation above which a
            colour counts as highly saturated (0–1).
        saturated_color_count_threshold: Number of highly saturated colours
            at or above which an advisory warning is emitted.

    Returns:
        List of ``FigureDefect`` instances.  Empty means no defects detected.
    """
    try:
        import matplotlib.figure as _mfig  # noqa: PLC0415
    except ImportError:
        return []

    if not isinstance(fig, _mfig.Figure):
        return []

    renderer = fig.canvas.get_renderer()
    if renderer is None:
        return []

    defects: list[FigureDefect] = []

    for ax in fig.axes:
        ax_bbox = ax.get_window_extent(renderer)
        _check_text_out_of_axes(ax, renderer, ax_bbox, defects)

        text_artists = [child for child in ax.get_children() if _is_meaningful_text(child)]
        _check_text_text_overlap(text_artists, renderer, text_overlap_tolerance_px, defects)
        _check_text_line_overlap(
            ax, text_artists, renderer, ax_bbox, text_line_overlap_tolerance_px, defects
        )
        _check_text_marker_overlap(
            ax, text_artists, renderer, ax_bbox, text_marker_overlap_tolerance_px, defects
        )
        _check_marker_crowding(ax, renderer, ax_bbox, marker_min_separation_px, defects)

    _check_saturated_color_count(
        fig, saturated_color_saturation_threshold, saturated_color_count_threshold, defects
    )

    return defects


def assert_clean(fig: object, *, max_severity: str = _SEVERITY_ERROR) -> None:
    """Assert that *fig* has no defects at or above *max_severity*.

    Args:
        fig: A matplotlib ``Figure`` instance.
        max_severity: ``"error"`` (default) or ``"warn"``.

    Raises:
        AssertionError: When at least one defect meets the severity threshold.
    """
    severity_order = {_SEVERITY_WARN: 0, _SEVERITY_ERROR: 1}
    threshold = severity_order.get(max_severity, 1)
    defects = lint_figure(
        fig,
        text_overlap_tolerance_px=0.0,
        text_line_overlap_tolerance_px=0.0,
        text_marker_overlap_tolerance_px=0.0,
        marker_min_separation_px=0.0,
    )
    failing = [d for d in defects if severity_order.get(d.severity, 1) >= threshold]
    summary = "; ".join(f"[{d.defect_type}] {d.message}" for d in failing)
    assert not failing, (
        f"Figure has {len(failing)} defect(s) at severity >= {max_severity}: {summary}"
    )


# ---------------------------------------------------------------------------
# Internal linting helpers
# ---------------------------------------------------------------------------


def _is_text_artist(artist: object) -> bool:
    """Return ``True`` when *artist* is a matplotlib ``Text`` instance."""
    try:
        import matplotlib.text as _mtext  # noqa: PLC0415
    except ImportError:
        return False
    return isinstance(artist, _mtext.Text)


def _is_legend_text(artist: object) -> bool:
    """Return ``True`` when *artist* is a ``Text`` owned by an axes legend.

    Legend text sits in a fixed (often crowded) box and overlaps are expected /
    intended, so it must be excluded from legibility overlap checks.
    """
    parent = getattr(artist, "get_parent", lambda: None)()
    if parent is None:
        return False
    from matplotlib.legend import Legend  # noqa: PLC0415

    return isinstance(parent, Legend)


def _is_meaningful_text(artist: object) -> bool:
    """Return ``True`` for visible, non-empty text not owned by a legend."""
    return (
        _is_text_artist(artist)
        and artist.get_visible()
        and bool(artist.get_text())
        and not _is_legend_text(artist)
    )


def _bbox_center(bbox: object) -> tuple[float, float]:
    """Return the display-space centre ``(x, y)`` of a ``Bbox``."""
    return float((bbox.x0 + bbox.x1) / 2.0), float((bbox.y0 + bbox.y1) / 2.0)


def _bboxes_overlap(a: object, b: object, tolerance: float) -> bool:
    """Return ``True`` when two ``Bbox`` objects overlap within *tolerance*."""
    return bool(
        (a.x0 - tolerance) < b.x1
        and (a.x1 + tolerance) > b.x0
        and (a.y0 - tolerance) < b.y1
        and (a.y1 + tolerance) > b.y0
    )


def _check_text_out_of_axes(
    ax: object,
    renderer: object,
    ax_bbox: object,
    defects: list[FigureDefect],
) -> None:
    """Detect text artists whose bounding box extends outside the axes area."""
    for child in ax.get_children():
        if not _is_meaningful_text(child):
            continue
        text_bbox = child.get_window_extent(renderer)
        if (
            text_bbox.x0 < ax_bbox.x0
            or text_bbox.x1 > ax_bbox.x1
            or text_bbox.y0 < ax_bbox.y0
            or text_bbox.y1 > ax_bbox.y1
        ):
            cx, cy = _bbox_center(text_bbox)
            label = child.get_text()[:60] if child.get_text() else "<empty>"
            defects.append(
                FigureDefect(
                    defect_type=_DEFECT_TYPE_TEXT_OUT_OF_AXES,
                    severity=_SEVERITY_WARN,
                    message=f"Text {label!r} extends outside axes bounds.",
                    location=(cx, cy),
                )
            )


def _check_text_text_overlap(
    text_artists: list[object],
    renderer: object,
    tolerance: float,
    defects: list[FigureDefect],
) -> None:
    """Detect overlapping text artists."""
    bboxes = [(artist, artist.get_window_extent(renderer)) for artist in text_artists]
    for i, (a, bbox_a) in enumerate(bboxes):
        for b, bbox_b in bboxes[i + 1 :]:
            if _bboxes_overlap(bbox_a, bbox_b, tolerance):
                cx, cy = _bbox_center(bbox_a)
                a_label = a.get_text()[:40] if a.get_text() else "<empty>"
                b_label = b.get_text()[:40] if b.get_text() else "<empty>"
                defects.append(
                    FigureDefect(
                        defect_type=_DEFECT_TYPE_TEXT_TEXT_OVERLAP,
                        severity=_SEVERITY_ERROR,
                        message=f"Text {a_label!r} overlaps {b_label!r}.",
                        location=(cx, cy),
                    )
                )


def _check_text_line_overlap(
    ax: object,
    text_artists: list[object],
    renderer: object,
    ax_bbox: object,
    tolerance: float,
    defects: list[FigureDefect],
) -> None:
    """Detect text artists that overlap line artists."""
    from matplotlib.lines import Line2D  # noqa: PLC0415

    lines = [
        child for child in ax.get_children() if isinstance(child, Line2D) and child.get_visible()
    ]
    if not lines or not text_artists:
        return

    for text_artist in text_artists:
        text_bbox = text_artist.get_window_extent(renderer)
        for line in lines:
            if _line_bbox_overlaps_text(line, text_bbox, ax, renderer, tolerance):
                cx, cy = _bbox_center(text_bbox)
                label = text_artist.get_text()[:40] if text_artist.get_text() else "<empty>"
                defects.append(
                    FigureDefect(
                        defect_type=_DEFECT_TYPE_TEXT_LINE_OVERLAP,
                        severity=_SEVERITY_ERROR,
                        message=f"Text {label!r} overlaps a line artist.",
                        location=(cx, cy),
                    )
                )
                break


def _line_bbox_overlaps_text(
    line: object,
    text_bbox: object,
    ax: object,
    renderer: object,
    tolerance: float,
) -> bool:
    """Return ``True`` when a line segment enters the text bounding box."""

    xdata = line.get_xdata()
    ydata = line.get_ydata()
    if len(xdata) < 2:
        return False

    transform = ax.transData
    for idx in range(len(xdata) - 1):
        p0 = transform.transform((float(xdata[idx]), float(ydata[idx])))
        p1 = transform.transform((float(xdata[idx + 1]), float(ydata[idx + 1])))
        if _segment_intersects_bbox(p0, p1, text_bbox, tolerance):
            return True
    return False


def _segment_intersects_bbox(
    p0: tuple[float, float],
    p1: tuple[float, float],
    bbox: object,
    tolerance: float,
) -> bool:
    """Return ``True`` when the segment ``p0→p1`` enters *bbox* (with tolerance).

    Uses the Cyrus–Beck/Liang–Barsky parametric clipping approach to test
    whether any portion of the segment lies inside the expanded bounding box.
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]

    t_enter = 0.0
    t_exit = 1.0

    for p, q in [
        (-dx, p0[0] - (bbox.x0 - tolerance)),
        (dx, (bbox.x1 + tolerance) - p0[0]),
        (-dy, p0[1] - (bbox.y0 - tolerance)),
        (dy, (bbox.y1 + tolerance) - p0[1]),
    ]:
        if abs(p) < 1e-12:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                t_enter = max(t_enter, t)
            else:
                t_exit = min(t_exit, t)
            if t_enter > t_exit:
                return False

    return True


def _check_text_marker_overlap(
    ax: object,
    text_artists: list[object],
    renderer: object,
    ax_bbox: object,
    tolerance: float,
    defects: list[FigureDefect],
) -> None:
    """Detect text artists that overlap scatter/collection markers."""
    from matplotlib.collections import PathCollection  # noqa: PLC0415

    collections = [
        child
        for child in ax.get_children()
        if isinstance(child, PathCollection) and child.get_visible()
    ]
    if not collections or not text_artists:
        return

    transform = ax.transData
    for text_artist in text_artists:
        text_bbox = text_artist.get_window_extent(renderer)
        for coll in collections:
            offsets = coll.get_offsets()
            if len(offsets) == 0:
                continue
            for point in offsets:
                px, py = transform.transform((float(point[0]), float(point[1])))
                if _point_in_bbox(px, py, text_bbox, tolerance):
                    cx, cy = _bbox_center(text_bbox)
                    label = text_artist.get_text()[:40] if text_artist.get_text() else "<empty>"
                    defects.append(
                        FigureDefect(
                            defect_type=_DEFECT_TYPE_TEXT_MARKER_OVERLAP,
                            severity=_SEVERITY_ERROR,
                            message=f"Text {label!r} overlaps a scatter marker.",
                            location=(cx, cy),
                        )
                    )
                    break
            else:
                continue
            break


def _point_in_bbox(px: float, py: float, bbox: object, tolerance: float) -> bool:
    """Return ``True`` when ``(px, py)`` is inside *bbox* expanded by *tolerance*."""
    return bool(
        (bbox.x0 - tolerance) <= px <= (bbox.x1 + tolerance)
        and (bbox.y0 - tolerance) <= py <= (bbox.y1 + tolerance)
    )


def _check_marker_crowding(
    ax: object,
    renderer: object,
    ax_bbox: object,
    min_separation_px: float,
    defects: list[FigureDefect],
) -> None:
    """Detect scatter markers closer than *min_separation_px* in display space."""
    import numpy as np  # noqa: PLC0415
    from matplotlib.collections import PathCollection  # noqa: PLC0415

    collections = [
        child
        for child in ax.get_children()
        if isinstance(child, PathCollection) and child.get_visible()
    ]
    transform = ax.transData
    all_points: list[tuple[float, float]] = []
    for coll in collections:
        offsets = coll.get_offsets()
        for point in offsets:
            px, py = transform.transform((float(point[0]), float(point[1])))
            all_points.append((px, py))

    if len(all_points) < 2:
        return

    pts = np.array(all_points)
    reported: set[int] = set()
    for i in range(len(pts)):
        if i in reported:
            continue
        dists = np.sqrt(np.sum((pts[i + 1 :] - pts[i]) ** 2, axis=1))
        close_indices = np.where(dists < min_separation_px)[0]
        if len(close_indices) > 0:
            reported.add(i)
            defects.append(
                FigureDefect(
                    defect_type=_DEFECT_TYPE_MARKER_CROWDING,
                    severity=_SEVERITY_WARN,
                    message=(
                        f"Marker at ({pts[i][0]:.1f}, {pts[i][1]:.1f}) has "
                        f"{len(close_indices)} neighbour(s) closer than "
                        f"{min_separation_px:.1f}px."
                    ),
                    location=(float(pts[i][0]), float(pts[i][1])),
                )
            )


def _check_saturated_color_count(
    fig: object,
    saturation_threshold: float,
    count_threshold: int,
    defects: list[FigureDefect],
) -> None:
    """Warn when too many highly-saturated colours are used."""
    import matplotlib.colors as mcolors  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    colors_seen: set[str] = set()
    saturated_count = 0

    for ax in fig.axes:
        for rgba in _iter_artist_colors(ax):
            key_str = str(tuple(float(c) for c in rgba))
            if key_str not in colors_seen:
                colors_seen.add(key_str)
                if _is_saturated_rgba(rgba, saturation_threshold, mcolors, np):
                    saturated_count += 1

    if saturated_count >= count_threshold:
        defects.append(
            FigureDefect(
                defect_type=_DEFECT_TYPE_SATURATED_COLOR_COUNT,
                severity=_SEVERITY_WARN,
                message=(
                    f"Figure uses {saturated_count} highly-saturated colour(s) "
                    f"(threshold: {count_threshold}). Consider reducing saturation."
                ),
                location=None,
            )
        )


def _iter_artist_colors(ax: object) -> list[object]:
    """Yield all RGBA colour tuples used by lines and scatter collections.

    Returns:
        List of RGBA tuples/arrays collected from visible line and scatter
        artists in the axes.
    """
    import matplotlib.colors as mcolors  # noqa: PLC0415
    from matplotlib.collections import PathCollection  # noqa: PLC0415

    colors: list[object] = []
    for line in ax.get_lines():
        try:
            colors.append(mcolors.to_rgba(line.get_color()))
        except (ValueError, TypeError):
            continue
    for coll in ax.get_children():
        if not isinstance(coll, PathCollection):
            continue
        colors.extend(coll.get_facecolors())
    return colors


def _is_saturated_rgba(
    rgba: object,
    threshold: float,
    mcolors: object,
    np: object,
) -> bool:
    """Return ``True`` when an RGBA array exceeds the saturation threshold."""
    import colorsys  # noqa: PLC0415

    r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
    _, s, _ = colorsys.rgb_to_hsv(r, g, b)
    return s > threshold


__all__ = [
    "FigureDefect",
    "FigureQA",
    "assert_clean",
    "build_arg_parser",
    "check_figure_entry",
    "check_figure_file",
    "lint_figure",
    "main",
    "validate_figures_in_catalog",
]
