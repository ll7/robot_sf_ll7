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


__all__ = [
    "FigureQA",
    "build_arg_parser",
    "check_figure_entry",
    "check_figure_file",
    "main",
    "validate_figures_in_catalog",
]
