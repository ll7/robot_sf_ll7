#!/usr/bin/env python3
"""Render one diagnostic multi-panel dossier from ``simulation_trace_export.v1``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExportValidationError,
)
from robot_sf.analysis_workbench.trace_dossier_renderer import (
    DEFAULT_DOSSIER_FIXTURE,
    TraceDossierRenderError,
    render_trace_dossier,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        default=DEFAULT_DOSSIER_FIXTURE,
        help="Input simulation_trace_export.v1 JSON path.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output PNG dossier path.")
    parser.add_argument(
        "--svg",
        type=Path,
        default=None,
        help="Optional deterministic SVG dossier path; supply with --pdf and --caption.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Optional deterministic PDF dossier path; supply with --svg and --caption.",
    )
    parser.add_argument(
        "--caption",
        type=Path,
        default=None,
        help="Optional Markdown caption path; supply with --svg and --pdf.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Output deterministic trace_dossier_manifest.v1 JSON path.",
    )
    parser.add_argument(
        "--command",
        default=None,
        help="Generation command recorded in the manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the diagnostic trace dossier renderer CLI.

    Returns:
        Process exit code.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)
    command = args.command or "uv run python scripts/tools/render_trace_dossier.py"
    try:
        result = render_trace_dossier(
            args.trace,
            output_png=args.output,
            output_svg=args.svg,
            output_pdf=args.pdf,
            caption_path=args.caption,
            manifest_path=args.manifest,
            command=command,
        )
    except (
        OSError,
        json.JSONDecodeError,
        SimulationTraceExportValidationError,
        TraceDossierRenderError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"wrote trace dossier {result.png_path}")
    if result.svg_path is not None:
        print(f"wrote trace dossier {result.svg_path}")
    if result.pdf_path is not None:
        print(f"wrote trace dossier {result.pdf_path}")
    if result.caption_path is not None:
        print(f"wrote trace dossier caption {result.caption_path}")
    print(f"wrote trace dossier manifest {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
