#!/usr/bin/env python3
"""Render a deterministic, evidence-gated Chapter 7 case dossier."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from robot_sf.benchmark.case_dossier_figure import CaseDossierError, render_case_dossier

_BUNDLE_PATHS = {
    "svg": "svg_path",
    "pdf": "pdf_path",
    "caption": "caption_path",
    "sidecar": "sidecar_path",
    "manifest": "manifest_path",
    "artifact_catalog": "catalog_path",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the case-dossier command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Versioned case_dossier_input.v1 JSON manifest.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory for SVG, PDF, caption, sidecar, manifest, and artifact catalog.",
    )
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help="Render once more in isolation and require every output byte to match.",
    )
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Render one dossier, optionally proving byte determinism."""

    args = build_parser().parse_args(argv)
    try:
        bundle = render_case_dossier(args.input, args.out_dir)
        deterministic: bool | None = None
        if args.check_determinism:
            with tempfile.TemporaryDirectory(prefix="case-dossier-determinism-") as raw_dir:
                comparison = render_case_dossier(args.input, Path(raw_dir) / "comparison")
                mismatches = [
                    name
                    for name, attribute in _BUNDLE_PATHS.items()
                    if getattr(bundle, attribute).read_bytes()
                    != getattr(comparison, attribute).read_bytes()
                ]
            if mismatches:
                raise CaseDossierError(
                    "case_dossier_nondeterministic",
                    ", ".join(sorted(mismatches)),
                )
            deterministic = True
    except (CaseDossierError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = {
        "schema_version": "case_dossier_cli_report.v1",
        "deterministic": deterministic,
        "manifest": bundle.manifest_path.as_posix(),
        "output_sha256": {
            name: _sha256(getattr(bundle, attribute))
            for name, attribute in sorted(_BUNDLE_PATHS.items())
        },
        "scientific_admission": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
