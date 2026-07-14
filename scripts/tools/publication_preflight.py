#!/usr/bin/env python3
"""Audit a built publication bundle directory with the final release preflight (issue #5530).

This is a CPU-only, no-simulation check. It reads an already-exported
``<name>_publication_bundle`` directory (produced by
:func:`robot_sf.benchmark.artifact_publication.export_publication_bundle`) and
fails closed when the bundle is internally self-inconsistent:

1. ``release/release_result.json`` and ``reports/campaign_summary.json`` must
   agree on status, evidence_status, total_episodes, and successful_runs.
2. every ``checksums.sha256`` entry must verify against a file present relative
   to the bundle root, and every manifest-listed file must be signed (so
   ``sha256sum -c checksums.sha256`` works from the bundle root).
3. episode-ledger ``software_commit`` values must equal the publication manifest's
   repository commit, unless the manifest carries an explicit non-runtime-diff
   explanation.
4. ``publication_channels`` must not retain DOI/release-tag placeholders.
5. ambiguous goal-reached + timeout rows must carry timing evidence or a note.

The checksums-from-bundle-root requirement is what makes the documented
one-command verification possible; prior 0.0.3 bundles shipped checksums
relative to ``payload/`` and therefore failed ``sha256sum -c`` from the root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.artifact_publication import (
    PublicationPreflightError,
    verify_publication_bundle_preflight,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        required=True,
        type=Path,
        help="Built publication bundle directory (contains payload/, "
        "publication_manifest.json, checksums.sha256).",
    )
    parser.add_argument(
        "--no-require-release-reconciliation",
        action="store_true",
        help="Skip the blocking requirement that release_result.json + campaign_summary.json "
        "exist; only fail when they are present but disagree (useful for ad-hoc bundles).",
    )
    parser.add_argument("--output", type=Path, help="Optional path to write the JSON report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the publication preflight CLI and return a POSIX exit code."""
    args = _parser().parse_args(argv)
    try:
        report = verify_publication_bundle_preflight(
            args.bundle_dir,
            require_release_reconciliation=not args.no_require_release_reconciliation,
        )
    except PublicationPreflightError as exc:
        report = {
            "schema_version": "publication-preflight.v1",
            "status": "fail",
            "violations": [str(exc)],
            "warnings": [],
        }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
