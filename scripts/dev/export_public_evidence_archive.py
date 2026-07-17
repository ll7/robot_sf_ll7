#!/usr/bin/env python3
"""Export a public-safe adversarial archive from a private source (issue #5911).

Certified adversarial archives produced on private runners embed
infrastructure-specific absolute paths (home directories, worktree paths,
campaign ``output/`` roots) inside otherwise public evidence JSON. Registering
those archives verbatim leaks internal layout and makes the evidence
non-reproducible across machines. This helper is the reusable public-evidence
export/sanitization entry point promoted out of the one-off projection used to
register job 13518 (PR #5905, issue #5305).

What it does
------------

1. Reads a source archive JSON (``--source``) containing private absolute paths.
2. Projects every absolute private path prefix to a stable
   ``<scheme>://job-<id>/`` artifact URI while preserving candidate, metric,
   certification, family, seed, and archive-ID values exactly.
3. Records source and projected archive SHA-256 digests plus a path-only
   transformation record.
4. Fails closed if the projected archive still contains any absolute
   user/home/worktree path.
5. Writes the projected archive and transformation record through the shared
   evidence-writer convention (``robot_sf.evidence.writers``) so the required
   ``AI-GENERATED NEEDS-REVIEW`` marker is injected, and emits a SHA256SUMS
   manifest for the bundle.

Usage
-----

::

    # Explicit private root (deterministic; recommended):
    scripts/dev/export_public_evidence_archive.py \\
        --source /private/runs/job-13518/archive.json \\
        --output-dir docs/context/evidence/issue_5305_certified_archive \\
        --private-root /private/runs/job-13518 \\
        --scheme private-artifact --job-id 13518

    # Auto-detect the private root from the offending strings:
    scripts/dev/export_public_evidence_archive.py \\
        --source archive.json --output-dir out --auto-detect-private-root \\
        --job-id 13518

Exit codes
----------

* ``0``: the public-safe bundle was written and passes the fail-closed guard.
* ``1``: the source could not be read, parsed, or still leaks a private path
  after projection (the projected files are not written on failure).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.adversarial.public_projection import (
    DEFAULT_PUBLIC_SCHEME,
    PrivatePathLeakError,
    PublicProjectionConfig,
    find_offending_paths,
    project_archive_to_public,
)
from robot_sf.evidence.writers import write_json, write_review_sidecar, write_sha256sums


def _load_archive(path: Path) -> dict[str, Any]:
    """Load and validate a JSON object archive payload from ``path``."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read source archive {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"source archive {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(
            f"source archive {path} must be a JSON object, got {type(payload).__name__}"
        )
    return payload


def _write_bundle(
    *,
    output_dir: Path,
    result,
) -> list[Path]:
    """Write the projected archive and transformation record through shared writers.

    The projected ``archive.json`` is written verbatim so its committed bytes
    match the recorded ``projected_archive_sha256`` digest exactly; the required
    ``AI-GENERATED NEEDS-REVIEW`` marker is carried by a companion
    ``archive.json.review.json`` sidecar via the shared evidence-writer
    convention (issue #5911). The transformation record is metadata, so it is
    written with an inline marker through ``write_json``.

    Returns the list of files written so the caller can emit SHA256SUMS over
    them.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "archive.json"
    record_path = output_dir / "public_projection.json"

    # Verbatim bytes: do NOT inject a marker into the archive, or its digest
    # would diverge from the recorded projected_archive_sha256.
    archive_path.write_text(
        json.dumps(result.projected_archive, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar_path = write_review_sidecar(archive_path)
    write_json(record_path, result.projection)

    return [archive_path, sidecar_path, record_path]


def main(argv: list[str] | None = None) -> int:
    """Run the public-evidence archive export CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="path to the private source archive JSON to project",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory to write the public-safe archive.json and public_projection.json",
    )
    parser.add_argument(
        "--private-root",
        default="",
        help=(
            "absolute private filesystem prefix to strip and replace with the "
            "public URI (deterministic; recommended). When omitted, enable "
            "--auto-detect-private-root to derive it from offending strings."
        ),
    )
    parser.add_argument(
        "--auto-detect-private-root",
        action="store_true",
        help="derive the private root as the longest common absolute-path prefix",
    )
    parser.add_argument(
        "--scheme",
        default=DEFAULT_PUBLIC_SCHEME,
        help=f"public artifact URI scheme (default: {DEFAULT_PUBLIC_SCHEME})",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="optional job/campaign id embedded in the URI as <scheme>://job-<id>/",
    )
    parser.add_argument(
        "--no-sha256sums",
        action="store_true",
        help="do not write a SHA256SUMS manifest for the bundle",
    )
    args = parser.parse_args(argv)

    if not args.private_root and not args.auto_detect_private_root:
        parser.error(
            "provide --private-root for deterministic projection, or "
            "--auto-detect-private-root to derive it from the archive"
        )

    source_path: Path = args.source
    output_dir: Path = args.output_dir
    archive = _load_archive(source_path)

    offending_before = find_offending_paths(archive)
    if not offending_before:
        print(
            f"export_public_evidence_archive: source {source_path} contains no "
            "private absolute paths; nothing to project."
        )

    config = PublicProjectionConfig(
        private_root=args.private_root,
        scheme=args.scheme,
        job_id=args.job_id,
        auto_detect_private_root=args.auto_detect_private_root,
    )
    try:
        result = project_archive_to_public(archive, config=config)
    except PrivatePathLeakError as exc:
        sys.stderr.write(
            f"export_public_evidence_archive: FAIL-CLOSED. {exc}\n"
            "The projected archive still contains private absolute paths; "
            "no files were written. Widen --private-root or inspect the "
            "offending fields.\n"
        )
        return 1

    written = _write_bundle(output_dir=output_dir, result=result)
    if not args.no_sha256sums:
        write_sha256sums(output_dir)

    print(
        "export_public_evidence_archive: public-safe bundle written.\n"
        f"  source:                 {source_path}\n"
        f"  source_sha256:          {result.source_sha256}\n"
        f"  projected_sha256:       {result.projected_sha256}\n"
        f"  scheme:                 {result.projection['private_pointer_scheme']}\n"
        f"  offending paths found:  {len(offending_before)}\n"
        f"  replacements applied:   {result.projection['replacement_count']}\n"
        f"  values changed:         {result.projection['candidate_or_metric_values_changed']}\n"
        f"  changed fields:         {result.projection['changed_fields']}\n"
        f"  output dir:             {output_dir}\n"
        f"  files written:          {', '.join(p.name for p in written)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
