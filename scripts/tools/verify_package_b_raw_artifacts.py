#!/usr/bin/env python3
"""Verify Package B raw candidate/replay artifact inventory and file digests.

Provides a deterministic verification and retrieval status check for the 4,761-entry
candidate/replay artifact tree (Issue #6131).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from robot_sf.benchmark.adversarial_package_b_report import (
    RAW_ARTIFACT_METADATA_FILENAME,
    retrieve_package_b_raw_artifacts,
    verify_package_b_candidate_replay_inventory,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_BUNDLE = Path("docs/context/evidence/issue_5785_package_b_27cell_replication_2026-07-15")


def build_parser() -> argparse.ArgumentParser:
    """Build parser for verify_package_b_raw_artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
        help="Path to Package B evidence bundle directory or candidate_replay_SHA256SUMS.txt inventory file.",
    )
    parser.add_argument(
        "--raw-tree-dir",
        type=Path,
        default=None,
        help="Existing raw candidate/replay tree to byte-verify instead of retrieving the archive.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Pinned raw-artifact metadata JSON. Defaults to raw_artifact_bundle.json in --bundle "
            "when retrieving."
        ),
    )
    parser.add_argument(
        "--retrieve-to",
        type=Path,
        default=None,
        help="Directory for the retrieved archive and extracted tree; must be empty.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the verification summary payload.",
    )
    parser.add_argument(
        "--fail-closed",
        action="store_true",
        help="Deprecated compatibility flag; verification now always exits non-zero on failure.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run candidate/replay inventory verification."""
    args = build_parser().parse_args(argv)
    if args.raw_tree_dir is not None and args.retrieve_to is not None:
        raise SystemExit("--raw-tree-dir and --retrieve-to cannot be used together")

    retrieval = None
    if args.raw_tree_dir is not None:
        result = verify_package_b_candidate_replay_inventory(args.bundle, args.raw_tree_dir)
    else:
        metadata_path = args.metadata or args.bundle / RAW_ARTIFACT_METADATA_FILENAME
        if args.retrieve_to is not None:
            retrieval = retrieve_package_b_raw_artifacts(
                args.bundle, args.retrieve_to, metadata_path=metadata_path
            )
            result = verify_package_b_candidate_replay_inventory(
                args.bundle, retrieval.raw_tree_dir if retrieval.is_valid else None
            )
        else:
            with TemporaryDirectory(prefix="robot_sf_package_b_raw_") as temporary_dir:
                retrieval = retrieve_package_b_raw_artifacts(
                    args.bundle, temporary_dir, metadata_path=metadata_path
                )
                result = verify_package_b_candidate_replay_inventory(
                    args.bundle, retrieval.raw_tree_dir if retrieval.is_valid else None
                )
        if retrieval is not None and not retrieval.is_valid:
            result = replace(
                result,
                is_valid=False,
                errors=(*retrieval.errors, *result.errors),
            )
    payload = result.to_payload()
    if retrieval is not None:
        payload["retrieval"] = retrieval.to_payload()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")

    return 0 if result.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
