#!/usr/bin/env python3
"""Assemble or validate the real issue #6412 visualization-only package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.issue_6412_real_reexport import (
    RealReexportPackageError,
    assemble_real_reexport_package,
    finalize_real_reexport_package,
    materialize_resolver_mapping,
    verify_complete_package,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    assemble = subparsers.add_parser("assemble", help="assemble the local 88/2 package")
    assemble.add_argument("--binding-receipt", type=Path, required=True)
    assemble.add_argument("--request-manifest", type=Path, required=True)
    assemble.add_argument("--expected-outcomes", type=Path, required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument(
        "--external-root",
        default="benchmark-results/robot_sf_ll7/issue5756",
        help="Durable external-data-hub root; no local absolute paths are recorded.",
    )
    assemble.add_argument("--host-alias", default="imech156-u")
    assemble.add_argument("--retrieval-key", default="issue5756")

    materialize = subparsers.add_parser(
        "materialize", help="write an absolute-path resolver receipt outside the package"
    )
    materialize.add_argument("--package-dir", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)

    finalize = subparsers.add_parser("finalize", help="finalize after resolver and figure QA")
    finalize.add_argument("--package-dir", type=Path, required=True)
    finalize.add_argument("--resolution-json", type=Path, required=True)
    finalize.add_argument("--figure-qa", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="verify a finalized local package")
    verify.add_argument("--package-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one #6412 package operation and return a process exit code."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "assemble":
            manifest = assemble_real_reexport_package(
                binding_receipt=args.binding_receipt,
                request_manifest=args.request_manifest,
                expected_outcomes=args.expected_outcomes,
                output_dir=args.output_dir,
                external_root=args.external_root,
                host_alias=args.host_alias,
                retrieval_key=args.retrieval_key,
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
        elif args.command == "materialize":
            materialized = materialize_resolver_mapping(args.package_dir, args.output)
            print(json.dumps({"n_rows": len(materialized["rows"]), "output": str(args.output)}))
        elif args.command == "finalize":
            resolution = json.loads(args.resolution_json.read_text(encoding="utf-8"))
            figure_qa = json.loads(args.figure_qa.read_text(encoding="utf-8"))
            complete = finalize_real_reexport_package(
                args.package_dir,
                resolution=resolution,
                figure_qa=figure_qa,
            )
            print(json.dumps(complete, indent=2, sort_keys=True))
        else:
            complete = verify_complete_package(args.package_dir)
            print(json.dumps(complete, indent=2, sort_keys=True))
    except (OSError, TypeError, ValueError, json.JSONDecodeError, RealReexportPackageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
