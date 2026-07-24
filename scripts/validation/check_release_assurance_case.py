#!/usr/bin/env python3
"""Generate or check release assurance case JSON.

The default mode checks a checked-in worked example and fails closed if any
evidence leaf is missing or its recorded SHA-256 no longer matches the file on
disk.  Generation is explicit so CI does not silently refresh stale evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.release_assurance_case import (
    build_release_assurance_case_from_paths,
    load_release_assurance_case,
    validate_release_assurance_case_references,
    validate_release_assurance_case_schema,
    write_release_assurance_case,
)

DEFAULT_CASE = Path("docs/context/evidence/issue_4683_release_assurance_case_example.json")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE, help="Assurance case JSON path.")
    parser.add_argument("--manifest", type=Path, help="Release manifest used when generating.")
    parser.add_argument("--gate-spec", type=Path, help="Release gate spec used when generating.")
    parser.add_argument(
        "--gate-report", type=Path, help="Optional gate report used when generating."
    )
    parser.add_argument(
        "--generated-at-utc",
        help="Stable timestamp override for reproducible generated examples.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Generate --case from --manifest/--gate-spec instead of checking an existing case.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the release assurance case checker."""

    args = _parse_args(argv)
    if args.write:
        if args.manifest is None:
            raise SystemExit("--manifest is required with --write")
        payload = build_release_assurance_case_from_paths(
            manifest_path=args.manifest,
            gate_spec_path=args.gate_spec,
            release_gate_report_path=args.gate_report,
            generated_at_utc=args.generated_at_utc,
        )
        validate_release_assurance_case_schema(payload)
        problems = validate_release_assurance_case_references(payload)
        if problems:
            for problem in problems:
                print(
                    f"{problem.evidence_id}: {problem.path}: {problem.reason}",
                    file=sys.stderr,
                )
            return 1
        write_release_assurance_case(payload, args.case)
        print(f"wrote {args.case}")
        return 0

    payload = load_release_assurance_case(args.case)
    validate_release_assurance_case_schema(payload)
    problems = validate_release_assurance_case_references(payload)
    if problems:
        for problem in problems:
            print(f"{problem.evidence_id}: {problem.path}: {problem.reason}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "valid",
                "case": str(args.case),
                "evidence_count": len(payload["evidence"]),
                "claim_count": len(payload["claims"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
