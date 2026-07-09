#!/usr/bin/env python3
"""Preflight a scenario perturbation manifest with fail-closed validity checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.scenario_certification import (
    preflight_perturbation_manifest,
    preflight_to_dict,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the scenario perturbation preflight parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Scenario perturbation manifest YAML.")
    parser.add_argument("--output", type=Path, help="Write the JSON preflight report to this path.")
    parser.add_argument(
        "--fail-on-excluded",
        action="store_true",
        help="Exit non-zero when any variant is excluded from success evidence.",
    )
    return parser


def main() -> int:
    """Run scenario perturbation preflight."""
    args = _build_parser().parse_args()
    try:
        report = preflight_perturbation_manifest(args.manifest)
    except (FileNotFoundError, ValueError) as exc:
        # Bad input (nonexistent manifest path or malformed/non-mapping
        # manifest): print one actionable line to stderr and exit non-zero
        # instead of dumping a raw traceback. Mirrors validate_report.py's clean
        # failure UX. ``str(exc)`` renders OSError readably (e.g.
        # ``[Errno 2] No such file or directory: ...``) and the loader's
        # ValueError wording verbatim.
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2
    payload = preflight_to_dict(report)
    output = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    if args.fail_on_excluded and any(
        result["benchmark_evidence_status"] != "eligible_success_evidence_candidate"
        for result in payload["results"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
