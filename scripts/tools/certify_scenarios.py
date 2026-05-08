#!/usr/bin/env python3
"""Generate ``scenario_cert.v1`` certificates for scenario manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.scenario_certification import (
    CERT_SCHEMA_VERSION,
    certificate_to_dict,
    certify_scenario_file,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for scenario certification.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_config", type=Path, help="Scenario YAML manifest to certify.")
    parser.add_argument("--scenario-id", help="Optional scenario id/name selector.")
    parser.add_argument("--output", type=Path, help="Write JSON output to this path.")
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Emit one certificate per JSONL line instead of a batch object.",
    )
    parser.add_argument(
        "--fail-on-excluded",
        action="store_true",
        help="Exit non-zero when any certificate is benchmark-excluded.",
    )
    return parser


def main() -> int:
    """Run the scenario certification CLI."""

    args = _build_parser().parse_args()
    certificates = certify_scenario_file(args.scenario_config, scenario_id=args.scenario_id)
    payloads = [certificate_to_dict(certificate) for certificate in certificates]
    if args.jsonl:
        output = "\n".join(json.dumps(payload, sort_keys=True) for payload in payloads) + "\n"
    else:
        output = json.dumps(
            {
                "schema_version": f"{CERT_SCHEMA_VERSION}.batch",
                "scenario_config": args.scenario_config.as_posix(),
                "certificates": payloads,
            },
            indent=2,
            sort_keys=True,
        )
        output += "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    if args.fail_on_excluded and any(
        payload.get("benchmark_eligibility") == "excluded" for payload in payloads
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
