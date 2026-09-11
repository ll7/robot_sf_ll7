#!/usr/bin/env python3
"""Capture or check a deterministic, redacted environment manifest (issue #8894).

The command records the reproducibility-relevant state of one explicitly selected platform class
(local CPU, local GPU, SLURM CPU, SLURM GPU, CARLA) without installing, mutating, or running any
campaign. Environment variables are only read from explicit allowlists, and home directories,
hostnames, private cluster roots, credential-like values, signed-URL parameters, and scheduler
account/partition identities are redacted before output. Repeated captures with identical probes
share one ``semantic_digest``; ``captured_at_utc`` stays outside that digest.

Usage:

    uv run python scripts/tools/capture_environment_manifest.py capture \
        --platform-class local_cpu --output /tmp/environment.json
    uv run python scripts/tools/capture_environment_manifest.py check \
        --manifest /tmp/environment.json --platform-class local_cpu

Exit codes:

- ``capture``: ``0`` success, ``2`` unknown platform class or unreadable input.
- ``check``: ``0`` compatible, ``1`` incompatible, ``2`` malformed manifest or unknown class.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.evidence.environment_manifest import (
    ENVIRONMENT_MANIFEST_SCHEMA_VERSION,
    PLATFORM_CONTRACTS,
    REASON_UNKNOWN_PLATFORM_CLASS,
    build_environment_manifest,
    evaluate_environment_manifest,
)


def _parse_declared(pairs: list[str] | None) -> dict[str, str]:
    declared_values: dict[str, str] = {}
    for pair in pairs or []:
        key, separator, value = pair.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"declared_fields_invalid: --declare expects KEY=VALUE, got {pair!r}")
        declared_values[key.strip()] = value
    return declared_values


def _summarize(manifest: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict) and "status" in value:
            reason = f" ({value['reason']})" if value.get("reason") else ""
            lines.append(f"{prefix}: {value['status']}{reason}")
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), item)

    for section in (
        "repository",
        "runtime",
        "packages",
        "accelerator",
        "scheduler",
        "thread_controls",
    ):
        walk(section, manifest.get(section, {}))
    return lines


def _capture(args: argparse.Namespace) -> int:
    try:
        manifest = build_environment_manifest(
            args.platform_class,
            repo_root=args.repo_root,
            declared_values=_parse_declared(args.declare),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output = Path(args.output)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"error: manifest_unwritable: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    print(f"{ENVIRONMENT_MANIFEST_SCHEMA_VERSION} {manifest['platform_class']}")
    for line in _summarize(manifest):
        print(f"  {line}")
    print(f"semantic_digest: {manifest['semantic_digest']}")
    print(f"wrote: {output}")
    return 0


def _check(args: argparse.Namespace) -> int:
    try:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: manifest_unreadable: {exc}", file=sys.stderr)
        return 2
    if not isinstance(manifest, dict):
        print("error: manifest_unreadable: expected a JSON object", file=sys.stderr)
        return 2
    result = evaluate_environment_manifest(manifest, expected_platform_class=args.platform_class)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"compatible: {str(result['compatible']).lower()}")
        print(
            "platform_class: "
            f"{result['manifest_platform_class']} (expected {result['expected_platform_class']})"
        )
        reasons = ", ".join(result["reasons"]) if result["reasons"] else "none"
        print(f"reasons: {reasons}")
    if REASON_UNKNOWN_PLATFORM_CLASS in result["reasons"]:
        return 2
    return 0 if result["compatible"] else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the capture/check argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Capture a manifest for one platform class.")
    capture.add_argument("--platform-class", required=True, choices=sorted(PLATFORM_CONTRACTS))
    capture.add_argument("--output", required=True, help="Destination JSON path.")
    capture.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Checkout to fingerprint (default: current working directory).",
    )
    capture.add_argument(
        "--declare",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Record a caller-declared value outside observed probes (repeatable).",
    )
    capture.add_argument(
        "--json", action="store_true", help="Print the manifest JSON instead of a summary."
    )
    capture.set_defaults(handler=_capture)

    check = subparsers.add_parser("check", help="Check a manifest against a platform class.")
    check.add_argument("--manifest", required=True, help="Manifest JSON path.")
    check.add_argument("--platform-class", required=True, help="Expected platform class.")
    check.add_argument(
        "--json", action="store_true", help="Print the check result JSON instead of a summary."
    )
    check.set_defaults(handler=_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the capture/check command and return the process exit code."""
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
