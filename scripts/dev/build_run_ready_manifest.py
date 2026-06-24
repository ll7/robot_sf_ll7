#!/usr/bin/env python3
"""Build a machine-readable run-ready manifest from launch packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.dev.preflight_launch_packet import preflight_launch_packet

SCHEMA_VERSION = "run-ready-launch-packet-manifest.v1"
DEFAULT_GLOB = "configs/**/*launch_packet*.yaml"


def _manifest_entry(report: dict[str, Any]) -> dict[str, Any]:
    """Return compact manifest entry from one preflight report."""

    configs = [
        {
            "field": item.get("field"),
            "path": item.get("path"),
            "exists": item.get("exists"),
            "sha256": item.get("sha256_expected"),
            "sha256_matches": item.get("sha256_matches"),
        }
        for item in report.get("configs", [])
    ]
    return {
        "issue": report.get("issue"),
        "packet_path": report.get("packet_path"),
        "schema_version": report.get("packet_schema_version"),
        "kind": report.get("kind"),
        "ready": bool(report.get("ready")),
        "reasons": list(report.get("reasons", [])),
        "configs": configs,
        "seed_budget": report.get("seed_budget"),
        "command": report.get("command"),
        "claim_gate": report.get("claim_gate"),
    }


def build_run_ready_manifest(
    *,
    repo_root: Path,
    pattern: str = DEFAULT_GLOB,
) -> dict[str, Any]:
    """Build a run-ready manifest payload."""

    repo_root = repo_root.resolve()
    packets = sorted(repo_root.glob(pattern))
    entries = []
    for packet in packets:
        report = preflight_launch_packet(packet, repo_root=repo_root)
        entries.append(_manifest_entry(report))
    return {
        "schema_version": SCHEMA_VERSION,
        "packet_glob": pattern,
        "packet_count": len(entries),
        "ready_count": sum(1 for entry in entries if entry["ready"]),
        "entries": entries,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument("--glob", default=DEFAULT_GLOB, help="Launch-packet glob relative to repo.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/run_ready_manifest.yaml"),
        help="Output YAML path.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of YAML.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = _parse_args(argv)
    manifest = build_run_ready_manifest(repo_root=args.repo_root, pattern=args.glob)
    if args.json:
        text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    else:
        text = yaml.safe_dump(manifest, sort_keys=False)
    output = args.output if args.output.is_absolute() else args.repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
