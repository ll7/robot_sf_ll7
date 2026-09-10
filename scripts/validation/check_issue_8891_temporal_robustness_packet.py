#!/usr/bin/env python3
"""Check or materialize the outcome-free issue #8891 packet in memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from robot_sf.adversarial.matched_budget_packet import (
    build_canary_packet,
    build_expected_identities,
    validate_packet,
)

DEFAULT_PACKET = Path("configs/adversarial/issue_8891_temporal_robustness_packet.yaml")


def main() -> int:
    """Run a read-only packet operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--identities", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--format", choices=("json", "text"), default="text")
    args = parser.parse_args()
    if args.check_only and not args.canary:
        parser.error("--check-only requires --canary")
    root = args.repo_root.resolve()
    path = args.packet if args.packet.is_absolute() else root / args.packet
    packet = yaml.safe_load(path.read_text(encoding="utf-8"))
    summary = validate_packet(packet, repo_root=root)
    if args.identities:
        summary = build_expected_identities(packet, repo_root=root)
    elif args.canary:
        summary = build_canary_packet(packet, repo_root=root)
    if args.format == "text":
        summary = {
            key: value
            for key, value in summary.items()
            if key not in {"runs", "rows", "candidates"}
        }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
