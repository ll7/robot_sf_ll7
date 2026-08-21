"""Maintainer CLI for benchmark-release diagnostics and direct Zenodo publication."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark import zenodo_publisher
from robot_sf.benchmark.release_doctor import collect_release_doctor_report

if TYPE_CHECKING:
    import argparse


def build_subparser(subparsers: Any) -> None:
    """Register the ``robot-sf release`` command tree."""
    release = subparsers.add_parser("release", help="Benchmark-data release operations.")
    modes = release.add_subparsers(dest="release_cmd", required=True)
    zenodo = modes.add_parser("zenodo", help="Direct Zenodo benchmark-dataset publisher.")
    zenodo_modes = zenodo.add_subparsers(dest="zenodo_mode", required=True)
    for mode in ("reserve", "upload", "publish", "verify"):
        parser = zenodo_modes.add_parser(mode)
        parser.add_argument("--token-file", type=Path, required=True)
        parser.add_argument("--state", type=Path, required=True)
        parser.add_argument("--api-base", default=zenodo_publisher.ZENODO_API_BASE)
        if mode in {"reserve", "verify"}:
            parser.add_argument("--metadata", type=Path, required=True)
        if mode == "upload":
            parser.add_argument("files", nargs="+", type=Path)
    doctor = modes.add_parser("doctor", help="Fail-closed full benchmark-release diagnostics.")
    doctor.add_argument("--repo", type=Path, default=Path.cwd())
    doctor.add_argument("--manifest", type=Path, required=True)
    doctor.add_argument("--expected-release-sha", required=True)
    doctor.add_argument("--expected-base-sha", required=True)
    doctor.add_argument("--tag", required=True)
    doctor.add_argument("--checkpoint-receipt", type=Path)
    doctor.add_argument("--private-launch-packet", type=Path)
    doctor.add_argument("--dissertation", type=Path)
    doctor.add_argument("--token-file", type=Path)
    doctor.add_argument("--expected-cells", type=int, default=20160)
    doctor.add_argument("--minimum-free-gib", type=float, default=100.0)
    doctor.add_argument("--require-zenodo-webhook-disabled", action="store_true")


def _print(payload: dict[str, Any]) -> None:
    """Print a stable JSON object without credentials."""
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def handle(args: argparse.Namespace) -> int:
    """Dispatch release operations and return a process exit code.

    Returns:
        Zero for success and two for a blocked or failed publication operation.
    """
    if args.release_cmd == "doctor":
        report = collect_release_doctor_report(
            repo=args.repo.resolve(),
            manifest_path=args.manifest.resolve(),
            expected_release_sha=args.expected_release_sha,
            expected_base_sha=args.expected_base_sha,
            tag=args.tag,
            checkpoint_receipt=args.checkpoint_receipt,
            private_launch_packet=args.private_launch_packet,
            dissertation=args.dissertation,
            token_file=args.token_file,
            expected_cells=args.expected_cells,
            minimum_free_gib=args.minimum_free_gib,
            require_zenodo_webhook_disabled=args.require_zenodo_webhook_disabled,
        )
        _print(report)
        return 0 if report["status"] == "pass" else 2
    try:
        session = zenodo_publisher.build_session(args.token_file)
        if args.zenodo_mode == "reserve":
            metadata = zenodo_publisher.load_dataset_metadata(args.metadata)
            state = zenodo_publisher.reserve(session, metadata, api_base=args.api_base)
            zenodo_publisher.write_state(args.state, state)
            _print(state)
            return 0
        state = zenodo_publisher.load_state(args.state)
        if args.zenodo_mode == "upload":
            state = zenodo_publisher.upload(session, state, args.files, api_base=args.api_base)
            zenodo_publisher.write_state(args.state, state)
            _print(state)
            return 0
        if args.zenodo_mode == "publish":
            state = zenodo_publisher.publish(session, state, api_base=args.api_base)
            zenodo_publisher.write_state(args.state, state)
            _print(state)
            return 0
        metadata = zenodo_publisher.load_dataset_metadata(args.metadata)
        report = zenodo_publisher.verify(session, state, metadata, api_base=args.api_base)
        _print(report)
        return 0 if report["status"] == "pass" else 2
    except zenodo_publisher.ZenodoPublisherError as exc:
        _print({"status": "blocked", "reason": str(exc)})
        return 2


__all__ = ["build_subparser", "handle"]
