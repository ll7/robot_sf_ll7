"""Validate a durable oracle-imitation trace-URI registry.

The registry records, per split, the durable trace URI, its SHA-256 checksum, the trace/split
identity, the schema version, and a retrieval status. Use ``--require-training-ready`` to fail
closed unless every required split is concretely, durably resolvable, which is the mechanical
gate for leaving the ``artifact_retrieval_blocked`` lane state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.training.oracle_trace_uri_registry import (
    OracleTraceUriRegistryError,
    validate_trace_uri_registry,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Validate a durable oracle-imitation trace-URI registry."
    )
    parser.add_argument("--config", required=True, type=Path, help="Trace-URI registry YAML path.")
    parser.add_argument(
        "--repo-root",
        default=Path.cwd(),
        type=Path,
        help="Repository root used to resolve relative local-mirror paths.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON validation report.")
    parser.add_argument(
        "--require-training-ready",
        action="store_true",
        help=(
            "Fail closed unless every required split has a concrete durable resolvable trace "
            "URI and checksum, i.e. the registry mechanically satisfies training_ready."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate a trace-URI registry and return a shell-friendly exit code."""
    args = build_arg_parser().parse_args(argv)
    try:
        report = validate_trace_uri_registry(
            args.config,
            repo_root=args.repo_root,
            require_training_ready=args.require_training_ready,
        )
    except OracleTraceUriRegistryError as exc:
        if args.json:
            print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(str(exc))
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "oracle-imitation trace-URI registry valid: "
            f"{report['dataset_id']} ({report['trace_count']} traces, "
            f"training_ready={report['training_ready']})"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
