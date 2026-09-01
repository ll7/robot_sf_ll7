#!/usr/bin/env python3
"""Generate or verify a non-self-referential benchmark release identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from robot_sf.benchmark.release_protocol import (
    RESOLVED_RELEASE_METADATA_FILENAME,
    verify_resolved_release_identity,
    write_resolved_release_identity,
)
from robot_sf.common.artifact_paths import get_repository_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve a tracked benchmark v0.2 template into deterministic ignored JSON "
            "artifacts, or reproduce and verify those bytes at the exact clean source commit."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="Generate resolved identity JSON.")
    generate.add_argument("--template", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--source-commit", required=True)
    generate.add_argument("--release-tag", required=True)
    generate.add_argument("--concept-doi", required=True)
    generate.add_argument("--version-doi", required=True)
    generate.add_argument("--repository-root", type=Path, default=None)

    verify = subparsers.add_parser("verify", help="Reproduce and verify resolved identity bytes.")
    verify.add_argument("--identity", type=Path, required=True)
    verify.add_argument("--repository-root", type=Path, default=None)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    """Run the deterministic generate/verify command."""
    args = _parser().parse_args(argv)
    repository_root = (args.repository_root or get_repository_root()).resolve()
    try:
        if args.command == "generate":
            output = args.output if args.output.is_absolute() else repository_root / args.output
            payload = write_resolved_release_identity(
                template_path=args.template,
                output_path=output,
                source_commit=args.source_commit,
                release_tag=args.release_tag,
                concept_doi=args.concept_doi,
                version_doi=args.version_doi,
                repository_root=repository_root,
            )
            metadata = output.parent / RESOLVED_RELEASE_METADATA_FILENAME
            _print(
                {
                    "schema_version": "benchmark-release-identity-command.v1",
                    "status": "generated",
                    "source_commit": payload["source_commit"],
                    "latest_main_base_commit": payload["resolved_manifest"][
                        "latest_main_base_commit"
                    ],
                    "release_tag": payload["release_tag"],
                    "identity_path": output.resolve().relative_to(repository_root).as_posix(),
                    "identity_sha256": _sha256(output),
                    "metadata_path": metadata.resolve().relative_to(repository_root).as_posix(),
                    "metadata_sha256": _sha256(metadata),
                }
            )
            return 0

        identity = args.identity if args.identity.is_absolute() else repository_root / args.identity
        manifest = verify_resolved_release_identity(identity, repository_root=repository_root)
        _print(
            {
                "schema_version": "benchmark-release-identity-command.v1",
                "status": "verified",
                "source_commit": manifest.source_sha,
                "latest_main_base_commit": manifest.latest_main_base_commit,
                "release_tag": manifest.release_tag,
                "identity_path": identity.resolve().relative_to(repository_root).as_posix(),
                "identity_sha256": _sha256(identity),
                "metadata_sha256": manifest.metadata_sha256,
            }
        )
        return 0
    except (OSError, TypeError, ValueError, KeyError) as exc:
        _print(
            {
                "schema_version": "benchmark-release-identity-command.v1",
                "status": "rejected",
                "reason": str(exc),
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
