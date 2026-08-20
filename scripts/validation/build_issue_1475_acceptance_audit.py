#!/usr/bin/env python3
"""Build the issue #1475 acceptance-criteria audit from tracked evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.evidence.writers import write_json
from scripts.validation.acceptance_audit_catalog import (
    DEFAULT_1475_CLOSURE_AUDIT,
    DEFAULT_1475_STATE_SURFACE,
    DEFAULT_SMOKE_SUMMARY,
    DEFAULT_SOURCE_CHECKSUMS,
    build_issue_1475_audit,
    check_issue_contract,
)
from scripts.validation.acceptance_audit_runner import (
    ContractValidationError,
    CriterionAudit,  # noqa: F401
)

SCHEMA_VERSION = "issue-1475-acceptance-audit.v1"
DEFAULT_CLOSURE_AUDIT = DEFAULT_1475_CLOSURE_AUDIT
DEFAULT_OUTPUT = Path("docs/context/evidence/issue_1475_acceptance_audit_2026-07-06.json")
DEFAULT_STATE_SURFACE = DEFAULT_1475_STATE_SURFACE


def _repo_root_from(path: Path) -> Path:
    return path.resolve()


def build_audit(
    *,
    repo_root: Path,
    smoke_summary_path: Path = DEFAULT_SMOKE_SUMMARY,
    source_checksums_path: Path = DEFAULT_SOURCE_CHECKSUMS,
    closure_audit_path: Path = DEFAULT_CLOSURE_AUDIT,
    state_surface_path: Path = DEFAULT_STATE_SURFACE,
) -> dict[str, Any]:
    """Build a fail-closed issue #1475 acceptance audit."""

    return build_issue_1475_audit(
        repo_root=repo_root,
        smoke_summary_path=smoke_summary_path,
        source_checksums_path=source_checksums_path,
        closure_audit_path=closure_audit_path,
        state_surface_path=state_surface_path,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Build the CPU-only issue #1475 criterion-to-evidence acceptance audit."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--smoke-summary", type=Path, default=DEFAULT_SMOKE_SUMMARY)
    parser.add_argument("--source-checksums", type=Path, default=DEFAULT_SOURCE_CHECKSUMS)
    parser.add_argument("--closure-audit", type=Path, default=DEFAULT_CLOSURE_AUDIT)
    parser.add_argument("--state-surface", type=Path, default=DEFAULT_STATE_SURFACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true", help="Write the JSON artifact to --output.")
    parser.add_argument(
        "--check-contract",
        action="store_true",
        help="Validate the declarative contract without reading evidence or writing output.",
    )
    return parser


def main() -> int:
    """Build and print or write the audit artifact."""

    args = build_parser().parse_args()
    if args.check_contract:
        try:
            check_issue_contract(1475)
        except ContractValidationError as exc:
            print(f"issue #1475 acceptance-audit contract invalid: {exc}", file=sys.stderr)
            return 2
        print("issue #1475 acceptance-audit contract valid")
        return 0

    repo_root = _repo_root_from(args.repo_root)
    report = build_audit(
        repo_root=repo_root,
        smoke_summary_path=args.smoke_summary,
        source_checksums_path=args.source_checksums,
        closure_audit_path=args.closure_audit,
        state_surface_path=args.state_surface,
    )

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.write:
        output_path = repo_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_path, report)
    else:
        sys.stdout.write(payload)
    return 0 if report["status"] in {"complete", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
