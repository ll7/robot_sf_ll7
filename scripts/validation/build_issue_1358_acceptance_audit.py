#!/usr/bin/env python3
"""Build issue #1358 parent acceptance-criteria audit evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.evidence.writers import write_json
from scripts.validation.acceptance_audit_catalog import (
    DEFAULT_1358_CLOSURE_AUDIT,
    DEFAULT_1358_STATE_SURFACE,
    build_issue_1358_audit,
    check_issue_contract,
)
from scripts.validation.acceptance_audit_runner import (
    ContractValidationError,
    CriterionAudit,  # noqa: F401
)

SCHEMA_VERSION = "issue-1358-acceptance-audit.v1"
DEFAULT_CLOSURE_AUDIT = DEFAULT_1358_CLOSURE_AUDIT
DEFAULT_STATE_SURFACE = DEFAULT_1358_STATE_SURFACE
DEFAULT_OUTPUT = Path("docs/context/evidence/issue_1358_acceptance_audit_2026-07-07.json")


def _repo_root_from(path: Path) -> Path:
    return path.resolve()


def build_audit(
    *,
    repo_root: Path,
    closure_audit_path: Path = DEFAULT_CLOSURE_AUDIT,
    state_surface_path: Path = DEFAULT_STATE_SURFACE,
) -> dict[str, Any]:
    """Build a fail-closed issue #1358 acceptance audit."""

    return build_issue_1358_audit(
        repo_root=repo_root,
        closure_audit_path=closure_audit_path,
        state_surface_path=state_surface_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root resolving relative paths (default: current directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON artifact path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the audit JSON artifact instead of printing to stdout.",
    )
    parser.add_argument(
        "--check-contract",
        action="store_true",
        help="Validate the declarative contract without reading evidence or writing output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run CLI."""

    args = build_arg_parser().parse_args(argv)
    if args.check_contract:
        try:
            check_issue_contract(1358)
        except ContractValidationError as exc:
            print(f"issue #1358 acceptance-audit contract invalid: {exc}", file=sys.stderr)
            return 2
        print("issue #1358 acceptance-audit contract valid")
        return 0

    repo_root = _repo_root_from(args.repo_root)
    report = build_audit(repo_root=repo_root)

    if report["state_surface"]["status"] != "valid":
        print(
            f"issue #1358 state surface mismatch: {report['state_surface']['errors']}",
            file=sys.stderr,
        )
        return 2

    if args.write:
        output = args.output if args.output.is_absolute() else repo_root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    else:
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
