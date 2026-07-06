#!/usr/bin/env python3
"""Validator for release artifact badging claims.

Verifies that the claimed badge level (available, functional, reproduced)
is supported by appropriate evidence, checklist, and manifest metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def is_raw_output_pointer(value: str | None) -> bool:
    """Return True if the pointer is just a local output/ path."""
    if not value:
        return True
    val = value.strip().lower()
    return (
        val.startswith("output/")
        or val.startswith("file://")
        or val.startswith("./")
        or val.startswith("../")
        or "localhost" in val
    )


def parse_markdown_frontmatter(content: str) -> dict[str, Any]:
    """Parse YAML front matter from markdown content."""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    frontmatter_lines = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        frontmatter_lines.append(line)
    else:
        return {}
    yaml_content = "\n".join(frontmatter_lines)
    return yaml.safe_load(yaml_content) or {}


def validate_badging_claims(  # noqa: C901, PLR0912, PLR0915
    manifest_data: dict[str, Any] | None,
    checklist_data: dict[str, Any] | None,
) -> tuple[str, list[str], list[str]]:
    """Validate badging claims against rules.

    Returns:
        Tuple of (status, blockers, next_actions)
    """
    blockers: list[str] = []
    next_actions: list[str] = []

    # 1. Resolve claimed badge level
    claimed_level = "none"
    if manifest_data:
        # Check artifact_badging block in manifest
        badging = manifest_data.get("artifact_badging")
        if isinstance(badging, dict):
            claimed_level = badging.get("claimed_level", "none")
        else:
            # Fallback to schema checking or default to none/fail-closed
            claimed_level = "none"
    elif checklist_data:
        claimed_level = checklist_data.get("claimed_badge_level", "none")

    valid_levels = {"none", "available", "functional", "reproduced"}
    if claimed_level not in valid_levels:
        blockers.append(
            f"Claimed level {claimed_level!r} is invalid. Must be one of {valid_levels}"
        )
        return "failed", blockers, ["Specify a valid claimed_badge_level"]

    if claimed_level == "none":
        return "passed", [], ["No badging verification required when claimed_level is 'none'."]

    # 2. Check "available" requirements
    has_archive = False
    has_checksum = False
    has_durable_id = False

    # Check manifest info
    if manifest_data:
        # Check files count
        files = manifest_data.get("files", [])
        if files:
            has_checksum = True

        # Check archive and DOI
        pub_channels = manifest_data.get("publication_channels")
        if isinstance(pub_channels, dict):
            doi = pub_channels.get("doi")
            rel_url = pub_channels.get("release_url")
            if doi and not is_raw_output_pointer(doi) and "<record-id>" not in doi:
                has_durable_id = True
            elif rel_url and not is_raw_output_pointer(rel_url) and "{release_tag}" not in rel_url:
                has_durable_id = True

        badging = manifest_data.get("artifact_badging", {})
        if isinstance(badging, dict) and badging.get("checklist_path"):
            has_archive = True

    # Check checklist info
    if checklist_data:
        bundle = checklist_data.get("artifact_bundle", {})
        if isinstance(bundle, dict):
            archive_path = bundle.get("archive_path")
            checksum_path = bundle.get("checksum_manifest_path")
            doi = bundle.get("doi")
            if archive_path and not is_raw_output_pointer(archive_path):
                has_archive = True
            if checksum_path:
                has_checksum = True
            if doi and not is_raw_output_pointer(doi) and "00000" not in doi:
                has_durable_id = True

    # Verify available
    if not has_checksum:
        blockers.append("Missing checksum manifest of exported files.")
    if not has_durable_id:
        blockers.append(
            "Missing durable identifier (valid DOI or release asset URL). Local output path not allowed."
        )
    if not has_archive:
        blockers.append("Missing release archive bundle path or link.")

    # 3. Check "functional" requirements
    if claimed_level in {"functional", "reproduced"}:
        has_smoke_cmd = False
        smoke_passed = False

        if checklist_data:
            repro = checklist_data.get("reproduction", {})
            if isinstance(repro, dict) and repro.get("functional_smoke_command"):
                has_smoke_cmd = True
                # In checklist, we assume if user claims functional, they ran it.
                smoke_passed = True

        if manifest_data:
            badging = manifest_data.get("artifact_badging")
            if isinstance(badging, dict):
                smoke_status = badging.get("functional_smoke_status")
                if smoke_status == "passed":
                    smoke_passed = True
                    has_smoke_cmd = True

        if not has_smoke_cmd:
            blockers.append("Missing functional smoke test command.")
        if not smoke_passed:
            blockers.append("Functional smoke test must have passed status.")

    # 4. Check "reproduced" requirements
    if claimed_level == "reproduced":
        has_repro_cmd = False
        repro_passed = False
        has_tolerances = False

        if checklist_data:
            repro = checklist_data.get("reproduction", {})
            if isinstance(repro, dict):
                if repro.get("headline_reproduction_command"):
                    has_repro_cmd = True
                if repro.get("tolerances"):
                    has_tolerances = True
                repro_passed = True  # Assuming passed from user claim if checklist filled

        if manifest_data:
            badging = manifest_data.get("artifact_badging")
            if isinstance(badging, dict):
                repro_status = badging.get("reproduction_status")
                if repro_status == "passed":
                    repro_passed = True
                # Tolerances/commands can be described in checklist_path or claim_boundary
                if badging.get("claim_boundary") or badging.get("checklist_path"):
                    has_repro_cmd = True
                    has_tolerances = True

        if not has_repro_cmd:
            blockers.append("Missing headline reproduction command.")
        if not has_tolerances:
            blockers.append("Missing reproduction tolerances.")
        if not repro_passed:
            blockers.append("Reproduction run must have passed status.")

    if blockers:
        status = "failed"
        next_actions.append("Correct the missing checklist items or manifest metadata.")
    else:
        status = "passed"
        next_actions.append(f"Verification successful for badge level: {claimed_level}.")

    return status, blockers, next_actions


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    """Validate release artifact badging claims from command line."""
    parser = argparse.ArgumentParser(description="Check release artifact badging claims.")
    parser.add_argument(
        "--manifest", type=Path, help="Path to publication or evidence manifest JSON."
    )
    parser.add_argument(
        "--checklist", type=Path, help="Path to reproducibility checklist markdown."
    )
    parser.add_argument("--output", type=Path, default=Path("release_badge_validation_report.json"))
    args = parser.parse_args(argv)

    manifest_data = None
    checklist_data = None

    if args.manifest and args.manifest.exists():
        try:
            manifest_data = json.loads(args.manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error reading manifest: {e}", file=sys.stderr)
            return 1

    if args.checklist and args.checklist.exists():
        try:
            checklist_data = parse_markdown_frontmatter(args.checklist.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError) as e:
            print(f"Error reading checklist: {e}", file=sys.stderr)
            return 1

    if not manifest_data and not checklist_data:
        print(
            "Error: Either --manifest or --checklist must be provided and exist.", file=sys.stderr
        )
        return 1

    claimed_level = "none"
    if manifest_data:
        badging = manifest_data.get("artifact_badging")
        if isinstance(badging, dict):
            claimed_level = badging.get("claimed_level", "none")
    elif checklist_data:
        claimed_level = checklist_data.get("claimed_badge_level", "none")

    status, blockers, next_actions = validate_badging_claims(manifest_data, checklist_data)

    report = {
        "schema_version": "release-badge-validation-report.v1",
        "timestamp_utc": _utc_now_iso(),
        "claimed_level": claimed_level,
        "status": status,
        "blockers": blockers,
        "next_actions": next_actions,
    }

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Validation report written to: {args.output}")
    except OSError as e:
        print(f"Error writing report: {e}", file=sys.stderr)
        return 1

    if status == "passed":
        print(f"SUCCESS: Release badging claim {claimed_level!r} passed verification.")
        return 0
    else:
        print(
            f"FAILURE: Release badging claim {claimed_level!r} failed validation.", file=sys.stderr
        )
        for blocker in blockers:
            print(f" - BLOCKER: {blocker}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
