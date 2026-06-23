#!/usr/bin/env python3
"""Thin compatibility wrapper for SDD staging (Issue #2657, consolidated under Issue #3473).

The SDD staging subsystem (manifest parsing, pinned-checksum validation, the proxy-vs-dataset-backed
scenario-prior gate, the disk-space check, and the no-auto-download safety contract) now lives in
the canonical external-data subsystem:

    scripts/tools/manage_external_data.py

That module owns the ``sdd`` asset specification and is the single source of truth for the SDD
checksum policy and availability states. This file is preserved only as a thin shim so existing
callers, tests, and the documented CLI surface keep working. New code should import the canonical
``manage_external_data`` functions (``load_sdd_staging_spec``, ``validate_sdd_staging``,
``resolve_sdd_scenario_prior_mode``, ``run_sdd_download`` ...) directly.

SAFETY CONTRACT (unchanged; enforced in the canonical module):

* **Never auto-download.** A default invocation only PLANS/REPORTS and exits without touching the
  network.
* A network fetch requires an explicit ``--confirm-download`` flag AND an interactive ``y/N`` prompt
  (skipped only when ``--yes`` is also given for non-interactive use).
* Free disk space is checked against the manifest's expected size BEFORE any fetch; the tool fails
  closed when free space is insufficient.
* A staged tree is validated against the pinned ``expected_tree_sha256`` before being marked
  ``staged``; a mismatch refuses to mark SDD as dataset-backed.
* Raw data is staged into a **git-ignored** subfolder (``output/external_data/sdd`` by default), or
  into ``$ROBOT_SF_EXTERNAL_DATA_ROOT/sdd`` when that shared root is configured.

Usage::

    # Plan only -- NEVER downloads (default):
    uv run python scripts/data/stage_sdd_dataset_issue_2657.py
    # Availability status (no download):
    uv run python scripts/data/stage_sdd_dataset_issue_2657.py --check
    # Validate a locally-staged copy (no download):
    uv run python scripts/data/stage_sdd_dataset_issue_2657.py validate
    # Real staging requires explicit confirmation AND a configured download URL:
    uv run python scripts/data/stage_sdd_dataset_issue_2657.py download --confirm-download

The canonical CLI equivalents are ``manage_external_data.py sdd-plan|sdd-status|sdd-validate|
sdd-mode|sdd-download``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.tools import manage_external_data as _canonical
from scripts.tools.manage_external_data import (
    DEFAULT_SDD_STAGING_MANIFEST as DEFAULT_MANIFEST_PATH,
)
from scripts.tools.manage_external_data import (
    ExternalDataError,
)
from scripts.tools.manage_external_data import (
    SddStagingSpec as SddManifest,
)

# Backwards-compatible alias: the #2657 module raised SddStagingError. The canonical subsystem uses
# ExternalDataError; alias it so callers/tests catching SddStagingError keep working.
SddStagingError = ExternalDataError

REPO_ROOT = _canonical.REPO_ROOT

# Re-export canonical names under their historical #2657 names so existing imports/tests and the
# documented CLI surface keep working after consolidation. These are module-level assignments (not
# `import as`) so the linter does not strip them as unused re-exports.
MODE_PROXY = _canonical.SDD_MODE_PROXY
MODE_DATASET_BACKED = _canonical.SDD_MODE_DATASET_BACKED
STAGING_STATUS_FILENAME = _canonical.SDD_STAGING_STATUS_FILENAME
DEFAULT_STAGING_ROOT = _canonical.DEFAULT_STAGING_ROOT
ExpectedFile = _canonical.SddExpectedFile

# Re-export canonical primitives under their historical #2657 names.
_tree_checksum = _canonical._tree_checksum
load_manifest = _canonical.load_sdd_staging_spec
validate_staging = _canonical.validate_sdd_staging
resolve_scenario_prior_mode = _canonical.resolve_sdd_scenario_prior_mode
check_disk_space = _canonical.check_sdd_disk_space
build_plan = _canonical.build_sdd_plan
_write_staging_status = _canonical.write_sdd_staging_status


def run_download(manifest: SddManifest, args: argparse.Namespace) -> dict[str, Any]:
    """Compatibility wrapper preserving the #2657 ``run_download(manifest, args)`` signature."""
    return _canonical.run_sdd_download(
        manifest,
        confirm_download=getattr(args, "confirm_download", False),
        yes=getattr(args, "yes", False),
    )


def _confirm_download(args: argparse.Namespace) -> bool:
    """Compatibility wrapper around the canonical confirmation gate."""
    return _canonical.confirm_sdd_download(
        confirm_download=getattr(args, "confirm_download", False),
        yes=getattr(args, "yes", False),
    )


def _emit(payload: dict[str, Any], *, as_json: bool, human_lines: list[str]) -> None:
    """Print JSON or human-readable output."""
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("\n".join(human_lines))


def _plan_lines(plan: dict[str, Any]) -> list[str]:
    """Human-readable lines for a plan/report payload."""
    disk = plan["disk_check"]
    return [
        f"SDD staging PLAN (asset_id={plan['asset_id']}, version={plan['version_tag']})",
        f"  source:        {plan['source_url']}",
        f"  license:       {plan['license']} ({plan['license_url']})",
        f"  readme:        {plan['readme_pointer']}",
        f"  staging_dir:   {plan['staging_dir']} (git-ignored)",
        f"  expected size: {plan['expected_total_size_mib']} MiB",
        f"  disk check:    required {disk['required_mib']} MiB, "
        f"available {disk['available_mib']} MiB -> "
        f"{'OK' if disk['sufficient'] else 'INSUFFICIENT'}",
        f"  availability:  {plan['local_availability']}",
        f"  prior mode:    {plan['scenario_prior_mode']}",
        "  download:      NO (plan only; never auto-downloads)",
        f"  next step:     {plan['next_step']}",
    ]


def _validation_lines(report: dict[str, Any]) -> list[str]:
    """Human-readable lines for a validation/check payload."""
    lines = [
        f"SDD staging status (asset_id={report['asset_id']}, version={report['version_tag']})",
        f"  staging_dir:   {report['staging_dir']} (exists={report['staging_dir_exists']})",
        f"  availability:  {report['local_availability']}",
        f"  prior mode:    {report['mode']}",
    ]
    if report.get("tree_sha256"):
        lines.append(f"  tree_sha256:   {report['tree_sha256']}")
    if report["missing_expected"]:
        lines.append(f"  missing:       {', '.join(report['missing_expected'])}")
    lines.append(f"  action:        {report['action']}")
    return lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments (preserved #2657 surface)."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Staging manifest path (default: {DEFAULT_MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report local availability status and checksums WITHOUT downloading.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("plan", help="Plan/report only (default); never downloads.")
    sub.add_parser("validate", help="Validate the locally-staged SDD copy; never downloads.")
    sub.add_parser("status", help="Alias for --check: availability status; never downloads.")
    sub.add_parser("mode", help="Print the scenario-prior mode gate (proxy vs dataset-backed).")
    download_parser = sub.add_parser(
        "download", help="Guarded staging: requires explicit confirmation; fails closed."
    )
    download_parser.add_argument(
        "--confirm-download",
        action="store_true",
        help="Explicitly confirm a network download (still prompts unless --yes).",
    )
    download_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive y/N prompt for non-interactive confirmation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the SDD staging tool via the canonical subsystem (preserved #2657 CLI)."""
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        command = args.command
        if args.check and command in (None, "plan"):
            command = "status"

        if command in (None, "plan"):
            plan = build_plan(manifest)
            _emit(plan, as_json=args.json, human_lines=_plan_lines(plan))
            return 0

        if command in ("status", "validate"):
            report = validate_staging(manifest)
            if command == "validate" and report["ok"]:
                _write_staging_status(manifest, report)
            _emit(report, as_json=args.json, human_lines=_validation_lines(report))
            return 0 if report["ok"] else 2

        if command == "mode":
            gate = resolve_scenario_prior_mode(manifest)
            _emit(
                gate,
                as_json=args.json,
                human_lines=[
                    f"scenario_prior_mode: {gate['mode']}",
                    f"dataset_backed: {gate['dataset_backed']}",
                    f"reason: {gate['reason']}",
                ],
            )
            return 0 if gate["dataset_backed"] else 2

        if command == "download":
            report = run_download(manifest, args)
            _emit(report, as_json=args.json, human_lines=_validation_lines(report))
            return 0 if report.get("ok") else 2

        raise SddStagingError(f"Unknown command: {command}")
    except SddStagingError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
