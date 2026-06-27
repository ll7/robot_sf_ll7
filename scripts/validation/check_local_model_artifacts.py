#!/usr/bin/env python3
"""Check configs for silent local-only model artifact dependencies.

The checker targets benchmark/config portability for Issue #1638. It flags
``model_path`` and ``resume_from`` values under ``output/`` unless the exact
reference is listed in the blocklist with an artifact-promotion follow-up.

The ``--audit-blocklist`` mode (Issue #1764) inverts the check: it reports
blocklist entries that no longer cover a present local reference because the
named config was retired/removed or migrated to a durable ``model_id``. Those
orphaned entries can be pruned so the preflight allowlist shrinks as configs are
recovered or retired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.local_model_artifacts import (
    BLOCKLIST_ACTIVE,
    PROMOTED_BLOCKED_STATUS,
    BlocklistAuditEntry,
    LocalModelReference,
    audit_blocklist_coverage,
    classify_local_model_references,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCAN_ROOTS = (REPO_ROOT / "configs/baselines",)
DEFAULT_BLOCKLIST = REPO_ROOT / "configs/baselines/local_model_artifact_blocklist.yaml"
DEFAULT_PROMOTED_SURFACES = REPO_ROOT / "configs/benchmarks/promoted_config_surfaces.yaml"


def check_local_model_artifacts(
    scan_paths: list[Path],
    *,
    blocklist_path: Path = DEFAULT_BLOCKLIST,
    promoted_surfaces_path: Path = DEFAULT_PROMOTED_SURFACES,
) -> list[LocalModelReference]:
    """Inspect YAML configs and classify local model references.

    Returns:
        list[LocalModelReference]: Classified references. ``status=unblocked`` rows should fail
        preflight; ``status=blocked`` rows are intentionally explicit follow-up work.
    """
    return classify_local_model_references(
        scan_paths,
        repo_root=REPO_ROOT,
        blocklist_path=blocklist_path,
        promoted_surfaces_path=promoted_surfaces_path,
    )


def audit_blocklist(
    *,
    blocklist_path: Path = DEFAULT_BLOCKLIST,
) -> list[BlocklistAuditEntry]:
    """Audit the blocklist for orphaned (retired/migrated) entries.

    Returns:
        list[BlocklistAuditEntry]: One entry per blocklist triple. Non-``active`` rows name a
        retired or migrated config whose allowlist entry can now be pruned.
    """
    return audit_blocklist_coverage(blocklist_path, repo_root=REPO_ROOT)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=list(DEFAULT_SCAN_ROOTS),
        help="YAML files or directories to scan; defaults to configs/baselines",
    )
    parser.add_argument(
        "--blocklist",
        type=Path,
        default=DEFAULT_BLOCKLIST,
        help="Explicit blocklist for known local-only artifact references.",
    )
    parser.add_argument(
        "--promoted-surfaces",
        type=Path,
        default=DEFAULT_PROMOTED_SURFACES,
        help="YAML file listing benchmark-promoted config surfaces that must never use output/ model paths.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON rows.")
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Also fail when all local references are explicitly blocked.",
    )
    parser.add_argument(
        "--audit-blocklist",
        action="store_true",
        help=(
            "Audit the blocklist for orphaned entries (configs that were retired or migrated to a "
            "durable model_id). Fails when any orphaned entry remains."
        ),
    )
    return parser.parse_args()


def _run_blocklist_audit(blocklist_path: Path, *, as_json: bool) -> int:
    """Print the blocklist coverage audit and fail when orphaned entries remain."""
    entries = audit_blocklist(blocklist_path=blocklist_path)
    if as_json:
        print(json.dumps([entry.__dict__ for entry in entries], indent=2, sort_keys=True))
    else:
        if not entries:
            print("OK: blocklist has no entries to audit.")
        for entry in entries:
            print(f"{entry.status.upper()}: {entry.path}:{entry.field}: {entry.value}")
            print(f"  detail: {entry.detail}")
    orphaned = [entry for entry in entries if entry.status != BLOCKLIST_ACTIVE]
    return 1 if orphaned else 0


def main() -> int:
    """Run the local model artifact preflight."""
    args = _parse_args()
    if args.audit_blocklist:
        return _run_blocklist_audit(args.blocklist, as_json=args.json)
    rows = check_local_model_artifacts(
        args.paths,
        blocklist_path=args.blocklist,
        promoted_surfaces_path=args.promoted_surfaces,
    )
    if args.json:
        print(json.dumps([row.__dict__ for row in rows], indent=2, sort_keys=True))
    else:
        if not rows:
            print("OK: no local output model_path/resume_from references found.")
        for row in rows:
            print(f"{row.status.upper()}: {row.path}:{row.field}: {row.value}")
            print(f"  reason: {row.reason}")

    has_unblocked = any(row.status in {"unblocked", PROMOTED_BLOCKED_STATUS} for row in rows)
    has_blocked = any(row.status == "blocked" for row in rows)
    return 1 if has_unblocked or (args.fail_on_blocked and has_blocked) else 0


if __name__ == "__main__":
    raise SystemExit(main())
