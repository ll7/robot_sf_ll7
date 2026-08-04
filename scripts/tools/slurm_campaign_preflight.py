#!/usr/bin/env python3
"""Validate a Slurm campaign packet without allocating compute.

The input is a JSON launch manifest.  This command checks packet identity,
campaign-cell completeness, paired horizons, native planner availability, and
output/artifact boundaries.  It never invokes ssh, sbatch, or the scheduler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "robot-sf-slurm-campaign-preflight.v1"
PLACEHOLDER_RE = re.compile(r"<[^<>\s][^<>]*>")
BRACE_PLACEHOLDER_RE = re.compile(r"\{[^{}\s][^{}]*\}")
PERCENT_PLACEHOLDER_RE = re.compile(r"%[A-Za-z][A-Za-z0-9_]*")
GOOD_STATUS = {"native", "available", "ok"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("campaign manifest must be a JSON object")
    return value


def _placeholder_blockers(value: Any, prefix: str = "manifest") -> list[str]:
    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            result.extend(_placeholder_blockers(item, f"{prefix}.{key}"))
        return result
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            result.extend(_placeholder_blockers(item, f"{prefix}[{index}]"))
        return result
    text = str(value or "")
    if (
        PLACEHOLDER_RE.search(text)
        or BRACE_PLACEHOLDER_RE.search(text)
        or PERCENT_PLACEHOLDER_RE.search(text)
    ):
        return [f"unresolved placeholder in {prefix}"]
    return []


def _cell_status(cell: dict[str, Any]) -> tuple[bool, str]:
    status = str(cell.get("status", "") or "").strip().lower()
    if status in GOOD_STATUS:
        return True, status
    required = {key: cell.get(key) is True for key in ("native", "available", "ok")}
    if all(required.values()):
        return True, "native/available/ok"
    missing = ",".join(key for key, present in required.items() if not present)
    return False, f"missing status proof: {missing}"


def _validate_cell(cell: dict[str, Any], path: str, blockers: list[str]) -> str:
    key = str(cell.get("key", "") or "").strip()
    if not key:
        blockers.append(f"{path}.key is missing")
    for field in ("output_root", "artifact_contract"):
        if not str(cell.get(field, "") or "").strip():
            blockers.append(f"{path}.{field} is missing")
    valid_status, status = _cell_status(cell)
    if not valid_status:
        blockers.append(f"{path}: {status}")
    declared = cell.get("declared_rows")
    instantiated = cell.get("instantiated_rows")
    if declared is None or instantiated is None:
        blockers.append(f"{path}: declared_rows and instantiated_rows are required")
    else:
        try:
            rows_match = int(declared) == int(instantiated)
        except (TypeError, ValueError):
            rows_match = False
        if not rows_match:
            blockers.append(
                f"{path}: declared_rows={declared} differs from instantiated_rows={instantiated}"
            )
    return key


def preflight(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    canary_key: str = "",
    expected_public_commit: str = "",
    actual_public_commit: str = "",
    output_root: Path | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    blockers.extend(_placeholder_blockers(manifest))
    campaign_id = str(manifest.get("campaign_id", "") or "").strip()
    if not campaign_id:
        blockers.append("campaign_id is missing")

    expected = (
        expected_public_commit
        or str(manifest.get("expected_public_commit", "") or "").strip()
    )
    actual = (
        actual_public_commit or str(manifest.get("public_commit", "") or "").strip()
    )
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", expected):
        blockers.append("expected public commit is missing or invalid")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", actual):
        blockers.append("actual public commit is missing or invalid")
    if (
        expected
        and actual
        and not (
            expected.lower().startswith(actual.lower())
            or actual.lower().startswith(expected.lower())
        )
    ):
        blockers.append(f"public commit mismatch: expected {expected}, actual {actual}")

    packet = manifest.get("packet")
    if not isinstance(packet, dict):
        blockers.append("packet identity is missing")
        packet = {}
    config_value = str(packet.get("config", "") or "").strip()
    config = Path(config_value) if config_value else None
    if not config_value:
        blockers.append("packet.config is missing")
    elif config.is_absolute() and not config.is_file():
        blockers.append(f"packet config does not exist: {config}")
    packet_hash = str(packet.get("sha256", "") or "").strip().lower()
    if not packet_hash:
        blockers.append("packet.sha256 is missing")
    elif not re.fullmatch(r"[0-9a-f]{64}", packet_hash):
        blockers.append("packet.sha256 is not a SHA-256 digest")
    if (
        config is not None
        and config.is_file()
        and packet_hash
        and _sha256(config) != packet_hash
    ):
        blockers.append("packet.sha256 does not match packet.config")

    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        blockers.append("campaign cells are missing")
        cells = []
    keys: list[str] = []
    output_roots: dict[str, str] = {}
    selected = [
        cell
        for cell in cells
        if isinstance(cell, dict)
        and (not canary_key or str(cell.get("key", "")) == canary_key)
    ]
    if canary_key and not selected:
        blockers.append(f"canary key not found: {canary_key}")
    for index, cell in enumerate(selected):
        key = _validate_cell(cell, f"cells[{index}]", blockers)
        if key in keys:
            blockers.append(f"duplicate campaign cell key: {key}")
        keys.append(key)
        output_root_value = str(cell.get("output_root", "") or "").strip()
        if output_root_value and output_root_value in output_roots:
            blockers.append(
                f"campaign cells {output_roots[output_root_value]} and {key} share output_root {output_root_value}"
            )
        elif output_root_value:
            output_roots[output_root_value] = key

    if not canary_key:
        for cell in cells:
            if isinstance(cell, dict) and str(cell.get("key", "")) not in keys:
                keys.append(str(cell.get("key", "")))

    paired = manifest.get("paired_keys", [])
    if paired:
        all_keys = {
            str(cell.get("key", "")) for cell in cells if isinstance(cell, dict)
        }
        for key in paired:
            if str(key) not in all_keys:
                blockers.append(f"paired campaign cell is missing: {key}")

    aggregate = manifest.get("aggregate")
    if not isinstance(aggregate, dict):
        blockers.append("aggregate artifact contract is missing")
        aggregate = {}
    if str(aggregate.get("status", "") or "").lower() not in GOOD_STATUS:
        blockers.append("aggregate status is not ready")
    if not str(aggregate.get("artifact_contract", "") or "").strip():
        blockers.append("aggregate artifact_contract is missing")

    aggregate_validated = str(
        aggregate.get("status", "") or ""
    ).lower() in GOOD_STATUS and bool(
        str(aggregate.get("artifact_contract", "") or "").strip()
    )
    if output_root is not None:
        if output_root.is_symlink() or (
            output_root.exists()
            and (not output_root.is_dir() or any(output_root.iterdir()))
        ):
            blockers.append(f"output root is not empty/reservable: {output_root}")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready" if not blockers else "blocked",
        "submit_safe": not blockers,
        "no_submit": True,
        "blockers": list(dict.fromkeys(blockers)),
        "remediation": [
            "regenerate the exact campaign packet and rerun the canary",
            "resolve native/available/ok and declared-versus-instantiated row mismatches",
            "do not allocate compute while this report is blocked",
        ]
        if blockers
        else [],
        "identities": {
            "manifest": str(manifest_path.resolve()),
            "campaign_id": campaign_id,
            "expected_public_commit": expected,
            "actual_public_commit": actual,
            "packet_config": config_value,
            "packet_sha256": packet_hash,
        },
        "canary_coverage": {
            "mode": "canary" if canary_key else "full-structure",
            "selected_key": canary_key,
            "validated_keys": keys,
            "aggregate_validated": aggregate_validated,
        },
        "planner_keys": keys,
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--canary-key", default="")
    parser.add_argument("--expected-public-commit", default="")
    parser.add_argument("--actual-public-commit", default="")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = _load(args.manifest)
        report = preflight(
            manifest,
            manifest_path=args.manifest,
            canary_key=args.canary_key,
            expected_public_commit=args.expected_public_commit,
            actual_public_commit=args.actual_public_commit,
            output_root=args.output_root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "input_error",
            "submit_safe": False,
            "no_submit": True,
            "blockers": [str(exc)],
            "remediation": ["provide a readable JSON campaign manifest"],
            "identities": {"manifest": str(args.manifest)},
            "canary_coverage": {
                "mode": "unknown",
                "selected_key": args.canary_key,
                "validated_keys": [],
            },
            "planner_keys": [],
        }
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"campaign-preflight: status={report['status']} submit_safe={report['submit_safe']}"
        )
        for blocker in report["blockers"]:
            print(f"blocker: {blocker}", file=sys.stderr)
    return 0 if report["submit_safe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
