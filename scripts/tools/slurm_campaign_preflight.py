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
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "robot-sf-slurm-campaign-preflight.v1"
PLACEHOLDER_RE = re.compile(r"<[^<>\s][^<>]*>")
BRACE_PLACEHOLDER_RE = re.compile(r"\{[^{}\s][^{}]*\}")
PERCENT_PLACEHOLDER_RE = re.compile(r"%[A-Za-z][A-Za-z0-9_]*")
GOOD_STATUS = {"native", "available", "ok"}
FULL_COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _anchored_path(value: str, anchor: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = anchor / candidate
    return candidate


def _repository_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", os.fspath(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _check_output_root(
    value: str,
    *,
    label: str,
    anchor: Path,
    blockers: list[str],
) -> Path:
    candidate = _anchored_path(value, anchor)
    normalized = candidate.resolve(strict=False)
    if candidate.is_symlink():
        blockers.append(f"{label} is a symlink and cannot be reserved: {candidate}")
    elif candidate.exists() and not candidate.is_dir():
        blockers.append(f"{label} is not a directory: {candidate}")
    elif candidate.exists() and any(candidate.iterdir()):
        blockers.append(f"{label} is not empty/reservable: {candidate}")
    return normalized


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


def _commit_identity(
    manifest: dict[str, Any],
    *,
    expected_public_commit: str,
    actual_public_commit: str,
    public_repo: Path | None,
    blockers: list[str],
) -> tuple[str, str, str]:
    expected = (
        expected_public_commit or str(manifest.get("expected_public_commit", "") or "").strip()
    )
    actual_source = "argument" if actual_public_commit else "manifest"
    actual = actual_public_commit or str(manifest.get("public_commit", "") or "").strip()
    if public_repo is not None:
        repository_actual = _repository_head(public_repo)
        actual_source = "repository"
        if not repository_actual:
            blockers.append(f"public repository HEAD could not be resolved: {public_repo}")
        elif actual_public_commit and actual_public_commit.lower() != repository_actual.lower():
            blockers.append(
                "actual public commit does not match the supplied public repository HEAD"
            )
        actual = repository_actual
    elif not actual_public_commit:
        blockers.append(
            "actual public commit must be supplied explicitly or bound with public_repo"
        )
    if not FULL_COMMIT_RE.fullmatch(expected):
        blockers.append("expected public commit is missing or invalid")
    if not FULL_COMMIT_RE.fullmatch(actual):
        blockers.append("actual public commit is missing or invalid")
    if (
        expected
        and actual
        and FULL_COMMIT_RE.fullmatch(expected)
        and FULL_COMMIT_RE.fullmatch(actual)
        and expected.lower() != actual.lower()
    ):
        blockers.append(f"public commit mismatch: expected {expected}, actual {actual}")
    return expected, actual, actual_source


def _packet_identity(
    manifest: dict[str, Any], *, manifest_path: Path, blockers: list[str]
) -> tuple[str, Path | None, str]:
    packet = manifest.get("packet")
    if not isinstance(packet, dict):
        blockers.append("packet identity is missing")
        packet = {}
    config_value = str(packet.get("config", "") or "").strip()
    config = _anchored_path(config_value, manifest_path.resolve().parent) if config_value else None
    if not config_value:
        blockers.append("packet.config is missing")
    elif config.is_symlink():
        blockers.append(f"packet config must be a regular file: {config}")
    elif not config.is_file():
        blockers.append(f"packet config does not exist: {config}")
    packet_hash = str(packet.get("sha256", "") or "").strip().lower()
    if not packet_hash:
        blockers.append("packet.sha256 is missing")
    elif not re.fullmatch(r"[0-9a-f]{64}", packet_hash):
        blockers.append("packet.sha256 is not a SHA-256 digest")
    if config is not None and config.is_file() and packet_hash and _sha256(config) != packet_hash:
        blockers.append("packet.sha256 does not match packet.config")
    return config_value, config, packet_hash


def _cell_contract(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    canary_key: str,
    blockers: list[str],
) -> tuple[list[tuple[int, dict[str, Any]]], list[str], dict[str, str]]:
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        blockers.append("campaign cells are missing")
        cells = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            blockers.append(f"cells[{index}] must be an object")
    dict_cells = [(index, cell) for index, cell in enumerate(cells) if isinstance(cell, dict)]
    selected = [
        (index, cell)
        for index, cell in dict_cells
        if not canary_key or str(cell.get("key", "")) == canary_key
    ]
    if canary_key and not selected:
        blockers.append(f"canary key not found: {canary_key}")
    keys: list[str] = []
    output_roots: dict[str, str] = {}
    for index, cell in selected:
        key = _validate_cell(cell, f"cells[{index}]", blockers)
        if key in keys:
            blockers.append(f"duplicate campaign cell key: {key}")
        keys.append(key)
        _record_cell_output_root(
            cell,
            index=index,
            key=key,
            manifest_path=manifest_path,
            output_roots=output_roots,
            blockers=blockers,
        )
    if not canary_key:
        for _index, cell in dict_cells:
            if str(cell.get("key", "")) not in keys:
                keys.append(str(cell.get("key", "")))
    return dict_cells, keys, output_roots


def _record_cell_output_root(
    cell: dict[str, Any],
    *,
    index: int,
    key: str,
    manifest_path: Path,
    output_roots: dict[str, str],
    blockers: list[str],
) -> None:
    output_root_value = str(cell.get("output_root", "") or "").strip()
    if not output_root_value:
        return
    normalized_root = _check_output_root(
        output_root_value,
        label=f"cells[{index}].output_root",
        anchor=manifest_path.resolve().parent,
        blockers=blockers,
    )
    normalized_key = os.fspath(normalized_root)
    if normalized_key in output_roots:
        blockers.append(
            f"campaign cells {output_roots[normalized_key]} and {key} share output_root {normalized_key}"
        )
    else:
        output_roots[normalized_key] = key


def _campaign_contract(
    manifest: dict[str, Any],
    *,
    dict_cells: list[tuple[int, dict[str, Any]]],
    blockers: list[str],
) -> bool:
    paired = manifest.get("paired_keys", [])
    if paired:
        all_keys = {str(cell.get("key", "")) for _index, cell in dict_cells}
        for key in paired:
            if str(key) not in all_keys:
                blockers.append(f"paired campaign cell is missing: {key}")
    aggregate = manifest.get("aggregate")
    if not isinstance(aggregate, dict):
        blockers.append("aggregate artifact contract is missing")
        aggregate = {}
    aggregate_status_ok = str(aggregate.get("status", "") or "").lower() in GOOD_STATUS
    aggregate_contract = str(aggregate.get("artifact_contract", "") or "").strip()
    if not aggregate_status_ok:
        blockers.append("aggregate status is not ready")
    if not aggregate_contract:
        blockers.append("aggregate artifact_contract is missing")
    return aggregate_status_ok and bool(aggregate_contract)


def _output_root_contract(
    output_root: Path | None,
    *,
    manifest_path: Path,
    output_roots: dict[str, str],
    blockers: list[str],
) -> None:
    if output_root is None:
        return
    normalized_output_root = _check_output_root(
        os.fspath(output_root),
        label="output root",
        anchor=manifest_path.resolve().parent,
        blockers=blockers,
    )
    normalized_output_key = os.fspath(normalized_output_root)
    if normalized_output_key in output_roots:
        blockers.append(
            f"output root overlaps campaign cell {output_roots[normalized_output_key]}: {normalized_output_key}"
        )


def preflight(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    canary_key: str = "",
    expected_public_commit: str = "",
    actual_public_commit: str = "",
    public_repo: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a campaign manifest and return a no-submit admission report."""
    blockers = _placeholder_blockers(manifest)
    campaign_id = str(manifest.get("campaign_id", "") or "").strip()
    if not campaign_id:
        blockers.append("campaign_id is missing")
    expected, actual, actual_source = _commit_identity(
        manifest,
        expected_public_commit=expected_public_commit,
        actual_public_commit=actual_public_commit,
        public_repo=public_repo,
        blockers=blockers,
    )
    config_value, config, packet_hash = _packet_identity(
        manifest, manifest_path=manifest_path, blockers=blockers
    )
    dict_cells, keys, output_roots = _cell_contract(
        manifest,
        manifest_path=manifest_path,
        canary_key=canary_key,
        blockers=blockers,
    )
    aggregate_validated = _campaign_contract(manifest, dict_cells=dict_cells, blockers=blockers)
    _output_root_contract(
        output_root,
        manifest_path=manifest_path,
        output_roots=output_roots,
        blockers=blockers,
    )
    canary_safe = not blockers
    full_campaign_safe = canary_safe and not canary_key
    submit_safe = full_campaign_safe

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked" if blockers else "canary_ready" if canary_key else "ready",
        "submit_safe": submit_safe,
        "canary_safe": canary_safe,
        "full_campaign_safe": full_campaign_safe,
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
            "actual_commit_source": actual_source,
            "packet_config": config_value,
            "packet_config_resolved": str(config.resolve()) if config else "",
            "packet_sha256": packet_hash,
        },
        "canary_coverage": {
            "mode": "canary" if canary_key else "full-structure",
            "selected_key": canary_key,
            "validated_keys": keys,
            "unvalidated_keys": [
                str(cell.get("key", ""))
                for _index, cell in dict_cells
                if str(cell.get("key", "")) not in keys
            ],
            "aggregate_validated": aggregate_validated,
        },
        "planner_keys": keys,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--canary-key", default="")
    parser.add_argument("--expected-public-commit", default="")
    parser.add_argument("--actual-public-commit", default="")
    parser.add_argument(
        "--public-repo",
        type=Path,
        help="bind actual_public_commit to git HEAD in this local repository",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the no-submit campaign preflight CLI."""
    args = _parser().parse_args(argv)
    try:
        manifest = _load(args.manifest)
        report = preflight(
            manifest,
            manifest_path=args.manifest,
            canary_key=args.canary_key,
            expected_public_commit=args.expected_public_commit,
            actual_public_commit=args.actual_public_commit,
            public_repo=args.public_repo,
            output_root=args.output_root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "input_error",
            "submit_safe": False,
            "canary_safe": False,
            "full_campaign_safe": False,
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
            f"campaign-preflight: status={report['status']} submit_safe={report['submit_safe']} canary_safe={report['canary_safe']}"
        )
        for blocker in report["blockers"]:
            print(f"blocker: {blocker}", file=sys.stderr)
    safe_for_invocation = report["canary_safe"] if args.canary_key else report["submit_safe"]
    return 0 if safe_for_invocation else 2


if __name__ == "__main__":
    raise SystemExit(main())
