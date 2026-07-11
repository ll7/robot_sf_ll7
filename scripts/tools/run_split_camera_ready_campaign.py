#!/usr/bin/env python3
"""Plan per-arm camera-ready runs and aggregate compatible native results.

The ``plan`` command is CPU-only: it validates a splitter manifest and writes exact
commands for independently scheduled child campaigns.  The ``aggregate`` command
reads those campaign roots, excludes adapter/fallback/degraded rows, and feeds the
compatible native rows to the existing camera-ready report writer.

This tool never submits compute or runs a benchmark campaign itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.camera_ready._artifacts import _write_json
from robot_sf.benchmark.camera_ready._reporting import (
    build_campaign_credibility_scorecard,
    write_campaign_report,
)

PACKET_SCHEMA = "split-camera-ready-execution-packet.v1"
AGGREGATE_SCHEMA = "split-camera-ready-native-aggregate.v1"
CLAIM_BOUNDARY = (
    "Native-row aggregation of independently executed split campaigns. Adapter, fallback, "
    "degraded, unavailable, failed, missing, and incompatible rows are excluded and cannot "
    "support benchmark-success claims in this aggregate."
)
COMPATIBILITY_FIELDS = (
    "scenario_matrix",
    "scenario_matrix_hash",
    "git_hash",
    "paper_interpretation_profile",
    "paper_profile_version",
    "observation_noise_hash",
    "snqi_weights_sha256",
    "snqi_baseline_sha256",
    "kinematics_matrix",
)
AGGREGATE_COMPATIBILITY_FIELDS = (*COMPATIBILITY_FIELDS, "seed_policy")
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object or fail with a path-specific error."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    """Return the raw-byte SHA-256 digest for ``path``."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_path(path: Path) -> str:
    """Return a repository-relative path when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _validated_children(  # noqa: C901
    split_manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a split manifest and prove every declared child still matches its digest."""
    manifest = _read_json(split_manifest_path)
    source_config = manifest.get("source_config")
    source_digest = manifest.get("source_sha256")
    if not isinstance(source_config, str) or not source_config:
        raise ValueError(f"{split_manifest_path}.source_config must be a non-empty path")
    if not isinstance(source_digest, str) or len(source_digest) != 64:
        raise ValueError(f"{split_manifest_path}.source_sha256 must be a SHA-256 hex digest")
    source_path = Path(source_config)
    if not source_path.is_file():
        raise ValueError(f"split source config is missing: {source_path}")
    observed_source_digest = _sha256(source_path)
    if observed_source_digest != source_digest:
        raise ValueError(
            f"split source digest mismatch for {source_path}: expected {source_digest}, "
            f"observed {observed_source_digest}"
        )
    children = manifest.get("children")
    if not isinstance(children, list) or not children:
        raise ValueError(f"{split_manifest_path} must declare a non-empty children list")

    manifest_dir = split_manifest_path.resolve().parent
    validated: list[dict[str, Any]] = []
    seen_planners: set[str] = set()
    for index, child in enumerate(children):
        if not isinstance(child, dict):
            raise ValueError(f"children[{index}] must be an object")
        filename = child.get("filename")
        digest = child.get("sha256")
        planners = child.get("planner_keys")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"children[{index}].filename must be a non-empty string")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"children[{index}].sha256 must be a SHA-256 hex digest")
        if (
            not isinstance(planners, list)
            or not planners
            or not all(isinstance(key, str) and key for key in planners)
        ):
            raise ValueError(f"children[{index}].planner_keys must contain planner names")
        overlap = seen_planners.intersection(planners)
        if overlap:
            raise ValueError(f"duplicate planner keys across split children: {sorted(overlap)}")
        seen_planners.update(planners)

        config_path = manifest_dir / filename
        if not config_path.is_file():
            raise ValueError(f"split child is missing: {config_path}")
        actual_digest = _sha256(config_path)
        if actual_digest != digest:
            raise ValueError(
                f"split child digest mismatch for {config_path}: expected {digest}, "
                f"observed {actual_digest}"
            )
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"split child must contain a YAML mapping: {config_path}")
        seed_policy = payload.get("seed_policy")
        if not isinstance(seed_policy, dict):
            raise ValueError(f"split child is missing seed_policy: {config_path}")
        validated.append(
            {
                "config_path": _repo_path(config_path),
                "config_sha256": digest,
                "planner_keys": list(planners),
                "seed_policy": deepcopy(seed_policy),
            }
        )
    return manifest, validated


def build_execution_packet(
    split_manifest_path: Path,
    *,
    output_root: Path,
    campaign_prefix: str,
) -> dict[str, Any]:
    """Build an exact, deterministic command packet for all split children."""
    if not _SAFE_LABEL.fullmatch(campaign_prefix):
        raise ValueError("campaign_prefix must be a non-empty directory-safe label")
    manifest, children = _validated_children(split_manifest_path)
    output_root_ref = _repo_path(output_root)
    arms: list[dict[str, Any]] = []
    for child in children:
        arm_label = "__".join(child["planner_keys"])
        if not _SAFE_LABEL.fullmatch(arm_label):
            raise ValueError(f"planner keys do not form a directory-safe arm label: {arm_label}")
        campaign_id = f"{campaign_prefix}__arm_{arm_label}"
        command = [
            "uv",
            "run",
            "python",
            "scripts/tools/run_camera_ready_benchmark.py",
            "--config",
            child["config_path"],
            "--output-root",
            output_root_ref,
            "--campaign-id",
            campaign_id,
            "--mode",
            "run",
            "--skip-publication-bundle",
        ]
        arms.append(
            {
                **child,
                "campaign_id": campaign_id,
                "campaign_root": str(Path(output_root_ref) / campaign_id),
                "command": shlex.join(command),
                "status": "planned_not_executed",
            }
        )
    return {
        "schema_version": PACKET_SCHEMA,
        "status": "planned_not_executed",
        "claim_boundary": (
            "Execution plan only; no campaign was run and no benchmark result is claimed."
        ),
        "split_manifest": _repo_path(split_manifest_path),
        "split_manifest_sha256": _sha256(split_manifest_path),
        "source_config": manifest.get("source_config"),
        "source_sha256": manifest.get("source_sha256"),
        "campaign_prefix": campaign_prefix,
        "output_root": output_root_ref,
        "arms": arms,
    }


def _load_packet(path: Path) -> dict[str, Any]:
    packet = _read_json(path)
    if packet.get("schema_version") != PACKET_SCHEMA:
        raise ValueError(f"{path} schema_version must be {PACKET_SCHEMA!r}")
    manifest_path = Path(str(packet.get("split_manifest", "")))
    if not manifest_path.is_file():
        raise ValueError(f"packet split manifest is missing: {manifest_path}")
    expected_manifest_sha = packet.get("split_manifest_sha256")
    if _sha256(manifest_path) != expected_manifest_sha:
        raise ValueError("packet split manifest digest no longer matches")
    _, current_children = _validated_children(manifest_path)
    packet_arms = packet.get("arms")
    if not isinstance(packet_arms, list) or len(packet_arms) != len(current_children):
        raise ValueError("packet arms no longer match the split manifest")
    for packet_arm, current in zip(packet_arms, current_children, strict=True):
        if not isinstance(packet_arm, dict):
            raise ValueError("packet arm must be an object")
        for field in ("config_path", "config_sha256", "planner_keys", "seed_policy"):
            if packet_arm.get(field) != current[field]:
                raise ValueError(f"packet arm {field} no longer matches the split manifest")
    return packet


def _row_exclusion_reason(row: dict[str, Any]) -> str | None:
    status = str(row.get("status", "")).strip().lower()
    execution_mode = str(row.get("execution_mode", "")).strip().lower()
    readiness = str(row.get("readiness_status", "")).strip().lower()
    availability = str(row.get("availability_status", "")).strip().lower()
    if status != "ok":
        return f"row_status_{status or 'missing'}"
    if execution_mode != "native":
        return f"execution_mode_{execution_mode or 'missing'}"
    if readiness != "native":
        return f"readiness_status_{readiness or 'missing'}"
    if availability != "available":
        return f"availability_status_{availability or 'missing'}"
    if row.get("benchmark_success") not in {True, "true", "True"}:
        return "benchmark_success_false"
    return None


def _compatibility_values(campaign: dict[str, Any]) -> dict[str, Any]:
    return {field: campaign.get(field) for field in COMPATIBILITY_FIELDS}


def aggregate_execution_packet(  # noqa: C901, PLR0915
    packet_path: Path, *, output_dir: Path
) -> dict[str, Any]:
    """Aggregate only compatible native rows from completed packet campaign roots."""
    packet = _load_packet(packet_path)
    included_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    reference_compatibility: dict[str, Any] | None = None

    for arm in packet["arms"]:
        campaign_root = Path(str(arm["campaign_root"]))
        summary_path = campaign_root / "reports" / "campaign_summary.json"
        campaign_manifest_path = campaign_root / "campaign_manifest.json"
        source = {
            "planner_keys": list(arm["planner_keys"]),
            "campaign_id": arm["campaign_id"],
            "summary_path": _repo_path(summary_path),
            "campaign_manifest_path": _repo_path(campaign_manifest_path),
        }
        missing = [
            path.name for path in (campaign_manifest_path, summary_path) if not path.is_file()
        ]
        if missing:
            exclusions.append(
                {
                    **source,
                    "reason": "missing_campaign_artifacts",
                    "missing": missing,
                    "blocking": True,
                }
            )
            sources.append({**source, "status": "missing"})
            continue

        campaign_manifest = _read_json(campaign_manifest_path)
        summary = _read_json(summary_path)
        campaign = summary.get("campaign")
        rows = summary.get("planner_rows")
        if not isinstance(campaign, dict) or not isinstance(rows, list):
            exclusions.append({**source, "reason": "malformed_campaign_summary", "blocking": True})
            sources.append({**source, "status": "malformed"})
            continue
        observed_keys = [str(row.get("planner_key")) for row in rows if isinstance(row, dict)]
        if sorted(observed_keys) != sorted(arm["planner_keys"]):
            exclusions.append(
                {
                    **source,
                    "reason": "planner_key_mismatch",
                    "observed_planner_keys": observed_keys,
                    "blocking": True,
                }
            )
            sources.append({**source, "status": "incompatible"})
            continue
        if campaign.get("campaign_id") != arm["campaign_id"]:
            exclusions.append({**source, "reason": "campaign_id_mismatch", "blocking": True})
            sources.append({**source, "status": "incompatible"})
            continue
        if campaign_manifest.get("campaign_id") != arm["campaign_id"]:
            exclusions.append(
                {**source, "reason": "manifest_campaign_id_mismatch", "blocking": True}
            )
            sources.append({**source, "status": "incompatible"})
            continue
        if campaign_manifest.get("config_hash") != arm["config_sha256"][:16]:
            exclusions.append({**source, "reason": "config_hash_mismatch", "blocking": True})
            sources.append({**source, "status": "incompatible"})
            continue
        manifest_seed_policy = campaign_manifest.get("seed_policy")
        if not isinstance(manifest_seed_policy, dict):
            exclusions.append({**source, "reason": "missing_seed_provenance", "blocking": True})
            sources.append({**source, "status": "incompatible"})
            continue
        planned_seed_policy = arm["seed_policy"]
        seed_mismatches = sorted(
            key
            for key, value in planned_seed_policy.items()
            if manifest_seed_policy.get(key) != value
        )
        if seed_mismatches:
            exclusions.append(
                {
                    **source,
                    "reason": "seed_policy_mismatch",
                    "mismatched_fields": seed_mismatches,
                    "blocking": True,
                }
            )
            sources.append({**source, "status": "incompatible"})
            continue

        compatibility = _compatibility_values(campaign)
        compatibility["seed_policy"] = manifest_seed_policy
        if reference_compatibility is None:
            reference_compatibility = compatibility
        elif compatibility != reference_compatibility:
            mismatches = sorted(
                field
                for field in AGGREGATE_COMPATIBILITY_FIELDS
                if compatibility[field] != reference_compatibility[field]
            )
            exclusions.append(
                {
                    **source,
                    "reason": "campaign_contract_mismatch",
                    "mismatched_fields": mismatches,
                    "blocking": True,
                }
            )
            sources.append({**source, "status": "incompatible"})
            continue

        row = rows[0]
        reason = _row_exclusion_reason(row)
        if reason is None:
            included_rows.append(deepcopy(row))
            sources.append({**source, "status": "included_native"})
        else:
            exclusions.append(
                {
                    **source,
                    "reason": reason,
                    "blocking": reason.startswith(("row_status_", "availability_status_"))
                    or reason == "benchmark_success_false",
                }
            )
            sources.append({**source, "status": "excluded_non_native"})

    blocking = [item for item in exclusions if item["blocking"]]
    complete = bool(included_rows) and not blocking
    status = "native_aggregate_complete" if complete else "blocked"
    campaign_id = f"{packet['campaign_prefix']}__native_aggregate"
    campaign = {
        "schema_version": "benchmark-camera-ready-campaign.v1",
        "campaign_id": campaign_id,
        "name": campaign_id,
        **(reference_compatibility or {}),
        "campaign_execution_status": "completed" if complete else "partial_or_blocked",
        "evidence_status": "nominal" if complete else "blocked",
        "benchmark_success": complete,
        "status": status,
        "status_reason": (
            "all available compatible native rows aggregated; non-native rows are explicit exclusions"
            if complete
            else "one or more required split arms are missing, failed, malformed, or incompatible"
        ),
        "successful_runs": len(included_rows),
        "total_runs": len(packet["arms"]),
        "non_success_runs": len(exclusions),
        "accepted_unavailable_runs": sum(not item["blocking"] for item in exclusions),
        "unexpected_failed_runs": len(blocking),
        "row_status_summary": {
            "successful_evidence_rows": len(included_rows),
            "accepted_unavailable_rows": sum(not item["blocking"] for item in exclusions),
            "unexpected_failed_rows": len(blocking),
            "fallback_or_degraded_rows": sum(
                "fallback" in item["reason"] or "degraded" in item["reason"] for item in exclusions
            ),
        },
    }
    summary = {
        "campaign": campaign,
        "planner_rows": sorted(included_rows, key=lambda row: str(row.get("planner_key", ""))),
        "runs": [],
        "warnings": [f"Excluded {item['campaign_id']}: {item['reason']}" for item in exclusions],
        "claim_boundary": CLAIM_BOUNDARY,
        "source_campaigns": sources,
        "excluded_rows": exclusions,
        "artifacts": {
            "execution_packet": _repo_path(packet_path),
            "aggregate_manifest": _repo_path(output_dir / "aggregate_manifest.json"),
            "campaign_summary_json": _repo_path(output_dir / "reports/campaign_summary.json"),
            "campaign_report_md": _repo_path(output_dir / "reports/campaign_report.md"),
        },
    }
    summary["credibility_scorecard"] = build_campaign_credibility_scorecard(summary)
    reports_dir = output_dir / "reports"
    _write_json(reports_dir / "campaign_summary.json", summary)
    write_campaign_report(reports_dir / "campaign_report.md", summary)
    aggregate_manifest = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": status,
        "benchmark_success": complete,
        "claim_boundary": CLAIM_BOUNDARY,
        "execution_packet": _repo_path(packet_path),
        "execution_packet_sha256": _sha256(packet_path),
        "compatibility_fields": list(AGGREGATE_COMPATIBILITY_FIELDS),
        "compatibility_values": reference_compatibility,
        "included_planner_keys": [row.get("planner_key") for row in summary["planner_rows"]],
        "excluded_rows": exclusions,
        "source_campaigns": sources,
        "reports": summary["artifacts"],
    }
    _write_json(output_dir / "aggregate_manifest.json", aggregate_manifest)
    return aggregate_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Write a CPU-only per-arm execution packet.")
    plan.add_argument("--split-manifest", type=Path, required=True)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--campaign-prefix", required=True)
    plan.add_argument("--packet", type=Path, required=True)

    aggregate = subparsers.add_parser(
        "aggregate", help="Aggregate completed compatible native split-campaign rows."
    )
    aggregate.add_argument("--packet", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected planning or aggregation command."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            result = build_execution_packet(
                args.split_manifest,
                output_root=args.output_root,
                campaign_prefix=args.campaign_prefix,
            )
            _write_json(args.packet, result)
        else:
            result = aggregate_execution_packet(args.packet, output_dir=args.output_dir)
    except ValueError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") in {"planned_not_executed", "native_aggregate_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
