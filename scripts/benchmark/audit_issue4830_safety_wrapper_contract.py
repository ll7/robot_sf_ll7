#!/usr/bin/env python3
"""Audit issue #4830 campaign artifacts without inferring missing metric semantics.

The camera-ready runner emits valid arm-level and episode-level artifacts.  The
older issue #3501 paired report contract expects a normalized row with
``metric_values`` for every paired outcome.  This audit makes that boundary
explicit: it validates the completed campaign surface, reports the available
source fields, and fails closed when the normalized factorial metrics are not
present.  It does not convert similarly named metrics into new scientific
quantities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "robot_sf.issue_4830_safety_wrapper_contract_audit.v1"
EXPECTED_WRAPPER_ARMS = ("wrapper_off", "wrapper_on")
EXPECTED_PLANNERS = ("orca", "social_force", "prediction_planner")
EXPECTED_ARM_KEYS = tuple(
    f"{planner}__{arm}" for planner in EXPECTED_PLANNERS for arm in EXPECTED_WRAPPER_ARMS
)
REQUIRED_FACTORIAL_METRICS = (
    "exact_collision_probability",
    "near_miss_probability",
    "min_predicted_separation_m",
    "completion_probability",
    "progress_at_timeout",
    "false_positive_stop_rate",
    "stop_yield_latency_s",
    "wrapper_intervention_rate",
)
STANDARD_ARTIFACTS = (
    "campaign_manifest.json",
    "manifest.json",
    "preflight.json",
    "run_meta.json",
    "reports/campaign_summary.json",
    "reports/campaign_integrity.json",
    "reports/campaign_credibility_scorecard.json",
    "reports/campaign_report.md",
    "reports/matrix_summary.json",
    "reports/comparability_matrix.json",
    "reports/post_campaign_stage_status.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and fail with a path-specific error."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one source artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arm_identity(path: Path) -> tuple[str, str, str] | None:
    """Parse ``<planner>__<wrapper_arm>__<kinematics>`` run directories."""
    try:
        planner, wrapper_arm, kinematics = path.name.rsplit("__", 2)
    except ValueError:
        return None
    if wrapper_arm not in EXPECTED_WRAPPER_ARMS:
        return None
    return planner, wrapper_arm, kinematics


def _iter_episode_records(campaign_root: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield arm identity and episode records from the external campaign root."""
    runs_root = campaign_root / "runs"
    for episode_path in sorted(runs_root.glob("*__*__*/episodes.jsonl")):
        identity = _arm_identity(episode_path.parent)
        if identity is None:
            continue
        arm_key = "__".join(identity[:2])
        for line_number, line in enumerate(
            episode_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{episode_path}:{line_number} must contain an object")
            yield arm_key, record


def _episode_source_presence(records: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Summarize source fields without assigning them new metric meanings."""
    paths = {
        "metrics.wrapper_intervention_rate": lambda row: row.get("metrics", {}).get(
            "wrapper_intervention_rate"
        ),
        "metrics.success": lambda row: row.get("metrics", {}).get("success"),
        "metrics.collisions": lambda row: row.get("metrics", {}).get("collisions"),
        "metrics.near_misses": lambda row: row.get("metrics", {}).get("near_misses"),
        "metrics.clearing_distance_min": lambda row: row.get("metrics", {}).get(
            "clearing_distance_min"
        ),
        "event_ledger.exact_events.collision": lambda row: (
            row.get("event_ledger", {}).get("exact_events", {}).get("collision")
        ),
        "algorithm_metadata.safety_wrapper": lambda row: row.get("algorithm_metadata", {}).get(
            "safety_wrapper"
        ),
    }
    counts: dict[str, int] = {}
    for source_path, getter in paths.items():
        counts[source_path] = sum(getter(row) is not None for _, row in records)
    return {"record_count": len(records), "non_null_record_counts": counts}


def _load_campaign_status(campaign_root: Path) -> dict[str, Any]:
    """Load the standard campaign status fields used by the audit."""
    summary_path = campaign_root / "reports/campaign_summary.json"
    integrity_path = campaign_root / "reports/campaign_integrity.json"
    stage_path = campaign_root / "reports/post_campaign_stage_status.json"
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    integrity = _read_json(integrity_path) if integrity_path.is_file() else {}
    stage = _read_json(stage_path) if stage_path.is_file() else {}
    campaign = summary.get("campaign") if isinstance(summary.get("campaign"), Mapping) else {}
    return {
        "campaign_id": campaign.get("campaign_id"),
        "git_hash": campaign.get("git_hash"),
        "campaign_execution_status": campaign.get("campaign_execution_status"),
        "evidence_status": campaign.get("evidence_status"),
        "benchmark_success": campaign.get("benchmark_success"),
        "total_episodes": campaign.get("total_episodes"),
        "total_runs": campaign.get("total_runs"),
        "successful_runs": campaign.get("successful_runs"),
        "unexpected_failed_runs": campaign.get("unexpected_failed_runs"),
        "fallback_or_degraded_rows": (
            campaign.get("row_status_summary", {}).get("fallback_or_degraded_rows")
            if isinstance(campaign.get("row_status_summary"), Mapping)
            else None
        ),
        "integrity_status": integrity.get("status"),
        "post_campaign_stage_status": stage.get("post_campaign_stage", {}).get("status")
        if isinstance(stage.get("post_campaign_stage"), Mapping)
        else None,
    }


def audit_campaign(
    campaign_root: str | Path,
    *,
    config_path: str | None = None,
    config_sha256: str | None = None,
    artifact_prefix: str | None = None,
    source_location: str | None = None,
) -> dict[str, Any]:
    """Audit a completed #4830 campaign root and return a JSON-safe report."""
    root = Path(campaign_root).resolve()
    if not root.is_dir():
        raise ValueError(f"campaign root does not exist: {root}")

    missing_artifacts = [path for path in STANDARD_ARTIFACTS if not (root / path).is_file()]
    records = list(_iter_episode_records(root))
    arm_counts = Counter(arm_key for arm_key, _ in records)
    arm_pair_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pairing_keys: Counter[tuple[str, str, int]] = Counter()
    malformed_records = 0
    direct_metric_values = 0
    for arm_key, record in records:
        planner, wrapper_arm = arm_key.rsplit("__", 1)
        scenario_id = record.get("scenario_id")
        seed = record.get("seed")
        if isinstance(scenario_id, str) and isinstance(seed, int) and not isinstance(seed, bool):
            pairing_keys[(planner, scenario_id, seed)] += 1
            arm_pair_counts[planner][wrapper_arm] += 1
        else:
            malformed_records += 1
        if isinstance(record.get("metric_values"), Mapping):
            direct_metric_values += 1

    source_presence = _episode_source_presence(records)
    missing_metric_contract = list(REQUIRED_FACTORIAL_METRICS)
    standard_status = _load_campaign_status(root)
    standard_valid = (
        not missing_artifacts
        and standard_status.get("campaign_execution_status") == "completed"
        and standard_status.get("evidence_status") == "valid"
        and standard_status.get("integrity_status") == "valid"
        and standard_status.get("unexpected_failed_runs") == 0
        and standard_status.get("fallback_or_degraded_rows") == 0
    )
    expected_arm_set = set(EXPECTED_ARM_KEYS)
    observed_arm_set = set(arm_counts)
    duplicate_pairings = sorted(
        {
            "planner": planner,
            "scenario_id": scenario_id,
            "seed": seed,
            "count": count,
        }
        for (planner, scenario_id, seed), count in pairing_keys.items()
        if count != 2
    )
    campaign_id = standard_status.get("campaign_id")
    producing_commit = standard_status.get("git_hash")
    source_root = source_location or str(root)
    source_artifacts: dict[str, dict[str, Any]] = {}
    for relative in STANDARD_ARTIFACTS:
        source_path = root / relative
        if not source_path.is_file():
            continue
        artifact_path = (
            f"{source_root.rstrip('/')}/{relative}"
            if source_location
            else ((Path(artifact_prefix) / relative).as_posix() if artifact_prefix else relative)
        )
        source_artifacts[relative] = {
            "artifact_path": artifact_path,
            "bytes": source_path.stat().st_size,
            "location": f"{source_root.rstrip('/')}/{relative}",
            "sha256": _sha256(source_path),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 4830,
        "campaign_id": campaign_id,
        "campaign_root": source_root,
        "config_path": config_path,
        "config_sha256": config_sha256,
        "producing_commit": producing_commit,
        "source_location": source_root,
        "claim_boundary": (
            "This audit validates campaign execution and records the boundary between the "
            "camera-ready artifact surface and the issue #3501 normalized paired-row report. "
            "It does not infer missing metric semantics, promote evidence, or make a safety claim."
        ),
        "standard_campaign": {
            "status": "valid" if standard_valid else "incomplete",
            **standard_status,
            "missing_artifacts": missing_artifacts,
        },
        "factorial_contract": {
            "status": "blocked" if direct_metric_values != len(records) else "present",
            "required_metric_names": list(REQUIRED_FACTORIAL_METRICS),
            "normalized_metric_values_record_count": direct_metric_values,
            "blocked_metrics": missing_metric_contract
            if direct_metric_values != len(records)
            else [],
            "reason": (
                "Camera-ready episode records do not contain the issue #3501 metric_values row "
                "contract. Candidate fields with similar names remain unadmitted because their "
                "semantics are not identical to the required quantities."
                if direct_metric_values != len(records)
                else None
            ),
        },
        "roster": {
            "expected_arm_keys": list(EXPECTED_ARM_KEYS),
            "observed_arm_keys": sorted(observed_arm_set),
            "missing_arm_keys": sorted(expected_arm_set - observed_arm_set),
            "unexpected_arm_keys": sorted(observed_arm_set - expected_arm_set),
            "arm_episode_counts": dict(sorted(arm_counts.items())),
            "planner_wrapper_pair_counts": {
                planner: dict(sorted(arms.items()))
                for planner, arms in sorted(arm_pair_counts.items())
            },
            "episode_record_count": len(records),
            "unique_pairing_key_count": len(pairing_keys),
            "malformed_record_count": malformed_records,
            "non_pairing_counts": duplicate_pairings,
        },
        "source_presence": source_presence,
        "source_artifacts": source_artifacts,
        "stop_conditions": [
            "Do not run the existing #3501 paired report builder until normalized metric_values rows exist.",
            "Do not map clearing distance, progress, or diagnostic false-stop fields to different metric names without a reviewed semantic contract.",
            "Do not promote this campaign to dissertation evidence from this audit alone.",
        ],
    }


def _render_readme(audit: Mapping[str, Any]) -> str:
    """Render a compact human-readable audit note."""
    standard = audit["standard_campaign"]
    factorial = audit["factorial_contract"]
    roster = audit["roster"]
    lines = [
        "# Issue #4830 safety-wrapper campaign evidence audit",
        "",
        str(audit["claim_boundary"]),
        "",
        f"- Standard campaign status: `{standard['status']}`",
        f"- Campaign execution: `{standard.get('campaign_execution_status')}`",
        f"- Evidence status: `{standard.get('evidence_status')}`",
        f"- Public commit: `{standard.get('git_hash')}`",
        f"- Episodes: `{standard.get('total_episodes')}`",
        f"- Arms: `{standard.get('total_runs')}`",
        f"- Observed episode records: `{roster['episode_record_count']}`",
        f"- Paired-row contract: `{factorial['status']}`",
        "",
        "## Paired-row gate",
        "",
        "The existing issue #3501 report builder requires normalized `metric_values` "
        "for every `(planner, scenario_id, seed, wrapper_arm)` row. The camera-ready "
        "episode records do not contain that object. Similar fields are listed in "
        "`summary.json` as source presence only; this audit does not reinterpret them.",
        "",
        "Blocked required metrics:",
        "",
    ]
    lines.extend(f"- `{metric}`" for metric in factorial["blocked_metrics"])
    lines.extend(
        [
            "",
            "This is an artifact-contract stop, not a failed campaign run. The standard "
            "camera-ready campaign artifacts remain separate from any dissertation claim.",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit(audit: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write the compact audit JSON and README."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.json"
    readme = out / "README.md"
    summary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme.write_text(_render_readme(audit), encoding="utf-8")
    return {"summary": summary, "readme": readme}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--config-sha256", default=None)
    parser.add_argument("--artifact-prefix", default=None)
    parser.add_argument("--source-location", default=None)
    return parser.parse_args()


def main() -> int:
    """Audit and write a campaign contract report."""
    args = parse_args()
    audit = audit_campaign(
        args.campaign_root,
        config_path=args.config_path,
        config_sha256=args.config_sha256,
        artifact_prefix=args.artifact_prefix,
        source_location=args.source_location,
    )
    paths = write_audit(audit, args.output_dir)
    print(
        f"issue_4830_safety_wrapper_contract_audit standard={audit['standard_campaign']['status']} "
        f"factorial={audit['factorial_contract']['status']} summary={paths['summary']}"
    )
    return 0 if audit["standard_campaign"]["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
