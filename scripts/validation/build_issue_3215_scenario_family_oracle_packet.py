#!/usr/bin/env python3
"""Build the issue #3215 scenario-family/oracle-arm packet evidence.

The script is intentionally CPU-only. It validates the tracked launch-packet
YAML and emits a compact JSON/Markdown packet that reviewers can inspect before
any paired-seed campaign or Slurm/GPU work exists.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET = Path("configs/benchmarks/predictive_scenario_family_oracle_arm_issue_3215.yaml")
DEFAULT_OUTPUT_JSON = Path(
    "docs/context/evidence/issue_3215_scenario_family_oracle_packet_2026-07-05/packet.json"
)
DEFAULT_OUTPUT_MD = Path(
    "docs/context/evidence/issue_3215_scenario_family_oracle_packet_2026-07-05/README.md"
)
REQUIRED_FORECAST_ARMS = {"none", "constant_velocity", "interaction_aware", "oracle_future"}
REQUIRED_OUTCOMES = {
    "collision_rate",
    "near_miss_rate",
    "false_positive_stop_rate",
    "progress_loss",
    "stop_timing",
    "forecast_risk_calibration",
}


@dataclass(frozen=True)
class Check:
    """One packet validation check."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible check record."""
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def _git_head() -> str:
    """Return current Git head SHA when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        # git binary missing/unavailable or the call timed out: this SHA is
        # best-effort provenance, so degrade to "unknown" rather than crashing.
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _load_packet(path: Path) -> dict[str, Any]:
    """Load a YAML packet mapping."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected YAML mapping at top level")
    return payload


def _path_exists(payload: dict[str, Any], dotted_key: str) -> Check:
    """Check that a repo-relative path field exists."""
    value: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return Check(dotted_key, False, "missing field")
        value = value[part]
    if not isinstance(value, str) or not value:
        return Check(dotted_key, False, "field is not a non-empty string")
    # Fail closed if the tracked path escapes the repo root (defensive: the
    # value comes from a versioned packet config, but resolve+containment keeps
    # a stray "../" or symlink from silently validating an out-of-tree path).
    resolved = (REPO_ROOT / value).resolve(strict=False)
    if not resolved.is_relative_to(REPO_ROOT):
        return Check(dotted_key, False, f"path escapes repository root: {value}")
    exists = resolved.exists()
    return Check(dotted_key, exists, value if exists else f"missing path: {value}")


def validate_packet(payload: dict[str, Any]) -> list[Check]:
    """Validate launch-packet contract without running a campaign."""
    scenario_family = payload.get("scenario_family") or {}
    paired_seeds = payload.get("paired_seeds") or {}
    forecast_arms = payload.get("forecast_arms") or []
    outcomes = payload.get("outcomes") or {}

    arm_ids = {arm.get("id") for arm in forecast_arms if isinstance(arm, dict)}
    outcome_ids = set(outcomes.get("primary") or []) | set(outcomes.get("secondary") or [])
    oracle_arm = next(
        (
            arm
            for arm in forecast_arms
            if isinstance(arm, dict) and arm.get("id") == "oracle_future"
        ),
        {},
    )
    factor_sizes = {
        "occlusion_geometries": len(scenario_family.get("occlusion_geometries") or []),
        "approach_angles_deg": len(scenario_family.get("approach_angles_deg") or []),
        "actor_speeds_mps": len(scenario_family.get("actor_speeds_mps") or []),
        "time_to_arrival_offsets_s": len(scenario_family.get("time_to_arrival_offsets_s") or []),
        "sensor_latency_ms": len(scenario_family.get("sensor_latency_ms") or []),
        "uncertainty_levels": len(scenario_family.get("uncertainty_levels") or []),
    }
    scenario_row_count = 1
    for size in factor_sizes.values():
        scenario_row_count *= max(size, 0)

    checks = [
        Check(
            "schema_version",
            payload.get("schema_version") == "predictive_scenario_family_oracle_packet.v1",
            str(payload.get("schema_version")),
        ),
        Check("issue", payload.get("issue") == 3215, str(payload.get("issue"))),
        Check(
            "evidence_tier",
            payload.get("evidence_tier") == "diagnostic-only",
            str(payload.get("evidence_tier")),
        ),
        Check(
            "forecast_arms",
            REQUIRED_FORECAST_ARMS.issubset(arm_ids),
            f"found={sorted(str(arm) for arm in arm_ids)}",
        ),
        Check(
            "oracle_boundary",
            bool(oracle_arm.get("oracle_state_allowed"))
            and "diagnostic" in str(oracle_arm.get("interpretation_boundary", "")).lower(),
            str(oracle_arm.get("interpretation_boundary", "")),
        ),
        Check(
            "outcomes",
            REQUIRED_OUTCOMES.issubset(outcome_ids),
            f"found={sorted(outcome_ids)}",
        ),
        Check(
            "paired_seed_count",
            isinstance(paired_seeds.get("count"), int) and paired_seeds["count"] >= 30,
            str(paired_seeds.get("count")),
        ),
        Check(
            "scenario_family_factorization",
            all(size >= 2 for size in factor_sizes.values()) and scenario_row_count > 0,
            f"factor_sizes={factor_sizes}, rows_per_seed={scenario_row_count}",
        ),
        Check(
            "claim_boundary",
            "does not run paired seeds" in str(payload.get("claim_boundary", "")).lower()
            and (
                "paper/dissertation" in json.dumps(payload.get("out_of_scope", [])).lower()
                or "paper or dissertation" in json.dumps(payload.get("out_of_scope", [])).lower()
            ),
            str(payload.get("claim_boundary", "")),
        ),
        _path_exists(payload, "scenario_family.base_scenario_set"),
        _path_exists(payload, "source_context.scenario_seed_manifest"),
        _path_exists(payload, "source_context.forecast_risk_gate"),
        _path_exists(payload, "source_context.prerequisite_synthesis"),
    ]
    return checks


def build_manifest(packet_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Build the compact evidence manifest."""
    checks = validate_packet(payload)
    status = "ready" if all(check.passed for check in checks) else "blocked"
    scenario_family = payload["scenario_family"]
    paired_seeds = payload["paired_seeds"]
    factor_sizes = {
        key: len(scenario_family.get(key) or [])
        for key in (
            "occlusion_geometries",
            "approach_angles_deg",
            "actor_speeds_mps",
            "time_to_arrival_offsets_s",
            "sensor_latency_ms",
            "uncertainty_levels",
        )
    }
    rows_per_seed = 1
    for size in factor_sizes.values():
        rows_per_seed *= size

    return {
        "schema_version": "issue_3215_scenario_family_oracle_packet_evidence.v1",
        "issue": 3215,
        "status": status,
        "packet": packet_path.as_posix(),
        "git_head": _git_head(),
        "evidence_tier": payload["evidence_tier"],
        "claim_boundary": payload["claim_boundary"],
        "scenario_family": {
            "name": scenario_family["name"],
            "base_scenario_set": scenario_family["base_scenario_set"],
            "factor_sizes": factor_sizes,
            "rows_per_seed": rows_per_seed,
            "paired_seed_count": paired_seeds["count"],
            "planned_arm_count": len(payload["forecast_arms"]),
            "planned_paired_rows": rows_per_seed
            * paired_seeds["count"]
            * len(payload["forecast_arms"]),
        },
        "forecast_arms": payload["forecast_arms"],
        "outcomes": payload["outcomes"],
        "stop_rules": payload["stop_rules"],
        "out_of_scope": payload["out_of_scope"],
        "checks": [check.to_dict() for check in checks],
        "blocking_issues": [check.detail for check in checks if not check.passed],
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    """Render a reviewable Markdown packet summary."""
    scenario = manifest["scenario_family"]
    lines = [
        "# Issue #3215 Scenario-Family Oracle Packet",
        "",
        "This is a CPU-generated launch packet for the next diagnostic step after the hard-case portfolio synthesis. It does not run paired seeds or establish benchmark evidence.",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Evidence tier: `{manifest['evidence_tier']}`",
        f"- Packet: `{manifest['packet']}`",
        f"- Git head: `{manifest['git_head']}`",
        f"- Scenario family: `{scenario['name']}`",
        f"- Rows per seed: `{scenario['rows_per_seed']}`",
        f"- Paired seeds: `{scenario['paired_seed_count']}`",
        f"- Forecast arms: `{scenario['planned_arm_count']}`",
        f"- Planned paired rows: `{scenario['planned_paired_rows']}`",
        "",
        "## Forecast Arms",
        "",
        "| Arm | Role | Oracle state |",
        "| --- | --- | --- |",
    ]
    for arm in manifest["forecast_arms"]:
        lines.append(
            f"| `{arm['id']}` | {arm['role']} | `{str(arm['oracle_state_allowed']).lower()}` |"
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| Check | Status | Detail |",
            "| --- | --- | --- |",
        ]
    )
    for check in manifest["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        detail = str(check["detail"]).replace("\n", " ")
        lines.append(f"| `{check['name']}` | {status} | {detail} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            manifest["claim_boundary"],
            "",
            "Out of scope: " + ", ".join(manifest["out_of_scope"]) + ".",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--json", action="store_true", help="Print manifest JSON to stdout.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    packet_path = args.packet if args.packet.is_absolute() else REPO_ROOT / args.packet
    payload = _load_packet(packet_path)
    manifest = build_manifest(args.packet, payload)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(manifest), encoding="utf-8")

    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"issue #3215 scenario-family oracle packet: {manifest['status']}")
        print(f"json: {args.output_json}")
        print(f"markdown: {args.output_md}")
    return 0 if manifest["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
