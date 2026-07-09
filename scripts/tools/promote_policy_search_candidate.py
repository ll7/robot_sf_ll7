#!/usr/bin/env python3
"""Evaluate a candidate summary against the configured promotion gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.identity.hash_utils import load_json as _load_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", type=Path)
    parser.add_argument(
        "--candidate-registry",
        type=Path,
        default=Path("docs/context/policy_search/candidate_registry.yaml"),
    )
    parser.add_argument(
        "--promotion-gates",
        type=Path,
        default=Path("configs/policy_search/promotion_gates.yaml"),
    )
    parser.add_argument(
        "--gate-name",
        help=(
            "Override the candidate registry promotion gate. Use this for stricter "
            "diagnostic checks such as nominal_sanity on full_matrix_h500 summaries."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("output/policy_search/promotion"))
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk.

    Returns:
        dict[str, Any]: Parsed YAML mapping.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def main() -> int:
    """Evaluate one candidate summary against configured promotion gates."""
    args = parse_args()
    payload = _load_json(args.summary_json)
    registry = _load_yaml(args.candidate_registry)
    gates = _load_yaml(args.promotion_gates)

    candidate_name = str(payload.get("candidate", "unknown"))
    candidates = registry.get("candidates") if isinstance(registry.get("candidates"), dict) else {}
    candidate_registered = candidate_name in candidates if isinstance(candidates, dict) else False
    candidate_meta = candidates.get(candidate_name, {}) if isinstance(candidates, dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    family = (
        summary.get("scenario_family") if isinstance(summary.get("scenario_family"), dict) else {}
    )
    classic = family.get("classic", {}) if isinstance(family.get("classic"), dict) else {}
    francis = family.get("francis2023", {}) if isinstance(family.get("francis2023"), dict) else {}

    gate_name = str(args.gate_name or candidate_meta.get("promotion_gate", "tier_b"))
    gate_map = gates.get("gates") if isinstance(gates.get("gates"), dict) else {}
    gate_cfg = gate_map.get(gate_name, {}) if isinstance(gate_map, dict) else {}
    gate_configured = bool(gate_cfg)
    stratified = gate_map.get("scenario_stratified", {}) if isinstance(gate_map, dict) else {}

    stage = str(payload.get("stage", "unknown"))
    runner_decision = str(payload.get("decision", "")).strip().lower()
    promotion_scale_stage = stage in {"full_matrix", "full_matrix_h500", "robustness_extension"}
    stage_decision_passed = promotion_scale_stage or runner_decision == "pass"

    checks = {
        "candidate_registered": candidate_registered,
        "gate_configured": gate_configured,
        "stage_decision_passed": stage_decision_passed,
        "success_rate": float(summary.get("success_rate", 0.0))
        >= float(gate_cfg.get("min_success_rate", 0.0)),
        "collision_rate": float(summary.get("collision_rate", 1.0))
        <= float(gate_cfg.get("max_collision_rate", 1.0)),
        "classic_collision_rate": float(classic.get("collision_rate", 1.0))
        <= float(stratified.get("classic_collision_rate_max", 1.0)),
        "francis_collision_rate": float(francis.get("collision_rate", 1.0))
        <= float(stratified.get("francis_collision_rate_max", 1.0)),
    }
    decision = "promote" if all(checks.values()) else "revise"

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{candidate_name}_{payload.get('stage', 'unknown')}_promotion.json"
    json_path.write_text(
        json.dumps(
            {
                "candidate": candidate_name,
                "stage": stage,
                "runner_decision": runner_decision or None,
                "gate": gate_name,
                "checks": checks,
                "decision": decision,
                "summary_json": str(args.summary_json),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        f"# Promotion Decision: {candidate_name}",
        "",
        f"- Stage: `{stage}`",
        f"- Runner decision: `{runner_decision or 'n/a'}`",
        f"- Gate: `{gate_name}`",
        f"- Decision: `{decision}`",
        "",
        "| Check | Passed |",
        "|---|---|",
    ]
    for key, value in checks.items():
        lines.append(f"| {key} | {'yes' if value else 'no'} |")
    md_path = output_dir / f"{candidate_name}_{payload.get('stage', 'unknown')}_promotion.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
