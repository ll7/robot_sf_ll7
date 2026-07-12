#!/usr/bin/env python3
"""Validate that two benchmark configs form a valid horizon ablation pair.

A valid horizon ablation pair differs ONLY in campaign-level horizon while
sharing the same planner roster, seed budget, scenario matrix, comparability
mapping, and all other execution-relevant fields.  This is the minimal check
needed to conclude "does horizon change planner conclusions?" from the paired
campaign outputs.

Usage::

    uv run python scripts/benchmark/validate_horizon_ablation_pair.py \
        configs/benchmarks/issue_5409_horizon_ablation_h500.yaml \
        configs/benchmarks/issue_5409_horizon_ablation_h600.yaml

Exit codes:
    0 — configs form a valid horizon ablation pair
    1 — at least one mismatch was found (printed to stderr)
    2 — config loading or parsing error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

_EXECUTION_RELEVANT_KEYS = frozenset(
    {
        "scenario_matrix",
        "comparability_mapping",
        "route_clearance_certifications",
        "scenario_horizons",
        "dt",
        "workers",
        "record_forces",
        "resume",
        "stop_on_failure",
        "bootstrap_samples",
        "bootstrap_confidence",
        "bootstrap_seed",
        "kinematics_matrix",
        "export_publication_bundle",
        "include_videos_in_publication",
        "snqi_weights",
        "snqi_baseline",
        "paper_facing",
        "paper_profile_version",
    }
)


def _load_raw_config(path: Path) -> dict[str, Any]:
    """Load a YAML campaign config as a raw dict without full validation."""
    text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root is not a mapping: {path}")
    return payload


def _planner_roster(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Extract the normalized planner roster from a raw config payload."""
    planners_raw = payload.get("planners")
    if not isinstance(planners_raw, list) or not planners_raw:
        raise ValueError("Config has no 'planners' list")
    roster: list[dict[str, str]] = []
    for entry in planners_raw:
        if not isinstance(entry, dict):
            raise ValueError("Each planner entry must be a mapping")
        key = str(entry.get("key") or entry.get("algo") or "").strip()
        algo = str(entry.get("algo") or "").strip()
        algo_config = str(entry.get("algo_config") or "").strip()
        if not key or not algo:
            raise ValueError(f"Planner entry missing key or algo: {entry}")
        roster.append({"key": key, "algo": algo, "algo_config": algo_config})
    return roster


def _seed_policy_signature(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract a comparable seed-policy signature."""
    raw = payload.get("seed_policy")
    if not isinstance(raw, dict):
        return {}
    return {
        "mode": str(raw.get("mode", "")),
        "seed_set": str(raw.get("seed_set", "")),
        "seeds": list(raw.get("seeds") or []),
        "seed_sets_path": str(raw.get("seed_sets_path", "")),
    }


def _compare_planner_rosters(
    roster_a: list[dict[str, str]],
    roster_b: list[dict[str, str]],
) -> list[str]:
    """Return mismatches between two planner rosters."""
    errors: list[str] = []
    keys_a = [p["key"] for p in roster_a]
    keys_b = [p["key"] for p in roster_b]
    if keys_a != keys_b:
        only_a = [k for k in keys_a if k not in keys_b]
        only_b = [k for k in keys_b if k not in keys_a]
        if only_a:
            errors.append(f"Planners only in config A: {only_a}")
        if only_b:
            errors.append(f"Planners only in config B: {only_b}")
        if not only_a and not only_b and len(keys_a) == len(keys_b):
            errors.append(f"Planner order differs: A={keys_a} vs B={keys_b}")
    for pa, pb in zip(roster_a, roster_b, strict=False):
        if pa["key"] != pb["key"]:
            break
        for field in ("algo", "algo_config"):
            if pa[field] != pb[field]:
                errors.append(
                    f"Planner '{pa['key']}' {field} differs: A={pa[field]!r} vs B={pb[field]!r}"
                )
    return errors


def _check_horizon_differs(
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
) -> tuple[int | None, int | None, list[str]]:
    """Check that horizons are fixed and differ. Return (h_a, h_b, errors)."""
    mismatches: list[str] = []
    horizon_a = payload_a.get("horizon")
    horizon_b = payload_b.get("horizon")
    if horizon_a is None and payload_a.get("scenario_horizons"):
        mismatches.append(
            "Config A uses scenario_horizons (per-scenario), not a fixed horizon; "
            "a clean ablation requires fixed horizons."
        )
    if horizon_b is None and payload_b.get("scenario_horizons"):
        mismatches.append(
            "Config B uses scenario_horizons (per-scenario), not a fixed horizon; "
            "a clean ablation requires fixed horizons."
        )
    if horizon_a is not None and horizon_b is not None and horizon_a == horizon_b:
        mismatches.append(
            f"Horizons are identical ({horizon_a} == {horizon_b}); "
            "an ablation requires different horizons."
        )
    return (
        int(horizon_a) if horizon_a is not None else None,
        int(horizon_b) if horizon_b is not None else None,
        mismatches,
    )


def _check_field_parity(
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
) -> list[str]:
    """Compare all execution-relevant, SNQI, and AMV fields for parity."""
    mismatches: list[str] = []
    for key in sorted(_EXECUTION_RELEVANT_KEYS):
        va = payload_a.get(key)
        vb = payload_b.get(key)
        if va != vb:
            mismatches.append(f"Field '{key}' differs: A={va!r} vs B={vb!r}")

    snqi_a = dict(sorted((payload_a.get("snqi_contract") or {}).items()))
    snqi_b = dict(sorted((payload_b.get("snqi_contract") or {}).items()))
    if snqi_a != snqi_b:
        mismatches.append(f"snqi_contract differs: A={snqi_a} vs B={snqi_b}")

    amv_a = dict(sorted((payload_a.get("amv_profile") or {}).items()))
    amv_b = dict(sorted((payload_b.get("amv_profile") or {}).items()))
    if amv_a != amv_b:
        mismatches.append(f"amv_profile differs: A={amv_a} vs B={amv_b}")

    return mismatches


class HorizonAblationPairResult:
    """Structured result of a horizon ablation pair validation."""

    def __init__(
        self,
        config_a: str,
        config_b: str,
        horizon_a: int | None,
        horizon_b: int | None,
        mismatches: list[str],
    ) -> None:
        """Store validation inputs and computed mismatches."""
        self.config_a = config_a
        self.config_b = config_b
        self.horizon_a = horizon_a
        self.horizon_b = horizon_b
        self.mismatches = mismatches

    @property
    def is_valid(self) -> bool:
        """Return True when the pair is a valid horizon ablation."""
        return (
            not self.mismatches
            and self.horizon_a is not None
            and self.horizon_b is not None
            and self.horizon_a != self.horizon_b
        )

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable validation result."""
        return {
            "schema_version": "horizon_ablation_pair_validation.v1",
            "config_a": self.config_a,
            "config_b": self.config_b,
            "horizon_a": self.horizon_a,
            "horizon_b": self.horizon_b,
            "is_valid": self.is_valid,
            "mismatch_count": len(self.mismatches),
            "mismatches": self.mismatches,
        }


def validate_horizon_ablation_pair(
    path_a: str | Path,
    path_b: str | Path,
) -> HorizonAblationPairResult:
    """Validate that two configs form a valid horizon ablation pair.

    Returns:
        Structured validation result with mismatches list.
    """
    path_a = Path(path_a).resolve()
    path_b = Path(path_b).resolve()

    try:
        payload_a = _load_raw_config(path_a)
        payload_b = _load_raw_config(path_b)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        return HorizonAblationPairResult(
            config_a=str(path_a),
            config_b=str(path_b),
            horizon_a=None,
            horizon_b=None,
            mismatches=[f"Config loading error: {exc}"],
        )

    mismatches: list[str] = []

    horizon_a, horizon_b, horizon_errors = _check_horizon_differs(payload_a, payload_b)
    mismatches.extend(horizon_errors)

    try:
        roster_a = _planner_roster(payload_a)
        roster_b = _planner_roster(payload_b)
        mismatches.extend(_compare_planner_rosters(roster_a, roster_b))
    except ValueError as exc:
        mismatches.append(f"Roster extraction error: {exc}")

    seed_a = _seed_policy_signature(payload_a)
    seed_b = _seed_policy_signature(payload_b)
    if seed_a != seed_b:
        mismatches.append(f"Seed policy differs: A={seed_a} vs B={seed_b}")

    mismatches.extend(_check_field_parity(payload_a, payload_b))

    return HorizonAblationPairResult(
        config_a=str(path_a),
        config_b=str(path_b),
        horizon_a=horizon_a,
        horizon_b=horizon_b,
        mismatches=mismatches,
    )


def main() -> None:
    """CLI entry point for horizon ablation pair validation."""
    parser = argparse.ArgumentParser(
        description="Validate two benchmark configs as a horizon ablation pair.",
    )
    parser.add_argument("config_a", type=str, help="Path to the first config (e.g. h500).")
    parser.add_argument("config_b", type=str, help="Path to the second config (e.g. h600).")
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print the result as JSON to stdout.",
    )
    args = parser.parse_args()

    result = validate_horizon_ablation_pair(args.config_a, args.config_b)

    if args.json_output:
        print(json.dumps(result.to_payload(), indent=2, sort_keys=False))
    elif result.is_valid:
        print(f"VALID horizon ablation pair: h{result.horizon_a} vs h{result.horizon_b}")
    else:
        print("INVALID horizon ablation pair:", file=sys.stderr)
        for mismatch in result.mismatches:
            print(f"  - {mismatch}", file=sys.stderr)

    sys.exit(0 if result.is_valid else 1)


if __name__ == "__main__":
    main()
