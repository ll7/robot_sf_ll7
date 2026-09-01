"""Emit deterministic implementation-integrity smoke for issue #8068."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from robot_sf.prediction._contract_utils import stable_digest
from robot_sf.prediction.goal_intention import (
    GoalCandidate,
    GoalCandidateRole,
    GoalCandidateSet,
    HeadingGoalPosteriorConfig,
    update_heading_goal_posterior,
)

CLAIM_BOUNDARY = (
    "implementation-integrity smoke: H=1 observation-only heading posterior over public "
    "candidates; no calibrated intention, planner-performance, or benchmark claim"
)
SCHEMA_VERSION = "issue_8068_goal_posterior_actor_smoke.v1"


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("smoke config must be a YAML mapping")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"config schema_version must be {SCHEMA_VERSION}")
    return payload


def _rotate_vector(vector: tuple[float, float], angle: float) -> tuple[float, float]:
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return (
        cos_angle * vector[0] - sin_angle * vector[1],
        sin_angle * vector[0] + cos_angle * vector[1],
    )


def _rotate_point(
    point: tuple[float, float], angle: float, offset: tuple[float, float]
) -> tuple[float, float]:
    rotated = _rotate_vector(point, angle)
    return (rotated[0] + offset[0], rotated[1] + offset[1])


def _xy(value: Any, field_name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two values")
    values = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{field_name} must contain finite values")
    return values


def _candidate_set(
    raw_candidates: Any,
    *,
    scenario_id: str,
    position_transform: tuple[float, tuple[float, float]] | None = None,
) -> GoalCandidateSet:
    if not isinstance(raw_candidates, list):
        raise ValueError(f"{scenario_id}.candidates must be a list")
    angle, offset = position_transform or (0.0, (0.0, 0.0))
    candidates: list[GoalCandidate] = []
    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, dict):
            raise ValueError(f"{scenario_id}.candidates[{index}] must be a mapping")
        candidate_id = raw_candidate.get("id")
        position = _xy(raw_candidate["position"], f"{scenario_id}.candidates[{index}].position")
        if position_transform is not None:
            position = _rotate_point(position, angle, offset)
        role = GoalCandidateRole(raw_candidate.get("role", GoalCandidateRole.FINAL_DESTINATION))
        candidates.append(
            GoalCandidate(
                id=candidate_id,
                position=position,
                source="public_smoke_fixture",
                role=role,
            )
        )
    return GoalCandidateSet(
        candidates=tuple(candidates),
        source="public_smoke_fixture",
    )


def _run_case(
    scenario: dict[str, Any],
    *,
    config: HeadingGoalPosteriorConfig,
    rotated: bool = False,
) -> dict[str, Any]:
    scenario_id = str(scenario.get("id", ""))
    transform: tuple[float, tuple[float, float]] | None = None
    if rotated:
        raw_transform = scenario.get("rotate")
        if not isinstance(raw_transform, dict):
            raise ValueError(f"{scenario_id}.rotate must be present for rotated cases")
        transform = (
            float(raw_transform["angle_rad"]),
            _xy(raw_transform["offset"], f"{scenario_id}.rotate.offset"),
        )
    position = _xy(scenario["position"], f"{scenario_id}.position")
    velocity = _xy(scenario["velocity"], f"{scenario_id}.velocity")
    if transform is not None:
        position = _rotate_point(position, transform[0], transform[1])
        velocity = _rotate_vector(velocity, transform[0])
    candidate_set = _candidate_set(
        scenario["candidates"],
        scenario_id=scenario_id,
        position_transform=transform,
    )
    prior = scenario.get("prior")
    if prior is not None and not isinstance(prior, dict):
        raise ValueError(f"{scenario_id}.prior must be a mapping")
    belief = update_heading_goal_posterior(
        track_id=scenario_id,
        observed_position_global=position,
        observed_velocity_global=velocity,
        candidate_set=candidate_set,
        prior=prior,
        config=config,
    )
    return {
        "case_id": f"{scenario_id}{'_rotated' if rotated else ''}",
        "candidate_set": candidate_set.to_dict(),
        "candidate_set_digest": stable_digest(candidate_set.to_dict()),
        "candidate_ids": [candidate.id for candidate in candidate_set.candidates],
        "probabilities": {
            candidate.candidate_id: candidate.probability
            for candidate in belief.candidate_probabilities
        },
        "entropy": belief.entropy,
        "unknown_candidate_probability": belief.unknown_candidate_probability,
        "blockers": list(belief.blockers),
        "source": belief.source.value,
        "mode": belief.mode.value,
        "content_digest": belief.content_digest,
    }


def build_report(config_path: Path) -> dict[str, Any]:
    """Build a compact deterministic actor-only smoke report."""

    payload = _load_config(config_path)
    raw_config = payload.get("posterior_config", {})
    if not isinstance(raw_config, dict):
        raise ValueError("posterior_config must be a mapping")
    config = HeadingGoalPosteriorConfig(**raw_config)
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenarios must be a non-empty list")
    reports: list[dict[str, Any]] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("each scenario must be a mapping")
        reports.append(_run_case(scenario, config=config))
        reports.append(_run_case(scenario, config=config, rotated=True))
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "posterior_config": {**config.to_dict(), "config_hash": config.config_hash},
        "source_contract": "observation_only",
        "oracle_identity_input_present": False,
        "scenarios": reports,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the issue #8068 actor-posterior smoke CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/issue_8068_goal_posterior_actor_smoke.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/benchmarks/issue_8068_goal_posterior_actor_smoke.json"),
    )
    args = parser.parse_args(argv)
    report = build_report(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
