"""Counterexample bundle writing for adversarial search."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.adversarial.config import CandidateEvaluation, CandidateSpec, SearchConfig

MANIFEST_SCHEMA_VERSION = "adversarial-search-manifest.v1"


def _load_template(path: Path) -> dict[str, Any]:
    """Load a scenario-template YAML file."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Scenario template must be a mapping: {path}")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"Scenario template must contain a non-empty scenarios list: {path}")
    if not isinstance(scenarios[0], dict):
        raise ValueError("Scenario template first scenario must be a mapping")
    return payload


def _candidate_route_payload(candidate: CandidateSpec, *, index: int) -> dict[str, Any]:
    """Build a route-overrides payload for the candidate start and goal."""
    route_id = 100_000 + int(index)
    return {
        "robot_routes": [
            {
                "spawn_id": route_id,
                "goal_id": route_id,
                "waypoints": [candidate.start.as_waypoint(), candidate.goal.as_waypoint()],
            }
        ],
        "ped_routes": [],
    }


def _apply_candidate_to_scenario(
    scenario: dict[str, Any],
    candidate: CandidateSpec,
    *,
    index: int,
    route_file_name: str,
    pedestrian_id: str | None,
) -> dict[str, Any]:
    """Return a scenario dictionary specialized for one candidate."""
    updated = deepcopy(scenario)
    base_name = str(updated.get("name") or updated.get("scenario_id") or "scenario")
    updated["name"] = f"{base_name}_adversarial_{index:04d}"
    updated["route_overrides_file"] = route_file_name
    updated["seeds"] = [int(candidate.scenario_seed)]
    sim_config = dict(updated.get("simulation_config") or {})
    sim_config["route_spawn_seed"] = int(candidate.scenario_seed)
    sim_config["peds_speed_mult"] = float(candidate.pedestrian_speed_mps)
    updated["simulation_config"] = sim_config
    metadata = dict(updated.get("metadata") or {})
    metadata["adversarial_candidate"] = {
        **candidate.to_json(),
        "candidate_index": int(index),
        "spawn_time_s_note": "stored for search provenance; route-spawn timing support is adapter-dependent",
    }
    updated["metadata"] = metadata

    if pedestrian_id:
        entries = list(updated.get("single_pedestrians") or [])
        replacement = {
            "id": pedestrian_id,
            "speed_m_s": float(candidate.pedestrian_speed_mps),
            "wait_at": [{"waypoint_index": 0, "wait_s": float(candidate.pedestrian_delay_s)}],
        }
        for entry_index, entry in enumerate(entries):
            if isinstance(entry, dict) and entry.get("id") == pedestrian_id:
                merged = dict(entry)
                merged.update(replacement)
                entries[entry_index] = merged
                break
        else:
            entries.append(replacement)
        updated["single_pedestrians"] = entries
    return updated


def write_candidate_inputs(
    *,
    config: SearchConfig,
    candidate: CandidateSpec,
    candidate_dir: Path,
    index: int,
) -> tuple[Path, Path]:
    """Write replayable scenario and route-override files for a candidate."""
    candidate_dir.mkdir(parents=True, exist_ok=True)
    route_path = candidate_dir / "route_overrides.yaml"
    scenario_path = candidate_dir / "scenario.yaml"
    route_payload = _candidate_route_payload(candidate, index=index)
    route_path.write_text(yaml.safe_dump(route_payload, sort_keys=False), encoding="utf-8")

    template = _load_template(config.scenario_template)
    scenario = _apply_candidate_to_scenario(
        dict(template["scenarios"][0]),
        candidate,
        index=index,
        route_file_name=route_path.name,
        pedestrian_id=config.search_space.pedestrian_id,
    )
    scenario_payload = {"scenarios": [scenario]}
    scenario_path.write_text(yaml.safe_dump(scenario_payload, sort_keys=False), encoding="utf-8")
    return scenario_path, route_path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def write_trajectory_csv(path: Path, record: dict[str, Any] | None) -> Path:
    """Write a small CSV replay index for the evaluated candidate.

    The current benchmark episode record does not expose full per-step
    trajectories. This CSV therefore records replay-identifying fields and can
    be replaced by a denser trajectory export once the runner exposes one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if record is None:
        path.write_text("episode_id,seed,status,steps,termination_reason\n", encoding="utf-8")
        return path
    row = [
        str(record.get("episode_id", "")),
        str(record.get("seed", "")),
        str(record.get("status", "")),
        str(record.get("steps", "")),
        str(record.get("termination_reason", "")),
    ]
    path.write_text(
        "episode_id,seed,status,steps,termination_reason\n" + ",".join(row) + "\n",
        encoding="utf-8",
    )
    return path


def write_search_manifest(
    *,
    config: SearchConfig,
    manifest_path: Path,
    evaluations: list[CandidateEvaluation],
    best: CandidateEvaluation | None,
    num_invalid_candidates: int,
    num_failed_evaluations: int,
) -> Path:
    """Write the top-level search manifest."""
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "config": config.to_json(),
        "summary": {
            "num_candidates": len(evaluations),
            "num_valid_candidates": len(evaluations) - num_invalid_candidates,
            "num_invalid_candidates": num_invalid_candidates,
            "num_failed_evaluations": num_failed_evaluations,
            "best_objective_value": best.objective_value if best else None,
            "best_bundle_path": best.bundle_path.as_posix() if best and best.bundle_path else None,
        },
        "candidates": [_evaluation_to_json(item) for item in evaluations],
    }
    return write_json(manifest_path, payload)


def _evaluation_to_json(evaluation: CandidateEvaluation) -> dict[str, Any]:
    """Convert an evaluation dataclass into a manifest payload."""
    attribution = (
        evaluation.failure_attribution.to_json() if evaluation.failure_attribution else None
    )
    return {
        "candidate": evaluation.candidate.to_json(),
        "certification_status": evaluation.certification_status.to_json(),
        "objective_value": evaluation.objective_value,
        "failure_attribution": attribution,
        "episode_record_path": evaluation.episode_record_path.as_posix()
        if evaluation.episode_record_path
        else None,
        "trajectory_csv_path": evaluation.trajectory_csv_path.as_posix()
        if evaluation.trajectory_csv_path
        else None,
        "scenario_yaml_path": evaluation.scenario_yaml_path.as_posix()
        if evaluation.scenario_yaml_path
        else None,
        "bundle_path": evaluation.bundle_path.as_posix() if evaluation.bundle_path else None,
        "error": evaluation.error,
    }


def _json_safe(value: Any) -> Any:
    """Recursively convert common Python objects to JSON-safe values."""
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(nested) for nested in value]
    return value
