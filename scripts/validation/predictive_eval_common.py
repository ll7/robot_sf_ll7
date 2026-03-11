"""Shared helpers for predictive planner validation/evaluation scripts."""

from __future__ import annotations

from pathlib import Path

import yaml

from robot_sf.training.scenario_loader import load_scenarios


def load_seed_manifest(path: Path) -> dict[str, list[int]]:
    """Load scenario->seed map from YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Seed manifest must be a mapping: {path}")
    out: dict[str, list[int]] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            out[str(key)] = [int(v) for v in value]
    return out


def make_subset_scenarios(scenario_matrix: Path, seed_manifest: dict[str, list[int]]) -> list[dict]:
    """Load scenarios and apply explicit seed lists for selected entries."""
    scenarios = load_scenarios(scenario_matrix)
    selected: list[dict] = []
    base_dir = scenario_matrix.parent.resolve()
    for scenario in scenarios:
        scenario_id = str(
            scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
        )
        if scenario_id not in seed_manifest:
            continue
        scenario_copy = dict(scenario)
        map_file = scenario_copy.get("map_file")
        if isinstance(map_file, str):
            map_path = Path(map_file)
            if not map_path.is_absolute():
                scenario_copy["map_file"] = str((base_dir / map_path).resolve())
        scenario_copy["seeds"] = list(seed_manifest[scenario_id])
        selected.append(scenario_copy)
    return selected
