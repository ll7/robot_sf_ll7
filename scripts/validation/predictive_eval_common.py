"""Shared helpers for predictive planner validation/evaluation scripts."""

from __future__ import annotations

from pathlib import Path

import yaml

from robot_sf.training.scenario_loader import load_scenarios


def load_seed_manifest(path: Path) -> dict[str, list[int]]:
    """Load and validate scenario->seed map from YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Seed manifest must be a mapping: {path}")
    out: dict[str, list[int]] = {}
    for key, value in payload.items():
        scenario_id = str(key).strip()
        if not scenario_id:
            raise ValueError(f"Seed manifest contains an empty scenario id: {path}")
        if not isinstance(value, list):
            raise TypeError(f"Seed manifest entry must be a seed list: {scenario_id}")
        if not value:
            raise ValueError(f"Seed manifest entry has no seeds: {scenario_id}")

        seeds = [int(v) for v in value]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Seed manifest entry contains duplicate seeds: {scenario_id}")
        out[scenario_id] = seeds
    return out


def _scenario_id(scenario: dict) -> str:
    """Return the stable scenario identifier used by seed manifests."""
    return str(
        scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
    )


def make_subset_scenarios(scenario_matrix: Path, seed_manifest: dict[str, list[int]]) -> list[dict]:
    """Load scenarios and apply explicit seed lists for selected entries."""
    scenarios = load_scenarios(scenario_matrix)
    selected: list[dict] = []
    matched_ids: set[str] = set()
    base_dir = scenario_matrix.parent.resolve()
    for scenario in scenarios:
        scenario_id = _scenario_id(scenario)
        if scenario_id not in seed_manifest:
            continue
        matched_ids.add(scenario_id)
        scenario_copy = dict(scenario)
        map_file = scenario_copy.get("map_file")
        if isinstance(map_file, str):
            map_path = Path(map_file)
            if not map_path.is_absolute():
                scenario_copy["map_file"] = str((base_dir / map_path).resolve())
        scenario_copy["seeds"] = list(seed_manifest[scenario_id])
        selected.append(scenario_copy)
    missing_ids = sorted(set(seed_manifest) - matched_ids)
    if missing_ids:
        available_ids = sorted(_scenario_id(scenario) for scenario in scenarios)
        raise ValueError(
            f"Seed manifest references scenario ids not present in {scenario_matrix}: "
            f"{', '.join(missing_ids)}. Available scenario ids: {', '.join(available_ids)}"
        )
    if not selected:
        raise ValueError(f"Seed manifest selected no scenarios from {scenario_matrix}")
    return selected
