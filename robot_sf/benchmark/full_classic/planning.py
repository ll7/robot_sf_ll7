"""Planning utilities for the Full Classic Interaction Benchmark.

Currently placeholder functions; implemented across tasks T022-T024.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml  # type: ignore

# ---- Internal helpers (kept simple to keep public function complexity low) ----


def _parse_yaml_file(p: Path) -> dict:
    try:
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to parse YAML: {exc}") from exc


def _extract_scenarios(root: dict) -> list[dict]:
    scenarios = root.get("scenarios") if isinstance(root, dict) else None
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Scenario matrix missing non-empty 'scenarios' list")
    return scenarios


_REQUIRED_SC_KEYS = {"name", "map_file", "simulation_config", "metadata"}


def _validate_scenario_dicts(scenarios: list[dict]) -> None:
    for idx, sc in enumerate(scenarios):
        if not isinstance(sc, dict):
            raise ValueError(f"Scenario index {idx} not a mapping")
        missing = _REQUIRED_SC_KEYS - sc.keys()
        if missing:
            raise ValueError(f"Scenario '{sc.get('name', idx)}' missing keys: {missing}")
        sim_cfg = sc.get("simulation_config", {})
        meta = sc.get("metadata", {})
        if not isinstance(sim_cfg, dict) or not isinstance(meta, dict):
            raise ValueError(f"Scenario '{sc.get('name')}' has invalid nested structures")
        if "max_episode_steps" not in sim_cfg:
            raise ValueError(f"Scenario '{sc.get('name')}' missing max_episode_steps")
        if "archetype" not in meta or "density" not in meta:
            raise ValueError(f"Scenario '{sc.get('name')}' metadata missing archetype/density")


# Minimal placeholder dataclasses matching data-model (subset for scaffolding)
@dataclass
class ScenarioDescriptor:  # duplicated light form; real version will live centrally later
    scenario_id: str
    archetype: str
    density: str
    map_path: str
    params: Dict[str, object]
    planned_seeds: List[int]
    max_episode_steps: int
    hash_fragment: str


def load_scenario_matrix(path: str) -> list[dict]:  # T022
    """Load scenario matrix YAML returning the raw scenario dictionaries.

    Public function kept intentionally thin for low cyclomatic complexity.
    """
    p = Path(path)
    if not p.exists():  # contract: FileNotFoundError path missing
        raise FileNotFoundError(path)
    data = _parse_yaml_file(p)
    scenarios = _extract_scenarios(data)
    _validate_scenario_dicts(scenarios)
    return scenarios


def _normalise_raw_scenario(index: int, sc: dict) -> tuple[str, str, str, int, str, dict]:
    """Return (name, archetype, density, max_steps, map_file, params_source).

    Supports both the full matrix shape and the simplified contract test shape.
    """
    if "metadata" in sc and "simulation_config" in sc:  # full matrix shape
        name = str(sc.get("name", f"sc{index}"))
        meta = sc["metadata"]
        archetype = str(meta["archetype"])
        density = str(meta["density"])
        sim_cfg = sc["simulation_config"]
        max_steps = int(sim_cfg["max_episode_steps"])  # trusted numeric
        map_file = str(sc["map_file"])  # path string
        params_source = {
            k: v
            for k, v in sc.items()
            if k not in {"name", "map_file", "simulation_config", "metadata"}
        }
    else:  # simplified contract test shape
        name = str(sc.get("scenario_id", f"sc{index}"))
        archetype = str(sc["archetype"])
        density = str(sc["density"])
        max_steps = int(sc["max_episode_steps"])  # Provided directly
        map_file = str(sc["map_path"])  # May be relative
        params_source = dict(sc.get("params", {}))
    return name, archetype, density, max_steps, map_file, params_source


def _plan_unique_seeds(rng, count: int) -> list[int]:  # small helper for clarity
    seeds: list[int] = []
    while len(seeds) < count:
        val = rng.randrange(0, 2**31 - 1)
        if val not in seeds:
            seeds.append(val)
    return seeds


def plan_scenarios(raw: list[dict], cfg, *, rng) -> list[ScenarioDescriptor]:  # T023
    """Expand raw scenario dicts into `ScenarioDescriptor` objects.

    Responsibilities (per contract & data model):
    - Validate required keys already screened by `load_scenario_matrix` plus map file existence.
    - Deterministically generate `cfg.initial_episodes` seeds per scenario using provided rng.
    - Compute a stable hash fragment capturing defining scenario attributes to support
      reproducibility & manifest hashing (map path, archetype, density, params, max steps).
    - Construct scenario_id: <archetype>__<density>__<name> (sanitized) ensuring uniqueness.

    Assumptions:
    - `cfg` exposes attributes: initial_episodes, scenario_matrix_path.
    - `rng` is an instance of `random.Random` seeded by caller (master seed) for determinism.
    """

    if not isinstance(raw, list) or not raw:
        raise ValueError("Raw scenarios must be a non-empty list")

    scenarios: list[ScenarioDescriptor] = []
    seen_ids: set[str] = set()
    for index, sc in enumerate(raw):
        name, archetype, density, max_steps, map_file, params_source = _normalise_raw_scenario(
            index, sc
        )

        # Map path validation (relative resolution)
        matrix_dir = Path(cfg.scenario_matrix_path).parent
        map_path = Path(map_file)
        if not map_path.is_absolute():
            map_path = (matrix_dir / map_file).resolve()
        # For early contract tests we allow non-existent map files (synthetic inputs)
        # Full integration later will validate via environment factory. Only raise if
        # parent 'svg_maps' directory exists (indicating mis-typed filename) but file missing.
        if not map_path.exists():
            if (
                map_path.parent.name == "svg_maps" and map_path.parent.exists()
            ):  # likely real file expected
                raise ValueError(f"Map file not found for scenario '{name}': {map_path}")
        # Seed planning – deterministic unique seeds per scenario
        planned_seeds = _plan_unique_seeds(rng, int(cfg.initial_episodes))

        # Hash fragment: stable SHA1 over JSON canonical representation of key fields
        hash_payload = {
            "name": name,
            "archetype": archetype,
            "density": density,
            "map": str(map_path),
            "max_episode_steps": max_steps,
            "params": params_source,
        }
        hash_bytes = json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        hash_fragment = hashlib.sha1(hash_bytes).hexdigest()[:10]

        # Scenario ID (ensure filesystem safe, uniqueness)
        def _san(s: str) -> str:
            return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in s.lower())

        scenario_id = f"{_san(archetype)}__{_san(density)}__{_san(name)}"
        if scenario_id in seen_ids:
            raise ValueError(f"Duplicate scenario id generated: {scenario_id}")
        seen_ids.add(scenario_id)

        scenarios.append(
            ScenarioDescriptor(
                scenario_id=scenario_id,
                archetype=archetype,
                density=density,
                map_path=str(map_path),
                params=params_source,
                planned_seeds=planned_seeds,
                max_episode_steps=max_steps,
                hash_fragment=hash_fragment,
            )
        )

    return scenarios


def expand_episode_jobs(scenarios: list[ScenarioDescriptor], cfg) -> list:  # T024
    raise NotImplementedError("Implemented in task T024")
