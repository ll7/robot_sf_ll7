"""Planning utilities for the Full Classic Interaction Benchmark.

Currently placeholder functions; implemented across tasks T022-T024.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore

# ---- Internal helpers (kept simple to keep public function complexity low) ----


def _parse_yaml_file(p: Path) -> dict:
    """TODO docstring. Document this function.

    Args:
        p: TODO docstring.

    Returns:
        TODO docstring.
    """
    try:
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as exc:
        raise ValueError(f"Failed to parse YAML: {exc}") from exc


def _extract_scenarios(root: dict) -> list[dict]:
    """TODO docstring. Document this function.

    Args:
        root: TODO docstring.

    Returns:
        TODO docstring.
    """
    scenarios = root.get("scenarios") if isinstance(root, dict) else None
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Scenario matrix missing non-empty 'scenarios' list")
    return scenarios


_REQUIRED_SC_KEYS = {"name", "map_file", "simulation_config", "metadata"}


def _validate_scenario_dicts(scenarios: list[dict]) -> None:
    """TODO docstring. Document this function.

    Args:
        scenarios: TODO docstring.
    """
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
    """TODO docstring. Document this class."""

    scenario_id: str
    archetype: str
    density: str
    map_path: str
    params: dict[str, object]
    raw: dict
    planned_seeds: list[int]
    max_episode_steps: int
    hash_fragment: str


@dataclass
class EpisodeJob:  # lightweight form for planning layer
    """TODO docstring. Document this class."""

    job_id: str
    scenario_id: str
    seed: int
    archetype: str
    density: str
    horizon: int
    scenario: ScenarioDescriptor


def load_scenario_matrix(path: str) -> list[dict]:  # T022
    """Load scenario matrix YAML returning the raw scenario dictionaries.

    Public function kept intentionally thin for low cyclomatic complexity.

    Returns:
        List of raw scenario dictionaries loaded from the YAML file.
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

    Returns:
        Tuple of (name, archetype, density, max_steps, map_file, params_source).
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
    """TODO docstring. Document this function.

    Args:
        rng: TODO docstring.
        count: TODO docstring.

    Returns:
        TODO docstring.
    """
    seeds: list[int] = []
    while len(seeds) < count:
        val = rng.randrange(0, 2**31 - 1)
        if val not in seeds:
            seeds.append(val)
    return seeds


def _resolve_map_path(map_file: str, cfg, name: str) -> Path:
    """TODO docstring. Document this function.

    Args:
        map_file: TODO docstring.
        cfg: TODO docstring.
        name: TODO docstring.

    Returns:
        TODO docstring.
    """
    matrix_dir = Path(cfg.scenario_matrix_path).parent
    map_path = Path(map_file)
    if not map_path.is_absolute():
        map_path = (matrix_dir / map_file).resolve()
    if not map_path.exists() and map_path.parent.name == "svg_maps" and map_path.parent.exists():
        raise ValueError(f"Map file not found for scenario '{name}': {map_path}")
    return map_path


def _plan_seeds(sc: dict, cfg, rng) -> list[int]:
    """TODO docstring. Document this function.

    Args:
        sc: TODO docstring.
        cfg: TODO docstring.
        rng: TODO docstring.

    Returns:
        TODO docstring.
    """
    cfg_seeds = getattr(cfg, "seeds", None)
    seeds_from_matrix = sc.get("seeds")
    planned: list[int] = []
    if isinstance(cfg_seeds, list) and cfg_seeds:
        planned = [int(s) for s in cfg_seeds]
    elif isinstance(seeds_from_matrix, list) and seeds_from_matrix:
        planned = [int(s) for s in seeds_from_matrix]
    target = int(getattr(cfg, "initial_episodes", 0) or 0)
    while len(planned) < target:
        next_seed = _plan_unique_seeds(rng, 1)[0]
        if next_seed not in planned:
            planned.append(next_seed)
    return planned


def _scenario_id(archetype: str, density: str, name: str) -> str:
    """TODO docstring. Document this function.

    Args:
        archetype: TODO docstring.
        density: TODO docstring.
        name: TODO docstring.

    Returns:
        TODO docstring.
    """

    def _san(s: str) -> str:
        """TODO docstring. Document this function.

        Args:
            s: TODO docstring.

        Returns:
            TODO docstring.
        """
        return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in s.lower())

    return f"{_san(archetype)}__{_san(density)}__{_san(name)}"


def _hash_fragment(payload: dict) -> str:
    """TODO docstring. Document this function.

    Args:
        payload: TODO docstring.

    Returns:
        TODO docstring.
    """
    hash_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(hash_bytes).hexdigest()[:10]


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

    Returns:
        List of validated ScenarioDescriptor objects with generated seeds and identifiers.
    """

    if not isinstance(raw, list) or not raw:
        raise ValueError("Raw scenarios must be a non-empty list")

    scenarios: list[ScenarioDescriptor] = []
    seen_ids: set[str] = set()
    for index, sc in enumerate(raw):
        name, archetype, density, max_steps, map_file, params_source = _normalise_raw_scenario(
            index,
            sc,
        )

        map_path = _resolve_map_path(map_file, cfg, name)
        planned_seeds = _plan_seeds(sc, cfg, rng)
        hash_fragment = _hash_fragment(
            {
                "name": name,
                "archetype": archetype,
                "density": density,
                "map": str(map_path),
                "max_episode_steps": max_steps,
                "params": params_source,
            },
        )
        scenario_id = _scenario_id(archetype, density, name)
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
                raw=sc,
                planned_seeds=planned_seeds,
                max_episode_steps=max_steps,
                hash_fragment=hash_fragment,
            ),
        )

    return scenarios


def expand_episode_jobs(scenarios: list[ScenarioDescriptor], cfg) -> list[EpisodeJob]:  # T024
    """Create concrete episode jobs for initial execution plan.

    - One job per planned seed per scenario.
    - Horizon override: if cfg.horizon_override provided, use it; else scenario.max_episode_steps.
    - job_id construction: sha1 of scenario_id + seed + horizon for determinism; truncated.

    Returns:
        List of EpisodeJob objects, one per seed per scenario.
    """
    jobs: list[EpisodeJob] = []
    for sc in scenarios:
        horizon = (
            int(cfg.horizon_override)
            if getattr(cfg, "horizon_override", None)
            else sc.max_episode_steps
        )
        for seed in sc.planned_seeds:
            base = f"{sc.scenario_id}:{seed}:{horizon}"
            job_hash = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
            jobs.append(
                EpisodeJob(
                    job_id=job_hash,
                    scenario_id=sc.scenario_id,
                    seed=seed,
                    archetype=sc.archetype,
                    density=sc.density,
                    horizon=horizon,
                    scenario=sc,
                ),
            )
    return jobs
