"""Public API facade for Robot SF.

Provides ergonomic, lightweight entry points for scenario loading,
environment construction, and episode execution.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.baselines.interface import PlannerProtocol
from robot_sf.benchmark.types import EpisodeRecord, MetricsBundle, ScenarioSpec


def _find_repo_root() -> Path:
    """Resolve the repository root directory.

    Returns:
        Path: Absolute path to the repository root.
    """
    return Path(__file__).resolve().parent.parent


def _resolve_scenario_path(scenario_id: str | Path) -> Path:
    """Locate a scenario YAML file from an ID, name, or path.

    Args:
        scenario_id: Direct file path, path relative to configs/scenarios, or stem name.

    Returns:
        Path: Resolved absolute path to the scenario YAML file.

    Raises:
        FileNotFoundError: If the scenario file cannot be located.
    """
    candidate_path = Path(scenario_id)
    if candidate_path.is_file():
        return candidate_path.resolve()

    scenarios_root = _find_repo_root() / "configs" / "scenarios"
    candidates = (
        scenarios_root / candidate_path,
        scenarios_root / f"{candidate_path}.yaml",
        scenarios_root / "single" / candidate_path,
        scenarios_root / "single" / f"{candidate_path}.yaml",
    )
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()

    stem_query = candidate_path.stem
    for match in scenarios_root.glob("**/*.yaml"):
        if match.stem == stem_query or match.name == str(candidate_path):
            return match.resolve()

    raise FileNotFoundError(
        f"Scenario {scenario_id!r} could not be resolved to a valid scenario YAML file."
    )


def load_scenario(scenario_id: str | Path) -> dict[str, Any]:
    """Resolve and load a scenario definition from configs/scenarios/*.yaml.

    Args:
        scenario_id: Either a direct file path, a path relative to configs/scenarios,
            or a scenario identifier / file stem.

    Returns:
        Loaded scenario definition dictionary with ``__scenario_path__`` attached.

    Raises:
        FileNotFoundError: If the scenario file cannot be located.
        ValueError: If no matching scenario is found within the file.
    """
    from robot_sf.training.scenario_loader import load_scenarios  # noqa: PLC0415

    resolved_path = _resolve_scenario_path(scenario_id)
    loaded = load_scenarios(resolved_path)
    if not loaded:
        raise ValueError(f"No scenarios found in {resolved_path}")

    target_id = Path(scenario_id).stem
    selected: dict[str, Any] | None = None
    for sc in loaded:
        if sc.get("id") == target_id or sc.get("name") == target_id:
            selected = dict(sc)
            break
    if selected is None:
        selected = dict(loaded[0])

    selected["__scenario_path__"] = str(resolved_path)
    return selected


def make_env(
    *,
    scenario: str | Path | Mapping[str, Any] | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> Any:
    """Create a robot simulation environment with an ergonomic keyword-only API.

    This function wraps :func:`robot_sf.gym_env.environment_factory.make_robot_env`
    and optionally builds the required simulation configuration directly from a scenario ID or file.

    Args:
        scenario: Optional scenario identifier, path, or dictionary.
        seed: Deterministic random seed.
        **kwargs: Additional options forwarded to :func:`make_robot_env`.

    Returns:
        Gymnasium environment instance.
    """
    from robot_sf.gym_env.environment_factory import make_robot_env  # noqa: PLC0415
    from robot_sf.training.scenario_loader import (  # noqa: PLC0415
        build_robot_config_from_scenario,
    )

    scenario_name = "default"
    if scenario is not None:
        if isinstance(scenario, (str, Path)):
            sc_dict = load_scenario(scenario)
        elif isinstance(scenario, Mapping):
            sc_dict = dict(scenario)
        else:
            raise TypeError(
                f"scenario must be a str, Path, or Mapping, got {type(scenario).__name__}"
            )

        scenario_path_str = sc_dict.get("__scenario_path__")
        sc_path = (
            Path(scenario_path_str)
            if scenario_path_str is not None
            else _find_repo_root() / "configs" / "scenarios" / "scenario.yaml"
        )
        scenario_name = str(sc_dict.get("name") or sc_dict.get("id") or scenario)
        config = build_robot_config_from_scenario(sc_dict, scenario_path=sc_path)
        kwargs.setdefault("config", config)
        kwargs.setdefault("scenario_name", scenario_name)

    env = make_robot_env(seed=seed, **kwargs)
    env.scenario_id = scenario_name
    if seed is not None:
        env.applied_seed = seed
    return env


def _extract_action(planner: Any, obs: Any, env: Any) -> Any:
    """Generate an action from planner, callable, or default zeros.

    Returns:
        Action compatible with the environment's action space.
    """
    if planner is not None:
        if hasattr(planner, "step"):
            return planner.step(obs)
        if callable(planner):
            return planner(obs)
        return env.action_space.sample()
    return np.zeros(2, dtype=np.float32)


def _extract_metrics(
    last_info: dict[str, Any], steps: int, total_reward: float, duration: float
) -> dict[str, float]:
    """Extract metrics dictionary from step outcomes.

    Returns:
        Dictionary mapping metric names to scalar float values.
    """
    meta = last_info.get("meta", {}) if isinstance(last_info, dict) else {}
    metrics_values: dict[str, float] = {
        "success": float(last_info.get("success", meta.get("is_route_complete", False))),
        "collision": float(last_info.get("collision", meta.get("collision", False))),
        "steps": float(steps),
        "total_reward": float(total_reward),
        "duration_s": float(duration),
    }
    for k, v in meta.items():
        if isinstance(v, (int, float, bool)) and not isinstance(v, (list, dict)):
            metrics_values[str(k)] = float(v)
    return metrics_values


def run_episode(
    env: Any,
    *,
    planner: Any = None,
    max_steps: int | None = None,
    seed: int | None = None,
) -> EpisodeRecord:
    """Execute a single episode in the provided environment and return an EpisodeRecord.

    Args:
        env: The Gymnasium simulation environment.
        planner: Optional planner implementing :class:`PlannerProtocol` or a callable action generator.
        max_steps: Maximum step limit before truncating the episode.
        seed: Seed to pass to ``env.reset()``. Defaults to ``env.applied_seed`` or 0.

    Returns:
        An :class:`EpisodeRecord` containing episode metrics and metadata.
    """
    resolved_seed = seed if seed is not None else getattr(env, "applied_seed", None) or 0

    obs, info = env.reset(seed=resolved_seed)
    if planner is not None and hasattr(planner, "reset"):
        planner.reset(seed=resolved_seed)

    steps = 0
    total_reward = 0.0
    done = False
    start_time = time.perf_counter()
    last_info = info or {}

    while not done:
        action = _extract_action(planner, obs, env)
        obs, reward, terminated, truncated, last_info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            truncated = True
        done = terminated or truncated

    duration = time.perf_counter() - start_time
    metrics_values = _extract_metrics(last_info, steps, total_reward, duration)

    meta = last_info.get("meta", {}) if isinstance(last_info, dict) else {}
    scenario_id = (
        getattr(env, "scenario_id", None)
        or meta.get("scenario_name")
        or meta.get("scenario_id")
        or "default"
    )
    episode_id = f"{scenario_id}_{resolved_seed}_{int(time.time() * 1000)}"

    algo_name = None
    if planner is not None:
        algo_name = (
            getattr(planner, "name", None) or getattr(planner, "__class__", type(planner)).__name__
        )
    else:
        algo_name = "zero_action"

    return EpisodeRecord(
        version="v1",
        episode_id=episode_id,
        scenario_id=str(scenario_id),
        seed=int(resolved_seed),
        metrics=MetricsBundle(values=metrics_values),
        algo=algo_name,
        horizon=steps,
        timing={"wall_time_s": duration},
        raw={
            "last_info_scalars": {
                k: v for k, v in last_info.items() if isinstance(v, (int, float, str, bool))
            }
        },
    )


__all__ = [
    "EpisodeRecord",
    "PlannerProtocol",
    "ScenarioSpec",
    "load_scenario",
    "make_env",
    "run_episode",
]
