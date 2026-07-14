"""Execution helpers for the issue #3574 mean-matched heterogeneity harness.

The dry-run manifest built by :mod:`heterogeneous_population_ablation` describes
paired population rows but carries no runtime trace. This module turns one such
manifest row into the runtime scenario the map runner expects and executes it so
that the emitted episode record carries the ``algorithm_metadata.pedestrian_control_trace``
(with per-step ``clearance_m`` / ``near_field_exposure_s``) that the readiness
gate in :func:`assess_mean_matched_episode_records` demands.

Extracted from ``scripts/benchmark/run_heterogeneous_population_ablation_issue_3574.py``
so both the campaign runner and the acceptance tests exercise the same scenario
assembly and emission path (issue #5397): the trace the readiness gate requires
is produced on the harness path, not by a parallel test-only reconstruction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robot_sf.benchmark.map_runner import _build_policy
from robot_sf.benchmark.map_runner_episode import run_map_episode

# Per-archetype desired-speed factors used when materializing a manifest row into
# a runtime population. The ``mean_matched_homogeneous`` value is the weighted mean
# of the mixture arm and keeps the two arms mean-matched (see the smoke config).
DEFAULT_ARCHETYPE_SPEED_FACTORS = {
    "cautious": 0.7,
    "standard": 1.0,
    "hurried": 1.4,
    "mean_matched_homogeneous": 1.025,
}

DEFAULT_ARCHETYPE_SEED = 3574
DEFAULT_MAX_EPISODE_STEPS = 600


def build_episode_scenario(
    row: dict[str, Any],
    *,
    map_path: Path | str | None = None,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    archetype_speed_factors: dict[str, float] | None = None,
    archetype_seed: int = DEFAULT_ARCHETYPE_SEED,
) -> dict[str, Any]:
    """Assemble the runtime scenario dict for one paired manifest row.

    The scenario carries the row's ``pedestrian_control_trace_labels`` so the map
    runner attaches a per-pedestrian control trace to the emitted record.

    Returns:
        A scenario dictionary consumable by :func:`run_map_episode`.
    """

    arm_population = row["arm_population"]
    population_size = sum(arm_population["counts"].values())
    resolved_map_path = _resolve_row_map_path(row, fallback_map_path=map_path)
    return {
        "name": row["scenario_id"],
        "map_file": str(resolved_map_path),
        "simulation_config": {
            "max_episode_steps": int(max_episode_steps),
            "ped_density": float(row["density"]),
            "population_size": population_size,
            "response_law_composition": arm_population.get("response_law_composition"),
            "response_law_seed": arm_population.get("response_law_seed"),
            "archetype_composition": arm_population.get("composition"),
            "archetype_speed_factors": dict(
                DEFAULT_ARCHETYPE_SPEED_FACTORS
                if archetype_speed_factors is None
                else archetype_speed_factors
            ),
            "archetype_seed": archetype_seed,
        },
        "pedestrian_control_trace_labels": arm_population["pedestrian_control_trace_labels"],
        "robot_config": {},
    }


def run_manifest_row(
    row: dict[str, Any],
    *,
    map_path: Path | str | None = None,
    scenario_path: Path,
    horizon: int = DEFAULT_MAX_EPISODE_STEPS,
    dt: float = 0.1,
    max_episode_steps: int | None = None,
) -> dict[str, Any]:
    """Run one paired manifest row and return its annotated episode record.

    The record is annotated with the campaign keys the readiness gate indexes by
    (``scenario_id``/``planner``/``seed``/``population_arm``/``response_law_fraction``)
    so :func:`assess_mean_matched_episode_records` can pair it against the manifest.

    Returns:
        The emitted episode record with campaign metadata attached.
    """

    response_law_fraction_value = row.get("response_law_fraction")
    response_law_fraction = float(
        0.0 if response_law_fraction_value is None else response_law_fraction_value
    )
    scenario = build_episode_scenario(
        row,
        map_path=map_path,
        max_episode_steps=horizon if max_episode_steps is None else max_episode_steps,
    )

    record = run_map_episode(
        scenario=scenario,
        seed=int(row["seed"]),
        horizon=horizon,
        dt=dt,
        record_forces=True,
        snqi_weights=None,
        snqi_baseline=None,
        algo=row["planner"],
        scenario_path=scenario_path,
        record_planner_decision_trace=False,
        record_simulation_step_trace=True,
        policy_builder=_build_policy,
    )

    record["population_arm"] = row["population_arm"]
    record["planner"] = row["planner"]
    record["seed"] = int(row["seed"])
    record["scenario_id"] = row["scenario_id"]
    record["response_law_fraction"] = response_law_fraction

    scenario_params = record.setdefault("scenario_params", {})
    scenario_params["population_arm"] = row["population_arm"]
    scenario_params["planner"] = row["planner"]
    scenario_params["seed"] = int(row["seed"])
    scenario_params["scenario_id"] = row["scenario_id"]
    scenario_params["response_law_fraction"] = response_law_fraction

    return record


def _resolve_row_map_path(row: dict[str, Any], *, fallback_map_path: Path | str | None) -> Path:
    """Resolve a row-owned map, retaining explicit fallback support for legacy inline rows.

    Returns:
        Existing absolute map path selected from the row or explicit fallback.
    """

    raw_map_path = row.get("map_file", fallback_map_path)
    if raw_map_path is None:
        raise ValueError(
            "manifest row has no map_file; matrix-derived manifests must carry a per-row map "
            "and legacy inline rows require an explicit map_path"
        )
    map_path = Path(raw_map_path).expanduser()
    if map_path.is_absolute():
        candidates = [map_path]
    else:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [Path.cwd() / map_path, repo_root / map_path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise ValueError(f"manifest row map_file does not resolve to a file; searched: {searched}")


__all__ = [
    "DEFAULT_ARCHETYPE_SEED",
    "DEFAULT_ARCHETYPE_SPEED_FACTORS",
    "DEFAULT_MAX_EPISODE_STEPS",
    "build_episode_scenario",
    "run_manifest_row",
]
