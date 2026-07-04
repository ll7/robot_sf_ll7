#!/usr/bin/env python3
"""Run a bounded actual fidelity-sensitivity campaign for issue #3207.

This runner deliberately avoids the broad benchmark CLI because minimal local
environments may not install learned-policy dependencies. It executes real
Robot SF simulator episodes for a compact non-learned planner slice and writes
raw JSONL under ``output/`` plus a compact evidence bundle under
``docs/context/evidence``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from robot_sf.baselines.interface import Observation
from robot_sf.baselines.social_force import SFPlannerConfig, SocialForcePlanner
from robot_sf.benchmark.fidelity_fixed_scope_preflight import build_fixed_scope_preflight
from robot_sf.benchmark.fidelity_rank_stability import (
    analyze_fidelity_sensitivity,
    check_rank_identifiability_contract,
    write_rank_identifiability_report,
)
from robot_sf.benchmark.fidelity_sensitivity import (
    DIAGNOSTIC_SMOKE_CLAIM_BOUNDARY,
    validate_fidelity_sensitivity_config,
)
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    build_hybrid_rule_local_planner_config,
)
from robot_sf.planner.socnav import ORCAPlannerAdapter, SocNavPlannerConfig
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "issue_3207_fidelity_sensitivity_campaign_slice.v1"
DEFAULT_CONFIG = "configs/research/fidelity_sensitivity_v1.yaml"
DEFAULT_SCENARIO_SET = "configs/scenarios/sets/paper_cross_kinematics_v1.yaml"
DEFAULT_RAW_ROOT = "output/fidelity_sensitivity/issue_3207_actual_slice_2026-06-20"
DEFAULT_EVIDENCE_DIR = (
    "docs/context/evidence/issue_3207_fidelity_sensitivity_actual_slice_2026-06-20"
)
PLANNERS = ("goal_seek", "baseline_social_force")
METRICS = (
    "success_rate",
    "collision_rate",
    "min_clearance",
    "mean_clearance",
    "near_miss_rate",
    "comfort_exposure_mean",
    "time_to_goal_norm",
)
CLAIM_BOUNDARY = (
    "bounded_actual_campaign_slice_not_full_benchmark_evidence: executes real Robot SF "
    "episodes for a compact two-planner local fidelity-sensitivity slice. It measures "
    "internal sensitivity on this slice only; it is not simulator-realism, sim-to-real, "
    "paper-facing planner-ranking, or full #3207 acceptance evidence."
)
ARCHETYPE_SPEED_FACTORS = {"cautious": 0.8, "standard": 1.0, "hurried": 1.2}

FIXED_SCOPE_PLAN_SCHEMA_VERSION = "issue_3207_fidelity_fixed_scope_run_plan.v1"
FIXED_SCOPE_PLAN_CLAIM_BOUNDARY = (
    "fixed_scope_run_plan_enumeration_only: consumes the issue #3207 fixed-scope preflight "
    "packet and enumerates the concrete planner_group x axis-variant x seed run cells that the "
    "full fixed-scope campaign would execute. It launches no episode and promotes no claim; "
    "execution stays fail-closed behind unmet launch prerequisites (ORCA/rvo2 runtime "
    "dependency, hybrid-rule explicit opt-in, and the post-run rank-identifiability recheck). "
    "It is not benchmark evidence, not simulator-realism evidence, not sim-to-real evidence, "
    "and not paper-facing evidence."
)


class FixedScopeNotLaunchableError(RuntimeError):
    """Raised when a fixed-scope run plan still has unmet launch gates.

    The bounded slice runner intentionally leaves the ORCA/rvo2 runtime
    dependency, the hybrid-rule explicit opt-in, and the post-run
    rank-identifiability recheck unsatisfied, so an actual full campaign launch
    must fail closed rather than silently run against unresolved prerequisites.
    """


@dataclass(frozen=True)
class FixedScopeRunCell:
    """One planner_group x axis-variant x seed cell of the full fixed scope.

    A cell describes a unit of work the full campaign would execute for each
    scenario in the fixed scenario set. It carries the resolved catalog
    algorithm and its availability/opt-in state so downstream launch logic can
    reason about the cell without re-resolving the planner group.
    """

    planner_group: str
    algorithm: str
    planner_available: bool
    planner_tier: str | None
    requires_explicit_opt_in: bool
    axis: str
    variant: str
    baseline_variant: bool
    seed: int
    scenario_set: str


def enumerate_fixed_scope_run_cells(
    validated: Mapping[str, Any],
    resolution_by_group: Mapping[str, Mapping[str, Any]],
) -> list[FixedScopeRunCell]:
    """Enumerate the full fixed-scope run cells from a validated study config.

    The enumeration mirrors the preflight materialization order
    (planner_group x every axis variant x seed, including baseline variants) so
    the cell count matches ``materialized_scope.run_cells_per_scenario``.

    Args:
        validated: Config returned by ``validate_fidelity_sensitivity_config``.
        resolution_by_group: Planner-group name to preflight resolution record.

    Returns:
        Run cells in deterministic planner/axis/variant/seed order.
    """
    fixed_scope = validated["fixed_scope"]
    seeds = [int(seed) for seed in fixed_scope["seeds"]]
    scenario_set = str(fixed_scope["scenario_set"])
    planner_groups = [str(group) for group in fixed_scope["planner_groups"]]
    cells: list[FixedScopeRunCell] = []
    for group in planner_groups:
        record = resolution_by_group.get(group, {})
        for axis in validated["axes"]:
            axis_key = str(axis["key"])
            for variant in axis["variants"]:
                variant_key = str(variant["key"])
                baseline = bool(variant.get("baseline", False))
                for seed in seeds:
                    cells.append(
                        FixedScopeRunCell(
                            planner_group=group,
                            algorithm=str(
                                record.get("canonical_name") or record.get("algorithm", group)
                            ),
                            planner_available=bool(record.get("available", False)),
                            planner_tier=record.get("tier"),
                            requires_explicit_opt_in=bool(
                                record.get("requires_explicit_opt_in", False)
                            ),
                            axis=axis_key,
                            variant=variant_key,
                            baseline_variant=baseline,
                            seed=seed,
                            scenario_set=scenario_set,
                        )
                    )
    return cells


def build_fixed_scope_run_plan(
    config: Mapping[str, Any],
    *,
    config_path: str,
    git_head: str,
    date: str | None = None,
) -> dict[str, Any]:
    """Consume the #3207 fixed-scope preflight and build an executable run plan.

    This is the runner-side counterpart to
    :func:`robot_sf.benchmark.fidelity_fixed_scope_preflight.build_fixed_scope_preflight`:
    the preflight owns the launch/readiness gate, while this function turns the
    materialized scope into the concrete run cells the campaign runner would
    iterate. It runs no episode. ``executable`` is only ``True`` when the
    preflight is ready and *all* launch prerequisites and blockers are cleared.
    The shipped config carries the fixed-scope hybrid-rule opt-in and structured
    post-run rank-identifiability contract, while ORCA/rvo2 remains
    runtime-checked.

    Args:
        config: Raw fidelity-sensitivity study config mapping.
        config_path: Repo-relative config path, recorded for provenance.
        git_head: Git head recorded for provenance.
        date: Optional ISO date string recorded for provenance.

    Returns:
        JSON-serializable run-plan packet embedding the preflight decision, the
        enumerated run cells, and the residual launch gates.
    """
    validated = validate_fidelity_sensitivity_config(config)
    preflight = build_fixed_scope_preflight(
        config, config_path=config_path, git_head=git_head, date=date
    )
    resolution_by_group = {
        str(record["planner_group"]): record for record in preflight["planner_resolution"]
    }
    cells = enumerate_fixed_scope_run_cells(validated, resolution_by_group)

    materialized = preflight["materialized_scope"]
    expected_cells = int(materialized["run_cells_per_scenario"])
    if len(cells) != expected_cells:
        # Internal contract: the runner plan and the preflight materialization
        # must agree on scope, otherwise one of them is stale.
        raise ValueError(
            "fixed-scope run-plan enumeration disagrees with preflight materialization: "
            f"{len(cells)} cells vs run_cells_per_scenario={expected_cells}"
        )

    blockers = list(preflight["blockers"])
    launch_prerequisites = list(preflight["launch_prerequisites"])
    gate_reasons = blockers + launch_prerequisites
    executable = bool(preflight["preflight_ready"]) and not gate_reasons

    return {
        "schema_version": FIXED_SCOPE_PLAN_SCHEMA_VERSION,
        "issue": int(preflight.get("issue", 3207)),
        "study_id": str(validated["study_id"]),
        "claim_boundary": FIXED_SCOPE_PLAN_CLAIM_BOUNDARY,
        "config_path": config_path,
        "git_head": git_head,
        "date": date,
        "preflight_decision": preflight["decision"],
        "preflight_ready": bool(preflight["preflight_ready"]),
        "preflight_claim_boundary": preflight["claim_boundary"],
        "executable": executable,
        "launched": False,
        "run_cell_count": len(cells),
        "run_cells_per_scenario_expected": expected_cells,
        "materialized_scope": materialized,
        "planner_resolution": preflight["planner_resolution"],
        "primary_metric": preflight["primary_metric"],
        "blockers": blockers,
        "launch_prerequisites": launch_prerequisites,
        "post_run_contracts": preflight["post_run_contracts"],
        "post_run_contract_specs": preflight["post_run_contract_specs"],
        "gate_reasons": gate_reasons,
        "run_cells": [asdict(cell) for cell in cells],
    }


def ensure_fixed_scope_launchable(plan: Mapping[str, Any]) -> None:
    """Fail closed unless a fixed-scope run plan has cleared every launch gate.

    Args:
        plan: Packet from :func:`build_fixed_scope_run_plan`.

    Raises:
        FixedScopeNotLaunchableError: If the plan is not executable or any
            blocker/launch-prerequisite remains.
    """
    gate_reasons = list(plan.get("gate_reasons") or [])
    if plan.get("executable") and not gate_reasons:
        return
    reasons = gate_reasons or ["preflight_not_ready"]
    raise FixedScopeNotLaunchableError(
        "fixed-scope campaign is not launchable; unmet gates: " + "; ".join(reasons)
    )


def write_fixed_scope_run_plan(
    plan: Mapping[str, Any], output_dir: str | pathlib.Path
) -> pathlib.Path:
    """Write a deterministic fixed-scope run-plan JSON to gitignored output.

    Returns:
        Path to the written ``fidelity_fixed_scope_run_plan.json`` file.
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "fidelity_fixed_scope_run_plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plan_path


# Catalog algorithm -> concrete runner planner. This is the only place a
# fixed-scope planner group becomes an actual episode-executing planner. The
# bounded slice runner has a native, dependency-free implementation only for the
# social-force baseline, so that is the only binding here. ORCA (needs rvo2) and
# the experimental hybrid-rule planner deliberately have NO binding: a cell that
# resolves to them stays unbound and fails closed rather than silently running a
# substitute planner (no silent fallback).
RUNNER_ALGORITHM_PLANNERS: Mapping[str, str] = {
    "orca": "orca",
    "social_force": "baseline_social_force",
    "hybrid_rule_local_planner": "hybrid_rule_v0_minimal",
}


@dataclass(frozen=True)
class RunnerCellBinding:
    """Concrete runner inputs bound to one fixed-scope run cell.

    ``runner_bound`` is ``True`` only when the cell's resolved catalog algorithm
    has a native runner planner *and* the preflight marked the planner available
    and not pending an explicit opt-in *and* the variant has a supported runtime
    binding. When it is ``False``, ``planner_name`` is ``None`` and
    ``unbound_reason`` explains why; the runner must never substitute a supported
    planner for an unbound cell.
    """

    cell: FixedScopeRunCell
    variant: VariantSpec
    planner_name: str | None
    runner_bound: bool
    unbound_reason: str | None


def build_fixed_scope_variant_index(
    config: Mapping[str, Any],
) -> dict[tuple[str, str], VariantSpec]:
    """Index one runtime-bound :class:`VariantSpec` per ``(axis, source key)``.

    Unlike :func:`load_variant_specs` — which collapses to a single nominal
    baseline for the bounded slice — this materializes *every* configured
    variant, including each axis's own baseline, and keys them by
    ``(axis_key, source_variant_key)``. That is exactly the grain a fixed-scope
    run cell carries (``cell.axis`` / ``cell.variant``), so every plan cell maps
    to exactly one variant runtime binding.

    Args:
        config: Raw fidelity-sensitivity study config mapping.

    Returns:
        Mapping of ``(axis_key, source_variant_key)`` to its runtime-bound spec.
    """
    index: dict[tuple[str, str], VariantSpec] = {}
    for axis in config["axes"]:
        axis_key = str(axis["key"])
        for raw_variant in axis["variants"]:
            source_key = str(raw_variant["key"])
            baseline = bool(raw_variant.get("baseline", False))
            patch = copy.deepcopy(raw_variant.get("patch") or {})
            observation_noise = copy.deepcopy(raw_variant.get("observation_noise") or {})
            runtime_binding = _runtime_binding(axis_key, patch, observation_noise)
            index[(axis_key, source_key)] = VariantSpec(
                axis=axis_key,
                # Keep every variant key distinct (baselines included) so per-axis
                # nominal cells stay traceable in output rows and stable seeds.
                key=f"{axis_key}__{source_key}",
                source_key=source_key,
                baseline=baseline,
                patch=patch,
                observation_noise=observation_noise,
                runtime_binding=runtime_binding,
            )
    return index


def bind_fixed_scope_run_cell(
    cell: FixedScopeRunCell,
    variant_index: Mapping[tuple[str, str], VariantSpec],
) -> RunnerCellBinding:
    """Bind one fixed-scope run cell to concrete runner inputs, fail-closed.

    The cell's ``(axis, variant)`` must resolve to a materialized variant spec;
    a miss is an internal contract violation (the plan and the variant index
    disagree), raised rather than silently dropped. The planner side is
    fail-closed: a cell is only ``runner_bound`` when its resolved algorithm has
    a native runner planner and the preflight found it available and not pending
    an explicit opt-in. Unbound cells carry ``planner_name = None`` and never
    inherit a substitute planner.

    Args:
        cell: One enumerated fixed-scope run cell.
        variant_index: Output of :func:`build_fixed_scope_variant_index`.

    Returns:
        The resolved :class:`RunnerCellBinding`.

    Raises:
        KeyError: If the cell's ``(axis, variant)`` is absent from the index.
    """
    variant = variant_index.get((cell.axis, cell.variant))
    if variant is None:
        raise KeyError(
            "fixed-scope run cell has no materialized variant: "
            f"axis={cell.axis!r} variant={cell.variant!r}"
        )

    reasons: list[str] = []
    planner_name = RUNNER_ALGORITHM_PLANNERS.get(cell.algorithm)
    if planner_name is None:
        reasons.append(f"no_native_runner_planner_for_algorithm:{cell.algorithm}")
    if not cell.planner_available:
        reasons.append(f"planner_unavailable:{cell.planner_group}")
    if cell.requires_explicit_opt_in:
        reasons.append(f"planner_requires_explicit_opt_in:{cell.planner_group}")
    if variant.runtime_binding == "unsupported":
        reasons.append(f"variant_runtime_binding_unsupported:{cell.axis}__{cell.variant}")

    runner_bound = not reasons
    return RunnerCellBinding(
        cell=cell,
        variant=variant,
        planner_name=planner_name if runner_bound else None,
        runner_bound=runner_bound,
        unbound_reason=None if runner_bound else "; ".join(reasons),
    )


def _run_cells_from_plan(plan: Mapping[str, Any]) -> list[FixedScopeRunCell]:
    """Reconstruct :class:`FixedScopeRunCell` objects from a serialized plan."""
    return [FixedScopeRunCell(**cell) for cell in plan["run_cells"]]


def bind_fixed_scope_run_plan(
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[RunnerCellBinding]:
    """Bind every cell of a fixed-scope run plan to concrete runner inputs.

    Args:
        plan: Packet from :func:`build_fixed_scope_run_plan`.
        config: Raw fidelity-sensitivity study config the plan was built from.

    Returns:
        One :class:`RunnerCellBinding` per plan cell, in plan order.
    """
    variant_index = build_fixed_scope_variant_index(config)
    return [bind_fixed_scope_run_cell(cell, variant_index) for cell in _run_cells_from_plan(plan)]


def execute_fixed_scope_cells(
    bindings: Sequence[RunnerCellBinding],
    *,
    cell_runner: Callable[[RunnerCellBinding], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Run each runner-bound cell via ``cell_runner``; fail closed on unbound.

    This is the per-cell execution seam: ``cell_runner`` receives one binding and
    returns its episode rows. It is injected so the mapping from plan cell to
    runner inputs (planner, variant, seed) is testable without the heavy real
    simulator, and so the campaign runner never grows a second silent-fallback
    path. If *any* binding is unbound this raises before running a single cell,
    so a launch that slipped past the plan-level gate still cannot substitute a
    supported planner for an ORCA/hybrid cell.

    Args:
        bindings: Bound cells from :func:`bind_fixed_scope_run_plan`.
        cell_runner: Callable that executes one bound cell and returns its rows.

    Returns:
        Flattened episode rows across all bound cells.

    Raises:
        FixedScopeNotLaunchableError: If any binding is not ``runner_bound``.
    """
    unbound = [b for b in bindings if not b.runner_bound]
    if unbound:
        reasons = "; ".join(
            f"{b.cell.planner_group}/{b.cell.axis}/{b.cell.variant}: {b.unbound_reason}"
            for b in unbound[:5]
        )
        raise FixedScopeNotLaunchableError(
            f"{len(unbound)} of {len(bindings)} fixed-scope cells are unbound; "
            f"refusing to run with a substitute planner. Examples: {reasons}"
        )
    rows: list[dict[str, Any]] = []
    for binding in bindings:
        rows.extend(cell_runner(binding))
    return rows


@dataclass(frozen=True)
class VariantSpec:
    """Runtime-bound fidelity variant used by the compact campaign."""

    axis: str
    key: str
    source_key: str
    baseline: bool
    patch: Mapping[str, Any]
    observation_noise: Mapping[str, Any]
    runtime_binding: str


class GoalSeekPlanner:
    """Deterministic goal-facing unicycle command policy."""

    def __init__(self, *, max_linear_speed: float, max_angular_speed: float) -> None:
        """Store command limits used by the goal-facing controller."""
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)

    def reset(self, *, seed: int | None = None) -> None:
        """Accept a seed for interface parity."""
        del seed

    def step(self, obs: Observation) -> dict[str, float]:
        """Return a bounded command toward the current goal."""
        robot_pos = np.asarray(obs.robot["position"], dtype=float)
        robot_goal = np.asarray(obs.robot["goal"], dtype=float)
        heading = float(obs.robot.get("heading", 0.0))
        delta = robot_goal - robot_pos
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-6:
            return {"v": 0.0, "omega": 0.0}
        desired_heading = float(math.atan2(delta[1], delta[0]))
        heading_error = _wrap_angle(desired_heading - heading)
        angular = float(
            np.clip(2.0 * heading_error, -self.max_angular_speed, self.max_angular_speed)
        )
        alignment = max(0.0, 1.0 - abs(heading_error) / math.pi)
        linear = float(np.clip(distance * alignment, 0.0, self.max_linear_speed))
        return {"v": linear, "omega": angular}


def _wrap_angle(value: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_status_short() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ["unknown"]
    if result.returncode != 0:
        return ["unknown"]
    return result.stdout.splitlines()


def _git_provenance() -> dict[str, Any]:
    status_short = _git_status_short()
    return {
        "git_head": _git_head(),
        "git_worktree_dirty": bool(status_short),
        "git_status_short_at_generation": status_short,
    }


def _repo_rel(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_variant_specs(
    config: Mapping[str, Any], *, include_all_variants: bool
) -> list[VariantSpec]:
    """Build runtime-bound variant specs from the checked-in #3207 config."""
    variants: list[VariantSpec] = []
    seen_baseline = False
    included_perturbation_axes: set[str] = set()
    for axis in config["axes"]:
        axis_key = str(axis["key"])
        for raw_variant in axis["variants"]:
            baseline = bool(raw_variant.get("baseline", False))
            if baseline:
                if seen_baseline:
                    continue
                seen_baseline = True
            elif not include_all_variants and axis_key in included_perturbation_axes:
                continue
            patch = copy.deepcopy(raw_variant.get("patch") or {})
            observation_noise = copy.deepcopy(raw_variant.get("observation_noise") or {})
            runtime_binding = _runtime_binding(axis_key, patch, observation_noise)
            if baseline or runtime_binding != "unsupported":
                variants.append(
                    VariantSpec(
                        axis=axis_key,
                        key=("baseline" if baseline else f"{axis_key}__{raw_variant['key']}"),
                        source_key=str(raw_variant["key"]),
                        baseline=baseline,
                        patch=patch,
                        observation_noise=observation_noise,
                        runtime_binding=runtime_binding,
                    )
                )
                if not baseline:
                    included_perturbation_axes.add(axis_key)
    if not any(variant.baseline for variant in variants):
        raise ValueError("no baseline variant found in config")
    return variants


def _runtime_binding(
    axis: str,
    patch: Mapping[str, Any],
    observation_noise: Mapping[str, Any],
) -> str:
    if axis == "integration_timestep" and "dt" in patch:
        return "sim_config.time_per_step_in_secs"
    if axis == "clearance_radius" and isinstance(patch.get("sim_config"), Mapping):
        return "sim_config.ped_radius"
    if axis == "social_force_speed_archetypes" and "pedestrian_archetypes" in patch:
        return "sim_config.archetype_composition"
    if axis == "observation_noise" and observation_noise:
        return "planner_observation_noise"
    return "unsupported"


def apply_variant(config: Any, variant: VariantSpec, *, seed: int) -> None:
    """Apply one runtime-bound fidelity variant to an environment config."""
    if variant.runtime_binding == "sim_config.time_per_step_in_secs":
        dt_value = float(variant.patch["dt"])
        original_duration = float(config.sim_config.sim_time_in_secs)
        config.sim_config.time_per_step_in_secs = dt_value
        config.sim_config.sim_time_in_secs = original_duration
    elif variant.runtime_binding == "sim_config.ped_radius":
        config.sim_config.ped_radius = float(variant.patch["sim_config"]["ped_radius"])
    elif variant.runtime_binding == "sim_config.archetype_composition":
        config.sim_config.archetype_composition = {
            str(key): float(value) for key, value in variant.patch["pedestrian_archetypes"].items()
        }
        config.sim_config.archetype_speed_factors = dict(ARCHETYPE_SPEED_FACTORS)
        config.sim_config.archetype_seed = int(seed)


def _build_observation(
    env: Any, *, noise: Mapping[str, Any], rng: np.random.Generator
) -> Observation:
    robot = env.simulator.robots[0]
    robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float)
    heading = float(robot.pose[1])
    linear, _angular = robot.current_speed
    robot_vel = np.array([float(linear) * math.cos(heading), float(linear) * math.sin(heading)])
    ped_positions = np.asarray(env.simulator.ped_pos, dtype=float)
    ped_velocities = np.asarray(
        getattr(env.simulator, "ped_vel", np.zeros_like(ped_positions)), dtype=float
    )
    if ped_velocities.shape != ped_positions.shape:
        ped_velocities = np.zeros_like(ped_positions)

    pose_std = float(noise.get("pose_noise_std_m", 0.0) or 0.0)
    heading_std = float(noise.get("heading_noise_std_rad", 0.0) or 0.0)
    dropout = float(noise.get("pedestrian_false_negative_prob", 0.0) or 0.0)
    observed_robot_pos = (
        robot_pos + rng.normal(0.0, pose_std, size=2) if pose_std > 0 else robot_pos
    )
    observed_heading = heading + float(rng.normal(0.0, heading_std)) if heading_std > 0 else heading
    keep_mask = np.ones((ped_positions.shape[0],), dtype=bool)
    if dropout > 0 and keep_mask.size:
        keep_mask = rng.random(keep_mask.shape[0]) >= dropout

    agents = [
        {
            "position": ped_positions[idx].tolist(),
            "velocity": ped_velocities[idx].tolist(),
            "goal": ped_positions[idx].tolist(),
            "radius": float(env.env_config.sim_config.ped_radius),
        }
        for idx in range(ped_positions.shape[0])
        if keep_mask[idx]
    ]
    return Observation(
        dt=float(env.env_config.sim_config.time_per_step_in_secs),
        robot={
            "position": observed_robot_pos.tolist(),
            "velocity": robot_vel.tolist(),
            "goal": np.asarray(env.simulator.goal_pos[0], dtype=float).tolist(),
            "heading": observed_heading,
            "radius": float(robot.config.radius),
        },
        agents=agents,
        obstacles=[],
    )


class AdapterPlanner:
    """Bridge benchmark adapter ``plan(dict)`` planners into this runner's ``step`` API."""

    def __init__(self, adapter: Any) -> None:
        """Store a benchmark adapter behind the compact runner planner API."""
        self._adapter = adapter

    def bind_env(self, env: Any) -> None:
        """Bind environment geometry when the wrapped adapter supports it."""
        bind_env = getattr(self._adapter, "bind_env", None)
        if callable(bind_env):
            bind_env(env)

    def step(self, obs: Observation) -> dict[str, float]:
        """Return a unicycle command for the compact runner episode loop."""
        linear, angular = self._adapter.plan(_socnav_adapter_observation(obs))
        return {"v": float(linear), "omega": float(angular)}

    def close(self) -> None:
        """Release wrapped adapter resources when available."""
        close = getattr(self._adapter, "close", None)
        if callable(close):
            close()


def _socnav_adapter_observation(obs: Observation) -> dict[str, Any]:
    """Convert compact baseline observation into SocNav adapter observation."""
    robot = dict(obs.robot)
    agents = list(obs.agents)
    return {
        "dt": float(obs.dt),
        "robot": {
            "position": robot.get("position", [0.0, 0.0]),
            "heading": [float(robot.get("heading", 0.0))],
            "speed": robot.get("velocity", [0.0, 0.0]),
            "radius": [float(robot.get("radius", 0.3))],
        },
        "goal": {"current": robot.get("goal", [0.0, 0.0])},
        "pedestrians": {
            "positions": [agent.get("position", [0.0, 0.0]) for agent in agents],
            "velocities": [agent.get("velocity", [0.0, 0.0]) for agent in agents],
            "count": [len(agents)],
            "radius": float(agents[0].get("radius", 0.3)) if agents else 0.3,
        },
        "obstacles": list(obs.obstacles),
    }


def _load_yaml_mapping(path: pathlib.Path) -> dict[str, Any]:
    """Load a YAML mapping for local adapter configuration."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"planner config must be a mapping: {_repo_rel(path)}")
    return data


def _socnav_config(config: Any) -> SocNavPlannerConfig:
    """Build a SocNav adapter config using the active scenario speed caps."""
    return SocNavPlannerConfig(
        max_linear_speed=float(config.robot_config.max_linear_speed),
        max_angular_speed=float(config.robot_config.max_angular_speed),
    )


def _planner(planner: str, config: Any, *, seed: int) -> Any:
    if planner == "goal_seek":
        return GoalSeekPlanner(
            max_linear_speed=float(config.robot_config.max_linear_speed),
            max_angular_speed=float(config.robot_config.max_angular_speed),
        )
    if planner == "baseline_social_force":
        return SocialForcePlanner(
            SFPlannerConfig(
                action_space="unicycle",
                mode="unicycle",
                dt=float(config.sim_config.time_per_step_in_secs),
                v_max=float(config.robot_config.max_linear_speed),
                omega_max=float(config.robot_config.max_angular_speed),
            ),
            seed=seed,
        )
    if planner == "orca":
        return AdapterPlanner(
            ORCAPlannerAdapter(config=_socnav_config(config), allow_fallback=False)
        )
    if planner == "hybrid_rule_v0_minimal":
        algo_config = _load_yaml_mapping(REPO_ROOT / "configs/algos/hybrid_rule_v0_minimal.yaml")
        if not bool(algo_config.get("allow_testing_algorithms")):
            raise ValueError("hybrid_rule_v0_minimal requires allow_testing_algorithms opt-in")
        return AdapterPlanner(
            HybridRuleLocalPlannerAdapter(
                config=build_hybrid_rule_local_planner_config(algo_config)
            )
        )
    raise ValueError(f"unsupported planner: {planner}")


def _env_action(env: Any, command: Mapping[str, float]) -> np.ndarray:
    current_linear, current_angular = env.simulator.robots[0].current_speed
    desired_linear = float(command.get("v", command.get("linear", 0.0)))
    desired_angular = float(command.get("omega", command.get("angular", 0.0)))
    return np.array(
        [desired_linear - float(current_linear), desired_angular - float(current_angular)],
        dtype=float,
    )


def _surface_clearances(env: Any) -> np.ndarray:
    """Return robot-pedestrian surface clearances for the current simulator state."""
    ped_positions = np.asarray(env.simulator.ped_pos, dtype=float)
    if ped_positions.size == 0:
        return np.asarray([], dtype=float)
    robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float)
    center_distances = np.linalg.norm(ped_positions - robot_pos, axis=1)
    robot_radius = float(env.simulator.robots[0].config.radius)
    ped_radius = float(env.env_config.sim_config.ped_radius)
    return center_distances - robot_radius - ped_radius


def run_episode(
    scenario: Mapping[str, Any],
    *,
    scenario_path: pathlib.Path,
    variant: VariantSpec,
    planner_name: str,
    seed: int,
    horizon: int,
) -> dict[str, Any]:
    """Run one actual simulator episode and return a compact row."""
    config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
    baseline_dt = float(config.sim_config.time_per_step_in_secs)
    target_duration = min(
        float(horizon) * baseline_dt,
        float(config.sim_config.sim_time_in_secs),
    )
    apply_variant(config, variant, seed=seed)
    variant_dt = float(config.sim_config.time_per_step_in_secs)
    max_steps = max(1, math.ceil(target_duration / variant_dt))
    env = make_robot_env(config=config, seed=seed, debug=False)
    rng = np.random.default_rng(_stable_seed(seed, variant.key, planner_name))
    planner = _planner(planner_name, config, seed=seed)
    bind_env = getattr(planner, "bind_env", None)
    if callable(bind_env):
        bind_env(env)
    reset = getattr(planner, "reset", None)
    if callable(reset):
        reset(seed=seed)

    clearances: list[float] = []
    near_misses = 0
    comfort_exposure: list[float] = []
    terminated = False
    info: dict[str, Any] = {}
    steps = 0
    try:
        env.reset(seed=seed)
        for step_idx in range(max_steps):
            obs = _build_observation(env, noise=variant.observation_noise, rng=rng)
            command = planner.step(obs)
            _obs, _reward, terminated, _truncated, info = env.step(_env_action(env, command))
            steps = step_idx + 1
            surface_clearances = _surface_clearances(env)
            if surface_clearances.size:
                clearances.append(float(np.min(surface_clearances)))
            meta = info.get("meta", {}) if isinstance(info, dict) else {}
            near_misses += int(float(meta.get("near_misses", 0.0) or 0.0) > 0.0)
            comfort_exposure.append(float(meta.get("comfort_exposure", 0.0) or 0.0))
            if terminated:
                break
    finally:
        close = getattr(planner, "close", None)
        if callable(close):
            close()
        env.close()

    meta = info.get("meta", {}) if isinstance(info, dict) else {}
    collision = bool(info.get("collision", False)) if isinstance(info, dict) else False
    route_success = (
        bool(info.get("is_success", False) or info.get("success", False))
        if isinstance(info, dict)
        else False
    )
    success = route_success and not collision
    min_clearance = min(clearances) if clearances else None
    return {
        "variant": variant.key,
        "axis": variant.axis,
        "variant_source_key": variant.source_key,
        "baseline_variant": variant.baseline,
        "runtime_binding": variant.runtime_binding,
        "planner": planner_name,
        "scenario_id": str(scenario.get("name") or scenario.get("scenario_id") or "unknown"),
        "seed": int(seed),
        "steps": int(steps),
        "terminated": bool(terminated),
        "success": success,
        "route_success": route_success,
        "collision": collision,
        "metrics": {
            "success_rate": 1.0 if success else 0.0,
            "collision_rate": 1.0 if collision else 0.0,
            "min_clearance": min_clearance,
            "mean_clearance": float(np.mean(clearances)) if clearances else None,
            "near_miss_rate": near_misses / float(max(1, steps)),
            "comfort_exposure_mean": float(np.mean(comfort_exposure)) if comfort_exposure else 0.0,
            "time_to_goal_norm": steps / float(max(1, max_steps)),
        },
        "terminal_meta": {
            "is_route_complete": bool(meta.get("is_route_complete", False)),
            "is_timesteps_exceeded": bool(meta.get("is_timesteps_exceeded", False)),
            "distance_to_goal": _finite_or_none(meta.get("distance_to_goal")),
        },
    }


def _stable_seed(seed: int, *parts: str) -> int:
    text = "::".join((str(seed), *parts))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate episode rows into variant/planner metric means."""
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        variant = str(row["variant"])
        planner = str(row["planner"])
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        bucket = summary.setdefault(variant, {}).setdefault(planner, {})
        counts = summary.setdefault(f"{variant}__counts", {}).setdefault(planner, {})
        for metric in METRICS:
            value = _finite_or_none(metrics.get(metric))
            if value is None:
                continue
            bucket[metric] = bucket.get(metric, 0.0) + value
            counts[metric] = counts.get(metric, 0.0) + 1.0
    for variant in list(summary):
        if variant.endswith("__counts"):
            continue
        count_variant = summary.get(f"{variant}__counts", {})
        for planner, metrics in summary[variant].items():
            for metric, total in list(metrics.items()):
                count = count_variant.get(planner, {}).get(metric, 0.0)
                metrics[metric] = total / count if count else 0.0
    for key in [key for key in summary if key.endswith("__counts")]:
        del summary[key]
    return summary


def build_report(
    *,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    variants: Sequence[VariantSpec],
    scenario_set: str,
    horizon: int,
    raw_rows_path: pathlib.Path,
    git_provenance: Mapping[str, Any],
    date: str,
) -> dict[str, Any]:
    """Build compact JSON evidence from actual episode rows."""
    aggregates = aggregate_rows(rows)
    baseline_variant = next(variant.key for variant in variants if variant.baseline)
    axis_tables = {
        variant.key: aggregates[variant.key]
        for variant in variants
        if not variant.baseline and variant.key in aggregates
    }
    rank_report = analyze_fidelity_sensitivity(
        aggregates[baseline_variant],
        axis_tables,
        primary_metric="success_rate",
        drift_metrics=METRICS,
    ).to_dict()
    all_success_rates = [
        float(metrics.get("success_rate", 0.0))
        for by_planner in aggregates.values()
        for metrics in by_planner.values()
    ]
    all_collision_rates = [
        float(metrics.get("collision_rate", 0.0))
        for by_planner in aggregates.values()
        for metrics in by_planner.values()
    ]
    result_caveats = [
        "ranking_stability_is_on_bounded_two_planner_slice_only",
        "full_fixed_scope_planners_not_run",
    ]
    if not rank_report.get("rank_identifiable", True):
        reason = str(rank_report.get("rank_identifiability_reason") or "unknown")
        result_caveats.append(f"rank_non_identifiable_{reason}")
    if all_success_rates and max(all_success_rates) <= 0.0:
        result_caveats.append("all_observed_success_rates_zero")
    if all_collision_rates and min(all_collision_rates) >= 1.0:
        result_caveats.append("all_observed_collision_rates_one")
    elif all_collision_rates and max(all_collision_rates) >= 1.0:
        result_caveats.append("some_observed_collision_rates_one")
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 3207,
        "status": "actual_campaign_slice",
        "date": date,
        **git_provenance,
        "claim_boundary": CLAIM_BOUNDARY,
        "prior_smoke_boundary": DIAGNOSTIC_SMOKE_CLAIM_BOUNDARY,
        "config_path": DEFAULT_CONFIG,
        "scenario_set": scenario_set,
        "raw_rows_path": str(_repo_rel(raw_rows_path)).replace("output/", "ignored_output/"),
        "raw_output_policy": "raw JSONL remains ignored under ignored_output/",
        "study_id": str(config.get("study_id", "issue_3207_fidelity_sensitivity_v1")),
        "scope": {
            "classification": "bounded_actual_slice",
            "scenario_count": len({str(row["scenario_id"]) for row in rows}),
            "seeds": sorted({int(row["seed"]) for row in rows}),
            "planners": sorted({str(row["planner"]) for row in rows}),
            "horizon": int(horizon),
            "episode_count": len(rows),
            "not_full_fixed_scope_reason": (
                "local torch/rvo2-independent slice uses two non-learned planners; "
                "full config fixed_scope planners remain future work"
            ),
        },
        "variants": [
            {
                "axis": variant.axis,
                "variant": variant.key,
                "source_key": variant.source_key,
                "baseline": variant.baseline,
                "runtime_binding": variant.runtime_binding,
            }
            for variant in variants
        ],
        "aggregates": aggregates,
        "rank_stability": rank_report,
        "result_caveats": result_caveats,
    }


def format_markdown(report: Mapping[str, Any]) -> str:
    """Render compact evidence Markdown."""
    rank_stability = report["rank_stability"]
    rank_identifiable = bool(rank_stability.get("rank_identifiable", True))
    rank_reason = str(rank_stability.get("rank_identifiability_reason") or "none")
    rank_stable = rank_stability.get("rank_stable")
    rank_stable_text = "not_applicable" if rank_stable is None else str(rank_stable)
    rank_status = "identifiable" if rank_identifiable else "non-identifiable"
    nominal_label = (
        "Nominal ranking"
        if rank_identifiable
        else "Nominal deterministic order (ties broken by name)"
    )
    lines = [
        f"# Issue #3207 Fidelity Sensitivity Actual Slice {report['date']}",
        "",
        f"- Status: `{report['status']}`",
        f"- Evidence classification: `{report['scope']['classification']}`",
        f"- Git head: `{report['git_head']}`",
        f"- Git worktree dirty at generation: `{report['git_worktree_dirty']}`",
        f"- Raw rows: `{report['raw_rows_path']}`",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Scope",
        "",
        f"- Scenario set: `{report['scenario_set']}`",
        f"- Episodes: `{report['scope']['episode_count']}`",
        f"- Horizon: `{report['scope']['horizon']}`",
        f"- Seeds: `{', '.join(str(seed) for seed in report['scope']['seeds'])}`",
        f"- Planners: `{', '.join(report['scope']['planners'])}`",
        f"- Limitation: {report['scope']['not_full_fixed_scope_reason']}.",
        f"- Result caveats: `{', '.join(report['result_caveats'])}`",
        "",
        "## Rank Stability",
        "",
        f"- {nominal_label}: `{', '.join(rank_stability['nominal_ranking'])}`",
        f"- Rank evidence status: `{rank_status}`",
        f"- Rank identifiability reason: `{rank_reason}`",
        f"- Rank stable on this slice: `{rank_stable_text}`",
        f"- Flipping variants: `{', '.join(rank_stability['flipping_axes']) or 'none'}`",
        f"- Non-identifiable variants: `{', '.join(rank_stability.get('non_identifiable_axes', [])) or 'none'}`",
        "",
        "| Variant | Rank evidence | Kendall tau | Rank flips | Top-1 changed |",
        "|---|---|---:|---:|---|",
    ]
    for axis in rank_stability["axes"]:
        axis_identifiable = bool(axis.get("rank_identifiable", True))
        axis_reason = str(axis.get("rank_identifiability_reason") or "none")
        axis_status = "identifiable" if axis_identifiable else f"non-identifiable: {axis_reason}"
        tau = axis.get("kendall_tau")
        flips = axis.get("rank_flips")
        top1_changed = axis.get("top1_changed")
        tau_text = "NA" if tau is None else f"{float(tau):.6g}"
        flips_text = "NA" if flips is None else str(int(flips))
        top1_text = "NA" if top1_changed is None else f"`{top1_changed}`"
        lines.append(
            f"| `{axis['axis']}` | `{axis_status}` | {tau_text} | {flips_text} | {top1_text} |"
        )
    lines.extend(
        [
            "",
            "This evidence measures internal simulator-fidelity sensitivity for the bounded local slice only.",
            "It must not be cited as simulator-realism, sim-to-real, full benchmark, or paper-facing ranking evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    *,
    rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    raw_root: pathlib.Path,
    evidence_dir: pathlib.Path,
) -> None:
    """Write raw ignored rows and compact tracked evidence."""
    raw_root.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rows_path = raw_root / "episode_rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (evidence_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "README.md").write_text(format_markdown(report), encoding="utf-8")
    with (evidence_dir / "planner_variant_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "planner", *METRICS],
            lineterminator="\n",
        )
        writer.writeheader()
        for variant, by_planner in sorted(report["aggregates"].items()):
            for planner, metrics in sorted(by_planner.items()):
                writer.writerow({"variant": variant, "planner": planner, **metrics})
    # Standalone post-run contract artifact: rank-identifiability report.
    rank_report = report.get("rank_stability")
    if isinstance(rank_report, Mapping):
        write_rank_identifiability_report(rank_report, evidence_dir)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--scenario-set", default=DEFAULT_SCENARIO_SET)
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--horizon", type=int, default=180)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument(
        "--first-variant-per-axis-only",
        action="store_true",
        help="Run only the baseline plus the first runtime-bound perturbation per fidelity axis.",
    )
    parser.add_argument(
        "--fixed-scope-plan-only",
        action="store_true",
        help=(
            "Consume the #3207 fixed-scope preflight and enumerate the full run plan "
            "(planner_group x axis-variant x seed) without running any episode."
        ),
    )
    parser.add_argument(
        "--plan-out",
        default="output/fidelity_sensitivity/issue_3207_fixed_scope_run_plan",
        help="Directory for the emitted fixed-scope run-plan JSON (gitignored output).",
    )
    parser.add_argument(
        "--require-launchable",
        action="store_true",
        help=(
            "With --fixed-scope-plan-only, exit non-zero (fail closed) if any launch "
            "prerequisite remains. The fixed-scope packet records residual gates such "
            "as ORCA/rvo2 runtime availability."
        ),
    )
    parser.add_argument(
        "--fixed-scope-execute",
        action="store_true",
        help=(
            "Explicit opt-in to drive per-cell episode execution from the fixed-scope run "
            "plan. Fails closed via ensure_fixed_scope_launchable unless every launch gate is "
            "cleared, so on the shipped config it runs zero episodes. When launchable, each "
            "plan cell is bound to concrete runner inputs (planner, variant, seed) with no "
            "silent fallback for unbound (ORCA/hybrid) planners."
        ),
    )
    parser.add_argument("--date", default=dt.datetime.now(tz=dt.UTC).date().isoformat())
    return parser.parse_args(argv)


def _run_fixed_scope_plan_only(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    """Emit the fixed-scope run plan without running any episode.

    Returns:
        ``0`` on success, or ``1`` when ``--require-launchable`` is set and the
        plan still has unmet launch gates (fail-closed).
    """
    plan = build_fixed_scope_run_plan(
        config,
        config_path=args.config,
        git_head=_git_head(),
        date=str(args.date),
    )
    plan_path = write_fixed_scope_run_plan(plan, REPO_ROOT / args.plan_out)
    print(f"wrote fixed-scope run plan: {_repo_rel(plan_path)}")
    print(
        f"preflight_decision={plan['preflight_decision']} executable={plan['executable']} "
        f"launched={plan['launched']} run_cells={plan['run_cell_count']}"
    )
    for reason in plan["gate_reasons"]:
        print(f"  launch gate: {reason}")
    if args.require_launchable:
        try:
            ensure_fixed_scope_launchable(plan)
        except FixedScopeNotLaunchableError as exc:
            print(f"fail-closed: {exc}")
            return 1
    return 0


def _default_fixed_scope_cell_runner(
    *,
    scenarios: Sequence[Mapping[str, Any]],
    scenario_path: pathlib.Path,
    horizon: int,
) -> Callable[[RunnerCellBinding], list[dict[str, Any]]]:
    """Build the real per-cell runner: one episode per scenario for a bound cell.

    Only reached after :func:`ensure_fixed_scope_launchable` passes, so every
    binding here is guaranteed ``runner_bound`` with a concrete ``planner_name``.
    """

    def _run(binding: RunnerCellBinding) -> list[dict[str, Any]]:
        assert binding.planner_name is not None  # guaranteed by execute guard
        return [
            run_episode(
                scenario,
                scenario_path=scenario_path,
                variant=binding.variant,
                planner_name=binding.planner_name,
                seed=int(binding.cell.seed),
                horizon=horizon,
            )
            for scenario in scenarios
        ]

    return _run


def _run_fixed_scope_execute(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    """Drive per-cell episode execution from the fixed-scope plan, fail-closed.

    Returns:
        ``1`` (fail-closed, zero episodes) while any launch gate remains — the
        state on the shipped config — or ``0`` after executing every bound cell
        when the plan is fully launchable.
    """
    plan = build_fixed_scope_run_plan(
        config,
        config_path=args.config,
        git_head=_git_head(),
        date=str(args.date),
    )
    try:
        ensure_fixed_scope_launchable(plan)
    except FixedScopeNotLaunchableError as exc:
        print(f"fail-closed: {exc}")
        print(
            "no episodes executed; fixed-scope per-cell execution stays gated behind "
            "launch prerequisites (ORCA/rvo2, hybrid opt-in, runtime rank-identifiability recheck)"
        )
        return 1

    # Reached only when every launch gate is cleared (not on the shipped config).
    bindings = bind_fixed_scope_run_plan(plan, config)
    scenario_set = str(plan["materialized_scope"]["scenario_set"])
    scenario_path = REPO_ROOT / scenario_set
    scenarios = list(load_scenarios(scenario_path))
    if not scenarios:
        raise ValueError(f"fixed-scope scenario set produced no scenarios: {scenario_path}")
    rows = execute_fixed_scope_cells(
        bindings,
        cell_runner=_default_fixed_scope_cell_runner(
            scenarios=scenarios,
            scenario_path=scenario_path,
            horizon=int(args.horizon),
        ),
    )
    raw_root = REPO_ROOT / args.raw_root
    raw_rows_path = raw_root / "episode_rows.jsonl"
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = build_report(
        config=config,
        rows=rows,
        variants=list(build_fixed_scope_variant_index(config).values()),
        scenario_set=scenario_set,
        horizon=int(args.horizon),
        raw_rows_path=raw_rows_path,
        git_provenance=_git_provenance(),
        date=str(args.date),
    )
    write_outputs(
        rows=rows,
        report=report,
        raw_root=raw_root,
        evidence_dir=REPO_ROOT / args.evidence_dir,
    )
    # Post-run contract gate: validate rank-identifiability against plan spec.
    contract_specs = plan.get("post_run_contract_specs") or []
    rank_contract_spec = next(
        (
            spec
            for spec in contract_specs
            if spec.get("id") == "runtime_rank_identifiability_recheck"
        ),
        None,
    )
    if rank_contract_spec is not None:
        contract_result = check_rank_identifiability_contract(
            report["rank_stability"], rank_contract_spec
        )
        if not contract_result.satisfied:
            print(
                f"fail-closed: post-run contract '{contract_result.contract_id}' failed: "
                f"{contract_result.reason}"
            )
            return 1
    elif not bool(report["rank_stability"].get("rank_identifiable")):
        reason = report["rank_stability"].get("rank_identifiability_reason", "unknown")
        print(f"fail-closed: post-run rank identifiability recheck failed: {reason}")
        return 1
    print(f"executed {len(rows)} fixed-scope episode rows across {len(bindings)} cells")
    print(f"wrote raw rows: {_repo_rel(raw_rows_path)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded actual campaign."""
    args = _parse_args(argv)
    config_path = REPO_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, Mapping):
        raise ValueError(f"config must be a mapping: {config_path}")
    if args.fixed_scope_plan_only:
        # Plan-consumption mode: enumerate the full fixed-scope run plan and stop
        # before any episode. Execution stays fail-closed behind launch gates.
        return _run_fixed_scope_plan_only(args, config)
    if args.fixed_scope_execute:
        # Explicit opt-in: bind the plan to concrete per-cell runner inputs and
        # execute, but only after ensure_fixed_scope_launchable passes. On the
        # shipped config this fails closed and runs zero episodes.
        return _run_fixed_scope_execute(args, config)
    scenario_path = REPO_ROOT / args.scenario_set
    scenarios = list(load_scenarios(scenario_path))
    if not scenarios:
        raise ValueError(f"scenario set produced no scenarios: {scenario_path}")
    scenario = scenarios[0]
    seeds = tuple(args.seeds or config.get("fixed_scope", {}).get("seeds", [111, 112, 113]))
    variants = load_variant_specs(
        config,
        include_all_variants=not bool(args.first_variant_per_axis_only),
    )
    raw_root = REPO_ROOT / args.raw_root
    raw_rows_path = raw_root / "episode_rows.jsonl"

    rows = [
        run_episode(
            scenario,
            scenario_path=scenario_path,
            variant=variant,
            planner_name=planner,
            seed=int(seed),
            horizon=int(args.horizon),
        )
        for variant in variants
        for planner in PLANNERS
        for seed in seeds
    ]
    report = build_report(
        config=config,
        rows=rows,
        variants=variants,
        scenario_set=args.scenario_set,
        horizon=int(args.horizon),
        raw_rows_path=raw_rows_path,
        git_provenance=_git_provenance(),
        date=str(args.date),
    )
    write_outputs(
        rows=rows,
        report=report,
        raw_root=raw_root,
        evidence_dir=REPO_ROOT / args.evidence_dir,
    )
    print(f"wrote raw rows: {_repo_rel(raw_rows_path)}")
    print(f"wrote compact evidence: {args.evidence_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
