#!/usr/bin/env python3
"""Episode-level ScenarioBelief uncertainty -> planner-safety experiment (#3471).

Plain-language summary: PR #3450 showed, in a *single synthetic step*, that ScenarioBelief
uncertainty changes the stream_gap planner's decision. This is the bounded **episode-level**
follow-up it named: roll a controlled crossing scenario over many timesteps with the real
``StreamGapPlannerAdapter`` driving the robot, and measure whether *dropping* uncertain agents
produces unsafe ``COMMIT`` behavior compared with *retaining* them under conservative handling.

Three belief modes share the identical ground-truth scenario; only what the planner is allowed to
trust differs:

* ``oracle`` -- certain belief, planner reacts to every agent (safe baseline).
* ``uncertain_retained`` -- the corridor agent's existence confidence is degraded, but the planner's
  uncertainty gate is OFF, so it fail-closed *keeps* the uncertain agent (conservative).
* ``uncertain_dropped`` -- same degraded belief, planner gate ON, so the low-confidence corridor
  agent is dropped from gap reasoning. The hypothesis: this raises unsafe commitment.

Ground truth (used only for safety scoring, never shown to the planner in the dropped mode) is the
true simulated pedestrian position. The result is classified ``revise`` / ``continue`` / ``stop`` /
``inconclusive`` and is **diagnostic-tier, controlled-scenario evidence** -- not the full benchmark
environment, not paper-grade, no traffic-realism or trained-policy claim.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.planner.scenario_belief_adapter import project_scenario_belief_for_planner
from robot_sf.planner.stream_gap import StreamGapPlannerAdapter, StreamGapPlannerConfig
from robot_sf.representation import Estimate2D, scenario_belief_from_simulator_oracle

SCHEMA_VERSION = "scenario-belief-episode-safety.v1"
ISSUE = 3471
PLANNER_KEY = "stream_gap"
_CORRIDOR_AGENT_INDEX = 0

#: Belief modes -> (degrade the corridor agent's existence?, planner uncertainty gating on?).
MODES: dict[str, dict[str, bool]] = {
    "oracle": {"degrade": False, "gate": False},
    "uncertain_retained": {"degrade": True, "gate": False},
    "uncertain_dropped": {"degrade": True, "gate": True},
}

#: Existence confidence assigned to the degraded corridor agent (below the 0.5 gate threshold).
_DEGRADED_EXISTENCE = 0.2

#: Position confidence assigned to the conformal-radius proxy (below the 0.5 gate threshold).
_DEGRADED_POSITION_CONFIDENCE = 0.2

#: Position variance assigned to the inflated-envelope proxy (above the 1.0 gate threshold).
_INFLATED_POSITION_VARIANCE = 4.0

UNCERTAINTY_REPRESENTATIONS: dict[str, dict[str, str]] = {
    "belief_drop": {
        "gate_field": "uncertainty_min_existence_probability",
        "description": "lower corridor-agent existence probability below the stream_gap gate",
    },
    "conformal_radius": {
        "gate_field": "uncertainty_min_position_confidence",
        "description": (
            "lower corridor-agent position confidence as a controlled episode-level "
            "conformal-radius proxy"
        ),
    },
    "envelope_inflation": {
        "gate_field": "uncertainty_max_position_variance",
        "description": "inflate corridor-agent position variance above the stream_gap envelope gate",
    },
}

#: The four stream_gap uncertainty-gate thresholds the #3558 sweep is allowed to override.
GATE_THRESHOLD_FIELDS = (
    "uncertainty_min_existence_probability",
    "uncertainty_min_position_confidence",
    "uncertainty_min_class_probability",
    "uncertainty_max_position_variance",
)


@dataclass
class EpisodeParams:
    """Controlled crossing-scenario parameters (frozen for reproducibility).

    Geometry lives in positive map coordinates because ``ScenarioBelief.to_socnav_struct``
    clips pedestrian positions to the non-negative map frame.
    """

    max_steps: int = 120
    dt: float = 0.1
    start_x: float = 2.0
    path_y: float = 5.0
    goal_x: float = 9.0
    corridor_x: float = 5.0
    robot_radius: float = 0.4
    ped_radius: float = 0.3
    near_miss_margin: float = 0.6
    ped_cross_speed: float = 0.7


@dataclass
class ScenarioState:
    """Mutable ground-truth state advanced each timestep."""

    robot_pos: np.ndarray
    robot_heading: float
    ped_pos: np.ndarray
    ped_vel: np.ndarray
    goal: np.ndarray


def build_initial_state(seed: int, params: EpisodeParams) -> ScenarioState:
    """Build the initial ground-truth state; ``seed`` perturbs the crosser's phase and speed."""
    rng = np.random.default_rng(seed)
    # Corridor pedestrian crosses the robot's path (path_y, +y motion) near corridor_x; a distractor
    # sits off-corridor. The crosser starts below the path and is timed to contest the crossing point.
    y_start = params.path_y - float(rng.uniform(2.0, 3.0))
    cross_speed = params.ped_cross_speed * float(rng.uniform(0.85, 1.15))
    ped_pos = np.array(
        [[params.corridor_x, y_start], [params.corridor_x + 3.0, params.path_y + 3.0]],
        dtype=np.float32,
    )
    ped_vel = np.array([[0.0, cross_speed], [0.0, 0.0]], dtype=np.float32)
    return ScenarioState(
        robot_pos=np.array([params.start_x, params.path_y], dtype=np.float32),
        robot_heading=0.0,
        ped_pos=ped_pos,
        ped_vel=ped_vel,
        goal=np.array([params.goal_x, params.path_y], dtype=np.float32),
    )


def _make_simulator(state: ScenarioState, params: EpisodeParams) -> SimpleNamespace:
    """Build the lightweight simulator stand-in consumed by the belief constructor."""
    return SimpleNamespace(
        ped_pos=np.asarray(state.ped_pos, dtype=np.float32),
        ped_vel=np.asarray(state.ped_vel, dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=(
                    (float(state.robot_pos[0]), float(state.robot_pos[1])),
                    float(state.robot_heading),
                ),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=params.robot_radius),
            )
        ],
        goal_pos=[np.asarray(state.goal, dtype=np.float32)],
        # Use the real goal as the next goal too: a None/zero next-goal is emitted as (0,0) by
        # to_socnav_struct, which the stream_gap planner would steer toward.
        next_goal_pos=[np.asarray(state.goal, dtype=np.float32)],
        map_def=SimpleNamespace(width=14.0, height=10.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=params.dt),
    )


def _degrade_agent_for_representation(agent: Any, uncertainty_representation: str) -> Any:
    """Apply one issue #3557 uncertainty representation to an agent belief."""
    if uncertainty_representation == "belief_drop":
        return replace(agent, existence_probability=_DEGRADED_EXISTENCE)
    if uncertainty_representation == "conformal_radius":
        return replace(
            agent,
            position=replace(agent.position, confidence=_DEGRADED_POSITION_CONFIDENCE),
        )
    if uncertainty_representation == "envelope_inflation":
        return replace(
            agent,
            position=Estimate2D.point(
                agent.position.mean_xy,
                confidence=agent.position.confidence,
                variance=_INFLATED_POSITION_VARIANCE,
                frame_id=agent.position.frame_id,
                units=agent.position.units,
                covariance_units=agent.position.covariance_units,
            ),
        )
    raise ValueError(f"unknown uncertainty representation: {uncertainty_representation}")


def build_belief_for_mode(
    state: ScenarioState,
    mode: str,
    params: EpisodeParams,
    uncertainty_representation: str = "belief_drop",
) -> Any:
    """Build the ScenarioBelief for ``mode`` from the current ground-truth state.

    The dropped/retained modes degrade the corridor agent through the selected uncertainty
    representation; oracle leaves it certain.
    """
    if uncertainty_representation not in UNCERTAINTY_REPRESENTATIONS:
        raise ValueError(f"unknown uncertainty representation: {uncertainty_representation}")
    simulator = _make_simulator(state, params)
    belief = scenario_belief_from_simulator_oracle(
        simulator, env_config=RobotSimulationConfig(), max_pedestrians=4
    )
    if not MODES[mode]["degrade"]:
        return belief
    agents = list(belief.agents)
    if not agents:
        return belief
    # Degrade the agent nearest the true corridor pedestrian so matching survives any reordering.
    corridor_true = np.asarray(state.ped_pos[_CORRIDOR_AGENT_INDEX], dtype=float)
    idx = min(
        range(len(agents)),
        key=lambda i: float(
            np.linalg.norm(np.asarray(agents[i].position.mean_xy, dtype=float) - corridor_true)
        ),
    )
    agents[idx] = _degrade_agent_for_representation(agents[idx], uncertainty_representation)
    return replace(belief, agents=tuple(agents))


def _planner_config(
    mode: str, gate_thresholds: dict[str, float] | None = None
) -> StreamGapPlannerConfig:
    """Return a stream_gap config with uncertainty gating set per ``mode``.

    ``gate_thresholds`` optionally overrides one or more of the four uncertainty-gate
    thresholds (``GATE_THRESHOLD_FIELDS``) so the #3558 calibration sweep can probe
    whether any threshold setting makes dropping at least as safe as conservative
    retention. Unknown keys fail closed rather than being silently ignored.
    """
    overrides = dict(gate_thresholds or {})
    unknown = set(overrides) - set(GATE_THRESHOLD_FIELDS)
    if unknown:
        raise ValueError(f"unknown gate threshold override(s): {sorted(unknown)}")
    return StreamGapPlannerConfig(uncertainty_gating_enabled=MODES[mode]["gate"], **overrides)


def _advance(
    state: ScenarioState, speed: float, angular: float, params: EpisodeParams
) -> ScenarioState:
    """Advance robot (unicycle) and pedestrians one timestep."""
    heading = state.robot_heading + angular * params.dt
    robot_pos = (
        state.robot_pos
        + speed * np.array([np.cos(heading), np.sin(heading)], dtype=np.float32) * params.dt
    )
    ped_pos = state.ped_pos + state.ped_vel * params.dt
    return replace(
        state,
        robot_pos=robot_pos.astype(np.float32),
        robot_heading=float(heading),
        ped_pos=ped_pos.astype(np.float32),
    )


def min_separation(state: ScenarioState, params: EpisodeParams) -> float:
    """Return the true minimum surface separation between robot and any pedestrian."""
    deltas = np.asarray(state.ped_pos, dtype=float) - np.asarray(state.robot_pos, dtype=float)
    center = float(np.min(np.linalg.norm(deltas, axis=1)))
    return center - params.robot_radius - params.ped_radius


def _is_commit(speed: float) -> bool:
    """A stream_gap command is a COMMIT when it returns the configured commit speed."""
    return abs(speed - StreamGapPlannerConfig.commit_speed) < 1e-6


def run_episode(
    mode: str,
    seed: int,
    params: EpisodeParams,
    gate_thresholds: dict[str, float] | None = None,
    uncertainty_representation: str = "belief_drop",
) -> dict[str, Any]:
    """Roll one controlled episode under ``mode`` and return episode-level safety metrics.

    ``gate_thresholds`` optionally overrides the uncertainty-gate thresholds (only meaningful
    when the mode enables gating); see :func:`_planner_config`.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    state = build_initial_state(seed, params)
    planner = StreamGapPlannerAdapter(_planner_config(mode, gate_thresholds))

    start = time.perf_counter()
    initial_goal_dist = float(np.linalg.norm(state.goal - state.robot_pos))
    min_sep = float("inf")
    collision = False
    near_miss_steps = 0
    commit_steps = 0
    unsafe_commit_steps = 0
    first_yield_step: int | None = None
    reached_goal = False
    uncertainty_consumed = False
    fail_closed = False

    for step in range(params.max_steps):
        belief = build_belief_for_mode(state, mode, params, uncertainty_representation)
        projection = project_scenario_belief_for_planner(belief, planner_key=PLANNER_KEY)
        status = projection.compatibility.get("status")
        if status == "fail_closed":
            fail_closed = True
        uncertainty_consumed = uncertainty_consumed or bool(
            projection.compatibility.get("uncertainty_consumed")
        )
        speed, angular = planner.plan(projection.observation)

        committed = _is_commit(speed)
        if committed:
            commit_steps += 1
        elif speed <= StreamGapPlannerConfig.creep_speed and first_yield_step is None:
            first_yield_step = step

        state = _advance(state, speed, angular, params)
        sep = min_separation(state, params)
        min_sep = min(min_sep, sep)
        if sep <= 0.0:
            collision = True
        elif sep <= params.near_miss_margin:
            near_miss_steps += 1
        # Unsafe commit: planner committed while a true pedestrian is within the near-miss band.
        if committed and sep <= params.near_miss_margin:
            unsafe_commit_steps += 1

        if (
            float(np.linalg.norm(state.goal - state.robot_pos))
            <= StreamGapPlannerConfig.goal_tolerance
        ):
            reached_goal = True
            break

    final_goal_dist = float(np.linalg.norm(state.goal - state.robot_pos))
    progress = max(0.0, (initial_goal_dist - final_goal_dist) / initial_goal_dist)
    return {
        "mode": mode,
        "uncertainty_representation": uncertainty_representation,
        "seed": seed,
        "collision": collision,
        "min_separation": round(min_sep, 4),
        "near_miss_steps": near_miss_steps,
        "commit_steps": commit_steps,
        "unsafe_commit_steps": unsafe_commit_steps,
        "first_yield_step": first_yield_step,
        "reached_goal": reached_goal,
        "progress": round(progress, 4),
        "runtime_steps": step + 1,
        "uncertainty_consumed": uncertainty_consumed,
        "fail_closed": fail_closed,
        "runtime_sec": round(time.perf_counter() - start, 4),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed episode rows for one mode."""
    n = len(rows)
    return {
        "episodes": n,
        "collision_rate": round(sum(r["collision"] for r in rows) / n, 4),
        "mean_min_separation": round(float(np.mean([r["min_separation"] for r in rows])), 4),
        "worst_min_separation": round(min(r["min_separation"] for r in rows), 4),
        "total_unsafe_commit_steps": sum(r["unsafe_commit_steps"] for r in rows),
        "mean_progress": round(float(np.mean([r["progress"] for r in rows])), 4),
        "reached_goal_rate": round(sum(r["reached_goal"] for r in rows) / n, 4),
        "uncertainty_consumed_any": any(r["uncertainty_consumed"] for r in rows),
        "fail_closed_any": any(r["fail_closed"] for r in rows),
    }


def classify_decision(by_mode: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Classify the dropped-vs-retained safety contrast into a maintainer-facing decision."""
    retained = by_mode.get("uncertain_retained")
    dropped = by_mode.get("uncertain_dropped")
    if not retained or not dropped:
        return {"decision": "blocked", "reason": "missing required modes"}
    if not dropped["uncertainty_consumed_any"]:
        return {
            "decision": "inconclusive",
            "reason": "dropped mode never consumed the uncertainty sidecar; gate not exercised",
        }
    unsafe_delta = dropped["total_unsafe_commit_steps"] - retained["total_unsafe_commit_steps"]
    sep_delta = retained["worst_min_separation"] - dropped["worst_min_separation"]
    if unsafe_delta > 0 or dropped["collision_rate"] > retained["collision_rate"]:
        decision = "revise"
        reason = (
            f"dropping uncertain agents increased unsafe commitment "
            f"(unsafe_commit_steps +{unsafe_delta}, worst_min_sep delta {round(sep_delta, 3)}m); "
            "the uncertainty-dropping default should be revised/blocked for safety-relevant use"
        )
    elif unsafe_delta == 0 and abs(sep_delta) < 1e-6:
        decision = "inconclusive"
        reason = "no measurable safety difference between retained and dropped at this matrix"
    else:
        decision = "continue"
        reason = "dropping uncertain agents did not increase unsafe commitment in this matrix"
    return {
        "decision": decision,
        "reason": reason,
        "unsafe_commit_delta_dropped_minus_retained": unsafe_delta,
        "worst_min_sep_delta_retained_minus_dropped": round(sep_delta, 4),
    }


def run_matrix(
    seeds: list[int],
    params: EpisodeParams,
    uncertainty_representation: str = "belief_drop",
) -> dict[str, Any]:
    """Run all modes across the seed matrix and assemble the classified report."""
    if uncertainty_representation not in UNCERTAINTY_REPRESENTATIONS:
        raise ValueError(f"unknown uncertainty representation: {uncertainty_representation}")
    episodes: dict[str, list[dict[str, Any]]] = {}
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        rows = [
            run_episode(
                mode,
                seed,
                params,
                uncertainty_representation=uncertainty_representation,
            )
            for seed in seeds
        ]
        episodes[mode] = rows
        by_mode[mode] = _aggregate(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "followup_issue": 3557,
        "evidence_tier": "diagnostic",
        "uncertainty_representation": uncertainty_representation,
        "uncertainty_representation_contract": {
            "available_representations": sorted(UNCERTAINTY_REPRESENTATIONS),
            "selected": UNCERTAINTY_REPRESENTATIONS[uncertainty_representation],
            "claim_boundary": (
                "Harness-level representation parameterization only; each run remains a "
                "controlled #3471 episode contrast and is not a cross-representation "
                "generalization claim."
            ),
        },
        "claim_boundary": (
            "Controlled crossing scenario with the real stream_gap planner + ScenarioBelief "
            "uncertainty gate; not the full benchmark environment, not paper-grade, no trained "
            "policy or traffic-realism claim."
        ),
        "seeds": seeds,
        "seed_count": len(seeds),
        "params": vars(params),
        "by_mode": by_mode,
        "decision": classify_decision(by_mode),
        "episodes": episodes,
    }


def load_config(path: Path) -> tuple[list[int], EpisodeParams]:
    """Load the predeclared seed matrix and episode params from a YAML config."""
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    seeds = [int(s) for s in data.get("seeds", range(101, 113))]
    param_fields = {f.name for f in EpisodeParams.__dataclass_fields__.values()}
    overrides = {k: v for k, v in (data.get("params") or {}).items() if k in param_fields}
    return seeds, EpisodeParams(**overrides)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=None, help="Predeclared scenario/seed config."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--uncertainty-representation",
        choices=sorted(UNCERTAINTY_REPRESENTATIONS),
        default="belief_drop",
        help=(
            "Issue #3557 harness parameter. Selects the ScenarioBelief uncertainty "
            "representation applied to the corridor agent."
        ),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the matrix and emit the classified report."""
    args = parse_args(argv)
    if args.config is not None:
        seeds, params = load_config(args.config)
    else:
        seeds, params = list(range(101, 113)), EpisodeParams()
    if args.seeds is not None:
        seeds = args.seeds
    if args.max_steps is not None:
        params = replace(params, max_steps=args.max_steps)
    report = run_matrix(
        seeds,
        params,
        uncertainty_representation=args.uncertainty_representation,
    )
    report["generated_at_utc"] = datetime.now(UTC).isoformat()

    compact = {k: v for k, v in report.items() if k != "episodes"}
    print(json.dumps(compact, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
