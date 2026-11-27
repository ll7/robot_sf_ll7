"""Record and report the opt-in doorway witness and matched solo controls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml
from shapely.geometry import LineString, Point, Polygon

from robot_sf.benchmark.map_runner.map_runner import _run_map_episode
from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.benchmark.narrow_doorway_radius_audit import derive_doorway_geometry
from robot_sf.benchmark.seed_bands import DEV_SEEDS
from robot_sf.common.seed import set_global_seed
from robot_sf.scenario_certification.feasibility_diagnostics import make_actor_free_scenario
from robot_sf.sim.simulator import Simulator
from robot_sf.training.scenario_loader import load_scenarios

Opening = tuple[float, float, float, float]
Frame = dict[str, Any]


def _trace_actor(frame: Frame, expected_actors: tuple[bool, bool]) -> dict | None:
    """Read the actor inventory for one frame.

    Returns:
        The one pedestrian, when present.
    """
    robot, pedestrian = expected_actors
    if (frame.get("robot") is not None) != robot or len(frame["pedestrians"]) != int(pedestrian):
        raise ValueError("unexpected actor count")
    return frame["pedestrians"][0] if pedestrian else None


def _trace_tracks(
    frames: list[Frame], dt: float, expected_actors: tuple[bool, bool]
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Validate cadence, actor count and pedestrian identity.

    Returns:
        Synchronized robot and pedestrian center tracks, empty for absent actors.
    """
    if len(frames) < 2:
        raise ValueError("trace needs reset and at least one post-reset sample")
    reset_ped = _trace_actor(frames[0], expected_actors)
    actor_id = reset_ped["id"] if reset_ped else None
    if expected_actors[1] and (not isinstance(actor_id, str) or not actor_id):
        raise ValueError("missing stable pedestrian actor ID")
    robots, pedestrians = [], []
    for index, frame in enumerate(frames):
        if not math.isclose(frame["time_s"], index * dt, abs_tol=1e-9):
            raise ValueError("trace time is not synchronized with the simulation step")
        ped = _trace_actor(frame, expected_actors)
        if ped is not None:
            if ped["id"] != actor_id:
                raise ValueError("pedestrian actor ID changed")
            pedestrians.append(tuple(ped["position"]))
        if expected_actors[0]:
            robots.append(tuple(frame["robot"]["position"]))
    if not all(len(xy) == 2 and all(map(math.isfinite, xy)) for xy in robots + pedestrians):
        raise ValueError("invalid actor position")
    return robots, pedestrians


def _crossing(
    positions: list[tuple[float, float]], opening: Opening, direction: int
) -> tuple[int | None, int | None]:
    """Find entry and exit on opposite sides of the throat.

    Returns:
        The indices of the first entry and first opposite-side exit.
    """
    west, east, low, high = opening
    if not positions:
        return None, None
    start_x = positions[0][0]
    boundary = west if direction == 1 else east
    if direction * (start_x - boundary) >= 0:
        raise ValueError("invalid actor starting side")
    entry = next(
        (i for i, (x, y) in enumerate(positions) if west <= x <= east and low < y < high),
        None,
    )
    if entry is None:
        return None, None
    opposite = east if direction == 1 else west
    exit_index = next(
        (
            i
            for i in range(entry + 1, len(positions))
            if direction * (positions[i][0] - opposite) > 0
        ),
        None,
    )
    return entry, exit_index


def _approach_encounter(
    robots: list[tuple[float, float]], pedestrians: list[tuple[float, float]], opening: Opening
) -> int | None:
    """Return the first encounter frame before either actor passes the opening."""
    west, east, low, high = opening
    for index, ((rx, ry), (px, py)) in enumerate(zip(robots, pedestrians, strict=False)):
        if rx > east or px < west:
            break
        if px <= east and low < py < high and west - 3.0 <= rx and low < ry < high:
            return index
    return None


def summarize_trace(
    frames: list[Frame],
    *,
    opening: Opening,
    dt: float,
    expected_actors: tuple[bool, bool],
) -> dict[str, Any]:
    """Report crossings, contacts and an encounter at the doorway.

    A pedestrian must reach the throat while the robot is in it or within
    3 m on the west side, aligned with the opening. The robot may pause
    during its approach.

    Returns:
        First event times in seconds, plus contact flags.
    """
    robots, pedestrians = _trace_tracks(frames, dt, expected_actors)
    robot_entry, robot_exit = _crossing(robots, opening, direction=1)
    ped_entry, ped_exit = _crossing(pedestrians, opening, direction=-1)
    encounter = _approach_encounter(robots, pedestrians, opening)
    events = {
        "robot_entry_time_s": robot_entry,
        "robot_exit_time_s": robot_exit,
        "pedestrian_entry_time_s": ped_entry,
        "pedestrian_exit_time_s": ped_exit,
        "encounter_time_s": encounter,
    }
    return {
        **{
            name: frames[index]["time_s"] if index is not None else None
            for name, index in events.items()
        },
        "robot_pedestrian_contact": any(frame["collision"].get("pedestrian") for frame in frames),
        "robot_wall_contact": any(frame["collision"].get("obstacle") for frame in frames),
    }


def _pedestrian_control(config: Any, seed: int, steps: int, dt: float) -> list[Frame]:
    """Advance the same live pedestrian model with an empty robot list.

    Returns:
        Reset and synchronized post-step poses for the sole actor.
    """
    set_global_seed(seed)
    (map_def,) = config.map_pool.map_defs.values()
    sim = Simulator(
        config=config.sim_config,
        map_def=map_def,
        robots=[],
        goal_proximity_threshold=0.0,
        random_start_pos=False,
        peds_have_obstacle_forces=config.peds_have_static_obstacle_forces,
    )
    frames = []
    for index in range(steps + 1):
        if index:
            sim.step_once([])
        (position,) = sim.ped_pos
        frames.append(
            {
                "time_s": index * dt,
                "robot": None,
                "pedestrians": [{"id": "simulator-slot-0", "position": position.tolist()}],
                "collision": {},
            }
        )
    return frames


def _robot_trial(
    scenario: dict, path: Path, seed: int, planner: str, settings: dict
) -> tuple[list[Frame], str, dict]:
    """Use the ordinary benchmark episode and retain the trace poses and collision events.

    Returns:
        Frames, termination reason and recorded effective physics.
    """
    record = _run_map_episode(
        scenario,
        seed,
        horizon=None,
        dt=None,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo=planner,
        algo_config=settings,
        scenario_path=path,
        record_simulation_step_trace=True,
    )
    metadata = record["algorithm_metadata"]
    if metadata["status"] != "ok" or metadata["algorithm"] != planner:
        raise ValueError(f"planner {planner} did not run as requested")
    trace = metadata["simulation_step_trace"]
    frames = [
        {
            "time_s": 0.0,
            "robot": trace["reset"]["robot"],
            "pedestrians": trace["reset"]["pedestrians"],
            "collision": {},
        },
        *trace["steps"],
    ]
    # The map runner appends this provenance field to the episode wire record.
    physics = dict(record)["effective_physics"]
    if not isinstance(physics, dict):
        raise ValueError("invalid effective physics in the episode record")
    return frames, str(record["termination_reason"]), physics


def _wall_clearance(frames: list[Frame], walls: list[Polygon], actor: str, radius: float) -> float:
    """Compute minimum disc-to-wall clearance along the sampled path.

    Returns:
        Separation in metres, using straight segments between centers.
    """
    positions = [
        frame["robot"]["position"] if actor == "robot" else frame["pedestrians"][0]["position"]
        for frame in frames
    ]
    path = Point(positions[0]) if len(positions) == 1 else LineString(positions)
    return min(path.distance(wall) for wall in walls) - radius


def run_proof(config_path: Path, trace_path: Path) -> dict[str, Any]:
    """Run the fixed development proof and retain each trial as one JSONL record.

    Returns:
        The static route clearance and per-trial findings.
    """
    launch = yaml.safe_load(config_path.read_text())
    scenario_path = (config_path.parent / launch["scenario_matrix"]).resolve()
    (loaded_scenario,) = load_scenarios(scenario_path)
    scenario = dict(loaded_scenario)
    seeds = scenario["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in DEV_SEEDS for seed in seeds):
        raise ValueError("the proof requires unique development seeds")
    config = build_env_config(scenario, scenario_path=scenario_path)
    radius = config.robot_config.radius
    ped_radius = config.sim_config.ped_radius
    dt = config.sim_config.time_per_step_in_secs
    steps = scenario["simulation_config"]["max_episode_steps"]
    (map_def,) = config.map_pool.map_defs.values()
    walls = [Polygon(obstacle.vertices) for obstacle in map_def.obstacles]
    geometry = derive_doorway_geometry(scenario_path, scenario)
    door = geometry.obstacle_rects[0]
    opening = (
        door["x"],
        door["x"] + door["width"],
        geometry.gap_lower_edge_m,
        geometry.gap_upper_edge_m,
    )
    clearance = geometry.route_min_center_distance_m - radius
    findings = []
    robot_only = make_actor_free_scenario(scenario)
    with trace_path.open("x") as stream:
        for seed in seeds:
            ped_only = _pedestrian_control(config, seed, steps, dt)
            for planner, settings in launch["planners"].items():
                robot_control_start = None
                for condition, test_scenario, actors in (
                    ("robot_only", robot_only, (True, False)),
                    ("paired", scenario, (True, True)),
                ):
                    frames, termination, physics = _robot_trial(
                        test_scenario, scenario_path, seed, planner, settings
                    )
                    row = _trial_findings(frames, walls, opening, dt, actors, (radius, ped_radius))
                    if condition == "robot_only":
                        robot_control_start = frames[0]["robot"]
                    else:
                        row["matched_robot_reset"] = frames[0]["robot"] == robot_control_start
                        row["matched_pedestrian_reset"] = (
                            frames[0]["pedestrians"][0]["position"]
                            == ped_only[0]["pedestrians"][0]["position"]
                        )
                    row.update(
                        seed=seed, condition=condition, planner=planner, termination=termination
                    )
                    findings.append(row)
                    stream.write(
                        json.dumps({"trial": row, "frames": frames, "physics": physics}) + "\n"
                    )
            row = _trial_findings(ped_only, walls, opening, dt, (False, True), (radius, ped_radius))
            row.update(
                seed=seed, condition="pedestrian_only", planner=None, termination="fixed_horizon"
            )
            findings.append(row)
            stream.write(json.dumps({"trial": row, "frames": ped_only}) + "\n")
    return {
        "opening_width_m": geometry.gap_width_m,
        "route_clearance_m": clearance,
        "robot_radius_m": radius,
        "pedestrian_radius_m": ped_radius,
        "dt_s": dt,
        "trials": findings,
    }


def _trial_findings(
    frames: list[Frame],
    walls: list[Polygon],
    opening: Opening,
    dt: float,
    actors: tuple[bool, bool],
    radii: tuple[float, float],
) -> dict[str, Any]:
    """Summarize the trial and independently check the initial and swept wall clearance.

    Returns:
        Doorway events, clearances and initial-placement validity.
    """
    result = summarize_trace(frames, opening=opening, dt=dt, expected_actors=actors)
    initial_clear = True
    for actor, present, radius in zip(("robot", "pedestrian"), actors, radii, strict=True):
        if present:
            initial_clear &= _wall_clearance(frames[:1], walls, actor, radius) > 0.0
            result[f"{actor}_min_wall_clearance_m"] = _wall_clearance(frames, walls, actor, radius)
    if all(actors):
        start = frames[0]
        distance = math.dist(start["robot"]["position"], start["pedestrians"][0]["position"])
        initial_clear &= distance > sum(radii)
    result["initial_placement_clear"] = initial_clear
    return result


def main() -> int:
    """Write findings for a standalone development run.

    Returns:
        Zero when the report was written. Findings are reported as data.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_proof(args.config, args.traces), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
