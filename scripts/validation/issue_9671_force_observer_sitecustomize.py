"""Opt-in, separately pinned observer for frozen 0.0.7 robot force calls.

Stage this file as ``sitecustomize.py`` and set ISSUE9671_OBSERVER_OUTPUT plus
ISSUE9671_OBSERVER_SHA256. It does not modify the frozen runner or force law.
An episode with ambiguous slot identity fails closed before a sidecar is written.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

FROZEN_SOURCE = "07f7e8d43084de748915e1b1eb8b2a1603357c6e"
FORCE_FILE = "robot_sf/ped_npc/ped_robot_force.py"
SIM_FILE = "robot_sf/sim/simulator.py"
RUNNER_FILE = "robot_sf/benchmark/map_runner/map_runner_episode.py"
SOURCE_MODULES = {
    FORCE_FILE: "robot_sf.ped_npc.ped_robot_force",
    SIM_FILE: "robot_sf.sim.simulator",
    RUNNER_FILE: "robot_sf.benchmark.map_runner.map_runner_episode",
}
FORCE_LINES = {"Simulator": 1699, "PedSimulator": 2087}


class ObserverIdentityError(RuntimeError):
    """A force slot cannot be bound to a stable trace actor."""


def _vectors(value: Any, count: int | None = None) -> list[list[float]]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or (count is not None and len(array) != count):
        raise ObserverIdentityError(f"invalid vector shape: {array.shape}")
    if not np.isfinite(array).all():
        raise ObserverIdentityError("non-finite vector")
    return array.tolist()


def _positions(pedestrians: Any) -> list[list[float]]:
    return _vectors([ped["position"] for ped in pedestrians])


def _require_distinct(positions: list[list[float]]) -> None:
    if len({tuple(position) for position in positions}) != len(positions):
        raise ObserverIdentityError(
            "coincident pedestrian positions cannot establish slot identity"
        )


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ObserverIdentityError(f"{label} differs from trace slot order or state")


def bind_episode(capture: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Bind pre-step force slots to reset IDs and every later trace state.

    Equality is intentional: an approximate position match could accept two
    nearby actors in a permutation. Unobservable population changes fail closed.
    """
    trace = row["algorithm_metadata"]["simulation_step_trace"]
    reset = trace["reset"]["pedestrians"]
    steps = trace["steps"]
    actors = [ped["actor_id"] for ped in reset]
    if not actors or len(set(actors)) != len(actors):
        raise ObserverIdentityError("reset actor IDs missing or duplicated")
    _require_distinct(_positions(reset))
    _same(capture["episode_id"], row["episode_id"], "episode ID")
    _same(capture["seed"], row["seed"], "seed")
    _same(capture["algo"], row["algo"], "algorithm")
    _same(capture["scenario_id"], row["scenario_id"], "scenario ID")
    _same(len(capture["steps"]), len(steps), "step count")
    _same(row["steps"], len(steps), "recorded episode step count")
    prior = _positions(reset)
    bound_steps = []
    component_roster: list[str] | None = None
    for index, (sample, step) in enumerate(zip(capture["steps"], steps, strict=True)):
        _same(step["step"], index, "trace step index")
        _same(sample["step"], index, "capture step index")
        _same([ped["actor_id"] for ped in step["pedestrians"]], actors, "actor IDs")
        _same(sample["step_entry_positions"], prior, "pre-step positions")
        _same(sample["force_input_positions"], prior, "force input positions")
        _require_distinct(sample["force_input_positions"])
        _same(sample["post_step_positions"], _positions(step["pedestrians"]), "post-step positions")
        _same(
            sample["total_forces"],
            step["planner"]["ammv"]["pedestrian_force_vectors"],
            "total force vectors",
        )
        _same(len(sample["robot_forces"]), len(actors), "robot force cardinality")
        roster = sample["component_ids"]
        if not roster or len(set(roster)) != len(roster):
            raise ObserverIdentityError("robot force component IDs missing or duplicated")
        if component_roster is None:
            component_roster = roster
        _same(roster, component_roster, "robot force component roster")
        bound_steps.append({"step": index, "actor_ids": actors, **sample})
        prior = _positions(step["pedestrians"])
    return {
        "schema_version": "issue-9671-robot-force-observer.v1",
        "episode_id": row["episode_id"],
        "scenario_id": row["scenario_id"],
        "seed": row["seed"],
        "algorithm": row["algo"],
        "episode_record_canonical_sha256": hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "actor_ids": actors,
        "steps": bound_steps,
    }


class ForceObserver:
    """Trace frozen call/return events without calling the force component again."""

    def __init__(self, output: Path, provenance: dict[str, Any], source_root: Path) -> None:
        """Keep one process/thread and one episode active at a time."""
        self.output = output
        self.provenance = provenance
        self.source_root = source_root.resolve(strict=True)
        self.verified_frames: set[Any] = set()
        self.pid = os.getpid()
        self.thread = threading.get_ident()
        self.episode: dict[str, Any] | None = None
        self.step: dict[str, Any] | None = None
        self.force_returns: dict[int, dict[str, Any]] = {}
        self.component_object_ids: tuple[int, ...] | None = None

    def _check_process(self) -> None:
        if os.getpid() != self.pid or threading.get_ident() != self.thread:
            raise ObserverIdentityError("episode crossed process or thread boundary")

    def __call__(self, frame: Any, event: str, arg: Any) -> Any:
        """Dispatch only the three pinned frozen source functions."""
        filename = frame.f_code.co_filename
        name = frame.f_code.co_name
        if filename.endswith(RUNNER_FILE) and name == "run_map_episode":
            self._verify_source_frame(frame, event, RUNNER_FILE)
            self._episode_event(frame, event, arg)
        elif filename.endswith(SIM_FILE) and name == "step_once":
            self._verify_source_frame(frame, event, SIM_FILE)
            self._step_event(frame, event)
        elif filename.endswith(FORCE_FILE) and name == "__call__":
            self._verify_source_frame(frame, event, FORCE_FILE)
            if event == "return":
                self._force_event(frame, arg)
        else:
            return None
        if event == "return":
            self.verified_frames.remove(frame)
        return self

    def _verify_source_frame(self, frame: Any, event: str, relative_path: str) -> None:
        if event != "call":
            if frame not in self.verified_frames:
                raise ObserverIdentityError("traced frame has no verified frozen-source call")
            return
        expected_path = (self.source_root / relative_path).resolve(strict=True)
        declared_file = frame.f_globals.get("__file__")
        if (
            Path(frame.f_code.co_filename).resolve(strict=True) != expected_path
            or not isinstance(declared_file, str)
            or Path(declared_file).resolve(strict=True) != expected_path
            or frame.f_globals.get("__name__") != SOURCE_MODULES[relative_path]
        ):
            raise ObserverIdentityError("traced frame is outside pinned frozen source/module")
        self.verified_frames.add(frame)

    def _episode_event(self, frame: Any, event: str, arg: Any) -> None:
        self._check_process()
        if event == "call":
            if self.episode is not None:
                raise ObserverIdentityError("nested episode")
            self.episode = {
                "seed": frame.f_locals["seed"],
                "algo": frame.f_locals["algo"],
                "scenario_id": str(
                    frame.f_locals["scenario"].get("name")
                    or frame.f_locals["scenario"].get("scenario_id")
                    or frame.f_locals["scenario"].get("id")
                    or "unknown"
                ),
                "steps": [],
            }
            self.component_object_ids = None
        elif event == "return" and self.episode is not None:
            if not isinstance(arg, dict):
                raise ObserverIdentityError("episode returned without a record")
            if self.step is not None:
                raise ObserverIdentityError("episode ended during a step")
            capture = self.episode
            capture["episode_id"] = arg["episode_id"]
            bound = bind_episode(capture, arg)
            bound["observer_provenance"] = self.provenance
            self.output.mkdir(parents=True, exist_ok=True)
            target = self.output / f"{arg['episode_id']}.robot-force.json"
            if target.exists():
                raise ObserverIdentityError("duplicate episode sidecar")
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(bound, sort_keys=True, separators=(",", ":")) + "\n")
            temporary.replace(target)
            self.episode = None

    def _step_event(self, frame: Any, event: str) -> None:  # noqa: C901
        if self.episode is None:
            return
        self._check_process()
        simulator = frame.f_locals["self"]
        if event == "call":
            if self.step is not None:
                raise ObserverIdentityError("nested simulator step")
            self.step = {
                "step": len(self.episode["steps"]),
                "step_entry_positions": _vectors(simulator.ped_pos),
            }
            self.force_returns = {}
        elif event == "line" and frame.f_lineno == FORCE_LINES.get(type(simulator).__name__):
            if self.step is None:
                raise ObserverIdentityError("force event without a step")
            from robot_sf.ped_npc.ped_robot_force import PedRobotForce

            components = [
                force
                for force in simulator.pysf_sim.forces
                if isinstance(force, PedRobotForce)
                and getattr(force, "component_type", None) == "pedestrian_robot"
            ]
            if not components:
                raise ObserverIdentityError("no registered robot force component")
            if len(self.force_returns) != len(components):
                raise ObserverIdentityError("robot force call count differs from roster")
            count = len(self.step["step_entry_positions"])
            values = np.zeros((count, 2), dtype=float)
            input_positions = None
            component_ids = []
            component_inputs = []
            for component in components:
                sample = self.force_returns.get(id(component))
                if sample is None:
                    raise ObserverIdentityError("uncaptured robot force component")
                if input_positions is None:
                    input_positions = sample["positions"]
                _same(sample["positions"], input_positions, "force component slot positions")
                values += np.asarray(sample["forces"], dtype=float)
                component_ids.append(getattr(component, "component_id", None))
                component_inputs.append(sample)
            if not all(component_ids) or len(set(component_ids)) != len(component_ids):
                raise ObserverIdentityError("robot force component IDs missing or duplicated")
            object_ids = tuple(id(component) for component in components)
            if self.component_object_ids is None:
                self.component_object_ids = object_ids
            _same(object_ids, self.component_object_ids, "robot force component object roster")
            self.step["force_input_positions"] = input_positions
            self.step["robot_forces"] = _vectors(values, count)
            self.step["component_ids"] = component_ids
            self.step["component_inputs"] = component_inputs
        elif event == "return":
            if self.step is None or "robot_forces" not in self.step:
                raise ObserverIdentityError("step ended without robot force capture")
            self.step["post_step_positions"] = _vectors(simulator.ped_pos)
            self.step["total_forces"] = _vectors(simulator.last_ped_forces)
            self.episode["steps"].append(self.step)
            self.step = None

    def _force_event(self, frame: Any, arg: Any) -> None:
        if self.step is None or self.episode is None:
            return
        self._check_process()
        component = frame.f_locals["self"]
        from robot_sf.ped_npc.ped_robot_force import PedRobotForce

        if (
            not isinstance(component, PedRobotForce)
            or getattr(component, "component_type", None) != "pedestrian_robot"
        ):
            return
        if id(component) in self.force_returns:
            raise ObserverIdentityError("component evaluated twice in one step")
        count = len(self.step["step_entry_positions"])
        returned = _vectors(arg, count)
        _same(returned, _vectors(component.last_forces, count), "last_forces return")
        raw_multipliers = frame.f_locals.get("multipliers")
        multipliers = None
        if raw_multipliers is not None:
            multiplier_array = np.asarray(raw_multipliers, dtype=float)
            if multiplier_array.ndim != 1 or not np.isfinite(multiplier_array).all():
                raise ObserverIdentityError("invalid pedestrian response multipliers")
            multipliers = multiplier_array.tolist()
        self.force_returns[id(component)] = {
            "positions": _vectors(frame.f_locals["ped_positions"], count),
            "robot_position": _vectors([frame.f_locals["robot_pos"]], 1)[0],
            "multipliers": multipliers,
            "multipliers_applied": multipliers is not None and len(multipliers) == count,
            "config": {
                key: value
                for key, value in vars(component.config).items()
                if isinstance(value, bool | int | float | str)
            },
            "forces": returned,
        }


def install_from_environment() -> ForceObserver | None:
    """Install only when an explicit, checksum-bound output is supplied."""
    output = os.environ.get("ISSUE9671_OBSERVER_OUTPUT")
    if not output:
        return None
    expected = os.environ["ISSUE9671_OBSERVER_SHA256"]
    actual = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if actual != expected:
        raise ObserverIdentityError("observer byte hash mismatch")
    source = Path(os.environ["ISSUE9671_FROZEN_SOURCE_ROOT"]).resolve(strict=True)
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != FROZEN_SOURCE:
        raise ObserverIdentityError("frozen source commit mismatch")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip():
        raise ObserverIdentityError("frozen source tracked files are dirty")
    source_hashes = {}
    for relative_path in SOURCE_MODULES:
        committed = subprocess.check_output(
            ["git", "-C", str(source), "show", f"{FROZEN_SOURCE}:{relative_path}"]
        )
        local = (source / relative_path).read_bytes()
        if local != committed:
            raise ObserverIdentityError(f"frozen source bytes differ: {relative_path}")
        source_hashes[relative_path] = hashlib.sha256(local).hexdigest()
    config = Path(os.environ["ISSUE9671_DIAGNOSTIC_CONFIG_PATH"])
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    if config_sha != os.environ["ISSUE9671_DIAGNOSTIC_CONFIG_SHA256"]:
        raise ObserverIdentityError("diagnostic config byte hash mismatch")
    provenance = {
        "observer_sha256": actual,
        "frozen_source_commit": head,
        "frozen_source_file_sha256": source_hashes,
        "diagnostic_config_sha256": config_sha,
        "campaign_id": os.environ["ISSUE9671_CAMPAIGN_ID"],
        "pid": str(os.getpid()),
    }
    observer = ForceObserver(Path(output), provenance, source)
    sys.settrace(observer)
    threading.settrace(observer)
    return observer


_OBSERVER = install_from_environment()
