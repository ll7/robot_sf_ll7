"""Fail-closed social-force window continuation over the existing typed snapshot.

This supports a single RobotEnv with perfect SocNav observations and the analytic
SocialForcePlannerAdapter. It is not a generic environment checkpoint contract.
The simulator snapshot owns actors, navigators and RNGs; the small companion owns
the environment clock/history, observation and adapter's steering filter state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.simulator_counterfactual_adapter import SimulatorCounterfactualModel
from robot_sf.benchmark.typed_snapshot import (
    SnapshotBoundary,
    SnapshotCompatibility,
    _decode_value,
    _encode_value,
    capture_typed_snapshot,
    read_typed_snapshot,
    restore_typed_snapshot,
    write_typed_snapshot,
)
from robot_sf.gym_env.snqi_proxy import StepSNQIProxyState
from robot_sf.planner.socnav_social_force import SocialForcePlannerAdapter

SCHEMA = "robot_sf.generated_continuation.v1"
RUNTIME_SCHEMA = "robot_sf.generated_replay_runtime.v2"
_CLOCK_FIELDS = (
    "episode",
    "timestep",
    "sim_time_elapsed",
    "is_timeout",
    "is_collision_with_ped",
    "is_collision_with_obst",
    "is_collision_with_robot",
    "distance_to_goal",
    "prev_distance_to_goal",
    "episode_step_limit",
    "sim_time_limit",
    "d_t",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sensor(env: Any) -> Any:
    return getattr(env.state.sensors, "wrapped_adapter", env.state.sensors)


def _adapter(policy: Any) -> SocialForcePlannerAdapter:
    adapter = getattr(policy, "_planner_adapter", None)
    if type(adapter) is not SocialForcePlannerAdapter:
        raise ValueError("window continuation supports only SocialForcePlannerAdapter")
    return adapter


def require_supported_continuation(env: Any, policy: Any, loop_config: Any) -> None:
    """Reject unsupported runtime histories rather than resetting them silently."""
    _adapter(policy)
    sensor = _sensor(env)
    if type(env).__name__ != "RobotEnv" or type(sensor).__name__ != "SocNavObservationFusion":
        raise ValueError("window continuation requires RobotEnv with SocNav observations")
    if len(env.simulator.robots) != 1 or env._action_latency_steps:
        raise ValueError("window continuation requires one robot and no action latency")
    if sensor._pedestrian_tracker is not None or sensor._predictive_foresight is not None:
        raise ValueError("window continuation does not support tracking or foresight histories")
    if sensor._lost_pedestrian_memory:
        raise ValueError("window continuation does not support lost-actor memory")
    if (
        loop_config.actuation_controller is not None
        or loop_config.noise_spec.get("enabled", False)
        or loop_config.tracking_precision_spec.get("enabled", False)
        or loop_config.safety_wrapper_runtime.enabled
        or loop_config.cbf_runtime.enabled
        or loop_config.active_harness is not None
    ):
        raise ValueError("window continuation does not support controller/noise/wrapper histories")
    if env.recording_enabled or env.use_jsonl_recording:
        raise ValueError("window continuation does not support recording-session histories")


def _compatibility(env: Any, policy: Any, map_sha256: str) -> SnapshotCompatibility:
    """Bind code bytes and immutable settings without process-local object IDs.

    Returns:
        Compatibility fields for the supported environment and adapter.
    """
    root = Path(__file__).resolve().parents[3]
    code_files = [
        "robot_sf/benchmark/scenario_generation/continuation.py",
        "robot_sf/benchmark/map_runner/map_runner_episode.py",
        "robot_sf/planner/socnav_social_force.py",
        "robot_sf/planner/socnav.py",
        "robot_sf/gym_env/robot_env.py",
        "robot_sf/robot/robot_state.py",
    ]
    code = {name: _sha(root / name) for name in code_files}
    config = asdict(env.config)
    # Factories are checked by the explicit supported subset; map bytes are
    # independently bound. Neither Python object addresses nor mutable map
    # navigator objects belong in the immutable configuration digest.
    for name in ("map_pool", "robot_factory", "ped_factory"):
        config.pop(name, None)
    config["planner"] = asdict(_adapter(policy).config)
    config["planner_drive_settings"] = asdict(_adapter(policy)._sampling_drive_settings)
    settings = json.dumps(config, sort_keys=True, default=str).replace(
        str(root), "<repository_root>"
    )
    return SnapshotCompatibility(
        map_sha256=map_sha256,
        config_sha256=hashlib.sha256(settings.encode()).hexdigest(),
        code_revision=hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest(),
        dt_s=env.state.d_t,
        planner_id="social_force",
    )


def capture_window_checkpoint(
    path: Path,
    *,
    env: Any,
    state: Any,
    loop_config: Any,
    source_episode_id: str,
    seed: int,
    source_map: Path,
    window_steps: int,
) -> dict[str, Any]:
    """Persist a real pre-step boundary; no trace positions are used as state.

    Returns:
        The completed, digest-bound continuation record.
    """
    policy = loop_config.policy_fn
    require_supported_continuation(env, policy, loop_config)
    if path.exists():
        raise FileExistsError(path)
    if env.state.is_terminal or state.collision_seen or state.reached_goal_step is not None:
        raise ValueError("cannot capture a terminal window boundary")
    start_step = env.state.timestep
    if window_steps < 1 or start_step + window_steps > env.state.max_sim_steps:
        raise ValueError("window must fit inside the original episode budget")
    path.parent.mkdir(parents=True, exist_ok=True)
    model = SimulatorCounterfactualModel(env.simulator, capture_rng=True)
    model._step_index = start_step
    compatibility = _compatibility(env, policy, _sha(source_map))
    snapshot = capture_typed_snapshot(
        model,
        compatibility,
        boundary=SnapshotBoundary(
            step_index=start_step,
            absolute_time_s=start_step * env.state.d_t,
            remaining_budget_steps=env.state.max_sim_steps - start_step,
            phase="pre_step",
            next_observation_ready=True,
        ),
    )
    simulator_file = path.with_name(path.stem + ".simulator.json")
    artifact = write_typed_snapshot(snapshot, simulator_file)
    adapter_fields = {
        k: deepcopy(v)
        for k, v in vars(_adapter(policy)).items()
        if k not in {"config", "_sampling_drive_settings"}
    }
    # The adapter uses +inf as a deliberate disabled speed-limit sentinel.
    adapter_fields = {
        k: {"$positive_infinity": True} if isinstance(v, float) and v == math.inf else v
        for k, v in adapter_fields.items()
    }
    aux = {
        "clock": {k: getattr(env.state, k) for k in _CLOCK_FIELDS},
        "last_action": env.last_action,
        "snqi_history": asdict(env._snqi_proxy.state),
        "sensor_heading": _sensor(env)._last_heading,
        "observation": deepcopy(state.obs),
        "planner": adapter_fields,
    }
    arrays: dict[str, np.ndarray] = {}
    encoded = _encode_value(aux, arrays, "continuation")
    aux_file = path.with_suffix(".npz")
    np.savez_compressed(aux_file, **arrays)
    record = {
        "schema_version": SCHEMA,
        "source_episode_id": source_episode_id,
        "source_seed": seed,
        "start_step": start_step,
        "window_steps": window_steps,
        "original_budget_steps": env.state.max_sim_steps,
        "compatibility": snapshot.compatibility.to_dict(),
        "simulator_file": simulator_file.name,
        "simulator_sha256": _sha(simulator_file),
        "simulator_payload_sha256": _sha(artifact.payload_path),
        "aux_file": aux_file.name,
        "aux_sha256": _sha(aux_file),
        "state": encoded,
    }
    # Metadata is the final commit marker. Interrupted companion writes cannot
    # become a usable checkpoint; readers bind every completed companion hash.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(record, sort_keys=True, indent=2) + "\n")
    try:
        os.replace(temporary, path)
        read_window_checkpoint(path, expected_sha256=_sha(path))
    finally:
        temporary.unlink(missing_ok=True)
    return record


def _validate_checkpoint_boundary(record: dict[str, Any]) -> None:
    """Validate the complete boundary inventory independently of file custody."""
    required = {
        "schema_version",
        "source_episode_id",
        "source_seed",
        "start_step",
        "window_steps",
        "original_budget_steps",
        "compatibility",
        "simulator_file",
        "simulator_sha256",
        "simulator_payload_sha256",
        "aux_file",
        "aux_sha256",
        "state",
    }
    if set(record) != required:
        raise ValueError("window checkpoint fields mismatch")
    for name, minimum in (
        ("source_seed", 0),
        ("start_step", 0),
        ("window_steps", 1),
        ("original_budget_steps", 1),
    ):
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError("window checkpoint boundary fields are invalid")
    if record["start_step"] + record["window_steps"] > record["original_budget_steps"]:
        raise ValueError("window checkpoint exceeds the original episode budget")


def read_window_checkpoint(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    """Read only digest-bound companion files under the checkpoint directory.

    Returns:
        The validated continuation record.
    """
    if _sha(path) != expected_sha256:
        raise ValueError("window checkpoint digest mismatch")
    record = json.loads(path.read_text())
    if record.get("schema_version") != SCHEMA:
        raise ValueError("unsupported window checkpoint schema")
    _validate_checkpoint_boundary(record)
    for key in ("simulator_file", "aux_file"):
        name = record.get(key)
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("checkpoint companion must be a basename")
        digest_key = "simulator_sha256" if key == "simulator_file" else "aux_sha256"
        if _sha(path.parent / name) != record[digest_key]:
            raise ValueError("window checkpoint companion digest mismatch")
    simulator_payload = (path.parent / record["simulator_file"]).with_suffix(".json.npz")
    if _sha(simulator_payload) != record["simulator_payload_sha256"]:
        raise ValueError("window simulator payload digest mismatch")
    return record


def restore_window_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    env: Any,
    policy: Any,
    loop_config: Any,
    seed: int,
    source_map: Path,
) -> tuple[Any, dict[str, Any]]:
    """Validate the complete supported packet before restoring a fresh environment.

    Returns:
        Restored next observation and the checkpoint provenance record.
    """
    require_supported_continuation(env, policy, loop_config)
    record = read_window_checkpoint(path, expected_sha256=expected_sha256)
    if record["source_seed"] != seed or record["original_budget_steps"] != env.state.max_sim_steps:
        raise ValueError("window checkpoint seed/budget mismatch")
    snapshot = read_typed_snapshot(path.parent / record["simulator_file"])
    # The typed snapshot binds map/config/code/platform and validates all actor
    # buffers before mutation. Resolve the actual map file from loaded custody.
    expected = _compatibility(env, policy, _sha(source_map))
    snapshot.compatibility.assert_compatible(expected)
    if (
        snapshot.boundary.step_index != record["start_step"]
        or snapshot.boundary.remaining_budget_steps
        != record["original_budget_steps"] - record["start_step"]
        or not snapshot.boundary.next_observation_ready
    ):
        raise ValueError("window checkpoint simulator boundary mismatch")
    with np.load(path.parent / record["aux_file"], allow_pickle=False) as payload:
        aux = _decode_value(record["state"], {k: payload[k] for k in payload.files}, "continuation")
    if set(aux) != {
        "clock",
        "last_action",
        "snqi_history",
        "sensor_heading",
        "observation",
        "planner",
    }:
        raise ValueError("window checkpoint runtime fields mismatch")
    if set(aux["clock"]) != set(_CLOCK_FIELDS):
        raise ValueError("window checkpoint clock fields mismatch")
    if aux["clock"]["timestep"] != record["start_step"] or any(
        aux["clock"][k]
        for k in (
            "is_timeout",
            "is_collision_with_ped",
            "is_collision_with_obst",
            "is_collision_with_robot",
        )
    ):
        raise ValueError("window checkpoint has a terminal or inconsistent clock")
    planner = _adapter(policy)
    required_planner_fields = set(vars(planner)) - {"config", "_sampling_drive_settings"}
    # The steering target is created on the first policy step rather than reset.
    if not required_planner_fields <= set(aux["planner"]) or set(
        aux["planner"]
    ) - required_planner_fields - {"_target_velocity"}:
        raise ValueError("window checkpoint planner fields mismatch")
    if set(aux["snqi_history"]) != {f.name for f in fields(StepSNQIProxyState)}:
        raise ValueError("window checkpoint reward history fields mismatch")
    model = SimulatorCounterfactualModel(env.simulator, capture_rng=True)
    restore_typed_snapshot(model, snapshot, expected)
    for name, value in aux["clock"].items():
        setattr(env.state, name, value)
    env.last_action = aux["last_action"]
    env._snqi_proxy.state = StepSNQIProxyState(**aux["snqi_history"])
    _sensor(env)._last_heading = aux["sensor_heading"]
    for name, value in aux["planner"].items():
        setattr(
            planner,
            name,
            math.inf
            if isinstance(value, dict) and value == {"$positive_infinity": True}
            else value,
        )
    env._latest_observation = deepcopy(aux["observation"])
    return aux["observation"], record
