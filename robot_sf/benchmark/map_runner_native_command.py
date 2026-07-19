"""Native-command execution path and deadlock/stall metric for the benchmark runner.

This module adds a first-class *native-command* arm to the map-runner benchmark harness
(Issue #5887). The arm runs a declared external planner binary as a subprocess, either once
per episode (per-episode invocation) or as a long-lived persistent process (persistent-process
mode). Both modes expose the same argv + env contract, timeout handling, exit-code semantics,
and provenance capture (the command string plus a content hash of the resolved binary).

The module also implements a parameterized no-progress-over-window deadlock/stall detector. The
detector is intentionally distinct from the episode timeout: a stall is "no meaningful progress
toward the goal over a parameterized number of consecutive steps", while a timeout is the global
horizon limit. The deadlock boolean is emitted into ``metrics.deadlock`` so it flows through to
campaign tables without changing any existing metric semantics.

Nothing here changes existing planner behavior or metric definitions; every emitted field is
additive.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import selectors
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.algorithm_metadata import (
    enrich_algorithm_metadata,
    resolve_learned_checkpoint_observation_contract,
)

_NATIVE_COMMAND_ALGO = "native_command"

# Planner-diagnostics keys the issue #5416 paired-outcome analyzer requires on every
# native row (see scripts/analysis/analyze_issue_5416_sipp_four_geometry.py).
_DIAGNOSTICS_KEYS = (
    "expansion_limit_hits",
    "runtime_bound_exits",
    "fallback_count",
    "commitment_invalidations",
)
_RUNTIME_FIELD = "planner_step_runtime_seconds"

_DEFAULT_DEADLOCK_WINDOW_STEPS = 40
_DEFAULT_DEADLOCK_PROGRESS_THRESHOLD_M = 0.05
_DEFAULT_STEP_TIMEOUT_SEC = 30.0


class NativeCommandContractError(ValueError):
    """Raised when a native-command contract field is missing or invalid."""


class NativeCommandStepError(RuntimeError):
    """Raised when a native-command subprocess fails to return a usable command."""


def _resolve_binary_hash(command: list[str]) -> tuple[str, str]:
    """Return ``(label, hash)`` provenance for a native planner command.

    Args:
        command: Resolved argv list (first element is the binary path).

    Returns:
        The resolved binary label and its full sha256 content hash. An unresolved
        command has an empty content hash and is still allowed to fail at launch time.
    """

    if not command:
        raise NativeCommandContractError("native command must be a non-empty argv list")
    binary = Path(command[0])
    if not binary.is_file():
        resolved = shutil.which(command[0])
        if resolved:
            binary = Path(resolved)
    label = str(binary)
    if not binary.is_file():
        return label, ""
    try:
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    except OSError:
        return label, ""
    return label, digest


def _resolve_entrypoint_hash(value: object, command: list[str]) -> tuple[str, str]:
    """Return a content hash for the tracked command entrypoint when declared.

    ``command[0]`` is often a Python interpreter rather than the planner source.
    Native planner configurations can therefore declare ``entrypoint_path`` to
    preserve provenance for the actual tracked executable.
    """
    raw_path = str(value).strip() if value is not None else command[0]
    path = Path(raw_path)
    if not path.is_file():
        return raw_path, ""
    try:
        return str(path), hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return str(path), ""


@dataclass
class NativeCommandSpec:
    """Parsed argv + env contract for the native-command planner arm.

    Attributes:
        command: Resolved subprocess argv list.
        env: Extra environment variables merged over the process environment.
        mode: ``"per_episode"`` spawns a fresh process each episode; ``"persistent"``
            keeps one process alive across steps for the episode (request/response over
            stdin/stdout).
        step_timeout_sec: Per-step wall-clock budget before the call is declared stalled.
        binary_label: Provenance label of the resolved binary.
        binary_hash: Full content hash of the resolved binary, when available.
    """

    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    mode: str = "per_episode"
    step_timeout_sec: float = _DEFAULT_STEP_TIMEOUT_SEC
    binary_label: str = ""
    binary_hash: str = ""
    entrypoint_label: str = ""
    entrypoint_hash: str = ""
    require_static_geometry: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> NativeCommandSpec:
        """Build a spec from the planner algo config.

        The expected config is::

            native_command:
              command: ["/path/sipp", "--scenario", "{scenario_id}"]
              env: {SIPP_SEED: "{seed}"}
              mode: per_episode | persistent
              step_timeout_sec: 30.0

        ``{scenario_id}``, ``{seed}``, ``{horizon}``, and ``{dt}`` template tokens in the
        command and env are resolved per episode.

        Returns:
            A validated ``NativeCommandSpec``.
        """
        payload = config or {}
        nested = payload.get("native_command")
        if isinstance(nested, dict):
            payload = nested
        raw_command = payload.get("command", payload.get("argv"))
        if not isinstance(raw_command, (list, tuple)) or not raw_command:
            raise NativeCommandContractError(
                "native_command requires a non-empty 'command' or 'argv' list",
            )
        command = [str(token) for token in raw_command]
        env = payload.get("env")
        env = {str(k): str(v) for k, v in env.items()} if isinstance(env, dict) else {}
        raw_mode = payload.get("mode")
        if raw_mode is None and "persistent" in payload:
            raw_mode = "persistent" if bool(payload["persistent"]) else "per_episode"
        mode = str(raw_mode if raw_mode is not None else "per_episode").strip().lower()
        if mode not in {"per_episode", "persistent"}:
            raise NativeCommandContractError(
                f"native_command mode must be 'per_episode' or 'persistent', got {mode!r}",
            )
        step_timeout_sec = float(
            payload.get("step_timeout_sec", payload.get("timeout_s", _DEFAULT_STEP_TIMEOUT_SEC))
        )
        if not math.isfinite(step_timeout_sec) or step_timeout_sec <= 0.0:
            raise NativeCommandContractError(
                "native_command step timeout must be positive and finite",
            )
        label, binary_hash = _resolve_binary_hash(command)
        entrypoint_label, entrypoint_hash = _resolve_entrypoint_hash(
            payload.get("entrypoint_path"), command
        )
        return cls(
            command=command,
            env=env,
            mode=mode,
            step_timeout_sec=step_timeout_sec,
            binary_label=label,
            binary_hash=binary_hash,
            entrypoint_label=entrypoint_label,
            entrypoint_hash=entrypoint_hash,
            require_static_geometry=bool(payload.get("require_static_geometry", False)),
        )

    def resolve_templates(
        self, *, scenario_id: str, seed: int, horizon: int, dt: float
    ) -> NativeCommandSpec:
        """Return a copy with episode-level template tokens substituted."""
        mapping = {
            "{scenario_id}": scenario_id,
            "{seed}": str(seed),
            "{horizon}": str(horizon),
            "{dt}": str(dt),
        }
        command = list(self.command)
        for idx, token in enumerate(command):
            for key, value in mapping.items():
                token = token.replace(key, value)
            command[idx] = token
        env = dict(self.env)
        for key in env:
            value = env[key]
            for token, replacement in mapping.items():
                value = value.replace(token, replacement)
            env[key] = value
        return NativeCommandSpec(
            command=command,
            env=env,
            mode=self.mode,
            step_timeout_sec=self.step_timeout_sec,
            binary_label=self.binary_label,
            binary_hash=self.binary_hash,
            entrypoint_label=self.entrypoint_label,
            entrypoint_hash=self.entrypoint_hash,
            require_static_geometry=self.require_static_geometry,
        )


def _render_request(  # noqa: C901
    obs: dict[str, Any], static_geometry: dict[str, Any] | None = None
) -> str:
    """Render a policy observation as the subprocess request payload.

    Returns:
        JSON string. The robot/goal position and heading are normalized into a canonical
        ``{robot: {position, heading}, goal: {current}, pedestrians: [...]}`` shape so a
        native planner can compute a goal-directed command without depending on the full
        env observation schema (which may be nested or flat).
    """
    robot_block = obs.get("robot")
    if isinstance(robot_block, dict):
        robot_block = dict(robot_block)
        if "speed" not in robot_block and obs.get("robot_speed") is not None:
            robot_block["speed"] = obs["robot_speed"]
        if "angular_velocity" not in robot_block and obs.get("robot_angular_velocity") is not None:
            robot_block["angular_velocity"] = obs["robot_angular_velocity"]
    else:
        pos = obs.get("robot_position")
        heading = obs.get("robot_heading")
        speed = obs.get("robot_speed")
        angular_velocity = obs.get("robot_angular_velocity")
        robot_block = {
            "position": list(pos) if _is_sequence(pos) else None,
            "heading": list(heading) if _is_sequence(heading) else None,
            "speed": list(speed) if _is_sequence(speed) else None,
            "angular_velocity": (
                list(angular_velocity) if _is_sequence(angular_velocity) else None
            ),
        }
    goal_block = obs.get("goal")
    if isinstance(goal_block, dict):
        goal_block = dict(goal_block)
        if "next" not in goal_block and obs.get("goal_next") is not None:
            goal_block["next"] = obs["goal_next"]
    else:
        gpos = obs.get("goal_position") or obs.get("goal_current")
        gnext = obs.get("goal_next")
        goal_block = {
            "current": list(gpos) if _is_sequence(gpos) else None,
            "next": list(gnext) if _is_sequence(gnext) else None,
        }
    request: dict[str, Any] = {"robot": robot_block, "goal": goal_block}
    sim_block = obs.get("sim")
    if isinstance(sim_block, dict):
        request["sim"] = dict(sim_block)
    elif obs.get("dt") is not None:
        request["dt"] = obs["dt"]
    ped = obs.get("pedestrians")
    if isinstance(ped, dict):
        request["pedestrians"] = ped
    elif obs.get("pedestrians_positions") is not None:
        request["pedestrians"] = {
            "positions": obs.get("pedestrians_positions"),
            "velocities": obs.get("pedestrians_velocities"),
            "count": obs.get("pedestrians_count"),
        }
    if static_geometry is not None:
        request["static_geometry"] = static_geometry
    return json.dumps(
        request, default=lambda obj: obj.tolist() if hasattr(obj, "tolist") else str(obj)
    )


def _is_sequence(value: object) -> bool:
    """Return whether a value is a non-string sequence (list/tuple/ndarray)."""
    return isinstance(value, (list, tuple)) or (
        hasattr(value, "__len__") and hasattr(value, "__getitem__")
    )


def _parse_response_payload(
    text: str,
) -> tuple[float, float, dict[str, Any] | None]:
    """Parse a subprocess response and optional geometry-consumption proof.

    Returns:
        Linear and angular command parsed from the ``linear_velocity`` /
        ``angular_velocity`` keys (numpy-style tuples also accepted).

    Raises:
        NativeCommandStepError: If the payload is not usable.
    """
    text = text.strip()
    if not text:
        raise NativeCommandStepError("native command returned an empty response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise NativeCommandStepError(f"native command returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise NativeCommandStepError("native command response must be a JSON object")
    linear = payload.get(
        "linear_velocity",
        payload.get("linear", payload.get("v", payload.get("vx"))),
    )
    angular = payload.get(
        "angular_velocity",
        payload.get("angular", payload.get("omega", payload.get("vy"))),
    )
    try:
        values = (float(linear), float(angular))
    except (TypeError, ValueError) as exc:
        raise NativeCommandStepError(
            "native command response missing a recognized linear/angular command pair",
        ) from exc
    if not all(math.isfinite(value) for value in values):
        raise NativeCommandStepError("native command response must contain finite velocities")
    geometry_consumption = payload.get("geometry_consumption")
    if geometry_consumption is not None and not isinstance(geometry_consumption, dict):
        raise NativeCommandStepError("native command geometry_consumption must be a JSON object")
    return values[0], values[1], geometry_consumption


def _parse_response(text: str) -> tuple[float, float]:
    """Parse a subprocess response into a ``(linear, angular)`` command pair.

    Returns:
        Linear and angular velocity command.
    """
    linear, angular, _geometry_consumption = _parse_response_payload(text)
    return linear, angular


class _NoProgressDeadlockDetector:
    """Parameterized no-progress-over-window deadlock/stall detector.

    The detector is distinct from the episode timeout: it flags a *stall* when the robot
    makes no meaningful progress toward the goal over a parameterized number of consecutive
    steps, regardless of whether the global horizon has been reached.
    """

    def __init__(
        self,
        *,
        window_steps: int = _DEFAULT_DEADLOCK_WINDOW_STEPS,
        progress_threshold_m: float = _DEFAULT_DEADLOCK_PROGRESS_THRESHOLD_M,
    ) -> None:
        if window_steps < 1:
            raise ValueError("deadlock window_steps must be >= 1")
        if progress_threshold_m < 0.0:
            raise ValueError("deadlock progress_threshold_m must be >= 0")
        self._window_steps = int(window_steps)
        self._threshold = float(progress_threshold_m)
        self._distances: list[float] = []
        self._active = False

    def update(self, distance_to_goal_m: float) -> None:
        """Record the current distance-to-goal and refresh stall state.

        Args:
            distance_to_goal_m: Euclidean robot-to-goal distance at this step.
        """
        self._distances.append(float(distance_to_goal_m))
        if len(self._distances) <= self._window_steps:
            return
        window_start_idx = len(self._distances) - 1 - self._window_steps
        start_distance = self._distances[window_start_idx]
        final_distance = self._distances[-1]
        progress_delta = start_distance - final_distance
        self._active = progress_delta <= self._threshold

    @property
    def active(self) -> bool:
        """Whether the stall condition currently holds."""
        return self._active

    def as_field(self) -> dict[str, Any]:
        """Return the typed episode diagnostics field for this detector."""
        return {
            "schema_version": "native-command-deadlock.v1",
            "window_steps": self._window_steps,
            "progress_threshold_m": self._threshold,
            "active": bool(self._active),
            "distance_samples": len(self._distances),
        }


class NativeCommandPlanner:
    """Bridge a declared native planner command into the map-runner policy contract.

    The planner either spawns a fresh subprocess per episode (``per_episode``) or keeps one
    persistent subprocess alive across steps (``persistent``), sending one JSON request per
    step over stdin and reading one JSON response per step from stdout.
    """

    def __init__(self, spec: NativeCommandSpec) -> None:
        """Bind the resolved command spec and initialize run-state buffers."""
        self._spec = spec
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_buffer = bytearray()
        self._last_runtime_sec: float = 0.0
        self._last_decision: dict[str, Any] = {}
        self._static_geometry: dict[str, Any] | None = None
        self._diagnostics: dict[str, Any] = dict.fromkeys(_DIAGNOSTICS_KEYS, 0)
        self._diagnostics["exit_codes"] = []
        self._diagnostics["last_exit_code"] = None
        # How many times a subprocess was actually launched. This is the observable
        # that makes the persistent-mode respawn bug (issue #5957) directly visible
        # in episode diagnostics: a healthy persistent run records exactly one spawn
        # regardless of horizon, while the per-episode mode records one spawn per step.
        self._process_spawns = 0
        self._runtimes: list[float] = []
        self._deadlock = _NoProgressDeadlockDetector()
        # Plain serializable run state, updated every step so episode finalization can
        # read it after the metadata dict is deep-copied (the deep copy would otherwise
        # sever the link to the live planner instance).
        self.run_state: dict[str, Any] = {
            "deadlock_active": False,
            "deadlock_field": self._deadlock.as_field(),
            "planner_diagnostics": self.diagnostics,
            "geometry_input_received": False,
            "geometry_input_verified": False,
        }

    def bind_env(self, env: Any, *, scenario_id: str) -> None:
        """Bind a serializable static-map representation from the live environment.

        The native contract must not infer geometry from a process-local path: a
        planner receives the exact obstacle and boundary segments selected by the
        map runner. Missing or malformed map geometry is a fail-closed contract
        error when the configuration requires geometry-aware execution.
        """
        simulator = getattr(env, "simulator", None)
        segments_fn = getattr(simulator, "iter_obstacle_segments", None)
        map_def = getattr(simulator, "map_def", None)
        if simulator is None or not callable(segments_fn):
            if self._spec.require_static_geometry:
                raise NativeCommandContractError("native command requires live static geometry")
            return
        segments = _serialize_segments(segments_fn())
        bounds = _serialize_segments(getattr(map_def, "bounds", []) if map_def is not None else [])
        if self._spec.require_static_geometry and not (segments or bounds):
            raise NativeCommandContractError("native command received no static geometry segments")
        payload = {
            "schema_version": "native-command-static-geometry.v1",
            "scenario_id": scenario_id,
            "obstacle_segments": segments,
            "boundary_segments": bounds,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        self._static_geometry = payload
        self.run_state["static_geometry"] = {
            "schema_version": payload["schema_version"],
            "scenario_id": scenario_id,
            "sha256": payload["sha256"],
            "obstacle_segment_count": len(segments),
            "boundary_segment_count": len(bounds),
        }
        self.run_state["geometry_input_received"] = True
        self.run_state["geometry_input_verified"] = False

    def _verify_geometry_consumption(
        self,
        geometry_consumption: dict[str, Any] | None,
    ) -> None:
        """Verify the subprocess populated the planner-consumed geometry channels."""
        if not self._spec.require_static_geometry:
            return
        if self._static_geometry is None:
            raise NativeCommandStepError("native command static geometry was not bound")
        if not isinstance(geometry_consumption, dict):
            raise NativeCommandStepError(
                "native command omitted required geometry-consumption proof"
            )
        if geometry_consumption.get("schema_version") != "native-command-geometry-consumption.v1":
            raise NativeCommandStepError("native command geometry-consumption schema mismatch")
        if geometry_consumption.get("geometry_sha256") != self._static_geometry.get("sha256"):
            raise NativeCommandStepError("native command geometry-consumption hash mismatch")
        try:
            obstacle_cells = int(geometry_consumption["obstacle_occupied_cells"])
            pedestrian_cells = int(geometry_consumption["pedestrian_occupied_cells"])
            combined_cells = int(geometry_consumption["combined_occupied_cells"])
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeCommandStepError(
                "native command geometry-consumption cell counts are invalid"
            ) from exc
        if obstacle_cells <= 0:
            raise NativeCommandStepError("native command planner-facing obstacle channel is empty")
        if pedestrian_cells < 0 or combined_cells < max(obstacle_cells, pedestrian_cells):
            raise NativeCommandStepError(
                "native command planner-facing combined channel is incomplete"
            )
        if geometry_consumption.get("combined_matches_union") is not True:
            raise NativeCommandStepError(
                "native command combined channel does not match geometry union"
            )
        self.run_state["geometry_consumption"] = dict(geometry_consumption)
        self.run_state["geometry_input_verified"] = True

    def reset(self, *, seed: int | None = None) -> None:
        """Open the persistent process (no-op for per-episode mode)."""
        if self._spec.mode == "persistent":
            self._start_process()

    def _start_process(self) -> None:
        self._stop_process()
        try:
            self._process = subprocess.Popen(
                self._spec.command,
                env={**_process_environ(), **self._spec.env},
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            self._process_spawns += 1
        except OSError as exc:
            raise NativeCommandContractError(
                f"failed to launch native command {self._spec.command}: {exc}",
            ) from exc

    def _stop_process(self) -> None:
        if self._process is not None:
            try:
                self._process.stdin.close() if self._process.stdin else None
            except OSError:
                pass
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            self._process = None
        self._stdout_buffer.clear()

    def close(self) -> None:
        """Release any persistent subprocess."""
        self._stop_process()

    def plan(
        self, obs: dict[str, Any], *, distance_to_goal_m: float | None = None
    ) -> tuple[float, float]:
        """Produce a ``(linear, angular)`` command from the native planner.

        Args:
            obs: Map-runner policy observation for this step.
            distance_to_goal_m: Optional precomputed goal distance used by the stall
                detector (falls back to the observation when omitted).

        Returns:
            Linear and angular command parsed from the subprocess response.
        """
        if self._spec.require_static_geometry and self._static_geometry is None:
            raise NativeCommandContractError("native command static geometry was not bound")
        if self._spec.require_static_geometry:
            self.run_state["geometry_input_verified"] = False
        request = _render_request(obs, self._static_geometry)
        started = time.perf_counter()
        try:
            if self._spec.mode == "persistent":
                _cmd, linear, angular, geometry_consumption = self._plan_persistent(request)
            else:
                _cmd, linear, angular, geometry_consumption = self._plan_per_episode(request)
            self._verify_geometry_consumption(geometry_consumption)
        except NativeCommandStepError:
            # A malformed response, timeout, non-zero exit, or broken persistent
            # stream is a per-step fallback, not a batch-wide crash. Contract and
            # launch errors remain fail-closed and are intentionally not caught.
            if self._spec.mode == "persistent":
                self._stop_process()
            linear, angular = 0.0, 0.0
            self._diagnostics["fallback_count"] += 1
        self._last_runtime_sec = float(time.perf_counter() - started)
        self._runtimes.append(self._last_runtime_sec)
        self._last_decision = {
            "linear_velocity": linear,
            "angular_velocity": angular,
            "mode": self._spec.mode,
        }
        if distance_to_goal_m is not None:
            self._deadlock.update(distance_to_goal_m)
        else:
            robot_block = obs.get("robot")
            goal_block = obs.get("goal")
            pos = (
                robot_block.get("position")
                if isinstance(robot_block, dict)
                else obs.get("robot_position")
            )
            goal = (
                goal_block.get("current")
                if isinstance(goal_block, dict)
                else obs.get("goal_position") or obs.get("goal_current")
            )
            if _is_sequence(pos) and _is_sequence(goal) and len(pos) >= 2 and len(goal) >= 2:
                self._deadlock.update(
                    math.hypot(float(pos[0]) - float(goal[0]), float(pos[1]) - float(goal[1])),
                )
        self.run_state["deadlock_active"] = bool(self._deadlock.active)
        self.run_state["deadlock_field"] = self._deadlock.as_field()
        self.run_state["planner_diagnostics"] = self.diagnostics
        return linear, angular

    def _plan_per_episode(
        self, request: str
    ) -> tuple[list[str], float, float, dict[str, Any] | None]:
        try:
            proc = subprocess.run(
                self._spec.command,
                input=request.encode("utf-8"),
                env={**_process_environ(), **self._spec.env},
                capture_output=True,
                timeout=self._spec.step_timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            # The child was launched but exceeded the step budget, so it counts as a
            # spawn (matches the persistent path's "only count an actual launch" rule).
            self._process_spawns += 1
            self._diagnostics["runtime_bound_exits"] += 1
            self._diagnostics.setdefault("exit_codes", []).append(None)
            self._diagnostics["last_exit_code"] = None
            raise NativeCommandStepError(
                f"native command exceeded step timeout {self._spec.step_timeout_sec}s",
            ) from exc
        except OSError as exc:
            # Not counted: the subprocess never launched (contract/launch failure).
            raise NativeCommandContractError(
                f"failed to launch native command {self._spec.command}: {exc}",
            ) from exc
        # The child launched and returned; count it even on a nonzero exit, since it
        # did run (consistent with runner.py's _spawn-then-communicate per-episode path).
        self._process_spawns += 1
        self._diagnostics.setdefault("exit_codes", []).append(proc.returncode)
        self._diagnostics["last_exit_code"] = proc.returncode
        if proc.returncode != 0:
            raise NativeCommandStepError(
                f"native command exited with code {proc.returncode}: "
                f"{proc.stderr.decode('utf-8', errors='replace').strip()}",
            )
        linear, angular, geometry_consumption = _parse_response_payload(
            proc.stdout.decode("utf-8", errors="replace")
        )
        return self._spec.command, linear, angular, geometry_consumption

    def _plan_persistent(
        self, request: str
    ) -> tuple[list[str], float, float, dict[str, Any] | None]:
        if self._process is None or self._process.poll() is not None:
            self._start_process()
        assert self._process is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        try:
            self._process.stdin.write((request + "\n").encode("utf-8"))
            self._process.stdin.flush()
            line = self._readline_with_timeout()
        except subprocess.TimeoutExpired as exc:
            self._diagnostics["runtime_bound_exits"] += 1
            self._diagnostics.setdefault("exit_codes", []).append(None)
            self._diagnostics["last_exit_code"] = self._process.poll()
            self._stop_process()
            raise NativeCommandStepError(
                f"persistent native command exceeded step timeout {self._spec.step_timeout_sec}s",
            ) from exc
        except (OSError, ValueError) as exc:
            self._diagnostics.setdefault("exit_codes", []).append(self._process.poll())
            self._diagnostics["last_exit_code"] = self._process.poll()
            self._stop_process()
            raise NativeCommandStepError(f"persistent native command I/O failed: {exc}") from exc
        if not line:
            self._diagnostics.setdefault("exit_codes", []).append(self._process.poll())
            self._diagnostics["last_exit_code"] = self._process.poll()
            self._stop_process()
            raise NativeCommandStepError("persistent native command closed its stdout")
        exit_code = self._process.poll()
        self._diagnostics.setdefault("exit_codes", []).append(exit_code)
        self._diagnostics["last_exit_code"] = exit_code
        if exit_code not in (0, None):
            self._stop_process()
            raise NativeCommandStepError(
                f"persistent native command exited with code {exit_code}",
            )
        linear, angular, geometry_consumption = _parse_response_payload(line)
        return self._spec.command, linear, angular, geometry_consumption

    def _readline_with_timeout(self) -> str:
        """Read one persistent response line without allowing an unbounded block.

        Returns:
            Decoded response text without the line terminator.
        """
        assert self._process is not None
        assert self._process.stdout is not None
        deadline = time.monotonic() + self._spec.step_timeout_sec
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._stdout_buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise subprocess.TimeoutExpired(self._spec.command, self._spec.step_timeout_sec)
                if not selector.select(remaining):
                    raise subprocess.TimeoutExpired(self._spec.command, self._spec.step_timeout_sec)
                chunk = os.read(self._process.stdout.fileno(), 4096)
                if not chunk:
                    break
                self._stdout_buffer.extend(chunk)
        if b"\n" not in self._stdout_buffer:
            line = bytes(self._stdout_buffer)
            self._stdout_buffer.clear()
            return line.decode("utf-8", errors="replace")
        line, remainder = bytes(self._stdout_buffer).split(b"\n", 1)
        self._stdout_buffer = bytearray(remainder)
        return line.decode("utf-8", errors="replace")

    def planner_stats(self) -> dict[str, Any]:
        """Return live diagnostics snapshot for the episode metadata/analyzer."""
        return {
            "last_decision": dict(self._last_decision),
            "last_step_runtime_sec": self._last_runtime_sec,
            "mode": self._spec.mode,
            "binary_label": self._spec.binary_label,
            "binary_hash": self._spec.binary_hash,
            "entrypoint_label": self._spec.entrypoint_label,
            "entrypoint_hash": self._spec.entrypoint_hash,
        }

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Planner-diagnostics block required by the issue #5416 analyzer."""
        return {
            **{key: int(self._diagnostics[key]) for key in _DIAGNOSTICS_KEYS},
            _RUNTIME_FIELD: [float(value) for value in self._runtimes],
            "exit_codes": list(self._diagnostics["exit_codes"]),
            "last_exit_code": self._diagnostics["last_exit_code"],
            "process_spawns": int(self._process_spawns),
        }

    @property
    def deadlock_field(self) -> dict[str, Any]:
        """Typed deadlock/stall diagnostics field for the episode row."""
        return self._deadlock.as_field()


def _process_environ() -> dict[str, str]:
    """Return a copy of the current process environment for subprocess launches."""
    return {str(k): str(v) for k, v in os.environ.items()}


def _serialize_segments(values: Any) -> list[list[list[float]]]:
    """Normalize simulator line segments into finite JSON-compatible point pairs.

    Returns:
        Finite ``[[x, y], [x, y]]`` segment pairs, omitting malformed values.
    """
    serialized: list[list[list[float]]] = []
    try:
        iterator = iter(values)
    except TypeError:
        return serialized
    for value in iterator:
        try:
            points = list(value)
            if len(points) == 4:
                points = [points[:2], points[2:]]
            if len(points) != 2:
                continue
            start, end = list(points[0]), list(points[1])
            if len(start) < 2 or len(end) < 2:
                continue
            flat = [float(start[0]), float(start[1]), float(end[0]), float(end[1])]
            if not all(math.isfinite(item) for item in flat):
                continue
            serialized.append([[flat[0], flat[1]], [flat[2], flat[3]]])
        except (TypeError, ValueError):
            continue
    return serialized


def build_native_command_policy(  # noqa: PLR0913
    algo_key: str,
    algo_config: dict[str, Any],
    *,
    scenario_id: str,
    seed: int,
    horizon: int,
    dt: float,
    robot_kinematics: str | None = None,
    observation_mode: str | None = None,
    observation_level: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build the native-command policy callable and enriched algorithm metadata.

    Returns:
        tuple[Callable, dict[str, Any]]: Policy callable ``plan(obs) -> (linear, angular)``
        and enriched algorithm metadata whose ``planner_kinematics.execution_mode`` is
        ``"native"`` and which carries the required ``planner_diagnostics`` block.
    """
    spec = NativeCommandSpec.from_config(algo_config).resolve_templates(
        scenario_id=scenario_id,
        seed=seed,
        horizon=horizon,
        dt=dt,
    )
    planner = NativeCommandPlanner(spec)

    learned_observation_contract = resolve_learned_checkpoint_observation_contract(
        algo_key,
        algo_config,
        observation_mode=observation_mode,
        observation_level=observation_level,
    )
    logical_algorithm = str(algo_config.get("planner_variant") or algo_key)
    meta = enrich_algorithm_metadata(
        algo=algo_key,
        metadata={
            "algorithm": logical_algorithm,
            "status": "ok",
            "fallback_or_degraded": False,
            "config": algo_config,
            "config_hash": _config_hash(algo_config),
        },
        execution_mode="native",
        adapter_name="NativeCommandPlanner",
        robot_kinematics=robot_kinematics,
        observation_mode=observation_mode,
        observation_level=observation_level,
    )
    meta["learned_checkpoint_observation_contract"] = learned_observation_contract
    meta["native_command"] = {
        "schema_version": "native-command.v1",
        "mode": spec.mode,
        "persistent": spec.mode == "persistent",
        "command": spec.command,
        "argv": spec.command,
        "env": spec.env,
        "binary_label": spec.binary_label,
        "binary_hash": spec.binary_hash,
        "binary_path": spec.binary_label,
        "binary_hash_sha256": spec.binary_hash or None,
        "entrypoint_path": spec.entrypoint_label,
        "entrypoint_hash_sha256": spec.entrypoint_hash or None,
        "require_static_geometry": spec.require_static_geometry,
        "step_timeout_sec": spec.step_timeout_sec,
        "timeout_s": spec.step_timeout_sec,
    }
    meta["planner_diagnostics"] = planner.diagnostics
    meta["_native_run_state"] = planner.run_state

    def _policy(obs: dict[str, Any]) -> tuple[float, float]:
        """Run the native planner and return a ``(linear, angular)`` command.

        Returns:
            Linear and angular velocity command.
        """
        return planner.plan(obs)

    _policy._planner_reset = planner.reset  # type: ignore[attr-defined]
    _policy._planner_close = planner.close  # type: ignore[attr-defined]
    _policy._planner_stats = planner.planner_stats  # type: ignore[attr-defined]

    def _bind_env(env: Any) -> None:
        planner.bind_env(env, scenario_id=scenario_id)
        static_geometry = planner.run_state.get("static_geometry")
        if isinstance(static_geometry, dict):
            meta["native_command"]["static_geometry"] = static_geometry
            meta["native_command"]["geometry_input_received"] = True
            meta["native_command"]["geometry_input_verified"] = False

    _policy._planner_bind_env = _bind_env  # type: ignore[attr-defined]
    _policy._native_planner = planner  # type: ignore[attr-defined]
    _policy._execution_mode = "native"  # type: ignore[attr-defined]
    return _policy, meta


def _config_hash(config: dict[str, Any]) -> str:
    """Return a short stable hash of an algo config mapping."""
    try:
        text = yaml.safe_dump(config, sort_keys=True)
    except (TypeError, ValueError):
        text = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def native_command_metadata_for_record(
    algo_meta: dict[str, Any],
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """Extract native-command deadlock + diagnostics fields from finalized algo metadata.

    Returns:
        Tuple of ``(is_native, deadlock_field, planner_diagnostics)``. When the episode was
        not run through the native-command arm, ``is_native`` is ``False`` and the other two
        payloads are empty so callers can leave existing rows untouched.
    """
    if not isinstance(algo_meta, dict):
        return False, {}, {}
    run_state = algo_meta.get("_native_run_state")
    if isinstance(run_state, dict) and "deadlock_field" in run_state:
        diagnostics = run_state.get("planner_diagnostics", {})
        fallback_count = (
            diagnostics.get("fallback_count", 0) if isinstance(diagnostics, dict) else 0
        )
        fallback_or_degraded = not isinstance(fallback_count, int) or fallback_count != 0
        algo_meta["fallback_or_degraded"] = fallback_or_degraded
        native_metadata = algo_meta.get("native_command")
        if isinstance(native_metadata, dict):
            native_metadata["fallback_or_degraded"] = fallback_or_degraded
            native_metadata["geometry_input_received"] = bool(
                run_state.get("geometry_input_received", False)
            )
            native_metadata["geometry_input_verified"] = bool(
                run_state.get("geometry_input_verified", False)
            )
            geometry_consumption = run_state.get("geometry_consumption")
            if isinstance(geometry_consumption, dict):
                native_metadata["geometry_consumption"] = dict(geometry_consumption)
        return True, run_state.get("deadlock_field", {}), diagnostics
    kinematics = algo_meta.get("planner_kinematics")
    is_native = isinstance(kinematics, dict) and kinematics.get("execution_mode") == "native"
    return bool(is_native), {}, {}


__all__ = [
    "NativeCommandContractError",
    "NativeCommandPlanner",
    "NativeCommandSpec",
    "NativeCommandStepError",
    "build_native_command_policy",
    "native_command_metadata_for_record",
]
