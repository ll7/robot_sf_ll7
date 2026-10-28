"""Single perceived-pedestrian snapshot shared by every planner-facing pedestrian channel.

Diagnostic stressor for matched pedestrian-perception degradation. One snapshot is
computed per simulator step from simulator truth and then consumed by both the
SocNav pedestrian rows and the occupancy-grid pedestrian layer, so no planner input
keeps exposing true pedestrian positions under a treatment condition.

Conditions (one effect at a time; declared synthetic stressors, not a sensor model):

- ``oracle``: perceived state equals simulator truth (control; same code path).
- ``position_noise``: i.i.d. Gaussian offset on each pedestrian position, indexed by
  (episode key, step, simulator slot). Velocities stay exact.
- ``delay``: positions and world-frame velocities from ``delay_steps`` steps earlier;
  the first snapshot is held until enough history exists. Robot state stays current.
- ``static_occlusion``: pedestrians whose robot-to-pedestrian segment crosses a static
  obstacle interior are removed. Static map knowledge stays exact.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import LineString
from shapely.prepared import prep

PERCEPTION_CONDITIONS = ("oracle", "position_noise", "delay", "static_occlusion")
PERCEPTION_INTERPRETATION = "declared_synthetic_pedestrian_perception_stressor_not_a_sensor_model"
CONSUMER_CHANNELS = ("socnav_pedestrian_rows", "occupancy_grid_pedestrians")


@dataclass(frozen=True)
class PedestrianPerceptionSpec:
    """Normalized one-effect-at-a-time perception condition."""

    condition: str = "oracle"
    position_noise_std_m: float = 0.0
    delay_steps: int = 0
    record_trace: bool = False

    def __post_init__(self) -> None:
        """Validate that exactly the declared effect is active."""
        if self.condition not in PERCEPTION_CONDITIONS:
            raise ValueError(f"unknown pedestrian perception condition: {self.condition!r}")
        if self.position_noise_std_m < 0.0 or self.delay_steps < 0:
            raise ValueError("noise std and delay steps must be non-negative")
        if (self.position_noise_std_m > 0.0) != (self.condition == "position_noise"):
            raise ValueError("position_noise_std_m > 0 exactly when condition is position_noise")
        if (self.delay_steps > 0) != (self.condition == "delay"):
            raise ValueError("delay_steps > 0 exactly when condition is delay")

    @classmethod
    def from_mapping(cls, raw: Any) -> PedestrianPerceptionSpec | None:
        """Build a spec from an observation-noise payload entry, or ``None`` when absent.

        Returns:
            Validated spec, or ``None`` when no perception block is configured.
        """
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise TypeError("pedestrian_perception must be a mapping")
        unknown = set(raw) - {"condition", "position_noise_std_m", "delay_steps", "record_trace"}
        if unknown:
            raise ValueError(f"unknown pedestrian_perception keys: {sorted(unknown)}")
        return cls(
            condition=str(raw.get("condition", "oracle")),
            position_noise_std_m=float(raw.get("position_noise_std_m", 0.0) or 0.0),
            delay_steps=int(raw.get("delay_steps", 0) or 0),
            record_trace=bool(raw.get("record_trace", False)),
        )


@dataclass(frozen=True)
class PerceivedPedestrianSnapshot:
    """World-frame perceived pedestrian rows for one step."""

    step: int
    positions: np.ndarray
    velocities: np.ndarray
    source_slots: np.ndarray


def perception_episode_key(*, scenario_id: str, seed: int, condition: str) -> int:
    """Return a 64-bit key shared by all planners for one (cell, seed, condition).

    Returns:
        Unsigned 64-bit integer derived from SHA-256.
    """
    material = json.dumps(
        {"scenario_id": str(scenario_id), "seed": int(seed), "condition": str(condition)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def position_noise_offsets(
    episode_key: int, step: int, slots: np.ndarray, std_m: float
) -> np.ndarray:
    """Return reproducible Gaussian offsets indexed by (episode, step, slot).

    Each offset comes from its own counter-based stream, so a pedestrian's offset
    does not depend on how many other pedestrians exist.

    Returns:
        ``(N, 2)`` offsets in meters.
    """
    out = np.zeros((slots.shape[0], 2), dtype=float)
    for row, slot in enumerate(slots):
        bitgen = np.random.Philox(key=int(episode_key), counter=[int(step), int(slot), 0, 0])
        out[row] = np.random.Generator(bitgen).normal(0.0, std_m, size=2)
    return out


@dataclass
class PedestrianPerception:
    """Compute one perceived snapshot per step and account for every consumer."""

    spec: PedestrianPerceptionSpec
    scenario_id: str
    seed: int
    episode_key: int = field(init=False)
    current: PerceivedPedestrianSnapshot | None = field(init=False, default=None)
    _step: int = field(init=False, default=-1)
    _history: deque = field(init=False)
    _obstacles: list[tuple[Any, Any]] | None = field(init=False, default=None)
    expected_channels: tuple[str, ...] = CONSUMER_CHANNELS
    _consumed: dict[str, int] = field(init=False)
    _last_consumed_step: dict[str, int] = field(init=False)
    _stats: dict[str, float] = field(init=False)
    trace: list[dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        """Derive the episode key and zero the counters."""
        self.episode_key = perception_episode_key(
            scenario_id=self.scenario_id, seed=self.seed, condition=self.spec.condition
        )
        self.reset()

    def reset(self) -> None:
        """Start a new episode (called by the environment before the reset snapshot)."""
        self.current = None
        self._step = -1
        self._history = deque(maxlen=self.spec.delay_steps + 1)
        self._consumed = dict.fromkeys(CONSUMER_CHANNELS, 0)
        self._last_consumed_step = dict.fromkeys(CONSUMER_CHANNELS, -1)
        self._stats = {
            "snapshots": 0,
            "true_rows": 0,
            "perceived_rows": 0,
            "rows_occluded": 0,
            "steps_with_occlusion": 0,
            "rows_position_differs": 0,
            "steps_perceived_differs": 0,
            "abs_position_error_sum_m": 0.0,
            "max_position_error_m": 0.0,
            "delay_held_initial_steps": 0,
        }
        self.trace = []

    def _static_obstacles(self, simulator: Any) -> list[tuple[Any, Any]]:
        if self._obstacles is None:
            prepared: list[tuple[Any, Any]] = []
            map_def = getattr(simulator, "map_def", None)
            for obstacle in getattr(map_def, "obstacles", None) or []:
                polygons = (
                    obstacle.iter_polygons()
                    if hasattr(obstacle, "iter_polygons")
                    else [getattr(obstacle, "geometry", None)]
                )
                for polygon in polygons:
                    if polygon is None or getattr(polygon, "is_empty", False):
                        continue
                    prepared.append((prep(polygon), polygon))
            self._obstacles = prepared
        return self._obstacles

    def _occluded(self, simulator: Any, robot_pos: np.ndarray, ped_pos: np.ndarray) -> bool:
        """Same line-of-sight semantics as ``SocNavObservationFusion._statically_occluded``.

        Returns:
            True when the segment crosses a static obstacle interior.
        """
        segment = LineString(
            [(float(robot_pos[0]), float(robot_pos[1])), tuple(map(float, ped_pos))]
        )
        if segment.length <= 0.0:
            return False
        for prep_geom, raw_poly in self._static_obstacles(simulator):
            if prep_geom.intersects(segment) and not segment.touches(raw_poly):
                return True
        return False

    def observe(self, simulator: Any) -> PerceivedPedestrianSnapshot:
        """Compute this step's perceived snapshot from simulator truth.

        Returns:
            The snapshot every pedestrian channel must consume this step.
        """
        self._step += 1
        step = self._step
        true_pos = np.asarray(simulator.ped_pos, dtype=float).reshape(-1, 2).copy()
        try:
            true_vel = np.asarray(simulator.ped_vel, dtype=float).reshape(-1, 2).copy()
        except (AttributeError, TypeError, ValueError):
            true_vel = np.zeros_like(true_pos)
        slots = np.arange(true_pos.shape[0], dtype=np.int64)
        truth = PerceivedPedestrianSnapshot(step, true_pos, true_vel, slots)
        robot_pos = np.asarray(simulator.robots[0].pose[0], dtype=float).reshape(2)

        condition = self.spec.condition
        occluded_rows = 0
        if condition == "oracle":
            snap = truth
        elif condition == "position_noise":
            offsets = position_noise_offsets(
                self.episode_key, step, slots, self.spec.position_noise_std_m
            )
            snap = PerceivedPedestrianSnapshot(step, true_pos + offsets, true_vel, slots)
        elif condition == "delay":
            self._history.append(truth)
            if len(self._history) <= self.spec.delay_steps:
                self._stats["delay_held_initial_steps"] += 1
            old = self._history[0]
            snap = PerceivedPedestrianSnapshot(
                step, old.positions, old.velocities, old.source_slots
            )
        else:  # static_occlusion
            keep = np.array(
                [not self._occluded(simulator, robot_pos, p) for p in true_pos], dtype=bool
            )
            occluded_rows = int(true_pos.shape[0] - np.count_nonzero(keep))
            snap = PerceivedPedestrianSnapshot(step, true_pos[keep], true_vel[keep], slots[keep])
        self.current = snap
        self._account(truth, snap, occluded_rows, robot_pos)
        return snap

    def _account(
        self,
        truth: PerceivedPedestrianSnapshot,
        snap: PerceivedPedestrianSnapshot,
        occluded_rows: int,
        robot_pos: np.ndarray,
    ) -> None:
        stats = self._stats
        stats["snapshots"] += 1
        stats["true_rows"] += int(truth.positions.shape[0])
        stats["perceived_rows"] += int(snap.positions.shape[0])
        stats["rows_occluded"] += occluded_rows
        stats["steps_with_occlusion"] += int(occluded_rows > 0)
        errors = np.zeros((0,), dtype=float)
        if snap.positions.shape[0] > 0 and truth.positions.shape[0] > 0:
            valid = snap.source_slots < truth.positions.shape[0]
            errors = np.linalg.norm(
                snap.positions[valid] - truth.positions[snap.source_slots[valid]], axis=1
            )
        differs = bool(
            snap.positions.shape != truth.positions.shape
            or not np.array_equal(snap.positions, truth.positions)
            or not np.array_equal(snap.velocities, truth.velocities)
        )
        stats["steps_perceived_differs"] += int(differs)
        stats["rows_position_differs"] += int(np.count_nonzero(errors > 0.0))
        stats["abs_position_error_sum_m"] += float(errors.sum())
        if errors.size:
            stats["max_position_error_m"] = max(stats["max_position_error_m"], float(errors.max()))
        if self.spec.record_trace:
            self.trace.append(
                {
                    "step": truth.step,
                    "robot_xy": [round(float(v), 4) for v in robot_pos],
                    "true_xy": np.round(truth.positions, 4).tolist(),
                    "true_vel": np.round(truth.velocities, 4).tolist(),
                    "perceived_slots": snap.source_slots.tolist(),
                    "perceived_xy": np.round(snap.positions, 4).tolist(),
                    "perceived_vel": np.round(snap.velocities, 4).tolist(),
                }
            )

    def consume(self, channel: str) -> PerceivedPedestrianSnapshot:
        """Return the current snapshot and count each step a channel consumed it.

        Returns:
            Current snapshot.
        """
        if self.current is None:
            raise RuntimeError("pedestrian perception consumed before the first snapshot")
        if channel not in self._consumed:
            raise ValueError(f"unknown perception consumer channel: {channel!r}")
        if self._last_consumed_step[channel] != self.current.step:
            self._last_consumed_step[channel] = self.current.step
            self._consumed[channel] += 1
        return self.current

    def summary(self) -> dict[str, Any]:
        """Return JSON-safe per-episode accounting.

        Returns:
            Spec, episode key and counters, including per-channel consumption counts.
        """
        stats = dict(self._stats)
        rows = max(1, int(stats["perceived_rows"]))
        stats["mean_abs_position_error_m"] = stats["abs_position_error_sum_m"] / rows
        snapshots = int(stats["snapshots"])
        return {
            "interpretation": PERCEPTION_INTERPRETATION,
            "condition": self.spec.condition,
            "position_noise_std_m": self.spec.position_noise_std_m,
            "delay_steps": self.spec.delay_steps,
            "episode_key": str(self.episode_key),
            "expected_channels": list(self.expected_channels),
            "consumed_steps": dict(self._consumed),
            "all_channels_consumed_every_snapshot": snapshots > 0
            and all(self._consumed[channel] == snapshots for channel in self.expected_channels),
            "stats": stats,
        }
