"""Deterministic recording fixture for the trajectory-visualization example.

This module owns the headless smoke path of
``examples/advanced/14_trajectory_visualization.py`` (issue #8734). It builds a
small generated recording with the exact ``(states, map_def)`` tuple schema the
canonical environment writer persists (see ``robot_sf/gym_env/base_env.py``),
so the fixture is consumed through the canonical ``load_states`` reader
without a hand-written pickle or a pre-existing recording.

Headless execution renders a deterministic frame sequence through
``robot_sf.render.frame_export`` and emits a JSON summary with source schema,
frame count, actor count, coordinate frame, and output digests. Malformed,
legacy-ambiguous, non-finite, path-escaping, and empty recordings fail closed
with stable reason codes.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from robot_sf.nav.map_config import MapDefinition
from robot_sf.render.frame_export import export_pickle_frames
from robot_sf.render.playback_recording import load_states
from robot_sf.render.sim_state import VisualizableSimState
from robot_sf.render.sim_view import _empty_map_definition

FIXTURE_SCHEMA = "trajectory-viz-fixture.v1"
CANONICAL_TUPLE_SCHEMA = "playback-recording-tuple.v1"
COORDINATE_FRAME = "meters, origin bottom-left, x right, y up"

REASON_OK = "ok"
REASON_MISSING_FILE = "missing_file"
REASON_NOT_A_FILE = "not_a_file"
REASON_PATH_ESCAPE = "path_escape"
REASON_MALFORMED_PICKLE = "malformed_pickle"
REASON_LEGACY_AMBIGUOUS_SHAPE = "legacy_ambiguous_shape"
REASON_EMPTY_RECORDING = "empty_recording"
REASON_NON_FINITE_VALUES = "non_finite_values"

_FIXTURE_STEPS_DEFAULT = 12
_FIXTURE_PEDESTRIANS_DEFAULT = 2


def _repo_root() -> Path:
    """Return the repository root inferred from this file location."""
    return Path(__file__).resolve().parents[2]


def _allowed_recording_roots() -> list[Path]:
    """Return roots a recording path may resolve inside (repo, tmp, cwd)."""
    import tempfile

    roots = [_repo_root(), Path.cwd(), Path(tempfile.gettempdir()).resolve()]
    return [root.resolve() for root in roots]


def resolve_recording_path(raw: str | Path) -> tuple[Path | None, str]:
    """Resolve a recording path fail-closed with a stable reason code."""
    candidate = Path(raw).expanduser()
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        return None, REASON_PATH_ESCAPE
    if not candidate.exists():
        return None, REASON_MISSING_FILE
    if not resolved.is_file():
        return None, REASON_NOT_A_FILE
    if not any(resolved == root or root in resolved.parents for root in _allowed_recording_roots()):
        return None, REASON_PATH_ESCAPE
    return resolved, REASON_OK


def resolve_out_dir(raw: str | Path) -> tuple[Path | None, str]:
    """Resolve a caller-owned output directory fail-closed."""
    candidate = Path(raw).expanduser()
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        return None, REASON_PATH_ESCAPE
    if not any(resolved == root or root in resolved.parents for root in _allowed_recording_roots()):
        return None, REASON_PATH_ESCAPE
    return resolved, REASON_OK


def build_fixture_states(
    *, steps: int = _FIXTURE_STEPS_DEFAULT, pedestrians: int = _FIXTURE_PEDESTRIANS_DEFAULT
) -> list[VisualizableSimState]:
    """Build deterministic playback states (no RNG): robot walks the diagonal."""
    states = []
    for step in range(steps):
        x = 0.1 + 0.07 * step
        y = 0.1 + 0.05 * step
        ped_positions = np.array(
            [[0.7 - 0.03 * step, 0.3 + 0.04 * step] for _ in range(pedestrians)],
            dtype=float,
        )
        ped_actions = np.zeros((pedestrians, 2, 2), dtype=float)
        states.append(
            VisualizableSimState(
                timestep=step,
                robot_action=None,
                robot_pose=((x, y), 0.0),
                pedestrian_positions=ped_positions,
                ray_vecs=np.zeros((0, 2), dtype=float),
                ped_actions=ped_actions,
            )
        )
    return states


def build_fixture_recording(
    path: str | Path,
    *,
    steps: int = _FIXTURE_STEPS_DEFAULT,
    pedestrians: int = _FIXTURE_PEDESTRIANS_DEFAULT,
) -> Path:
    """Write a generated fixture with the canonical ``(states, map_def)`` schema."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    states = build_fixture_states(steps=steps, pedestrians=pedestrians)
    map_def: MapDefinition = _empty_map_definition()
    with target.open("wb") as handle:
        pickle.dump((states, map_def), handle)
    return target


def check_recording_values(states: list[VisualizableSimState]) -> str:
    """Fail closed on empty or non-finite state payloads."""
    if not states:
        return REASON_EMPTY_RECORDING
    for state in states:
        arrays = [state.pedestrian_positions, state.ray_vecs, state.ped_actions]
        for array in arrays:
            values = np.asarray(array, dtype=float)
            if values.size and not np.all(np.isfinite(values)):
                return REASON_NON_FINITE_VALUES
    return REASON_OK


def summarize_recording(
    states: list[VisualizableSimState],
    map_def: MapDefinition,
    frame_paths: list[Path],
    *,
    schema: str = FIXTURE_SCHEMA,
) -> dict[str, object]:
    """Build the deterministic JSON summary for a headless run."""
    digests = {}
    for frame_path in frame_paths:
        digest = hashlib.sha256(frame_path.read_bytes()).hexdigest()
        digests[frame_path.name] = digest
    pedestrian_counts = {
        len(np.asarray(state.pedestrian_positions).reshape(-1, 2)) for state in states
    }
    return {
        "schema": schema,
        "tuple_schema": CANONICAL_TUPLE_SCHEMA,
        "frame_count": len(states),
        "rendered_frames": len(frame_paths),
        "actor_count": {"robot": 1, "pedestrians": sorted(pedestrian_counts)},
        "coordinate_frame": COORDINATE_FRAME,
        "map": {"width": map_def.width, "height": map_def.height},
        "output_digests": digests,
    }


def _load_validated_states(
    recording_path: Path,
) -> tuple[tuple[list[VisualizableSimState], MapDefinition] | None, str]:
    """Load a recording fail-closed, mapping every failure to a reason code."""
    try:
        data = load_states(str(recording_path))
    except ValueError as exc:
        if "empty" in str(exc).lower():
            return None, REASON_EMPTY_RECORDING
        return None, REASON_LEGACY_AMBIGUOUS_SHAPE
    except (TypeError, AttributeError, IndexError, KeyError):
        return None, REASON_LEGACY_AMBIGUOUS_SHAPE
    except Exception:
        return None, REASON_MALFORMED_PICKLE
    if not isinstance(data, tuple) or len(data) != 2:
        return None, REASON_LEGACY_AMBIGUOUS_SHAPE
    states, map_def = data
    if not isinstance(states, list) or not isinstance(map_def, MapDefinition):
        return None, REASON_LEGACY_AMBIGUOUS_SHAPE
    if not all(isinstance(state, VisualizableSimState) for state in states):
        return None, REASON_LEGACY_AMBIGUOUS_SHAPE
    return (states, map_def), REASON_OK


def run_headless(
    recording: str | Path,
    out_dir: str | Path,
    *,
    max_frames: int = 6,
) -> tuple[dict[str, object] | None, str]:
    """Validate, render, and summarize a recording; fail closed with reason codes."""
    recording_path, reason = resolve_recording_path(recording)
    if recording_path is None:
        return None, reason
    target_dir, reason = resolve_out_dir(out_dir)
    if target_dir is None:
        return None, reason
    loaded, reason = _load_validated_states(recording_path)
    if loaded is None:
        return None, reason
    states, map_def = loaded
    reason = check_recording_values(states)
    if reason != REASON_OK:
        return None, reason
    frame_paths, _filmstrip = export_pickle_frames(
        recording_path, target_dir / "frames", count=min(max_frames, len(states))
    )
    summary = summarize_recording(states, map_def, frame_paths)
    summary_path = target_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary, REASON_OK
