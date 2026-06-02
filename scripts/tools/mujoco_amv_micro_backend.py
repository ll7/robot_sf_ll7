"""Replay a tiny AMV command trace through an optional MuJoCo diagnostic path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.analysis_workbench.simulation_trace_export import (
    SIMULATION_TRACE_EXPORT_SCHEMA_VERSION,
    load_simulation_trace_export,
)

if TYPE_CHECKING:
    from types import ModuleType

SCHEMA_VERSION = "mujoco_amv_micro_backend.v1"
CLAIM_BOUNDARY = (
    "diagnostic-only MuJoCo micro-backend probe; not Robot SF simulator replacement, "
    "not social-navigation benchmark evidence, and not calibrated AMV hardware evidence"
)
UNSUPPORTED_SEMANTICS = (
    "pedestrian dynamics",
    "social-navigation benchmark outcomes",
    "map geometry and obstacle contacts",
    "hardware-calibrated AMV response",
    "controller latency calibration",
)
_MODEL_XML = """\
<mujoco model="robot_sf_amv_micro_backend">
  <option timestep="{timestep:.8f}" gravity="0 0 0"/>
  <worldbody>
    <body name="amv" pos="0 0 0">
      <freejoint name="root"/>
      <geom name="body" type="box" size="0.35 0.18 0.05" mass="20"/>
    </body>
  </worldbody>
</mujoco>
"""


@dataclass(frozen=True, slots=True)
class CommandSegment:
    """Constant command over a duration."""

    duration_s: float
    v_m_s: float
    omega_rad_s: float
    source_step: int | None = None


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Limits and timing for the micro-backend replay."""

    timestep_s: float
    max_linear_accel_m_s2: float
    max_linear_decel_m_s2: float
    max_angular_accel_rad_s2: float
    max_yaw_rate_rad_s: float
    latency_steps: int


@dataclass(slots=True)
class ReplayState:
    """Integrated AMV state for the diagnostic replay."""

    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0
    v_m_s: float = 0.0
    omega_rad_s: float = 0.0


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--commands",
        type=Path,
        help="CSV with duration_s,v_m_s,omega_rad_s columns. Use --demo-fixture to omit.",
    )
    inputs.add_argument(
        "--trace",
        type=Path,
        help=(
            "simulation_trace_export.v1 JSON trace. Reads "
            "frames[].planner.selected_action linear/angular velocities."
        ),
    )
    inputs.add_argument(
        "--demo-fixture",
        action="store_true",
        help="Use a built-in stop/go/turn command trace instead of --commands.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--max-linear-accel", type=float, default=2.819)
    parser.add_argument("--max-linear-decel", type=float, default=3.429)
    parser.add_argument("--max-angular-accel", type=float, default=4.0)
    parser.add_argument("--max-yaw-rate", type=float, default=1.2)
    parser.add_argument("--latency-steps", type=int, default=1)
    return parser


def _load_mujoco() -> ModuleType:
    """Import MuJoCo or raise a clear runtime error."""
    try:
        return importlib.import_module("mujoco")
    except Exception as exc:  # pragma: no cover - exact import errors vary by host
        raise RuntimeError(
            "MuJoCo is not available. Install it in an explicit diagnostic environment "
            "and rerun; Robot SF does not add MuJoCo to routine dependencies."
        ) from exc


def _load_commands(
    path: Path | None,
    *,
    trace_path: Path | None,
    demo_fixture: bool,
    timestep_s: float,
) -> tuple[list[CommandSegment], dict[str, Any] | None]:
    """Load command segments from CSV or return the built-in demo trace."""
    if demo_fixture:
        return [
            CommandSegment(duration_s=0.5, v_m_s=0.0, omega_rad_s=0.0),
            CommandSegment(duration_s=1.0, v_m_s=2.0, omega_rad_s=0.0),
            CommandSegment(duration_s=1.0, v_m_s=2.0, omega_rad_s=0.8),
            CommandSegment(duration_s=0.8, v_m_s=0.2, omega_rad_s=-0.6),
            CommandSegment(duration_s=0.6, v_m_s=0.0, omega_rad_s=0.0),
        ], None
    if trace_path is not None:
        return _load_trace_commands(trace_path, fallback_duration_s=timestep_s)
    if path is None:
        raise ValueError("one of --commands, --trace, or --demo-fixture is required")

    segments: list[CommandSegment] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"duration_s", "v_m_s", "omega_rad_s"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            try:
                segment = CommandSegment(
                    duration_s=float(row["duration_s"]),
                    v_m_s=float(row["v_m_s"]),
                    omega_rad_s=float(row["omega_rad_s"]),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row_number} contains a non-numeric command") from exc
            if segment.duration_s <= 0:
                raise ValueError(f"{path}:{row_number} duration_s must be positive")
            segments.append(segment)
    if not segments:
        raise ValueError(f"{path} has no command rows")
    return segments, None


def _load_trace_commands(
    path: Path,
    *,
    fallback_duration_s: float,
) -> tuple[list[CommandSegment], dict[str, Any]]:
    """Load command segments from ``simulation_trace_export.v1`` selected actions."""
    if fallback_duration_s <= 0:
        raise ValueError("--timestep must be positive")
    trace = load_simulation_trace_export(path)
    if trace.schema_version != SIMULATION_TRACE_EXPORT_SCHEMA_VERSION:
        raise ValueError(f"{path} schema_version must be {SIMULATION_TRACE_EXPORT_SCHEMA_VERSION}")
    if not trace.frames:
        raise ValueError(f"{path} has no frames")

    segments: list[CommandSegment] = []
    previous_delta_s = fallback_duration_s
    for index, frame in enumerate(trace.frames):
        selected_action = frame.planner["selected_action"]
        missing_fields = {"linear_velocity", "angular_velocity"}.difference(selected_action)
        if missing_fields:
            raise ValueError(
                f"{path}: frame {frame.step} selected_action missing "
                f"{', '.join(sorted(missing_fields))}"
            )
        if index + 1 < len(trace.frames):
            duration_s = trace.frames[index + 1].time_s - frame.time_s
            previous_delta_s = duration_s
        else:
            duration_s = previous_delta_s
        if duration_s <= 0:
            raise ValueError(f"{path}: frame {frame.step} duration must be positive")
        segments.append(
            CommandSegment(
                duration_s=duration_s,
                v_m_s=float(selected_action["linear_velocity"]),
                omega_rad_s=float(selected_action["angular_velocity"]),
                source_step=frame.step,
            )
        )

    source_trace = {
        "schema_version": trace.schema_version,
        "trace_id": trace.trace_id,
        "source_path": str(path),
        "command_fields": {
            "linear_velocity": "frames[].planner.selected_action.linear_velocity",
            "angular_velocity": "frames[].planner.selected_action.angular_velocity",
        },
        "evidence_boundary": trace.evidence_boundary,
        "source": {
            "scenario_id": trace.source.scenario_id,
            "seed": trace.source.seed,
            "planner_id": trace.source.planner_id,
            "episode_id": trace.source.episode_id,
            "generated_by": trace.source.generated_by,
        },
    }
    return segments, source_trace


def _expand_commands(segments: list[CommandSegment], timestep_s: float) -> list[dict[str, Any]]:
    """Expand duration-coded commands into fixed-timestep command rows."""
    if timestep_s <= 0:
        raise ValueError("--timestep must be positive")
    rows: list[dict[str, float]] = []
    time_s = 0.0
    for segment in segments:
        steps = max(1, math.ceil(segment.duration_s / timestep_s))
        for _ in range(steps):
            rows.append(
                {
                    "time_s": round(time_s, 10),
                    "commanded_v_m_s": segment.v_m_s,
                    "commanded_omega_rad_s": segment.omega_rad_s,
                    "source_step": segment.source_step,
                }
            )
            time_s += timestep_s
    return rows


def _clip_delta(
    target: float,
    current: float,
    *,
    positive_limit: float,
    negative_limit: float | None = None,
    dt: float,
) -> tuple[float, bool]:
    """Move ``current`` toward ``target`` under asymmetric rate limits."""
    down_limit = positive_limit if negative_limit is None else negative_limit
    delta = target - current
    max_up = max(float(positive_limit), 0.0) * dt
    max_down = max(float(down_limit), 0.0) * dt
    clipped_delta = min(delta, max_up) if delta >= 0 else max(delta, -max_down)
    return current + clipped_delta, not math.isclose(delta, clipped_delta, abs_tol=1e-12)


def _build_mujoco_model(mujoco: ModuleType, timestep_s: float) -> tuple[Any, Any, str]:
    """Build a tiny MuJoCo model and data pair for runtime proof."""
    xml = _MODEL_XML.format(timestep=timestep_s)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data, hashlib.sha256(xml.encode("utf-8")).hexdigest()


def replay_commands(
    segments: list[CommandSegment],
    config: ReplayConfig,
    *,
    mujoco: ModuleType | None = None,
    source_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay command segments and return diagnostic payload."""
    mujoco_module = mujoco or _load_mujoco()
    model, data, model_sha256 = _build_mujoco_model(mujoco_module, config.timestep_s)
    commands = _expand_commands(segments, config.timestep_s)
    latency_queue = [
        {"commanded_v_m_s": 0.0, "commanded_omega_rad_s": 0.0}
        for _ in range(max(config.latency_steps, 0))
    ]
    state = ReplayState()
    rows: list[dict[str, Any]] = []
    clip_count = 0
    previous_v = 0.0

    for command in commands:
        latency_queue.append(command)
        applied_command = latency_queue.pop(0)
        target_v = float(applied_command["commanded_v_m_s"])
        target_omega = float(applied_command["commanded_omega_rad_s"])

        next_v, linear_clipped = _clip_delta(
            target_v,
            state.v_m_s,
            positive_limit=config.max_linear_accel_m_s2,
            negative_limit=config.max_linear_decel_m_s2,
            dt=config.timestep_s,
        )
        next_omega, angular_clipped = _clip_delta(
            target_omega,
            state.omega_rad_s,
            positive_limit=config.max_angular_accel_rad_s2,
            dt=config.timestep_s,
        )
        yaw_limited_omega = max(
            -config.max_yaw_rate_rad_s,
            min(config.max_yaw_rate_rad_s, next_omega),
        )
        yaw_clipped = not math.isclose(yaw_limited_omega, next_omega, abs_tol=1e-12)
        state.v_m_s = next_v
        state.omega_rad_s = yaw_limited_omega
        state.yaw_rad = _normalize_angle(state.yaw_rad + state.omega_rad_s * config.timestep_s)
        state.x_m += state.v_m_s * math.cos(state.yaw_rad) * config.timestep_s
        state.y_m += state.v_m_s * math.sin(state.yaw_rad) * config.timestep_s
        mujoco_module.mj_step(model, data)

        clipped = linear_clipped or angular_clipped or yaw_clipped
        clip_count += int(clipped)
        linear_accel = (state.v_m_s - previous_v) / config.timestep_s
        previous_v = state.v_m_s
        rows.append(
            {
                "time_s": command["time_s"],
                "commanded_v_m_s": command["commanded_v_m_s"],
                "commanded_omega_rad_s": command["commanded_omega_rad_s"],
                "source_step": command["source_step"],
                "applied_v_m_s": state.v_m_s,
                "applied_omega_rad_s": state.omega_rad_s,
                "linear_accel_m_s2": linear_accel,
                "x_m": state.x_m,
                "y_m": state.y_m,
                "yaw_rad": state.yaw_rad,
                "linear_rate_clipped": linear_clipped,
                "angular_rate_clipped": angular_clipped,
                "yaw_rate_clipped": yaw_clipped,
            }
        )

    return _payload(
        rows,
        model_sha256=model_sha256,
        mujoco_version=str(getattr(mujoco_module, "__version__", "unknown")),
        config=config,
        clip_count=clip_count,
        source_trace=source_trace,
    )


def _payload(
    rows: list[dict[str, Any]],
    *,
    model_sha256: str,
    mujoco_version: str,
    config: ReplayConfig,
    clip_count: int,
    source_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the output payload."""
    max_accel = max((row["linear_accel_m_s2"] for row in rows), default=0.0)
    max_decel = min((row["linear_accel_m_s2"] for row in rows), default=0.0)
    max_yaw = max((abs(row["applied_omega_rad_s"]) for row in rows), default=0.0)
    final_pose = rows[-1] if rows else {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "claim_boundary": CLAIM_BOUNDARY,
        "source_trace": source_trace,
        "runtime": {
            "backend": "mujoco",
            "mujoco_version": mujoco_version,
            "model_xml_sha256": model_sha256,
            "routine_dependency": False,
        },
        "config": {
            "timestep_s": config.timestep_s,
            "max_linear_accel_m_s2": config.max_linear_accel_m_s2,
            "max_linear_decel_m_s2": config.max_linear_decel_m_s2,
            "max_angular_accel_rad_s2": config.max_angular_accel_rad_s2,
            "max_yaw_rate_rad_s": config.max_yaw_rate_rad_s,
            "latency_steps": config.latency_steps,
        },
        "command_contract": {
            "action_space": "unicycle_vw",
            "units": {
                "time": "s",
                "linear_velocity": "m/s",
                "angular_velocity": "rad/s",
            },
        },
        "unsupported_semantics": list(UNSUPPORTED_SEMANTICS),
        "summary": {
            "steps": len(rows),
            "duration_s": len(rows) * config.timestep_s,
            "command_clip_fraction": clip_count / len(rows) if rows else 0.0,
            "max_linear_accel_m_s2": max_accel,
            "max_linear_decel_m_s2": max_decel,
            "max_abs_yaw_rate_rad_s": max_yaw,
            "final_pose": {
                "x_m": final_pose["x_m"],
                "y_m": final_pose["y_m"],
                "yaw_rad": final_pose["yaw_rad"],
            },
        },
        "rows": rows,
    }


def _normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    """Write a compact Markdown summary for the diagnostic payload."""
    summary = payload["summary"]
    lines = [
        "# MuJoCo AMV Micro-Backend Diagnostic",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"Claim boundary: {payload['claim_boundary']}",
        "",
        "## Summary",
        "",
        f"- steps: {summary['steps']}",
        f"- duration_s: {summary['duration_s']:.3f}",
        f"- command_clip_fraction: {summary['command_clip_fraction']:.4f}",
        f"- max_linear_accel_m_s2: {summary['max_linear_accel_m_s2']:.4f}",
        f"- max_linear_decel_m_s2: {summary['max_linear_decel_m_s2']:.4f}",
        f"- max_abs_yaw_rate_rad_s: {summary['max_abs_yaw_rate_rad_s']:.4f}",
        "",
        "## Unsupported Semantics",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["unsupported_semantics"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = ReplayConfig(
            timestep_s=float(args.timestep),
            max_linear_accel_m_s2=float(args.max_linear_accel),
            max_linear_decel_m_s2=float(args.max_linear_decel),
            max_angular_accel_rad_s2=float(args.max_angular_accel),
            max_yaw_rate_rad_s=float(args.max_yaw_rate),
            latency_steps=int(args.latency_steps),
        )
        segments, source_trace = _load_commands(
            args.commands,
            trace_path=args.trace,
            demo_fixture=bool(args.demo_fixture),
            timestep_s=config.timestep_s,
        )
        payload = replay_commands(
            segments,
            config,
            source_trace=source_trace,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(args.output_md, payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
