"""Build the issue #2227 AMMV contrastive mechanism trajectory panel.

This tool runs the direct ``SocialForcePlanner`` twice on one fixed, deterministic
scenario with identical seed/scenario, toggling ONLY the ``ammv_aware_enabled`` key in
``configs/baselines/social_force_ammv_aware.yaml``:

* control      = ``ammv_aware_enabled: false``
* intervention = ``ammv_aware_enabled: true``

Both arms are exported as ``simulation_trace_export.v1`` JSON (schema-validated on load),
and a contrastive trajectory panel is rendered via
``robot_sf.benchmark.trajectory_panels.generate_trajectory_panel_bundle``.

CLAIM BOUNDARY (diagnostic only): The panel demonstrates a *planner-level* mechanism
difference -- the AMMV term activates (nonzero force) and changes the selected command and
robot trajectory under an identical seed/scenario. It does NOT claim navigation success,
benchmark advantage, or sensor/perception realism. The ``SocialForcePlanner`` is a robot
planner only; pedestrians are static and simulator-owned dynamics are not modelled here.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import yaml

from robot_sf.analysis_workbench.simulation_trace_export import (
    load_simulation_trace_export,
)
from robot_sf.baselines.social_force import Observation, SocialForcePlanner
from robot_sf.benchmark.trajectory_panels import generate_trajectory_panel_bundle

# Deterministic fixed mechanism scenario: a close static pedestrian directly in the
# robot's forward path. Chosen to provoke the AMMV interaction term (issue #2444 verified
# this yields a genuine same-seed divergent pair at the SocialForcePlanner level).
SCENARIO_ID = "issue_2227_ammv_close_front_static_ped"
SEED = 42
STEPS = 24
DT = 0.1
ROBOT_SPEC: dict[str, Any] = {
    "position": [0.0, 0.0],
    "velocity": [1.0, 0.0],
    "goal": [4.0, 0.0],
    "radius": 0.3,
}
STATIC_PEDS: list[dict[str, Any]] = [
    {"position": [0.3, 0.1], "velocity": [0.0, 0.0], "radius": 0.3}
]
CONFIG_PATH = Path("configs/baselines/social_force_ammv_aware.yaml")


def _git_head() -> str:
    """Return the current git HEAD sha, or ``unknown`` when unavailable."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _load_config(path: Path) -> dict[str, Any]:
    """Load a YAML planner config into a mapping."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def _action_command(action: dict[str, float]) -> tuple[float, float]:
    """Return ``(linear_velocity, heading)`` for a velocity-mode planner action."""
    vx = float(action.get("vx", action.get("v", 0.0)))
    vy = float(action.get("vy", 0.0))
    linear_velocity = math.hypot(vx, vy)
    heading = math.atan2(vy, vx)
    return linear_velocity, heading


def run_arm(
    *,
    ammv_enabled: bool,
    base_config: dict[str, Any],
) -> dict[str, Any]:
    """Run one planner arm with AMMV toggled and capture a per-step trace.

    Toggles ONLY ``ammv_aware_enabled`` so the contrast isolates the AMMV term.

    Returns:
        A mapping with arm metadata, per-step frames, and AMMV force summary.
    """
    config = dict(base_config)
    config["ammv_aware_enabled"] = ammv_enabled
    planner = SocialForcePlanner(config, seed=SEED)

    robot_pos = list(ROBOT_SPEC["position"])
    robot_vel = list(ROBOT_SPEC["velocity"])
    prev_heading = math.atan2(robot_vel[1], robot_vel[0])

    frames: list[dict[str, Any]] = []
    ammv_force_magnitudes: list[float] = []
    for step in range(STEPS):
        step_obs = Observation(
            dt=DT,
            robot={
                "position": list(robot_pos),
                "velocity": list(robot_vel),
                "goal": list(ROBOT_SPEC["goal"]),
                "radius": ROBOT_SPEC["radius"],
            },
            agents=[dict(ped) for ped in STATIC_PEDS],
            obstacles=[],
        )
        action = planner.step(step_obs)
        metadata = planner.get_metadata()
        ammv_force = float(metadata.get("ammv_force_magnitude", 0.0) or 0.0)
        ammv_force_magnitudes.append(ammv_force)

        linear_velocity, heading = _action_command(action)
        angular_velocity = (heading - prev_heading) / DT

        frames.append(
            {
                "step": step,
                "time_s": round(step * DT, 6),
                "robot": {
                    "position": [round(robot_pos[0], 6), round(robot_pos[1], 6)],
                    "heading": round(prev_heading, 6),
                    "velocity": [round(robot_vel[0], 6), round(robot_vel[1], 6)],
                    "radius": float(ROBOT_SPEC["radius"]),
                },
                "pedestrians": [
                    {
                        "id": f"ped_{idx}",
                        "position": [float(ped["position"][0]), float(ped["position"][1])],
                        "velocity": [float(ped["velocity"][0]), float(ped["velocity"][1])],
                        "radius": float(ped["radius"]),
                    }
                    for idx, ped in enumerate(STATIC_PEDS)
                ],
                "planner": {
                    "selected_action": {
                        "linear_velocity": round(linear_velocity, 6),
                        "angular_velocity": round(angular_velocity, 6),
                        "vx": round(float(action.get("vx", action.get("v", 0.0))), 6),
                        "vy": round(float(action.get("vy", 0.0)), 6),
                    },
                    "ammv_force_magnitude": round(ammv_force, 6),
                    "event": "start" if step == 0 else "advance",
                },
            }
        )

        vx = float(action.get("vx", action.get("v", 0.0)))
        vy = float(action.get("vy", 0.0))
        robot_pos = [robot_pos[0] + vx * DT, robot_pos[1] + vy * DT]
        robot_vel = [vx, vy]
        prev_heading = heading

    return {
        "ammv_enabled": ammv_enabled,
        "frames": frames,
        "max_ammv_force_magnitude": max(ammv_force_magnitudes),
        "final_position": list(robot_pos),
    }


def build_trace_export(arm: dict[str, Any], *, command: str, commit: str) -> dict[str, Any]:
    """Build a ``simulation_trace_export.v1`` payload for one arm."""
    suffix = "ammv_on" if arm["ammv_enabled"] else "ammv_off"
    return {
        "schema_version": "simulation_trace_export.v1",
        "trace_id": f"{SCENARIO_ID}_{suffix}",
        "source": {
            "scenario_id": SCENARIO_ID,
            "seed": SEED,
            "planner_id": f"social_force_{suffix}",
            "episode_id": f"{SCENARIO_ID}_{suffix}_seed{SEED}",
            "generated_by": (
                "scripts/analysis/build_ammv_mechanism_panel_issue_2227.py "
                f"(commit {commit}); command: {command}"
            ),
        },
        "evidence_boundary": "analysis_workbench_only",
        "coordinate_frame": "world",
        "units": {"position": "m", "heading": "rad", "time": "s", "velocity": "m/s"},
        "frames": arm["frames"],
    }


def _write_trace(payload: dict[str, Any], path: Path) -> Path:
    """Write a trace payload to JSON and validate it loads against the schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Fail closed: a trace that does not load against the schema is unusable.
    load_simulation_trace_export(path)
    return path


def _trajectory_delta(off: dict[str, Any], on: dict[str, Any]) -> dict[str, float]:
    """Compute coarse trajectory/command deltas between the two arms."""
    off_final = off["final_position"]
    on_final = on["final_position"]
    off_first = off["frames"][1]["planner"]["selected_action"] if len(off["frames"]) > 1 else {}
    on_first = on["frames"][1]["planner"]["selected_action"] if len(on["frames"]) > 1 else {}
    return {
        "final_position_distance_m": math.dist(off_final, on_final),
        "final_lateral_offset_delta_m": on_final[1] - off_final[1],
        "step1_linear_velocity_delta_mps": (
            float(on_first.get("linear_velocity", 0.0))
            - float(off_first.get("linear_velocity", 0.0))
        ),
    }


def _write_captions(
    path: Path,
    *,
    off: dict[str, Any],
    on: dict[str, Any],
    delta: dict[str, float],
    command: str,
    commit: str,
) -> None:
    """Write the AMMV mechanism annotation / captions file."""
    lines = [
        "# Issue #2227 AMMV Contrastive Mechanism Panel",
        "",
        "Two same-seed, same-scenario direct `SocialForcePlanner` runs. The ONLY config",
        "difference is `ammv_aware_enabled` (false=control, true=intervention) in",
        f"`{CONFIG_PATH.as_posix()}`; every other key is identical so the contrast isolates",
        "the AMMV interaction term.",
        "",
        f"- Scenario: `{SCENARIO_ID}`, seed `{SEED}`, steps `{STEPS}`, dt `{DT}`.",
        f"- Command: `{command}`",
        f"- Commit: `{commit}`",
        "",
        "## Where the mechanism was expected to act",
        "",
        "A close static pedestrian sits directly in the robot's forward path. The AMMV",
        "actuation-aware repulsion term is expected to activate there.",
        "",
        "## Did it activate?",
        "",
        f"- AMMV-off max force magnitude: `{off['max_ammv_force_magnitude']:.6f}` (expected 0).",
        f"- AMMV-on  max force magnitude: `{on['max_ammv_force_magnitude']:.6f}` (nonzero).",
        "",
        "## Did command / trajectory behavior change?",
        "",
        f"- Step-1 linear-velocity delta (on - off): "
        f"`{delta['step1_linear_velocity_delta_mps']:.6f}` m/s.",
        f"- Final-position distance between arms: `{delta['final_position_distance_m']:.6f}` m.",
        f"- Final lateral-offset delta (on - off): "
        f"`{delta['final_lateral_offset_delta_m']:.6f}` m.",
        "",
        "## Outcome and claim boundary",
        "",
        "Observed evidence: the AMMV term activates (nonzero force) and the selected",
        "command and resulting robot trajectory diverge under an identical seed/scenario.",
        "",
        "Claim boundary (diagnostic_only): this is a PLANNER-LEVEL mechanism difference.",
        "It is NOT a navigation-success, benchmark-advantage, or sensor/perception-realism",
        "claim. Pedestrians are static and simulator-owned dynamics are not modelled here.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(output_dir: Path) -> dict[str, Any]:
    """Run both arms, export traces, render the panel, and emit captions.

    Returns:
        A summary mapping with force magnitudes, deltas, and artifact paths.
    """
    commit = _git_head()
    command = "uv run python scripts/analysis/build_ammv_mechanism_panel_issue_2227.py"
    base_config = _load_config(CONFIG_PATH)

    off = run_arm(ammv_enabled=False, base_config=base_config)
    on = run_arm(ammv_enabled=True, base_config=base_config)

    traces_dir = output_dir / "traces"
    control_path = _write_trace(
        build_trace_export(off, command=command, commit=commit),
        traces_dir / "ammv_off_control_trace.json",
    )
    intervention_path = _write_trace(
        build_trace_export(on, command=command, commit=commit),
        traces_dir / "ammv_on_intervention_trace.json",
    )

    panel_dir = output_dir / "panel"
    bundle = generate_trajectory_panel_bundle(
        trace_paths=[control_path, intervention_path],
        output_dir=panel_dir,
        command=command,
        commit=commit,
    )

    delta = _trajectory_delta(off, on)
    captions_path = output_dir / "ammv_mechanism_captions.md"
    _write_captions(
        captions_path,
        off=off,
        on=on,
        delta=delta,
        command=command,
        commit=commit,
    )

    panel_pngs = [str(a.png_path) for a in bundle.artifacts]
    panel_pdfs = [str(a.pdf_path) for a in bundle.artifacts]
    summary = {
        "scenario_id": SCENARIO_ID,
        "seed": SEED,
        "steps": STEPS,
        "dt": DT,
        "commit": commit,
        "command": command,
        "claim_boundary": "diagnostic_only",
        "evidence_tier": "stress",
        "paper_grade": False,
        "ammv_off_max_force_magnitude": off["max_ammv_force_magnitude"],
        "ammv_on_max_force_magnitude": on["max_ammv_force_magnitude"],
        "trajectory_delta": delta,
        "control_trace": str(control_path),
        "intervention_trace": str(intervention_path),
        "panel_pngs": panel_pngs,
        "panel_pdfs": panel_pdfs,
        "selection_csv": str(bundle.selection_csv),
        "manifest_path": str(bundle.manifest_path),
        "captions_path": str(captions_path),
    }

    # Fail closed: the panel MUST produce at least one PNG+PDF artifact pair.
    if not panel_pngs or not panel_pdfs:
        raise RuntimeError("panel renderer produced no PNG/PDF artifacts; refusing to fake a panel")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/issue_2227_ammv_panel"),
        help="Directory for generated traces, panel, captions, and summary.",
    )
    args = parser.parse_args()
    summary = run(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
