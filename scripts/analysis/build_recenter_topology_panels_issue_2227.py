#!/usr/bin/env python3
"""Contrastive mechanism panels for static recentering and topology-guided recovery.

Issue #2227 asks for contrastive mechanism panels that show, per mechanism:

* where the mechanism was **expected to act**,
* whether it **activated** (with the planner's own activation diagnostic),
* whether the selected **command/source changed**,
* whether the **outcome changed**,

while keeping observed evidence separate from hypothesis wording.

This script completes the two remaining #2227 sub-targets (the AMV/AMMV sub-target
was delivered in a sibling PR):

1. ``static-recenter`` panel (HybridRuleLocalPlanner ``static_recenter_enabled``)
2. ``topology-guided recovery`` panel
   (TopologyGuidedHybridRulePlannerAdapter ``topology_command_enabled``)

For each mechanism the planner is run **twice** on one fixed scenario with an
identical seed, toggling **only** the mechanism flag. The "on" arm uses a
scenario where the mechanism is *expected to act* (a static deadlock for
recentering; a bottleneck route-ambiguity slice for topology). Each arm is
exported as a schema-valid ``simulation_trace_export.v1`` document, validated by
the canonical loader, and rendered into a contrastive trajectory panel bundle.

HONESTY: traces come from actual planner runs. Only the mechanism flag differs
between arms. A mechanism that is expected here but does *not* activate, or
activates without changing the outcome, is reported as an honest null - the
script never fabricates a difference. This is **diagnostic-only / stress** tier,
planner-level evidence; it is not a navigation-success, benchmark, or perception
claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.simulation_trace_export import (
    load_simulation_trace_export,
)
from robot_sf.benchmark.map_runner import _run_map_episode
from robot_sf.benchmark.trajectory_panels import generate_trajectory_panel_bundle
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.run_policy_search_candidate import load_candidate_definition

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = Path("docs/context/policy_search/candidate_registry.yaml")
DEFAULT_OUTPUT_DIR = Path("output/issue_2227_recenter_topology")
CLAIM_BOUNDARY = "diagnostic_only"
EVIDENCE_TIER = "stress"


@dataclass(frozen=True)
class MechanismSpec:
    """Declarative description of one contrastive mechanism panel run."""

    mechanism_id: str
    flag: str
    algo: str
    candidate: str
    scenario_set: Path
    scenario_id: str
    seed: int
    horizon: int
    dt: float
    expected_reason: str


# Static-recentering activation-capable row from Issue #2592 controlled-trace
# evidence (classic_bottleneck_low, seed 113). Topology route-ambiguity slice
# (classic_bottleneck_medium) from the stress slice matrix used by the
# #2742/#2716 topology reselection diagnostics.
MECHANISMS: tuple[MechanismSpec, ...] = (
    MechanismSpec(
        mechanism_id="static_recenter",
        flag="static_recenter_enabled",
        algo="hybrid_rule_local_planner",
        candidate="issue_2170_static_recenter_only",
        scenario_set=Path("configs/scenarios/sets/issue_2592_static_deadlock_active_row_h500.yaml"),
        scenario_id="classic_bottleneck_low",
        seed=113,
        horizon=160,
        dt=0.1,
        expected_reason=(
            "static deadlock / local-minimum bottleneck where the recenter probe is "
            "expected to perturb the robot off the wall (Issue #2592 active row)"
        ),
    ),
    MechanismSpec(
        mechanism_id="topology_command",
        flag="topology_command_enabled",
        algo="topology_guided_hybrid_rule_v0",
        candidate="topology_guided_hybrid_rule_v0",
        scenario_set=Path("configs/policy_search/stress_slice_matrix.yaml"),
        scenario_id="classic_bottleneck_medium",
        seed=111,
        horizon=160,
        dt=0.1,
        expected_reason=(
            "bottleneck route-ambiguity slice where >=2 distinct masked-route "
            "hypotheses are expected, allowing a topology-hypothesis command to be "
            "selected (topology reselection hard slice)"
        ),
    ),
)


@dataclass
class ArmResult:
    """Per-arm captured trace, activation diagnostic, and terminal outcome."""

    arm: str  # "on" or "off"
    planner_id: str
    enabled: bool
    trace_path: Path
    frame_count: int
    activation: dict[str, Any]
    selected_source_counts: dict[str, int]
    terminal: dict[str, Any]


@dataclass
class MechanismResult:
    """Contrastive result for one mechanism (both arms plus the honest delta)."""

    spec: MechanismSpec
    on: ArmResult
    off: ArmResult
    activated: bool
    command_source_changed: bool
    outcome_changed: bool
    trajectory_delta_m: float
    classification: str
    null_reason: str | None = None
    panel_artifacts: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Durable (git-ignored) output root for traces and panels.",
    )
    parser.add_argument(
        "--candidate-registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Policy-search candidate registry used to seed the activation-capable config.",
    )
    parser.add_argument(
        "--mechanism",
        choices=[spec.mechanism_id for spec in MECHANISMS],
        action="append",
        help="Restrict to specific mechanism(s). Default: all.",
    )
    return parser.parse_args()


def _git_head() -> str:
    """Return the current git HEAD short SHA, or ``unknown`` when unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _load_scenario(scenario_set: Path, scenario_id: str) -> dict[str, Any]:
    """Load the single named scenario mapping from a scenario set."""
    matches = [
        dict(scenario)
        for scenario in load_scenarios(scenario_set)
        if str(scenario.get("name") or scenario.get("scenario_id") or scenario.get("id"))
        == scenario_id
    ]
    if not matches:
        raise SystemExit(f"FAIL-CLOSED: scenario '{scenario_id}' not found in {scenario_set}")
    return matches[0]


def _runtime_config(registry_path: Path, candidate: str) -> dict[str, Any]:
    """Return the merged base+params runtime config for a registry candidate."""
    _entry, _payload, config, _config_path = load_candidate_definition(registry_path, candidate)
    return dict(config)


def _decision_steps(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the planner decision-trace steps from an episode record."""
    metadata = record.get("algorithm_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    trace = metadata.get("planner_decision_trace")
    trace = trace if isinstance(trace, dict) else {}
    steps = trace.get("steps")
    return [step for step in steps if isinstance(step, dict)] if isinstance(steps, list) else []


def _sim_steps(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the simulation-step-trace frames from an episode record."""
    metadata = record.get("algorithm_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    trace = metadata.get("simulation_step_trace")
    if isinstance(trace, dict):
        trace = trace.get("steps")
    return [frame for frame in trace if isinstance(frame, dict)] if isinstance(trace, list) else []


def _topology_runtime(record: dict[str, Any]) -> dict[str, Any]:
    """Return the topology-guided planner runtime diagnostic, if present."""
    metadata = record.get("algorithm_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    runtime = metadata.get("planner_runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    topology = runtime.get("topology_guided")
    return topology if isinstance(topology, dict) else {}


def _selected_source_counts(decision_steps: list[dict[str, Any]]) -> dict[str, int]:
    """Tally selected command-source per decision step."""
    counts: dict[str, int] = {}
    for step in decision_steps:
        source = str(step.get("selected_source", "unknown"))
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def _terminal_outcome(record: dict[str, Any]) -> dict[str, Any]:
    """Return compact terminal-outcome fields for one episode record."""
    metrics = record.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    return {
        "status": record.get("status"),
        "termination_reason": record.get("termination_reason"),
        "success": bool(metrics.get("success", False)),
        "steps": int(record.get("steps", 0) or 0),
        "collisions": int(metrics.get("collisions", 0) or 0),
        "near_misses": int(metrics.get("near_misses", 0) or 0),
    }


def _activation_diagnostic(
    spec: MechanismSpec,
    decision_steps: list[dict[str, Any]],
    record: dict[str, Any],
) -> dict[str, Any]:
    """Return the planner's own activation diagnostic for the mechanism arm.

    Returns:
        dict[str, Any]: ``active`` flag plus the supporting raw counters.
    """
    if spec.mechanism_id == "static_recenter":
        recenter_steps = [
            int(step["step"])
            for step in decision_steps
            if float(step.get("static_recenter", 0.0)) > 0.0
        ]
        source_steps = [
            int(step["step"])
            for step in decision_steps
            if str(step.get("selected_source")) == "static_recenter"
        ]
        return {
            "diagnostic": "static_recenter_term_positive_in_decision_trace",
            "active": bool(recenter_steps),
            "recenter_term_activation_count": len(recenter_steps),
            "first_activation_step": recenter_steps[0] if recenter_steps else None,
            "static_recenter_command_selected_count": len(source_steps),
        }
    topology = _topology_runtime(record)
    status_counts = topology.get("status_counts") if isinstance(topology, dict) else {}
    status_counts = status_counts if isinstance(status_counts, dict) else {}
    source_counts = _selected_source_counts(decision_steps)
    topology_command_count = int(source_counts.get("topology_hypothesis", 0))
    return {
        "diagnostic": "topology_status_counts_and_topology_hypothesis_source",
        "active": topology_command_count > 0,
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "topology_hypothesis_command_count": topology_command_count,
        "selected_hypothesis_counts": {
            str(key): int(value)
            for key, value in (topology.get("selected_hypothesis_counts") or {}).items()
        },
        "topology_command_enabled": bool(topology.get("topology_command_enabled", False)),
    }


def _enriched_planner_payload(
    sim_frame: dict[str, Any],
    decision_step: dict[str, Any] | None,
    *,
    activation_step: bool,
) -> dict[str, Any]:
    """Merge selected-source / activation flags into a sim-trace planner payload.

    Returns:
        dict[str, Any]: Schema-valid ``planner`` payload with extra diagnostics.
    """
    planner = sim_frame.get("planner")
    planner = dict(planner) if isinstance(planner, dict) else {}
    if "selected_action" not in planner:
        planner["selected_action"] = {"linear_velocity": 0.0, "angular_velocity": 0.0}
    planner.setdefault("event", "step")
    if decision_step is not None:
        planner["selected_source"] = str(decision_step.get("selected_source", "unknown"))
        planner["static_recenter_term"] = float(decision_step.get("static_recenter", 0.0))
        score = decision_step.get("selected_score")
        if isinstance(score, int | float):
            planner["selected_score"] = float(score)
    planner["mechanism_active_this_step"] = bool(activation_step)
    return planner


def _build_trace_export(
    *,
    spec: MechanismSpec,
    arm: str,
    enabled: bool,
    planner_id: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Build a schema-valid ``simulation_trace_export.v1`` document for one arm.

    Returns:
        dict[str, Any]: The export payload (not yet written to disk).
    """
    sim_frames = _sim_steps(record)
    decision_steps = {int(step["step"]): step for step in _decision_steps(record)}
    if not sim_frames:
        raise SystemExit(
            f"FAIL-CLOSED: {spec.mechanism_id} arm '{arm}' produced no simulation-step frames"
        )

    frames: list[dict[str, Any]] = []
    for frame in sim_frames:
        step_idx = int(frame.get("step", 0))
        decision_step = decision_steps.get(step_idx)
        if spec.mechanism_id == "static_recenter":
            activation_step = bool(
                decision_step is not None and float(decision_step.get("static_recenter", 0.0)) > 0.0
            )
        else:
            activation_step = bool(
                decision_step is not None
                and str(decision_step.get("selected_source")) == "topology_hypothesis"
            )
        robot = frame.get("robot")
        robot = dict(robot) if isinstance(robot, dict) else {}
        peds_raw = frame.get("pedestrians")
        pedestrians: list[dict[str, Any]] = []
        for index, ped in enumerate(peds_raw if isinstance(peds_raw, list) else []):
            if not isinstance(ped, dict):
                continue
            entry: dict[str, Any] = {
                "id": str(ped.get("id", f"ped_{index}")),
                "position": [float(ped["position"][0]), float(ped["position"][1])],
                "velocity": [float(ped["velocity"][0]), float(ped["velocity"][1])],
            }
            frames_radius = ped.get("radius")
            if isinstance(frames_radius, int | float):
                entry["radius"] = float(frames_radius)
            pedestrians.append(entry)
        frames.append(
            {
                "step": step_idx,
                "time_s": float(frame.get("time_s", (step_idx + 1) * spec.dt)),
                "robot": {
                    "position": [
                        float(robot["position"][0]),
                        float(robot["position"][1]),
                    ],
                    "heading": float(robot.get("heading", 0.0)),
                    "velocity": [
                        float(robot["velocity"][0]),
                        float(robot["velocity"][1]),
                    ],
                },
                "pedestrians": pedestrians,
                "planner": _enriched_planner_payload(
                    frame, decision_step, activation_step=activation_step
                ),
            }
        )

    return {
        "schema_version": "simulation_trace_export.v1",
        "trace_id": f"issue_2227_{spec.mechanism_id}_{arm}",
        "source": {
            "scenario_id": spec.scenario_id,
            "seed": spec.seed,
            "planner_id": planner_id,
            "episode_id": f"{spec.mechanism_id}_{arm}_seed{spec.seed}",
            "generated_by": "scripts/analysis/build_recenter_topology_panels_issue_2227.py",
        },
        "evidence_boundary": "analysis_workbench_only",
        "coordinate_frame": "world",
        "units": {"position": "m", "heading": "rad", "time": "s", "velocity": "m/s"},
        "frames": frames,
    }


def _run_arm(
    spec: MechanismSpec,
    scenario: dict[str, Any],
    base_config: dict[str, Any],
    *,
    arm: str,
    enabled: bool,
    traces_dir: Path,
) -> ArmResult:
    """Run one mechanism arm, export and validate its trace, return diagnostics.

    Returns:
        ArmResult: Captured trace path and activation/terminal diagnostics.
    """
    config = deepcopy(base_config)
    config[spec.flag] = enabled
    record = _run_map_episode(
        scenario,
        spec.seed,
        horizon=spec.horizon,
        dt=spec.dt,
        record_forces=True,
        snqi_weights=None,
        snqi_baseline=None,
        algo=spec.algo,
        algo_config=config,
        scenario_path=spec.scenario_set,
        record_planner_decision_trace=True,
        record_simulation_step_trace=True,
    )
    decision_steps = _decision_steps(record)
    planner_id = f"{spec.algo}_{spec.flag}_{'on' if enabled else 'off'}"
    export = _build_trace_export(
        spec=spec, arm=arm, enabled=enabled, planner_id=planner_id, record=record
    )
    trace_path = traces_dir / f"{spec.mechanism_id}_{arm}.json"
    trace_path.write_text(json.dumps(export, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Fail closed if the export does not load cleanly against the schema.
    load_simulation_trace_export(trace_path)
    return ArmResult(
        arm=arm,
        planner_id=planner_id,
        enabled=enabled,
        trace_path=trace_path,
        frame_count=len(export["frames"]),
        activation=_activation_diagnostic(spec, decision_steps, record),
        selected_source_counts=_selected_source_counts(decision_steps),
        terminal=_terminal_outcome(record),
    )


def _final_xy(arm: ArmResult) -> tuple[float, float]:
    """Return the final robot xy position from an arm's exported trace."""
    trace = load_simulation_trace_export(arm.trace_path)
    last = trace.frames[-1].robot.get("position")
    return float(last[0]), float(last[1])


def _classify(spec: MechanismSpec, on: ArmResult, off: ArmResult) -> MechanismResult:
    """Build the honest contrastive classification for a mechanism.

    Returns:
        MechanismResult: Contrastive result with activation, deltas, null reason.
    """
    activated = bool(on.activation.get("active", False))
    command_source_changed = on.selected_source_counts != off.selected_source_counts
    outcome_changed = on.terminal != off.terminal
    on_xy = _final_xy(on)
    off_xy = _final_xy(off)
    trajectory_delta = ((on_xy[0] - off_xy[0]) ** 2 + (on_xy[1] - off_xy[1]) ** 2) ** 0.5

    null_reason: str | None = None
    if not activated:
        classification = "expected_here_did_not_activate"
        null_reason = (
            "Mechanism was expected to act in this scenario but its activation "
            "diagnostic never fired; treat as an honest null (no fabricated delta)."
        )
    elif outcome_changed:
        classification = "activated_outcome_changed"
    elif command_source_changed or trajectory_delta > 1e-6:
        classification = "activated_trace_changed_outcome_unchanged"
    else:
        classification = "activated_no_observable_change"
        null_reason = (
            "Mechanism activated but neither command source, trajectory, nor terminal "
            "outcome changed; honest null delta."
        )
    return MechanismResult(
        spec=spec,
        on=on,
        off=off,
        activated=activated,
        command_source_changed=command_source_changed,
        outcome_changed=outcome_changed,
        trajectory_delta_m=trajectory_delta,
        classification=classification,
        null_reason=null_reason,
    )


def _caption(result: MechanismResult) -> str:
    """Return the contrastive caption for one mechanism panel bundle."""
    spec = result.spec
    activation = result.on.activation
    lines = [
        f"## {spec.mechanism_id} contrastive mechanism panel (Issue #2227)",
        "",
        f"- Mechanism flag toggled (isolation): `{spec.flag}` (on vs off, only this flag differs).",
        f"- Scenario: `{spec.scenario_id}` (seed {spec.seed}, horizon {spec.horizon}, dt {spec.dt}).",
        f"- Expected to act here? YES - {spec.expected_reason}.",
        f"- Activated? {'YES' if result.activated else 'NO'} "
        f"(diagnostic: `{activation.get('diagnostic')}`; raw={json.dumps({k: v for k, v in activation.items() if k != 'diagnostic'}, sort_keys=True)}).",
        f"- Command/source changed between arms? {'YES' if result.command_source_changed else 'NO'} "
        f"(on={json.dumps(result.on.selected_source_counts, sort_keys=True)}; "
        f"off={json.dumps(result.off.selected_source_counts, sort_keys=True)}).",
        f"- Outcome changed? {'YES' if result.outcome_changed else 'NO'} "
        f"(on={json.dumps(result.on.terminal, sort_keys=True)}; "
        f"off={json.dumps(result.off.terminal, sort_keys=True)}).",
        f"- Final-pose trajectory delta: {result.trajectory_delta_m:.4f} m.",
        f"- Classification: `{result.classification}`.",
    ]
    if result.null_reason:
        lines.append(f"- Honest null: {result.null_reason}")
    lines += [
        "",
        f"Claim boundary: `{CLAIM_BOUNDARY}` / evidence tier `{EVIDENCE_TIER}`. "
        "Planner-level activation accounting only; NOT a navigation-success, benchmark, "
        "ranking, or perception claim. Traces come from actual planner runs; only the "
        "mechanism flag differs between arms.",
    ]
    return "\n".join(lines)


def _run_mechanism(
    spec: MechanismSpec,
    *,
    registry_path: Path,
    output_dir: Path,
    commit: str,
    command: str,
) -> MechanismResult:
    """Execute both arms for one mechanism and render its contrastive panel bundle.

    Returns:
        MechanismResult: Contrastive result including rendered panel artifact ids.
    """
    scenario = _load_scenario(spec.scenario_set, spec.scenario_id)
    base_config = _runtime_config(registry_path, spec.candidate)
    traces_dir = output_dir / "traces" / spec.mechanism_id
    traces_dir.mkdir(parents=True, exist_ok=True)

    off_arm = _run_arm(spec, scenario, base_config, arm="off", enabled=False, traces_dir=traces_dir)
    on_arm = _run_arm(spec, scenario, base_config, arm="on", enabled=True, traces_dir=traces_dir)
    result = _classify(spec, on_arm, off_arm)

    panel_dir = output_dir / "panels" / spec.mechanism_id
    bundle = generate_trajectory_panel_bundle(
        trace_paths=[off_arm.trace_path, on_arm.trace_path],
        output_dir=panel_dir,
        command=command,
        commit=commit,
    )
    pdf_count = 0
    for artifact in bundle.artifacts:
        result.panel_artifacts.append(artifact.artifact_id)
        if not artifact.png_path.exists() or not artifact.pdf_path.exists():
            raise SystemExit(
                f"FAIL-CLOSED: {spec.mechanism_id} panel '{artifact.artifact_id}' "
                "missing PNG or PDF output"
            )
        pdf_count += 1
    if pdf_count == 0:
        raise SystemExit(f"FAIL-CLOSED: {spec.mechanism_id} produced no rendered panels")

    caption_path = panel_dir / "mechanism_caption.md"
    caption_path.write_text(_caption(result) + "\n", encoding="utf-8")
    return result


def _write_selection_csv(path: Path, results: list[MechanismResult]) -> None:
    """Write a compact per-arm selection CSV for the evidence bundle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mechanism_id",
                "arm",
                "flag",
                "enabled",
                "planner_id",
                "frame_count",
                "activated",
                "terminal_success",
                "terminal_steps",
                "termination_reason",
                "trace_path",
            ]
        )
        for result in results:
            for arm in (result.off, result.on):
                writer.writerow(
                    [
                        result.spec.mechanism_id,
                        arm.arm,
                        result.spec.flag,
                        arm.enabled,
                        arm.planner_id,
                        arm.frame_count,
                        bool(arm.activation.get("active", False)),
                        arm.terminal.get("success"),
                        arm.terminal.get("steps"),
                        arm.terminal.get("termination_reason"),
                        arm.trace_path.as_posix(),
                    ]
                )


def _summary_payload(
    results: list[MechanismResult], *, commit: str, command: str
) -> dict[str, Any]:
    """Build the machine-readable run summary payload."""
    return {
        "schema_version": "issue_2227_recenter_topology_panels.v1",
        "issue": 2227,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_tier": EVIDENCE_TIER,
        "paper_grade": False,
        "commit": commit,
        "command": command,
        "mechanisms": [
            {
                "mechanism_id": result.spec.mechanism_id,
                "flag": result.spec.flag,
                "scenario_id": result.spec.scenario_id,
                "seed": result.spec.seed,
                "horizon": result.spec.horizon,
                "dt": result.spec.dt,
                "expected_to_act": True,
                "expected_reason": result.spec.expected_reason,
                "activated": result.activated,
                "activation_diagnostic": result.on.activation,
                "command_source_changed": result.command_source_changed,
                "outcome_changed": result.outcome_changed,
                "trajectory_delta_m": result.trajectory_delta_m,
                "classification": result.classification,
                "null_reason": result.null_reason,
                "on_terminal": result.on.terminal,
                "off_terminal": result.off.terminal,
                "on_selected_source_counts": result.on.selected_source_counts,
                "off_selected_source_counts": result.off.selected_source_counts,
                "panel_artifacts": result.panel_artifacts,
                "on_trace": result.on.trace_path.as_posix(),
                "off_trace": result.off.trace_path.as_posix(),
            }
            for result in results
        ],
    }


def main() -> None:
    """Build contrastive static-recenter and topology mechanism panels."""
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = _git_head()
    command = "uv run python scripts/analysis/build_recenter_topology_panels_issue_2227.py"

    selected = args.mechanism or [spec.mechanism_id for spec in MECHANISMS]
    specs = [spec for spec in MECHANISMS if spec.mechanism_id in selected]

    results: list[MechanismResult] = []
    for spec in specs:
        result = _run_mechanism(
            spec,
            registry_path=args.candidate_registry,
            output_dir=output_dir,
            commit=commit,
            command=command,
        )
        results.append(result)

    _write_selection_csv(output_dir / "representative_episode_selection.csv", results)
    summary = _summary_payload(results, commit=commit, command=command)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
