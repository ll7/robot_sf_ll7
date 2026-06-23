"""Bounded diagnostic for ScenarioBelief uncertainty semantics (issue #2546).

This diagnostic answers one bounded question on a tiny fixed scenario set:
does ScenarioBelief uncertainty change policy-observation semantics, the
stream_gap planner decision, or safety-relevant failure predicates -- and where
does uncertainty consumption fail closed?

It compares five belief conditions against an oracle baseline:

1. ``oracle``                  -- deterministic observation (zero covariance,
                                   full visibility, existence 1.0, single class).
2. ``visibility_limited``      -- one corridor pedestrian marked OCCLUDED so it
                                   drops out of the visibility-filtered projection.
3. ``covariance_inflated``     -- large isotropic position covariance.
4. ``class_probability``       -- spread / ambiguous class probabilities.
5. ``existence_degraded``      -- low existence-confidence on the corridor agent.

For each condition the diagnostic emits:

* an uncertainty report (``ScenarioBelief.to_uncertainty_report()``),
* a projected policy-observation summary (SOCNAV_STRUCT shape),
* a planner behavior + compatibility summary for a *consuming* planner
  (``stream_gap``, with uncertainty gating enabled) and an *unsupported*
  planner key (fail-closed), and
* failure-predicate changes versus the oracle condition.

CLAIM BOUNDARY: diagnostic only / stress evidence. This does NOT claim real
sensor validation, perception-model fidelity, trained-policy behavior, planner
performance improvement, SNQI movement, safety improvement, or any
benchmark/paper-grade result. Unsupported planner consumption is reported as a
fail-closed status, never as success.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

import numpy as np

from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.planner.scenario_belief_adapter import (
    SCENARIO_BELIEF_PLANNER_PROJECTION_SCHEMA_VERSION,
    project_scenario_belief_for_planner,
)
from robot_sf.planner.stream_gap import StreamGapPlannerAdapter, StreamGapPlannerConfig
from robot_sf.representation import (
    Estimate2D,
    ScenarioBelief,
    VisibilityState,
    scenario_belief_from_simulator_oracle,
)

DIAGNOSTIC_SCHEMA_VERSION = "scenario-belief-uncertainty-diagnostic.v1"
ISSUE = 2546
CONSUMING_PLANNER_KEY = "stream_gap"
UNSUPPORTED_PLANNER_KEY = "predictive_planner_v2"
DEFAULT_SEED = 2546

# The corridor pedestrian is the agent the planner actually reacts to; it is the
# one we perturb across uncertainty conditions so behavior changes are legible.
_CORRIDOR_AGENT_INDEX = 0


@contextmanager
def _temporary_numpy_seed(seed: int) -> Iterator[None]:
    """Set NumPy's global RNG seed for this diagnostic, then restore caller state."""
    previous_state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(previous_state)


def _oracle_belief() -> ScenarioBelief:
    """Build a deterministic oracle ScenarioBelief for one fixed crossing scenario.

    The scenario places one pedestrian inside the goal corridor (the agent the
    stream_gap planner reacts to) and one pedestrian off to the side.

    Returns:
        ScenarioBelief: Oracle belief with near-zero synthetic uncertainty.
    """
    simulator = SimpleNamespace(
        ped_pos=np.array([[1.5, 0.0], [6.0, 5.0]], dtype=np.float32),
        ped_vel=np.array([[0.0, 0.6], [0.0, 0.0]], dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=0.4),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=12.0, height=10.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )
    return scenario_belief_from_simulator_oracle(
        simulator,
        env_config=RobotSimulationConfig(),
        max_pedestrians=4,
    )


def _with_corridor_agent(belief: ScenarioBelief, new_agent: Any) -> ScenarioBelief:
    """Return a belief with the corridor agent replaced, others unchanged."""
    agents = list(belief.agents)
    agents[_CORRIDOR_AGENT_INDEX] = new_agent
    return replace(belief, agents=tuple(agents))


def _condition_oracle(oracle: ScenarioBelief) -> ScenarioBelief:
    """Return the oracle condition unchanged (baseline)."""
    return oracle


def _condition_visibility_limited(oracle: ScenarioBelief) -> ScenarioBelief:
    """Mark the corridor agent OCCLUDED so it leaves the visibility projection."""
    agent = oracle.agents[_CORRIDOR_AGENT_INDEX]
    occluded = replace(
        agent,
        visibility_state=VisibilityState.OCCLUDED,
        last_observed_age_s=1.0,
        missing_fields=("policy_position", "policy_velocity"),
    )
    return _with_corridor_agent(oracle, occluded)


def _condition_covariance_inflated(oracle: ScenarioBelief) -> ScenarioBelief:
    """Inflate the corridor agent position covariance well above the gate bound."""
    agent = oracle.agents[_CORRIDOR_AGENT_INDEX]
    inflated = replace(
        agent,
        position=Estimate2D.point(
            agent.position.mean_xy,
            confidence=agent.position.confidence,
            variance=4.0,
        ),
    )
    return _with_corridor_agent(oracle, inflated)


def _condition_class_probability(oracle: ScenarioBelief) -> ScenarioBelief:
    """Spread the corridor agent class probabilities below the gate threshold."""
    agent = oracle.agents[_CORRIDOR_AGENT_INDEX]
    ambiguous = replace(
        agent,
        class_probabilities=(("pedestrian", 0.3), ("cyclist", 0.4), ("unknown", 0.3)),
    )
    return _with_corridor_agent(oracle, ambiguous)


def _condition_existence_degraded(oracle: ScenarioBelief) -> ScenarioBelief:
    """Degrade the corridor agent existence confidence below the gate threshold."""
    agent = oracle.agents[_CORRIDOR_AGENT_INDEX]
    degraded = replace(agent, existence_probability=0.2)
    return _with_corridor_agent(oracle, degraded)


CONDITION_BUILDERS = {
    "oracle": _condition_oracle,
    "visibility_limited": _condition_visibility_limited,
    "covariance_inflated": _condition_covariance_inflated,
    "class_probability": _condition_class_probability,
    "existence_degraded": _condition_existence_degraded,
}


def _observation_summary(observation: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, JSON-safe summary of a SOCNAV_STRUCT projection."""
    pedestrians = observation.get("pedestrians", {})
    count = float(np.asarray(pedestrians.get("count", [0.0]), dtype=float).reshape(-1)[0])
    positions = np.asarray(pedestrians.get("positions"), dtype=float)
    nonzero_rows = int(np.count_nonzero(np.any(positions != 0.0, axis=1)))
    return {
        "policy_keys": sorted(observation.keys()),
        "pedestrian_count": count,
        "nonzero_pedestrian_rows": nonzero_rows,
        "first_pedestrian_position": [round(float(v), 6) for v in positions[0]]
        if positions.size
        else [],
        "robot_position": [
            round(float(v), 6)
            for v in np.asarray(
                observation.get("robot", {}).get("position", []), dtype=float
            ).reshape(-1)[:2]
        ],
    }


def _failure_predicates(
    *,
    command: tuple[float, float],
    gate: dict[str, Any],
    observation_summary: dict[str, Any],
) -> dict[str, Any]:
    """Derive safety-relevant boolean failure predicates from a planner outcome.

    These are diagnostic predicates over the planner decision and uncertainty
    gate, not validated safety metrics.

    Returns:
        dict[str, Any]: Boolean / scalar predicate values for this condition.
    """
    linear, angular = command
    return {
        "is_waiting": bool(abs(linear) < 1e-6),
        "is_committing": bool(linear >= 0.9),
        "linear_speed": round(float(linear), 6),
        "angular_speed": round(float(angular), 6),
        "gate_dropped_corridor_agent": bool(int(gate.get("dropped_count", 0)) > 0),
        "gate_dropped_count": int(gate.get("dropped_count", 0)),
        "corridor_agent_in_projection": bool(
            observation_summary.get("nonzero_pedestrian_rows", 0) > 0
        ),
    }


def _predicate_diff(oracle_pred: dict[str, Any], cond_pred: dict[str, Any]) -> dict[str, Any]:
    """Return per-predicate changes between the oracle and a condition."""
    changed: dict[str, Any] = {}
    for key in sorted(set(oracle_pred) | set(cond_pred)):
        ov = oracle_pred.get(key)
        cv = cond_pred.get(key)
        if ov != cv:
            changed[key] = {"oracle": ov, "condition": cv}
    return changed


def _run_condition(
    name: str,
    belief: ScenarioBelief,
    *,
    consuming_config: StreamGapPlannerConfig,
) -> dict[str, Any]:
    """Run one belief condition through both planners and collect diagnostics.

    Returns:
        dict[str, Any]: Per-condition uncertainty report, observation summary,
        planner behavior, and compatibility status.
    """
    uncertainty_report = belief.to_uncertainty_report()

    # Consuming planner: stream_gap with uncertainty gating enabled.
    consuming_projection = project_scenario_belief_for_planner(
        belief, planner_key=CONSUMING_PLANNER_KEY
    )
    consuming_planner = StreamGapPlannerAdapter(consuming_config)
    consuming_command = consuming_planner.plan(consuming_projection.observation)
    gate = dict(consuming_planner.last_uncertainty_gate)
    obs_summary = _observation_summary(consuming_projection.observation)

    # Unsupported planner: must fail closed (no uncertainty sidecar consumed).
    unsupported_projection = project_scenario_belief_for_planner(
        belief, planner_key=UNSUPPORTED_PLANNER_KEY
    )

    predicates = _failure_predicates(
        command=consuming_command,
        gate=gate,
        observation_summary=obs_summary,
    )

    return {
        "condition": name,
        "uncertainty_report": uncertainty_report,
        "policy_observation_summary": obs_summary,
        "consuming_planner": {
            "planner_key": CONSUMING_PLANNER_KEY,
            "compatibility": consuming_projection.compatibility,
            "command_linear": round(float(consuming_command[0]), 6),
            "command_angular": round(float(consuming_command[1]), 6),
            "uncertainty_gate": gate,
        },
        "unsupported_planner": {
            "planner_key": UNSUPPORTED_PLANNER_KEY,
            "compatibility": unsupported_projection.compatibility,
            "fail_closed": unsupported_projection.compatibility.get("status") == "fail_closed",
            "uncertainty_consumed": bool(
                unsupported_projection.compatibility.get("uncertainty_consumed", False)
            ),
        },
        "failure_predicates": predicates,
    }


def _git_head() -> str:
    """Return the current git HEAD SHA, or 'unknown' when unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _follow_up_decision(
    *,
    any_behavior_difference: bool,
    fail_closed_ok: bool,
) -> dict[str, str]:
    """Choose an honest continue|revise|stop follow-up decision.

    Returns:
        dict[str, str]: ``decision`` plus a short ``rationale``.
    """
    if not fail_closed_ok:
        return {
            "decision": "revise",
            "rationale": (
                "Unsupported planner did not fail closed as required; the uncertainty "
                "consumption contract must be revised before continuing."
            ),
        }
    if any_behavior_difference:
        return {
            "decision": "continue",
            "rationale": (
                "Uncertainty conditions changed at least one planner decision or failure "
                "predicate versus oracle while unsupported consumption failed closed. The "
                "representation carries decision-relevant uncertainty; a runtime producer + "
                "end-to-end stress run is the next bounded step. Diagnostic only."
            ),
        }
    return {
        "decision": "revise",
        "rationale": (
            "Null result: no condition changed a planner decision or failure predicate "
            "versus oracle. The uncertainty representation does not yet influence the "
            "consuming planner; revise the uncertainty consumption mapping before further work."
        ),
    }


def run_diagnostic(*, seed: int) -> dict[str, Any]:
    """Run the full diagnostic across all conditions and return the report dict.

    Returns:
        dict[str, Any]: Deterministic JSON-ready diagnostic report.
    """
    with _temporary_numpy_seed(seed):
        oracle = _oracle_belief()

        # The consuming planner opts into uncertainty gating so degraded uncertainty
        # can actually change which pedestrians it reacts to.
        consuming_config = StreamGapPlannerConfig(uncertainty_gating_enabled=True)

        conditions: dict[str, dict[str, Any]] = {}
        for name, builder in CONDITION_BUILDERS.items():
            belief = builder(oracle)
            conditions[name] = _run_condition(name, belief, consuming_config=consuming_config)

    oracle_predicates = conditions["oracle"]["failure_predicates"]
    predicate_diffs: dict[str, dict[str, Any]] = {}
    any_behavior_difference = False
    for name, result in conditions.items():
        if name == "oracle":
            continue
        diff = _predicate_diff(oracle_predicates, result["failure_predicates"])
        predicate_diffs[name] = diff
        if diff:
            any_behavior_difference = True

    # Every non-oracle uncertain condition must fail closed on the unsupported
    # planner, and the unsupported planner must never consume uncertainty.
    fail_closed_ok = all(
        result["unsupported_planner"]["fail_closed"]
        and not result["unsupported_planner"]["uncertainty_consumed"]
        for result in conditions.values()
    )

    follow_up = _follow_up_decision(
        any_behavior_difference=any_behavior_difference,
        fail_closed_ok=fail_closed_ok,
    )

    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "issue": ISSUE,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_head": _git_head(),
        "seed": seed,
        "claim_boundary": "diagnostic_only",
        "evidence_tier": "stress",
        "paper_grade": False,
        "not_benchmark_evidence": True,
        "planner_projection_schema": SCENARIO_BELIEF_PLANNER_PROJECTION_SCHEMA_VERSION,
        "consuming_planner_key": CONSUMING_PLANNER_KEY,
        "unsupported_planner_key": UNSUPPORTED_PLANNER_KEY,
        "conditions": conditions,
        "predicate_diffs_vs_oracle": predicate_diffs,
        "any_behavior_difference": any_behavior_difference,
        "unsupported_fail_closed_ok": fail_closed_ok,
        "follow_up_decision": follow_up,
        "limitations": [
            "Synthetic fixed scenario set; not drawn from real sensor data.",
            "No perception model and no trained policy are involved.",
            "Planner behavior differences are diagnostic decision-shifts, not "
            "validated safety or performance improvements.",
            "Failure predicates are derived from the planner command and uncertainty "
            "gate, not from an episode-level safety evaluation.",
            "stream_gap is the only uncertainty-consuming planner; all other planner "
            "keys fail closed by design.",
        ],
    }


def _render_markdown(report: dict[str, Any]) -> str:
    """Render a compact human-readable markdown summary of the diagnostic report."""
    lines: list[str] = []
    lines.append(f"# Issue #{ISSUE} ScenarioBelief Uncertainty Diagnostic")
    lines.append("")
    lines.append(f"- schema: `{report['schema_version']}`")
    lines.append(f"- git HEAD: `{report['git_head']}`")
    lines.append(f"- seed: `{report['seed']}`")
    lines.append(f"- generated (UTC): `{report['generated_at_utc']}`")
    lines.append(
        "- claim boundary: **diagnostic_only** / evidence_tier: **stress** / paper_grade: **false**"
    )
    lines.append(
        f"- consuming planner: `{report['consuming_planner_key']}` (uncertainty gating enabled)"
    )
    lines.append(f"- unsupported planner: `{report['unsupported_planner_key']}` (must fail closed)")
    lines.append("")
    lines.append("## Per-condition summary")
    lines.append("")
    lines.append(
        "| condition | consuming status | linear | angular | gate status | dropped | unsupported fail-closed |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, result in report["conditions"].items():
        cp = result["consuming_planner"]
        gate = cp["uncertainty_gate"]
        up = result["unsupported_planner"]
        lines.append(
            "| {name} | {status} | {lin} | {ang} | {gstatus} | {dropped} | {fc} |".format(
                name=name,
                status=cp["compatibility"].get("status"),
                lin=cp["command_linear"],
                ang=cp["command_angular"],
                gstatus=gate.get("status"),
                dropped=gate.get("dropped_count", 0),
                fc=up["fail_closed"],
            )
        )
    lines.append("")
    lines.append("## Failure-predicate changes vs oracle")
    lines.append("")
    diffs = report["predicate_diffs_vs_oracle"]
    any_diff = False
    for name, diff in diffs.items():
        if not diff:
            lines.append(f"- `{name}`: no change vs oracle")
            continue
        any_diff = True
        lines.append(f"- `{name}`:")
        for key, change in diff.items():
            lines.append(
                f"  - `{key}`: oracle=`{change['oracle']}` -> condition=`{change['condition']}`"
            )
    lines.append("")
    if any_diff:
        lines.append(
            "At least one uncertainty condition changed a planner decision or failure "
            "predicate versus oracle (difference found)."
        )
    else:
        lines.append(
            "NULL RESULT: no uncertainty condition changed a planner decision or failure "
            "predicate versus oracle."
        )
    lines.append("")
    lines.append("## Follow-up decision")
    lines.append("")
    fu = report["follow_up_decision"]
    lines.append(f"- decision: **{fu['decision']}**")
    lines.append(f"- rationale: {fu['rationale']}")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for lim in report["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the diagnostic runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Bounded ScenarioBelief uncertainty diagnostic (issue #2546). "
            "Diagnostic only / stress evidence; no benchmark, safety, or paper claim."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic seed for the diagnostic run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/issue_2546_belief_uncertainty"),
        help="Directory for JSON + markdown report artifacts.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the markdown summary to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: run the diagnostic and write JSON + markdown artifacts.

    Returns:
        int: Process exit code (0 on success).
    """
    args = _parse_args(argv)
    report = run_diagnostic(seed=args.seed)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "scenario_belief_uncertainty_diagnostic_issue_2546.json"
    md_path = output_dir / "scenario_belief_uncertainty_diagnostic_issue_2546.md"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = _render_markdown(report)
    md_path.write_text(markdown)

    fu = report["follow_up_decision"]
    print(f"[issue-2546] wrote {json_path}")
    print(f"[issue-2546] wrote {md_path}")
    print(
        "[issue-2546] any_behavior_difference="
        f"{report['any_behavior_difference']} "
        f"unsupported_fail_closed_ok={report['unsupported_fail_closed_ok']} "
        f"follow_up={fu['decision']}"
    )
    if args.print_summary:
        print()
        print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
