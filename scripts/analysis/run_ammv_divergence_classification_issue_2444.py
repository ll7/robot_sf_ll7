"""Issue #2444 AMMV / default Social Force divergence classification.

Background
----------
Issue #2434 already closed with a *negative* result. Its scenario sweep compared 15 matched
default-vs-AMMV episode pairs through the differential-drive benchmark *adapter* and found
``max_per_frame_abs_delta = 0.0`` and ``max_episode_metric_abs_delta = 0.0`` for every pair
(see ``docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/``). Those pairs are
therefore rendering fixtures, NOT behavioral-difference evidence.

This tool runs a *more sensitive* slice: a deterministic, direct ``SocialForcePlanner`` mechanism
probe (reused from ``scripts/tools/run_ammv_social_force_pair_diagnostic.py``) that bypasses the
benchmark adapter and exposes the AMMV force term directly. For each probe it records the compact
``ammv_divergence_selection`` block and classifies the outcome as one of:

* ``nonzero_divergence_found`` -- at least one probe has an AMMV force > 0 AND a nonzero paired
  delta between the default and AMMV traces (a genuine same-seed behavioral difference).
* ``ammv_inactive_under_tested_settings`` -- every probe shows zero AMMV force / zero delta, i.e.
  the #2434 negative baseline reproduces even on the more sensitive direct surface.
* ``blocked_missing_instrumentation`` -- a concrete, named missing input prevents the slice from
  running at all.

Claim boundary: diagnostic_only. This is not benchmark-strength or paper-facing evidence. A
nonzero divergence here proves the AMMV term is *active and behaviorally non-identical at the
direct-planner level*; it does NOT claim a benchmark advantage.

Zero-divergence guard
---------------------
``classify_divergence`` refuses to label a probe as behavioral evidence unless its AMMV force is
strictly positive AND its paired delta is strictly nonzero. This is the machine-checkable guard
that stops future #2159 / #2227 mechanism panels from treating an identical (zero-divergence)
trace pair as behavioral evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

# Reuse the existing, verified probe machinery instead of reinventing it.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts" / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "tools"))

from run_ammv_social_force_pair_diagnostic import (  # noqa: E402
    _load_config,
    _mechanism_probe_spec,
    _run_mechanism_probe,
)

SCHEMA_VERSION = "issue_2444_ammv_divergence_classification.v1"
DEFAULT_AMMV_CONFIG = Path("configs/baselines/social_force_ammv_aware.yaml")
DEFAULT_OUTPUT_DIR = Path("output/issue_2444_ammv_divergence")
ISSUE_2434_EVIDENCE = "docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/"

# Deterministic, more-sensitive slice: the two existing direct mechanism probes.
DEFAULT_PROBES = (
    "issue_2168_close_front_agent_probe",
    "issue_3202_anticipatory_crossing_probe",
)

RESULT_CLASSIFICATIONS = (
    "nonzero_divergence_found",
    "ammv_inactive_under_tested_settings",
    "blocked_missing_instrumentation",
)

# Numerical floor below which a delta is treated as identical (matches the probe machinery).
DELTA_EPSILON = 1e-9


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _max_abs_delta(paired_delta: dict[str, float]) -> float:
    return max(
        (abs(delta) for v in paired_delta.values() if math.isfinite(delta := float(v))),
        default=0.0,
    )


def build_selection_block(probe_result: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Construct the compact ``ammv_divergence_selection`` block for one probe.

    The robot/pedestrian/action deltas are derived from the same-seed default-vs-AMMV traces. The
    pedestrian-state delta is reported as ``None`` because the direct robot-planner probe does not
    own pedestrian trajectories (documented limitation, not a hidden zero).
    """
    paired_delta = probe_result["paired_delta"]
    ammv_trace = probe_result["traces"]["ammv_social_force"]
    max_force = float(ammv_trace["max_ammv_force_magnitude"])
    # Robot-state delta: pooled max over kinematic deltas the probe actually computes.
    robot_state_keys = (
        "mean_robot_speed_mps",
        "max_abs_lateral_velocity_mps",
        "final_robot_lateral_offset_m",
        "min_robot_ped_clearance_m",
    )
    max_robot_state_delta = max(
        (
            abs(delta)
            for k in robot_state_keys
            if k in paired_delta and math.isfinite(delta := float(paired_delta[k]))
        ),
        default=0.0,
    )
    # The "selected action" surface here is the robot speed/lateral-velocity response.
    max_selected_action_delta = max(
        (
            abs(delta)
            for key in ("mean_robot_speed_mps", "max_abs_lateral_velocity_mps")
            if math.isfinite(delta := float(paired_delta.get(key, 0.0)))
        ),
        default=0.0,
    )
    mechanism_activation_observed = max_force > 0.0
    behavioral = mechanism_activation_observed and _max_abs_delta(paired_delta) > DELTA_EPSILON
    # outcome_changed: did the min-clearance sign flip / change materially between traces?
    default_trace = probe_result["traces"]["default_social_force"]
    outcome_changed = (
        abs(
            float(ammv_trace["min_robot_ped_clearance_m"])
            - float(default_trace["min_robot_ped_clearance_m"])
        )
        > DELTA_EPSILON
    )
    return {
        "scenario_id": probe_result["name"],
        "seed": seed,
        "default_candidate": (
            "ammv config with ammv_aware_enabled=false (AMMV term off; all other params identical)"
        ),
        "ammv_candidate": "ammv config with ammv_aware_enabled=true (AMMV term on)",
        "frame_count": int(probe_result["steps"]),
        "max_robot_state_delta": max_robot_state_delta,
        # Pedestrian state is simulator-owned in this robot-planner-only probe; not measured here.
        "max_pedestrian_state_delta": None,
        "max_selected_action_delta": max_selected_action_delta,
        "max_ammv_force_delta": max_force,
        "outcome_changed": outcome_changed,
        "mechanism_activation_observed": mechanism_activation_observed,
        "result_classification": (
            "nonzero_divergence_found" if behavioral else "ammv_inactive_under_tested_settings"
        ),
    }


def classify_divergence(selection_blocks: list[dict[str, Any]]) -> str:
    """Aggregate per-probe selections into a single ``result_classification``.

    Zero-divergence guard: a block only counts as behavioral evidence when its AMMV force is
    strictly positive AND its per-probe classification is ``nonzero_divergence_found``. An empty
    list (no probe could run) is treated as a missing-instrumentation block by the caller.
    """
    if not selection_blocks:
        return "blocked_missing_instrumentation"
    any_behavioral = any(
        block["mechanism_activation_observed"]
        and block["result_classification"] == "nonzero_divergence_found"
        for block in selection_blocks
    )
    if any_behavioral:
        return "nonzero_divergence_found"
    return "ammv_inactive_under_tested_settings"


def run_classification(
    *,
    ammv_config: Path,
    probes: tuple[str, ...] = DEFAULT_PROBES,
) -> dict[str, Any]:
    """Run the more-sensitive direct-probe slice and build the full summary payload."""
    if not ammv_config.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "classification": "diagnostic",
            "claim_boundary": "diagnostic_only",
            "result_classification": "blocked_missing_instrumentation",
            "blocker": (
                f"AMMV planner config not found at '{ammv_config.as_posix()}'. "
                "A real divergence slice cannot run without an AMMV-enabled SocialForce config."
            ),
            "issue_2434_baseline": ISSUE_2434_EVIDENCE,
            "git_head": _git_head(),
            "ammv_config": ammv_config.as_posix(),
            "selections": [],
        }

    # Isolate the AMMV term: the control arm is the SAME config with ``ammv_aware_enabled``
    # toggled off, so the paired delta reflects only the AMMV interaction term -- not unrelated
    # differences between bare ``SFPlannerConfig()`` defaults and the AMMV-aware config.
    ammv_payload = _load_config(ammv_config)
    control_payload = {**ammv_payload, "ammv_aware_enabled": False}

    selections: list[dict[str, Any]] = []
    for probe_name in probes:
        spec = _mechanism_probe_spec(probe_name)
        result = _run_mechanism_probe(
            ammv_config, probe_name=probe_name, default_config=control_payload
        )
        selections.append(build_selection_block(result, seed=int(spec["seed"])))

    result_classification = classify_divergence(selections)
    return {
        "schema_version": SCHEMA_VERSION,
        "title": "Issue #2444 AMMV Divergence Classification",
        "classification": "diagnostic",
        "claim_boundary": "diagnostic_only",
        "benchmark_evidence": False,
        "paper_facing": False,
        "git_head": _git_head(),
        "ammv_config": ammv_config.as_posix(),
        "issue_2434_baseline": {
            "path": ISSUE_2434_EVIDENCE,
            "result": "max_per_frame_abs_delta=0.0, max_episode_metric_abs_delta=0.0 over 15 "
            "matched default-vs-AMMV adapter-mode pairs (rendering fixtures, not behavioral "
            "evidence).",
        },
        "slice_description": (
            "Direct SocialForcePlanner mechanism probes (bypassing the differential-drive "
            "benchmark adapter that produced the #2434 zero-delta result). The control arm is the "
            "same AMMV-aware config with ammv_aware_enabled toggled off, so the paired delta "
            "isolates the AMMV interaction term."
        ),
        "result_classification": result_classification,
        "selections": selections,
        "zero_divergence_guard": (
            "A probe is treated as behavioral evidence only when max_ammv_force_delta > 0 AND a "
            "paired delta is strictly nonzero. Identical (zero-divergence) pairs are rejected."
        ),
        "limitations": [
            "Direct robot-planner probe only; pedestrian dynamics are simulator-owned and not "
            "measured here (max_pedestrian_state_delta is None by construction).",
            "Diagnostic-only: a nonzero divergence proves the AMMV term is active and "
            "behaviorally non-identical at the planner level, not a benchmark advantage.",
            "The control arm toggles only ammv_aware_enabled on the same config, so the paired "
            "delta isolates the AMMV term and is not confounded by other config differences.",
            "Two-probe slice; not a benchmark matrix or parameter sweep.",
        ],
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        f"# {summary.get('title', 'Issue #2444 AMMV Divergence Classification')}",
        "",
        f"- Result classification: `{summary['result_classification']}`.",
        f"- Claim boundary: `{summary['claim_boundary']}`.",
        f"- Git HEAD: `{summary['git_head']}`.",
        f"- AMMV config: `{summary['ammv_config']}`.",
        "",
        "## #2434 negative baseline (cited)",
        "",
    ]
    baseline = summary.get("issue_2434_baseline")
    if isinstance(baseline, dict):
        lines += [f"- Path: `{baseline['path']}`.", f"- Result: {baseline['result']}", ""]
    else:
        lines += [f"- Path: `{baseline}`.", ""]
    lines += ["## Per-probe `ammv_divergence_selection` blocks", ""]
    for block in summary.get("selections", []):
        lines += [
            f"### `{block['scenario_id']}` (seed `{block['seed']}`)",
            "",
            f"- frame_count: `{block['frame_count']}`.",
            f"- max_robot_state_delta: `{block['max_robot_state_delta']:.6g}`.",
            f"- max_selected_action_delta: `{block['max_selected_action_delta']:.6g}`.",
            f"- max_ammv_force_delta: `{block['max_ammv_force_delta']:.6g}`.",
            f"- mechanism_activation_observed: `{block['mechanism_activation_observed']}`.",
            f"- outcome_changed: `{block['outcome_changed']}`.",
            f"- per-probe classification: `{block['result_classification']}`.",
            "",
        ]
    lines += ["## Zero-divergence guard", "", summary.get("zero_divergence_guard", ""), ""]
    lines += ["## Limitations", ""]
    lines += [f"- {item}" for item in summary.get("limitations", [])]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "ammv_divergence_classification.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(summary, output_dir / "README.md")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the divergence classification runner."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ammv-config",
        type=Path,
        default=DEFAULT_AMMV_CONFIG,
        help="AMMV-enabled SocialForce planner config (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the JSON+markdown evidence pack (default: %(default)s).",
    )
    parser.add_argument(
        "--probe",
        dest="probes",
        action="append",
        choices=sorted(DEFAULT_PROBES),
        help="Mechanism probe(s) to run; repeatable. Defaults to the full deterministic slice.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the divergence classification slice and write the evidence pack."""
    args = build_parser().parse_args(argv)
    probes = tuple(args.probes) if args.probes else DEFAULT_PROBES
    summary = run_classification(ammv_config=args.ammv_config, probes=probes)
    _write_outputs(summary, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": args.output_dir.as_posix(),
                "result_classification": summary["result_classification"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
