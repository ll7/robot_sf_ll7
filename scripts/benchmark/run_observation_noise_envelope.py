#!/usr/bin/env python3
"""Near-field observation-noise robustness envelope from an occluded-emergence trace.

Reads the durable occluded-emergence trace fixture and evaluates near-field
observation-noise perturbation conditions against the robot-pedestrian pair.
This is diagnostic trace-derived evidence only, not a full planner benchmark.

Usage::

    uv run python scripts/benchmark/run_observation_noise_envelope.py \\
        --output-dir docs/context/evidence/issue_2755_observation_noise_envelope_2026-06-13
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
from typing import Any

import numpy as np

from robot_sf.benchmark.observation_perturbation import (
    ObservationPerturbationSpec,
    ObservationPerturbationState,
    perturb_ground_truth,
)
from robot_sf.benchmark.observation_quality import ObservationQuality

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    REPO_ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1/"
    "occluded_emergence_episode_0000.json"
)
OBSERVATION_QUALITY_SCHEMA_VERSION = "observation_quality.v1"
TRACE_DT_S = 0.1

CONDITIONS: dict[str, dict[str, Any]] = {
    "noop": {"description": "No perturbation applied (baseline)."},
    "low_noise": {
        "description": "Bounded Gaussian position noise with std=0.10 m, bound=0.20 m.",
        "spec_kw": {"position_noise_std_m": 0.10, "position_noise_bound_m": 0.20, "seed": 2755},
    },
    "medium_noise": {
        "description": "Bounded Gaussian position noise with std=0.30 m, bound=0.60 m.",
        "spec_kw": {"position_noise_std_m": 0.30, "position_noise_bound_m": 0.60, "seed": 2755},
    },
    "missed_detection_only": {
        "description": "Single pedestrian fully missed (probability=1.0).",
        "spec_kw": {"missed_detection_probability": 1.0, "seed": 2755},
    },
    "false_positive_only": {
        "description": "One observed-only pedestrian injected into the replay observation.",
        "spec_kw": {
            "false_positive_positions": [[26.0, 7.0]],
            "false_positive_velocities": [[0.0, 0.0]],
            "false_positive_ids": ["false_positive_0"],
        },
    },
    "occlusion_only": {
        "description": "Single pedestrian position/velocity zeroed by occlusion mask.",
        "spec_kw": {"occlusion_mask": np.array([True])},
    },
    "delay_only": {
        "description": "2-step delayed observation for the single pedestrian.",
        "spec_kw": {"delay_steps": 2},
    },
    "combined": {
        "description": "Medium Gaussian noise + occlusion mask on the pedestrian.",
        "spec_kw": {
            "position_noise_std_m": 0.30,
            "position_noise_bound_m": 0.60,
            "occlusion_mask": np.array([True]),
            "seed": 2755,
        },
    },
}


def _load_fixture(path: pathlib.Path) -> dict[str, Any]:
    """Load the occluded-emergence trace fixture."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _git_head() -> str:
    """Return the short git HEAD, or empty string on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=5,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _extract_frame_arrays(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract ground-truth robot position and pedestrian arrays from a frame.

    Returns:
        (robot_pos, ped_pos, ped_ids)
    """
    robot_pos = np.array(frame["robot"]["position"], dtype=np.float64)
    ped_positions = np.array([p["position"] for p in frame["pedestrians"]], dtype=np.float64)
    ped_ids = [str(p["id"]) for p in frame["pedestrians"]]
    return robot_pos, ped_positions, ped_ids


def _extract_observed_arrays(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract fixture-visible pedestrian observations from a frame."""
    observed_peds = frame.get("observed_pedestrians", [])
    if not observed_peds:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            [],
        )
    positions = np.array([p["position"] for p in observed_peds], dtype=np.float64)
    velocities = np.array([p["velocity"] for p in observed_peds], dtype=np.float64)
    ids = [str(p["id"]) for p in observed_peds]
    return positions, velocities, ids


def _closest_distance(robot_pos: np.ndarray, ped_pos: np.ndarray) -> float:
    """Compute Euclidean distance from robot to the nearest pedestrian."""
    if ped_pos.size == 0:
        return float("inf")
    diffs = ped_pos - robot_pos
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    return float(np.min(dists))


def _spec_for_actor_count(
    spec: ObservationPerturbationSpec, actor_count: int
) -> ObservationPerturbationSpec:
    """Adapt fixed-size occlusion masks to frames before fixture visibility."""
    if spec.occlusion_mask is None:
        return spec
    if np.asarray(spec.occlusion_mask).size == actor_count:
        return spec
    if actor_count == 0:
        return ObservationPerturbationSpec(
            position_noise_std_m=spec.position_noise_std_m,
            position_noise_bound_m=spec.position_noise_bound_m,
            missed_detection_probability=spec.missed_detection_probability,
            occlusion_mask=np.zeros(0, dtype=bool),
            delay_steps=spec.delay_steps,
            seed=spec.seed,
        )
    return spec


def _stop_yield_feasibility(frame: dict[str, Any]) -> dict[str, bool]:
    """Extract stop and yield feasibility from frame conflict_timing."""
    ct = frame.get("conflict_timing", {})
    return {
        "stop_feasible": bool(ct.get("stop_feasible", False)),
        "yield_feasible": bool(ct.get("yield_feasible", False)),
    }


def _selected_action_proxy(frame: dict[str, Any]) -> dict[str, Any]:
    """Extract selected action and event from frame planner payload."""
    planner = frame.get("planner", {})
    return {
        "linear_velocity": planner.get("selected_action", {}).get("linear_velocity"),
        "angular_velocity": planner.get("selected_action", {}).get("angular_velocity"),
        "event": planner.get("event"),
    }


def evaluate_condition(
    condition_name: str,
    condition_cfg: dict[str, Any],
    frames: list[dict[str, Any]],
    first_visible_step: int,
) -> dict[str, Any]:
    """Evaluate one perturbation condition against the trace frames.

    Returns a compact result dict with condition metadata and aggregated classification.
    """
    spec_kw = dict(condition_cfg.get("spec_kw", {}))
    spec = ObservationPerturbationSpec(**spec_kw)
    needs_state = spec.delay_steps > 0
    state = ObservationPerturbationState(delay_steps=spec.delay_steps) if needs_state else None

    first_observed_step: int | None = None
    response_delay_steps: int | None = None
    missed_counts: list[int] = []
    occluded_counts: list[int] = []
    false_positive_counts: list[int] = []
    closest_distances: list[float] = []
    action_proxies: list[dict[str, Any]] = []
    observed_steps: list[int] = []
    perturbed_observed_steps: list[int] = []
    feasibility_by_step: dict[int, dict[str, bool]] = {}

    for frame in frames:
        step = frame["step"]
        robot_pos, ground_truth_ped_positions, _ped_ids = _extract_frame_arrays(frame)
        observed_positions, observed_velocities, observed_ids = _extract_observed_arrays(frame)
        if observed_positions.size > 0:
            observed_steps.append(step)
        step_spec = _spec_for_actor_count(spec, len(observed_ids))

        perturb_result = perturb_ground_truth(
            observed_positions,
            observed_velocities,
            observed_ids,
            spec=step_spec,
            step=step,
            state=state,
        )

        obs = perturb_result["observed"]
        meta = perturb_result["metadata"]

        # Determine if pedestrian is "observed" (non-zero position in observed state)
        ped_observed = False
        if obs["positions"].size > 0:
            ped_observed = bool(np.any(obs["positions"] != 0.0))

        if ped_observed and first_observed_step is None:
            first_observed_step = step
            if first_visible_step is not None:
                response_delay_steps = step - first_visible_step
        if ped_observed:
            perturbed_observed_steps.append(step)

        closest_dist = _closest_distance(robot_pos, ground_truth_ped_positions)
        fy = _stop_yield_feasibility(frame)
        action_proxy = _selected_action_proxy(frame)
        feasibility_by_step[step] = fy

        missed_counts.append(meta["missed_actor_count"])
        occluded_counts.append(meta["occluded_actor_count"])
        false_positive_counts.append(meta["false_positive_actor_count"])
        closest_distances.append(closest_dist)
        action_proxies.append(action_proxy)

    # Classification logic
    classification = _classify_condition(
        condition_name=condition_name,
        first_observed_step=first_observed_step,
        first_visible_step=first_visible_step,
        response_delay_steps=response_delay_steps,
        closest_distances=closest_distances,
        missed_counts=missed_counts,
        occluded_counts=occluded_counts,
    )

    result = {
        "condition": condition_name,
        "description": condition_cfg["description"],
        "spec": _spec_summary(spec),
        "first_observed_step": first_observed_step,
        "response_delay_steps": response_delay_steps,
        "first_visible_step_reference": first_visible_step,
        "total_frames": len(frames),
        "fixture_observed_steps": observed_steps,
        "perturbed_observed_steps": perturbed_observed_steps,
        "missed_actor_observations_total": sum(missed_counts),
        "occluded_actor_observations_total": sum(occluded_counts),
        "false_positive_actor_observations_total": sum(false_positive_counts),
        "delay_steps_configured": spec.delay_steps,
        "closest_distance_m": round(min(closest_distances), 4) if closest_distances else None,
        "stop_yield_feasibility": {
            "stop_feasible_first_observed": (
                feasibility_by_step[first_observed_step]["stop_feasible"]
                if first_observed_step is not None
                else None
            ),
            "yield_feasible_first_observed": (
                feasibility_by_step[first_observed_step]["yield_feasible"]
                if first_observed_step is not None
                else None
            ),
        },
        "action_proxy_changes": _action_proxy_summary(action_proxies),
        "classification": classification,
    }
    result["safety_effects"] = _safety_effect_summary(result)
    return result


def _spec_summary(spec: ObservationPerturbationSpec) -> dict[str, Any]:
    """Summarize a perturbation spec as a plain dict."""
    return {
        "position_noise_std_m": spec.position_noise_std_m,
        "position_noise_bound_m": spec.position_noise_bound_m,
        "missed_detection_probability": spec.missed_detection_probability,
        "has_occlusion_mask": spec.occlusion_mask is not None,
        "false_positive_actor_count": spec.false_positive_actor_count,
        "delay_steps": spec.delay_steps,
        "noise_profile": spec.noise_profile,
        "is_noop": spec.is_noop,
        "observation_quality": _observation_quality_group(spec),
    }


def _observation_quality_group(spec: ObservationPerturbationSpec) -> dict[str, Any]:
    """Return the bounded observation-quality field group for one perturbation spec."""

    quality = ObservationQuality(
        visibility=["trace_fixture_observed_pedestrians"],
        occlusion=(
            ["explicit_occlusion_mask"]
            if spec.occlusion_mask is not None
            else ["fixture_declared_visibility_boundary"]
        ),
        latency_s=float(spec.delay_steps) * TRACE_DT_S,
        dropout_probability=float(spec.missed_detection_probability),
        range_limit_m=None,
        angular_noise_std_rad=0.0,
        false_negative_rate=float(spec.missed_detection_probability),
        false_positive_rate=1.0 if spec.false_positive_actor_count > 0 else 0.0,
        notes=(
            "Diagnostic simulator observation-quality metadata only; "
            "not hardware-calibrated sensor realism."
        ),
    )
    return {
        "schema_version": OBSERVATION_QUALITY_SCHEMA_VERSION,
        "fields": quality.to_dict(),
    }


def _classify_condition(
    *,
    condition_name: str,
    first_observed_step: int | None,
    first_visible_step: int,
    response_delay_steps: int | None,
    closest_distances: list[float],
    missed_counts: list[int],
    occluded_counts: list[int],
) -> dict[str, str]:
    """Classify the condition result into one of the allowed categories."""
    # Noop is always diagnostic_only (it is the reference)
    if condition_name == "noop":
        return {
            "label": "diagnostic_only",
            "rationale": (
                "No-perturbation baseline. Provides reference robot-pedestrian "
                "trajectory and action selection without observation noise."
            ),
        }

    never_observed = first_observed_step is None
    has_missed_observations = any(c > 0 for c in missed_counts)
    has_occluded_observations = any(c > 0 for c in occluded_counts)

    # If pedestrian never observed (full occlusion or full miss)
    if never_observed:
        if has_missed_observations:
            return {
                "label": "scenario_too_weak",
                "rationale": (
                    "Pedestrian fully missed after fixture visibility begins; no observation "
                    "signal reaches the policy. Cannot test policy robustness."
                ),
            }
        if has_occluded_observations:
            return {
                "label": "scenario_too_weak",
                "rationale": (
                    "Pedestrian position/velocity zeroed by occlusion after fixture visibility "
                    "begins; no observation signal reaches the policy."
                ),
            }

    min_dist = min(closest_distances) if closest_distances else float("inf")

    if first_observed_step is not None and min_dist > 2.0:
        return {
            "label": "scenario_too_weak",
            "rationale": (
                f"Closest robot-pedestrian distance ({min_dist:.2f} m) exceeds "
                "near-field threshold (2.0 m). Scenario too weak for observation-noise "
                "sensitivity test."
            ),
        }

    # If perturbation caused observable differences, it is robustness evidence
    if (
        first_observed_step is not None
        and response_delay_steps is not None
        and response_delay_steps > 0
    ):
        return {
            "label": "robustness_evidence",
            "rationale": (
                f"Pedestrian observed at step {first_observed_step} "
                f"({response_delay_steps} steps after first-visible). "
                "Perturbation delayed/masked the observation and may have affected "
                "policy response timing."
            ),
        }

    return {
        "label": "diagnostic_only",
        "rationale": (
            "Perturbation produced mixed effects. Classified as diagnostic-only "
            "pending broader seed/scenario evidence."
        ),
    }


def _safety_effect_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Summarize false-negative and false-positive safety effects honestly."""

    missed_total = int(result["missed_actor_observations_total"])
    occluded_total = int(result["occluded_actor_observations_total"])
    false_positive_total = int(result["false_positive_actor_observations_total"])
    delay_steps = result["response_delay_steps"]
    if result["first_observed_step"] is None and (missed_total > 0 or occluded_total > 0):
        false_negative_effect = "full_miss_or_occlusion"
        false_negative_rationale = (
            "The actor never reached the observed state after the fixture visibility boundary."
        )
    elif delay_steps is not None and delay_steps > 0:
        false_negative_effect = "delayed_detection"
        false_negative_rationale = (
            f"The actor reached the observed state {delay_steps} steps after first visibility."
        )
    elif missed_total > 0 or occluded_total > 0:
        false_negative_effect = "partial_observation_loss"
        false_negative_rationale = "Some actor observations were missed or occluded."
    else:
        false_negative_effect = "none_observed"
        false_negative_rationale = "No missed or occluded actor observations were recorded."

    return {
        "false_negative": {
            "effect": false_negative_effect,
            "missed_actor_observations_total": missed_total,
            "occluded_actor_observations_total": occluded_total,
            "rationale": false_negative_rationale,
        },
        "false_positive": {
            "effect": "actor_injected" if false_positive_total > 0 else "none_observed",
            "false_positive_actor_observations_total": false_positive_total,
            "rationale": (
                "Observed-only actors were injected into the perturbation replay."
                if false_positive_total > 0
                else "No false-positive actor observations were recorded."
            ),
        },
    }


def _action_proxy_summary(action_proxies: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize stored-trace action proxies without embedding every action."""
    if not action_proxies:
        return {"linear_velocity_changed": False, "events": [], "velocity_range": None}
    events = [ap.get("event") for ap in action_proxies]
    unique_events = list(dict.fromkeys(events))  # preserve order, dedupe
    velocities = [v for ap in action_proxies if (v := ap.get("linear_velocity")) is not None]
    changed = len(set(velocities)) > 1
    return {
        "linear_velocity_changed": changed,
        "events": unique_events,
        "velocity_range": [min(velocities), max(velocities)] if velocities else None,
    }


def _build_report(
    results: list[dict[str, Any]],
    fixture_meta: dict[str, Any],
    repro: dict[str, Any],
    *,
    issue: int | None = None,
) -> dict[str, Any]:
    """Build the full JSON report."""
    issue_id = int(issue if issue is not None else repro.get("issue", 2755))
    return {
        "schema_version": "observation_noise_envelope.v1",
        "issue": issue_id,
        "claim_boundary": (
            "Diagnostic trace-derived evidence only. Not paper-facing benchmark proof. "
            "Evaluates near-field observation-noise robustness on a single "
            "occluded-emergence trace fixture."
        ),
        "reproducibility": repro,
        "fixture": fixture_meta,
        "conditions": results,
        "summary": {
            "total_conditions": len(results),
            "classifications": {r["condition"]: r["classification"]["label"] for r in results},
            "safety_effects": {r["condition"]: r["safety_effects"] for r in results},
        },
    }


def _generate_markdown(
    results: list[dict[str, Any]],
    fixture_meta: dict[str, Any],
    repro: dict[str, Any],
) -> str:
    """Generate the Markdown evidence report."""
    lines: list[str] = [
        "# Observation-Noise Robustness Envelope",
        "",
        "## Claim Boundary",
        "",
        "**Diagnostic trace-derived evidence only. Not paper-facing benchmark proof.**",
        "This evaluates near-field observation-noise robustness on a single "
        "occluded-emergence trace fixture. Results are diagnostic, not statistically "
        "powered population claims.",
        "",
        "## Reproducibility",
        "",
        f"- **Issue:** #{repro['issue']}",
        f"- **Generated at (UTC):** {repro['generated_at_utc']}",
        f"- **Command:** `{repro['command']}`",
        f"- **Repo HEAD:** `{repro['repo_head']}`",
        f"- **Fixture:** `{fixture_meta['trace_path']}`",
        f"- **Scenario:** {fixture_meta.get('scenario_id', '')}",
        f"- **First visible step:** {fixture_meta.get('first_visible_step', '?')}",
        "",
        "## Conditions",
        "",
    ]

    for r in results:
        spec = r["spec"]
        lines.append(f"### {r['condition']}")
        lines.append("")
        lines.append(f"- **Description:** {r['description']}")
        lines.append(f"- **Noise profile:** {spec['noise_profile']}")
        lines.append(f"- **First observed step:** {r['first_observed_step']}")
        if r["response_delay_steps"] is not None:
            lines.append(
                f"- **Response delay:** {r['response_delay_steps']} steps from first-visible"
            )
        lines.append(
            f"- **Closest distance:** {r['closest_distance_m']}"
            if r["closest_distance_m"] is not None
            else "- **Closest distance:** N/A"
        )
        fy = r["stop_yield_feasibility"]
        lines.append(f"- **Stop feasible (first observed):** {fy['stop_feasible_first_observed']}")
        lines.append(
            f"- **Yield feasible (first observed):** {fy['yield_feasible_first_observed']}"
        )
        effects = r["safety_effects"]
        lines.append(f"- **False-negative safety effect:** `{effects['false_negative']['effect']}`")
        lines.append(f"- **False-positive safety effect:** `{effects['false_positive']['effect']}`")
        cl = r["classification"]
        lines.append(f"- **Classification:** `{cl['label']}`")
        lines.append(f"  - {cl['rationale']}")
        lines.append("")

    lines.extend(
        [
            "## Classification Legend",
            "",
            "- **robustness_evidence**: perturbation affected observation and may impact policy.",
            "- **scenario_too_weak**: scenario too weak (pedestrian never observed or too distant).",
            "- **policy_insensitive**: perturbation did not change policy action sequence.",
            "- **diagnostic_only**: mixed effects; needs broader evidence.",
            "- **blocked**: not applicable for this trace-derived evaluation.",
            "",
            "## Caveats",
            "",
            "- Single deterministic fixture (seed=111), single scenario family.",
            "- No live planner replay; action proxies are from the stored trace, not re-executed.",
            "- Stop/yield feasibility is from fixture metadata, not re-derived.",
            "- Not paper-facing benchmark evidence.",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Run observation-noise envelope evaluation and write evidence."""
    parser = argparse.ArgumentParser(
        description="Evaluate near-field observation-noise robustness envelope."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/context/evidence/issue_2755_observation_noise_envelope_2026-06-13",
        help="Output directory for evidence artifacts.",
    )
    parser.add_argument(
        "--issue",
        type=int,
        default=2755,
        help="Issue number to record in generated provenance.",
    )
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    trace = _load_fixture(FIXTURE_PATH)
    frames = trace["frames"]
    occlusion = trace["occlusion"]
    first_visible_step = occlusion["first_visible_step"]

    fixture_meta = {
        "trace_path": str(FIXTURE_PATH.relative_to(REPO_ROOT)),
        "scenario_id": trace["source"]["scenario_id"],
        "seed": trace["source"]["seed"],
        "planner_id": trace["source"]["planner_id"],
        "episode_id": trace["source"]["episode_id"],
        "frame_count": len(frames),
        "first_visible_step": first_visible_step,
        "conflict_time_s": occlusion.get("conflict_time_s"),
        "stop_feasible_before_step": occlusion.get("stop_feasible_before_step"),
        "yield_feasible_before_step": occlusion.get("yield_feasible_before_step"),
    }

    repo_head = _git_head()
    generated_at = datetime.datetime.now(datetime.UTC).isoformat()
    command = (
        f"uv run python scripts/benchmark/run_observation_noise_envelope.py "
        f"--output-dir {args.output_dir}"
    )
    if args.issue != 2755:
        command = f"{command} --issue {args.issue}"

    repro = {
        "issue": args.issue,
        "generated_at_utc": generated_at,
        "command": command,
        "repo_head": repo_head,
    }

    results: list[dict[str, Any]] = []
    for name, cfg in CONDITIONS.items():
        result = evaluate_condition(name, cfg, frames, first_visible_step)
        results.append(result)

    report = _build_report(results, fixture_meta, repro, issue=args.issue)

    json_path = output_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    md_content = _generate_markdown(results, fixture_meta, repro)
    md_path = output_dir / "README.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md_content)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    for r in results:
        cl = r["classification"]
        print(f"  [{cl['label']:>20s}] {r['condition']}: {r['description'][:60]}")


if __name__ == "__main__":
    main()
