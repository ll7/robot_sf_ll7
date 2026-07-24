#!/usr/bin/env python3
"""Multi-pedestrian dense-stress observation-noise robustness envelope.

Reads the dense-pedestrian-stress trace fixture and evaluates observation-noise
perturbation conditions against the multi-actor scene.  This extends the
single-pedestrian occluded-emergence envelope to test observation-noise
sensitivity in dense, overlapping-pedestrian scenarios.

This is diagnostic trace-derived evidence only, not a full planner benchmark.

Usage::

    uv run python scripts/benchmark/run_dense_stress_observation_envelope.py \\
        --output-dir docs/context/evidence/issue_2765_dense_pedestrian_stress_2026-06-14
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import shutil
import subprocess
from typing import Any

import numpy as np

from robot_sf.benchmark.observation_perturbation import (
    ObservationPerturbationSpec,
    ObservationPerturbationState,
    perturb_ground_truth,
)
from robot_sf.benchmark.pedestrian_forecast import (
    PedestrianState,
    chi_square_2d_threshold,
    constant_velocity_gaussian_baseline,
    ellipse_overlaps_point,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    REPO_ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1/"
    "dense_pedestrian_stress_episode_0000.json"
)

DT_S = 0.1
PEDESTRIAN_COUNT = 3

CONDITIONS: dict[str, dict[str, Any]] = {
    "noop": {"description": "No perturbation applied (baseline)."},
    "low_noise": {
        "description": "Bounded Gaussian position noise std=0.10 m, bound=0.20 m on all actors.",
        "spec_kw": {"position_noise_std_m": 0.10, "position_noise_bound_m": 0.20, "seed": 2765},
    },
    "medium_noise": {
        "description": "Bounded Gaussian position noise std=0.30 m, bound=0.60 m on all actors.",
        "spec_kw": {"position_noise_std_m": 0.30, "position_noise_bound_m": 0.60, "seed": 2765},
    },
    "high_noise": {
        "description": "Bounded Gaussian position noise std=0.50 m, bound=1.00 m on all actors.",
        "spec_kw": {"position_noise_std_m": 0.50, "position_noise_bound_m": 1.00, "seed": 2765},
    },
    "partial_missed_detection": {
        "description": "50% missed detection probability on each actor independently.",
        "spec_kw": {"missed_detection_probability": 0.5, "seed": 2765},
    },
    "full_missed_detection": {
        "description": "100% missed detection probability (all actors dropped).",
        "spec_kw": {"missed_detection_probability": 1.0, "seed": 2765},
    },
    "single_actor_occlusion": {
        "description": "Occlude only the first actor (ped_a), others visible.",
        "spec_kw": {"occlusion_mask": np.array([True, False, False])},
    },
    "two_actor_occlusion": {
        "description": "Occlude first two actors (ped_a, ped_b), third visible.",
        "spec_kw": {"occlusion_mask": np.array([True, True, False])},
    },
    "delay_2_steps": {
        "description": "2-step delayed observation for all actors.",
        "spec_kw": {"delay_steps": 2},
    },
    "medium_noise_with_occlusion": {
        "description": "Medium Gaussian noise (std=0.30 m) + occlusion of the first actor.",
        "spec_kw": {
            "position_noise_std_m": 0.30,
            "position_noise_bound_m": 0.60,
            "occlusion_mask": np.array([True, False, False]),
            "seed": 2765,
        },
    },
}


def _load_fixture(path: pathlib.Path) -> dict[str, Any]:
    """Load the dense-pedestrian-stress trace fixture."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _git_head() -> str:
    """Return the short git HEAD, or empty string on failure."""
    git_exe = shutil.which("git")
    if git_exe is None:
        return ""
    try:
        result = subprocess.run(
            [git_exe, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=5,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _extract_frame_arrays(
    frame: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract ground-truth robot position and pedestrian arrays from a frame."""
    robot_pos = np.array(frame["robot"]["position"], dtype=np.float64)
    ped_positions = np.array([p["position"] for p in frame["pedestrians"]], dtype=np.float64)
    ped_ids = [str(p["id"]) for p in frame["pedestrians"]]
    return robot_pos, ped_positions, ped_ids


def _extract_observed_arrays(
    frame: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
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


def _all_pairwise_distances(robot_pos: np.ndarray, ped_pos: np.ndarray) -> list[float]:
    """Compute distances from robot to each pedestrian."""
    if ped_pos.size == 0:
        return []
    diffs = ped_pos - robot_pos
    return [float(np.sqrt(np.sum(d**2))) for d in diffs]


def _spec_for_actor_count(
    spec: ObservationPerturbationSpec, actor_count: int
) -> ObservationPerturbationSpec:
    """Adapt fixed-size occlusion masks to frames before fixture visibility."""
    if spec.occlusion_mask is None:
        return spec
    mask = np.asarray(spec.occlusion_mask, dtype=bool)
    if mask.size == actor_count:
        return spec
    if mask.size > actor_count:
        new_mask = mask[:actor_count]
    else:
        new_mask = np.pad(
            mask,
            (0, actor_count - mask.size),
            mode="constant",
            constant_values=False,
        )
    if actor_count == 0:
        new_mask = np.zeros(0, dtype=bool)
    return ObservationPerturbationSpec(
        position_noise_std_m=spec.position_noise_std_m,
        position_noise_bound_m=spec.position_noise_bound_m,
        missed_detection_probability=spec.missed_detection_probability,
        occlusion_mask=new_mask,
        delay_steps=spec.delay_steps,
        seed=spec.seed,
    )


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


def _count_forecast_overlaps(
    forecasts: list[Any],
) -> int:
    """Count pairwise overlap events between forecast ellipses."""
    threshold = chi_square_2d_threshold(0.95)
    overlap_count = 0
    for i in range(len(forecasts)):
        for j in range(i + 1, len(forecasts)):
            overlap_count += _count_horizon_overlaps(forecasts[i], forecasts[j], threshold)
    return overlap_count


def _count_horizon_overlaps(forecast_a: Any, forecast_b: Any, threshold: float) -> int:
    """Count overlap events between two forecasts across horizons."""
    count = 0
    for pred_a in forecast_a.predictions:
        for pred_b in forecast_b.predictions:
            if pred_a.horizon_s != pred_b.horizon_s:
                continue
            if ellipse_overlaps_point(
                mean=pred_a.mean,
                covariance=pred_a.covariance,
                point=pred_b.mean,
                confidence_threshold=threshold,
                radius_m=0.3,
            ):
                count += 1
    return count


def _count_pairwise_ped_distances(pedestrians: list[dict[str, Any]]) -> float:
    """Return the max pairwise distance between pedestrians."""
    max_dist = 0.0
    for i in range(len(pedestrians)):
        for j in range(i + 1, len(pedestrians)):
            pos_i = np.array(pedestrians[i]["position"], dtype=np.float64)
            pos_j = np.array(pedestrians[j]["position"], dtype=np.float64)
            dist = float(np.linalg.norm(pos_i - pos_j))
            max_dist = max(max_dist, dist)
    return max_dist


def _compute_forecast_ambiguity(
    frame: dict[str, Any],
) -> dict[str, Any]:
    """Compute forecast ambiguity metric for a multi-pedestrian frame.

    Measures whether constant-velocity forecast ellipses from multiple
    pedestrians overlap near the robot position.
    """
    pedestrians = frame.get("pedestrians", [])
    if len(pedestrians) < 2:
        return {"overlap_count": 0, "max_pairwise_distance_m": 0.0, "evaluable": False}

    forecasts = []
    for ped in pedestrians:
        state = PedestrianState(
            id=int(ped["id"]),
            position=np.array(ped["position"], dtype=np.float64),
            velocity=np.array(ped["velocity"], dtype=np.float64),
        )
        forecasts.append(constant_velocity_gaussian_baseline(state, horizons_s=(0.5, 1.0)))

    overlap_count = _count_forecast_overlaps(forecasts)
    max_pairwise = _count_pairwise_ped_distances(pedestrians)

    return {
        "overlap_count": overlap_count,
        "max_pairwise_distance_m": round(max_pairwise, 4),
        "evaluable": len(pedestrians) >= 2,
    }


def evaluate_condition(
    condition_name: str,
    condition_cfg: dict[str, Any],
    frames: list[dict[str, Any]],
    dense_stress_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one perturbation condition against the dense-stress trace."""
    spec_kw = dict(condition_cfg.get("spec_kw", {}))
    spec = ObservationPerturbationSpec(**spec_kw)
    needs_state = spec.delay_steps > 0
    state = ObservationPerturbationState(delay_steps=spec.delay_steps) if needs_state else None

    first_observed_step: int | None = None
    missed_counts: list[int] = []
    occluded_counts: list[int] = []
    closest_distances: list[float] = []
    action_proxies: list[dict[str, Any]] = []
    observed_steps: list[int] = []
    perturbed_observed_steps: list[int] = []
    feasibility_by_step: dict[int, dict[str, bool]] = {}
    forecast_ambiguities: list[dict[str, Any]] = []

    for idx, frame in enumerate(frames):
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

        ped_observed = len(obs["ids"]) > 0

        if ped_observed and first_observed_step is None:
            first_observed_step = step
        if ped_observed:
            perturbed_observed_steps.append(step)

        closest_dist = _closest_distance(robot_pos, ground_truth_ped_positions)
        fy = _stop_yield_feasibility(frame)
        action_proxy = _selected_action_proxy(frame)
        feasibility_by_step[step] = fy

        forecast_amb = _compute_forecast_ambiguity(frame)

        missed_counts.append(meta["missed_actor_count"])
        occluded_counts.append(meta["occluded_actor_count"])
        closest_distances.append(closest_dist)
        action_proxies.append(action_proxy)
        forecast_ambiguities.append(forecast_amb)

    total_overlap = sum(fa["overlap_count"] for fa in forecast_ambiguities)
    min_closest = round(min(closest_distances), 4) if closest_distances else None

    classification = _classify_condition(
        condition_name=condition_name,
        first_observed_step=first_observed_step,
        closest_distances=closest_distances,
        missed_counts=missed_counts,
        occluded_counts=occluded_counts,
        total_forecast_overlap=total_overlap,
    )

    return {
        "condition": condition_name,
        "description": condition_cfg["description"],
        "spec": _spec_summary(spec),
        "first_observed_step": first_observed_step,
        "first_visible_step_reference": 0,
        "total_frames": len(frames),
        "pedestrian_count": PEDESTRIAN_COUNT,
        "fixture_observed_steps": observed_steps,
        "perturbed_observed_steps": perturbed_observed_steps,
        "missed_actor_observations_total": sum(missed_counts),
        "occluded_actor_observations_total": sum(occluded_counts),
        "delay_steps_configured": spec.delay_steps,
        "closest_distance_m": min_closest,
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
        "forecast_ambiguity": {
            "total_overlap_events": total_overlap,
            "evaluable_frames": sum(1 for fa in forecast_ambiguities if fa["evaluable"]),
        },
        "classification": classification,
    }


def _spec_summary(spec: ObservationPerturbationSpec) -> dict[str, Any]:
    """Summarize a perturbation spec as a plain dict."""
    return {
        "position_noise_std_m": spec.position_noise_std_m,
        "position_noise_bound_m": spec.position_noise_bound_m,
        "missed_detection_probability": spec.missed_detection_probability,
        "has_occlusion_mask": spec.occlusion_mask is not None,
        "occlusion_mask_size": (
            int(np.asarray(spec.occlusion_mask).size) if spec.occlusion_mask is not None else 0
        ),
        "delay_steps": spec.delay_steps,
        "noise_profile": spec.noise_profile,
        "is_noop": spec.is_noop,
    }


def _classify_condition(
    *,
    condition_name: str,
    first_observed_step: int | None,
    closest_distances: list[float],
    missed_counts: list[int],
    occluded_counts: list[int],
    total_forecast_overlap: int,
) -> dict[str, str]:
    """Classify the condition result into an allowed category."""
    if condition_name == "noop":
        return {
            "label": "diagnostic_only",
            "rationale": (
                "No-perturbation baseline. Provides reference multi-pedestrian "
                "trajectory and action selection without observation noise."
            ),
        }

    never_observed = first_observed_step is None
    has_missed = any(c > 0 for c in missed_counts)
    has_occluded = any(c > 0 for c in occluded_counts)

    if never_observed:
        if has_missed or has_occluded:
            return {
                "label": "scenario_too_weak",
                "rationale": (
                    "All pedestrians suppressed by missed detection or occlusion "
                    "mask; no observation signal reaches the policy."
                ),
            }
        return {
            "label": "inconclusive",
            "rationale": (
                "Pedestrians never observed but no missed-detection or "
                "occlusion signal recorded. Insufficient data."
            ),
        }

    min_dist = min(closest_distances) if closest_distances else float("inf")

    if min_dist > 2.0:
        return {
            "label": "scenario_too_weak",
            "rationale": (
                f"Closest robot-pedestrian distance ({min_dist:.2f} m) exceeds "
                "near-field threshold (2.0 m). Dense-stress scenario too weak."
            ),
        }

    if total_forecast_overlap > 0:
        return {
            "label": "forecast_ambiguity_detected",
            "rationale": (
                f"Multiple constant-velocity forecast ellipses overlap "
                f"({total_forecast_overlap} overlap events). Dense scene "
                "creates forecast ambiguity under observation perturbation."
            ),
        }

    return {
        "label": "diagnostic_only",
        "rationale": (
            "Perturbation produced mixed effects in the dense-stress scene. "
            "Classified as diagnostic-only pending broader seed evidence."
        ),
    }


def _action_proxy_summary(
    action_proxies: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize stored-trace action proxies."""
    if not action_proxies:
        return {
            "linear_velocity_changed": False,
            "events": [],
            "velocity_range": None,
        }
    events = [ap.get("event") for ap in action_proxies]
    unique_events = list(dict.fromkeys(events))
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
) -> dict[str, Any]:
    """Build the full JSON report."""
    return {
        "schema_version": "observation_noise_envelope.v1",
        "issue": 2765,
        "claim_boundary": (
            "Diagnostic trace-derived evidence only. Not paper-facing benchmark "
            "proof. Evaluates near-field observation-noise robustness on a "
            "multi-pedestrian dense-stress trace fixture."
        ),
        "reproducibility": repro,
        "fixture": fixture_meta,
        "conditions": results,
        "summary": {
            "total_conditions": len(results),
            "pedestrian_count": PEDESTRIAN_COUNT,
            "classifications": {r["condition"]: r["classification"]["label"] for r in results},
            "forecast_ambiguity_detected": any(
                r["classification"]["label"] == "forecast_ambiguity_detected" for r in results
            ),
        },
    }


def _generate_markdown(
    results: list[dict[str, Any]],
    fixture_meta: dict[str, Any],
    repro: dict[str, Any],
) -> str:
    """Generate the Markdown evidence report."""
    lines: list[str] = [
        "# Issue #2765: Dense-Pedestrian-Stress Observation-Noise Envelope - 2026-06-14",
        "",
        "## Claim Boundary",
        "",
        "**Diagnostic trace-derived evidence only. Not paper-facing benchmark "
        "proof.** This evaluates near-field observation-noise robustness on a "
        "multi-pedestrian dense-stress trace fixture with "
        f"{PEDESTRIAN_COUNT} converging actors.",
        "",
        "## Reproducibility",
        "",
        f"- **Issue:** #{repro['issue']}",
        f"- **Generated at (UTC):** {repro['generated_at_utc']}",
        f"- **Command:** `{repro['command']}`",
        f"- **Repo HEAD:** `{repro['repo_head']}`",
        f"- **Fixture:** `{fixture_meta['trace_path']}`",
        f"- **Scenario:** {fixture_meta.get('scenario_id', '')}",
        f"- **Pedestrian count:** {PEDESTRIAN_COUNT}",
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
        lines.append(f"- **Pedestrians:** {r['pedestrian_count']}")
        lines.append(f"- **First observed step:** {r['first_observed_step']}")
        lines.append(
            f"- **Closest distance:** {r['closest_distance_m']}"
            if r["closest_distance_m"] is not None
            else "- **Closest distance:** N/A"
        )
        fa = r.get("forecast_ambiguity", {})
        lines.append(f"- **Forecast overlap events:** {fa.get('total_overlap_events', 0)}")
        fy = r["stop_yield_feasibility"]
        lines.append(f"- **Stop feasible (first observed):** {fy['stop_feasible_first_observed']}")
        lines.append(
            f"- **Yield feasible (first observed):** {fy['yield_feasible_first_observed']}"
        )
        cl = r["classification"]
        lines.append(f"- **Classification:** `{cl['label']}`")
        lines.append(f"  - {cl['rationale']}")
        lines.append("")

    lines.extend(
        [
            "## Classification Legend",
            "",
            "- **forecast_ambiguity_detected**: overlapping forecast ellipses from multiple actors.",
            "- **robustness_evidence**: perturbation affected observation and may impact policy.",
            "- **scenario_too_weak**: scenario too weak (all pedestrians suppressed or too distant).",
            "- **policy_insensitive**: perturbation did not change policy action sequence.",
            "- **diagnostic_only**: mixed effects; needs broader evidence.",
            "- **inconclusive**: insufficient data for mechanism classification.",
            "",
            "## Caveats",
            "",
            "- Deterministic single-seed fixture (seed=2765), single scenario family.",
            "- No live planner replay; action proxies are from the stored trace.",
            "- Forecast ambiguity is computed from constant-velocity Gaussian baselines.",
            "- Not paper-facing benchmark evidence.",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Run dense-stress observation-noise envelope evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate dense-pedestrian-stress observation-noise envelope."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/context/evidence/issue_2765_dense_pedestrian_stress_2026-06-14",
        help="Output directory for evidence artifacts.",
    )
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    trace = _load_fixture(FIXTURE_PATH)
    frames = trace["frames"]
    dense_stress_meta = trace.get("dense_stress_metadata", {})

    fixture_meta = {
        "trace_path": str(FIXTURE_PATH.relative_to(REPO_ROOT)),
        "scenario_id": trace["source"]["scenario_id"],
        "seed": trace["source"]["seed"],
        "planner_id": trace["source"]["planner_id"],
        "episode_id": trace["source"]["episode_id"],
        "frame_count": len(frames),
        "pedestrian_count": dense_stress_meta.get("pedestrian_count", PEDESTRIAN_COUNT),
    }

    repo_head = _git_head()
    generated_at = datetime.datetime.now(datetime.UTC).isoformat()
    command = (
        f"uv run python scripts/benchmark/run_dense_stress_observation_envelope.py "
        f"--output-dir {args.output_dir}"
    )

    repro = {
        "issue": 2765,
        "generated_at_utc": generated_at,
        "command": command,
        "repo_head": repo_head,
    }

    results: list[dict[str, Any]] = []
    for name, cfg in CONDITIONS.items():
        result = evaluate_condition(name, cfg, frames, dense_stress_meta)
        results.append(result)

    report = _build_report(results, fixture_meta, repro)

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
        print(f"  [{cl['label']:>30s}] {r['condition']}: {r['description'][:50]}")


if __name__ == "__main__":
    main()
