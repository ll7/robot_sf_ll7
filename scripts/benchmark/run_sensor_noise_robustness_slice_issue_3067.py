#!/usr/bin/env python3
"""Issue #3067: bounded clean / noisy / partial observation-robustness slice.

Runs a same-seed clean / noisy / partial-observation comparison over a
pedestrian-dominated trace fixture and reports clean-vs-perturbed deltas for
safety (near-field), progress (observation continuity), runtime (perturbation
compute), and a social-compliance proxy.  Robustness is interpreted SEPARATELY
from nominal (stored-trace) performance.

This is a diagnostic slice, NOT a real-sensor certification and NOT a
sim-to-real transfer claim.  The perturbations are non-calibrated benchmark
robustness noise applied to ground-truth observed actor state; they are not a
hardware sensor model.  Every reported number comes from an actual run.

Fail-closed rules:
- If observation wrappers / perturbation metadata are incomplete -> ``blocked``.
- If the perturbation envelope is too narrow to move any observed-state metric
  -> ``diagnostic-only`` (null result documented honestly).
- Degraded / invalid / not-available rows are never counted as success.

Usage::

    uv run python scripts/benchmark/run_sensor_noise_robustness_slice_issue_3067.py
    uv run python scripts/benchmark/run_sensor_noise_robustness_slice_issue_3067.py \\
        --output-dir output/issue_3067_sensor_noise/run
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import shutil
import subprocess
import time
from typing import Any

import numpy as np

from robot_sf.benchmark.observation_levels import observation_level_spec
from robot_sf.benchmark.observation_perturbation import (
    ObservationPerturbationSpec,
    ObservationPerturbationState,
    perturb_ground_truth,
)
from robot_sf.benchmark.observation_quality import ObservationQuality

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    REPO_ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1/"
    "dense_pedestrian_stress_episode_0000.json"
)
SCHEMA_VERSION = "sensor_noise_robustness_slice.v1"
OBSERVATION_QUALITY_SCHEMA_VERSION = "observation_quality.v1"
DT_S = 0.1
SEED = 3067
NEAR_FIELD_THRESHOLD_M = 2.0

# Allowed overall-classification vocabulary (acceptance criterion).
OVERALL_CLASSIFICATIONS = ("benchmark", "diagnostic", "blocked", "non-claim")
# Allowed per-row status vocabulary (fail-closed).
ROW_STATUSES = ("ok", "degraded", "invalid", "not-available")

# Clean / noisy / partial matrix.  Each row carries an observation level + a
# perturbation spec.  "clean" is the same-seed reference all deltas compare to.
MATRIX: dict[str, dict[str, Any]] = {
    "clean": {
        "family": "clean",
        "observation_level": "oracle_full_state",
        "description": "No perturbation; same-seed clean reference observation.",
        "spec_kw": {},
    },
    "noisy_low": {
        "family": "noisy",
        "observation_level": "tracked_agents_with_noise",
        "description": "Bounded Gaussian position noise std=0.10 m, bound=0.20 m.",
        "spec_kw": {
            "position_noise_std_m": 0.10,
            "position_noise_bound_m": 0.20,
            "seed": SEED,
        },
    },
    "noisy_medium": {
        "family": "noisy",
        "observation_level": "tracked_agents_with_noise",
        "description": "Bounded Gaussian position noise std=0.30 m, bound=0.60 m.",
        "spec_kw": {
            "position_noise_std_m": 0.30,
            "position_noise_bound_m": 0.60,
            "seed": SEED,
        },
    },
    "partial_occlusion": {
        "family": "partial",
        "observation_level": "occluded_partial_state",
        "description": "Occlude the nearest crossing actor (ped_a); others visible.",
        "spec_kw": {"occlusion_mask": np.array([True, False, False])},
    },
    "partial_missed_detection": {
        "family": "partial",
        "observation_level": "occluded_partial_state",
        "description": "50% per-actor missed-detection probability (same seed).",
        "spec_kw": {"missed_detection_probability": 0.5, "seed": SEED},
    },
}


def _load_fixture(path: pathlib.Path) -> dict[str, Any]:
    """Load the pedestrian-dominated trace fixture."""
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


def _extract_frame_arrays(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract ground-truth robot position and observed pedestrian arrays."""
    robot_pos = np.array(frame["robot"]["position"], dtype=np.float64)
    observed = frame.get("observed_pedestrians", [])
    if not observed:
        return robot_pos, np.empty((0, 2), dtype=np.float64), []
    positions = np.array([p["position"] for p in observed], dtype=np.float64)
    ids = [str(p["id"]) for p in observed]
    return robot_pos, positions, ids


def _extract_observed_arrays(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract fixture-visible pedestrian observation positions and velocities."""
    observed = frame.get("observed_pedestrians", [])
    if not observed:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            [],
        )
    positions = np.array([p["position"] for p in observed], dtype=np.float64)
    velocities = np.array([p["velocity"] for p in observed], dtype=np.float64)
    ids = [str(p["id"]) for p in observed]
    return positions, velocities, ids


def _min_distance(robot_pos: np.ndarray, ped_pos: np.ndarray) -> float:
    """Return Euclidean distance from robot to the nearest pedestrian position."""
    if ped_pos.size == 0:
        return float("inf")
    diffs = ped_pos - robot_pos
    return float(np.min(np.sqrt(np.sum(diffs**2, axis=1))))


def _spec_for_actor_count(
    spec: ObservationPerturbationSpec, actor_count: int
) -> ObservationPerturbationSpec:
    """Adapt a fixed-size occlusion mask to a frame's observed-actor count."""
    if spec.occlusion_mask is None:
        return spec
    mask = np.asarray(spec.occlusion_mask, dtype=bool)
    if mask.size == actor_count:
        return spec
    if actor_count == 0:
        new_mask: np.ndarray = np.zeros(0, dtype=bool)
    elif mask.size > actor_count:
        new_mask = mask[:actor_count]
    else:
        new_mask = np.pad(mask, (0, actor_count - mask.size), constant_values=False)
    return ObservationPerturbationSpec(
        position_noise_std_m=spec.position_noise_std_m,
        position_noise_bound_m=spec.position_noise_bound_m,
        missed_detection_probability=spec.missed_detection_probability,
        occlusion_mask=new_mask,
        delay_steps=spec.delay_steps,
        seed=spec.seed,
    )


def _observation_quality_group(spec: ObservationPerturbationSpec) -> dict[str, Any]:
    """Return the bounded observation-quality metadata for a perturbation spec."""
    quality = ObservationQuality(
        visibility=["trace_fixture_observed_pedestrians"],
        occlusion=(
            ["explicit_occlusion_mask"]
            if spec.occlusion_mask is not None
            else ["fixture_declared_visibility_boundary"]
        ),
        latency_s=float(spec.delay_steps) * DT_S,
        dropout_probability=float(spec.missed_detection_probability),
        range_limit_m=None,
        angular_noise_std_rad=0.0,
        false_negative_rate=float(spec.missed_detection_probability),
        false_positive_rate=0.0,
        notes=(
            "Diagnostic simulator observation-quality metadata only; "
            "non-calibrated benchmark robustness noise, not a hardware sensor model."
        ),
    )
    return {
        "schema_version": OBSERVATION_QUALITY_SCHEMA_VERSION,
        "fields": quality.to_dict(),
    }


def _metadata_complete(meta: dict[str, Any]) -> bool:
    """Return True when per-step perturbation metadata carries the required keys."""
    required = {
        "noise_profile",
        "evidence_class",
        "missed_actor_count",
        "occluded_actor_count",
        "observed_actor_count",
        "actor_count",
        "step",
    }
    return required.issubset(meta.keys())


def evaluate_row(
    row_name: str,
    row_cfg: dict[str, Any],
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate one clean/noisy/partial row over the same-seed trace.

    Returns a per-row dict with observation-derived metrics, perturbation
    metadata, a fail-closed status, and runtime.  Metric values are derived
    from the perturbed observed state the policy would consume.
    """
    spec_kw = dict(row_cfg.get("spec_kw", {}))
    spec = ObservationPerturbationSpec(**spec_kw)
    state = (
        ObservationPerturbationState(delay_steps=spec.delay_steps) if spec.delay_steps > 0 else None
    )
    level_spec = observation_level_spec(row_cfg["observation_level"])

    metadata_complete = True
    perturb_failed = False
    min_observed_distances: list[float] = []
    observed_actor_counts: list[int] = []
    gt_observed_actor_counts: list[int] = []
    missed_total = 0
    occluded_total = 0
    frames_with_signal = 0

    start = time.perf_counter()
    for frame in frames:
        robot_pos = np.array(frame["robot"]["position"], dtype=np.float64)
        obs_pos, obs_vel, obs_ids = _extract_observed_arrays(frame)
        gt_observed_actor_counts.append(len(obs_ids))
        step_spec = _spec_for_actor_count(spec, len(obs_ids))
        try:
            result = perturb_ground_truth(
                obs_pos,
                obs_vel,
                obs_ids,
                spec=step_spec,
                step=int(frame["step"]),
                state=state,
            )
        except (ValueError, TypeError):
            perturb_failed = True
            break

        meta = result["metadata"]
        if not _metadata_complete(meta):
            metadata_complete = False
        observed = result["observed"]

        # Visible (non-zeroed) observed positions the policy would actually see.
        positions = np.asarray(observed["positions"], dtype=np.float64).reshape(-1, 2)
        if positions.size > 0:
            visible_mask = np.any(positions != 0.0, axis=1)
            visible_positions = positions[visible_mask]
        else:
            visible_positions = positions
        observed_actor_counts.append(int(visible_positions.shape[0]))
        if visible_positions.shape[0] > 0:
            frames_with_signal += 1
        min_observed_distances.append(_min_distance(robot_pos, visible_positions))
        missed_total += int(meta["missed_actor_count"])
        occluded_total += int(meta["occluded_actor_count"])
    runtime_s = time.perf_counter() - start

    # --- fail-closed status determination ---
    if perturb_failed:
        status = "invalid"
        status_reason = "Perturbation wrapper raised on a frame; observed state unavailable."
    elif not metadata_complete:
        status = "invalid"
        status_reason = "Per-step perturbation metadata incomplete; cannot certify the row."
    elif row_cfg["family"] != "clean" and frames_with_signal == 0:
        status = "degraded"
        status_reason = (
            "All actors suppressed after perturbation; no observation signal reaches the policy."
        )
    else:
        status = "ok"
        status_reason = "Row produced a usable perturbed observation stream."

    finite_dists = [d for d in min_observed_distances if np.isfinite(d)]
    metrics = {
        # Safety / near-miss proxy: closest observed actor across the episode.
        "min_observed_distance_m": (round(min(finite_dists), 4) if finite_dists else None),
        # Progress proxy: fraction of frames where the policy still sees an actor.
        "observation_continuity": (round(frames_with_signal / len(frames), 4) if frames else None),
        # Social-compliance proxy: near-field exposure count (frames within threshold).
        "near_field_exposure_frames": int(
            sum(1 for d in finite_dists if d <= NEAR_FIELD_THRESHOLD_M)
        ),
        # Runtime of the perturbation pass (diagnostic compute cost).
        "perturbation_runtime_s": round(runtime_s, 6),
        "total_observed_actor_observations": int(sum(observed_actor_counts)),
    }

    return {
        "row": row_name,
        "family": row_cfg["family"],
        "observation_level": level_spec.to_metadata(),
        "description": row_cfg["description"],
        "spec": {
            "position_noise_std_m": spec.position_noise_std_m,
            "position_noise_bound_m": spec.position_noise_bound_m,
            "missed_detection_probability": spec.missed_detection_probability,
            "has_occlusion_mask": spec.occlusion_mask is not None,
            "delay_steps": spec.delay_steps,
            "noise_profile": spec.noise_profile,
            "is_noop": spec.is_noop,
            "seed": spec.seed,
        },
        "perturbation_metadata": {
            "metadata_complete": metadata_complete,
            "missed_actor_observations_total": missed_total,
            "occluded_actor_observations_total": occluded_total,
            "frames_with_observation_signal": frames_with_signal,
            "total_frames": len(frames),
            "ground_truth_observed_actor_observations": int(sum(gt_observed_actor_counts)),
            "observation_quality": _observation_quality_group(spec),
        },
        "metrics": metrics,
        "status": status,
        "status_reason": status_reason,
    }


def _delta(perturbed: Any, clean: Any) -> float | None:
    """Return perturbed - clean for numeric metrics, else None."""
    if perturbed is None or clean is None:
        return None
    return round(float(perturbed) - float(clean), 6)


def compute_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute clean-vs-perturbed deltas per metric for every non-clean row.

    Deltas are only reported against an ``ok`` clean reference.  Degraded /
    invalid rows are surfaced but never folded into a success summary.
    """
    clean = next((r for r in rows if r["family"] == "clean"), None)
    if clean is None or clean["status"] != "ok":
        return {
            "available": False,
            "reason": "No usable clean reference row; deltas cannot be computed.",
            "rows": {},
        }

    clean_metrics = clean["metrics"]
    delta_rows: dict[str, Any] = {}
    any_metric_moved = False
    for row in rows:
        if row["family"] == "clean":
            continue
        m = row["metrics"]
        row_delta = {
            metric: _delta(m.get(metric), clean_metrics.get(metric))
            for metric in (
                "min_observed_distance_m",
                "observation_continuity",
                "near_field_exposure_frames",
                "total_observed_actor_observations",
            )
        }
        # Runtime delta tracked separately (diagnostic, not a robustness signal).
        row_delta["perturbation_runtime_s"] = _delta(
            m.get("perturbation_runtime_s"), clean_metrics.get("perturbation_runtime_s")
        )
        moved = any(
            d not in (None, 0, 0.0) for k, d in row_delta.items() if k != "perturbation_runtime_s"
        )
        if row["status"] == "ok" and moved:
            any_metric_moved = True
        delta_rows[row["row"]] = {
            "status": row["status"],
            "counts_as_success": row["status"] == "ok",
            "deltas": row_delta,
            "moved_a_metric": moved,
        }
    return {
        "available": True,
        "reference_row": clean["row"],
        "any_perturbed_metric_moved": any_metric_moved,
        "rows": delta_rows,
    }


def classify_overall(rows: list[dict[str, Any]], deltas: dict[str, Any]) -> dict[str, str]:
    """Return the overall classification in the stable vocabulary, fail-closed."""
    metadata_incomplete = any(not r["perturbation_metadata"]["metadata_complete"] for r in rows)
    clean = next((r for r in rows if r["family"] == "clean"), None)

    if metadata_incomplete or not deltas.get("available"):
        return {
            "label": "blocked",
            "rationale": (
                "Observation-perturbation metadata incomplete or no usable clean "
                "reference; fail-closed to blocked per issue #3067."
            ),
        }
    if clean is None or clean["status"] != "ok":
        return {
            "label": "blocked",
            "rationale": "Clean reference row did not produce a usable observation stream.",
        }

    ok_perturbed = [r for r in rows if r["family"] != "clean" and r["status"] == "ok"]
    if not ok_perturbed:
        return {
            "label": "non-claim",
            "rationale": (
                "Every perturbed row was degraded/invalid (all actors suppressed). "
                "No usable robustness signal; not a claim."
            ),
        }
    if not deltas.get("any_perturbed_metric_moved"):
        return {
            "label": "diagnostic",
            "rationale": (
                "Perturbations ran but moved no observed-state metric beyond the clean "
                "reference (null result). Envelope too narrow; diagnostic-only."
            ),
        }
    return {
        "label": "diagnostic",
        "rationale": (
            "Perturbations changed the observed state the policy would consume "
            "(non-null clean-vs-perturbed deltas). Diagnostic-only: trace-derived, "
            "single fixture, single seed; not a benchmark, certification, or "
            "sim-to-real claim."
        ),
    }


def build_report(
    rows: list[dict[str, Any]],
    deltas: dict[str, Any],
    overall: dict[str, str],
    fixture_meta: dict[str, Any],
    repro: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the full JSON report."""
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 3067,
        "claim_boundary": (
            "Diagnostic, trace-derived observation-robustness slice. "
            "NOT a real-sensor certification and NOT a sim-to-real transfer claim. "
            "Perturbations are non-calibrated benchmark robustness noise on observed "
            "actor state. Robustness is interpreted separately from nominal performance."
        ),
        "evidence_tier": "smoke",
        "paper_grade": False,
        "overall_classification": overall,
        "allowed_overall_classifications": list(OVERALL_CLASSIFICATIONS),
        "allowed_row_statuses": list(ROW_STATUSES),
        "reproducibility": repro,
        "fixture": fixture_meta,
        "matrix_families": sorted({r["family"] for r in rows}),
        "rows": rows,
        "clean_vs_perturbed_deltas": deltas,
        "nominal_vs_robustness_note": (
            "Nominal performance = stored-trace policy behavior on the clean observation. "
            "Robustness = how the observed state the policy consumes changes under "
            "perturbation. These are reported separately and must not be conflated."
        ),
    }


def generate_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown report from the JSON report."""
    repro = report["reproducibility"]
    fixture = report["fixture"]
    overall = report["overall_classification"]
    lines = [
        "# Issue #3067: Sensor-Noise Robustness Slice (clean / noisy / partial)",
        "",
        "## Claim Boundary",
        "",
        f"**{report['claim_boundary']}**",
        "",
        f"- **Evidence tier:** {report['evidence_tier']}",
        f"- **Paper-grade:** {report['paper_grade']}",
        f"- **Overall classification:** `{overall['label']}` — {overall['rationale']}",
        "",
        "## Reproducibility",
        "",
        f"- **Issue:** #{repro['issue']}",
        f"- **Generated at (UTC):** {repro['generated_at_utc']}",
        f"- **Command:** `{repro['command']}`",
        f"- **Repo HEAD:** `{repro['repo_head']}`",
        f"- **Seed:** {repro['seed']}",
        f"- **Fixture:** `{fixture['trace_path']}`",
        f"- **Scenario:** {fixture['scenario_id']} (pedestrian_count="
        f"{fixture['pedestrian_count']})",
        "",
        "## Matrix Rows",
        "",
        "| row | family | obs level | status | min_obs_dist_m | continuity | near_field_frames |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report["rows"]:
        m = r["metrics"]
        lines.append(
            f"| {r['row']} | {r['family']} | {r['observation_level']['key']} | "
            f"`{r['status']}` | {m['min_observed_distance_m']} | "
            f"{m['observation_continuity']} | {m['near_field_exposure_frames']} |"
        )
    lines += ["", "## Clean-vs-Perturbed Deltas", ""]
    deltas = report["clean_vs_perturbed_deltas"]
    if not deltas.get("available"):
        lines.append(f"_Deltas unavailable: {deltas.get('reason')}_")
    else:
        lines.append(f"Reference row: `{deltas['reference_row']}`")
        lines.append("")
        lines.append(
            "| row | status | counts_as_success | d(min_obs_dist) | d(continuity) | "
            "d(near_field) | moved? |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for name, d in deltas["rows"].items():
            dd = d["deltas"]
            lines.append(
                f"| {name} | `{d['status']}` | {d['counts_as_success']} | "
                f"{dd['min_observed_distance_m']} | {dd['observation_continuity']} | "
                f"{dd['near_field_exposure_frames']} | {d['moved_a_metric']} |"
            )
    lines += [
        "",
        "## Nominal vs Robustness",
        "",
        report["nominal_vs_robustness_note"],
        "",
        "## Caveats",
        "",
        "- Single deterministic pedestrian-dominated fixture, single seed.",
        "- Trace-derived: no live planner replay; metrics are observed-state proxies.",
        "- Non-calibrated benchmark robustness noise, not a hardware sensor model.",
        "- Degraded/invalid rows are fail-closed and never counted as success.",
        "- NOT a real-sensor certification and NOT a sim-to-real transfer claim.",
    ]
    return "\n".join(lines)


def run_slice(output_dir: pathlib.Path, *, command: str) -> dict[str, Any]:
    """Execute the full clean/noisy/partial slice and write artifacts.

    Returns the JSON report dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = _load_fixture(FIXTURE_PATH)
    frames = trace["frames"]
    dense_meta = trace.get("dense_stress_metadata", {})

    fixture_meta = {
        "trace_path": str(FIXTURE_PATH.relative_to(REPO_ROOT)),
        "scenario_id": trace["source"]["scenario_id"],
        "seed": trace["source"]["seed"],
        "planner_id": trace["source"]["planner_id"],
        "episode_id": trace["source"]["episode_id"],
        "frame_count": len(frames),
        "pedestrian_count": dense_meta.get("pedestrian_count"),
    }
    repro = {
        "issue": 3067,
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "command": command,
        "repo_head": _git_head(),
        "seed": SEED,
    }

    rows = [evaluate_row(name, cfg, frames) for name, cfg in MATRIX.items()]
    deltas = compute_deltas(rows)
    overall = classify_overall(rows, deltas)
    report = build_report(rows, deltas, overall, fixture_meta, repro)

    json_path = output_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    md_path = output_dir / "README.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(generate_markdown(report))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Overall classification: {overall['label']}")
    for r in rows:
        print(f"  [{r['status']:>13s}] {r['row']} ({r['family']})")
    return report


def main() -> None:
    """Parse arguments and run the issue #3067 robustness slice."""
    parser = argparse.ArgumentParser(
        description=(
            "Issue #3067 bounded clean / noisy / partial observation-robustness "
            "slice (diagnostic; not sensor certification or sim-to-real)."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/issue_3067_sensor_noise/run",
        help="Output directory for the JSON + Markdown report.",
    )
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    command = (
        "uv run python scripts/benchmark/run_sensor_noise_robustness_slice_issue_3067.py "
        f"--output-dir {args.output_dir}"
    )
    run_slice(output_dir, command=command)


if __name__ == "__main__":
    main()
