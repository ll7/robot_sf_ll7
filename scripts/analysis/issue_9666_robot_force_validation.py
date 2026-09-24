"""Analyze the preregistered diagnostic force-redundancy probe (#9666)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from robot_sf.benchmark.metrics import EpisodeData, experimental_human_interaction_proxy_metrics

FORCES = (
    "robot_force_impulse_total",
    "robot_force_peak",
    "robot_force_time_above_ref_s",
    "robot_force_pp_equiv_impulse_total",
    "robot_force_pp_equiv_peak",
    "robot_force_pp_equiv_time_above_ref_s",
)
COMPARATORS = ("min_distance", "near_misses", "human_discomfort_exposure_m_s")


def metric_value(row: dict, key: str) -> float | None:
    """Resolve the serializer's canonical nested human-proxy reduction."""
    metrics = row["metrics"]
    if key == "human_discomfort_exposure_m_s" and key not in metrics:
        return metrics.get("human_interaction_proxy", {}).get("canonical_reductions", {}).get(key)
    return metrics.get(key)


def posthoc_discomfort(row: dict) -> float | None:
    """Reuse the canonical proxy on recorded post-integration trajectories.

    Reject inconsistent trace footprints; never use pre-integration force positions
    as a substitute for the trajectory metric's snapshots.
    """
    trace = row.get("algorithm_metadata", {}).get("simulation_step_trace", {})
    steps = trace.get("steps")
    model = row["metrics"].get("robot_force_metadata")
    if not steps or model is None:
        return None
    actor_ids = sorted({ped["id"] for step in steps for ped in step["pedestrians"]})
    if not actor_ids:
        return 0.0
    indices = {actor: i for i, actor in enumerate(actor_ids)}
    robot = np.asarray([step["robot"]["position"] for step in steps], dtype=float)
    peds = np.full((len(steps), len(actor_ids), 2), np.nan)
    robot_radius = model["prf_robot_radius_m"]
    first_step = next(step for step in steps if step["pedestrians"])
    first_ped = first_step["pedestrians"][0]
    # The force kernel's pedestrian radius can differ from the trajectory
    # footprint. Recover the latter from recorded geometric clearance instead.
    ped_radius = float(
        np.linalg.norm(
            np.asarray(first_ped["position"]) - np.asarray(first_step["robot"]["position"])
        )
        - first_ped["surface_clearance_m"]
        - robot_radius
    )
    if ped_radius < 0 or not np.isfinite(ped_radius):
        raise ValueError("invalid trace pedestrian footprint radius")
    for t, step in enumerate(steps):
        if len({ped["id"] for ped in step["pedestrians"]}) != len(step["pedestrians"]):
            raise ValueError("duplicate pedestrian trace identity")
        for ped in step["pedestrians"]:
            position = np.asarray(ped["position"], dtype=float)
            clearance = np.linalg.norm(position - robot[t]) - robot_radius - ped_radius
            if not np.isclose(clearance, ped["surface_clearance_m"], rtol=0, atol=1e-9):
                raise ValueError("trace surface clearance disagrees with constant footprint radii")
            peds[t, indices[ped["id"]]] = position
    data = EpisodeData(
        robot_pos=robot,
        robot_vel=np.asarray([step["robot"]["velocity"] for step in steps], dtype=float),
        robot_acc=np.zeros_like(robot),
        peds_pos=peds,
        ped_forces=np.zeros_like(peds),
        goal=np.zeros(2),
        dt=trace["dt"],
        robot_radius=robot_radius,
        ped_radius=ped_radius,
    )
    return experimental_human_interaction_proxy_metrics(
        data, proxemic_radius_m=1.2, yield_speed_mps=0.15
    )["human_discomfort_exposure_m_s"]


def trace_evidence(row: dict) -> dict:
    """Summarize recorded inputs without inferring a causal interaction mechanism."""
    samples = row["metrics"].get("robot_force_samples")
    dt = row.get("scenario_params", {}).get("run_dt")
    if not samples or dt is None:
        return {"status": "unavailable"}
    active_steps = active_pairs = max_concurrent = 0
    for sample in samples:
        forces = np.asarray(sample["forces"], dtype=float).reshape(-1, 2)
        active = int(np.count_nonzero(np.linalg.norm(forces, axis=-1) > 0))
        active_steps += active > 0
        active_pairs += active
        max_concurrent = max(max_concurrent, active)
    return {
        "status": "recorded",
        "steps": len(samples),
        "duration_s": len(samples) * dt,
        "force_active_duration_s": active_steps * dt,
        "force_active_pedestrian_seconds": active_pairs * dt,
        "max_concurrently_exposed_pedestrians": max_concurrent,
        "termination_reason": row.get("termination_reason"),
        "boundary": "pre-integration model force; descriptive duration and multiplicity, not causal proof",
    }


def _prepare_comparators(rows: list[dict]) -> tuple[list[dict], int]:
    """Add explicit post-hoc proxy values to copies, preserving producer rows."""
    derived_count = 0
    prepared = []
    for row in rows:
        if metric_value(row, "human_discomfort_exposure_m_s") is None:
            derived = posthoc_discomfort(row)
            if derived is not None:
                row = {
                    **row,
                    "metrics": {**row["metrics"], "human_discomfort_exposure_m_s": derived},
                }
                derived_count += 1
        prepared.append(row)
    return prepared, derived_count


def analyze(rows: list[dict]) -> dict:
    """Compute correlations and rank disagreements without fabricating missing observations.

    Returns:
        Diagnostic report with cohort sizes, nullable correlations, and 20 rank disagreements.
    """
    rows, derived_count = _prepare_comparators(rows)
    identities = [(row["scenario_id"], row["seed"], row.get("algo")) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate scenario/seed/planner identity")
    groups = {"overall": rows}
    for row in rows:
        groups.setdefault(f"planner:{row['algo']}", []).append(row)
        family = row.get("scenario_params", {}).get("metadata", {}).get("archetype", "undeclared")
        groups.setdefault(f"family:{family}", []).append(row)
    correlations = []
    for cohort, members in groups.items():
        for force in FORCES:
            for comparator in COMPARATORS:
                pairs = np.array(
                    [(metric_value(r, force), metric_value(r, comparator)) for r in members],
                    dtype=float,
                )
                pairs = pairs[np.isfinite(pairs).all(axis=1)]
                if comparator == "min_distance":
                    pairs[:, 1] *= -1
                rho = None
                if len(pairs) > 1 and np.ptp(pairs[:, 0]) > 0 and np.ptp(pairs[:, 1]) > 0:
                    rho = float(spearmanr(pairs[:, 0], pairs[:, 1]).statistic)
                correlations.append(
                    {
                        "cohort": cohort,
                        "force": force,
                        "comparator": comparator,
                        "min_distance_sign_flipped": comparator == "min_distance",
                        "n": len(pairs),
                        "spearman_rho": rho,
                    }
                )
    eligible = [
        r
        for r in rows
        if all(
            r["metrics"].get(key) is not None and np.isfinite(r["metrics"][key])
            for key in ("robot_force_impulse_total", "min_distance")
        )
    ]
    disagreements = []
    if eligible:
        force_rank = rankdata([r["metrics"]["robot_force_impulse_total"] for r in eligible])
        distance_rank = rankdata([-r["metrics"]["min_distance"] for r in eligible])
        for i in np.argsort(-np.abs(force_rank - distance_rank), kind="stable")[:20]:
            row = eligible[i]
            metrics = row["metrics"]
            disagreements.append(
                {
                    "scenario_id": row["scenario_id"],
                    "seed": row["seed"],
                    "algo": row["algo"],
                    "force_rank": float(force_rank[i]),
                    "distance_rank": float(distance_rank[i]),
                    "trace_evidence": trace_evidence(row),
                    "metrics": {
                        k: metric_value(row, k)
                        for k in (*FORCES, *COMPARATORS, "robot_force_exposed_ped_count")
                    },
                    "observed_pattern": "multiple_pedestrians_exposed"
                    if metrics.get("robot_force_exposed_ped_count", 0) > 1
                    else (
                        "one_pedestrian_exposed"
                        if metrics.get("robot_force_exposed_ped_count") == 1
                        else (
                            "no_pedestrians_exposed"
                            if metrics.get("robot_force_exposed_ped_count") == 0
                            else "exposure_count_unavailable"
                        )
                    ),
                    "interpretation": "descriptive pattern; causal mechanism needs aligned trace inspection",
                }
            )
    return {
        "classification": "diagnostic_not_release_evaluation",
        "episodes": len(rows),
        "posthoc_discomfort": {
            "rows": derived_count,
            "producer": "experimental_human_interaction_proxy_metrics",
            "proxemic_radius_m": 1.2,
            "yield_speed_mps": 0.15,
            "trajectory_source": "simulation_step_trace.steps (post-integration)",
            "radius_source": "geometric footprint recovered from recorded surface clearance; not force-kernel pedestrian radius",
            "radius_validation": "every recorded surface clearance agrees within 1e-9m",
            "original_episode_files_modified": False,
        },
        "correlations": correlations,
        "largest_rank_disagreements": disagreements,
    }


def main() -> None:
    """Read explicit episode artifacts and write a checksum-bound diagnostic report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=384)
    args = parser.parse_args()
    rows = []
    sources = []
    for path in args.episodes:
        raw = path.read_bytes()
        file_rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        rows.extend(file_rows)
        sources.append(
            {
                "name": f"{path.parent.name}/{path.name}",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "rows": len(file_rows),
                "algorithms": sorted({row["algo"] for row in file_rows}),
            }
        )
    if len(rows) != args.expected_episodes:
        raise ValueError(f"expected {args.expected_episodes} episodes, received {len(rows)}")
    report = analyze(rows)
    report["sources"] = sources
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
