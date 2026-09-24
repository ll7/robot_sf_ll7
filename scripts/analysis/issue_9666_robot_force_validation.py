"""Analyze the preregistered diagnostic force-redundancy probe (#9666)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

FORCES = (
    "robot_force_impulse_total",
    "robot_force_peak",
    "robot_force_time_above_ref_s",
    "robot_force_pp_equiv_impulse_total",
    "robot_force_pp_equiv_peak",
    "robot_force_pp_equiv_time_above_ref_s",
)
COMPARATORS = ("min_distance", "near_misses", "human_discomfort_exposure_m_s")


def analyze(rows: list[dict]) -> dict:
    """Compute correlations and rank disagreements without fabricating missing observations.

    Returns:
        Diagnostic report with cohort sizes, nullable correlations, and 20 rank disagreements.
    """
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
                    [
                        (r["metrics"].get(force, np.nan), r["metrics"].get(comparator, np.nan))
                        for r in members
                    ],
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
        if np.isfinite(r["metrics"].get("robot_force_impulse_total", np.nan))
        and np.isfinite(r["metrics"].get("min_distance", np.nan))
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
                    "metrics": {
                        k: metrics.get(k)
                        for k in (*FORCES, *COMPARATORS, "robot_force_exposed_ped_count")
                    },
                    "observed_pattern": "multiple_pedestrians_exposed"
                    if metrics.get("robot_force_exposed_ped_count", 0) > 1
                    else "single_pedestrian_duration_vs_peak",
                    "interpretation": "descriptive pattern; causal mechanism needs aligned trace inspection",
                }
            )
    return {
        "classification": "diagnostic_not_release_evaluation",
        "episodes": len(rows),
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
        rows.extend(json.loads(line) for line in raw.splitlines() if line.strip())
        sources.append({"name": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
    if len(rows) != args.expected_episodes:
        raise ValueError(f"expected {args.expected_episodes} episodes, received {len(rows)}")
    report = analyze(rows)
    report["sources"] = sources
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
