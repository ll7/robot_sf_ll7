"""Derive unfrozen SNQI-v2 diagnostics from a complete dev-seed campaign.

This command never remaps seeds, rewrites rows, pins anchors, or admits release evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.snqi.v2_calibration import (
    CalibrationGrid,
    _candidate_calibration_horizons,
    _compact_calibration_record,
    derive_calibration_anchors,
)
from robot_sf.benchmark.snqi.v2_reports import read_episode_files, score_episode
from robot_sf.benchmark.snqi.v2_spec import WEIGHTS, SnqiV2Spec


def diagnose(campaign_root: Path) -> dict:
    """Validate original dev rows and report diagnostic anchors, scores and alignment.

    Returns:
        Unfrozen anchors and a ranking over the same development rows.
    """
    manifest_path = campaign_root / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    source = manifest["git"]["commit"]
    paths = sorted(campaign_root.glob("runs/*/episodes.jsonl"))
    hashes = {str(path.relative_to(campaign_root)): sha256_file(path) for path in paths}
    digest = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    horizons = _candidate_calibration_horizons()
    arms = [path.parent.name.removesuffix("__differential_drive") for path in paths]
    algorithms = {entry["key"]: entry["algo"] for entry in manifest["planners"] if entry["enabled"]}
    records, seeds, budgets = [], set(), Counter()
    targets = defaultdict(list)
    for arm, path in zip(arms, paths, strict=True):
        for row in read_episode_files([path]):
            seed = row.get("seed")
            if type(seed) is not int or not 1001 <= seed <= 1030:
                raise ValueError("diagnostics require original dev seeds 1001..1030")
            if row.get("git_hash") != source or row.get("scenario_id") not in horizons:
                raise ValueError("diagnostic row source/scenario differs from manifest/schedule")
            seeds.add(seed)
            budgets[row["horizon"]] += 1
            metrics = row["metrics"]
            targets[arm].append(
                float(metrics["success"])
                - 0.7 * metrics["collisions"]
                - 0.25 * metrics["near_misses"]
                - 0.2 * metrics["comfort_exposure"]
            )
            records.append(
                _compact_calibration_record(
                    row,
                    arm,
                    expected_horizon=horizons[row["scenario_id"]],
                    expected_algorithm=algorithms[arm],
                )
            )
    document = derive_calibration_anchors(
        records,
        arms=arms,
        scenarios=sorted(horizons),
        run_id=manifest["campaign_id"],
        source_commit=source,
        episodes_sha256=digest,
        grid=CalibrationGrid(horizons, tuple(sorted(seeds)), diagnostic=True),
        expected_algorithms=algorithms,
    )
    document["calibration"]["episode_files_sha256"] = hashes
    spec = SnqiV2Spec(
        weights=WEIGHTS,
        upper_anchors={term: entry["upper"] for term, entry in document["anchors"].items()},
        force_source=document["force_decision"]["source"],
        calibration_rho=document["force_decision"]["spearman_rho_F_N"],
        calibration_split_id=document["calibration"]["split_id"],
        calibration_seeds=tuple(sorted(seeds)),
        paths={},
        hashes={},
        metric_schema_version=document["metric_schema_version"],
        diagnostic=True,
    )
    by_arm = defaultdict(list)
    for row in records:
        by_arm[row["planner_key"]].append(
            score_episode(row, spec, expected_algorithm=algorithms[row["planner_key"]])
        )
    ranking = []
    for arm, rows in by_arm.items():
        ranking.append(
            {
                "arm": arm,
                "episodes": len(rows),
                "snqi_v2_mean": float(np.mean([r["metrics"]["snqi_v2"] for r in rows])),
                "success_rate": float(np.mean([r["metrics"]["success"] for r in rows])),
                "collision_rate": float(
                    np.mean([r["metrics"]["total_collision_count"] > 0 for r in rows])
                ),
                "legacy_target_quality": float(np.mean(targets[arm])),
            }
        )
    ranking.sort(key=lambda row: (-row["snqi_v2_mean"], row["arm"]))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    scores = [row["snqi_v2_mean"] for row in ranking]
    alignment = {
        "success_minus_collision": float(
            spearmanr(
                scores, [row["success_rate"] - row["collision_rate"] for row in ranking]
            ).statistic
        ),
        "legacy_target_quality": float(
            spearmanr(scores, [row["legacy_target_quality"] for row in ranking]).statistic
        ),
        "success_rate": float(
            spearmanr(scores, [row["success_rate"] for row in ranking]).statistic
        ),
    }
    if hashes != {str(path.relative_to(campaign_root)): sha256_file(path) for path in paths}:
        raise ValueError("diagnostic source rows changed during analysis")
    return {
        "status": "diagnostic_only",
        "anchors": document,
        "ranking": ranking,
        "rank_alignment_spearman": alignment,
        "budget_counts": dict(budgets),
        "manifest_sha256": sha256_file(manifest_path),
        "claim_boundary": "Same-dev-data calibration and ranking; nothing pinned; not held-out or release evidence.",
    }


def main() -> None:
    """Write a standalone diagnostic without mutating campaign inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.campaign_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "episodes": result["anchors"]["calibration"]["episode_count"],
                "anchors": result["anchors"]["anchors"],
                "rank_alignment_spearman": result["rank_alignment_spearman"],
            }
        )
    )


if __name__ == "__main__":
    main()
