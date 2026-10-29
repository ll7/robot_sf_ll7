"""EXSEL-7.5-GUARD-PAIR screening helpers: row projection, class/band verdicts, ledger.

Class and band logic copies scripts/dev/diss_worked_example_nomination.py (ll7/diss
b95d41b52) guard_record/_normalized/class_matches/acceptance_count exactly: guard rows
classify the collision kind from ped_collision_count then obstacle_collision_count;
time_to_goal_s = round(steps * dt_s, 6) for successes only, dt_s = 0.1 (protocol
sources.guard_replication.dt_s); bands inclusive; null bands are not checked.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CLAIM_ID = "EXSEL-7.5-GUARD-PAIR"
CELL = "classic_cross_trap_high"
DT_S = 0.1
ARMS = ("G0E1", "G1E1")
CLASSES = {"G0E1": {"outcome": "C", "collision_kind": "obstacle"}, "G1E1": {"outcome": "S"}}
BANDS = {  # frozen block, protocol claims[EXSEL-7.5-GUARD-PAIR].frozen.bands
    "G0E1": {"min_surface_clearance_m": None, "exposure_steps": None, "time_to_goal_s": None},
    "G1E1": {"min_surface_clearance_m": None, "exposure_steps": None, "time_to_goal_s": [33.2, 41.6]},
}
MEASURES = ("min_surface_clearance_m", "exposure_steps", "time_to_goal_s")
ENVELOPE_RADIUS_SUM_M = 1.4
FALLBACK_KEYS = ("planner_runtime.fallback_count",)

LEDGER_FIELDS = [
    "claim_id", "screening_index", "seed", "arm", "scenario_id", "execution_commit",
    "config_hash", "algo_config_hash", "episode_row_sha256", "metrics_success",
    "collision_event", "timeout_event", "termination_reason", "steps", "run_dt",
    "ped_collision_count", "obstacle_collision_count", "wall_collisions", "near_misses",
    "min_ped_separation_m", "min_surface_clearance_m", "time_to_goal_s", "outcome_class",
    "collision_kind", "class_verdict", "band_min_surface_clearance_m", "band_exposure_steps",
    "band_time_to_goal_s", "arm_pass", "accepted", "mode", "fallback_count",
    "fallback_or_degraded_marker", "status",
]


def classify(m: dict, out: dict) -> tuple[str, str | None]:
    if m.get("success"):
        outcome = "S"
    elif out.get("collision_event"):
        outcome = "C"
    elif out.get("timeout_event"):
        outcome = "T"
    else:
        outcome = "?"
    kind = None
    if outcome == "C":
        if m.get("ped_collision_count"):
            kind = "pedestrian"
        elif m.get("obstacle_collision_count"):
            kind = "obstacle"
    return outcome, kind


def class_ok(outcome: str, kind: str | None, cls: dict) -> bool:
    if outcome != cls["outcome"]:
        return False
    if "collision_kind" in cls and kind != cls["collision_kind"]:
        return False
    return True


def fallback_marker(meta: dict):
    from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker

    runtime_meta = dict(meta)
    runtime_meta.pop("simulation_step_trace", None)
    runtime_meta.pop("paired_effect_metric_producer", None)
    return runtime_fallback_or_degraded_marker(runtime_meta, expected_algorithm="hybrid_rule_local_planner")


def project(raw_line: bytes, arm: str) -> dict:
    """Project one raw producer line onto ledger fields (no verdicts)."""
    raw = json.loads(raw_line)
    m, out = raw["metrics"], raw["outcome"]
    meta = raw["algorithm_metadata"]
    outcome, kind = classify(m, out)
    sep = m.get("min_distance")
    has_ped = sep is not None and sep == sep and sep != float("inf")
    steps = raw["steps"]
    return {
        "claim_id": CLAIM_ID, "seed": raw["seed"], "arm": arm, "scenario_id": raw["scenario_id"],
        "execution_commit": raw["git_hash"], "config_hash": raw.get("config_hash"),
        "algo_config_hash": meta.get("config_hash"),
        "episode_row_sha256": hashlib.sha256(raw_line.rstrip(b"\n")).hexdigest(),
        "metrics_success": bool(m.get("success")), "collision_event": bool(out.get("collision_event")),
        "timeout_event": bool(out.get("timeout_event")), "termination_reason": raw["termination_reason"],
        "steps": steps, "run_dt": DT_S,
        "ped_collision_count": m.get("ped_collision_count"),
        "obstacle_collision_count": m.get("obstacle_collision_count"),
        "wall_collisions": m.get("wall_collisions"), "near_misses": m.get("near_misses"),
        "min_ped_separation_m": sep if has_ped else None,
        "min_surface_clearance_m": round(sep - ENVELOPE_RADIUS_SUM_M, 6) if has_ped else None,
        "time_to_goal_s": round(steps * DT_S, 6) if outcome == "S" else None,
        "outcome_class": outcome, "collision_kind": kind,
        "mode": meta["planner_kinematics"]["execution_mode"],
        "fallback_count": meta["planner_runtime"]["fallback_count"],
        "fallback_or_degraded_marker": fallback_marker(meta),
        "status": raw.get("status"),
        "algo_ablation_disable_hard_guard": meta["config"]["ablation_disable_hard_guard"],
    }


def verdicts(rows: dict[str, dict]) -> bool:
    """Fill per-arm verdicts in place; return the seed's acceptance (rules.acceptance, typical)."""
    accepted = True
    for arm in ARMS:
        r = rows[arm]
        r["class_verdict"] = "match" if class_ok(r["outcome_class"], r["collision_kind"], CLASSES[arm]) else "no_match"
        arm_pass = r["class_verdict"] == "match"
        for measure in MEASURES:
            band = BANDS[arm][measure]
            key = "band_" + measure
            if band is None:
                r[key] = "not_checked"
                continue
            value = r["near_misses"] if measure == "exposure_steps" else r[measure]
            inside = value is not None and band[0] <= value <= band[1]
            r[key] = "inside" if inside else "outside"
            arm_pass = arm_pass and inside
        r["arm_pass"] = arm_pass
        accepted = accepted and arm_pass
    for arm in ARMS:
        rows[arm]["accepted"] = accepted
    return accepted


def find_episodes(run_root: Path, arm: str) -> Path:
    hits = sorted(run_root.glob(f"*/runs/{arm.lower()}__differential_drive/episodes.jsonl"))
    if len(hits) != 1:
        raise SystemExit(f"expected one producer file for {arm} under {run_root}, got {len(hits)}")
    return hits[0]


def load_run(run_root: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    for arm in ARMS:
        for line in find_episodes(run_root, arm).read_bytes().splitlines():
            if line.strip():
                r = project(line, arm)
                key = (arm, r["seed"])
                if key in rows:
                    raise SystemExit(f"duplicate row {key}")
                rows[key] = r
    return rows
