"""Build a claim's screening_ledger.csv from the raw screening outputs (deterministic).

One row per screened (seed, arm), in screening order then the claim's arm order, with the
fields of acquisition_spec_markdown step 2 plus the mode column. The ledger stops after
the first accepted seed; a seed without a successful run has no row.

    python build_ledger.py --contract <claim>/contract.yaml --runs <raw output dir> \
        --out <claim>/screening_ledger.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exsel
from screen import arm_rows

COLUMNS = [
    "claim_id",
    "screening_index",
    "seed",
    "arm",
    "scenario_id",
    "execution_commit",
    "config_hash",
    "episode_row_sha256",
    "mode",
    "metrics_success",
    "outcome_collision_event",
    "outcome_timeout_event",
    "termination_reason",
    "steps",
    "run_dt",
    "ped_collision_count",
    "obstacle_collision_count",
    "wall_collisions",
    "near_misses",
    "min_ped_separation_m",
    "outcome",
    "collision_kind",
    "min_surface_clearance_m",
    "exposure_steps",
    "time_to_goal_s",
    "class_outcome",
    "class_collision_kind",
    "class_min_exposure_steps",
    "band_min_surface_clearance_m",
    "band_exposure_steps",
    "band_time_to_goal_s",
    "arm_matches",
    "accepted",
    "mode_algorithm_status",
    "mode_pedestrian_model_status",
    "mode_effective_view_degraded",
    "mode_planner_execution_mode",
    "mode_fallback_counters",
]


def fmt(value: object) -> str:
    """Render one cell: empty for absent, lower-case booleans, repr-exact floats.

    Returns:
        Cell text.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def verdict(value: bool | None) -> str:
    """Render a predicate or band verdict.

    Returns:
        pass, fail or not_checked.
    """
    return "not_checked" if value is None else ("pass" if value else "fail")


def build(contract: dict, runs: Path) -> tuple[str, dict]:
    """Return the ledger text and a summary.

    Returns:
        (CSV text, summary mapping).
    """
    claim = exsel.load_claim(contract)
    arms = [a["arm"] for a in claim["arms"]]
    prefix = contract["claim_id"].lower()
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    screened, accepted_seed, accepted_index = 0, None, None
    modes: dict[str, dict[str, int]] = {arm: {} for arm in arms}
    for idx, seed in enumerate(contract["screening"]["order"], start=1):
        run_dir = runs / f"{prefix}__screen__{idx:02d}__{seed}"
        receipt = run_dir / "receipt.json"
        if not receipt.exists() or json.loads(receipt.read_text())["rc"] != 0:
            break
        rows = arm_rows(run_dir, arms)
        ev = exsel.evaluate({arm: rows[arm][1] for arm in arms}, claim)
        screened += 1
        for arm in arms:
            line, _raw = rows[arm]
            pa = ev["per_arm"][arm]
            rec, cls, bands, me = pa["record"], pa["class"], pa["bands"], pa["mode_evidence"]
            if rec["seed"] != seed or rec["scenario_id"] != claim["frozen"]["cell"]:
                raise ValueError(f"{run_dir}: row seed/cell differ from the screening slot")
            modes[arm][pa["mode"]] = modes[arm].get(pa["mode"], 0) + 1
            w.writerow(
                [
                    fmt(v)
                    for v in [
                        contract["claim_id"],
                        idx,
                        seed,
                        arm,
                        rec["scenario_id"],
                        rec["git_hash"],
                        rec["config_hash"],
                        exsel.row_sha256(line),
                        pa["mode"],
                        rec["success"],
                        rec["collision_event"],
                        rec["timeout_event"],
                        rec["termination_reason"],
                        rec["steps"],
                        rec["run_dt"],
                        rec["ped_collision_count"],
                        rec["obstacle_collision_count"],
                        rec["wall_collisions"],
                        rec["near_misses"],
                        rec["min_ped_separation_m"],
                        rec["outcome"],
                        rec["collision_kind"],
                        rec["min_surface_clearance_m"],
                        rec["exposure_steps"],
                        rec["time_to_goal_s"],
                        verdict(cls.get("outcome")),
                        verdict(cls.get("collision_kind")),
                        verdict(cls.get("min_exposure_steps")),
                        verdict(bands["min_surface_clearance_m"]),
                        verdict(bands["exposure_steps"]),
                        verdict(bands["time_to_goal_s"]),
                        all(cls.values()) and not any(b is False for b in bands.values()),
                        ev["accepted"],
                        me["algorithm_status"],
                        me["pedestrian_model_status"],
                        me["effective_view_degraded"],
                        me["planner_execution_mode"],
                        me["fallback_counters"],
                    ]
                ]
            )
        if ev["accepted"]:
            accepted_seed, accepted_index = seed, idx
            break
    summary = {
        "claim_id": contract["claim_id"],
        "budget": contract["screening"]["budget"],
        "seeds_screened": screened,
        "accepted_seed": accepted_seed,
        "accepted_screening_index": accepted_index,
        "result": (
            f"ACCEPTED seed {accepted_seed} at screening index {accepted_index} of "
            f"{contract['screening']['budget']}"
            if accepted_seed is not None
            else f"NO QUALIFYING EXAMPLE after {screened} seeds"
        ),
        "frozen_pattern": claim["frozen"]["pattern"],
        "frozen_cell": claim["frozen"]["cell"],
        "mode_counts": {arm: dict(sorted(modes[arm].items())) for arm in arms},
    }
    return buf.getvalue(), summary


def main() -> int:
    """Write the ledger and its summary next to it.

    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    contract = yaml.safe_load(a.contract.read_text(encoding="utf-8"))
    text, summary = build(contract, a.runs)
    a.out.write_text(text, encoding="utf-8")
    a.out.with_name("screening_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary["result"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
