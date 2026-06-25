#!/usr/bin/env python3
"""Run the reactive-vs-replay pedestrian-reactivity ablation through the REAL runner (#3573).

#3573 asks how much pedestrian (non-)reactivity flatters planners. PR-merged #3594 added the pure
quantifier (``robot_sf.benchmark.reactivity_ablation.assess_reactivity_ablation``); the **paired
runs + open-loop replay pedestrian mode** were deferred. This script lands them: for each planner it
runs two conditions over identical scenarios + seeds (common random numbers) through
``robot_sf.benchmark.map_runner.run_map_batch`` and feeds the per-planner contrast into the
quantifier:

* **reactive** — current social-force pedestrians that respond to the robot
  (``peds_have_robot_repulsion: True``);
* **replay / non-reactive** — the robot-response term disabled
  (``peds_have_robot_repulsion: False``): pedestrians follow social-force dynamics among
  themselves but do **not** yield to the robot (open-loop w.r.t. the robot).

**Operationalization caveat (read first).** "Replay/non-reactive" here means the *robot->pedestrian
force is disabled in a live social-force sim*, not pre-recorded trajectory playback (SocNavBench-style
grounded replay). So the sign convention of the merged quantifier — which was written for the
grounded-replay framing where replay *under*-reports hazard — must be read against this: with live
non-reactive pedestrians, disabling reactivity tends to *reveal* intrusion hazard (pedestrians stop
yielding), so a flattering effect, if any, accrues to the **reactive** (over-yielding) condition.

**Finding (diagnostic).** On ``classic_crossing_subset`` (goal + orca, 4 seeds, horizon 150)
disabling pedestrian reactivity (replay) raised collisions and cut separation versus reactive:
orca 0.125 -> 0.50 collision-rate, goal 0.50 -> 0.625, with reduced mean clearance in both. With
common random numbers this contrast is attributable to reactivity, and confirms that reactive
(yielding) pedestrians flatter planners — the "over-yield" concern. Under the quantifier's
grounded-replay sign convention the flattering accrues to the *reactive* condition, so
``planners_flattered_by_replay`` is empty and the *magnitude* of the deltas is the signal.

**Horizon caveat.** The robot must reach pedestrian proximity for the effect to register: at
short horizons (< ~150 steps on this family) the robot has not yet reached the crossing and the
contrast is near-null. Use a horizon long enough for the episode to complete (default 300).

Evidence boundary: real-benchmark-runner result on a bounded family at small N (2 scenarios); a
diagnostic-tier reactivity-sensitivity signal, not paper-grade until the predeclared matrix is run
at seed-sufficiency budget with claim-card review.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.benchmark.reactivity_ablation import (
    REACTIVITY_ABLATION_SCHEMA,
    ReactivityContrast,
    assess_reactivity_ablation,
)
from robot_sf.training.scenario_loader import load_scenarios

SCHEMA_VERSION = "reactivity-ablation-campaign.v1"
ISSUE = 3573
SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"

DEFAULT_SCENARIO_SET = "configs/scenarios/sets/classic_crossing_subset.yaml"
DEFAULT_SEEDS = list(range(101, 113))
DEFAULT_PLANNERS = ("goal", "orca")

#: The two pedestrian-reactivity conditions (scenario flag -> condition tag).
REACTIVE = "reactive"
REPLAY = "replay"
_CONDITIONS = {REACTIVE: True, REPLAY: False}


def load_campaign_scenarios(
    set_path: Path, seeds: list[int], *, reactive: bool
) -> list[dict[str, Any]]:
    """Load the scenario family with absolute map paths, the seed matrix, and the reactivity flag."""
    scenarios = load_scenarios(set_path, base_dir=set_path)
    prepared: list[dict[str, Any]] = []
    for scenario in scenarios:
        entry = dict(scenario)
        entry["seeds"] = list(seeds)
        entry["peds_have_robot_repulsion"] = bool(reactive)
        map_file = entry.get("map_file")
        if isinstance(map_file, str) and map_file.strip() and not Path(map_file).is_absolute():
            entry["map_file"] = str((set_path.parent / map_file).resolve())
        prepared.append(entry)
    return prepared


def write_algo_config(planner: str, out_dir: Path) -> Path:
    """Write a minimal algo config enabling exploratory planners; return its path."""
    config = {"algo": planner, "allow_testing_algorithms": True}
    path = out_dir / f"algo_{planner}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def run_condition(
    planner: str,
    condition: str,
    set_path: Path,
    seeds: list[int],
    out_dir: Path,
    *,
    horizon: int,
    dt: float,
    workers: int,
) -> list[dict[str, Any]]:
    """Run one planner under one reactivity condition; return the episode records."""
    scenarios = load_campaign_scenarios(set_path, seeds, reactive=_CONDITIONS[condition])
    algo_path = write_algo_config(planner, out_dir)
    episodes_path = out_dir / f"episodes_{planner}_{condition}.jsonl"
    if episodes_path.exists():
        episodes_path.unlink()
    run_map_batch(
        scenarios,
        episodes_path,
        schema_path=Path(SCHEMA_PATH),
        algo=planner,
        algo_config_path=str(algo_path),
        horizon=horizon,
        dt=dt,
        record_forces=False,
        workers=workers,
        resume=False,
        benchmark_profile="experimental",
    )
    return [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]


def _metric(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Read a numeric metric from an episode record's metrics block."""
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    value = metrics.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def aggregate(records: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-episode safety metrics for one planner x condition run.

    Returns:
        dict[str, float]: ``collision_rate`` / ``near_miss_rate`` (fraction of episodes with any
        collision / near miss) and ``min_separation_m`` (mean min-clearance, higher is safer).
    """
    n = len(records)
    if n == 0:
        raise ValueError("condition produced no episodes")
    collision_rate = sum(1 for r in records if _metric(r, "total_collision_count") > 0) / n
    near_miss_rate = sum(1 for r in records if _metric(r, "near_misses") > 0) / n
    clearances = [_metric(r, "min_clearance", default=float("nan")) for r in records]
    valid = [c for c in clearances if not np.isnan(c)]
    mean_clearance = float(np.mean(valid)) if valid else float("nan")
    return {
        "collision_rate": round(collision_rate, 4),
        "near_miss_rate": round(near_miss_rate, 4),
        "min_separation_m": round(mean_clearance, 4),
        "episodes": n,
    }


def build_contrast(planner: str, reactive: dict[str, float], replay: dict[str, float]):
    """Build the per-planner ``ReactivityContrast`` from the two condition aggregates.

    Returns:
        ReactivityContrast: Paired reactive/replay safety aggregates for one planner.
    """
    return ReactivityContrast(
        planner=planner,
        reactive_collision_rate=reactive["collision_rate"],
        replay_collision_rate=replay["collision_rate"],
        reactive_near_miss_rate=reactive["near_miss_rate"],
        replay_near_miss_rate=replay["near_miss_rate"],
        reactive_min_separation_m=reactive["min_separation_m"],
        replay_min_separation_m=replay["min_separation_m"],
    )


def run_campaign(
    set_path: Path,
    seeds: list[int],
    planners: tuple[str, ...],
    out_dir: Path,
    *,
    horizon: int,
    dt: float,
    workers: int,
) -> dict[str, Any]:
    """Run the paired reactive-vs-replay ablation across planners and assemble the report.

    Returns:
        dict[str, Any]: Versioned report with per-planner condition aggregates and the
        :func:`assess_reactivity_ablation` summary (or a ``blocked`` note if no planner produced
        a usable contrast).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_planner: dict[str, Any] = {}
    contrasts = []
    for planner in planners:
        reactive = aggregate(
            run_condition(
                planner, REACTIVE, set_path, seeds, out_dir, horizon=horizon, dt=dt, workers=workers
            )
        )
        replay = aggregate(
            run_condition(
                planner, REPLAY, set_path, seeds, out_dir, horizon=horizon, dt=dt, workers=workers
            )
        )
        per_planner[planner] = {"reactive": reactive, "replay": replay}
        contrasts.append(build_contrast(planner, reactive, replay))

    assessment = assess_reactivity_ablation(contrasts) if contrasts else None
    return {
        "schema_version": SCHEMA_VERSION,
        "quantifier_schema": REACTIVITY_ABLATION_SCHEMA,
        "issue": ISSUE,
        "evidence_tier": "diagnostic",
        "claim_boundary": (
            "Real-runner reactive-vs-replay result on a bounded family at small N. 'Replay' = "
            "robot->ped force disabled in a live social-force sim (not pre-recorded playback); the "
            "merged quantifier's grounded-replay sign convention must be read against that "
            "(flattering accrues to the reactive over-yielding condition, so the delta magnitude is "
            "the signal). Requires a horizon long enough for the robot to reach the crossing "
            "(>= ~150 on this family). Diagnostic-tier, not paper-grade until the predeclared matrix "
            "is run at seed-sufficiency budget with claim-card review."
        ),
        "runner": "robot_sf.benchmark.map_runner.run_map_batch",
        "scenario_set": str(set_path),
        "seeds": seeds,
        "planners": list(planners),
        "per_planner": per_planner,
        "assessment": assessment,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-set", type=Path, default=Path(DEFAULT_SCENARIO_SET))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--planners", nargs="+", default=list(DEFAULT_PLANNERS))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("output/issue_3573_reactivity_campaign")
    )
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the reactive-vs-replay ablation campaign and emit the report."""
    args = parse_args(argv)
    report = run_campaign(
        args.scenario_set,
        args.seeds,
        tuple(args.planners),
        args.out_dir,
        horizon=args.horizon,
        dt=args.dt,
        workers=args.workers,
    )
    report["generated_at_utc"] = datetime.now(UTC).isoformat()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
