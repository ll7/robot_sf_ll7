#!/usr/bin/env python3
"""Run ScenarioBelief drop-vs-retain safety contrast through the real runner.

Plain-language summary: issue #3471 showed in a controlled scenario that
dropping uncertain out-of-field-of-view agents from ``stream_gap`` can be less
safe than retaining them. Issue #3556 promotes that contrast to an opt-in real
benchmark-runner mode with explicit screening and provenance, without promoting
the result to a paper-grade claim.
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
from robot_sf.benchmark.scenario_belief_screening import (
    build_input_screening_report,
    build_screening_report,
)
from robot_sf.benchmark.scenario_belief_screening import (
    classify_screened_decision as _classify_screened_decision,
)
from robot_sf.planner.scenario_belief_adapter import SUPPORTED_UNCERTAINTY_PLANNER_KEYS
from robot_sf.training.scenario_loader import load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "belief-mode-safety-campaign.v1"
ISSUE = 3556
SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"

MODES = ("oracle", "uncertain_retained", "uncertain_dropped")
CAMPAIGN_ALGO = "stream_gap"
REQUIRED_MODES = ("oracle", "uncertain_retained", "uncertain_dropped")
_UNSUPPORTED_PROBE_KEY = "campaign_preflight_unsupported_probe"
DEFAULT_SCENARIO_SET = "configs/scenarios/sets/issue_3556_near_safe_occlusion_bearing_crossing.yaml"
DEFAULT_SEED_SET = "issue_3556_s3_smoke"
DEFAULT_SEEDS = [363501, 363502, 363503]
DEFAULT_LAUNCH_PACKET = (
    "configs/benchmarks/scenario_belief_drop_vs_retain_issue_3556_near_safe.yaml"
)
DEFAULT_FOV = 120.0
NEAR_SAFE_ORACLE_COLLISION_RATE = 0.25
NEAR_MISS_NOTE = "real benchmark near_misses + collisions + min_clearance"


def load_campaign_scenarios(set_path: Path, seeds: list[int]) -> list[dict[str, Any]]:
    """Load the predeclared scenario family and apply the pinned seed matrix."""
    scenarios = load_scenarios(set_path, base_dir=set_path)
    prepared: list[dict[str, Any]] = []
    for scenario in scenarios:
        entry = dict(scenario)
        entry["seeds"] = list(seeds)
        map_file = entry.get("map_file")
        if isinstance(map_file, str) and map_file.strip() and not Path(map_file).is_absolute():
            entry["map_file"] = str((set_path.parent / map_file).resolve())
        prepared.append(entry)
    return prepared


def write_algo_config(mode: str, out_dir: Path, *, fov_degrees: float) -> Path:
    """Write a per-mode stream_gap algorithm config consumed by map_runner."""
    config = {
        "algo": CAMPAIGN_ALGO,
        "allow_testing_algorithms": True,
        "belief_mode": mode,
        "belief_fov_degrees": fov_degrees,
    }
    path = out_dir / f"algo_{mode}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run_mode(
    mode: str,
    scenarios: list[dict[str, Any]],
    out_dir: Path,
    *,
    fov_degrees: float,
    horizon: int,
    dt: float,
    workers: int,
) -> list[dict[str, Any]]:
    """Run all scenarios x seeds for one belief mode through ``run_map_batch``."""
    algo_path = write_algo_config(mode, out_dir, fov_degrees=fov_degrees)
    episodes_path = out_dir / f"episodes_{mode}.jsonl"
    if episodes_path.exists():
        episodes_path.unlink()
    run_map_batch(
        scenarios,
        episodes_path,
        schema_path=Path(SCHEMA_PATH),
        algo=CAMPAIGN_ALGO,
        algo_config_path=str(algo_path),
        horizon=horizon,
        dt=dt,
        record_forces=False,
        workers=workers,
        resume=False,
        benchmark_profile="experimental",
    )
    if not episodes_path.exists():
        return []
    episodes = []
    for line_number, line in enumerate(
        episodes_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            episodes.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Error parsing JSONL on line {line_number} in {episodes_path}: {exc}"
            ) from exc
    return episodes


def _metric(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Read a numeric metric from a benchmark episode record."""
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return default


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode safety metrics for one belief mode."""
    n = len(records)
    if n == 0:
        return {"episodes": 0}
    collisions = [_metric(record, "total_collision_count") for record in records]
    near_misses = [_metric(record, "near_misses") for record in records]
    min_clear = [_metric(record, "min_clearance", default=float("nan")) for record in records]
    success = [_metric(record, "success") for record in records]
    valid_clear = [value for value in min_clear if np.isfinite(value)]
    return {
        "episodes": n,
        "collision_rate": round(sum(1 for count in collisions if count > 0) / n, 4),
        "total_collisions": int(sum(collisions)),
        "total_near_misses": int(sum(near_misses)),
        "mean_min_clearance": round(float(np.mean(valid_clear)), 4) if valid_clear else None,
        "worst_min_clearance": round(float(np.min(valid_clear)), 4) if valid_clear else None,
        "success_rate": round(sum(success) / n, 4),
    }


def classify_decision(by_mode: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Backward-compatible classifier entry point."""
    return classify_screened_decision(by_mode)


def classify_screened_decision(by_mode: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Classify the contrast with #3556 allowed decision labels."""
    return _classify_screened_decision(
        by_mode,
        oracle_near_safe_threshold=NEAR_SAFE_ORACLE_COLLISION_RATE,
    )


def _relative_repo_path(path: Path) -> str:
    """Return a stable repo-relative path when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def check_launch_packet_arm_contract(  # noqa: C901, PLR0912, PLR0915
    packet_path: Path,
    *,
    set_path: Path,
    seeds: list[int],
    fov_degrees: float,
) -> tuple[bool, str]:
    """Validate committed launch-packet arm metadata without rolling episodes."""
    if not packet_path.exists():
        return False, f"launch packet not found: {packet_path}"
    with packet_path.open(encoding="utf-8") as handle:
        packet = yaml.safe_load(handle)
    if not isinstance(packet, dict):
        return False, "launch packet must be a mapping"

    expected_family = _relative_repo_path(set_path)
    expected_modes = set(REQUIRED_MODES)
    failures: list[str] = []

    if packet.get("scenario_family") != expected_family:
        failures.append(
            f"top-level scenario_family must be {expected_family!r} "
            f"(got {packet.get('scenario_family')!r})"
        )

    seed_set_name = packet.get("seed_set")
    seed_sets = packet.get("seed_sets")
    if not isinstance(seed_set_name, str):
        failures.append("top-level seed_set name must be pinned")
    if not isinstance(seed_sets, dict) or seed_set_name not in seed_sets:
        failures.append(f"seed_sets must define {seed_set_name!r}")
    else:
        seed_entry = seed_sets[seed_set_name]
        if not isinstance(seed_entry, dict) or seed_entry.get("seeds") != seeds:
            failures.append(f"seed set {seed_set_name!r} must pin {seeds}")

    belief_modes = packet.get("belief_modes")
    if not isinstance(belief_modes, dict):
        failures.append("belief_modes must be a mapping")
    else:
        mode_keys = set(belief_modes)
        if mode_keys != expected_modes:
            failures.append(
                f"belief_modes must contain exactly {sorted(expected_modes)} "
                f"(got {sorted(mode_keys)})"
            )

    arms = packet.get("runner_arms")
    if not isinstance(arms, list):
        failures.append("runner_arms must be a list")
        arms = []
    arm_ids = {arm.get("arm_id") for arm in arms if isinstance(arm, dict)}
    if arm_ids != expected_modes:
        failures.append(
            f"runner_arms must contain exactly {sorted(expected_modes)} "
            f"(got {sorted(str(arm_id) for arm_id in arm_ids)})"
        )

    expected_gate = {
        "oracle": False,
        "uncertain_retained": False,
        "uncertain_dropped": True,
    }
    expected_visibility = {
        "oracle": "all_agents_retained",
        "uncertain_retained": "low_confidence_agents_retained",
        "uncertain_dropped": "low_confidence_agents_dropped",
    }
    for arm in arms:
        if not isinstance(arm, dict):
            failures.append("each runner_arms entry must be a mapping")
            continue
        arm_id = arm.get("arm_id")
        if arm_id not in expected_modes:
            continue
        if arm.get("scenario_family") != expected_family:
            failures.append(f"{arm_id} arm scenario_family must be {expected_family!r}")
        if arm.get("seed_set") != seed_set_name:
            failures.append(f"{arm_id} arm seed_set must be {seed_set_name!r}")
        if arm.get("algo") != CAMPAIGN_ALGO:
            failures.append(f"{arm_id} arm algo must be {CAMPAIGN_ALGO!r}")
        algo_config = arm.get("algo_config")
        if not isinstance(algo_config, dict):
            failures.append(f"{arm_id} arm algo_config must be a mapping")
            continue
        if algo_config.get("belief_mode") != arm_id:
            failures.append(f"{arm_id} arm belief_mode must match arm_id")
        if algo_config.get("belief_fov_degrees") != fov_degrees:
            failures.append(f"{arm_id} arm belief_fov_degrees must be {fov_degrees}")
        if arm.get("expected_gate_enabled") is not expected_gate[arm_id]:
            failures.append(f"{arm_id} arm expected_gate_enabled drifted")
        if arm.get("expected_planner_visibility") != expected_visibility[arm_id]:
            failures.append(f"{arm_id} arm expected_planner_visibility drifted")

    if failures:
        return False, "; ".join(failures)
    return True, (
        f"launch packet pins {len(expected_modes)} runner arms to {expected_family} "
        f"with seed set {seed_set_name!r}"
    )


def check_campaign_readiness(
    set_path: Path,
    seeds: list[int],
    *,
    fov_degrees: float,
    horizon: int,
    dt: float,
    workers: int,
    launch_packet: Path | None = None,
) -> dict[str, Any]:
    """Fail-closed CPU-only readiness gate over #3556 real-runner inputs."""
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    mode_set = set(MODES)
    add(
        "belief_modes_pinned",
        mode_set == set(REQUIRED_MODES),
        f"pinned exactly {sorted(REQUIRED_MODES)}"
        if mode_set == set(REQUIRED_MODES)
        else f"MODES must equal {sorted(REQUIRED_MODES)}",
    )

    seeds_ok = (
        isinstance(seeds, list)
        and bool(seeds)
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
        and len(set(seeds)) == len(seeds)
    )
    add(
        "seeds_pinned",
        seeds_ok,
        f"{len(seeds)} unique integer seeds pinned"
        if seeds_ok
        else "seeds must be non-empty list of unique integers",
    )

    set_exists = set_path.exists()
    add(
        "scenario_set_exists",
        set_exists,
        f"scenario set present at {set_path}"
        if set_exists
        else f"scenario set not found: {set_path}",
    )

    if set_exists:
        try:
            screening_scenarios = load_campaign_scenarios(set_path, seeds)
            screening_inputs = build_input_screening_report(
                scenarios=screening_scenarios,
                seeds=seeds,
                fov_degrees=fov_degrees,
                required_modes=REQUIRED_MODES,
                scenario_set=set_path,
                launch_packet=launch_packet,
            )
            screening_ok = bool(screening_inputs["ready"])
            screening_detail = (
                "scenario IDs, seeds, modes, and out-of-FOV sidecar contract pinned"
                if screening_ok
                else f"screening input checks failed: {screening_inputs['failed_checks']}"
            )
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            screening_ok = False
            screening_detail = f"failed to load scenario screening inputs: {exc}"
    else:
        screening_ok = False
        screening_detail = "scenario set missing; screening inputs unavailable"
    add("scenario_belief_screening_inputs", screening_ok, screening_detail)

    if launch_packet is None:
        add(
            "launch_packet_arm_contract",
            True,
            "launch packet arm contract not requested for this readiness call",
        )
    else:
        arm_contract_ok, arm_contract_detail = check_launch_packet_arm_contract(
            launch_packet,
            set_path=set_path,
            seeds=seeds,
            fov_degrees=fov_degrees,
        )
        add("launch_packet_arm_contract", arm_contract_ok, arm_contract_detail)

    geometry = {"fov_degrees": fov_degrees, "horizon": horizon, "dt": dt, "workers": workers}
    bad_geometry = [f"{key}={value} must be > 0" for key, value in geometry.items() if value <= 0]
    add(
        "run_geometry_positive",
        not bad_geometry,
        "fov/horizon/dt/workers all > 0" if not bad_geometry else "; ".join(bad_geometry),
    )

    planner_ok = (
        CAMPAIGN_ALGO in SUPPORTED_UNCERTAINTY_PLANNER_KEYS
        and _UNSUPPORTED_PROBE_KEY not in SUPPORTED_UNCERTAINTY_PLANNER_KEYS
    )
    add(
        "uncertainty_planner_contract",
        planner_ok,
        f"campaign algo {CAMPAIGN_ALGO!r} consumes uncertainty; unsupported keys fail closed"
        if planner_ok
        else f"campaign algo {CAMPAIGN_ALGO!r} not in supported set",
    )

    threshold_ok = 0.0 < NEAR_SAFE_ORACLE_COLLISION_RATE < 1.0
    unsafe_rate = min(1.0, NEAR_SAFE_ORACLE_COLLISION_RATE + 0.5)
    synthetic_unsafe = {
        "oracle": {"episodes": 3, "collision_rate": unsafe_rate, "total_near_misses": 0},
        "uncertain_retained": {"episodes": 3, "collision_rate": 0.0, "total_near_misses": 0},
        "uncertain_dropped": {"episodes": 3, "collision_rate": 1.0, "total_near_misses": 5},
    }
    screened = classify_screened_decision(synthetic_unsafe)
    contract_ok = (
        threshold_ok
        and screened.get("decision") == "inconclusive_oracle_unsafe"
        and screened.get("oracle_near_safe") is False
    )
    add(
        "oracle_near_safety_contract",
        contract_ok,
        f"unsafe oracle (rate {unsafe_rate}) blocks interpretation at threshold "
        f"{NEAR_SAFE_ORACLE_COLLISION_RATE}"
        if contract_ok
        else f"oracle-near-safety contract violated: {screened}",
    )

    failed = [check["name"] for check in checks if not check["passed"]]
    return {
        "schema_version": "belief-mode-campaign-readiness.v1",
        "issue": ISSUE,
        "campaign_algo": CAMPAIGN_ALGO,
        "ready": not failed,
        "checks": checks,
        "failed_checks": failed,
        "claim_boundary": (
            "Input-pinning fail-closed readiness gate for the real campaign runner; "
            "does not roll episodes or interpret drop-vs-retain outcomes."
        ),
    }


def run_campaign(
    set_path: Path,
    seeds: list[int],
    out_dir: Path,
    *,
    fov_degrees: float,
    horizon: int,
    dt: float,
    workers: int,
) -> dict[str, Any]:
    """Run all three belief modes and assemble the screened report."""
    readiness = check_campaign_readiness(
        set_path,
        seeds,
        fov_degrees=fov_degrees,
        horizon=horizon,
        dt=dt,
        workers=workers,
    )
    if not readiness["ready"]:
        raise RuntimeError(
            "campaign readiness gate failed (no episodes rolled); "
            f"failed checks: {readiness['failed_checks']}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = load_campaign_scenarios(set_path, seeds)
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        records = run_mode(
            mode,
            scenarios,
            out_dir,
            fov_degrees=fov_degrees,
            horizon=horizon,
            dt=dt,
            workers=workers,
        )
        by_mode[mode] = aggregate(records)

    screening = build_screening_report(
        scenarios=scenarios,
        seeds=seeds,
        by_mode=by_mode,
        oracle_near_safe_threshold=NEAR_SAFE_ORACLE_COLLISION_RATE,
        fov_degrees=fov_degrees,
        required_modes=REQUIRED_MODES,
        scenario_set=set_path,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "evidence_tier": "smoke_evidence",
        "claim_boundary": (
            "Real benchmark-runner smoke on bounded crossing family; FOV uncertainty source is "
            "not a calibrated perception model; not nominal or paper-grade until full "
            "predeclared matrix, seed-sufficiency budget, and claim-card review."
        ),
        "runner": "robot_sf.benchmark.map_runner.run_map_batch",
        "scenario_set": str(set_path),
        "scenario_names": [scenario.get("name") for scenario in scenarios],
        "seeds": list(seeds),
        "fov_degrees": fov_degrees,
        "metric_note": NEAR_MISS_NOTE,
        "by_mode": by_mode,
        "screening": screening,
        "decision": screening["decision"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-set", type=Path, default=Path(DEFAULT_SCENARIO_SET))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--launch-packet",
        type=Path,
        default=None,
        help=(
            "Optional launch packet to validate against runtime scenario/seed knobs. "
            "Defaults to the committed #3556 near-safe packet for default inputs."
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("output/issue_3556_belief_mode_campaign")
    )
    parser.add_argument("--fov-degrees", type=float, default=DEFAULT_FOV)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run only the CPU-only readiness gate. Rolls no episodes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    launch_packet = args.launch_packet
    if (
        launch_packet is None
        and _relative_repo_path(args.scenario_set) == DEFAULT_SCENARIO_SET
        and args.seeds == DEFAULT_SEEDS
    ):
        launch_packet = Path(DEFAULT_LAUNCH_PACKET)
    readiness = check_campaign_readiness(
        args.scenario_set,
        args.seeds,
        fov_degrees=args.fov_degrees,
        horizon=args.horizon,
        dt=args.dt,
        workers=args.workers,
        launch_packet=launch_packet,
    )
    if args.preflight_only or not readiness["ready"]:
        print(json.dumps(readiness, indent=2, sort_keys=True))
        return 0 if readiness["ready"] else 1

    report = run_campaign(
        args.scenario_set,
        args.seeds,
        args.out_dir,
        fov_degrees=args.fov_degrees,
        horizon=args.horizon,
        dt=args.dt,
        workers=args.workers,
    )
    report["readiness"] = readiness
    report["generated_at_utc"] = datetime.now(UTC).isoformat()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
