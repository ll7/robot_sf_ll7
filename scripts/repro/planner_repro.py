"""Replay the frozen development comparisons for two Robot SF planner candidates."""

# evidence-writer-exempt: Replay restores exact, hash-checked simulator inputs
# in a temporary directory. It publishes no generated evidence files.

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pysocialforce
import yaml
from loguru import logger

import robot_sf
from robot_sf.benchmark.map_runner.map_runner import _run_map_episode, build_map_policy
from robot_sf.evidence.writers import sha256_file
from robot_sf.training.scenario_loader import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "docs/context/evidence/planner_search_2026-11"
REPLAY_SEEDS = (7301, 7319, 7330)
ARCHIVED_ARMS = ("orca", "drammpi2", "drammpi3")
ARMS = (*ARCHIVED_ARMS, "sampler")
DEVELOPMENT_SEEDS = tuple(range(7301, 7331))
GRIDS = ("release", "v1", "v2", "v3")
OUTCOMES = ("route_complete", "collision_event", "timeout_event")


def read_json(path: Path, *, uncompressed_sha256: str | None = None) -> dict:
    """Load JSON and optionally verify the original, uncompressed bytes."""
    raw = path.read_bytes()
    content = gzip.decompress(raw) if path.suffix == ".gz" else raw
    if (
        uncompressed_sha256 is not None
        and hashlib.sha256(content).hexdigest() != uncompressed_sha256
    ):
        raise ValueError(f"Original JSON bytes differ for {path.name}")
    return json.loads(content)


def verify_environment(manifest: dict, *, allow_rebuilt_rvo2: bool) -> None:
    """Check the shared runtime and exact inputs before any episode runs."""
    import_root = Path(robot_sf.__file__).resolve().parent
    if import_root != ROOT / "robot_sf":
        raise ValueError("The interpreter must load Robot SF from this checkout.")

    env = manifest["environment"]
    if sys.version.split()[0] != env["python"].split()[0]:
        raise ValueError(f"Python version differs from archive: {sys.version}")
    for package, expected in env["packages"].items():
        if version(package) != expected:
            raise ValueError(f"{package} version {version(package)} != {expected}")
    if Path(pysocialforce.__file__).resolve() != ROOT / env["pysocialforce"]:
        raise ValueError("The interpreter must load the repository's fast-pysf source.")
    verify_rvo2(env["rvo2_extension_sha256"], allow_rebuilt=allow_rebuilt_rvo2)

    sim = manifest["simulator"]
    verify_archive(manifest)
    pins = {
        **{ROOT / name: digest for name, digest in env["lockfiles_sha256"].items()},
        ROOT / sim["release_horizon_schedule"]: sim["release_horizon_schedule_sha256"],
        **{ROOT / arm["config_path"]: arm["config_sha256"] for arm in manifest["arms"].values()},
    }
    for path, expected in pins.items():
        if sha256_file(path) != expected:
            raise ValueError(f"SHA-256 mismatch for {path.relative_to(ROOT)}")


def verify_archive(manifest: dict) -> None:
    """Reject changed or missing frozen inputs and archived outcomes."""
    for name, expected in manifest["files_sha256"].items():
        if sha256_file(ARCHIVE / name) != expected:
            raise ValueError(f"SHA-256 mismatch for archive file {name}")


def verify_rvo2(archived_hash: str, *, allow_rebuilt: bool) -> None:
    """Require native RVO2 and check the binary identity."""
    rvo2_path = import_module("rvo2").__file__
    if rvo2_path is None:
        raise ValueError("Native RVO2 module does not have a file.")
    binary = sha256_file(Path(rvo2_path))
    if binary != archived_hash:
        if not allow_rebuilt:
            raise ValueError(
                "Native RVO2 binary differs; use --allow-rebuilt-rvo2 for a source rebuild."
            )
        print(f"Native RVO2 binary was rebuilt (SHA-256 {binary}).", flush=True)


def prepare_cases(manifest: dict, scratch: Path, seeds: tuple[int, ...]) -> dict:
    """Bind each replay key to the hashed case and map used in acquisition."""
    hashes = read_json(ARCHIVE / "input_hashes.json.gz")
    challenge_files = read_json(ARCHIVE / "challenge_inputs.json.gz")
    sim = manifest["simulator"]
    cases = {}
    for case_id, case in read_json(ARCHIVE / "release_cases.json.gz").items():
        canonical = json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
        map_path = ROOT / case["map_file"]
        case_hashes = {"case": hashlib.sha256(canonical).hexdigest(), "map": sha256_file(map_path)}
        if case_hashes != hashes["release"][case_id]:
            raise ValueError(f"Release input changed for {case_id}")
        for seed in seeds:
            cases["release", case_id, seed] = (
                {**case, "map_file": str(map_path)},
                ROOT / sim["release_scenario_path"],
                case_hashes["map"],
                None,
            )
    for grid in GRIDS[1:]:
        suite = read_json(
            ARCHIVE / f"challenge_suite.{grid}.json.gz",
            uncompressed_sha256=manifest["challenge_suite_json_sha256"][grid],
        )
        for entry in suite["cases"]:
            case_id = entry["case"]["id"]
            for seed in seeds:
                key = f"{grid}/{case_id}/{seed}"
                case_dir = scratch / key
                case_dir.mkdir(parents=True)
                pin = hashes["challenge"][key]
                for name in ("map.json", "routes.json", "scenario.yaml"):
                    path = case_dir / name
                    path.write_bytes(challenge_files[key][name].encode("utf-8"))
                    if sha256_file(path) != pin["sha256"][name]:
                        raise ValueError(f"Challenge input changed for {key}/{name}")
                source = case_dir / "scenario.yaml"
                (scenario,) = load_scenarios(source)
                if scenario["seeds"] != [seed]:
                    raise ValueError(f"Challenge seed differs for {key}")
                cases[grid, case_id, seed] = (
                    scenario,
                    source,
                    pin["sha256"]["map.json"],
                    pin["pedestrian_count"],
                )
    return cases


def replay(task: tuple) -> tuple:
    """Run a native episode and return the fields that were archived for it."""
    key, inputs, name, arm, sim = task
    grid, case_id, seed = key
    if seed not in DEVELOPMENT_SEEDS or name not in ARMS:
        raise ValueError(f"Outside the fixed replay: {key}/{name}")
    scenario, scenario_path, map_hash, ped_count = inputs
    algo = arm["factory_key"]
    config = yaml.safe_load((ROOT / arm["config_path"]).read_bytes())
    config.update(arm.get("config_overrides", {}))
    record: dict[str, Any] = dict(
        _run_map_episode(
            scenario,
            seed,
            horizon=None,
            dt=sim["control_period_s"],
            record_forces=sim["record_forces"],
            snqi_weights=None,
            snqi_baseline=None,
            algo=algo,
            scenario_path=scenario_path,
            algo_config=config,
            policy_builder=build_map_policy,
        )
    )
    metadata = record["algorithm_metadata"]
    runtime = metadata["planner_runtime"]
    integrity = record["integrity"]
    physics = record["effective_physics"]
    clearance = physics["clearance_definition"]
    horizon = (
        scenario["simulation_config"]["max_episode_steps"]
        if grid == "release"
        else sim["challenge_horizon_steps"]
    )
    if (
        (record["scenario_id"], record["seed"], record["algo"]) != (case_id, seed, algo)
        or metadata["canonical_algorithm"] != algo
        or metadata["status"] != "ok"
        or runtime["planner_type"] != arm["adapter_class"]
        or runtime.get("fallback_triggered", False)
        or integrity["contradictions"]
        or integrity["effective_view"]["degraded"] is not False
        or record["pedestrian_model"]["fallback_degraded_status"] != "native"
        or record["selected_map_identity"]["sha256"] != map_hash
        or record["effective_budget_steps"] != horizon
        or record["observation_level"] != sim["observation_level"]
        or physics["robot_kinematics"] != sim["drive"]
        or physics["integration"]["dt_s"] != sim["control_period_s"]
        or clearance["robot_radius_m"] != sim["robot_radius_m"]
        or clearance["pedestrian_radius_m"] != sim["pedestrian_contact_radius_m"]
    ):
        raise ValueError(f"Runtime identity or integrity check failed for {key}/{name}")
    if ped_count is not None and (
        record["spawn_validity"]["reset_clearance_status"] != "available"
        or record["spawn_validity"]["invalid_run"] is not False
        or integrity["effective_view"]["extracted_ped_count"] != ped_count
    ):
        raise ValueError(f"Challenge reset check failed for {key}/{name}")
    flags = record["outcome"]
    if (
        any(type(flags[field]) is not bool for field in OUTCOMES)
        or sum(flags[field] for field in OUTCOMES) != 1
    ):
        raise ValueError(f"Invalid outcome for {key}/{name}: {flags}")
    metrics = record["metrics"]
    return (
        key,
        name,
        (
            next(field for field in OUTCOMES if flags[field]),
            int(metrics["ped_collision_count"] > 0),
            int(metrics["obstacle_collision_count"] > 0),
            record["steps"],
        ),
    )


def archived_subset(seeds: tuple[int, ...]) -> dict:
    """Index archived outcomes, rejecting duplicate and unpaired keys.

    Returns:
        Expected outcomes by grid, case, seed and planner.
    """
    expected = {}
    with gzip.open(
        ARCHIVE / "paired_outcomes.csv.gz", "rt", newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            seed = int(row["seed"])
            if seed in seeds:
                key = row["grid"], row["case"], seed
                if key in expected:
                    raise ValueError(f"Duplicate archive row: {key}")
                expected[key] = {
                    arm: (
                        row[arm],
                        *(int(row[f"{arm}_{f}"]) for f in ("pedestrian", "obstacle", "steps")),
                    )
                    for arm in ARCHIVED_ARMS
                }
    load_sampler_outcomes(expected, seeds)
    return expected


def load_sampler_outcomes(expected: dict, seeds: tuple[int, ...]) -> None:
    """Join archived sampler rows to their matching ORCA and DRA-MPPI inputs."""
    with gzip.open(
        ARCHIVE / "sampler_outcomes.csv.gz", "rt", newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            seed = int(row["seed"])
            if seed in seeds:
                key = row["grid"], row["case"], seed
                if "sampler" in expected[key]:
                    raise ValueError(f"Duplicate sampler row: {key}")
                expected[key]["sampler"] = (
                    row["outcome"],
                    int(row["pedestrian"]),
                    int(row["obstacle"]),
                    int(row["steps"]),
                )
    if any("sampler" not in block for block in expected.values()):
        raise ValueError("Sampler and comparator archives do not cover the same inputs.")


def parse_args() -> argparse.Namespace:
    """Parse the choice of planners, seed block and native extension policy.

    Returns:
        The requested replay or summary options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 25), metavar="1–24")
    parser.add_argument(
        "--allow-rebuilt-rvo2", action="store_true", help="Allow a different native binary hash."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all-seeds", action="store_true", help="Replay all 30 development seeds.")
    parser.add_argument("--arms", nargs="+", choices=ARMS, help="Select planners to replay.")
    mode.add_argument(
        "--summary", action="store_true", help="Recalculate paired results from archived rows."
    )
    mode.add_argument(
        "--smoke", action="store_true", help="Run one release setting on three seeds."
    )
    args = parser.parse_args()
    if args.arms and len(set(args.arms)) != len(args.arms):
        parser.error("--arms must list each planner once")
    return args


def run_episodes(
    manifest: dict, args: argparse.Namespace, seeds: tuple[int, ...], arms: tuple[str, ...]
) -> list:
    """Execute the requested archive rows with the normal map-runner episode entry.

    Returns:
        The observed outcomes, keyed by input and planner.
    """
    with TemporaryDirectory(prefix="planner-repro-") as directory:
        cases = prepare_cases(manifest, Path(directory), seeds)
        if len(cases) != len(seeds) * sum(manifest["grids"].values()):
            raise ValueError("The replay has missing or duplicate inputs.")
        if args.smoke:
            first_release = next(case for grid, case, _ in cases if grid == "release")
            cases = {
                key: value for key, value in cases.items() if key[:2] == ("release", first_release)
            }
        jobs = (
            (key, inputs, arm, manifest["arms"][arm], manifest["simulator"])
            for key, inputs in cases.items()
            for arm in arms
        )
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("fork")
        ) as pool:
            return list(pool.map(replay, jobs))


def compare_results(observed: list, expected: dict) -> int:
    """Check paired outcome, contact and step-count fields.

    Returns:
        Exit code 1 if any rows differ, or 0 if all rows match.
    """
    mismatches = 0
    for key, arm, result in observed:
        baseline = expected[key][arm]
        if result != baseline:
            print(f"MISMATCH {key}/{arm}: archived={baseline}, replay={result}")
            mismatches += 1
    print(
        f"Compared {len(observed)} episodes on {len({row[0] for row in observed})} paired inputs: "
        f"{mismatches} mismatches (outcome, pedestrian contact, obstacle contact, steps)."
    )
    return int(mismatches > 0)


def summarize(manifest: dict) -> None:
    """Print outcome counts and paired 95% intervals in percentage points."""
    rows = archived_subset(DEVELOPMENT_SEEDS)
    config = manifest["paired_intervals"]
    rng = np.random.default_rng(config["bootstrap_rng_seed"])
    n_seeds = len(DEVELOPMENT_SEEDS)
    draw = rng.integers(n_seeds, size=(config["draws"], n_seeds))
    quantiles = [(1 - config["confidence"]) / 2, (1 + config["confidence"]) / 2]
    print(
        "| Input | Planner | Complete / collision / timeout | Δ completion | Δ collision | Δ timeout | Δ pedestrian contact | Δ obstacle contact |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for grid in GRIDS:
        totals = {arm: np.zeros((n_seeds, len(OUTCOMES) + 2), dtype=int) for arm in ARMS}
        for (name, _, seed), block in rows.items():
            if name != grid:
                continue
            for arm, (outcome, pedestrian, obstacle, _) in block.items():
                values = [*(outcome == label for label in OUTCOMES), pedestrian, obstacle]
                totals[arm][seed - DEVELOPMENT_SEEDS[0]] += values
        reference = totals["orca"]
        if not all(
            np.all(values[:, :3].sum(axis=1) == manifest["grids"][grid])
            for values in totals.values()
        ):
            raise ValueError(f"Incomplete or invalid outcome totals for {grid}")
        trials = n_seeds * manifest["grids"][grid]
        for arm, values in totals.items():
            counts = " / ".join(str(count) for count in values.sum(axis=0)[:3])
            effects = ["—"] * 5
            if arm != "orca":
                delta = values - reference
                points = 100 * delta.sum(axis=0) / trials
                low, high = np.quantile(100 * delta[draw].sum(axis=1) / trials, quantiles, axis=0)
                effects = [
                    f"{point:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                    for point, lo, hi in zip(points, low, high, strict=True)
                ]
            print(f"| {grid} | {arm} | {counts} | {' | '.join(effects)} |")


def main() -> int:
    """Check simulator and input pins, then compare native outcomes."""
    args = parse_args()
    seeds = DEVELOPMENT_SEEDS if args.all_seeds else REPLAY_SEEDS
    arms = tuple(args.arms or ARMS)
    logger.remove()
    manifest = read_json(ARCHIVE / "manifest.json")
    if manifest["replay"] != {"arms": list(ARMS), "seeds": list(REPLAY_SEEDS)}:
        raise ValueError("Replay manifest differs from the fixed subset.")
    if tuple(manifest["final_seeds"]) != DEVELOPMENT_SEEDS:
        raise ValueError("Manifest differs from the development seed block.")
    if args.summary:
        verify_archive(manifest)
        summarize(manifest)
        return 0
    verify_environment(manifest, allow_rebuilt_rvo2=args.allow_rebuilt_rvo2)
    expected = archived_subset(seeds)
    observed = run_episodes(manifest, args, seeds, arms)
    if not args.smoke and {key for key, _, _ in observed} != expected.keys():
        raise ValueError("Replay inputs and archive rows do not cover the same fixed grid.")
    return compare_results(observed, expected)


if __name__ == "__main__":
    raise SystemExit(main())
