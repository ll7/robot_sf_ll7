"""Run and gate VV-2 reference planners against a pinned release matrix.

The run mode is intended for a cluster compute node. Preflight and gate modes
are read-only with respect to the simulator and can run in CI against preserved
episode artifacts. A failed gate still writes both report formats.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.reference_oracle_report import evaluate_oracles, render_markdown
from robot_sf.benchmark.runner import load_scenario_matrix

ROOT = Path(__file__).resolve().parents[2]
EPISODE_SCHEMA = ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"


def _repo_path(value: str) -> Path:
    """Resolve a repository-relative config path without allowing an escape."""
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path escapes repository: {value}")
    if not path.is_file():
        raise ValueError(f"Required file is missing: {value}")
    return path


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    """Read one Git identity value from the owning worktree."""
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _release_matrix_contract(  # noqa: C901 - fail-closed release manifest validation.
    config: dict[str, Any], manifest_path: Path
) -> tuple[dict[str, Any], Path, list[dict[str, Any]], list[str], list[int]]:
    """Bind the configured matrix, seed schedule, and horizon to the release."""
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Release manifest must be a mapping")
    scenario_spec = manifest.get("scenario")
    seed_spec = manifest.get("seed_policy")
    matrix_spec = manifest.get("matrix")
    if not all(isinstance(value, dict) for value in (scenario_spec, seed_spec, matrix_spec)):
        raise ValueError("Release manifest lacks scenario, seed_policy, or matrix contract")
    matrix_path = (manifest_path.parent / scenario_spec["matrix_path"]).resolve()
    if not matrix_path.is_relative_to(ROOT) or not matrix_path.is_file():
        raise ValueError("Pinned scenario matrix is missing or outside the repository")
    if _sha256(matrix_path) != scenario_spec.get("matrix_sha256"):
        raise ValueError("Pinned scenario matrix SHA-256 differs from release manifest")
    seeds = seed_spec.get("resolved_seeds")
    if seeds != config.get("expected_seeds") or seeds != list(range(111, 141)):
        raise ValueError("Oracle seeds must match the pinned release seeds 111–140")
    scenarios = load_scenario_matrix(matrix_path)
    names = [str(scenario.get("name", "")) for scenario in scenarios]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("Scenario matrix has missing or duplicate names")
    if len(names) != config.get("expected_scenarios") or len(names) != matrix_spec.get("scenarios"):
        raise ValueError("Oracle scenario count differs from the release matrix")
    if config.get("horizon") != matrix_spec.get("horizon_steps"):
        raise ValueError("Oracle horizon differs from the release matrix")
    if not isinstance(config.get("dt"), (int, float)) or float(config["dt"]) <= 0:
        raise ValueError("Oracle dt must be positive")
    if not isinstance(config.get("workers"), int) or config["workers"] < 1:
        raise ValueError("Oracle workers must be a positive integer")
    return manifest, matrix_path, scenarios, names, list(seeds)


def _validate_run_roster(  # noqa: C901 - explicit checks keep every arm reviewable.
    config: dict[str, Any], scenario_ids: list[str]
) -> list[dict[str, Any]]:
    """Validate population modes and exact coverage of configured oracle arms."""
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs or any(not isinstance(run, dict) for run in runs):
        raise ValueError("Oracle runs must be a non-empty list")
    keys = [run.get("key") for run in runs]
    if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("Oracle run keys must be unique, non-empty strings")
    aware_keys = config.get("aware_keys")
    if (
        not isinstance(aware_keys, list)
        or not aware_keys
        or any(not isinstance(key, str) or not key for key in aware_keys)
    ):
        raise ValueError("At least one pedestrian-aware comparator arm is required")
    required_keys = [
        config.get("goal_key"),
        config.get("stationary_key"),
        *aware_keys,
    ]
    if len(required_keys) != len(set(required_keys)) or set(required_keys) != set(keys):
        raise ValueError("Goal, stationary, and aware keys must cover exactly the configured runs")
    for run in runs:
        if run.get("population") not in {"original", "pedestrian_free_v1"}:
            raise ValueError(f"Invalid population mode for {run['key']}")
        if not isinstance(run.get("algo"), str) or not run["algo"]:
            raise ValueError(f"Invalid algorithm for {run['key']}")
    by_key = {run["key"]: run for run in runs}
    if (
        by_key[config["goal_key"]]["algo"] != "goal"
        or by_key[config["goal_key"]]["population"] != "pedestrian_free_v1"
    ):
        raise ValueError("Goal arm must use goal with pedestrian_free_v1")
    if (
        by_key[config["stationary_key"]]["algo"] != "stand_still"
        or by_key[config["stationary_key"]]["population"] != "original"
    ):
        raise ValueError("Stationary arm must use stand_still with original population")
    for key in aware_keys:
        if by_key[key]["population"] != "pedestrian_free_v1":
            raise ValueError(f"Aware arm {key} must use pedestrian_free_v1")
        if by_key[key]["algo"] in {"goal", "stand_still"}:
            raise ValueError(f"Aware arm {key} must use a pedestrian-aware algorithm")
    if not isinstance(config.get("probe_scenario_ids"), list) or not set(
        config["probe_scenario_ids"]
    ).issubset(scenario_ids):
        raise ValueError("Probe IDs must be an explicit subset of the pinned matrix")
    empty_ids = config.get("stationary_no_pedestrian_scenario_ids")
    if (
        not isinstance(empty_ids, list)
        or len(empty_ids) != len(set(empty_ids))
        or not set(empty_ids).issubset(scenario_ids)
    ):
        raise ValueError("Stationary empty-population IDs must be an explicit unique matrix subset")
    if not isinstance(config.get("thresholds"), dict):
        raise ValueError("Oracle thresholds must be a mapping")
    return runs


def load_contract(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the oracle contract against a pinned release manifest."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or config.get("schema_version") != "reference-planner-oracles.v1"
    ):
        raise ValueError("Expected reference-planner-oracles.v1 configuration")
    manifest_path = _repo_path(str(config.get("release_manifest", "")))
    manifest, matrix_path, scenarios, names, seeds = _release_matrix_contract(config, manifest_path)
    runs = _validate_run_roster(config, names)
    config["_resolved"] = {
        "release_id": str(manifest["release_id"]),
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": _sha256(manifest_path),
        "matrix_path": str(matrix_path.relative_to(ROOT)),
        "matrix_sha256": _sha256(matrix_path),
        "scenario_ids": names,
        "seeds": seeds,
        "expected_episode_rows": len(names) * len(seeds) * len(runs),
    }
    return config, scenarios


def _scenario_arm(
    scenarios: list[dict[str, Any]], seeds: list[int], population: str
) -> list[dict[str, Any]]:
    """Copy one arm's source scenarios and select the pinned seed schedule."""
    derived = deepcopy(scenarios)
    for scenario in derived:
        scenario["seeds"] = list(seeds)
        if population == "pedestrian_free_v1":
            scenario["reference_population_mode"] = population
            simulation = scenario.setdefault("simulation_config", {})
            if not isinstance(simulation, dict):
                raise ValueError(f"Scenario {scenario['name']} has invalid simulation_config")
            simulation["population_size"] = 0
    return derived


def _verify_population_overlays(config: dict[str, Any], scenarios: list[dict[str, Any]]) -> int:
    """Build population variants and verify declared source-matrix empty cells."""
    from robot_sf.benchmark.map_runner.map_runner_env import build_env_config

    matrix_path = ROOT / config["_resolved"]["matrix_path"]
    derived = _scenario_arm(scenarios, config["_resolved"]["seeds"], "pedestrian_free_v1")
    for scenario in derived:
        env_config = build_env_config(scenario, scenario_path=matrix_path)
        if env_config.sim_config.population_size != 0:
            raise ValueError(f"Pedestrian-free overlay is not exact for {scenario['name']}")
        if any(
            map_def.single_pedestrians or map_def.social_groups
            for map_def in env_config.map_pool.map_defs.values()
        ):
            raise ValueError(f"Pedestrian-free overlay retains fixed actors in {scenario['name']}")
    declared_empty = set(config["stationary_no_pedestrian_scenario_ids"])
    observed_empty: set[str] = set()
    for scenario in scenarios:
        source_config = build_env_config(scenario, scenario_path=matrix_path)
        sim = scenario.get("simulation_config", {})
        density = sim.get("ped_density", 0) if isinstance(sim, dict) else None
        fixed_actors = any(
            map_def.single_pedestrians or map_def.social_groups
            for map_def in source_config.map_pool.map_defs.values()
        )
        if (
            density == 0
            and source_config.sim_config.population_size in (None, 0)
            and not fixed_actors
        ):
            observed_empty.add(scenario["name"])
    if observed_empty != declared_empty:
        raise ValueError(
            "Stationary empty-population declaration differs from source matrix: "
            f"declared={sorted(declared_empty)}, observed={sorted(observed_empty)}"
        )
    return len(derived)


def _run(
    config: dict[str, Any], scenarios: list[dict[str, Any]], output_root: Path, config_path: Path
) -> None:
    """Execute configured arms, retaining resumable raw rows and source identity."""
    from robot_sf.benchmark.map_runner.map_runner import run_map_batch

    if _git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Cluster oracle run requires a clean exact-source worktree")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "oracle_run_manifest.json"
    source_sha = _git("rev-parse", "HEAD")
    identity = {
        "schema_version": "reference-planner-oracle-run.v1",
        "source_sha": source_sha,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "release": config["_resolved"],
        "runs": config["runs"],
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {key: previous.get(key) for key in identity} != identity:
            raise ValueError("Existing run manifest identity differs; use a new output root")
    else:
        manifest_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    matrix_path = ROOT / config["_resolved"]["matrix_path"]
    for run in config["runs"]:
        arm_dir = output_root / "runs" / run["key"]
        arm_dir.mkdir(parents=True, exist_ok=True)
        result = run_map_batch(
            _scenario_arm(scenarios, config["_resolved"]["seeds"], run["population"]),
            arm_dir / "episodes.jsonl",
            EPISODE_SCHEMA,
            scenario_path=matrix_path,
            provenance_scenario_path=matrix_path,
            horizon=int(config["horizon"]),
            dt=float(config["dt"]),
            record_forces=False,
            algo=run["algo"],
            algo_config_path=run.get("algo_config"),
            benchmark_profile=run.get("benchmark_profile", "baseline-safe"),
            workers=int(config["workers"]),
            resume=True,
        )
        (arm_dir / "batch_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )


def _load_rows(
    output_root: Path, keys: list[str]
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Read raw rows for each arm while preserving malformed-input failures."""
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    for key in keys:
        path = output_root / "runs" / key / "episodes.jsonl"
        rows: list[dict[str, Any]] = []
        if not path.is_file():
            errors.append(f"Missing episode artifact for {key}: {path}")
        else:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{key} line {line_number}: {exc.msg}")
                    continue
                if not isinstance(record, dict):
                    errors.append(f"{key} line {line_number}: expected object")
                    continue
                rows.append(record)
        rows_by_arm[key] = rows
    return rows_by_arm, errors


def _gate(
    config: dict[str, Any],
    output_root: Path,
    config_path: Path,
    *,
    run_error: str | None = None,
) -> dict[str, Any]:
    """Evaluate saved rows and write the machine and human oracle reports."""
    run_manifest = output_root / "oracle_run_manifest.json"
    input_errors: list[str] = []
    if run_error is not None:
        input_errors.append(run_error)
    source_sha = None
    if run_manifest.is_file():
        try:
            identity = json.loads(run_manifest.read_text(encoding="utf-8"))
            source_sha = identity.get("source_sha")
            if identity.get("config_sha256") != _sha256(config_path):
                input_errors.append("Run manifest config SHA-256 differs from current config")
            if identity.get("release") != config["_resolved"]:
                input_errors.append("Run manifest release identity differs from current release")
        except (OSError, ValueError, TypeError) as exc:
            input_errors.append(f"Run manifest is unreadable: {exc}")
    else:
        input_errors.append("Run manifest is missing")
    keys = [run["key"] for run in config["runs"]]
    rows_by_arm, row_errors = _load_rows(output_root, keys)
    input_errors.extend(row_errors)
    report = evaluate_oracles(
        release_id=config["_resolved"]["release_id"],
        scenario_ids=config["_resolved"]["scenario_ids"],
        seeds=config["_resolved"]["seeds"],
        rows_by_arm=rows_by_arm,
        goal_key=config["goal_key"],
        stationary_key=config["stationary_key"],
        aware_keys=config["aware_keys"],
        expected_algorithms_by_arm={run["key"]: run["algo"] for run in config["runs"]},
        probe_scenario_ids=config["probe_scenario_ids"],
        stationary_no_pedestrian_scenario_ids=config["stationary_no_pedestrian_scenario_ids"],
        thresholds=config["thresholds"],
        source_sha=source_sha,
    )
    report["input_errors"] = input_errors
    report["config"] = {
        "path": str(config_path.relative_to(ROOT)),
        "sha256": _sha256(config_path),
        "release_manifest_sha256": config["_resolved"]["manifest_sha256"],
        "scenario_matrix_sha256": config["_resolved"]["matrix_sha256"],
    }
    if input_errors:
        report["gate"]["status"] = "fail"
        report["gate"].setdefault("reasons", []).extend(input_errors)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "oracle_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "oracle_report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """Execute preflight, cluster run, or saved-artifact release gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("preflight", "run", "release-gate"), required=True)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    if not config_path.is_relative_to(ROOT):
        parser.error("--config must be inside the repository")
    config, scenarios = load_contract(config_path)
    if args.mode == "preflight":
        verified = _verify_population_overlays(config, scenarios)
        print(
            json.dumps(
                {**config["_resolved"], "population_overlays_checked": verified},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    run_error = None
    if args.mode == "run":
        try:
            _verify_population_overlays(config, scenarios)
            _run(config, scenarios, args.output_root.resolve(), config_path)
        except Exception as exc:  # noqa: BLE001 - preserve a failed-run report and exit nonzero.
            run_error = f"Run aborted: {type(exc).__name__}: {exc}"
    report = _gate(config, args.output_root.resolve(), config_path, run_error=run_error)
    print(
        json.dumps(
            {"gate": report["gate"], "report": str(args.output_root / "oracle_report.json")},
            sort_keys=True,
        )
    )
    return 0 if report["gate"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
