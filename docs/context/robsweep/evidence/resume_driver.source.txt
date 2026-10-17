"""Continue only missing MPPI dev episodes through canonical identity-aware resume.

The original outer pool must exit before continuation takes ownership. Completed
raw rows are preserved exactly before compression; interrupted in-flight episodes
without a completed row are eligible again. No benchmark runtime is changed.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import multiprocessing
import os
import subprocess
import tempfile
import time
from pathlib import Path

from loguru import logger

from robot_sf.benchmark.map_runner.map_runner import run_map_batch
from robot_sf.training.scenario_loader import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/benchmarks/robsweep"
EVIDENCE = ROOT / "docs/context/robsweep/evidence"
BASE = "33a4735900153e742828227cc2171436ea09cd90"


def inputs(arm: str) -> tuple[list[dict], dict, Path]:
    """Retain the exact acquisition inputs, including authored budgets."""
    study = json.loads((CONFIG / "study.json").read_text())
    planner = next(p for p in study["roster"] if p["key"] == "predictive_mppi")
    scenarios = [
        copy.deepcopy(dict(s)) for s in load_scenarios(CONFIG / "legacy.yaml", base_dir=ROOT)
    ]
    for scenario in scenarios:
        scenario["seeds"] = study["seeds"]
        scenario["map_file"] = str((ROOT / scenario["map_file"]).resolve())
        if arm == "swap":
            scenario["robot_config"]["max_linear_speed"] = 2.0
    return scenarios, planner, CONFIG / f"predictive_mppi_{arm}.yaml"


def canonical(scenarios: list[dict], planner: dict, config: Path, out: Path, workers: int) -> dict:
    """Use the existing engine with its native deduplication and worker isolation."""
    return run_map_batch(
        scenarios,
        out,
        ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json",
        scenario_path=CONFIG / "legacy.yaml",
        dt=0.1,
        record_forces=True,
        algo=planner["algo"],
        algo_config_path=str(config),
        benchmark_profile=planner.get("benchmark_profile", "experimental"),
        adapter_impact_eval=planner.get("adapter_impact_eval", False),
        workers=workers,
        multiprocessing_context=multiprocessing.get_context("spawn"),
        resume=True,
    )


def prefix_rows(prefix: bytes) -> list[dict]:
    """Reject corrupt or duplicate completed records before resuming."""
    rows = [json.loads(line) for line in prefix.splitlines()]
    keys = {(r["scenario_id"], r["seed"]) for r in rows}
    if len(keys) != len(rows) or any(not 1061 <= seed <= 1090 for _, seed in keys):
        raise ValueError("Prefix contains duplicate or out-of-scope identities")
    return rows


def verify_resume(arm: str, workers: int) -> None:
    """Prove the actual completed MPPI identities are skipped without execution."""
    source = EVIDENCE / "episodes" / f"{arm}_predictive_mppi.jsonl"
    prefix = source.read_bytes().rsplit(b"\n", 1)[0] + b"\n"
    rows = prefix_rows(prefix)
    completed: dict[str, list[int]] = {}
    for row in rows:
        completed.setdefault(row["scenario_id"], []).append(row["seed"])
    scenarios, planner, config = inputs(arm)
    selected = []
    for scenario in scenarios:
        if scenario["name"] in completed:
            scenario["seeds"] = completed[scenario["name"]]
            selected.append(scenario)
    with tempfile.TemporaryDirectory(prefix="examiner-resume-proof-") as folder:
        out = Path(folder) / "prefix.jsonl"
        out.write_bytes(prefix)
        result = canonical(selected, planner, config, out, workers)
        if result["written"] != 0 or result["total_jobs"] != 0 or out.read_bytes() != prefix:
            raise ValueError("Canonical resume did not skip every completed identity exactly")
    print(json.dumps({"arm": arm, "completed_identities_skipped": len(rows), "written": 0}))


def verify_parallel_canary(arm: str) -> None:
    """Compare real five-step MPPI outcomes and physics across dispatch paths."""
    scenarios, planner, config = inputs(arm)
    scenario = scenarios[0]
    scenario["seeds"] = [1061]
    scenario["simulation_config"]["max_episode_steps"] = 5
    folder = EVIDENCE / "resume/canary"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    paths = []
    for workers in (1, 2):
        path = folder / f"{arm}_w{workers}.jsonl"
        if path.exists() or path.with_suffix(".jsonl.gz").exists():
            raise ValueError("Existing canary cannot be repeated without diagnosis")
        result = canonical([scenario], planner, config, path, workers)
        if result["written"] != 1 or result.get("failed_jobs", 0):
            raise ValueError("Native MPPI dispatch canary failed")
        record = json.loads(path.read_text())
        records.append(record)
        compressed = path.with_suffix(".jsonl.gz")
        with gzip.open(compressed, "wt") as handle:
            handle.write(path.read_text().replace(str(ROOT) + "/", ""))
        path.unlink()
        paths.append(
            {
                "artifact_path": str(compressed.relative_to(ROOT)),
                "sha256": hashlib.sha256(compressed.read_bytes()).hexdigest(),
            }
        )
    fields = ("outcome", "steps", "effective_physics", "metric_values")
    if any(records[0][field] != records[1][field] for field in fields):
        raise ValueError("Serial and native process dispatch diverge on canary scientific fields")
    proof = {
        "arm": arm,
        "seed": 1061,
        "steps": 5,
        "workers": [1, 2],
        "equal_fields": list(fields),
        "artifacts": paths,
        "claim_boundary": "Dispatch binding canary only; not a full-trajectory determinism proof or ranking episode.",
    }
    (folder / f"{arm}_proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"arm": arm, "serial_parallel_canary": "equal"}))


def continue_arm(arm: str, workers: int, old_controller_pid: int) -> None:
    """Append missing identities after exclusive ownership and source checks."""
    try:
        os.kill(old_controller_pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("Original controller must exit before continuation takes ownership")
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            BASE,
            "--",
            "robot_sf",
            "fast-pysf",
            "configs/algos",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    identity = json.loads((EVIDENCE / "acquisition_source.json").read_text())
    for item in identity["source_files"]:
        path = item.get("path") or str(Path(*item["acquisition_path_parts"]))
        if path.startswith("configs/benchmarks/robsweep/") or path in ("pyproject.toml", "uv.lock"):
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Acquisition input changed: {path}")
    stem = f"{arm}_predictive_mppi"
    out = EVIDENCE / "episodes" / f"{stem}.jsonl"
    receipt = EVIDENCE / "episodes" / f"{stem}_execution.json"
    if receipt.exists():
        raise ValueError("Completed arm cannot be resubmitted")
    prefix = out.read_bytes()
    rows = prefix_rows(prefix)
    snapshots = EVIDENCE / "resume"
    snapshots.mkdir(exist_ok=True)
    snapshot = snapshots / f"{stem}_prefix.jsonl.gz"
    if snapshot.exists():
        raise ValueError("Continuation already attempted; diagnose before another attempt")
    with gzip.open(snapshot, "wt") as handle:
        handle.write(prefix.decode().replace(str(ROOT) + "/", ""))
    scenarios, planner, config = inputs(arm)
    started = time.monotonic()
    result = canonical(scenarios, planner, config, out, workers)
    final = out.read_bytes()
    final_rows = prefix_rows(final)
    if not final.startswith(prefix) or len(final_rows) != 300:
        raise ValueError(
            "Continuation did not preserve its exact prefix and complete 300 identities"
        )
    if result.get("failed_jobs", 0) or result["written"] + len(rows) != 300:
        raise ValueError("Continuation has failed or unreconciled identities")
    payload = {
        "treatment": arm,
        "planner": "predictive_mppi",
        "requested": 300,
        "elapsed_s": time.monotonic() - started,
        "completed_prefix_rows": len(rows),
        "total_retained_records": len(final_rows),
        "prefix_bytes_preserved": True,
        "original_controller_exit_verified": True,
        "prefix_artifact": {
            "artifact_path": str(snapshot.relative_to(ROOT)),
            "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        },
        "result": result,
    }
    receipt.write_text(json.dumps(payload, indent=2) + "\n")
    with gzip.open(out.with_suffix(".jsonl.gz"), "wt") as handle:
        handle.write(final.decode().replace(str(ROOT) + "/", ""))
    out.unlink()
    print(json.dumps({"arm": arm, "preserved": len(rows), "new": result["written"], "total": 300}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("legacy", "swap"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check-only", action="store_true")
    checks.add_argument("--parallel-canary", action="store_true")
    parser.add_argument("--old-controller-pid", type=int)
    args = parser.parse_args()
    logger.remove()
    logger.add(EVIDENCE / f"resume_{args.arm}.log", level="ERROR")
    if args.parallel_canary:
        verify_parallel_canary(args.arm)
    elif args.check_only:
        verify_resume(args.arm, args.workers)
    elif args.old_controller_pid is None:
        parser.error("--old-controller-pid is required for exclusive continuation")
    else:
        continue_arm(args.arm, args.workers, args.old_controller_pid)
