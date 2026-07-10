#!/usr/bin/env python3
"""Run the config-first issue #5088 braking-authority targeted smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

from robot_sf.benchmark.braking_authority_sensitivity import (
    analyze_smoke_results,
    git_head,
    load_selected_scenario,
    load_smoke_config,
    materialize_arm_scenario,
    write_report,
)
from robot_sf.benchmark.map_runner import run_map_batch

EPISODE_SCHEMA_PATH = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Tracked smoke config YAML.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh output directory for disposable episodes and compact report.",
    )
    return parser


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Execute each controlled arm and return the compact sensitivity report."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    repo_root = Path(__file__).resolve().parents[2]
    config = load_smoke_config(config_path)
    source_scenario = load_selected_scenario(config, repo_root=repo_root)
    planner = config["planner"]
    run_config = config["run"]
    arm_records: dict[str, list[dict[str, Any]]] = {}
    arm_summaries: dict[str, dict[str, Any]] = {}
    raw_artifacts: dict[str, dict[str, Any]] = {}
    for arm in config["arms"]:
        key = str(arm["key"])
        scenario = materialize_arm_scenario(source_scenario, arm=arm, seeds=config["seeds"])
        arm_dir = output_dir / "arms" / key
        episodes_path = arm_dir / "episodes.jsonl"
        summary = run_map_batch(
            [scenario],
            episodes_path,
            EPISODE_SCHEMA_PATH,
            scenario_path=repo_root / str(config["scenario"]["path"]),
            horizon=int(run_config["horizon"]),
            dt=float(run_config["dt"]),
            record_forces=bool(run_config["record_forces"]),
            algo=str(planner["algo"]),
            benchmark_profile=str(planner["benchmark_profile"]),
            record_simulation_step_trace=bool(run_config["record_simulation_step_trace"]),
            workers=int(run_config["workers"]),
            resume=False,
        )
        arm_summaries[key] = summary
        (arm_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        arm_records[key] = [
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        raw_artifacts[key] = {
            "relative_path": episodes_path.relative_to(output_dir).as_posix(),
            "sha256": hashlib.sha256(episodes_path.read_bytes()).hexdigest(),
            "row_count": len(arm_records[key]),
        }
    reproduction_command = "DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy " + shlex.join(
        [
            "uv",
            "run",
            "python",
            "scripts/tools/run_braking_authority_sensitivity_smoke.py",
            "--config",
            config_path.as_posix(),
            "--output",
            "<fresh-artifact-dir>",
        ]
    )
    report = analyze_smoke_results(
        config,
        arm_records=arm_records,
        arm_summaries=arm_summaries,
        config_path=config_path.as_posix(),
        run_commit=git_head(repo_root),
        raw_artifact_root="local-scratch-not-retained",
        raw_artifacts=raw_artifacts,
        reproduction_command=reproduction_command,
    )
    write_report(report, output_dir)
    return report


def main() -> int:
    """Run the smoke and print one compact machine-readable result."""
    args = _parser().parse_args()
    report = run(args.config, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "signal_activated": report["comparison"]["signal_activated"],
                "activated_metrics": report["comparison"]["activated_metrics"],
                "report": (args.output / "report.json").as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
