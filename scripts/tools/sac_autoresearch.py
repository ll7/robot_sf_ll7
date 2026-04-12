#!/usr/bin/env python3
"""Run SAC training experiments and benchmark evaluations in a repeatable loop."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

import yaml

from scripts.training.train_sac_sb3 import load_sac_training_config

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "output" / "ai" / "autoresearch" / "sac"
RESULTS_PATH = RESULTS_DIR / "results.tsv"
VERIFIED_SIMPLE_SUBSET = ROOT / "configs" / "scenarios" / "sets" / "verified_simple_subset_v1.yaml"


def _resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if base is None:
        base = ROOT
    return (base / candidate).resolve()


def _run_process(command: list[str], *, allow_failure: bool = False) -> int:
    completed = subprocess.run(command, check=not allow_failure)
    return int(completed.returncode)


def _load_experiment_list(path: Path) -> list[dict[str, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Experiment file must contain a YAML list of experiment entries.")
    return data


def _format_result_row(experiment: dict[str, str], summary: dict[str, object]) -> list[str]:
    return [
        experiment.get("name", "unnamed"),
        experiment.get("config", ""),
        experiment.get("description", ""),
        f"{summary.get('success_rate', 0.0):.4f}",
        f"{summary.get('mean_min_distance', 0.0):.4f}",
        f"{summary.get('mean_avg_speed', 0.0):.4f}",
        str(summary.get("gate_pass", False)),
        str(summary.get("total_episodes", summary.get("episodes", 0))),
        str(summary.get("duration_s", 0)),
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    ]


def _append_results(headers: list[str], rows: list[list[str]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if write_header:
            writer.writerow(headers)
        writer.writerows(rows)


def _parse_summary(summary_path: Path) -> dict[str, object]:
    if not summary_path.exists():
        return {
            "success_rate": 0.0,
            "mean_min_distance": 0.0,
            "mean_avg_speed": 0.0,
            "gate_pass": False,
            "total_episodes": 0,
        }
    content = summary_path.read_text(encoding="utf-8")
    try:
        return yaml.safe_load(content) if content.strip() else {}
    except yaml.YAMLError:
        return {}


def main(argv: list[str] | None = None) -> int:
    """Run the configured SAC experiment list and append benchmark summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        type=str,
        default="configs/training/sac/sac_autoresearch_experiments.yaml",
        help="Path to a YAML file listing SAC experiments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate experiment definitions without running training.",
    )
    args = parser.parse_args(argv)

    experiments_path = _resolve_path(args.experiments)
    experiments = _load_experiment_list(experiments_path)
    headers = [
        "name",
        "config",
        "description",
        "success_rate",
        "mean_min_distance",
        "mean_avg_speed",
        "gate_pass",
        "episodes",
        "duration_s",
        "timestamp",
    ]
    rows: list[list[str]] = []

    for experiment in experiments:
        config_path = _resolve_path(experiment["config"], base=experiments_path.parent)
        assert config_path.exists(), f"Config missing: {config_path}"
        config = load_sac_training_config(config_path)
        output_dir = config.output_dir
        policy_id = config.policy_id
        checkpoint = output_dir / f"{policy_id}.zip"
        eval_settings = config.evaluation

        if args.dry_run:
            print(f"Would run experiment: {experiment.get('name')} ({config_path})")
            continue

        start = time.time()
        print(f"Running training: {experiment.get('name')} ({config_path})")
        _run_process(
            [
                "uv",
                "run",
                "python",
                "scripts/training/train_sac_sb3.py",
                "--config",
                str(config_path),
            ]
        )

        if not checkpoint.exists():
            raise RuntimeError(f"Expected checkpoint not found after training: {checkpoint}")

        eval_output_dir = RESULTS_DIR / policy_id
        summary_path = eval_output_dir / "sac_eval_summary.json"
        if summary_path.exists():
            summary_path.unlink()
        scenario_matrix = eval_settings.scenario_matrix or VERIFIED_SIMPLE_SUBSET
        eval_cmd = [
            "uv",
            "run",
            "python",
            "scripts/validation/evaluate_sac.py",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(eval_output_dir),
            "--workers",
            str(eval_settings.workers),
            "--scenario-matrix",
            str(scenario_matrix),
            "--horizon",
            str(eval_settings.horizon),
            "--dt",
            str(eval_settings.dt),
            "--min-success-rate",
            str(eval_settings.min_success_rate),
        ]
        algo_config = eval_settings.algo_config
        if algo_config not in (None, ""):
            eval_cmd.extend(["--algo-config", str(algo_config)])
        device = eval_settings.device
        if device not in (None, ""):
            eval_cmd.extend(["--device", str(device)])
        print(f"Running evaluation for: {policy_id}")
        _run_process(eval_cmd, allow_failure=True)
        duration = time.time() - start

        if not summary_path.exists():
            print(f"WARNING: eval summary not written for {policy_id}; skipping result row.")
            continue
        summary = _parse_summary(summary_path)
        summary["duration_s"] = f"{duration:.2f}"
        rows.append(_format_result_row(experiment, summary))

    if rows:
        _append_results(headers, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
