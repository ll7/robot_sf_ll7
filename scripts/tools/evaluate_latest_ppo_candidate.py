#!/usr/bin/env python3
"""Resolve the latest PPO checkpoint from W&B and run promotion-grade evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from robot_sf.common.logging import configure_logging
from robot_sf.models import resolve_latest_wandb_model, upsert_registry_entry

POLICY_SUCCESS_THRESHOLD = 0.80
POLICY_COLLISION_THRESHOLD = 0.10
BENCHMARK_SUCCESS_THRESHOLD = 0.80
BENCHMARK_COLLISION_THRESHOLD = 0.10


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-entity", default="ll7")
    parser.add_argument("--wandb-project", default="robot_sf")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-job-type")
    parser.add_argument("--wandb-name-prefix")
    parser.add_argument(
        "--wandb-tags",
        default="",
        help="Comma-separated tags that must be present on the selected run.",
    )
    parser.add_argument("--wandb-file", default="model.zip")
    parser.add_argument(
        "--wandb-allowed-states",
        default="finished,running",
        help="Comma-separated W&B states allowed for latest model selection.",
    )
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--seed-set", choices=["dev", "eval"], default="eval")
    parser.add_argument("--max-seeds", type=int, default=3)
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("output/model_cache/latest_wandb"),
        help="Directory used for downloaded W&B checkpoints.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/benchmarks/latest_ppo_candidate_eval"),
        help="Root directory for analysis outputs.",
    )
    parser.add_argument("--benchmark-workers", type=int, default=1)
    parser.add_argument("--benchmark-horizon", type=int, default=120)
    parser.add_argument("--benchmark-dt", type=float, default=0.1)
    parser.add_argument("--registry-path", type=Path, default=Path("model/registry.yaml"))
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Update the model registry when both promotion gates pass.",
    )
    parser.add_argument(
        "--registry-model-id",
        help="Override model_id used when promoting into the registry.",
    )
    parser.add_argument(
        "--registry-display-name",
        help="Override display_name used when promoting into the registry.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _promotion_summary(summary_payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(summary_payload.get("summary") or {})
    aggregates = dict(summary_payload.get("aggregates") or {})
    problem_episodes = list(summary_payload.get("problem_episodes") or [])
    termination_counts = dict(summary.get("termination_reason_counts") or {})
    metric_means = dict(summary.get("metric_means") or {})

    weakest = sorted(
        (
            {
                "scenario_id": scenario_id,
                "success_rate": float((metrics.get("success_rate") or {}).get("mean", 0.0)),
                "collision_rate": float((metrics.get("collision_rate") or {}).get("mean", 0.0)),
            }
            for scenario_id, metrics in aggregates.items()
            if str(scenario_id) != "_meta"
        ),
        key=lambda row: (row["success_rate"], -row["collision_rate"], row["scenario_id"]),
    )[:5]

    success_rate = float(summary.get("success_rate", 0.0))
    collision_rate = float(summary.get("collision_rate", 0.0))
    return {
        "episodes": int(summary.get("episodes", 0)),
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "ped_collision_count": int(termination_counts.get("collision", 0)),
        "termination_reason_counts": termination_counts,
        "metric_means": metric_means,
        "problem_episode_count": len(problem_episodes),
        "weakest_scenarios": weakest,
        "gate_pass": (
            success_rate >= POLICY_SUCCESS_THRESHOLD
            and collision_rate <= POLICY_COLLISION_THRESHOLD
        ),
    }


def _build_benchmark_algo_config(
    *,
    training_config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> Path:
    """Build a temporary PPO algo-config matching the training observation contract."""
    training_payload = yaml.safe_load(training_config_path.read_text(encoding="utf-8")) or {}
    env_overrides = dict(training_payload.get("env_overrides") or {})
    config_payload: dict[str, Any] = {
        "model_path": str(checkpoint_path),
        "device": "auto",
        "deterministic": True,
        "obs_mode": "dict",
        "action_space": "unicycle",
        "fallback_to_goal": True,
    }
    for key, value in env_overrides.items():
        if str(key).startswith("predictive_foresight_"):
            config_payload[str(key)] = value
    output_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    return output_path


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Expected benchmark JSONL missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _record_metric(record: dict[str, Any], key: str) -> float:
    return float((record.get("metrics") or {}).get(key, 0.0) or 0.0)


def _benchmark_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "termination_reason_counts": {},
            "problem_episode_count": 0,
            "weakest_scenarios": [],
            "contradictions": [],
            "gate_pass": False,
        }

    termination_counts: dict[str, int] = {}
    scenario_rows: dict[str, dict[str, float]] = {}
    contradictions: list[dict[str, Any]] = []
    success_sum = 0.0
    collision_sum = 0.0

    for record in records:
        metrics = dict(record.get("metrics") or {})
        scenario_id = str(record.get("scenario_id") or "unknown")
        success = float(metrics.get("success_rate", metrics.get("success", 0.0)) or 0.0)
        collision = float(metrics.get("collision_rate", metrics.get("collisions", 0.0)) or 0.0)
        success_sum += success
        collision_sum += collision
        termination = str(record.get("termination_reason") or "unknown")
        termination_counts[termination] = termination_counts.get(termination, 0) + 1
        row = scenario_rows.setdefault(
            scenario_id,
            {"episodes": 0.0, "success_sum": 0.0, "collision_sum": 0.0},
        )
        row["episodes"] += 1.0
        row["success_sum"] += success
        row["collision_sum"] += collision
        if collision > 0.0 and success > 0.0:
            contradictions.append(
                {
                    "scenario_id": scenario_id,
                    "seed": record.get("seed"),
                    "termination_reason": termination,
                    "reason": "collision_and_success",
                }
            )
        if termination == "collision" and success > 0.0:
            contradictions.append(
                {
                    "scenario_id": scenario_id,
                    "seed": record.get("seed"),
                    "termination_reason": termination,
                    "reason": "collision_termination_with_success",
                }
            )

    weakest = sorted(
        (
            {
                "scenario_id": scenario_id,
                "success_rate": row["success_sum"] / row["episodes"],
                "collision_rate": row["collision_sum"] / row["episodes"],
            }
            for scenario_id, row in scenario_rows.items()
        ),
        key=lambda row: (row["success_rate"], -row["collision_rate"], row["scenario_id"]),
    )[:5]

    episodes = len(records)
    success_rate = success_sum / episodes
    collision_rate = collision_sum / episodes
    return {
        "episodes": episodes,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "termination_reason_counts": termination_counts,
        "problem_episode_count": len(contradictions),
        "weakest_scenarios": weakest,
        "contradictions": contradictions,
        "gate_pass": (
            success_rate >= BENCHMARK_SUCCESS_THRESHOLD
            and collision_rate <= BENCHMARK_COLLISION_THRESHOLD
            and not contradictions
        ),
    }


def _registry_entry_from_candidate(
    *,
    model_id: str,
    display_name: str,
    selection: dict[str, Any],
    checkpoint_path: Path,
    training_config: Path,
    decision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "display_name": display_name,
        "local_path": str(checkpoint_path),
        "config_path": str(training_config),
        "commit": None,
        "wandb_run_id": selection["run_id"],
        "wandb_run_path": selection["run_path"],
        "wandb_entity": selection["run_path"].split("/")[0],
        "wandb_project": selection["run_path"].split("/")[1],
        "wandb_file": Path(checkpoint_path).name,
        "tags": ["ppo", "candidate", "promotion-workflow"],
        "notes": [
            "Promoted via scripts/tools/evaluate_latest_ppo_candidate.py.",
            f"Policy analysis success={decision['policy_gate']['success_rate']:.4f} "
            f"collision={decision['policy_gate']['collision_rate']:.4f}.",
            f"Benchmark success={decision['benchmark_gate']['success_rate']:.4f} "
            f"collision={decision['benchmark_gate']['collision_rate']:.4f}.",
        ],
    }


def _write_promotion_report(
    *,
    output_root: Path,
    selection: dict[str, Any],
    policy_result: dict[str, Any],
    benchmark_result: dict[str, Any],
    decision: dict[str, Any],
) -> tuple[Path, Path]:
    json_path = output_root / "promotion_report.json"
    md_path = output_root / "promotion_report.md"
    payload = {
        "selection": selection,
        "policy_result": policy_result,
        "benchmark_result": benchmark_result,
        "decision": decision,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# PPO Promotion Report",
        "",
        "## Selected Run",
        f"- run_id: `{selection['run_id']}`",
        f"- run_name: `{selection['run_name']}`",
        f"- run_path: `{selection['run_path']}`",
        f"- state: `{selection['state']}`",
        f"- created_at: `{selection['created_at']}`",
        f"- downloaded_model: `{selection['downloaded_model']}`",
        "",
        "## Policy Analysis Gate",
        f"- summary_json: `{policy_result['summary_json']}`",
        f"- report_md: `{policy_result['report_md']}`",
        f"- success_rate: `{decision['policy_gate']['success_rate']:.4f}`",
        f"- collision_rate: `{decision['policy_gate']['collision_rate']:.4f}`",
        f"- gate_pass: `{decision['policy_gate']['gate_pass']}`",
        "",
        "## Benchmark Gate",
        f"- episodes_jsonl: `{benchmark_result['episodes_jsonl']}`",
        f"- summary_json: `{benchmark_result['summary_json']}`",
        f"- algo_config: `{benchmark_result['algo_config']}`",
        f"- success_rate: `{decision['benchmark_gate']['success_rate']:.4f}`",
        f"- collision_rate: `{decision['benchmark_gate']['collision_rate']:.4f}`",
        f"- contradictions: `{decision['benchmark_gate']['problem_episode_count']}`",
        f"- gate_pass: `{decision['benchmark_gate']['gate_pass']}`",
        "",
        "## Promotion Decision",
        f"- promote: `{decision['promote']}`",
        f"- rationale: `{decision['rationale']}`",
        f"- registry_updated: `{decision['registry_updated']}`",
        "",
        "## Weakest Benchmark Scenarios",
    ]
    for row in decision["benchmark_gate"]["weakest_scenarios"]:
        lines.append(
            f"- `{row['scenario_id']}`: success_rate={row['success_rate']:.4f}, "
            f"collision_rate={row['collision_rate']:.4f}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _decision_exit_code(decision: dict[str, Any]) -> int:
    """Map promotion decision outcome to a machine-readable process exit code."""
    return 0 if bool(decision.get("promote")) else 2


def main() -> int:
    """Resolve the newest PPO checkpoint from W&B, evaluate it, and write a promotion report."""
    args = _build_parser().parse_args()
    configure_logging(verbose=str(args.log_level).upper() == "DEBUG")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_root = output_root / "policy_analysis"
    videos_root = output_root / "videos"
    benchmark_root = output_root / "benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    allowed_states = tuple(
        state.strip().lower()
        for state in str(args.wandb_allowed_states).split(",")
        if state.strip()
    )
    tags = tuple(tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip())
    checkpoint_path, selection = resolve_latest_wandb_model(
        entity=str(args.wandb_entity),
        project=str(args.wandb_project),
        group=str(args.wandb_group) if args.wandb_group else None,
        job_type=str(args.wandb_job_type) if args.wandb_job_type else None,
        name_prefix=str(args.wandb_name_prefix) if args.wandb_name_prefix else None,
        tags=tags,
        file_name=str(args.wandb_file),
        allowed_states=allowed_states,
        cache_dir=args.cache_dir,
    )

    selection_payload = {
        "run_id": selection.run_id,
        "run_path": selection.run_path,
        "run_name": selection.run_name,
        "job_type": selection.job_type,
        "group": selection.group,
        "state": selection.state,
        "created_at": selection.created_at,
        "downloaded_model": str(checkpoint_path),
    }
    (output_root / "latest_model_selection.json").write_text(
        json.dumps(selection_payload, indent=2),
        encoding="utf-8",
    )

    policy_cmd = [
        sys.executable,
        "scripts/tools/policy_analysis_run.py",
        "--training-config",
        str(args.training_config),
        "--policy",
        "ppo",
        "--model-path",
        str(checkpoint_path),
        "--seed-set",
        str(args.seed_set),
        "--max-seeds",
        str(args.max_seeds),
        "--output",
        str(analysis_root),
        "--video-output",
        str(videos_root),
        "--all",
    ]
    if args.videos:
        policy_cmd.extend(["--videos", "--video-fps", str(args.video_fps)])
    logger.info("Running policy analysis via: {}", " ".join(policy_cmd))
    subprocess.run(policy_cmd, check=True)

    summary_path = analysis_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected policy analysis summary missing: {summary_path}")
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    policy_result = {
        "summary_json": str(summary_path),
        "report_md": str(analysis_root / "report.md"),
        "video_root": str(videos_root) if args.videos else None,
    }
    policy_gate = _promotion_summary(summary_payload)

    algo_config_path = _build_benchmark_algo_config(
        training_config_path=args.training_config.resolve(),
        checkpoint_path=checkpoint_path.resolve(),
        output_path=benchmark_root / "ppo_candidate_algo.yaml",
    )
    benchmark_jsonl = benchmark_root / "episodes.jsonl"
    benchmark_cmd = [
        sys.executable,
        "scripts/run_classic_interactions.py",
        "--algo",
        "ppo",
        "--algo-config",
        str(algo_config_path),
        "--output",
        str(benchmark_jsonl),
        "--workers",
        str(args.benchmark_workers),
        "--horizon",
        str(args.benchmark_horizon),
        "--dt",
        str(args.benchmark_dt),
        "--no-resume",
    ]
    logger.info("Running benchmark gate via: {}", " ".join(benchmark_cmd))
    subprocess.run(benchmark_cmd, check=True)
    benchmark_records = _load_jsonl_records(benchmark_jsonl)
    benchmark_gate = _benchmark_summary(benchmark_records)
    benchmark_summary_path = benchmark_root / "summary.json"
    benchmark_summary_path.write_text(json.dumps(benchmark_gate, indent=2), encoding="utf-8")
    benchmark_result = {
        "episodes_jsonl": str(benchmark_jsonl),
        "summary_json": str(benchmark_summary_path),
        "algo_config": str(algo_config_path),
    }

    decision = {
        "policy_gate": policy_gate,
        "benchmark_gate": benchmark_gate,
        "promote": bool(policy_gate["gate_pass"] and benchmark_gate["gate_pass"]),
        "rationale": (
            "both policy-analysis and benchmark gates passed"
            if policy_gate["gate_pass"] and benchmark_gate["gate_pass"]
            else "one or more promotion gates failed"
        ),
        "registry_updated": False,
    }

    if args.promote and decision["promote"]:
        model_id = str(args.registry_model_id or selection.run_name or selection.run_id)
        display_name = str(
            args.registry_display_name
            or f"PPO latest candidate promoted ({selection.run_name or selection.run_id})"
        )
        entry = _registry_entry_from_candidate(
            model_id=model_id,
            display_name=display_name,
            selection=selection_payload,
            checkpoint_path=checkpoint_path,
            training_config=args.training_config.resolve(),
            decision=decision,
        )
        upsert_registry_entry(entry, registry_path=args.registry_path)
        decision["registry_updated"] = True
        decision["registry_model_id"] = model_id

    report_json, report_md = _write_promotion_report(
        output_root=output_root,
        selection=selection_payload,
        policy_result=policy_result,
        benchmark_result=benchmark_result,
        decision=decision,
    )
    logger.info("Latest model selection written to {}", output_root / "latest_model_selection.json")
    logger.info("Promotion report written to {} and {}", report_md, report_json)
    if args.promote and not decision["promote"]:
        logger.warning("Promotion requested but gates failed; registry was not updated.")
    return _decision_exit_code(decision)


if __name__ == "__main__":
    raise SystemExit(main())
