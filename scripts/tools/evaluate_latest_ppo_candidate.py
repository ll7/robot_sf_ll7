#!/usr/bin/env python3
"""Resolve the latest PPO checkpoint from W&B and run a promotion-grade policy analysis."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from robot_sf.common.logging import configure_logging
from robot_sf.models import resolve_latest_wandb_model


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

    promotion = {
        "episodes": int(summary.get("episodes", 0)),
        "success_rate": float(summary.get("success_rate", 0.0)),
        "collision_rate": float(summary.get("collision_rate", 0.0)),
        "ped_collision_count": int(termination_counts.get("collision", 0)),
        "termination_reason_counts": termination_counts,
        "metric_means": metric_means,
        "problem_episode_count": len(problem_episodes),
        "weakest_scenarios": weakest,
        "gate_pass": (
            float(summary.get("success_rate", 0.0)) >= 0.80
            and float(summary.get("collision_rate", 0.0)) <= 0.10
        ),
    }
    return promotion


def _write_promotion_report(
    *,
    output_root: Path,
    selection: dict[str, Any],
    policy_result: dict[str, Any],
    promotion: dict[str, Any],
) -> tuple[Path, Path]:
    json_path = output_root / "promotion_report.json"
    md_path = output_root / "promotion_report.md"
    payload = {
        "selection": selection,
        "policy_result": policy_result,
        "promotion": promotion,
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
        "## Policy Analysis",
        f"- summary_json: `{policy_result['summary_json']}`",
        f"- report_md: `{policy_result['report_md']}`",
        f"- videos_root: `{policy_result.get('video_root')}`",
        "",
        "## Gate",
        f"- gate_pass: `{promotion['gate_pass']}`",
        f"- success_rate: `{promotion['success_rate']:.4f}`",
        f"- collision_rate: `{promotion['collision_rate']:.4f}`",
        f"- problem_episode_count: `{promotion['problem_episode_count']}`",
        "",
        "## Weakest Scenarios",
    ]
    for row in promotion["weakest_scenarios"]:
        lines.append(
            f"- `{row['scenario_id']}`: success_rate={row['success_rate']:.4f}, "
            f"collision_rate={row['collision_rate']:.4f}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    """Resolve the newest PPO checkpoint from W&B, analyze it, and write a promotion report."""
    args = _build_parser().parse_args()
    configure_logging(verbose=str(args.log_level).upper() == "DEBUG")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_root = output_root / "policy_analysis"
    videos_root = output_root / "videos"
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

    cmd = [
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
        cmd.extend(["--videos", "--video-fps", str(args.video_fps)])
    logger.info("Running policy analysis via: {}", " ".join(cmd))
    subprocess.run(cmd, check=True)

    summary_path = analysis_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected policy analysis summary missing: {summary_path}")
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    policy_result = {
        "summary_json": str(summary_path),
        "report_md": str(analysis_root / "report.md"),
        "video_root": str(videos_root) if args.videos else None,
    }
    promotion = _promotion_summary(summary_payload)
    report_json, report_md = _write_promotion_report(
        output_root=output_root,
        selection=selection_payload,
        policy_result=policy_result,
        promotion=promotion,
    )
    logger.info("Latest model selection written to {}", output_root / "latest_model_selection.json")
    logger.info("Promotion report written to {} and {}", report_md, report_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
