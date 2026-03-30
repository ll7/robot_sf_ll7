"""Run the stage-2 issue-739 PPO ablation matrix and summarize the results."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIGS = (
    Path("configs/training/ppo/ablations/expert_ppo_issue_739_stage2_opt_scale.yaml"),
    Path("configs/training/ppo/ablations/expert_ppo_issue_739_stage2_sampling.yaml"),
)


@dataclass(slots=True)
class AblationRunSummary:
    """Minimal result row for one ablation config."""

    policy_id: str
    config_path: str
    expert_manifest_path: str
    training_run_manifest_path: str
    eval_per_scenario_path: str | None
    success_rate: float | None
    collision_rate: float | None
    max_steps_rate: float | None
    snqi: float | None
    eval_episode_return: float | None
    notes: list[str]


def _metric_mean(metrics: dict[str, Any], key: str) -> float | None:
    aggregate = metrics.get(key)
    if aggregate is None:
        return None
    return float(getattr(aggregate, "mean", aggregate))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        help="Optional ablation config path. Repeat to override the default matrix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/validation/issue_739_stage2_ablations/latest"),
        help="Directory for the summary JSON and Markdown outputs.",
    )
    return parser.parse_args(argv)


def _write_markdown(path: Path, rows: list[AblationRunSummary]) -> None:
    lines = [
        "# Issue 739 Stage-2 Ablations",
        "",
        "| policy_id | success | collision | max_steps | snqi | eval_return |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {policy_id} | {success} | {collision} | {max_steps} | {snqi} | {ret} |".format(
                policy_id=row.policy_id,
                success="n/a" if row.success_rate is None else f"{row.success_rate:.4f}",
                collision="n/a" if row.collision_rate is None else f"{row.collision_rate:.4f}",
                max_steps="n/a" if row.max_steps_rate is None else f"{row.max_steps_rate:.4f}",
                snqi="n/a" if row.snqi is None else f"{row.snqi:.4f}",
                ret="n/a" if row.eval_episode_return is None else f"{row.eval_episode_return:.4f}",
            )
        )
    lines.extend(["", "## Artifacts", ""])
    for row in rows:
        lines.append(f"- `{row.policy_id}`")
        lines.append(f"  - config: `{row.config_path}`")
        lines.append(f"  - expert manifest: `{row.expert_manifest_path}`")
        lines.append(f"  - training run manifest: `{row.training_run_manifest_path}`")
        if row.eval_per_scenario_path is not None:
            lines.append(f"  - eval by scenario: `{row.eval_per_scenario_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the configured stage-2 ablations and write summary artifacts."""
    os.environ.setdefault("LOGURU_LEVEL", "WARNING")
    from scripts.training.train_ppo import load_expert_training_config, run_expert_training

    args = _parse_args(argv)
    config_paths = tuple(Path(path) for path in args.configs) if args.configs else DEFAULT_CONFIGS
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[AblationRunSummary] = []
    for config_path in config_paths:
        resolved = config_path.resolve()
        config = load_expert_training_config(resolved)
        result = run_expert_training(config, config_path=resolved, dry_run=False)
        rows.append(
            AblationRunSummary(
                policy_id=config.policy_id,
                config_path=str(resolved),
                expert_manifest_path=str(result.expert_manifest_path),
                training_run_manifest_path=str(result.training_run_manifest_path),
                eval_per_scenario_path=(
                    str(result.training_run_artifact.eval_per_scenario_path)
                    if result.training_run_artifact.eval_per_scenario_path is not None
                    else None
                ),
                success_rate=_metric_mean(result.metrics, "success_rate"),
                collision_rate=_metric_mean(result.metrics, "collision_rate"),
                max_steps_rate=_metric_mean(result.metrics, "max_steps_rate"),
                snqi=_metric_mean(result.metrics, "snqi"),
                eval_episode_return=_metric_mean(result.metrics, "eval_episode_return"),
                notes=list(result.training_run_artifact.notes),
            )
        )

    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(
        json.dumps({"rows": [asdict(row) for row in rows]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(summary_md, rows)
    print(summary_json)
    print(summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
