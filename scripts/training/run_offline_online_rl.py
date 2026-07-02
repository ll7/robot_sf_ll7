"""Run a diagnostic offline-to-online SAC smoke comparison."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from scripts.training.train_sac_sb3 import load_sac_training_config, run_sac_training

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class OfflineOnlineRunSummary:
    """Diagnostic-only summary for one offline-online smoke comparison."""

    schema_version: str
    evidence_tier: str
    eligible_for_claim: bool
    seed: int | None
    offline_online_checkpoint: str
    scratch_checkpoint: str
    offline_online_config: str
    scratch_config: str
    caveats: tuple[str, ...]


def run_offline_online_experiment(config_path: Path | str) -> OfflineOnlineRunSummary:
    """Run configured offline-online and scratch SAC arms."""

    path = Path(config_path).resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"offline-online experiment config must be a mapping; got {type(raw)!r}")

    unknown = set(raw) - {
        "schema_version",
        "offline_online_arm",
        "scratch_arm",
        "output_dir",
        "summary_json",
        "summary_markdown",
    }
    if unknown:
        raise ValueError(f"Unknown offline-online experiment config keys: {sorted(unknown)}")

    offline_config_path = _required_config_path(raw, "offline_online_arm", base_dir=path.parent)
    scratch_config_path = _required_config_path(raw, "scratch_arm", base_dir=path.parent)
    output_dir = _resolve_path(
        raw.get("output_dir", "output/issue_4012_offline_online_rl_smoke"), path.parent
    )
    summary_json = _resolve_path(
        raw.get("summary_json", "issue_4012_offline_online_summary.json"), output_dir
    )
    summary_markdown = _resolve_path(
        raw.get("summary_markdown", "issue_4012_offline_online_report.md"),
        output_dir,
    )

    offline_config = load_sac_training_config(offline_config_path)
    scratch_config = load_sac_training_config(scratch_config_path)
    if offline_config.seed != scratch_config.seed:
        raise ValueError("offline_online and scratch arms must use the same seed")
    if offline_config.total_timesteps != scratch_config.total_timesteps:
        raise ValueError("offline_online and scratch arms must use the same total_timesteps")
    if not offline_config.offline_online.enabled:
        raise ValueError("offline_online_arm must enable offline_online")
    if scratch_config.offline_online.enabled:
        raise ValueError("scratch_arm must keep offline_online disabled")

    offline_checkpoint = run_sac_training(offline_config)
    scratch_checkpoint = run_sac_training(scratch_config)

    summary = OfflineOnlineRunSummary(
        schema_version="offline_online_rl_comparison.v1",
        evidence_tier="diagnostic-smoke-only",
        eligible_for_claim=False,
        seed=offline_config.seed,
        offline_online_checkpoint=str(offline_checkpoint),
        scratch_checkpoint=str(scratch_checkpoint),
        offline_online_config=str(offline_config_path),
        scratch_config=str(scratch_config_path),
        caveats=(
            "No full benchmark campaign run.",
            "No Slurm or GPU submission.",
            "No paper-facing robustness or performance claim.",
            "Worktree-local datasets and checkpoints are not durable evidence.",
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n")
    summary_markdown.write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _required_config_path(raw: dict[str, Any], key: str, *, base_dir: Path) -> Path:
    block = raw.get(key)
    if not isinstance(block, dict):
        raise ValueError(f"{key} must be a mapping")
    config_path = block.get("config")
    if config_path in (None, ""):
        raise ValueError(f"{key}.config is required")
    return _resolve_path(config_path, base_dir)


def _resolve_path(value: object, base_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _summary_markdown(summary: OfflineOnlineRunSummary) -> str:
    return (
        "# Issue #4012 Offline-to-Online RL Smoke\n\n"
        "Claim boundary: diagnostic implementation lane only; not benchmark evidence and not "
        "paper-facing.\n\n"
        f"- Evidence tier: {summary.evidence_tier}\n"
        f"- Eligible for claim: {summary.eligible_for_claim}\n"
        f"- Offline-online checkpoint: `{summary.offline_online_checkpoint}`\n"
        f"- Scratch checkpoint: `{summary.scratch_checkpoint}`\n"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Offline-online experiment YAML config.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = build_arg_parser().parse_args(argv)
    run_offline_online_experiment(args.config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
