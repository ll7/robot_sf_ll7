"""Run a diagnostic offline-to-online SAC smoke comparison."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.benchmark.rl_trajectory_dataset import (
    RLTrajectoryEpisode,
    compute_return_to_go,
    write_rl_trajectory_dataset,
)
from scripts.training.train_sac_sb3 import (
    _build_env,
    load_sac_training_config,
    load_scenarios,
    run_sac_training,
)

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

    _materialize_smoke_dataset_if_missing(offline_config)

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


def _materialize_smoke_dataset_if_missing(config: Any) -> None:
    """Create the repository smoke dataset when the checked-in smoke input is absent."""

    offline = config.offline_online
    dataset_path = getattr(offline, "dataset_path", None)
    if not offline.enabled or dataset_path is None:
        return
    manifest_path = getattr(offline, "manifest_path", None)
    manifest_exists = manifest_path is None or manifest_path.exists()
    if dataset_path.exists() and manifest_exists:
        return
    if "issue_4012_offline_online_rl_smoke" not in dataset_path.as_posix():
        return

    scenario_definitions = load_scenarios(config.scenario_config)
    if isinstance(scenario_definitions, dict):
        scenario_definitions = list(scenario_definitions.values())
    env = _build_env(config, scenario_definitions=scenario_definitions)
    try:
        obs = env.reset()
        observations: list[Any] = []
        actions: list[Any] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        steps = max(int(offline.min_transitions), 1)
        for step_idx in range(steps):
            action = env.action_space.sample() * 0.0
            action_batch = action.reshape((1, *action.shape))
            next_obs, reward, done, _info = env.step(action_batch)
            observations.append(_jsonable_vecenv_observation(obs))
            actions.append(_jsonable(action))
            rewards.append(float(reward[0]))
            terminated.append(bool(done[0]) and step_idx == steps - 1)
            truncated.append(False)
            obs = next_obs
        if observations:
            observations[-1] = _jsonable_vecenv_observation(obs)
            terminated[-1] = True
    finally:
        env.close()

    episode = RLTrajectoryEpisode(
        dataset_id="issue_4012_smoke",
        episode_id="issue_4012_smoke_train_0",
        scenario_id="issue_4012_smoke",
        seed=int(config.seed or 4012),
        source_policy_id="zero_action_smoke_fixture",
        split=str(offline.dataset_split),
        observations=tuple(observations),
        actions=tuple(actions),
        rewards=tuple(rewards),
        return_to_go=tuple(compute_return_to_go(rewards)),
        terminated=tuple(terminated),
        truncated=tuple(truncated),
        pedestrians=tuple({} for _ in rewards),
        robot_states=tuple({} for _ in rewards),
        provenance={
            "schema_version": "issue_4012_smoke_fixture.v1",
            "claim_boundary": "local diagnostic smoke input only; not durable benchmark evidence",
        },
    )
    write_rl_trajectory_dataset([episode], dataset_path)
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "rl_trajectory_dataset_manifest.v1",
                    "dataset_path": str(dataset_path),
                    "dataset_id": "issue_4012_smoke",
                    "split_counts": {str(offline.dataset_split): 1},
                    "claim_boundary": "local diagnostic smoke input only",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _jsonable_vecenv_observation(observation: Any) -> Any:
    if isinstance(observation, dict):
        return {key: _jsonable(value[0]) for key, value in observation.items()}
    return _jsonable(observation[0])


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


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
