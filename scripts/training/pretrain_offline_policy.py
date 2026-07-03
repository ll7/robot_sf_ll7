"""Standalone offline SAC pretraining checkpoint manifest lane for issue #4245."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.rl_trajectory_dataset import (
    RLTrajectoryEpisode,
    compute_return_to_go,
    write_rl_trajectory_dataset,
)
from robot_sf.training.offline_online_rl import (
    load_offline_transition_batch,
    validate_batch_against_env_spaces,
)
from robot_sf.training.offline_pretraining_manifest import (
    build_offline_checkpoint_manifest,
    offline_dataset_manifest_summary,
    space_fingerprint,
    write_json,
    write_normalizer_state,
)
from scripts.training.train_sac_sb3 import (
    _DEFAULT_SAC_HYPERPARAMS,
    SAC,
    _build_env,
    _resolve_policy_name,
    _save_sac_checkpoint_with_gym_shim,
    _seed_offline_online_replay_buffer,
    _transform_offline_batch_for_sac,
    load_sac_training_config,
    load_scenarios,
)


def pretrain_offline_policy(
    *,
    config_path: Path | str,
    output_dir: Path | str,
    manifest_out: Path | str,
) -> dict[str, Any]:
    """Run offline-only SAC gradient updates and write a checkpoint manifest."""

    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    manifest_out = Path(manifest_out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_sac_training_config(config_path)
    if not config.offline_online.enabled:
        raise ValueError("pretrain_offline_policy requires offline_online.enabled=true")
    if config.offline_online.dataset_path is None or config.offline_online.manifest_path is None:
        raise ValueError("pretrain_offline_policy requires offline dataset and manifest paths")
    _materialize_issue_4245_smoke_dataset_if_missing(config)

    scenario_definitions = load_scenarios(config.scenario_config)
    vec_env = _build_env(config, scenario_definitions=scenario_definitions)
    try:
        hyperparams = {**_DEFAULT_SAC_HYPERPARAMS, **config.sac_hyperparams}
        policy_name = _resolve_policy_name(vec_env.observation_space)
        model = SAC(
            policy_name,
            vec_env,
            verbose=0,
            seed=config.seed,
            device=config.device,
            policy_kwargs={"net_arch": [64, 64]},
            **hyperparams,  # type: ignore[arg-type]
        )
        _seed_offline_online_replay_buffer(
            model,
            config=config,
            observation_space=vec_env.observation_space,
            action_space=vec_env.action_space,
            dry_run=False,
        )
        if config.offline_online.offline_gradient_steps <= 0:
            raise ValueError("offline_online.offline_gradient_steps must be > 0 for pretraining")
        model.train(
            gradient_steps=config.offline_online.offline_gradient_steps,
            batch_size=int(hyperparams["batch_size"]),
        )

        checkpoint_path = output_dir / f"{config.policy_id}_offline_pretrain_checkpoint.zip"
        normalizer_path = output_dir / f"{config.policy_id}_normalizer_state.json"
        _save_sac_checkpoint_with_gym_shim(model, checkpoint_path)
        write_normalizer_state(
            normalizer_path,
            present=False,
            reason="SAC smoke env does not wrap VecNormalize; explicit hashed absence state.",
        )

        batch = load_offline_transition_batch(
            config.offline_online.dataset_path,
            split=config.offline_online.dataset_split,
            min_transitions=config.offline_online.min_transitions,
            action_contract=config.offline_online.action_contract,
            observation_contract=config.offline_online.observation_contract,
        )
        batch = _transform_offline_batch_for_sac(batch, obs_transform=config.obs_transform)
        preflight = validate_batch_against_env_spaces(
            batch,
            observation_space=vec_env.observation_space,
            action_space=vec_env.action_space,
        )
        if not preflight.ok:
            raise ValueError(
                "offline dataset failed environment-space preflight: "
                + "; ".join(preflight.validation_errors)
            )
        manifest = build_offline_checkpoint_manifest(
            checkpoint_path=checkpoint_path,
            normalizer_path=normalizer_path,
            training_config_path=config_path,
            dataset=offline_dataset_manifest_summary(
                dataset_manifest_path=config.offline_online.manifest_path,
                dataset_path=config.offline_online.dataset_path,
                batch=batch,
            ),
            offline_training={
                "gradient_steps": config.offline_online.offline_gradient_steps,
                "batch_size": int(hyperparams["batch_size"]),
                "seed": config.seed,
                "observation_contract": config.offline_online.observation_contract,
                "action_contract": config.offline_online.action_contract,
            },
            environment_contract={
                "scenario_config": str(config.scenario_config),
                "observation_space_fingerprint": space_fingerprint(vec_env.observation_space),
                "action_space_fingerprint": space_fingerprint(vec_env.action_space),
            },
            policy_type=policy_name,
        )
        write_json(manifest_out, manifest)
        return manifest
    finally:
        vec_env.close()


def _materialize_issue_4245_smoke_dataset_if_missing(config: Any) -> None:
    """Create a tiny ignored smoke dataset for the checked-in #4245 configs."""

    offline = config.offline_online
    dataset_path = offline.dataset_path
    manifest_path = offline.manifest_path
    if dataset_path is None or manifest_path is None:
        return
    if dataset_path.exists() and manifest_path.exists():
        return
    if "issue_4245_offline_pretrain_smoke" not in dataset_path.as_posix():
        return

    scenario_definitions = load_scenarios(config.scenario_config)
    env = _build_env(config, scenario_definitions=scenario_definitions)
    try:
        obs = env.reset()
        observations: list[Any] = []
        actions: list[Any] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        for step_idx in range(max(int(offline.min_transitions), 2)):
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            next_obs, reward, done, _info = env.step(action.reshape((1, *action.shape)))
            observations.append(_jsonable_vecenv_observation(obs))
            actions.append(_jsonable(action))
            rewards.append(float(reward[0]))
            terminated.append(bool(done[0]) or step_idx == max(int(offline.min_transitions), 2) - 1)
            truncated.append(False)
            obs = next_obs
    finally:
        env.close()

    episode = RLTrajectoryEpisode(
        dataset_id="issue_4245_smoke",
        episode_id="issue_4245_smoke_train_0",
        scenario_id="issue_4245_smoke",
        seed=int(config.seed or 4245),
        source_policy_id="issue_4245_zero_action_smoke",
        split=str(offline.dataset_split),
        observations=tuple(observations),
        actions=tuple(actions),
        rewards=tuple(rewards),
        return_to_go=tuple(compute_return_to_go(rewards)),
        terminated=tuple(terminated),
        truncated=tuple(truncated),
        pedestrians=tuple({} for _ in rewards),
        robot_states=tuple({} for _ in rewards),
        provenance={"issue": 4245, "claim_boundary": "local diagnostic smoke input only"},
    )
    write_rl_trajectory_dataset([episode], dataset_path)
    manifest = {
        "schema_version": "rl_trajectory_dataset_manifest.v1",
        "dataset_path": str(dataset_path),
        "dataset_id": "issue_4245_smoke",
        "split_counts": {str(offline.dataset_split): 1},
        "episode_count": 1,
        "step_count": len(rewards),
        "claim_boundary": "local diagnostic smoke input only",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = build_arg_parser().parse_args(argv)
    pretrain_offline_policy(
        config_path=args.config,
        output_dir=args.output_dir,
        manifest_out=args.manifest_out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
