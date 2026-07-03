"""Manifest-driven online SAC fine-tuning from an offline checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from robot_sf.training.offline_pretraining_manifest import (
    assert_environment_compatible,
    build_finetune_manifest,
    load_offline_checkpoint_manifest,
    space_fingerprint,
    write_json,
    write_normalizer_state,
)
from scripts.training.train_sac_sb3 import (
    SAC,
    _build_env,
    _save_sac_checkpoint_with_gym_shim,
    load_sac_training_config,
    load_scenarios,
)


def offline_to_online_finetune(
    *,
    config_path: Path | str,
    pretrained_manifest: Path | str,
    output_dir: Path | str,
    manifest_out: Path | str,
) -> dict[str, Any]:
    """Load an offline checkpoint manifest, fine-tune online, and chain provenance."""

    config_path = Path(config_path).resolve()
    pretrained_manifest = Path(pretrained_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    manifest_out = Path(manifest_out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_sac_training_config(config_path)
    parent_manifest = load_offline_checkpoint_manifest(pretrained_manifest)
    checkpoint_path = Path(str(parent_manifest["checkpoint_path"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = pretrained_manifest.parent / checkpoint_path

    scenario_definitions = load_scenarios(config.scenario_config)
    vec_env = _build_env(config, scenario_definitions=scenario_definitions)
    try:
        current_contract = {
            "scenario_config": str(config.scenario_config),
            "observation_space_fingerprint": space_fingerprint(vec_env.observation_space),
            "action_space_fingerprint": space_fingerprint(vec_env.action_space),
        }
        assert_environment_compatible(
            parent_contract=parent_manifest["environment_contract"],
            current_contract=current_contract,
        )
        model = SAC.load(str(checkpoint_path), env=vec_env, device=config.device)
        model.learn(total_timesteps=int(config.total_timesteps), log_interval=100)

        output_checkpoint = output_dir / f"{config.policy_id}_offline_to_online_finetune.zip"
        output_normalizer = output_dir / f"{config.policy_id}_normalizer_state.json"
        _save_sac_checkpoint_with_gym_shim(model, output_checkpoint)
        write_normalizer_state(
            output_normalizer,
            present=False,
            reason="Fine-tune env does not wrap VecNormalize; explicit hashed absence state.",
        )
        manifest = build_finetune_manifest(
            parent_manifest_path=pretrained_manifest,
            parent_manifest=parent_manifest,
            checkpoint_path=output_checkpoint,
            normalizer_path=output_normalizer,
            training_config_path=config_path,
            online_timesteps=int(config.total_timesteps),
            seed=config.seed,
            environment_contract=current_contract,
        )
        write_json(manifest_out, manifest)
        return manifest
    finally:
        vec_env.close()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pretrained-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = build_arg_parser().parse_args(argv)
    offline_to_online_finetune(
        config_path=args.config,
        pretrained_manifest=args.pretrained_manifest,
        output_dir=args.output_dir,
        manifest_out=args.manifest_out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
