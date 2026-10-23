"""Add opt-in family weights to the canonical PPO trainer without shared-file edits."""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

from robot_sf.training.scenario_family_weights import with_family_weights
from scripts.training import train_ppo


def main() -> None:
    """Compile the config option and delegate training and manifests to their owner."""
    args = train_ppo.build_arg_parser().parse_args()
    if args.log_file:
        raise ValueError("Use shell log redirection for this additive entry point")
    config_path = Path(args.config)
    config = with_family_weights(train_ppo.load_expert_training_config(config_path))
    logger.info("Resolved training scenario weights: {}", config.scenario_sampling.get("weights"))
    train_ppo.run_expert_training(
        config,
        config_path=config_path,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        dry_run=args.dry_run,
        resume_from=Path(args.resume_from) if args.resume_from else None,
        training_seed=args.training_seed,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
