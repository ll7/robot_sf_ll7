#!/usr/bin/env python3
"""Train the issue #4013 diagnostic learned short-horizon pedestrian predictor.

Produces a real (but diagnostic-only) trained checkpoint plus a training manifest
and metrics so the learned-prediction MPC smoke lane can load learned weights
instead of the zero-initialized untrained smoke model.

This is CPU-only and does not run a benchmark campaign, submit Slurm/GPU jobs, or
make any paper/dissertation claim. The synthetic training task is a reproducible
learnability probe, not real pedestrian data.

Example:
    uv run python scripts/training/train_learned_short_horizon_predictor_issue_4013.py \
        --config configs/training/learned_short_horizon_predictor_issue_4013_smoke.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from robot_sf.planner.learned_short_horizon_trainer import (
    ShortHorizonTrainerConfig,
    train_short_horizon_predictor,
)

DEFAULT_CONFIG = Path("configs/training/learned_short_horizon_predictor_issue_4013_smoke.yaml")


def _load_config(config_path: Path) -> ShortHorizonTrainerConfig:
    """Build a trainer config from YAML, ignoring unknown keys.

    Returns:
        ShortHorizonTrainerConfig: Parsed trainer configuration.
    """

    raw: dict[str, Any] = {}
    if config_path is not None:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw = loaded
    allowed = {f.name for f in fields(ShortHorizonTrainerConfig)}
    kwargs = {key: value for key, value in raw.items() if key in allowed}
    return ShortHorizonTrainerConfig(**kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train the predictor and print a compact JSON summary.

    Returns:
        int: ``0`` when training reduced the loss, ``1`` otherwise.
    """

    args = parse_args(argv)
    config = _load_config(args.config)
    if args.output_dir:
        config = ShortHorizonTrainerConfig(**{**config.__dict__, "output_dir": args.output_dir})
    result = train_short_horizon_predictor(config)
    summary = {
        "checkpoint_path": str(result.checkpoint_path),
        "manifest_path": str(result.manifest_path),
        "metrics_path": str(result.metrics_path),
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "loss_reduction": result.loss_reduction,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.loss_reduction > 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
