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

import yaml

from robot_sf.planner.learned_short_horizon_trainer import (
    ShortHorizonTrainerConfig,
    train_short_horizon_predictor,
)

DEFAULT_CONFIG = Path("configs/training/learned_short_horizon_predictor_issue_4013_smoke.yaml")


def _load_config(config_path: Path) -> ShortHorizonTrainerConfig:
    """Build a trainer config from YAML, ignoring unknown keys.

    Fail-closed: a missing config path, a directory instead of a file, malformed
    YAML, or a non-mapping YAML document raises instead of silently falling back
    to all-default values (which would mask a mistyped ``--config`` path).

    Returns:
        ShortHorizonTrainerConfig: Parsed trainer configuration.

    Raises:
        FileNotFoundError: if ``config_path`` does not exist or is not a file.
        ValueError: if the YAML is malformed or is not a top-level mapping.
    """

    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"trainer config not found or not a file: {config_path}")
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # malformed YAML must fail loudly, not vacuously pass
        raise ValueError(f"malformed trainer config YAML: {config_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(
            f"trainer config must be a YAML mapping, got {type(loaded).__name__}: {config_path}"
        )
    allowed = {f.name for f in fields(ShortHorizonTrainerConfig)}
    # Drop explicit ``null``/None values so the dataclass default applies instead of
    # nullifying a required field (e.g. ``output_dir: null`` -> Path(None) crash,
    # ``device: null`` -> unintended device); keys are still normalized to defaults.
    kwargs = {key: value for key, value in loaded.items() if key in allowed and value is not None}
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
