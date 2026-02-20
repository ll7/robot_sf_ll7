"""Helpers for loading canonical predictive planner algorithm configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path("configs/algos/prediction_planner_camera_ready.yaml")


def load_predictive_planner_algo_config(
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Load predictive planner algorithm config from YAML.

    Returns:
        Config mapping copied from the YAML file.
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Predictive planner config must be a mapping: {path}")
    return dict(payload)


def build_predictive_planner_algo_config(
    *,
    checkpoint_path: str | Path | None = None,
    device: str | None = "cpu",
    overrides: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Build canonical predictive planner config with runtime overrides.

    Returns:
        Predictive planner algorithm config mapping.
    """
    config = load_predictive_planner_algo_config(config_path)
    if checkpoint_path is not None:
        config["predictive_checkpoint_path"] = str(checkpoint_path)
        config.pop("predictive_model_id", None)
    if device is not None:
        config["predictive_device"] = str(device)
    if overrides:
        config.update(dict(overrides))
    return config
