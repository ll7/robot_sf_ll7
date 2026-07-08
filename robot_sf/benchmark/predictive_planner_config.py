"""Helpers for loading canonical predictive planner algorithm configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from robot_sf.planner.obstacle_features import PREDICTIVE_LEGACY_FEATURE_SCHEMA

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "configs"
    / "algos"
    / "prediction_planner_camera_ready.yaml"
)


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


def infer_predictive_checkpoint_feature_schema_name(
    checkpoint_path: str | Path,
) -> str | None:
    """Return the saved predictive feature schema name from a checkpoint when available."""
    path = Path(checkpoint_path)
    if not path.exists():
        return None

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, EOFError, RuntimeError, ValueError, KeyError):
        return None
    if not isinstance(payload, dict):
        return None

    feature_schema = payload.get("feature_schema")
    if isinstance(feature_schema, dict):
        name = str(feature_schema.get("name", "") or "").strip()
        if name:
            return name

    config = payload.get("config")
    if isinstance(config, dict):
        name = str(config.get("feature_schema_name", "") or "").strip()
        if name:
            return name
    return None


def build_predictive_planner_algo_config(
    *,
    checkpoint_path: str | Path | None = None,
    device: str | None = "cpu",
    feature_schema_name: str | None = None,
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
        feature_schema_name = (
            feature_schema_name or infer_predictive_checkpoint_feature_schema_name(checkpoint_path)
        )
    if feature_schema_name is not None:
        config["predictive_feature_schema_name"] = str(feature_schema_name)
    elif "predictive_feature_schema_name" not in config:
        config["predictive_feature_schema_name"] = PREDICTIVE_LEGACY_FEATURE_SCHEMA
    if device is not None:
        config["predictive_device"] = str(device)
    if overrides:
        config.update(dict(overrides))
    return config
