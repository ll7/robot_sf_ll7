"""Torch-backed validation used only for enabled learned predictive campaign bindings.

This adapter is outside the slim compat lane's deferred import closure. Importing
campaign preflight does not execute it; validating a .pt checkpoint requires training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.planner.obstacle_features import validate_predictive_feature_schema_metadata

if TYPE_CHECKING:
    from pathlib import Path


def _required_positive_int(metadata: dict[str, Any], key: str, *, section: str) -> int:
    """Reject absent, coerced, or non-positive checkpoint dimensions.

    Returns:
        The explicitly serialized positive integer.
    """
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"Predictive checkpoint {section}.{key} must be a positive integer")
    return value


def checkpoint_forecast_steps(path: Path, *, expected_feature_schema_name: str | None) -> int:
    """Return the runtime model horizon only after metadata and state loading succeed.

    Returns:
        The horizon of a model whose serialized output head was accepted by the loader.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Predictive checkpoint not found: {path}")
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Predictive checkpoint validation requires torch; install the training extra"
        ) from exc

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Predictive checkpoint payload must be a mapping")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("Predictive checkpoint config metadata must be a mapping")
    input_dim = _required_positive_int(config, "input_dim", section="config")
    _required_positive_int(config, "horizon_steps", section="config")
    feature_schema = payload.get("feature_schema")
    if not isinstance(feature_schema, dict):
        raise ValueError("Predictive checkpoint feature_schema metadata must be a mapping")
    schema_name = feature_schema.get("name")
    if not isinstance(schema_name, str) or not schema_name.strip():
        raise ValueError("Predictive checkpoint feature_schema.name must be a non-empty string")
    _required_positive_int(feature_schema, "input_dim", section="feature_schema")
    validate_predictive_feature_schema_metadata(
        feature_schema, input_dim=input_dim, expected_schema_name=expected_feature_schema_name
    )
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Predictive checkpoint state_dict must be a mapping")

    # Check metadata first: the runtime loader retains legacy default inference.
    # Then use its state loading, including output-head shape and missing-key checks.
    from robot_sf.planner.predictive_model import load_predictive_checkpoint  # noqa: PLC0415

    model, _payload = load_predictive_checkpoint(
        path, map_location="cpu", expected_feature_schema_name=expected_feature_schema_name
    )
    return int(model.config.horizon_steps)
