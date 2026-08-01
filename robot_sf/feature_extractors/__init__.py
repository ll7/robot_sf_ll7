"""
Alternative feature extractors for robot environments.

This module provides various feature extraction architectures that can be used
as alternatives to the original ``DynamicsExtractor`` while maintaining
compatibility with Stable-Baselines3 and the sensor fusion system. The legacy
``DynamicsExtractor`` entrypoint is intentionally preserved in
``robot_sf.feature_extractor`` for backward compatibility.

All extractors implement the same interface and work with the same observation spaces.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "AttentionFeatureExtractor": "robot_sf.feature_extractors.attention_extractor",
    "GridSocNavExtractor": "robot_sf.feature_extractors.grid_socnav_extractor",
    "LightweightCNNExtractor": "robot_sf.feature_extractors.lightweight_cnn_extractor",
    "LSTMFeatureExtractor": "robot_sf.feature_extractors.lstm_extractor",
    "MambaFeatureExtractor": "robot_sf.feature_extractors.mamba_extractor",
    "MambaFeatureExtractorConfig": "robot_sf.feature_extractors.mamba_extractor",
    "MLPFeatureExtractor": "robot_sf.feature_extractors.mlp_extractor",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve extractor exports only when an extractor is requested.

    Returns:
        The requested extractor class or configuration type.
    """
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List lazy package exports without importing their optional dependencies.

    Returns:
        All package globals and deferred export names.
    """
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
