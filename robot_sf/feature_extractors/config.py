"""
Configuration system for feature extractor selection.

This module provides a standardized way to configure and create different
feature extractors while maintaining backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FeatureExtractorType(Enum):
    """Available feature extractor types."""

    DYNAMICS = "dynamics"  # Original DynamicsExtractor
    MLP = "mlp"  # Simple MLP-based extractor
    ATTENTION = "attention"  # Attention-based extractor
    LIGHTWEIGHT_CNN = "lightweight_cnn"  # Lightweight CNN extractor
    LSTM = "lstm"  # LSTM sequential extractor (spatial, not temporal with standard PPO)
    MAMBA = "mamba"  # Mamba/SSM sequence extractor (bounded observation sequence)


def _get_extractor_class(extractor_type: FeatureExtractorType) -> type[BaseFeaturesExtractor]:
    """Lazy import and return the feature extractor class for the given type.

    Returns:
        type[BaseFeaturesExtractor]: Feature extractor class.
    """
    if extractor_type == FeatureExtractorType.DYNAMICS:
        from robot_sf.feature_extractor import DynamicsExtractor  # noqa: PLC0415

        return DynamicsExtractor
    if extractor_type == FeatureExtractorType.MLP:
        from robot_sf.feature_extractors.mlp_extractor import MLPFeatureExtractor  # noqa: PLC0415

        return MLPFeatureExtractor
    if extractor_type == FeatureExtractorType.ATTENTION:
        from robot_sf.feature_extractors.attention_extractor import (  # noqa: PLC0415
            AttentionFeatureExtractor,
        )

        return AttentionFeatureExtractor
    if extractor_type == FeatureExtractorType.LIGHTWEIGHT_CNN:
        from robot_sf.feature_extractors.lightweight_cnn_extractor import (  # noqa: PLC0415
            LightweightCNNExtractor,
        )

        return LightweightCNNExtractor
    if extractor_type == FeatureExtractorType.LSTM:
        from robot_sf.feature_extractors.lstm_extractor import (  # noqa: PLC0415
            LSTMFeatureExtractor,
        )

        return LSTMFeatureExtractor
    if extractor_type == FeatureExtractorType.MAMBA:
        from robot_sf.feature_extractors.mamba_extractor import (  # noqa: PLC0415
            MambaFeatureExtractor,
        )

        return MambaFeatureExtractor
    raise ValueError(f"Unknown feature extractor type: {extractor_type}")


@dataclass
class FeatureExtractorConfig:
    """
    Configuration for feature extractors.

    This class provides a unified interface for configuring different
    feature extractor types with their specific parameters.

    Attributes:
        extractor_type: Type of feature extractor to use
        params: Type-specific parameters for the extractor
    """

    extractor_type: FeatureExtractorType = FeatureExtractorType.DYNAMICS
    params: dict[str, Any] = field(default_factory=dict)

    def get_extractor_class(self) -> type[BaseFeaturesExtractor]:
        """Get the feature extractor class for this configuration.

        Returns:
            type[BaseFeaturesExtractor]: Class object for the configured extractor.
        """
        return _get_extractor_class(self.extractor_type)

    def get_policy_kwargs(self) -> dict[str, Any]:
        """Get policy kwargs suitable for StableBaselines3.

        Returns:
            dict[str, Any]: Policy kwargs with extractor class and parameters.
        """
        return {
            "features_extractor_class": self.get_extractor_class(),
            "features_extractor_kwargs": self.params.copy(),
        }


# Predefined configurations for common use cases
class FeatureExtractorPresets:
    """Predefined feature extractor configurations."""

    @staticmethod
    def dynamics_original() -> FeatureExtractorConfig:
        """Original DynamicsExtractor with default parameters.

        Returns:
            FeatureExtractorConfig: Preset configuration instance.
        """
        return FeatureExtractorConfig(extractor_type=FeatureExtractorType.DYNAMICS, params={})

    @staticmethod
    def dynamics_no_conv() -> FeatureExtractorConfig:
        """Original DynamicsExtractor without convolution (flatten only).

        Returns:
            FeatureExtractorConfig: Preset configuration instance.
        """
        return FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.DYNAMICS, params={"use_ray_conv": False}
        )

    @staticmethod
    def mlp_small() -> FeatureExtractorConfig:
        """Small MLP extractor for fast training.

        Returns:
            FeatureExtractorConfig: Preset configuration instance.
        """
        return FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.MLP,
            params={"ray_hidden_dims": [64, 32], "drive_hidden_dims": [16, 8], "dropout_rate": 0.1},
        )

    @staticmethod
    def mlp_large() -> FeatureExtractorConfig:
        """Large MLP extractor for better performance.

        Returns:
            FeatureExtractorConfig: Preset configuration instance.
        """
        return FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.MLP,
            params={
                "ray_hidden_dims": [256, 128, 64],
                "drive_hidden_dims": [64, 32, 16],
                "dropout_rate": 0.15,
            },
        )

    @staticmethod
    def attention_small() -> FeatureExtractorConfig:
        """Small attention extractor.

        Returns:
            FeatureExtractorConfig: Preset configuration instance.
        """
        return FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.ATTENTION,
            params={"embed_dim": 32, "num_heads": 2, "num_layers": 1, "dropout_rate": 0.1},
        )

    @staticmethod
    def attention_large() -> FeatureExtractorConfig:
        """Large attention extractor.

        Returns:
            FeatureExtractorConfig: Preset configuration instance.
        """
        return FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.ATTENTION,
            params={"embed_dim": 128, "num_heads": 8, "num_layers": 3, "dropout_rate": 0.1},
        )

    @staticmethod
    def lightweight_cnn() -> FeatureExtractorConfig:
        """Lightweight CNN extractor.

        Returns:
            FeatureExtractorConfig: Preset configuration instance.
        """
        return FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.LIGHTWEIGHT_CNN,
            params={"num_filters": [32, 16], "kernel_sizes": [5, 3], "dropout_rate": 0.1},
        )

    @staticmethod
    def lstm_small() -> FeatureExtractorConfig:
        """Small LSTM extractor — fast, treats rays as a 1-D sequence.

        Returns:
            FeatureExtractorConfig: Preset configuration instance.
        """
        return FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.LSTM,
            params={"hidden_size": 64, "num_layers": 1, "drive_hidden_dims": [32, 16]},
        )

    @staticmethod
    def lstm_medium() -> FeatureExtractorConfig:
        """Medium LSTM extractor — deeper sequence encoding with bidirectional scan.

        Returns:
            FeatureExtractorConfig: Preset configuration instance.
        """
        return FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.LSTM,
            params={
                "hidden_size": 128,
                "num_layers": 2,
                "lstm_dropout": 0.1,
                "drive_hidden_dims": [64, 32],
                "bidirectional": True,
            },
        )

    @staticmethod
    def mamba_lite() -> FeatureExtractorConfig:
        """CPU-safe Mamba/SSM-lite extractor for issue #4014 smoke checks.

        Returns:
            FeatureExtractorConfig: Preset configuration instance.
        """
        return FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.MAMBA,
            params={
                "backend": "torch_ssm_lite",
                "d_model": 64,
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "num_layers": 1,
                "dropout_rate": 0.0,
                "sequence_source": "rays",
                "drive_hidden_dims": (32, 16),
            },
        )


def create_feature_extractor_config(
    extractor_type: Union[str, FeatureExtractorType], **params
) -> FeatureExtractorConfig:
    """
    Create a feature extractor configuration.

    Args:
        extractor_type: Type of extractor (string or enum)
        **params: Additional parameters for the extractor

    Returns:
        FeatureExtractorConfig: Created configuration instance.

    Example:
        config = create_feature_extractor_config("mlp", ray_hidden_dims=[128, 64])
    """
    if isinstance(extractor_type, str):
        extractor_type = FeatureExtractorType(extractor_type)

    return FeatureExtractorConfig(extractor_type=extractor_type, params=params)
