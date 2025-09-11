"""
Configuration system for feature extractor selection.

This module provides a standardized way to configure and create different
feature extractors while maintaining backward compatibility.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Type, Union

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from robot_sf.feature_extractor import DynamicsExtractor
from robot_sf.feature_extractors.attention_extractor import AttentionFeatureExtractor
from robot_sf.feature_extractors.lightweight_cnn_extractor import LightweightCNNExtractor
from robot_sf.feature_extractors.mlp_extractor import MLPFeatureExtractor


class FeatureExtractorType(Enum):
    """Available feature extractor types."""
    
    DYNAMICS = "dynamics"  # Original DynamicsExtractor
    MLP = "mlp"  # Simple MLP-based extractor
    ATTENTION = "attention"  # Attention-based extractor
    LIGHTWEIGHT_CNN = "lightweight_cnn"  # Lightweight CNN extractor


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
    params: Dict[str, Any] = field(default_factory=dict)
    
    def get_extractor_class(self) -> Type[BaseFeaturesExtractor]:
        """Get the feature extractor class for this configuration."""
        return _EXTRACTOR_REGISTRY[self.extractor_type]
    
    def get_policy_kwargs(self) -> Dict[str, Any]:
        """Get policy kwargs suitable for StableBaselines3."""
        return {
            "features_extractor_class": self.get_extractor_class(),
            "features_extractor_kwargs": self.params.copy()
        }


# Registry mapping extractor types to their classes
_EXTRACTOR_REGISTRY: Dict[FeatureExtractorType, Type[BaseFeaturesExtractor]] = {
    FeatureExtractorType.DYNAMICS: DynamicsExtractor,
    FeatureExtractorType.MLP: MLPFeatureExtractor,
    FeatureExtractorType.ATTENTION: AttentionFeatureExtractor,
    FeatureExtractorType.LIGHTWEIGHT_CNN: LightweightCNNExtractor,
}


# Predefined configurations for common use cases
class FeatureExtractorPresets:
    """Predefined feature extractor configurations."""
    
    @staticmethod
    def dynamics_original() -> FeatureExtractorConfig:
        """Original DynamicsExtractor with default parameters."""
        return FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.DYNAMICS,
            params={}
        )
    
    @staticmethod
    def dynamics_no_conv() -> FeatureExtractorConfig:
        """Original DynamicsExtractor without convolution (flatten only)."""
        return FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.DYNAMICS,
            params={"use_ray_conv": False}
        )
    
    @staticmethod
    def mlp_small() -> FeatureExtractorConfig:
        """Small MLP extractor for fast training."""
        return FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.MLP,
            params={
                "ray_hidden_dims": [64, 32],
                "drive_hidden_dims": [16, 8],
                "dropout_rate": 0.1
            }
        )
    
    @staticmethod
    def mlp_large() -> FeatureExtractorConfig:
        """Large MLP extractor for better performance."""
        return FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.MLP,
            params={
                "ray_hidden_dims": [256, 128, 64],
                "drive_hidden_dims": [64, 32, 16], 
                "dropout_rate": 0.15
            }
        )
    
    @staticmethod
    def attention_small() -> FeatureExtractorConfig:
        """Small attention extractor."""
        return FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.ATTENTION,
            params={
                "embed_dim": 32,
                "num_heads": 2,
                "num_layers": 1,
                "dropout_rate": 0.1
            }
        )
    
    @staticmethod
    def attention_large() -> FeatureExtractorConfig:
        """Large attention extractor."""
        return FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.ATTENTION,
            params={
                "embed_dim": 128,
                "num_heads": 8,
                "num_layers": 3,
                "dropout_rate": 0.1
            }
        )
    
    @staticmethod
    def lightweight_cnn() -> FeatureExtractorConfig:
        """Lightweight CNN extractor."""
        return FeatureExtractorConfig(
            extractor_type=FeatureExtractorType.LIGHTWEIGHT_CNN,
            params={
                "num_filters": [32, 16],
                "kernel_sizes": [5, 3],
                "dropout_rate": 0.1
            }
        )


def create_feature_extractor_config(
    extractor_type: Union[str, FeatureExtractorType],
    **params
) -> FeatureExtractorConfig:
    """
    Create a feature extractor configuration.
    
    Args:
        extractor_type: Type of extractor (string or enum)
        **params: Additional parameters for the extractor
        
    Returns:
        FeatureExtractorConfig instance
        
    Example:
        config = create_feature_extractor_config("mlp", ray_hidden_dims=[128, 64])
    """
    if isinstance(extractor_type, str):
        extractor_type = FeatureExtractorType(extractor_type)
    
    return FeatureExtractorConfig(
        extractor_type=extractor_type,
        params=params
    )