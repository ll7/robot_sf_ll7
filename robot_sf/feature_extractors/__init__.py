"""
Alternative feature extractors for robot environments.

This module provides various feature extraction architectures that can be used
as alternatives to the original DynamicsExtractor while maintaining compatibility
with StableBaselines3 and the sensor fusion system.

All extractors implement the same interface and work with the same observation spaces.
"""

from .attention_extractor import AttentionFeatureExtractor
from .lightweight_cnn_extractor import LightweightCNNExtractor
from .mlp_extractor import MLPFeatureExtractor

__all__ = [
    "MLPFeatureExtractor",
    "AttentionFeatureExtractor",
    "LightweightCNNExtractor",
]
