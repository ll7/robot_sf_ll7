"""
Alternative feature extractors for robot environments.

This module provides various feature extraction architectures that can be used
as alternatives to the original ``DynamicsExtractor`` while maintaining
compatibility with Stable-Baselines3 and the sensor fusion system. The legacy
``DynamicsExtractor`` entrypoint is intentionally preserved in
``robot_sf.feature_extractor`` for backward compatibility.

All extractors implement the same interface and work with the same observation spaces.
"""

from .attention_extractor import AttentionFeatureExtractor
from .grid_socnav_extractor import GridSocNavExtractor
from .lightweight_cnn_extractor import LightweightCNNExtractor
from .lstm_extractor import LSTMFeatureExtractor
from .mamba_extractor import MambaFeatureExtractor, MambaFeatureExtractorConfig
from .mlp_extractor import MLPFeatureExtractor

__all__ = [
    "AttentionFeatureExtractor",
    "GridSocNavExtractor",
    "LSTMFeatureExtractor",
    "LightweightCNNExtractor",
    "MLPFeatureExtractor",
    "MambaFeatureExtractor",
    "MambaFeatureExtractorConfig",
]
