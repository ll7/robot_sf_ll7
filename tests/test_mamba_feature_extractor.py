"""Tests for the issue #4014 Mamba / state-space feature extractor primitive."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from robot_sf.feature_extractors.config import FeatureExtractorPresets, FeatureExtractorType
from robot_sf.feature_extractors.mamba_extractor import (
    MAMBA_TEMPORAL_HISTORY_KEY,
    MambaFeatureExtractor,
    MambaFeatureExtractorConfig,
)
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS


@pytest.fixture
def observation_space() -> spaces.Dict:
    """Observation space with rays, drive state, and planned temporal-history slot."""
    return spaces.Dict(
        {
            OBS_DRIVE_STATE: spaces.Box(low=-np.inf, high=np.inf, shape=(5, 5), dtype=np.float32),
            OBS_RAYS: spaces.Box(low=0, high=10, shape=(5, 64), dtype=np.float32),
            MAMBA_TEMPORAL_HISTORY_KEY: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4, 7),
                dtype=np.float32,
            ),
        }
    )


@pytest.fixture
def sample_observation() -> dict[str, th.Tensor]:
    """Sample batched tensors matching ``observation_space``."""
    return {
        OBS_DRIVE_STATE: th.randn(2, 5, 5),
        OBS_RAYS: th.rand(2, 5, 64),
        MAMBA_TEMPORAL_HISTORY_KEY: th.randn(2, 4, 7),
    }


def test_mamba_config_defaults_are_cpu_safe() -> None:
    """Default dataclass values describe the conservative issue #4014 lane."""
    config = MambaFeatureExtractorConfig()

    assert config.backend == "auto"
    assert config.sequence_source == "rays"
    assert config.fail_if_exact_backend_missing is False


def test_mamba_lite_preset_routes_to_extractor() -> None:
    """The shared feature-extractor config registry exposes the Mamba primitive."""
    config = FeatureExtractorPresets.mamba_lite()

    assert config.extractor_type == FeatureExtractorType.MAMBA
    assert config.params["backend"] == "torch_ssm_lite"
    assert config.get_extractor_class() is MambaFeatureExtractor


def test_torch_ssm_lite_forward_returns_finite_features(
    observation_space: spaces.Dict,
    sample_observation: dict[str, th.Tensor],
) -> None:
    """The CPU-safe backend should initialize and return finite SB3 features."""
    extractor = MambaFeatureExtractor(
        observation_space,
        backend="torch_ssm_lite",
        d_model=12,
        d_state=4,
        d_conv=3,
        expand=2,
        drive_hidden_dims=(10, 6),
    )

    features = extractor(sample_observation)

    assert extractor.backend_name == "torch_ssm_lite"
    assert extractor.backend_exact is False
    assert extractor.features_dim == 18
    assert features.shape == (2, extractor.features_dim)
    assert th.isfinite(features).all()


def test_forward_accepts_non_contiguous_ray_tensors(
    observation_space: spaces.Dict,
    sample_observation: dict[str, th.Tensor],
) -> None:
    """Ray sequence preparation should not require contiguous input tensors."""
    extractor = MambaFeatureExtractor(observation_space, backend="torch_ssm_lite", d_model=8)
    non_contiguous_obs = {
        **sample_observation,
        OBS_RAYS: th.rand(2, 64, 5).transpose(1, 2),
    }

    assert not non_contiguous_obs[OBS_RAYS].is_contiguous()
    features = extractor(non_contiguous_obs)

    assert features.shape == (2, extractor.features_dim)
    assert th.isfinite(features).all()


def test_temporal_history_sequence_source(
    observation_space: spaces.Dict,
    sample_observation: dict[str, th.Tensor],
) -> None:
    """The primitive exposes the later temporal-history contract without adding a wrapper."""
    extractor = MambaFeatureExtractor(
        observation_space,
        backend="torch_ssm_lite",
        sequence_source="temporal_history",
        d_model=10,
        drive_hidden_dims=(),
    )

    features = extractor(sample_observation)

    assert extractor.sequence_key == MAMBA_TEMPORAL_HISTORY_KEY
    assert extractor.features_dim == 35
    assert features.shape == (2, extractor.features_dim)


def test_invalid_backend_fails_closed(observation_space: spaces.Dict) -> None:
    """Unknown backends should fail before training starts."""
    with pytest.raises(ValueError, match="backend must be one of"):
        MambaFeatureExtractor(observation_space, backend="not_a_backend")  # type: ignore[arg-type]


def test_exact_backend_missing_can_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    observation_space: spaces.Dict,
) -> None:
    """Callers can require exact mamba_ssm instead of silently using the lite backend."""
    monkeypatch.setattr(
        "robot_sf.feature_extractors.mamba_extractor.importlib.util.find_spec",
        lambda name: None if name == "mamba_ssm" else importlib.util.find_spec(name),
    )

    with pytest.raises(ImportError, match="mamba_ssm backend requested but unavailable"):
        MambaFeatureExtractor(
            observation_space,
            backend="mamba_ssm",
            fail_if_exact_backend_missing=True,
        )


def test_auto_backend_records_lite_fallback_when_exact_missing(
    monkeypatch: pytest.MonkeyPatch,
    observation_space: spaces.Dict,
) -> None:
    """Auto mode must make fallback status visible for later comparison reports."""
    monkeypatch.setattr(
        "robot_sf.feature_extractors.mamba_extractor.importlib.util.find_spec",
        lambda name: None if name == "mamba_ssm" else importlib.util.find_spec(name),
    )

    extractor = MambaFeatureExtractor(observation_space, backend="auto")

    assert extractor.backend_name == "torch_ssm_lite"
    assert extractor.backend_exact is False
