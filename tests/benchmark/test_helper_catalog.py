"""Tests for the benchmark helper catalog (T003)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from robot_sf.benchmark.helper_catalog import (
    load_trained_policy,
    prepare_classic_env,
    run_episodes_with_recording,
)


def test_prepare_classic_env():
    """Test basic environment preparation with default config."""
    with patch("robot_sf.benchmark.helper_catalog.make_robot_env") as mock_make_env:
        mock_env = MagicMock()
        mock_make_env.return_value = mock_env

        env, seeds = prepare_classic_env()

        assert env is mock_env
        assert isinstance(seeds, list)
        assert len(seeds) > 0
        assert all(isinstance(seed, int) for seed in seeds)
        mock_make_env.assert_called_once()


def test_load_trained_policy_success():
    """Test successful policy loading with caching."""
    test_model_path = Path("test_model.zip")

    with patch("stable_baselines3.PPO") as mock_ppo:
        with patch.object(Path, "exists", return_value=True):
            mock_model = MagicMock()
            mock_ppo.load.return_value = mock_model

            # First call should load from disk
            result1 = load_trained_policy(str(test_model_path))
            assert result1 is mock_model
            mock_ppo.load.assert_called_once_with(str(test_model_path))

            # Second call should use cache
            mock_ppo.load.reset_mock()
            result2 = load_trained_policy(str(test_model_path))
            assert result2 is mock_model
            mock_ppo.load.assert_not_called()


def test_load_trained_policy_not_found():
    """Test policy loading failure when file doesn't exist."""
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_trained_policy("nonexistent_model.zip")


def test_run_episodes_with_recording_success():
    """Test successful episode execution with recording."""
    mock_env = MagicMock()
    mock_policy = MagicMock()
    mock_env.reset.return_value = ([1, 2, 3], {})
    mock_env.step.return_value = ([1, 2, 3], 1.0, True, False, {})
    mock_policy.predict.return_value = ([0.5], None)

    output_dir = Path("/tmp/test_output")
    seeds = [42, 123]

    with patch.object(Path, "mkdir"):
        results = run_episodes_with_recording(
            env=mock_env, policy=mock_policy, seeds=seeds, record=True, output_dir=output_dir
        )

    assert len(results) == 2
    for i, result in enumerate(results):
        assert result["seed"] == seeds[i]
        assert "episode_reward" in result
        assert "steps" in result
        assert result["recording_enabled"] is True


def test_run_episodes_with_recording_error():
    """Test episode execution error handling."""
    mock_env = MagicMock()
    mock_policy = MagicMock()
    mock_env.reset.side_effect = Exception("Environment error")

    with pytest.raises(RuntimeError, match="Failed to run episodes"):
        run_episodes_with_recording(
            env=mock_env, policy=mock_policy, seeds=[42], record=False, output_dir=Path("/tmp/test")
        )
