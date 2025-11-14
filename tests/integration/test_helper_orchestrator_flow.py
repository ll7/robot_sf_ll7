"""Integration test for helper orchestrator flow (T006)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from robot_sf.benchmark.helper_catalog import (
    load_trained_policy,
    prepare_classic_env,
    run_episodes_with_recording,
)
from robot_sf.benchmark.helper_registry import ExampleOrchestrator
from robot_sf.common.artifact_paths import resolve_artifact_path


def test_quickstart_orchestrator_flow():
    """Test the quickstart flow: orchestrator imports helpers and runs episode."""
    # Create example orchestrator metadata
    orchestrator = ExampleOrchestrator(
        path="examples/test_orchestrator.py",
        owner="test_user",
        requires_recording=True,
        notes="Integration test orchestrator",
    )

    # Mock all the external dependencies
    with patch("robot_sf.benchmark.helper_catalog.make_robot_env") as mock_make_env:
        with patch("stable_baselines3.PPO") as mock_ppo:
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "mkdir"):
                    # Setup mocks
                    mock_env = MagicMock()
                    mock_make_env.return_value = mock_env
                    mock_env.reset.return_value = ([1, 2, 3], {})
                    mock_env.step.return_value = ([1, 2, 3], 1.0, True, False, {})

                    mock_model = MagicMock()
                    mock_ppo.load.return_value = mock_model
                    mock_model.predict.return_value = ([0.5], None)

                    # Test the quickstart flow as described in quickstart.md

                    # 1. Prepare environment
                    env, seeds = prepare_classic_env()
                    assert env is mock_env
                    assert isinstance(seeds, list)

                    # 2. Load policy
                    policy = load_trained_policy("model/test.zip")
                    assert policy is mock_model

                    # 3. Run episodes with recording
                    results = run_episodes_with_recording(
                        env=env,
                        policy=policy,
                        seeds=seeds[:2],  # Use first 2 seeds
                        record=orchestrator.requires_recording,
                        output_dir=resolve_artifact_path("tmp/vis_runs"),
                    )

                    # 4. Verify results structure
                    assert len(results) == 2
                    for result in results:
                        assert "seed" in result
                        assert "episode_reward" in result
                        assert "steps" in result
                        assert "recording_enabled" in result
                        assert result["recording_enabled"] == orchestrator.requires_recording

                    # Ensure all helper functions were called
                    mock_make_env.assert_called_once()
                    mock_ppo.load.assert_called_once()
                    assert mock_model.predict.call_count == sum(r["steps"] for r in results)
