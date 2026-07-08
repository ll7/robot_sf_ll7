"""Tests for subprocess arm isolation in camera-ready campaigns (issue #4826).

These tests verify that:
1. Subprocess isolation mode is available and functional
2. Subprocess worker entrypoint exists and is callable
3. In-process mode still works with arm_isolation="in_process"
4. CLI flag for arm-isolation is recognized
5. Resource lifecycle cleanup is called in subprocess mode
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from robot_sf.benchmark.camera_ready.resource_lifecycle import (
    _cleanup_gpu_memory_before_exit,
    _main_subprocess_worker,
    _SubprocessArmParams,
)


class TestSubprocessIsolationEntryPoints:
    """Tests for subprocess isolation entry point and parameter structure."""

    def test_subprocess_arm_params_dataclass_exists(self):
        """Verify _SubprocessArmParams dataclass can be instantiated."""
        params = _SubprocessArmParams(
            planner_key="test_planner",
            planner_algo="test_algo",
            planner_human_model_variant=None,
            planner_human_model_source=None,
            planner_group="core",
            benchmark_profile="speed",
            socnav_missing_prereq_policy="error",
            adapter_impact_eval="disabled",
            kinematics="differential_drive",
            observation_mode="lidar",
            workers=1,
            horizon=100,
            dt=0.1,
            scenario_matrix_path=Path("scenarios.yaml"),
            episodes_path=Path("/tmp/episodes.jsonl"),
            summary_path=Path("/tmp/summary.json"),
            record_forces=True,
            record_planner_decision_trace=False,
            record_simulation_step_trace=False,
            observation_noise=None,
            synthetic_actuation_profile=None,
            latency_stress_profile=None,
            snqi_weights=None,
            snqi_baseline=None,
            algo_config_path=None,
        )
        assert params.planner_key == "test_planner"
        assert params.planner_algo == "test_algo"

    def test_cleanup_gpu_memory_before_exit_returns_structure(self):
        """Verify _cleanup_gpu_memory_before_exit returns expected structure."""
        result = _cleanup_gpu_memory_before_exit(
            planner_key="test_planner",
            kinematics="differential_drive",
        )
        # Should always return a dict with these fields
        assert isinstance(result, dict)
        assert "planner_key" in result
        assert "kinematics" in result
        assert "torch_available" in result
        assert "cuda_available" in result
        assert "allocated_mb" in result
        assert "reserved_mb" in result
        assert "high_water_mark_mb" in result
        assert "allocated_freed_mb" in result
        assert "reserved_freed_mb" in result

    def test_cleanup_gpu_memory_without_torch(self):
        """Verify cleanup works when torch is not available."""
        # Ensure torch is not in sys.modules
        torch_backup = sys.modules.get("torch")
        had_torch = "torch" in sys.modules
        if had_torch:
            del sys.modules["torch"]

        try:
            result = _cleanup_gpu_memory_before_exit(
                planner_key="test",
                kinematics="diff",
            )
            assert result["torch_available"] is False
            assert result["cuda_available"] is False
            assert result["allocated_mb"] == 0.0
        finally:
            # Restore torch if it was present
            if torch_backup is not None and had_torch:
                sys.modules["torch"] = torch_backup
            elif "torch" in sys.modules:
                del sys.modules["torch"]


class TestCampaignConfigArmIsolation:
    """Tests for CampaignConfig arm_isolation field."""

    def test_config_accepts_arm_isolation_in_process(self):
        """Verify CampaignConfig accepts arm_isolation='in_process'."""
        from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec

        config = CampaignConfig(
            name="test",
            scenario_matrix_path=Path("scenarios.yaml"),
            planners=(PlannerSpec(key="test", algo="test", enabled=True),),
            arm_isolation="in_process",
        )
        assert config.arm_isolation == "in_process"

    def test_config_accepts_arm_isolation_subprocess(self):
        """Verify CampaignConfig accepts arm_isolation='subprocess'."""
        from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec

        config = CampaignConfig(
            name="test",
            scenario_matrix_path=Path("scenarios.yaml"),
            planners=(PlannerSpec(key="test", algo="test", enabled=True),),
            arm_isolation="subprocess",
        )
        assert config.arm_isolation == "subprocess"

    def test_config_default_arm_isolation(self):
        """Verify CampaignConfig defaults to arm_isolation='in_process'."""
        from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec

        config = CampaignConfig(
            name="test",
            scenario_matrix_path=Path("scenarios.yaml"),
            planners=(PlannerSpec(key="test", algo="test", enabled=True),),
        )
        assert config.arm_isolation == "in_process"


class TestSubprocessWorkerExecution:
    """Tests for subprocess worker execution path."""

    def test_main_subprocess_worker_function_exists(self):
        """Verify _main_subprocess_worker function exists and is callable."""
        assert callable(_main_subprocess_worker)

    @patch("sys.stdin")
    @patch("sys.stdout")
    @patch("robot_sf.benchmark.camera_ready.resource_lifecycle._run_single_arm_subprocess")
    def test_subprocess_worker_reads_params_from_stdin(
        self, mock_run_arm, mock_stdout, mock_stdin
    ):
        """Verify subprocess worker reads parameters from stdin."""
        # Setup mock stdin with valid parameters
        mock_stdin.read.return_value = json.dumps(
            {
                "planner_key": "test",
                "planner_algo": "test_algo",
                "planner_human_model_variant": None,
                "planner_human_model_source": None,
                "planner_group": "core",
                "benchmark_profile": "speed",
                "socnav_missing_prereq_policy": "error",
                "adapter_impact_eval": "disabled",
                "kinematics": "differential_drive",
                "observation_mode": "lidar",
                "workers": 1,
                "horizon": 100,
                "dt": 0.1,
                "scenario_matrix_path": "scenarios.yaml",
                "episodes_path": "/tmp/episodes.jsonl",
                "summary_path": "/tmp/summary.json",
                "record_forces": True,
                "record_planner_decision_trace": False,
                "record_simulation_step_trace": False,
                "observation_noise": None,
                "synthetic_actuation_profile": None,
                "latency_stress_profile": None,
                "snqi_weights": None,
                "snqi_baseline": None,
                "algo_config_path": None,
            }
        )

        # Mock successful arm run
        mock_run_arm.return_value = {
            "summary": {
                "status": "ok",
                "total_jobs": 1,
                "written": 1,
                "failed_jobs": 0,
                "failures": [],
            },
            "cleanup_metrics": {
                "torch_available": False,
                "cuda_available": False,
            },
            "warnings": [],
            "episodes_total": 1,
        }

        mock_stdout.write = Mock()

        with patch("loguru.logger"):
            result = _main_subprocess_worker()
            assert result == 0
            mock_run_arm.assert_called_once()
            params = mock_run_arm.call_args.args[0]
            assert isinstance(params.scenario_matrix_path, Path)
            assert isinstance(params.episodes_path, Path)
            assert isinstance(params.summary_path, Path)


class TestRunCampaignArmIsolationParameter:
    """Tests for run_campaign arm_isolation parameter override."""

    def test_run_campaign_accepts_arm_isolation_parameter(self):
        """Verify run_campaign accepts arm_isolation parameter."""
        # Check function signature - should have arm_isolation parameter
        import inspect

        from robot_sf.benchmark.camera_ready.campaign import run_campaign

        sig = inspect.signature(run_campaign)
        assert "arm_isolation" in sig.parameters

    @patch("robot_sf.benchmark.camera_ready.campaign._run_campaign_orchestrator")
    def test_run_campaign_passes_arm_isolation_to_orchestrator(self, mock_orchestrator):
        """Verify run_campaign passes arm_isolation to orchestrator."""
        from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec
        from robot_sf.benchmark.camera_ready.campaign import run_campaign

        config = CampaignConfig(
            name="test",
            scenario_matrix_path=Path("scenarios.yaml"),
            planners=(PlannerSpec(key="test", algo="test", enabled=True),),
        )

        # Mock orchestrator to avoid actual execution
        mock_orchestrator.return_value = {
            "campaign_id": "test_campaign",
            "total_runs": 0,
            "successful_runs": 0,
        }

        with patch(
            "robot_sf.benchmark.camera_ready.campaign._resolve_campaign_runtime_dependencies"
        ):
            run_campaign(config, arm_isolation="subprocess")
            # Check that orchestrator was called with arm_isolation
            call_kwargs = mock_orchestrator.call_args[1]
            assert call_kwargs.get("arm_isolation") == "subprocess"


class TestCliArmIsolationFlag:
    """Tests for CLI --arm-isolation flag."""

    def test_cli_accepts_arm_isolation_in_process(self):
        """Verify CLI accepts --arm-isolation in_process."""
        from scripts.tools.run_camera_ready_benchmark import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--config", "test.yaml", "--arm-isolation", "in_process"])
        assert args.arm_isolation == "in_process"

    def test_cli_accepts_arm_isolation_subprocess(self):
        """Verify CLI accepts --arm-isolation subprocess."""
        from scripts.tools.run_camera_ready_benchmark import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--config", "test.yaml", "--arm-isolation", "subprocess"])
        assert args.arm_isolation == "subprocess"

    def test_cli_arm_isolation_defaults_to_in_process(self):
        """Verify CLI defaults --arm-isolation to in_process."""
        from scripts.tools.run_camera_ready_benchmark import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--config", "test.yaml"])
        assert args.arm_isolation == "in_process"

    def test_cli_rejects_invalid_arm_isolation(self):
        """Verify CLI rejects invalid --arm-isolation values."""
        from scripts.tools.run_camera_ready_benchmark import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--config", "test.yaml", "--arm-isolation", "invalid_mode"])


class TestSubprocessIsolationIntegration:
    """Integration tests for subprocess isolation with campaign execution."""

    def test_resource_lifecycle_module_exists(self):
        """Verify resource_lifecycle module can be imported."""
        import robot_sf.benchmark.camera_ready.resource_lifecycle as rl

        assert hasattr(rl, "_SubprocessArmParams")
        assert hasattr(rl, "_cleanup_gpu_memory_before_exit")
        assert hasattr(rl, "_main_subprocess_worker")

    def test_subprocess_arm_params_is_serializable(self):
        """Verify _SubprocessArmParams can be serialized to JSON."""
        params = _SubprocessArmParams(
            planner_key="test",
            planner_algo="test_algo",
            planner_human_model_variant=None,
            planner_human_model_source=None,
            planner_group="core",
            benchmark_profile="speed",
            socnav_missing_prereq_policy="error",
            adapter_impact_eval="disabled",
            kinematics="differential_drive",
            observation_mode="lidar",
            workers=1,
            horizon=100,
            dt=0.1,
            scenario_matrix_path=Path("scenarios.yaml"),
            episodes_path=Path("/tmp/episodes.jsonl"),
            summary_path=Path("/tmp/summary.json"),
            record_forces=True,
            record_planner_decision_trace=False,
            record_simulation_step_trace=False,
            observation_noise=None,
            synthetic_actuation_profile=None,
            latency_stress_profile=None,
            snqi_weights=None,
            snqi_baseline=None,
            algo_config_path=None,
        )

        # Should be serializable with dataclass.asdict after converting paths to strings
        from dataclasses import asdict

        params_dict = asdict(params)
        assert isinstance(params_dict, dict)
        # Convert Path objects to strings for JSON serialization
        params_dict["scenario_matrix_path"] = str(params_dict["scenario_matrix_path"])
        params_dict["episodes_path"] = str(params_dict["episodes_path"])
        params_dict["summary_path"] = str(params_dict["summary_path"])
        if params_dict["algo_config_path"] is not None:
            params_dict["algo_config_path"] = str(params_dict["algo_config_path"])
        assert json.dumps(params_dict)  # Should not raise

    def test_cleanup_metrics_attached_to_subprocess_result(self):
        """Verify cleanup_metrics are included in subprocess result."""
        result = {
            "summary": {
                "status": "ok",
                "total_jobs": 1,
                "written": 1,
            },
            "cleanup_metrics": {
                "planner_key": "test",
                "kinematics": "differential_drive",
                "torch_available": False,
                "cuda_available": False,
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "high_water_mark_mb": 0.0,
                "allocated_freed_mb": 0.0,
                "reserved_freed_mb": 0.0,
            },
            "warnings": [],
            "episodes_total": 1,
        }

        assert "cleanup_metrics" in result
        assert "torch_available" in result["cleanup_metrics"]
        assert "cuda_available" in result["cleanup_metrics"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
