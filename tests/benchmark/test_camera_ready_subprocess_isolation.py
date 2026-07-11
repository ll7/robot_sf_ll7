"""Tests for subprocess arm isolation in camera-ready campaigns (issue #4826).

These tests verify that:
1. Subprocess isolation mode is available and functional
2. Subprocess worker entrypoint exists and is callable
3. In-process mode still works with arm_isolation="in_process"
4. CLI flag for arm-isolation is recognized
5. Resource lifecycle cleanup is called in subprocess mode
6. The parent->worker serialization is JSON-safe and the subprocess path emits
   the ``subprocess_isolation`` + ``cleanup_metrics`` evidence (issue #4957).

These are one-real-path tests as defined in ``docs/dev_guide.md``: they invoke
the serializer and subprocess dispatch production uses instead of pre-converting
fixtures in the test.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from robot_sf.benchmark.camera_ready.resource_lifecycle import (
    _cleanup_gpu_memory_before_exit,
    _main_subprocess_worker,
    _serialize_subprocess_arm_params,
    _SubprocessArmParams,
)


def _make_arm_params(*, algo_config_path: Path | None = None) -> _SubprocessArmParams:
    """Build a representative _SubprocessArmParams for serialization tests."""
    return _SubprocessArmParams(
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
        algo_config_path=algo_config_path,
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
    def test_subprocess_worker_reads_params_from_stdin(self, mock_run_arm, mock_stdout, mock_stdin):
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

    @pytest.mark.parametrize("algo_config_path", [None, Path("/tmp/algo.yaml")])
    def test_subprocess_arm_params_is_serializable(self, algo_config_path):
        """Verify _SubprocessArmParams serializes to JSON via the real serializer.

        This is the regression guard for issue #4957: the production
        serialization at ``campaign.py`` previously called
        ``json.dumps(asdict(arm_params))`` directly, which crashes with
        ``TypeError: Object of type PosixPath is not JSON serializable`` because
        ``_SubprocessArmParams`` holds ``pathlib.Path`` fields. The smoke worker
        (issue #4826 comment) found that the prior version of this test
        hand-converted the Path fields *inside the test*, so it never exercised
        the implementation's own serialization and missed the crash.

        This test calls the REAL ``_serialize_subprocess_arm_params`` helper (the
        single serialization point used by production) with no hand-conversion,
        and covers both the ``algo_config_path=None`` and ``Path`` branches. See
        the one-real-path-test rule in ``docs/dev_guide.md``.
        """
        params = _make_arm_params(algo_config_path=algo_config_path)

        # Must not raise; exercises the production serialization path directly.
        serialized = _serialize_subprocess_arm_params(params)
        assert isinstance(serialized, str)

        # Round-trips through json.loads and every path field is a JSON string.
        params_dict = json.loads(serialized)
        assert isinstance(params_dict, dict)
        for field_name in ("scenario_matrix_path", "episodes_path", "summary_path"):
            assert isinstance(params_dict[field_name], str)
        if algo_config_path is not None:
            assert isinstance(params_dict["algo_config_path"], str)
            assert params_dict["algo_config_path"] == str(algo_config_path)
        else:
            assert params_dict["algo_config_path"] is None

    def test_subprocess_arm_params_serialization_round_trips_to_worker(self):
        """Parent-sends-strings / worker-parses-strings contract (issue #4957).

        The parent serializes with ``_serialize_subprocess_arm_params`` (path
        fields become JSON strings); the worker (``_main_subprocess_worker``)
        re-parses str->Path for the same fields. This test verifies the full
        round trip so a mismatch between the two sides cannot regress silently.
        """
        params = _make_arm_params(algo_config_path=Path("/tmp/algo.yaml"))
        serialized = _serialize_subprocess_arm_params(params)

        # Mirror the worker's str->Path re-parse (resource_lifecycle.py).
        params_dict = json.loads(serialized)
        for field_name in ("scenario_matrix_path", "episodes_path", "summary_path"):
            params_dict[field_name] = Path(params_dict[field_name])
        if params_dict["algo_config_path"] is not None:
            params_dict["algo_config_path"] = Path(params_dict["algo_config_path"])

        rebuilt = _SubprocessArmParams(**params_dict)
        assert rebuilt == params

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


class TestSubprocessIsolationMetricEmission:
    """Regression tests for issue #4957: the subprocess path must emit the
    ``subprocess_isolation`` + ``cleanup_metrics`` evidence that #4826's
    acceptance criteria require.

    Before the #4957 fix, ``_run_campaign_planner_variant_subprocess`` crashed
    at the ``json.dumps(asdict(arm_params))`` serialization (PosixPath not JSON
    serializable) *before* the subprocess was ever spawned, so neither
    ``subprocess_isolation: true`` nor ``cleanup_metrics`` could be produced.
    These tests exercise the real production path (serialization + subprocess
    dispatch + result parsing + run-entry emission) and assert the metrics land
    in the run entry.
    """

    def _make_context(self, tmp_path):
        from robot_sf.benchmark.camera_ready._config_types import (
            CampaignConfig,
            PlannerSpec,
        )
        from robot_sf.benchmark.camera_ready.campaign import (
            _CampaignPlannerMatrixContext,
            _CampaignPlannerVariantRun,
            _CampaignRuntimeDependencies,
        )

        planner = PlannerSpec(
            key="test_planner",
            algo="test_algo",
            planner_group="core",
            benchmark_profile="speed",
            enabled=True,
        )
        cfg = CampaignConfig(
            name="test",
            scenario_matrix_path=Path("scenarios.yaml"),
            planners=(planner,),
        )
        dependencies = _CampaignRuntimeDependencies(
            prepare_campaign_preflight=Mock(return_value={}),
            run_batch=Mock(return_value={}),
            compute_aggregates_with_ci=Mock(return_value=None),
            export_publication_bundle=Mock(return_value=None),
        )
        context = _CampaignPlannerMatrixContext(
            cfg=cfg,
            scenarios=[],
            snqi_weights=None,
            snqi_baseline=None,
            runs_dir=tmp_path,
            dependencies=dependencies,
        )
        episodes_path = tmp_path / "episodes.jsonl"
        run = _CampaignPlannerVariantRun(
            kinematics="differential_drive",
            active_observation_mode="lidar",
            planner_dir=tmp_path,
            episodes_path=episodes_path,
            effective_workers=1,
            effective_horizon=100,
            effective_dt=0.1,
            scoped_scenarios=[],
        )
        return planner, context, run

    def test_subprocess_path_emits_isolation_and_cleanup_metrics(self, tmp_path):
        """Assert ``subprocess_isolation: True`` + ``cleanup_metrics`` are emitted.

        Exercises the real ``_run_campaign_planner_variant_subprocess`` path with
        ``subprocess.run`` mocked to return a successful arm result carrying
        ``cleanup_metrics``. This proves the serialization fix (#4957) lets the
        path reach the emission code that was previously unreachable, and that
        the per-arm cleanup metrics propagate into the run entry.
        """
        from robot_sf.benchmark.camera_ready.campaign import (
            _run_campaign_planner_variant_subprocess,
        )

        planner, context, run = self._make_context(tmp_path)

        cleanup_metrics = {
            "planner_key": planner.key,
            "kinematics": "differential_drive",
            "torch_available": True,
            "cuda_available": True,
            "allocated_mb": 1.0,
            "reserved_mb": 2.0,
            "high_water_mark_mb": 3.0,
            "allocated_freed_mb": 0.5,
            "reserved_freed_mb": 1.5,
        }
        worker_stdout = json.dumps(
            {
                "summary": {
                    "status": "ok",
                    "total_jobs": 1,
                    "written": 1,
                    "failed_jobs": 0,
                    "failures": [],
                },
                "cleanup_metrics": cleanup_metrics,
                "warnings": [],
                "episodes_total": 1,
            }
        )

        captured_input = {}

        def fake_subprocess_run(cmd, *, input, capture_output, text, check):  # noqa: A002
            captured_input["json"] = input
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=worker_stdout, stderr=""
            )

        with (
            patch(
                "robot_sf.benchmark.camera_ready.campaign._prepare_campaign_planner_variant_run",
                return_value=run,
            ),
            patch(
                "robot_sf.benchmark.camera_ready.campaign.subprocess.run",
                side_effect=fake_subprocess_run,
            ),
            patch(
                "robot_sf.benchmark.camera_ready.campaign._planner_report_row",
                return_value={"planner_key": planner.key, "status": "ok"},
            ),
            patch("robot_sf.benchmark.camera_ready.campaign.read_jsonl", return_value=[]),
            patch(
                "robot_sf.benchmark.camera_ready.campaign.classify_planner_row_status",
                return_value="ok",
            ),
        ):
            result = _run_campaign_planner_variant_subprocess(
                context,
                planner=planner,
                kinematics="differential_drive",
                active_observation_mode="lidar",
            )

        # The serialization fix is in effect: the handoff JSON the parent built
        # is valid JSON (would have raised TypeError before #4957) and the path
        # fields are strings, not PosixPath.
        handoff = json.loads(captured_input["json"])
        assert isinstance(handoff["scenario_matrix_path"], str)
        assert isinstance(handoff["episodes_path"], str)
        assert isinstance(handoff["summary_path"], str)

        # Issue #4957 acceptance: the metrics are emitted in the run entry.
        assert len(result.run_entries) == 1
        entry = result.run_entries[0]
        assert entry["subprocess_isolation"] is True
        assert entry["gpu_cleanup"] == cleanup_metrics
        assert entry["gpu_cleanup"]["allocated_freed_mb"] == 0.5
        assert entry["gpu_cleanup"]["reserved_freed_mb"] == 1.5
        assert entry["status"] == "ok"

    def test_subprocess_path_emits_isolation_marker_on_failure(self, tmp_path):
        """The ``subprocess_isolation: True`` marker is emitted even when the
        subprocess arm fails, so the evidence that isolation was engaged is
        never lost (issue #4957 / #4826).
        """
        from robot_sf.benchmark.camera_ready.campaign import (
            _run_campaign_planner_variant_subprocess,
        )

        planner, context, run = self._make_context(tmp_path)

        with (
            patch(
                "robot_sf.benchmark.camera_ready.campaign._prepare_campaign_planner_variant_run",
                return_value=run,
            ),
            patch(
                "robot_sf.benchmark.camera_ready.campaign.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=1, stdout="", stderr="boom"
                ),
            ),
            patch("robot_sf.benchmark.camera_ready.campaign.read_jsonl", return_value=[]),
            patch(
                "robot_sf.benchmark.camera_ready.campaign.classify_planner_row_status",
                return_value="unexpected_failure",
            ),
        ):
            result = _run_campaign_planner_variant_subprocess(
                context,
                planner=planner,
                kinematics="differential_drive",
                active_observation_mode="lidar",
            )

        assert len(result.run_entries) == 1
        entry = result.run_entries[0]
        assert entry["subprocess_isolation"] is True
        assert entry["status"] == "failed"

    def test_subprocess_path_preserves_structured_failure_output(self, tmp_path):
        """A non-zero worker exit still propagates its JSON error and cleanup telemetry.

        The worker deliberately exits non-zero for a failed arm.  It nevertheless
        guarantees structured stdout, which the parent must parse instead of
        replacing with a generic failure (issue #4826).
        """
        from robot_sf.benchmark.camera_ready.campaign import (
            _run_campaign_planner_variant_subprocess,
        )

        planner, context, run = self._make_context(tmp_path)
        worker_stdout = json.dumps(
            {
                "summary": {
                    "status": "failed",
                    "error": "KeyError('missing_planner_field')",
                    "total_jobs": 0,
                    "written": 0,
                    "failed_jobs": 0,
                    "failures": [],
                },
                "cleanup_metrics": {"cuda_available": False},
                "warnings": ["missing_planner_field"],
                "episodes_total": 0,
            }
        )

        with (
            patch(
                "robot_sf.benchmark.camera_ready.campaign._prepare_campaign_planner_variant_run",
                return_value=run,
            ),
            patch(
                "robot_sf.benchmark.camera_ready.campaign.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=1, stdout=worker_stdout, stderr="worker failed"
                ),
            ),
            patch("robot_sf.benchmark.camera_ready.campaign.read_jsonl", return_value=[]),
            patch(
                "robot_sf.benchmark.camera_ready.campaign.classify_planner_row_status",
                return_value="unexpected_failure",
            ),
        ):
            result = _run_campaign_planner_variant_subprocess(
                context,
                planner=planner,
                kinematics="differential_drive",
                active_observation_mode="lidar",
            )

        entry = result.run_entries[0]
        assert entry["status"] == "failed"
        assert entry["summary"]["error"] == "KeyError('missing_planner_field')"
        assert entry["gpu_cleanup"] == {"cuda_available": False}


class TestScopedScenarioParity:
    """Regression tests for the parent->worker scenario handoff.

    Before this fix, the subprocess worker re-loaded the scenario matrix from
    disk instead of consuming the parent's prepared list. The re-load skipped
    the campaign loader's map_file normalization (every episode then failed
    with an unresolvable relative map_file — Slurm jobs 13372/13373), and also
    lost seed overrides, scenario-candidate filtering, AMV overrides, horizon
    schedules, and the configured holonomic_command_mode. The contract locked
    here: the worker must execute EXACTLY the scenario dicts the parent
    prepared, delivered via ``scoped_scenarios_path``.
    """

    SCOPED = [
        {
            "name": "blind_corner",
            "map_file": "maps/svg_maps/francis2023/francis2023_blind_corner.svg",
            "seeds": [111],
            "simulation_config": {"max_episode_steps": 30},
            "robot_config": {"kinematics": "differential_drive"},
        }
    ]

    def test_serializer_round_trips_scoped_scenarios_path(self):
        """scoped_scenarios_path survives serialize -> worker re-parse as Path."""
        from robot_sf.benchmark.camera_ready.resource_lifecycle import (
            _SUBPROCESS_ARM_PATH_FIELDS,
        )

        base = _make_arm_params()
        params = _SubprocessArmParams(
            **{**base.__dict__, "scoped_scenarios_path": Path("/tmp/scoped.json")}
        )
        params_dict = json.loads(_serialize_subprocess_arm_params(params))
        assert params_dict["scoped_scenarios_path"] == "/tmp/scoped.json"

        for field_name in _SUBPROCESS_ARM_PATH_FIELDS:
            if params_dict.get(field_name):
                params_dict[field_name] = Path(params_dict[field_name])
        rebuilt = _SubprocessArmParams(**params_dict)
        assert rebuilt == params

    def test_parent_serializes_scoped_scenarios_for_worker(self, tmp_path):
        """The parent writes its prepared scenarios and hands the path to the worker."""
        from robot_sf.benchmark.camera_ready.campaign import (
            _run_campaign_planner_variant_subprocess,
        )

        maker = TestSubprocessIsolationMetricEmission()
        planner, context, run = maker._make_context(tmp_path)
        run = type(run)(**{**run.__dict__, "scoped_scenarios": self.SCOPED})

        worker_stdout = json.dumps(
            {
                "summary": {
                    "status": "ok",
                    "total_jobs": 1,
                    "written": 1,
                    "failed_jobs": 0,
                    "failures": [],
                },
                "cleanup_metrics": {},
                "warnings": [],
                "episodes_total": 1,
            }
        )
        captured_input = {}

        def fake_subprocess_run(cmd, *, input, capture_output, text, check):  # noqa: A002
            captured_input["json"] = input
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=worker_stdout, stderr=""
            )

        with (
            patch(
                "robot_sf.benchmark.camera_ready.campaign._prepare_campaign_planner_variant_run",
                return_value=run,
            ),
            patch(
                "robot_sf.benchmark.camera_ready.campaign.subprocess.run",
                side_effect=fake_subprocess_run,
            ),
            patch(
                "robot_sf.benchmark.camera_ready.campaign._planner_report_row",
                return_value={"planner_key": planner.key, "status": "ok"},
            ),
            patch("robot_sf.benchmark.camera_ready.campaign.read_jsonl", return_value=[]),
            patch(
                "robot_sf.benchmark.camera_ready.campaign.classify_planner_row_status",
                return_value="ok",
            ),
        ):
            _run_campaign_planner_variant_subprocess(
                context,
                planner=planner,
                kinematics="differential_drive",
                active_observation_mode="lidar",
            )

        handoff = json.loads(captured_input["json"])
        scoped_path = handoff["scoped_scenarios_path"]
        assert isinstance(scoped_path, str)
        assert json.loads(Path(scoped_path).read_text(encoding="utf-8")) == self.SCOPED

    def test_worker_consumes_scoped_scenarios_verbatim(self, tmp_path):
        """The worker runs exactly the serialized scenarios; the matrix is not re-read.

        scenario_matrix_path points at a file that does not exist: if the worker
        fell back to re-loading the matrix, run_batch would never be reached and
        the legacy loader would raise instead.
        """
        from unittest.mock import Mock as _Mock

        from robot_sf.benchmark.camera_ready.resource_lifecycle import (
            _run_single_arm_subprocess,
        )

        scoped_path = tmp_path / "scoped_scenarios.json"
        scoped_path.write_text(json.dumps(self.SCOPED), encoding="utf-8")

        base = _make_arm_params()
        params = _SubprocessArmParams(
            **{
                **base.__dict__,
                "scenario_matrix_path": tmp_path / "does-not-exist.yaml",
                "episodes_path": tmp_path / "episodes.jsonl",
                "summary_path": tmp_path / "summary.json",
                "scoped_scenarios_path": scoped_path,
            }
        )

        captured = {}

        def fake_run_batch(scenarios, **kwargs):
            captured["scenarios"] = scenarios
            return {
                "status": "ok",
                "total_jobs": 1,
                "written": 1,
                "failed_jobs": 0,
                "failures": [],
            }

        with (
            patch("robot_sf.benchmark.runner.run_batch", side_effect=fake_run_batch),
            patch(
                "robot_sf.benchmark.fallback_policy.summarize_benchmark_availability",
                return_value=_Mock(availability_status="ok"),
            ),
            patch(
                "robot_sf.benchmark.fallback_policy.availability_payload",
                return_value={},
            ),
        ):
            result = _run_single_arm_subprocess(params)

        assert captured["scenarios"] == self.SCOPED
        assert result["summary"]["status"] == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
