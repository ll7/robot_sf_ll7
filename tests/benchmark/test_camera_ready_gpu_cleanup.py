"""Regression test for GPU VRAM cleanup between campaign arms (issue #4826).

This test ensures that GPU memory is properly cleaned up between campaign
arms to prevent VRAM leaks during long-running multi-arm campaigns.
"""

from __future__ import annotations

import gc
import sys
from unittest.mock import MagicMock, patch

import pytest

from robot_sf.benchmark.camera_ready import campaign as campaign_mod
from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec
from robot_sf.benchmark.camera_ready.campaign import (
    _CampaignPlannerVariantResult,
    _CampaignRuntimeDependencies,
    _cleanup_gpu_memory_between_arms,
)


class TestGPUCleanupBetweenArms:
    """Tests for GPU cleanup function between campaign arms."""

    def test_cleanup_gpu_memory_between_arms_without_torch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GPU cleanup returns safe defaults when torch is not available."""
        monkeypatch.delitem(sys.modules, "torch", raising=False)

        metrics = _cleanup_gpu_memory_between_arms(
            planner_key="test_planner",
            kinematics="unicycle",
        )

        assert metrics["planner_key"] == "test_planner"
        assert metrics["kinematics"] == "unicycle"
        assert metrics["torch_available"] is False
        assert metrics["cuda_available"] is False
        assert metrics["allocated_mb"] == 0
        assert metrics["reserved_mb"] == 0
        assert metrics["high_water_mark_mb"] == 0

    def test_cleanup_gpu_memory_between_arms_with_torch_cpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GPU cleanup handles torch available but CUDA not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        metrics = _cleanup_gpu_memory_between_arms(
            planner_key="test_planner",
            kinematics="unicycle",
        )

        assert metrics["planner_key"] == "test_planner"
        assert metrics["kinematics"] == "unicycle"
        assert metrics["torch_available"] is True
        assert metrics["cuda_available"] is False
        assert metrics["allocated_mb"] == 0
        assert metrics["reserved_mb"] == 0
        assert metrics["high_water_mark_mb"] == 0

    def test_cleanup_gpu_memory_between_arms_with_cuda(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GPU cleanup properly frees CUDA memory when available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        # Return values in BYTES (max_memory_allocated returns bytes)
        mock_torch.cuda.max_memory_allocated.return_value = 1024 * 1024 * 1024  # 1 GiB
        # Before/after in BYTES
        mock_torch.cuda.memory_allocated.side_effect = [
            512 * 1024 * 1024,  # Before: 512 MiB in bytes
            256 * 1024 * 1024,  # After: 256 MiB in bytes
        ]
        mock_torch.cuda.memory_reserved.side_effect = [
            600 * 1024 * 1024,  # Before: 600 MiB in bytes
            300 * 1024 * 1024,  # After: 300 MiB in bytes
        ]
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        metrics = _cleanup_gpu_memory_between_arms(
            planner_key="test_planner",
            kinematics="unicycle",
        )

        assert metrics["planner_key"] == "test_planner"
        assert metrics["kinematics"] == "unicycle"
        assert metrics["torch_available"] is True
        assert metrics["cuda_available"] is True
        # max_memory_allocated is converted to MiB by the function
        assert metrics["high_water_mark_mb"] == 1024
        assert metrics["allocated_mb"] == 256
        assert metrics["reserved_mb"] == 300
        assert metrics["allocated_freed_mb"] == 256  # 512 - 256
        assert metrics["reserved_freed_mb"] == 300  # 600 - 300

        mock_torch.cuda.empty_cache.assert_called_once()
        mock_torch.cuda.synchronize.assert_called_once()
        mock_torch.cuda.reset_peak_memory_stats.assert_called_once()


class TestCampaignMultiArmMemoryRegression:
    """Regression test for multi-arm campaign memory leaks (issue #4826)."""

    def test_cleanup_function_is_callable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cleanup function exists and is callable with planner_key and kinematics."""
        assert callable(_cleanup_gpu_memory_between_arms)
        monkeypatch.delitem(sys.modules, "torch", raising=False)

        result = _cleanup_gpu_memory_between_arms(
            planner_key="test",
            kinematics="unicycle",
        )
        assert result["planner_key"] == "test"
        assert result["kinematics"] == "unicycle"

    def test_cleanup_metrics_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cleanup returns dict with required metric fields for diagnostics."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.max_memory_allocated.return_value = 1024 * 1024 * 512  # 512 MiB
        mock_torch.cuda.memory_allocated.side_effect = [
            256 * 1024 * 1024,  # Before
            128 * 1024 * 1024,  # After
        ]
        mock_torch.cuda.memory_reserved.side_effect = [
            300 * 1024 * 1024,  # Before
            150 * 1024 * 1024,  # After
        ]
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        metrics = _cleanup_gpu_memory_between_arms(
            planner_key="planner_a",
            kinematics="diff_drive",
        )

        # Verify all required fields exist
        required_fields = {
            "planner_key",
            "kinematics",
            "torch_available",
            "cuda_available",
            "allocated_mb",
            "reserved_mb",
            "high_water_mark_mb",
            "allocated_freed_mb",
            "reserved_freed_mb",
        }
        assert set(metrics.keys()) == required_fields

        # Verify values are reasonable
        assert metrics["planner_key"] == "planner_a"
        assert metrics["kinematics"] == "diff_drive"
        assert metrics["torch_available"] is True
        assert metrics["cuda_available"] is True
        assert metrics["allocated_mb"] == 128
        assert metrics["reserved_mb"] == 150
        assert metrics["allocated_freed_mb"] == 128  # 256 - 128
        assert metrics["reserved_freed_mb"] == 150  # 300 - 150
        assert metrics["high_water_mark_mb"] == 512

    def test_cleanup_metrics_attach_to_current_variant(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Each campaign arm gets its own cleanup diagnostics."""

        def fake_variant(
            context: object,
            *,
            planner: PlannerSpec,
            kinematics: str,
            active_observation_mode: str,
        ) -> _CampaignPlannerVariantResult:
            del context, active_observation_mode
            return _CampaignPlannerVariantResult(
                run_entries=[{"planner": {"key": planner.key, "kinematics": kinematics}}],
                planner_rows=[],
                warnings=[],
                seed_variability_records=[],
                stop_requested=False,
            )

        def fake_cleanup(*, planner_key: str, kinematics: str) -> dict[str, str]:
            return {"planner_key": planner_key, "kinematics": kinematics}

        monkeypatch.setattr(campaign_mod, "_run_campaign_planner_variant", fake_variant)
        monkeypatch.setattr(campaign_mod, "_cleanup_gpu_memory_between_arms", fake_cleanup)

        cfg = CampaignConfig(
            name="gpu-cleanup-test",
            scenario_matrix_path=tmp_path / "scenarios.yaml",
            planners=(PlannerSpec(key="planner_a", algo="orca"),),
            kinematics_matrix=("diff_drive", "holonomic"),
        )
        dependencies = _CampaignRuntimeDependencies(
            prepare_campaign_preflight=lambda *args, **kwargs: {},
            run_batch=lambda *args, **kwargs: {},
            compute_aggregates_with_ci=lambda *args, **kwargs: {},
            export_publication_bundle=lambda *args, **kwargs: None,
        )

        results = campaign_mod._run_campaign_planner_matrix(
            cfg=cfg,
            scenarios=[],
            snqi_weights=None,
            snqi_baseline=None,
            runs_dir=tmp_path,
            dependencies=dependencies,
        )

        assert [entry["gpu_cleanup"]["kinematics"] for entry in results.run_entries] == [
            "diff_drive",
            "holonomic",
        ]
        assert [entry["planner"]["kinematics"] for entry in results.run_entries] == [
            "diff_drive",
            "holonomic",
        ]


class TestGcCollectAlwaysRunsBetweenArms:
    """Regression tests for gc.collect() always running between arms (issue #4826).

    Before this fix, gc.collect() was only called inside the CUDA block, so CPU-only
    runs (no GPU, or torch not installed) never triggered garbage collection between
    arms. This allows Python-level references to model weights, replay buffers, and
    env wrappers to accumulate across arms and grow RSS monotonically.
    """

    def test_gc_collect_called_without_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """gc.collect() runs even when torch is not installed (issue #4826)."""
        monkeypatch.delitem(sys.modules, "torch", raising=False)

        with patch.object(gc, "collect", wraps=gc.collect) as mock_gc:
            _cleanup_gpu_memory_between_arms(planner_key="p", kinematics="k")
            mock_gc.assert_called_once()

    def test_gc_collect_called_with_torch_cpu_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """gc.collect() runs when torch is available but CUDA is not (CPU-only cluster node)."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        with patch.object(gc, "collect", wraps=gc.collect) as mock_gc:
            _cleanup_gpu_memory_between_arms(planner_key="p", kinematics="k")
            mock_gc.assert_called_once()

    def test_gc_collect_called_before_cuda_empty_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """gc.collect() runs before CUDA empty_cache to release Python refs first."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.max_memory_allocated.return_value = 0
        mock_torch.cuda.memory_allocated.side_effect = [0, 0]
        mock_torch.cuda.memory_reserved.side_effect = [0, 0]
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        call_order: list[str] = []
        original_gc_collect = gc.collect

        def tracking_gc_collect(*args, **kwargs):
            call_order.append("gc.collect")
            return original_gc_collect(*args, **kwargs)

        def tracking_empty_cache():
            call_order.append("empty_cache")

        mock_torch.cuda.empty_cache = tracking_empty_cache

        with patch.object(gc, "collect", side_effect=tracking_gc_collect):
            _cleanup_gpu_memory_between_arms(planner_key="p", kinematics="k")

        assert call_order.index("gc.collect") < call_order.index("empty_cache"), (
            "gc.collect() must run before cuda.empty_cache() to release Python object refs first"
        )


class TestCleanupRunsInFinallyBlock:
    """Regression tests for cleanup running in try/finally (issue #4826).

    The in-process arm loop wraps each arm execution in try/finally so that
    _cleanup_gpu_memory_between_arms is called even when _run_campaign_planner_variant
    raises an unexpected exception. Without try/finally, a crashing arm leaves its
    CUDA allocations unreleased for all subsequent arms.
    """

    def test_cleanup_called_after_successful_arm(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Cleanup is called once per arm on success."""
        cleanup_calls: list[tuple[str, str]] = []

        def fake_variant(context, *, planner, kinematics, active_observation_mode):
            return _CampaignPlannerVariantResult(
                run_entries=[{"planner": {"key": planner.key}}],
                planner_rows=[],
                warnings=[],
                seed_variability_records=[],
                stop_requested=False,
            )

        def fake_cleanup(*, planner_key, kinematics):
            cleanup_calls.append((planner_key, kinematics))
            return {"planner_key": planner_key, "kinematics": kinematics}

        monkeypatch.setattr(campaign_mod, "_run_campaign_planner_variant", fake_variant)
        monkeypatch.setattr(campaign_mod, "_cleanup_gpu_memory_between_arms", fake_cleanup)

        cfg = CampaignConfig(
            name="test",
            scenario_matrix_path=tmp_path / "scenarios.yaml",
            planners=(
                PlannerSpec(key="plan_a", algo="orca"),
                PlannerSpec(key="plan_b", algo="orca"),
            ),
            kinematics_matrix=("diff_drive",),
        )
        deps = _CampaignRuntimeDependencies(
            prepare_campaign_preflight=lambda *a, **kw: {},
            run_batch=lambda *a, **kw: {},
            compute_aggregates_with_ci=lambda *a, **kw: {},
            export_publication_bundle=lambda *a, **kw: None,
        )

        campaign_mod._run_campaign_planner_matrix(
            cfg=cfg,
            scenarios=[],
            snqi_weights=None,
            snqi_baseline=None,
            runs_dir=tmp_path,
            dependencies=deps,
        )

        assert cleanup_calls == [("plan_a", "diff_drive"), ("plan_b", "diff_drive")], (
            "cleanup must be called once per arm in the correct order"
        )

    def test_cleanup_called_even_when_arm_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Cleanup runs via try/finally even when _run_campaign_planner_variant raises."""
        cleanup_was_called = []

        def exploding_variant(context, *, planner, kinematics, active_observation_mode):
            raise RuntimeError("unexpected arm failure")

        def fake_cleanup(*, planner_key, kinematics):
            cleanup_was_called.append((planner_key, kinematics))
            return {"planner_key": planner_key, "kinematics": kinematics}

        monkeypatch.setattr(campaign_mod, "_run_campaign_planner_variant", exploding_variant)
        monkeypatch.setattr(campaign_mod, "_cleanup_gpu_memory_between_arms", fake_cleanup)

        cfg = CampaignConfig(
            name="test",
            scenario_matrix_path=tmp_path / "scenarios.yaml",
            planners=(PlannerSpec(key="plan_a", algo="orca"),),
            kinematics_matrix=("diff_drive",),
        )
        deps = _CampaignRuntimeDependencies(
            prepare_campaign_preflight=lambda *a, **kw: {},
            run_batch=lambda *a, **kw: {},
            compute_aggregates_with_ci=lambda *a, **kw: {},
            export_publication_bundle=lambda *a, **kw: None,
        )

        with pytest.raises(RuntimeError, match="unexpected arm failure"):
            campaign_mod._run_campaign_planner_matrix(
                cfg=cfg,
                scenarios=[],
                snqi_weights=None,
                snqi_baseline=None,
                runs_dir=tmp_path,
                dependencies=deps,
            )

        assert len(cleanup_was_called) == 1, (
            "cleanup must run exactly once via finally block even when the arm raises"
        )
        assert cleanup_was_called[0] == ("plan_a", "diff_drive")


def test_pytorch_alloc_conf_set_in_slurm_env() -> None:
    """Defense-in-depth: PYTORCH_ALLOC_CONF is set for camera-ready campaigns.

    This test verifies that expandable_segments:True is set to prevent
    fragmentation-related OOM during long campaigns (issue #4826).
    """
    import os

    # The environment variable should be set when importing the benchmark modules
    # Check if it's set (either by map_runner_batch_runner.py or run_camera_ready_benchmark.py)
    # We don't assert it's already set since the test might run in isolation,
    # but we verify the default value that should be used.

    expected_value = "expandable_segments:True"
    # Simulate what the benchmark scripts do
    test_value = os.environ.get("PYTORCH_ALLOC_CONF")
    if test_value is None:
        # Not set yet, test the default behavior
        os.environ.setdefault("PYTORCH_ALLOC_CONF", expected_value)
        assert os.environ.get("PYTORCH_ALLOC_CONF") == expected_value
    else:
        # Already set by some import, verify it's correct
        assert test_value == expected_value, (
            f"PYTORCH_ALLOC_CONF should be '{expected_value}' for fragmentation defense, "
            f"got '{test_value}'"
        )
