"""Tests for robot_sf.benchmark.camera_ready.resource_lifecycle — subprocess isolation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.camera_ready.resource_lifecycle import (
    _SUBPROCESS_ARM_PATH_FIELDS,
    _cleanup_gpu_memory_before_exit,
    _serialize_subprocess_arm_params,
    _SubprocessArmParams,
    _write_json,
)


def _arm_params(tmp_path: Path) -> _SubprocessArmParams:
    """Build a minimal _SubprocessArmParams for testing."""
    return _SubprocessArmParams(
        planner_key="sf",
        planner_algo="social_force",
        planner_human_model_variant=None,
        planner_human_model_source=None,
        planner_group="classic",
        benchmark_profile="standard",
        socnav_missing_prereq_policy="skip",
        adapter_impact_eval="none",
        kinematics="differential_drive",
        observation_mode="full",
        workers=1,
        horizon=500,
        dt=0.1,
        scenario_matrix_path=tmp_path / "matrix.yaml",
        episodes_path=tmp_path / "episodes.jsonl",
        summary_path=tmp_path / "summary.json",
        record_forces=False,
        record_planner_decision_trace=False,
        record_simulation_step_trace=False,
        observation_noise=None,
        synthetic_actuation_profile=None,
        latency_stress_profile=None,
        snqi_weights=None,
        snqi_baseline=None,
        algo_config_path=None,
    )


class TestSubprocessArmParams:
    """Tests for _SubprocessArmParams dataclass."""

    def test_frozen_dataclass(self) -> None:
        """_SubprocessArmParams must be immutable (frozen)."""
        params = _arm_params(Path("/tmp"))
        with pytest.raises(AttributeError):
            params.planner_key = "other"  # type: ignore[misc]

    def test_default_resume_false(self) -> None:
        """resume must default to False."""
        params = _arm_params(Path("/tmp"))
        assert params.resume is False

    def test_default_scoped_scenarios_path_none(self) -> None:
        """scoped_scenarios_path must default to None."""
        params = _arm_params(Path("/tmp"))
        assert params.scoped_scenarios_path is None

    def test_default_safety_wrapper_none(self) -> None:
        """safety_wrapper must default to None."""
        params = _arm_params(Path("/tmp"))
        assert params.safety_wrapper is None


class TestSerializeSubprocessArmParams:
    """Tests for _serialize_subprocess_arm_params JSON serialization."""

    def test_produces_valid_json(self, tmp_path: Path) -> None:
        """Serialization must produce valid JSON."""
        params = _arm_params(tmp_path)
        result = _serialize_subprocess_arm_params(params)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_path_fields_converted_to_str(self, tmp_path: Path) -> None:
        """Path fields must be serialized as strings, not PosixPath."""
        params = _arm_params(tmp_path)
        result = _serialize_subprocess_arm_params(params)
        parsed = json.loads(result)
        for field_name in _SUBPROCESS_ARM_PATH_FIELDS:
            value = parsed.get(field_name)
            if value is not None:
                assert isinstance(value, str), f"{field_name} must be str, got {type(value)}"

    def test_none_path_fields_stay_none(self, tmp_path: Path) -> None:
        """None path fields must remain None after serialization."""
        params = _arm_params(tmp_path)
        result = _serialize_subprocess_arm_params(params)
        parsed = json.loads(result)
        assert parsed["algo_config_path"] is None
        assert parsed["scoped_scenarios_path"] is None

    def test_non_path_fields_preserved(self, tmp_path: Path) -> None:
        """Non-path fields must be preserved in serialization."""
        params = _arm_params(tmp_path)
        result = _serialize_subprocess_arm_params(params)
        parsed = json.loads(result)
        assert parsed["planner_key"] == "sf"
        assert parsed["planner_algo"] == "social_force"
        assert parsed["workers"] == 1
        assert parsed["horizon"] == 500
        assert parsed["kinematics"] == "differential_drive"

    def test_round_trip_reconstruction(self, tmp_path: Path) -> None:
        """Serialized params must be reconstructable into _SubprocessArmParams."""
        params = _arm_params(tmp_path)
        result = _serialize_subprocess_arm_params(params)
        parsed = json.loads(result)
        for field_name in _SUBPROCESS_ARM_PATH_FIELDS:
            if parsed.get(field_name):
                parsed[field_name] = Path(parsed[field_name])
        rebuilt = _SubprocessArmParams(**parsed)
        assert rebuilt.planner_key == params.planner_key
        assert rebuilt.scenario_matrix_path == params.scenario_matrix_path


class TestCleanupGpuMemoryBeforeExit:
    """Tests for _cleanup_gpu_memory_before_exit."""

    def test_returns_metrics_dict(self) -> None:
        """Must return a dict with expected keys even without torch."""
        result = _cleanup_gpu_memory_before_exit(planner_key="sf", kinematics="diff")
        assert isinstance(result, dict)
        assert result["planner_key"] == "sf"
        assert result["kinematics"] == "diff"
        assert "torch_available" in result
        assert "cuda_available" in result
        assert "allocated_mb" in result
        assert "reserved_mb" in result

    def test_no_torch_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When torch is unavailable, cleanup must return deterministic zero metrics."""
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        result = _cleanup_gpu_memory_before_exit(planner_key="x", kinematics="y")
        assert result["torch_available"] is False
        assert result["cuda_available"] is False
        assert result["allocated_mb"] == 0.0
        assert result["reserved_mb"] == 0.0


class TestWriteJson:
    """Tests for _write_json helper."""

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        """_write_json must produce a readable JSON file."""
        target = tmp_path / "sub" / "output.json"
        _write_json(target, {"key": "value", "num": 42})
        assert target.exists()
        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert loaded["key"] == "value"
        assert loaded["num"] == 42

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """_write_json must create parent directories if missing."""
        target = tmp_path / "deep" / "nested" / "output.json"
        _write_json(target, {"a": 1})
        assert target.exists()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """_write_json must overwrite an existing file."""
        target = tmp_path / "output.json"
        _write_json(target, {"old": True})
        _write_json(target, {"new": True})
        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert "new" in loaded
        assert "old" not in loaded
