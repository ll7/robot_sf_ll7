"""Direct unit coverage for multi-extractor summary dataclass contracts.

Locks the JSON/serialization surface exposed by
``robot_sf.training.multi_extractor_models`` so downstream summary writers and
artifact emitters keep a stable contract:

- ``ExtractorConfigurationProfile.merged_parameters`` defensive-copy behavior
  for ``None`` and populated parameter dictionaries.
- ``HardwareProfile.to_dict`` optional ``gpu_model``/``cuda_version`` omission
  and inclusion.
- ``ExtractorRunRecord.to_dict`` and ``TrainingRunSummary.to_dict`` nested
  serialization, default empty collections, and optional ``reason``/``notes``
  omission versus populated inclusion.

All assertions use in-memory dataclass instances only. No training runs, model
loads, artifact files, or benchmark execution are performed.
"""

from __future__ import annotations

import json
from typing import Any

from robot_sf.training import (
    ExtractorConfigurationProfile,
    ExtractorRunRecord,
    HardwareProfile,
    TrainingRunSummary,
)


def _make_hardware_profile(*, with_gpu: bool = False) -> HardwareProfile:
    """Build a deterministic ``HardwareProfile`` for tests.

    Args:
        with_gpu: When True, populate the optional ``gpu_model`` and
            ``cuda_version`` fields; otherwise leave them ``None``.

    Returns:
        HardwareProfile with fixed platform, arch, python_version, and workers.
    """
    return HardwareProfile(
        platform="linux",
        arch="x86_64",
        python_version="3.13",
        workers=4,
        gpu_model="TestGPU-9000" if with_gpu else None,
        cuda_version="12.4" if with_gpu else None,
    )


def _make_run_record(
    *,
    config_name: str = "baseline",
    status: str = "success",
    metrics: dict[str, float] | None = None,
    artifacts: dict[str, str] | None = None,
    reason: str | None = None,
) -> ExtractorRunRecord:
    """Build a deterministic ``ExtractorRunRecord`` for tests.

    Only the optional ``metrics``/``artifacts``/``reason`` fields passed as
    non-None are forwarded so the dataclass defaults remain exercised when they
    are omitted.

    Args:
        config_name: Extractor configuration name.
        status: Run status string.
        metrics: Optional metrics dict; ``None`` lets the dataclass default apply.
        artifacts: Optional artifacts dict; ``None`` lets the dataclass default apply.
        reason: Optional reason string; ``None`` lets the dataclass default apply.

    Returns:
        ExtractorRunRecord wired to a deterministic HardwareProfile.
    """
    kwargs: dict[str, Any] = {
        "config_name": config_name,
        "status": status,
        "start_time": "2026-07-26T00:00:00Z",
        "end_time": "2026-07-26T00:05:00Z",
        "duration_seconds": 300.0,
        "hardware_profile": _make_hardware_profile(),
        "worker_mode": "single-thread",
        "training_steps": 128,
    }
    if metrics is not None:
        kwargs["metrics"] = metrics
    if artifacts is not None:
        kwargs["artifacts"] = artifacts
    if reason is not None:
        kwargs["reason"] = reason
    return ExtractorRunRecord(**kwargs)


# --- ExtractorConfigurationProfile.merged_parameters ---


def test_merged_parameters_with_none_returns_empty_dict() -> None:
    """``None`` parameters must yield a fresh empty dict, never ``None``."""
    profile = ExtractorConfigurationProfile(name="empty", parameters=None)

    merged = profile.merged_parameters()

    assert merged == {}


def test_merged_parameters_with_none_returns_independent_dict() -> None:
    """Mutating the empty dict returned for ``None`` parameters must not leak a
    mutation back onto the source dataclass."""
    profile = ExtractorConfigurationProfile(name="empty", parameters=None)

    merged = profile.merged_parameters()
    merged["injected"] = True

    assert profile.parameters is None
    assert profile.merged_parameters() == {}


def test_merged_parameters_with_populated_returns_equal_content() -> None:
    """Populated parameters must round-trip their content unchanged and return a
    distinct container rather than the stored reference."""
    parameters = {"lr": 3e-4, "layers": [64, 64], "active": True}
    profile = ExtractorConfigurationProfile(name="populated", parameters=parameters)

    merged = profile.merged_parameters()

    assert merged == parameters
    assert merged is not parameters


def test_merged_parameters_with_populated_returns_defensive_copy() -> None:
    """Top-level mutation of the returned dict must not affect the source."""
    profile = ExtractorConfigurationProfile(
        name="populated",
        parameters={"lr": 3e-4, "layers": [64, 64]},
    )

    merged = profile.merged_parameters()
    merged["lr"] = 999.0
    merged["injected"] = True
    del merged["layers"]

    assert profile.parameters == {"lr": 3e-4, "layers": [64, 64]}
    assert profile.merged_parameters() == {"lr": 3e-4, "layers": [64, 64]}


# --- HardwareProfile.to_dict ---


def test_hardware_profile_to_dict_omits_optional_gpu_and_cuda() -> None:
    """Optional GPU/CUDA fields must be omitted when ``None``."""
    profile = _make_hardware_profile(with_gpu=False)

    payload = profile.to_dict()

    assert payload == {
        "platform": "linux",
        "arch": "x86_64",
        "python_version": "3.13",
        "workers": 4,
    }
    assert "gpu_model" not in payload
    assert "cuda_version" not in payload


def test_hardware_profile_to_dict_includes_optional_gpu_and_cuda() -> None:
    """Optional GPU/CUDA fields must be included when populated."""
    profile = _make_hardware_profile(with_gpu=True)

    payload = profile.to_dict()

    assert payload == {
        "platform": "linux",
        "arch": "x86_64",
        "python_version": "3.13",
        "workers": 4,
        "gpu_model": "TestGPU-9000",
        "cuda_version": "12.4",
    }


def test_hardware_profile_to_dict_includes_only_populated_optional_field() -> None:
    """A populated ``gpu_model`` with ``None`` ``cuda_version`` includes only GPU."""
    profile = HardwareProfile(
        platform="linux",
        arch="x86_64",
        python_version="3.13",
        workers=2,
        gpu_model="OnlyGPU-1",
        cuda_version=None,
    )

    payload = profile.to_dict()

    assert payload["gpu_model"] == "OnlyGPU-1"
    assert "cuda_version" not in payload


# --- ExtractorRunRecord.to_dict ---


def test_extractor_run_record_to_dict_defaults_empty_and_omits_reason() -> None:
    """Default empty ``metrics``/``artifacts`` serialize as empty dicts and a
    ``None`` ``reason`` must be omitted."""
    record = _make_run_record()

    payload = record.to_dict()

    assert payload == {
        "config_name": "baseline",
        "status": "success",
        "start_time": "2026-07-26T00:00:00Z",
        "end_time": "2026-07-26T00:05:00Z",
        "duration_seconds": 300.0,
        "hardware_profile": _make_hardware_profile().to_dict(),
        "worker_mode": "single-thread",
        "training_steps": 128,
        "metrics": {},
        "artifacts": {},
    }
    assert "reason" not in payload


def test_extractor_run_record_to_dict_serializes_nested_hardware_and_reason() -> None:
    """Populated ``metrics``/``artifacts``/``reason`` and nested hardware must
    serialize through their own helpers."""
    record = _make_run_record(
        config_name="candidate",
        status="failed",
        reason="cuda out of memory",
        metrics={"best_mean_reward": 1.1, "convergence_timestep": 20},
        artifacts={
            "extractor_dir": "extractors/candidate",
            "learning_curve": "figs/curve.png",
        },
    )

    payload = record.to_dict()

    assert payload == {
        "config_name": "candidate",
        "status": "failed",
        "start_time": "2026-07-26T00:00:00Z",
        "end_time": "2026-07-26T00:05:00Z",
        "duration_seconds": 300.0,
        "hardware_profile": _make_hardware_profile().to_dict(),
        "worker_mode": "single-thread",
        "training_steps": 128,
        "metrics": {"best_mean_reward": 1.1, "convergence_timestep": 20},
        "artifacts": {
            "extractor_dir": "extractors/candidate",
            "learning_curve": "figs/curve.png",
        },
        "reason": "cuda out of memory",
    }


def test_extractor_run_record_to_dict_empty_string_reason_is_omitted() -> None:
    """The ``reason`` field is gated on truthiness, so an empty string is omitted."""
    record = _make_run_record(reason="")

    payload = record.to_dict()

    assert "reason" not in payload


# --- TrainingRunSummary.to_dict ---


def test_training_run_summary_to_dict_defaults_empty_and_omits_notes() -> None:
    """Default empty hardware/results collections serialize as empty lists and
    ``None`` ``notes`` must be omitted."""
    summary = TrainingRunSummary(
        run_id="run-1",
        created_at="2026-07-26T00:00:00Z",
        output_root="output/run-1",
        hardware_overview=[],
        extractor_results=[],
        aggregate_metrics={},
        notes=None,
    )

    payload = summary.to_dict()

    assert payload == {
        "run_id": "run-1",
        "created_at": "2026-07-26T00:00:00Z",
        "output_root": "output/run-1",
        "hardware_overview": [],
        "extractor_results": [],
        "aggregate_metrics": {},
    }
    assert "notes" not in payload


def test_training_run_summary_to_dict_serializes_nested_profiles_and_records() -> None:
    """Nested ``HardwareProfile`` and ``ExtractorRunRecord`` entries serialize
    through their own ``to_dict`` and populated ``notes`` are included."""
    hardware = _make_hardware_profile(with_gpu=True)
    record = _make_run_record(
        config_name="candidate",
        reason="ok",
        metrics={"best_mean_reward": 1.0},
        artifacts={"extractor_dir": "extractors/candidate"},
    )
    summary = TrainingRunSummary(
        run_id="run-2",
        created_at="2026-07-26T00:10:00Z",
        output_root="output/run-2",
        hardware_overview=[hardware],
        extractor_results=[record],
        aggregate_metrics={"mean_best_reward": 1.0},
        notes=["baseline beat candidate", "gpu available"],
    )

    payload = summary.to_dict()

    assert payload == {
        "run_id": "run-2",
        "created_at": "2026-07-26T00:10:00Z",
        "output_root": "output/run-2",
        "hardware_overview": [hardware.to_dict()],
        "extractor_results": [record.to_dict()],
        "aggregate_metrics": {"mean_best_reward": 1.0},
        "notes": ["baseline beat candidate", "gpu available"],
    }
    assert payload["hardware_overview"][0] == hardware.to_dict()
    assert payload["extractor_results"][0] == record.to_dict()


def test_training_run_summary_to_dict_empty_notes_list_is_omitted() -> None:
    """The ``notes`` field is gated on truthiness, so an empty list is omitted."""
    summary = TrainingRunSummary(
        run_id="run-3",
        created_at="2026-07-26T00:00:00Z",
        output_root="output/run-3",
        hardware_overview=[],
        extractor_results=[],
        aggregate_metrics={},
        notes=[],
    )

    payload = summary.to_dict()

    assert "notes" not in payload


# --- JSON contract ---


def test_training_run_summary_to_dict_is_json_serializable() -> None:
    """The nested summary payload must round-trip through ``json`` without error,
    locking the contract used by the summary JSON writer."""
    hardware = _make_hardware_profile(with_gpu=True)
    record = _make_run_record(
        config_name="candidate",
        reason="ok",
        metrics={"best_mean_reward": 1.0},
        artifacts={"extractor_dir": "extractors/candidate"},
    )
    summary = TrainingRunSummary(
        run_id="run-json",
        created_at="2026-07-26T00:00:00Z",
        output_root="output/run-json",
        hardware_overview=[hardware],
        extractor_results=[record],
        aggregate_metrics={"mean_best_reward": 1.0},
        notes=["json round trip"],
    )

    encoded = json.dumps(summary.to_dict(), sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded == summary.to_dict()
