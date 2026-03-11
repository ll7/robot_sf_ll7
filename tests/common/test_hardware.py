"""Tests for hardware-capacity helpers used by training workflows."""

from __future__ import annotations

import pytest

from robot_sf.common.hardware import (
    HardwareCapacity,
    detect_hardware_capacity,
    recommend_env_runners,
)


def test_detect_hardware_capacity_prefers_slurm_cpu_and_gpu(monkeypatch):
    """SLURM allocation metadata should override host-wide CPU/GPU defaults."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "24")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "gpu:a30:1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    capacity = detect_hardware_capacity()

    assert capacity.allocated_cpus == 24
    assert capacity.usable_cpus == 24
    assert capacity.allocated_gpus == 1
    assert capacity.visible_gpus == 1


def test_detect_hardware_capacity_respects_cuda_visible_devices(monkeypatch):
    """CUDA visibility should cap visible GPUs independently from allocation metadata."""
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "gpu:a30:4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")

    capacity = detect_hardware_capacity()

    assert capacity.allocated_gpus == 4
    assert capacity.visible_gpus == 2


def test_detect_hardware_capacity_counts_single_cuda_device_ids(monkeypatch):
    """Single CUDA device-id tokens should each represent one visible GPU."""
    monkeypatch.delenv("SLURM_GPUS_ON_NODE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    capacity = detect_hardware_capacity()

    assert capacity.visible_gpus == 1


def test_detect_hardware_capacity_counts_multiple_cuda_device_ids(monkeypatch):
    """Comma-separated CUDA device ids should count by token, not numeric value."""
    monkeypatch.delenv("SLURM_GPUS_ON_NODE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    capacity = detect_hardware_capacity()

    assert capacity.visible_gpus == 2


def test_detect_hardware_capacity_preserves_slurm_numeric_gpu_counts(monkeypatch):
    """Numeric Slurm allocation tokens should still be interpreted as counts."""
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "4")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    capacity = detect_hardware_capacity()

    assert capacity.allocated_gpus == 4
    assert capacity.visible_gpus == 4


def test_detect_hardware_capacity_handles_cuda_disabled_token(monkeypatch):
    """CUDA disabled token should force zero visible GPUs."""
    monkeypatch.delenv("SLURM_GPUS_ON_NODE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    capacity = detect_hardware_capacity()

    assert capacity.visible_gpus == 0


def test_detect_hardware_capacity_parses_cuda_device_range(monkeypatch):
    """CUDA range tokens should expand to inclusive device counts."""
    monkeypatch.delenv("SLURM_GPUS_ON_NODE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0-3")

    capacity = detect_hardware_capacity()

    assert capacity.visible_gpus == 4


def test_detect_hardware_capacity_rejects_invalid_cpu_parameters():
    """Invalid reserve/minimum CPU parameters should raise clear ValueError."""
    with pytest.raises(ValueError, match="minimum_cpus must be >= 1"):
        detect_hardware_capacity(minimum_cpus=0)

    with pytest.raises(ValueError, match="reserve_cpu_cores must be >= 0"):
        detect_hardware_capacity(reserve_cpu_cores=-1)


def test_recommend_env_runners_rejects_negative_headroom():
    """Negative headroom should be rejected."""
    capacity = HardwareCapacity(
        logical_cpus=8,
        allocated_cpus=8,
        usable_cpus=8,
        allocated_gpus=0,
        visible_gpus=0,
    )
    with pytest.raises(ValueError, match="cpu_headroom must be >= 0"):
        recommend_env_runners(capacity, cpu_headroom=-1)


def test_recommend_env_runners_leaves_headroom():
    """Runner recommendation should keep CPU headroom for learner/runtime overhead."""
    capacity = HardwareCapacity(
        logical_cpus=24,
        allocated_cpus=24,
        usable_cpus=24,
        allocated_gpus=1,
        visible_gpus=1,
    )
    runners = recommend_env_runners(capacity, cpu_headroom=4)
    assert runners == 20
