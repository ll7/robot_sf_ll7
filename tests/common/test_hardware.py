"""Tests for hardware-capacity helpers used by training workflows."""

from __future__ import annotations

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
