"""TODO docstring. Document this module."""

import platform
import sys
from types import SimpleNamespace

import pytest

from robot_sf.training import hardware_probe
from robot_sf.training.hardware_probe import collect_hardware_profile


@pytest.mark.parametrize("worker_count", [1, 4])
def test_collect_hardware_profile_cpu_only(monkeypatch, worker_count):
    """TODO docstring. Document this function.

    Args:
        monkeypatch: TODO docstring.
        worker_count: TODO docstring.
    """
    monkeypatch.setattr(hardware_probe, "_collect_gpu_metadata", lambda: (None, None))

    profile = collect_hardware_profile(worker_count=worker_count)

    assert isinstance(profile.platform, str) and profile.platform
    assert profile.platform.startswith(platform.platform().split("-", 1)[0])
    assert profile.arch == platform.machine()
    assert profile.python_version.startswith(str(sys.version_info.major))
    assert profile.workers == worker_count
    assert profile.gpu_model is None
    assert profile.cuda_version is None


def test_collect_hardware_profile_with_gpu(monkeypatch):
    """TODO docstring. Document this function.

    Args:
        monkeypatch: TODO docstring.
    """
    monkeypatch.setattr(hardware_probe, "_collect_gpu_metadata", lambda: ("Fake GPU", "12.3"))

    profile = collect_hardware_profile(worker_count=2)

    assert profile.gpu_model == "Fake GPU"
    assert profile.cuda_version == "12.3"
    assert profile.workers == 2


def test_collect_hardware_profile_rejects_invalid_worker_count():
    """Test that worker counts must be positive."""
    with pytest.raises(ValueError, match="worker_count must be >= 1"):
        collect_hardware_profile(worker_count=0)


def test_collect_hardware_profile_skip_gpu(monkeypatch):
    """Test that skip_gpu avoids the CUDA metadata probe."""

    def fail_probe():
        """Fail if GPU probing is unexpectedly invoked."""
        raise AssertionError("GPU metadata should not be collected")

    monkeypatch.setattr(hardware_probe, "_collect_gpu_metadata", fail_probe)

    profile = collect_hardware_profile(worker_count=3, skip_gpu=True)

    assert profile.workers == 3
    assert profile.gpu_model is None
    assert profile.cuda_version is None


def test_collect_gpu_metadata_without_torch(monkeypatch):
    """Test GPU metadata collection when torch cannot be imported."""
    monkeypatch.setattr(hardware_probe, "_load_torch", lambda: None)

    assert hardware_probe._collect_gpu_metadata() == (None, None)


def test_collect_gpu_metadata_cpu_only(monkeypatch):
    """Test GPU metadata collection when CUDA is unavailable."""
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(hardware_probe, "_load_torch", lambda: fake_torch)

    assert hardware_probe._collect_gpu_metadata() == (None, None)


def test_collect_gpu_metadata_cuda(monkeypatch):
    """Test GPU metadata collection when CUDA metadata is available."""
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 1,
        get_device_name=lambda device_index: f"Fake GPU {device_index}",
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda, version=SimpleNamespace(cuda="12.8"))
    monkeypatch.setattr(hardware_probe, "_load_torch", lambda: fake_torch)

    assert hardware_probe._collect_gpu_metadata() == ("Fake GPU 1", "12.8")


def test_collect_gpu_metadata_cuda_without_version_module(monkeypatch):
    """Test GPU metadata collection when torch.version is unavailable."""
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_name=lambda _device_index: "Versionless GPU",
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setattr(hardware_probe, "_load_torch", lambda: fake_torch)

    assert hardware_probe._collect_gpu_metadata() == ("Versionless GPU", None)


def test_collect_gpu_metadata_import_runtime_failure(monkeypatch):
    """Test GPU metadata collection when importing torch raises at runtime."""

    def fail_load():
        """Simulate torch import failing during CUDA initialization."""
        raise RuntimeError("CUDA runtime initialization failed")

    monkeypatch.setattr(hardware_probe, "_load_torch", fail_load)

    assert hardware_probe._collect_gpu_metadata() == (None, None)


def test_collect_gpu_metadata_device_runtime_failure(monkeypatch):
    """Test GPU metadata collection when querying the CUDA device fails."""

    def fail_device():
        """Simulate CUDA device lookup failing at runtime."""
        raise RuntimeError("device unavailable")

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=fail_device,
        get_device_name=lambda _device_index: "unreachable",
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda, version=SimpleNamespace(cuda="12.8"))
    monkeypatch.setattr(hardware_probe, "_load_torch", lambda: fake_torch)

    assert hardware_probe._collect_gpu_metadata() == (None, None)
