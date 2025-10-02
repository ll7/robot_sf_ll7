import platform
import sys

import pytest

from robot_sf.training import hardware_probe
from robot_sf.training.hardware_probe import collect_hardware_profile


@pytest.mark.parametrize("worker_count", [1, 4])
def test_collect_hardware_profile_cpu_only(monkeypatch, worker_count):
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
    monkeypatch.setattr(hardware_probe, "_collect_gpu_metadata", lambda: ("Fake GPU", "12.3"))

    profile = collect_hardware_profile(worker_count=2)

    assert profile.gpu_model == "Fake GPU"
    assert profile.cuda_version == "12.3"
    assert profile.workers == 2
