"""Deterministic admission and cleanup at the browser and child boundaries."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("version", ["v18.20.0", "v22.20.0"])
def test_browser_runtime_checks_supported_version(monkeypatch, version):
    from tests.support.browser_runtime import require_node_runtime

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, version + "\n", "")
    )
    if version.startswith("v18"):
        with pytest.raises(pytest.skip.Exception, match="requires Node >= 22"):
            require_node_runtime()
    else:
        assert require_node_runtime() == version


def test_child_completion_and_hang_cleanup(tmp_path):
    from tests.support.subprocess_events import capture_completion

    result = capture_completion([sys.executable, "-c", 'print("ready")'])
    assert result.returncode == 0 and result.stdout.strip() == "ready"
    with pytest.raises(subprocess.TimeoutExpired):
        capture_completion(
            [sys.executable, "-c", "import time; time.sleep(120)"], hang_guard_seconds=0.1
        )
