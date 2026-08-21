"""Tests for the CUDA readiness receipt CLI and runtime classification."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from robot_sf.telemetry import gpu as gpu_module
from robot_sf.telemetry.gpu import CudaRuntimeClass
from scripts.dev import check_cuda_runtime


def test_json_output_is_one_machine_readable_document(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """The JSON mode keeps receipt diagnostics off stdout."""
    monkeypatch.setattr(
        check_cuda_runtime,
        "classify_cuda_runtime",
        lambda: CudaRuntimeClass("unavailable", "torch.cuda.is_available() is False"),
    )

    def fake_run_git(args: list[str]) -> str:
        if args == ["branch", "--show-current"]:
            return "test/cuda-runtime"
        if args == ["rev-parse", "HEAD"]:
            return "abc123"
        return ""

    monkeypatch.setattr(check_cuda_runtime, "_run_git", fake_run_git)

    assert check_cuda_runtime.main(["--receipt-dir", str(tmp_path), "--json"]) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["schema"] == "cuda_runtime_readiness.v1"
    assert payload["status"] == "unavailable"
    assert "Receipt written:" in captured.err
    assert (tmp_path / "cuda_runtime_test-cuda-runtime.json").is_file()


def _stub_import(monkeypatch: pytest.MonkeyPatch, module) -> None:
    """Replace importlib.import_module with a stub returning *module*."""

    def fake_import(name: str):
        assert name == "torch"
        if isinstance(module, Exception):
            raise module
        return module

    monkeypatch.setattr(gpu_module.importlib, "import_module", fake_import)


def test_classify_unavailable_when_torch_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing/broken torch import classifies as unavailable."""
    _stub_import(monkeypatch, ModuleNotFoundError("No module named 'torch'"))

    result = gpu_module.classify_cuda_runtime()

    assert result.status == "unavailable"
    assert "torch unavailable" in result.reason
    assert not result.usable


def test_classify_unavailable_when_cuda_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch present but cuda.is_available() False classifies as unavailable."""
    torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    _stub_import(monkeypatch, torch_stub)

    result = gpu_module.classify_cuda_runtime()

    assert result.status == "unavailable"
    assert "is_available() is False" in result.reason
    assert not result.usable


def test_classify_unavailable_when_cuda_probe_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception inside the cuda probe classifies as unavailable."""
    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: (_ for _ in ()).throw(RuntimeError("driver issue"))
        )
    )
    _stub_import(monkeypatch, torch_stub)

    result = gpu_module.classify_cuda_runtime()

    assert result.status == "unavailable"
    assert "torch.cuda probe failed" in result.reason


def test_classify_unusable_nvml_on_device_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """An NVML-named failure in the real device op classifies as unusable_nvml."""
    calls: list[str] = []

    def raise_nvml():
        calls.append("device_op")
        raise RuntimeError("NVML_ERROR_DRIVER_NOT_LOADED: driver not loaded")

    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: None,
        ),
        zeros=lambda *args, **kwargs: raise_nvml(),
    )
    _stub_import(monkeypatch, torch_stub)

    result = gpu_module.classify_cuda_runtime()

    assert result.status == "unusable_nvml"
    assert "NVML/driver" in result.reason
    assert calls == ["device_op"]


def test_classify_unusable_nvml_on_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-NVML runtime failure in the device op classifies as unusable_nvml."""
    calls: list[str] = []

    def raise_runtime():
        calls.append("device_op")
        raise RuntimeError("invalid device ordinal")

    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: None,
        ),
        zeros=lambda *args, **kwargs: raise_runtime(),
    )
    _stub_import(monkeypatch, torch_stub)

    result = gpu_module.classify_cuda_runtime()

    assert result.status == "unusable_nvml"
    assert "runtime error" in result.reason
    assert calls == ["device_op"]


def test_classify_usable_after_real_device_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful allocation plus synchronize classifies as usable."""
    calls: list[str] = []

    def fake_zeros(*args, **kwargs):
        calls.append("zeros")
        return object()

    def fake_sync():
        calls.append("synchronize")

    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, synchronize=fake_sync),
        zeros=fake_zeros,
    )
    _stub_import(monkeypatch, torch_stub)

    result = gpu_module.classify_cuda_runtime()

    assert result.usable
    assert result.status == "usable"
    assert result.reason == "real CUDA device operation succeeded"
    assert calls == ["zeros", "synchronize"]


def test_cuda_runtime_class_to_dict_roundtrip() -> None:
    """The JSON-ready representation carries status and reason."""
    classification = CudaRuntimeClass("unavailable", "torch unavailable: x")

    assert classification.to_dict() == {"status": "unavailable", "reason": "torch unavailable: x"}
    assert not classification.usable
