"""Tests for the CUDA readiness receipt CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

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
