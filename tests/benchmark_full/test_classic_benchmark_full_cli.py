"""CLI behavior tests for the classic full benchmark wrapper."""

from __future__ import annotations

import pytest

from scripts import classic_benchmark_full


def test_classic_benchmark_full_help_exits_before_backend_execution(monkeypatch, capsys) -> None:
    """``--help`` should be usable without running benchmark logic."""

    def fail_if_called(_cfg):
        raise AssertionError("run_full_benchmark should not run for --help")

    monkeypatch.setattr(classic_benchmark_full, "run_full_benchmark", fail_if_called)

    with pytest.raises(SystemExit) as exc_info:
        classic_benchmark_full.main(["--help"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "Full Classic Interaction Benchmark" in captured.out


def test_classic_benchmark_full_delegates_to_backend(monkeypatch, tmp_path) -> None:
    """Supported invocation should call the real benchmark backend contract."""
    captured = {}

    def fake_backend(cfg):
        captured["cfg"] = cfg

    monkeypatch.setattr(classic_benchmark_full, "run_full_benchmark", fake_backend)

    exit_code = classic_benchmark_full.main(
        [
            "--output",
            str(tmp_path / "classic-full"),
            "--smoke",
            "--workers",
            "1",
        ]
    )

    assert exit_code == 0
    assert captured["cfg"].output_root == str(tmp_path / "classic-full")
    assert captured["cfg"].smoke is True
    assert captured["cfg"].workers == 1


def test_classic_benchmark_full_unavailable_backend_is_actionable(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    """Unavailable backend exits nonzero with an actionable error and no traceback."""

    def unavailable_backend(_cfg):
        raise classic_benchmark_full.FullBenchmarkUnavailableError("backend import missing")

    monkeypatch.setattr(classic_benchmark_full, "run_full_benchmark", unavailable_backend)

    exit_code = classic_benchmark_full.main(
        [
            "--output",
            str(tmp_path / "classic-full"),
            "--smoke",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "backend import missing" in captured.err
    assert "Traceback" not in captured.err
