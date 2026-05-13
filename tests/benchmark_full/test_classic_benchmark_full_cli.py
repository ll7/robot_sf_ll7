"""CLI behavior tests for the classic full benchmark wrapper."""

from __future__ import annotations

from scripts import classic_benchmark_full


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
