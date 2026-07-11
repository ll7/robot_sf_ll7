"""Regression tests for the importable ``scripts.benchmark`` package."""

from __future__ import annotations

from scripts.benchmark import build_heterogeneous_population_ablation_report as report


def test_report_module_imports_and_main_is_callable(tmp_path, monkeypatch, capsys) -> None:
    """Focused tests can import and invoke benchmark report modules directly."""
    missing_records = tmp_path / "missing-records.jsonl"
    monkeypatch.setattr("sys.argv", ["report", "--records", str(missing_records)])

    assert report.main() == 1
    assert str(missing_records) in capsys.readouterr().out
