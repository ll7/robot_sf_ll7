"""Regression tests for the importable ``scripts.benchmark`` package."""

from __future__ import annotations

from scripts.benchmark import build_heterogeneous_population_ablation_report as report


def test_report_module_imports_and_main_is_callable(tmp_path, monkeypatch, capsys) -> None:
    """Focused tests can import and invoke benchmark report modules directly.

    This is an import/callability contract, not an input-validation contract:
    ``main`` must return 1 and name the missing input it rejected. Both the
    manifest and the records paths are supplied as nonexistent tmp paths so the
    test does not depend on WHICH required-input check runs first — #5288 and
    #5291 merged three seconds apart and broke main because the original
    version pinned the records-check-first ordering.
    """
    missing_manifest = tmp_path / "missing-manifest.json"
    missing_records = tmp_path / "missing-records.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "report",
            "--manifest",
            str(missing_manifest),
            "--records",
            str(missing_records),
        ],
    )

    assert report.main() == 1
    out = capsys.readouterr().out
    assert str(missing_manifest) in out or str(missing_records) in out
