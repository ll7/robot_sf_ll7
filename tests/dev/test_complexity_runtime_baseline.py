"""Tests for the lightweight complexity and runtime baseline command."""

from __future__ import annotations

import json

from scripts.dev.complexity_runtime_baseline import (
    build_baseline,
    format_text_report,
    parse_pytest_durations,
)


def test_build_baseline_ranks_modules_and_functions(tmp_path) -> None:
    """Baseline scan should rank source files and functions by simple size metrics."""
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "small.py").write_text(
        "def tiny():\n    return 1\n",
        encoding="utf-8",
    )
    (source_root / "large.py").write_text(
        "class Example:\n"
        "    def long_method(self):\n"
        "        value = 0\n"
        "        value += 1\n"
        "        value += 2\n"
        "        return value\n",
        encoding="utf-8",
    )

    baseline = build_baseline([source_root], top=2)

    assert [module.path.name for module in baseline.modules] == ["large.py", "small.py"]
    assert baseline.functions[0].qualified_name == "Example.long_method"
    assert baseline.functions[0].length_lines == 5


def test_parse_pytest_durations_extracts_runtime_samples() -> None:
    """Pytest duration parsing should preserve phase, node id, and rank by wall time."""
    samples = parse_pytest_durations(
        "============================= slowest 10 durations =============================\n"
        "27.40s call     tests/examples/test_examples_run.py::test_image\n"
        "1.61s setup    tests/carla_bridge/test_t0_export_cli.py::test_schema\n"
    )

    assert [sample.nodeid for sample in samples] == [
        "tests/examples/test_examples_run.py::test_image",
        "tests/carla_bridge/test_t0_export_cli.py::test_schema",
    ]
    assert samples[0].duration_seconds == 27.40
    assert samples[0].phase == "call"


def test_format_text_report_includes_complexity_and_runtime_sections(tmp_path) -> None:
    """Text output should be useful as a copyable context-note summary."""
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "sample.py").write_text(
        "def example():\n    return 'ok'\n",
        encoding="utf-8",
    )
    pytest_log = tmp_path / "pytest.log"
    pytest_log.write_text(
        "3.25s call     tests/test_sample.py::test_example\n",
        encoding="utf-8",
    )

    baseline = build_baseline([source_root], top=1, pytest_log=pytest_log)
    report = format_text_report(baseline)

    assert "Largest modules" in report
    assert "Longest functions" in report
    assert "Test runtime indicators" in report
    assert "sample.py" in report
    assert "tests/test_sample.py::test_example" in report
    assert json.loads(baseline.to_json())["pytest_durations"][0]["phase"] == "call"
