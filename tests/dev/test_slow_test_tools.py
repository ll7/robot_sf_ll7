"""Process-boundary contracts for the slow-test capture utilities."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.collect_slow_tests import SlowTestCollectionError, parse
from scripts.compare_slow_tests import Sample, SlowTestCaptureError, index_by, load_any

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECT_SCRIPT = REPO_ROOT / "scripts" / "collect_slow_tests.py"
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_slow_tests.py"


def _run(
    script: Path, *args: str, input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run one utility with the repository interpreter and capture its streams."""
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=REPO_ROOT,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_json(path: Path, payload: object) -> None:
    """Write a JSON fixture, retaining Python's non-standard NaN spellings for rejection tests."""
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")


def test_collect_stdin_collapses_phases_and_sorts_by_max_duration() -> None:
    """Valid pytest output keeps max phase duration and ignores unrelated lines."""
    result = _run(
        COLLECT_SCRIPT,
        input_text=(
            "random pytest output\n"
            "0.250s setup tests/test_demo.py::test_fast\n"
            "1.500s call tests/test_demo.py::test_fast\n"
            "0.100s teardown tests/test_demo.py::test_fast\n"
            "0.750s call tests/test_demo.py::test_slow\n"
        ),
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert [
        (row["test_identifier"], row["duration_seconds"]) for row in json.loads(result.stdout)
    ] == [
        ("tests/test_demo.py::test_fast", 1.5),
        ("tests/test_demo.py::test_slow", 0.75),
    ]


def test_collect_file_input_preserves_valid_contract(tmp_path: Path) -> None:
    """The explicit input-file path produces the same structured capture as stdin."""
    source = tmp_path / "pytest-durations.log"
    source.write_text("2.0s call tests/test_demo.py::test_file\n", encoding="utf-8")

    result = _run(COLLECT_SCRIPT, "--input", str(source))

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload[0]["test_identifier"] == "tests/test_demo.py::test_file"
    assert payload[0]["duration_seconds"] == 2.0


def test_collect_missing_input_is_one_clean_diagnostic(tmp_path: Path) -> None:
    """A missing collector input exits non-zero without leaking a traceback."""
    result = _run(COLLECT_SCRIPT, "--input", str(tmp_path / "missing.log"))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("error: unable to read input file '")
    assert "Traceback" not in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1


def test_collect_rejects_overflowing_duration_line() -> None:
    """A matched duration that would serialize as infinity fails closed."""
    with pytest.raises(SlowTestCollectionError, match="duration_seconds"):
        parse([f"{'9' * 400}s call tests/test_demo.py::test_overflow"])


def test_compare_accepts_list_and_samples_wrapper(tmp_path: Path) -> None:
    """Both documented capture shapes remain accepted and produce the report."""
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    _write_json(
        before,
        [{"test_identifier": "tests/test_demo.py::test_a", "duration_seconds": 1.0}],
    )
    _write_json(
        after,
        {"samples": [{"test_identifier": "tests/test_demo.py::test_a", "duration_seconds": 2}]},
    )

    result = _run(COMPARE_SCRIPT, "--before", str(before), "--after", str(after))

    assert result.returncode == 0
    assert result.stderr == ""
    assert "Compared 1 common tests" in result.stdout
    assert "tests/test_demo.py::test_a: +1.000s" in result.stdout


def test_compare_missing_capture_is_one_clean_diagnostic(tmp_path: Path) -> None:
    """A missing comparison capture is reported at the process boundary."""
    valid = tmp_path / "valid.json"
    _write_json(valid, [])

    result = _run(
        COMPARE_SCRIPT,
        "--before",
        str(tmp_path / "missing.json"),
        "--after",
        str(valid),
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "unable to read file" in result.stderr
    assert "Traceback" not in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1


def test_compare_invalid_json_is_one_clean_diagnostic(tmp_path: Path) -> None:
    """Invalid JSON is converted to a location-aware validation error."""
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text("{not json", encoding="utf-8")
    _write_json(after, [])

    result = _run(COMPARE_SCRIPT, "--before", str(before), "--after", str(after))

    assert result.returncode == 2
    assert "invalid JSON" in result.stderr
    assert "line 1" in result.stderr
    assert "Traceback" not in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"metadata": {}}, "top level must be a list"),
        ({"samples": {"test_identifier": "x"}}, "field 'samples' must be a list"),
        (42, "top level must be a list"),
        ({"samples": ["not an object"]}, "sample 0 must be an object"),
        ({"samples": [{"duration_seconds": 1.0}]}, "missing required field 'test_identifier'"),
        ({"samples": [{"test_identifier": "", "duration_seconds": 1.0}]}, "non-empty string"),
        ({"samples": [{"test_identifier": "  ", "duration_seconds": 1.0}]}, "non-empty string"),
        ({"samples": [{"test_identifier": "x"}]}, "missing required field 'duration_seconds'"),
        (
            {"samples": [{"test_identifier": "x", "duration_seconds": "1.0"}]},
            "finite, non-negative number",
        ),
        (
            {"samples": [{"test_identifier": "x", "duration_seconds": True}]},
            "booleans and strings are not accepted",
        ),
        (
            {"samples": [{"test_identifier": "x", "duration_seconds": -1.0}]},
            "finite, non-negative number",
        ),
        (
            {"samples": [{"test_identifier": "x", "duration_seconds": float("nan")}]},
            "finite, non-negative number",
        ),
        (
            {"samples": [{"test_identifier": "x", "duration_seconds": float("inf")}]},
            "finite, non-negative number",
        ),
        (
            {
                "samples": [
                    {"test_identifier": "x", "duration_seconds": 1.0},
                    {"test_identifier": "x", "duration_seconds": 2.0},
                ]
            },
            "duplicate test_identifier",
        ),
    ],
)
def test_compare_rejects_malformed_capture(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    """Every malformed top-level, row, and field shape fails closed with one message."""
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    _write_json(before, payload)
    _write_json(after, [])

    result = _run(COMPARE_SCRIPT, "--before", str(before), "--after", str(after))

    assert result.returncode == 2
    assert message in result.stderr
    assert "sample" in result.stderr or "top level" in result.stderr or "field" in result.stderr
    assert "Traceback" not in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1


def test_load_any_rejects_duplicates_before_indexing(tmp_path: Path) -> None:
    """The Python API has the same duplicate policy as the CLI boundary."""
    path = tmp_path / "duplicate.json"
    _write_json(
        path,
        [
            {"test_identifier": "x", "duration_seconds": 1.0},
            {"test_identifier": "x", "duration_seconds": 1.0},
        ],
    )

    with pytest.raises(SlowTestCaptureError, match="duplicate test_identifier"):
        load_any(path)


def test_index_by_rejects_programmatic_duplicates() -> None:
    """Callers bypassing JSON loading cannot reintroduce last-write-wins behavior."""
    with pytest.raises(SlowTestCaptureError, match="duplicate test_identifier"):
        index_by([Sample("x", 1.0), Sample("x", 2.0)])
