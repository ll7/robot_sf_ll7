"""Tests for issue #4366 manuscript-number verification."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/validation/verify_manuscript_asserted_numbers.py"
DECLARATIONS = REPO_ROOT / "configs/validation/issue_4366_manuscript_asserted_numbers.yaml"

SPEC = importlib.util.spec_from_file_location("_issue_4366_verifier", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _MODULE
SPEC.loader.exec_module(_MODULE)


def test_checked_in_declarations_verify_and_report(tmp_path: Path) -> None:
    """Checked-in issue declarations produce the committed match/not-verifiable partition."""
    report_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"

    exit_code = _MODULE.main(
        [
            "--declarations",
            str(DECLARATIONS),
            "--report",
            str(report_path),
            "--json-output",
            str(json_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["overall_status"] == "pass"
    assert report["status_counts"] == {
        "match": 9,
        "mismatch": 0,
        "not_verifiable": 1,
        "blocked": 0,
    }
    text = report_path.read_text(encoding="utf-8")
    assert "AI-GENERATED" in text
    assert "heatmap_per_family_means_source" in text
    assert "`not_verifiable`" in text
    result_rows = [line for line in text.splitlines() if line.startswith("| ")][2:]
    assert len(result_rows) == 10
    assert all(line.count("|") == 8 for line in result_rows)


def test_mismatch_returns_failure_without_silent_source_fix(tmp_path: Path) -> None:
    """A wrong asserted number is reported as mismatch and exits nonzero."""
    payload = yaml.safe_load(DECLARATIONS.read_text(encoding="utf-8"))
    payload["entries"][0]["expected"] = -1
    declarations = tmp_path / "bad.yaml"
    declarations.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    exit_code = _MODULE.main(
        [
            "--declarations",
            str(declarations),
            "--report",
            str(tmp_path / "report.md"),
            "--json-output",
            str(tmp_path / "report.json"),
        ]
    )

    assert exit_code == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["status_counts"]["mismatch"] == 1
    assert report["results"][0]["actual"] == pytest.approx(0.19045845847432735)


def test_missing_source_key_fails_closed(tmp_path: Path) -> None:
    """Missing source-of-record pointers are declaration errors, not not-verifiable guesses."""
    payload = yaml.safe_load(DECLARATIONS.read_text(encoding="utf-8"))
    payload["entries"][0]["source"]["pointer"] = "missing_key"
    declarations = tmp_path / "missing.yaml"
    declarations.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    exit_code = _MODULE.main(
        [
            "--declarations",
            str(declarations),
            "--report",
            str(tmp_path / "report.md"),
        ]
    )

    assert exit_code == 2
