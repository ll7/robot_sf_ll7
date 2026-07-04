"""Tests for issue #4366 manuscript-number verification."""

from __future__ import annotations

import hashlib
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


def _write_locator_declaration(
    tmp_path: Path, *, recorded_sha256: str, expected_sha256: str
) -> Path:
    table = tmp_path / "table.md"
    table.write_text(
        "| planner_key | success_mean |\n| --- | --- |\n| ppo | 1.0 |\n", encoding="utf-8"
    )
    locator = tmp_path / "locator.yaml"
    locator.write_text(
        yaml.safe_dump(
            {
                "heatmap_per_family_means_locator": {
                    "artifact_id": "fixture_table",
                    "table_path": str(table),
                    "table_sha256": recorded_sha256,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    declarations = tmp_path / "declarations.yaml"
    declarations.write_text(
        yaml.safe_dump(
            {
                "schema_version": "manuscript-asserted-number-declarations.v1",
                "issue": 4366,
                "claim_boundary": "fixture locator hash validation only",
                "selection_assumption": "fixture",
                "entries": [
                    {
                        "id": "heatmap_per_family_means_source",
                        "manuscript_locator": "fixture / heatmap per-family means",
                        "expected": {
                            "artifact_id": "fixture_table",
                            "table_path": str(table),
                            "table_sha256": expected_sha256,
                        },
                        "source": {
                            "path": str(locator),
                            "pointer": "heatmap_per_family_means_locator",
                        },
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return declarations


def test_checked_in_declarations_verify_and_report(tmp_path: Path) -> None:
    """Checked-in issue declarations produce the committed verified partition."""
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
        "match": 10,
        "mismatch": 0,
        "not_verifiable": 0,
        "blocked": 0,
    }
    text = report_path.read_text(encoding="utf-8")
    assert "AI-GENERATED" in text
    assert "heatmap_per_family_means_source" in text
    assert "issue_4366_heatmap_per_family_means_source_locator.yaml" in text
    assert '"evidence_status": "locator_verified_only"' in text
    assert "marker-placement placeholder" in text
    assert "Stable locator declared" in text
    heatmap = next(
        result for result in report["results"] if result["id"] == "heatmap_per_family_means_source"
    )
    assert heatmap["source_locator_status"] == "match"
    assert heatmap["actual"]["artifact_id"] == "tab_issue_1023_scenario_family_breakdown"
    result_rows = [line for line in text.splitlines() if line.startswith("| ")][2:]
    assert len(result_rows) == 10
    assert all(line.count("|") == 9 for line in result_rows)


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


def test_locator_table_sha256_validates_against_recorded_artifact(tmp_path: Path) -> None:
    """Locator table rows hash the recorded artifact instead of comparing expected copies."""
    table_content = "| planner_key | success_mean |\n| --- | --- |\n| ppo | 1.0 |\n"
    table_sha256 = hashlib.sha256(table_content.encode("utf-8")).hexdigest()
    declarations = _write_locator_declaration(
        tmp_path,
        recorded_sha256=table_sha256,
        expected_sha256="0" * 64,
    )

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

    assert exit_code == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    result = report["results"][0]
    assert result["status"] == "match"
    assert result["expected"]["table_sha256"] == "0" * 64
    assert result["actual"]["table_sha256"] == table_sha256
    assert result["actual"]["computed_table_sha256"] == table_sha256


def test_locator_table_sha256_mismatch_fails(tmp_path: Path) -> None:
    """Locator table rows fail when recorded table_sha256 does not match the artifact."""
    declarations = _write_locator_declaration(
        tmp_path,
        recorded_sha256="0" * 64,
        expected_sha256="0" * 64,
    )

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
    result = report["results"][0]
    assert result["status"] == "mismatch"
    assert result["reason"] == "locator table_sha256 differs from table artifact"
    assert result["actual"]["table_sha256"] == "0" * 64
    assert result["actual"]["computed_table_sha256"] != "0" * 64


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
