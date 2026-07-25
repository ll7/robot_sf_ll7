"""Tests for scripts/dev/quality_report.py (issue #6213)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "docs" / "templates" / "quality_report.schema.json"
MODULE_PATH = REPO_ROOT / "scripts" / "dev" / "quality_report.py"

HEAD_SHA = "a" * 40
BASE_SHA = "b" * 40
BRANCH = "issue-6213-test"


def _load_quality_report_module():
    """Load scripts/dev/quality_report.py without requiring scripts to be a package."""
    spec = importlib.util.spec_from_file_location("quality_report", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def quality_report():
    """Provide a freshly loaded quality_report module."""
    return _load_quality_report_module()


def _build_mock_report(
    quality_report,
    monkeypatch,
    repo_root: Path,
    *,
    require_clean_tree: bool = False,
    tree_state: str = "clean",
):
    """Build a report with deterministic mocked freshness git helpers."""
    monkeypatch.setattr(quality_report.freshness, "_current_branch", lambda: BRANCH)
    monkeypatch.setattr(quality_report.freshness, "_head_sha", lambda: HEAD_SHA)
    monkeypatch.setattr(quality_report.freshness, "_resolve_base_sha", lambda base_ref: BASE_SHA)
    monkeypatch.setattr(quality_report.freshness, "_tree_state", lambda: tree_state)
    return quality_report.build_report(
        base_ref="origin/main",
        require_clean_tree=require_clean_tree,
        repo_root=repo_root,
    )


def test_schema_declares_draft07_and_required_top_level() -> None:
    """The schema is valid JSON declaring draft-07, QualityReport, and required keys."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert schema["title"] == "QualityReport"
    assert set(schema["required"]) == {"schema_version", "stamp", "signals"}


def test_schema_version_constant(quality_report) -> None:
    """The module exposes the v1 schema version constant."""
    assert quality_report.SCHEMA_VERSION == "quality_report.v1"


def test_build_report_assembles_stamp_and_signals(quality_report, monkeypatch, tmp_path) -> None:
    """build_report returns a stamped report with a signals mapping."""
    report = _build_mock_report(quality_report, monkeypatch, tmp_path)
    assert report["schema_version"] == "quality_report.v1"
    stamp = report["stamp"]
    assert stamp["head_sha"] == HEAD_SHA
    assert stamp["base_sha"] == BASE_SHA
    assert stamp["branch"] == BRANCH
    assert stamp["tree_state"] == "clean"
    assert isinstance(report["signals"], dict)


def test_every_defined_signal_present_with_required_fields(
    quality_report, monkeypatch, tmp_path
) -> None:
    """Every SIGNAL_DEFINITIONS key appears with label/gate_class/status."""
    report = _build_mock_report(quality_report, monkeypatch, tmp_path)
    for definition in quality_report.SIGNAL_DEFINITIONS:
        key = definition["key"]
        assert key in report["signals"]
        signal = report["signals"][key]
        assert signal["label"]
        assert signal["gate_class"]
        assert signal["status"]


def test_gate_class_and_status_within_allowed_enums(quality_report, monkeypatch, tmp_path) -> None:
    """gate_class and status stay within their allowed value sets."""
    report = _build_mock_report(quality_report, monkeypatch, tmp_path)
    for signal in report["signals"].values():
        assert signal["gate_class"] in {"required", "diagnostic", "unavailable"}
        assert signal["status"] in {"available", "unavailable", "deferred"}


def test_deferred_and_unavailable_signals_carry_source_gap(
    quality_report, monkeypatch, tmp_path
) -> None:
    """Deferred/unavailable signals record a non-empty source_gap."""
    report = _build_mock_report(quality_report, monkeypatch, tmp_path)
    for key, signal in report["signals"].items():
        if signal["status"] in {"deferred", "unavailable"}:
            assert signal["source_gap"], key


def test_mutation_keeps_categories_separately_visible(
    quality_report, monkeypatch, tmp_path
) -> None:
    """A provided mutation baseline keeps killed/survived/timeout categories visible."""
    baseline_dir = tmp_path / "scripts" / "validation"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "mutation_baseline.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "summary": {
                    "killed": 10,
                    "survived": 2,
                    "skipped": 1,
                    "suspicious": 0,
                    "timeout": 3,
                    "total_mutants": 16,
                },
            }
        ),
        encoding="utf-8",
    )
    report = _build_mock_report(quality_report, monkeypatch, tmp_path)
    mutation = report["signals"]["mutation"]
    assert mutation["status"] == "available"
    categories = mutation["categories"]
    assert categories["killed"] == 10
    assert categories["survived"] == 2
    assert categories["timeout"] == 3
    assert categories["total_mutants"] == 16


def test_no_aggregate_single_score_field(quality_report, monkeypatch, tmp_path) -> None:
    """The report carries no vanity/aggregate score at top level or per signal."""
    report = _build_mock_report(quality_report, monkeypatch, tmp_path)
    forbidden = {"score", "quality_score", "aggregate_score"}
    assert not forbidden.intersection(report.keys())
    for signal in report["signals"].values():
        assert not forbidden.intersection(signal.keys())


def test_require_clean_tree_fails_closed_on_dirty(quality_report, monkeypatch, tmp_path) -> None:
    """A dirty worktree with require_clean_tree returns the fail-closed payload."""
    report = _build_mock_report(
        quality_report, monkeypatch, tmp_path, require_clean_tree=True, tree_state="dirty"
    )
    assert report["ok"] is False
    assert report["reason"] == "dirty_worktree"


def test_report_validates_against_schema(quality_report, monkeypatch, tmp_path) -> None:
    """The built report validates against the JSON Schema when jsonschema is available."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    report = _build_mock_report(quality_report, monkeypatch, tmp_path)
    jsonschema.validate(instance=report, schema=schema)
