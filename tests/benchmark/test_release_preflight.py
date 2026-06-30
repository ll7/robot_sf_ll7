"""Fail-closed tests for the release preflight checklist (issue #3081).

These tests prove that each missing or invalid release prerequisite makes the
preflight report ``blocked`` rather than passing, and that a fully-satisfied
synthetic checklist passes. They also smoke-check the shipped July 2026 checklist
loads and evaluates (honestly ``blocked`` against the current checkout).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.release_preflight import (
    SCHEMA_VERSION,
    ReleasePreflightError,
    evaluate_release_preflight,
    load_release_preflight_checklist,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_CHECKLIST = (
    REPO_ROOT / "configs/benchmarks/releases/release_july_2026_preflight_issue_3081.yaml"
)


def _write(path: Path, payload: object) -> None:
    """Write a YAML/JSON payload, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _checklist(items: list[dict]) -> dict:
    """Build a minimal valid checklist payload around the given items."""
    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": "synthetic_release",
        "description": "synthetic",
        "items": items,
    }


def _load_and_eval(tmp_path: Path, payload: dict) -> dict:
    """Persist, load, and evaluate a checklist against ``tmp_path`` as repo root."""
    checklist_path = tmp_path / "checklist.yaml"
    _write(checklist_path, payload)
    checklist = load_release_preflight_checklist(checklist_path)
    return evaluate_release_preflight(checklist, tmp_path)


# --- loader fail-closed cases ------------------------------------------------


def test_loader_rejects_bad_schema_version(tmp_path: Path) -> None:
    """An unsupported schema_version fails closed at load."""
    payload = _checklist([])
    payload["schema_version"] = "release_preflight_checklist.v0"
    path = tmp_path / "c.yaml"
    _write(path, payload)
    with pytest.raises(ReleasePreflightError, match="schema_version"):
        load_release_preflight_checklist(path)


def test_loader_rejects_unknown_check(tmp_path: Path) -> None:
    """An unknown check kind is rejected at load."""
    payload = _checklist([{"item_id": "a", "criterion": "reproduction", "check": "nope"}])
    path = tmp_path / "c.yaml"
    _write(path, payload)
    with pytest.raises(ReleasePreflightError, match="unknown check"):
        load_release_preflight_checklist(path)


def test_loader_rejects_unknown_criterion(tmp_path: Path) -> None:
    """An unknown criterion bucket is rejected at load."""
    payload = _checklist([{"item_id": "a", "criterion": "bogus", "check": "artifact_present"}])
    path = tmp_path / "c.yaml"
    _write(path, payload)
    with pytest.raises(ReleasePreflightError, match="unknown criterion"):
        load_release_preflight_checklist(path)


def test_loader_rejects_duplicate_item_id(tmp_path: Path) -> None:
    """Duplicate item ids are rejected at load."""
    item = {"item_id": "dup", "criterion": "reproduction", "check": "artifact_present"}
    payload = _checklist([dict(item), dict(item)])
    path = tmp_path / "c.yaml"
    _write(path, payload)
    with pytest.raises(ReleasePreflightError, match="Duplicate item_id"):
        load_release_preflight_checklist(path)


def test_loader_rejects_empty_items(tmp_path: Path) -> None:
    """An empty items list is rejected at load."""
    path = tmp_path / "c.yaml"
    _write(path, _checklist([]))
    with pytest.raises(ReleasePreflightError, match="non-empty items"):
        load_release_preflight_checklist(path)


# --- artifact_present --------------------------------------------------------


def test_artifact_present_complete(tmp_path: Path) -> None:
    """A present regular file resolves to complete/passed."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/repro.md").write_text("ok", encoding="utf-8")
    result = _load_and_eval(
        tmp_path,
        _checklist(
            [
                {
                    "item_id": "repro",
                    "criterion": "reproduction",
                    "check": "artifact_present",
                    "path": "docs/repro.md",
                }
            ]
        ),
    )
    assert result["status"] == "passed"
    assert result["items"][0]["status"] == "complete"


def test_artifact_present_missing_fails_closed(tmp_path: Path) -> None:
    """A missing artifact fails closed to blocked."""
    result = _load_and_eval(
        tmp_path,
        _checklist(
            [
                {
                    "item_id": "repro",
                    "criterion": "reproduction",
                    "check": "artifact_present",
                    "path": "docs/repro.md",
                }
            ]
        ),
    )
    assert result["status"] == "blocked"
    assert "missing" in result["items"][0]["gaps"][0]


def test_artifact_present_rejects_output_path(tmp_path: Path) -> None:
    """A worktree-local output/ path fails closed."""
    (tmp_path / "output").mkdir()
    (tmp_path / "output/repro.md").write_text("ok", encoding="utf-8")
    result = _load_and_eval(
        tmp_path,
        _checklist(
            [
                {
                    "item_id": "repro",
                    "criterion": "reproduction",
                    "check": "artifact_present",
                    "path": "output/repro.md",
                }
            ]
        ),
    )
    assert result["status"] == "blocked"
    assert "output/" in result["items"][0]["gaps"][0]


def test_artifact_present_rejects_symlink(tmp_path: Path) -> None:
    """A symlinked artifact fails closed."""
    (tmp_path / "real.md").write_text("ok", encoding="utf-8")
    (tmp_path / "link.md").symlink_to(tmp_path / "real.md")
    result = _load_and_eval(
        tmp_path,
        _checklist(
            [
                {
                    "item_id": "repro",
                    "criterion": "reproduction",
                    "check": "artifact_present",
                    "path": "link.md",
                }
            ]
        ),
    )
    assert result["status"] == "blocked"
    assert "symlink" in result["items"][0]["gaps"][0]


def test_artifact_present_rejects_absolute_path(tmp_path: Path) -> None:
    """An absolute path is rejected and fails closed."""
    result = _load_and_eval(
        tmp_path,
        _checklist(
            [
                {
                    "item_id": "repro",
                    "criterion": "reproduction",
                    "check": "artifact_present",
                    "path": "/etc/hostname",
                }
            ]
        ),
    )
    assert result["status"] == "blocked"
    assert "absolute" in result["items"][0]["gaps"][0]


# --- checksum_manifest -------------------------------------------------------


def _manifest_item(path: str = "manifest.json") -> dict:
    """Build a checksum_manifest checklist item for the given manifest path."""
    return {
        "item_id": "tables",
        "criterion": "tables_figures",
        "check": "checksum_manifest",
        "path": path,
    }


def test_checksum_manifest_matches(tmp_path: Path) -> None:
    """A manifest whose entries match their digests passes."""
    (tmp_path / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    digest = hashlib.sha256((tmp_path / "table.csv").read_bytes()).hexdigest()
    _write(
        tmp_path / "manifest.json",
        {"entries": [{"path": "table.csv", "sha256": digest}]},
    )
    result = _load_and_eval(tmp_path, _checklist([_manifest_item()]))
    assert result["status"] == "passed"


def test_checksum_manifest_mismatch_fails_closed(tmp_path: Path) -> None:
    """A checksum mismatch fails closed to blocked."""
    (tmp_path / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    _write(
        tmp_path / "manifest.json",
        {"entries": [{"path": "table.csv", "sha256": "deadbeef"}]},
    )
    result = _load_and_eval(tmp_path, _checklist([_manifest_item()]))
    assert result["status"] == "blocked"
    assert "mismatch" in result["items"][0]["gaps"][0]


def test_checksum_manifest_missing_listed_file_fails_closed(tmp_path: Path) -> None:
    """A listed-but-missing file fails closed."""
    _write(
        tmp_path / "manifest.json",
        {"entries": [{"path": "table.csv", "sha256": "abc"}]},
    )
    result = _load_and_eval(tmp_path, _checklist([_manifest_item()]))
    assert result["status"] == "blocked"
    assert "missing" in result["items"][0]["gaps"][0]


# --- claim_audit -------------------------------------------------------------


def _claim_item() -> dict:
    """Build a claim_audit checklist item over claims.yaml."""
    return {
        "item_id": "claims",
        "criterion": "claim_audit",
        "check": "claim_audit",
        "path": "claims.yaml",
    }


def test_claim_audit_native_promoted_passes(tmp_path: Path) -> None:
    """A promoted native-mode claim passes the audit."""
    _write(
        tmp_path / "claims.yaml",
        {"claims": [{"claim_id": "c1", "promoted": True, "planner_mode": "native"}]},
    )
    result = _load_and_eval(tmp_path, _checklist([_claim_item()]))
    assert result["status"] == "passed"


def test_claim_audit_promoted_fallback_fails_closed(tmp_path: Path) -> None:
    """A promoted fallback-mode claim fails closed."""
    _write(
        tmp_path / "claims.yaml",
        {"claims": [{"claim_id": "c1", "promoted": True, "planner_mode": "fallback"}]},
    )
    result = _load_and_eval(tmp_path, _checklist([_claim_item()]))
    assert result["status"] == "blocked"
    assert "fallback" in result["items"][0]["gaps"][0]


def test_claim_audit_promoted_hyphenated_partial_failure_fails_closed(tmp_path: Path) -> None:
    """A promoted hyphenated ``partial-failure`` mode fails closed.

    The canonical exclusion vocabulary lists both spellings; the audit normalizes
    hyphen to underscore so neither escapes.
    """
    _write(
        tmp_path / "claims.yaml",
        {"claims": [{"claim_id": "c1", "promoted": True, "row_status": "partial-failure"}]},
    )
    result = _load_and_eval(tmp_path, _checklist([_claim_item()]))
    assert result["status"] == "blocked"
    assert "partial_failure" in result["items"][0]["gaps"][0]


def test_claim_audit_promoted_non_success_row_status_fails_closed(tmp_path: Path) -> None:
    """A promoted claim on a non-success ``row_status`` fails closed.

    ``excluded`` is in the canonical ``benchmark_row_claim._NON_SUCCESS_STATUSES``
    row-status vocabulary but not in the planner-mode set, so it must still block.
    """
    _write(
        tmp_path / "claims.yaml",
        {"claims": [{"claim_id": "c1", "promoted": True, "row_status": "excluded"}]},
    )
    result = _load_and_eval(tmp_path, _checklist([_claim_item()]))
    assert result["status"] == "blocked"
    assert "excluded" in result["items"][0]["gaps"][0]


def test_claim_audit_checks_both_fields(tmp_path: Path) -> None:
    """A native planner_mode does not mask a non-success row_status."""
    # Both descriptor fields are audited; a success planner_mode must not let a
    # non-success row_status slip through.
    _write(
        tmp_path / "claims.yaml",
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "promoted": True,
                    "planner_mode": "native",
                    "row_status": "blocked",
                }
            ]
        },
    )
    result = _load_and_eval(tmp_path, _checklist([_claim_item()]))
    assert result["status"] == "blocked"
    assert "row_status" in result["items"][0]["gaps"][0]


def test_claim_audit_unpromoted_fallback_is_ignored(tmp_path: Path) -> None:
    """A non-promoted fallback claim does not block."""
    # A non-promoted (diagnostic) claim may carry a fallback mode without blocking.
    _write(
        tmp_path / "claims.yaml",
        {"claims": [{"claim_id": "c1", "promoted": False, "planner_mode": "degraded"}]},
    )
    result = _load_and_eval(tmp_path, _checklist([_claim_item()]))
    assert result["status"] == "passed"


# --- issue_classification_ledger --------------------------------------------


def _ledger_item(required: list[int]) -> dict:
    """Build an issue_classification_ledger item requiring the given issues."""
    return {
        "item_id": "ledger",
        "criterion": "sprint_issue_closure",
        "check": "issue_classification_ledger",
        "path": "ledger.yaml",
        "required_issues": required,
    }


def test_issue_ledger_all_classified_passes(tmp_path: Path) -> None:
    """A ledger classifying every required issue passes."""
    _write(tmp_path / "ledger.yaml", {"issues": {"10": "closed", "11": "negative_result"}})
    result = _load_and_eval(tmp_path, _checklist([_ledger_item([10, 11])]))
    assert result["status"] == "passed"


def test_issue_ledger_missing_issue_fails_closed(tmp_path: Path) -> None:
    """A required issue absent from the ledger fails closed."""
    _write(tmp_path / "ledger.yaml", {"issues": {"10": "closed"}})
    result = _load_and_eval(tmp_path, _checklist([_ledger_item([10, 11])]))
    assert result["status"] == "blocked"
    assert "#11" in result["items"][0]["gaps"][0]


def test_issue_ledger_unknown_classification_fails_closed(tmp_path: Path) -> None:
    """An unknown classification value fails closed."""
    _write(tmp_path / "ledger.yaml", {"issues": {"10": "maybe_later"}})
    result = _load_and_eval(tmp_path, _checklist([_ledger_item([10])]))
    assert result["status"] == "blocked"
    assert "unknown classification" in result["items"][0]["gaps"][0]


# --- rendering + shipped checklist smoke -------------------------------------


def test_render_markdown_contains_status(tmp_path: Path) -> None:
    """The Markdown report includes the overall status."""
    result = _load_and_eval(
        tmp_path,
        _checklist([_ledger_item([10])]),  # will be blocked (no ledger file)
    )
    md = render_markdown(result)
    assert "Release preflight" in md
    assert "blocked" in md


def test_loader_rejects_malformed_checklist_yaml(tmp_path: Path) -> None:
    """A syntactically invalid checklist YAML fails closed as a structural error."""
    path = tmp_path / "c.yaml"
    path.write_text("schema_version: [unbalanced\n", encoding="utf-8")
    with pytest.raises(ReleasePreflightError, match="Could not parse checklist"):
        load_release_preflight_checklist(path)


def test_referenced_file_malformed_yaml_fails_closed(tmp_path: Path) -> None:
    """A referenced file with invalid YAML fails closed to blocked (not a crash)."""
    (tmp_path / "claims.yaml").write_text("claims: [unterminated\n", encoding="utf-8")
    result = _load_and_eval(tmp_path, _checklist([_claim_item()]))
    assert result["status"] == "blocked"
    assert "could not be parsed" in result["items"][0]["gaps"][0]


def test_shipped_checklist_loads_and_evaluates() -> None:
    """The shipped July 2026 checklist loads and honestly reports blocked.

    The release artifacts are not present on the current checkout (issue #3081 is
    blocked), so the preflight must fail closed rather than declare readiness.
    """
    checklist = load_release_preflight_checklist(SHIPPED_CHECKLIST)
    assert checklist.release_id == "research_package_july_2026"
    assert len(checklist.items) == 4
    result = evaluate_release_preflight(checklist, REPO_ROOT)
    assert result["status"] == "blocked"
    assert result["summary"]["blocked"] >= 1
