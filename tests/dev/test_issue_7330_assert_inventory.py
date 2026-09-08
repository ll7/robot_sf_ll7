"""Tests for the deterministic production-assert inventory in issue #7330."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.dev import audit_production_asserts

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "robot_sf"


def test_current_main_residuals_are_complete_and_internal() -> None:
    """Every exact-source residual is reviewed and routes to parent closure."""
    payload = audit_production_asserts.build_inventory(REPO_ROOT, SOURCE_ROOT)

    assert payload["schema"] == "production-assert-inventory.v1"
    assert isinstance(payload["source"]["clean"], bool)
    assert payload["counts"]["assertion_count"] == 26
    assert payload["counts"]["classification"] == {"genuine_internal_invariant": 26}
    assert payload["counts"]["ownership"] == {
        "completed_historical_review": 14,
        "unowned_residual": 12,
    }
    assert {
        (row["path"], row["scope"], row["expression"])
        for row in payload["assertions"]
        if row["expression"] == "state is not None"
    } == {
        (
            "robot_sf/benchmark/map_runner/map_runner_episode.py",
            "_setup_and_run_step_loop",
            "state is not None",
        )
    }
    assert payload["recommendation"]["code"] == "close_parent_residuals_internal_only"
    assert all(
        row["recommended_action"] == "retain_assert_as_internal_invariant"
        for row in payload["assertions"]
    )


def test_json_and_markdown_rendering_is_deterministic() -> None:
    """Two runs over one commit produce byte-identical serialized outputs."""
    first = audit_production_asserts.build_inventory(REPO_ROOT, SOURCE_ROOT)
    second = audit_production_asserts.build_inventory(REPO_ROOT, SOURCE_ROOT)

    first_json = json.dumps(first, indent=2, sort_keys=True) + "\n"
    second_json = json.dumps(second, indent=2, sort_keys=True) + "\n"
    assert first_json == second_json
    assert audit_production_asserts.render_markdown(
        first
    ) == audit_production_asserts.render_markdown(second)


def test_detached_source_reports_detached_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exact detached-main snapshot is a supported inventory source."""
    original_run = audit_production_asserts.subprocess.run

    def fake_run(command: list[str], *args: object, **kwargs: object) -> object:
        if command[-4:] == ["symbolic-ref", "--short", "-q", "HEAD"]:
            return subprocess.CompletedProcess(command, 1, "", "")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(audit_production_asserts.subprocess, "run", fake_run)
    payload = audit_production_asserts.build_inventory(REPO_ROOT, SOURCE_ROOT)

    assert payload["source"]["ref"] == "DETACHED"


def test_unknown_assertion_fails_closed() -> None:
    """A source assertion outside the reviewed matrix cannot become ready."""
    unknown = audit_production_asserts.AssertionRow(
        path="robot_sf/example.py",
        line=1,
        end_line=1,
        qualified_scope="example",
        expression="value is not None",
        message=None,
        control_flow=(),
    )

    with pytest.raises(audit_production_asserts.InventoryError, match="unreviewed"):
        audit_production_asserts._reviewed_rows([unknown])


def test_cli_writes_both_issue_outputs(tmp_path: Path) -> None:
    """The issue-scoped command emits the required JSON and Markdown packets."""
    json_path = tmp_path / "assert_inventory.json"
    markdown_path = tmp_path / "assert_inventory.md"

    result = audit_production_asserts.main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--root",
            "robot_sf",
            "--json",
            str(json_path),
            "--markdown",
            str(markdown_path),
        ]
    )

    assert result == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["counts"]["assertion_count"] == 26
    assert "# Production assert inventory (issue #7330)" in markdown_path.read_text(
        encoding="utf-8"
    )
