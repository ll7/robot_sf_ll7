"""Tests for the tracked asset-rights inventory contract in issue #7299."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.tools import check_asset_rights_inventory as inventory

REPO_ROOT = Path(__file__).resolve().parents[2]


def _row(row_id: str, pattern: str, *, status: str = "project-authored") -> dict[str, object]:
    """Build a compact valid fixture row."""
    row: dict[str, object] = {
        "id": row_id,
        "scope": "assets",
        "globs": [pattern],
        "status": status,
        "source": "synthetic fixture source",
        "source_revision_or_access_date": "synthetic fixture revision",
        "license_or_rights": "synthetic fixture rights record",
        "attribution": "synthetic fixture attribution",
        "checksum_policy": "synthetic fixture checksum policy",
        "modification_status": "synthetic fixture modification record",
        "evidence": ["evidence.txt"],
    }
    if status in inventory.KNOWN_BLOCKING_STATUSES:
        row["unblock_condition"] = "supply the missing synthetic rights evidence"
    return row


def _write_fixture_inventory(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    """Write a minimal inventory and its evidence file."""
    (tmp_path / "evidence.txt").write_text("fixture evidence\n", encoding="utf-8")
    path = tmp_path / "inventory.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": inventory.INVENTORY_SCHEMA,
                "claim_boundary": "synthetic test inventory only",
                "tracked_scopes": [
                    {
                        "id": "assets",
                        "globs": ["assets/**"],
                        "release_relevant": True,
                    }
                ],
                "rows": rows,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_live_inventory_classifies_tracked_paths_without_unknown_gaps() -> None:
    """The checked-in inventory has no unclassified or overlapping tracked path."""
    report = inventory.build_report(REPO_ROOT)

    assert report["schema"] == inventory.REPORT_SCHEMA
    assert report["read_only"] is True
    assert report["tracked_path_count"] > 0
    assert report["counts"]["unclassified_paths"] == 0
    assert not any(
        issue["code"] in {"unclassified_path", "row_overlap", "scope_overlap"}
        for issue in report["issues"]
    )
    assert report["known_blockers"]
    assert report["counts"]["known_blocker_paths"] == len(report["known_blocker_paths"])
    assert "examples/datasets/2024-12-06_15-39-44.json" in report["known_blocker_paths"]
    assert inventory.exit_code(report) == 2
    assert inventory.exit_code(report, allow_known_blockers=True) == 0


def test_valid_fixture_passes_structural_check(tmp_path: Path) -> None:
    """A fully classified project-authored fixture passes."""
    path = _write_fixture_inventory(tmp_path, [_row("asset", "assets/*.svg")])

    report = inventory.build_report(
        tmp_path,
        path,
        tracked_paths=["assets/example.svg"],
    )

    assert report["status"] == "passed"
    assert report["issues"] == []
    assert report["known_blockers"] == []
    assert inventory.exit_code(report) == 0


def test_unclassified_path_is_a_hard_error(tmp_path: Path) -> None:
    """A new path outside the row glob cannot silently enter the release surface."""
    path = _write_fixture_inventory(tmp_path, [_row("svg", "assets/*.svg")])

    report = inventory.build_report(
        tmp_path,
        path,
        tracked_paths=["assets/new.json"],
    )

    assert any(issue["code"] == "unclassified_path" for issue in report["issues"])
    assert inventory.exit_code(report, allow_known_blockers=True) == 2


def test_unscoped_asset_like_path_is_a_hard_error(tmp_path: Path) -> None:
    """A likely asset cannot bypass the inventory by avoiding a declared scope."""
    path = _write_fixture_inventory(tmp_path, [_row("svg", "assets/*.svg")])

    report = inventory.build_report(
        tmp_path,
        path,
        tracked_paths=["unscoped/image.png"],
    )

    assert any(issue["code"] == "unscoped_asset_path" for issue in report["issues"])
    assert inventory.exit_code(report, allow_known_blockers=True) == 2


def test_overlapping_rows_are_a_hard_error(tmp_path: Path) -> None:
    """Two rows claiming one path make the inventory ambiguous."""
    path = _write_fixture_inventory(
        tmp_path,
        [_row("first", "assets/*.svg"), _row("second", "assets/*.svg")],
    )

    report = inventory.build_report(
        tmp_path,
        path,
        tracked_paths=["assets/ambiguous.svg"],
    )

    assert any(issue["code"] == "row_overlap" for issue in report["issues"])
    assert inventory.exit_code(report, allow_known_blockers=True) == 2


def test_blocked_row_requires_an_explicit_unblock_condition(tmp_path: Path) -> None:
    """Known uncertainty must name the evidence needed to unblock it."""
    row = _row("blocked", "assets/*.svg", status="blocked")
    row.pop("unblock_condition")
    path = _write_fixture_inventory(tmp_path, [row])

    report = inventory.build_report(
        tmp_path,
        path,
        tracked_paths=["assets/blocked.svg"],
    )

    assert any(issue["code"] == "missing_unblock_condition" for issue in report["issues"])
    assert inventory.exit_code(report, allow_known_blockers=True) == 2
