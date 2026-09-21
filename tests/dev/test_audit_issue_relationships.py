"""Regression tests for the explicit issue-relationship audit contract."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from scripts.dev.audit_issue_relationships import audit_relationships, parse_relationships


def _result(
    payload: Any, *, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["gh", "api"], returncode, json.dumps(payload), stderr)


CANONICAL_BODY = """# Example

## Relationships

- Parent issue: #12
- Blocked by: https://github.com/ll7/robot_sf_ll7/issues/13
- Blocking: none
- Relates to: ll7/robot_sf_ll7#14

## Scope
Keep the task bounded.
"""


def test_parse_canonical_relationships_and_none_sentinel() -> None:
    parsed = parse_relationships(CANONICAL_BODY, issue=11)

    assert parsed["section_present"] is True
    assert {(row["kind"], row["target"]) for row in parsed["declarations"]} == {
        ("parent", 12),
        ("blocked_by", 13),
        ("relates_to", 14),
    }
    assert parsed["errors"] == []
    assert parsed["legacy_mentions"] == []


def test_legacy_relationship_prose_is_report_only() -> None:
    parsed = parse_relationships(
        """## Parent issue: #40
Child of #42.

## Related issues
- #43
""",
        issue=41,
    )

    assert parsed["declarations"] == []
    assert "missing canonical ## Relationships section" in parsed["errors"]
    assert {item["kind"] for item in parsed["legacy_mentions"]} == {"parent", "relates_to"}
    assert parsed["legacy_mentions"][0]["targets"] == (40, 42)
    assert parsed["legacy_mentions"][1]["targets"] == (43,)


def test_parse_rejects_ambiguous_or_cross_repository_declarations() -> None:
    parsed = parse_relationships(
        """## Relationships
- Parent issue: #10, #11
- Blocked by: https://github.com/other/repo/issues/12
- Blocking: #7
""",
        issue=7,
    )

    assert any("at most one" in error for error in parsed["errors"])
    assert any("cross-repository" in error for error in parsed["errors"])
    assert any("itself" in error for error in parsed["errors"])


def test_parse_requires_an_explicit_none_sentinel() -> None:
    parsed = parse_relationships(
        """## Relationships
- Parent issue:
- Blocked by: none
- Blocking: none
- Relates to: none
""",
        issue=1,
    )

    assert any("explicit `none`" in error for error in parsed["errors"])


def test_audit_dry_run_reads_native_state_without_writing() -> None:
    calls: list[tuple[str, object | None, str | None]] = []

    def api(
        path: str, payload: object | None, method: str | None
    ) -> subprocess.CompletedProcess[str]:
        calls.append((path, payload, method))
        if "issues?state=open" in path:
            return _result(
                [
                    {
                        "number": 11,
                        "id": 111,
                        "title": "child",
                        "body": CANONICAL_BODY,
                        "html_url": "https://github.com/ll7/robot_sf_ll7/issues/11",
                    }
                ]
            )
        if path.endswith("/parent"):
            return _result({})
        if path.endswith("/dependencies/blocked_by"):
            return _result([])
        if path.endswith("/dependencies/blocking"):
            return _result([])
        raise AssertionError(path)

    report = audit_relationships(api=api, per_page=10, max_pages=1)

    assert report["source"]["status"] == "complete"
    operations = report["issues"][0]["operations"]
    assert {(item["kind"], item["status"]) for item in operations} == {
        ("parent", "proposed"),
        ("blocked_by", "proposed"),
        ("relates_to", "manual"),
    }
    assert report["dry_run"] is True
    assert all(method is None for _, _, method in calls)


def test_apply_requires_confirmation_and_verifies_new_link() -> None:
    calls: list[tuple[str, object | None, str | None]] = []
    parent_added = False

    def api(
        path: str, payload: object | None, method: str | None
    ) -> subprocess.CompletedProcess[str]:
        nonlocal parent_added
        calls.append((path, payload, method))
        if "issues?state=open" in path:
            return _result(
                [
                    {
                        "number": 11,
                        "id": 111,
                        "title": "child",
                        "body": "## Relationships\n- Parent issue: #12\n- Blocked by: none\n- Blocking: none\n- Relates to: none\n",
                        "html_url": "https://github.com/ll7/robot_sf_ll7/issues/11",
                    },
                    {
                        "number": 12,
                        "id": 112,
                        "title": "parent",
                        "body": "## Relationships\n- Parent issue: none\n- Blocked by: none\n- Blocking: none\n- Relates to: none\n",
                        "html_url": "https://github.com/ll7/robot_sf_ll7/issues/12",
                    },
                ]
            )
        if path.endswith("/issues/11/parent"):
            return _result({"number": 12} if parent_added else {}, returncode=0)
        if path.endswith("/issues/11/dependencies/blocked_by"):
            return _result([])
        if path.endswith("/issues/11/dependencies/blocking"):
            return _result([])
        if path.endswith("/issues/12/sub_issues") and method == "POST":
            parent_added = True
            return _result({})
        raise AssertionError(path)

    report = audit_relationships(
        api=api,
        per_page=10,
        max_pages=1,
        apply=True,
        confirmation="RELATIONSHIP_MIGRATION",
    )

    assert report["apply"]["status"] == "complete"
    writes = [(path, method) for path, _, method in calls if method == "POST"]
    assert writes == [("repos/ll7/robot_sf_ll7/issues/12/sub_issues", "POST")]


def test_existing_different_parent_is_a_conflict_without_a_write() -> None:
    calls: list[tuple[str, object | None, str | None]] = []

    def api(
        path: str, payload: object | None, method: str | None
    ) -> subprocess.CompletedProcess[str]:
        calls.append((path, payload, method))
        if "issues?state=open" in path:
            return _result(
                [
                    {
                        "number": 11,
                        "id": 111,
                        "title": "child",
                        "body": "## Relationships\n- Parent issue: #12\n- Blocked by: none\n- Blocking: none\n- Relates to: none\n",
                        "html_url": "https://github.com/ll7/robot_sf_ll7/issues/11",
                    }
                ]
            )
        if path.endswith("/issues/11/parent"):
            return _result({"number": 99})
        if path.endswith("/dependencies/blocked_by"):
            return _result([])
        if path.endswith("/dependencies/blocking"):
            return _result([])
        raise AssertionError(path)

    report = audit_relationships(api=api, max_pages=1)

    assert report["issues"][0]["operations"][0]["status"] == "conflict"
    assert any("native parent differs" in error for error in report["errors"])
    assert all(method != "POST" for _, _, method in calls)


def test_apply_without_confirmation_does_not_call_post() -> None:
    def api(
        path: str, payload: object | None, method: str | None
    ) -> subprocess.CompletedProcess[str]:
        if "issues?state=open" in path:
            return _result(
                [
                    {
                        "number": 1,
                        "id": 101,
                        "title": "issue",
                        "body": "## Relationships\n- Parent issue: #2\n",
                        "html_url": "https://github.com/ll7/robot_sf_ll7/issues/1",
                    }
                ]
            )
        if path.endswith("/parent"):
            return _result({})
        if path.endswith("/dependencies/blocked_by"):
            return _result([])
        if path.endswith("/dependencies/blocking"):
            return _result([])
        raise AssertionError(path)

    calls: list[str] = []

    def recording_api(path: str, payload: object | None, method: str | None) -> Any:
        calls.append(method or "GET")
        return api(path, payload, method)

    report = audit_relationships(api=recording_api, apply=True, confirmation=None, max_pages=1)

    assert report["apply"]["status"] == "blocked"
    assert "POST" not in calls


def test_apply_reciprocal_dependencies_produces_one_write_and_already_applied() -> None:
    calls: list[tuple[str, object | None, str | None]] = []
    dependency_added = False

    def api(
        path: str, payload: object | None, method: str | None
    ) -> subprocess.CompletedProcess[str]:
        nonlocal dependency_added
        calls.append((path, payload, method))
        if "issues?state=open" in path:
            return _result(
                [
                    {
                        "number": 1,
                        "id": 101,
                        "title": "blocker",
                        "body": "## Relationships\n- Parent issue: none\n- Blocked by: none\n- Blocking: #2\n- Relates to: none\n",
                        "html_url": "https://github.com/ll7/robot_sf_ll7/issues/1",
                    },
                    {
                        "number": 2,
                        "id": 102,
                        "title": "blocked",
                        "body": "## Relationships\n- Parent issue: none\n- Blocked by: #1\n- Blocking: none\n- Relates to: none\n",
                        "html_url": "https://github.com/ll7/robot_sf_ll7/issues/2",
                    },
                ]
            )
        if path.endswith("/parent"):
            return _result({})
        if path.endswith("/issues/1/dependencies/blocked_by"):
            return _result([])
        if path.endswith("/issues/1/dependencies/blocking"):
            return _result([{"number": 2}] if dependency_added else [])
        if path.endswith("/issues/2/dependencies/blocked_by") and method != "POST":
            return _result([{"number": 1}] if dependency_added else [])
        if path.endswith("/issues/2/dependencies/blocking"):
            return _result([])
        if path.endswith("/issues/2/dependencies/blocked_by") and method == "POST":
            dependency_added = True
            return _result({})
        raise AssertionError(f"unexpected call: {path} (method={method})")

    report = audit_relationships(
        api=api,
        per_page=10,
        max_pages=1,
        apply=True,
        confirmation="RELATIONSHIP_MIGRATION",
    )

    assert report["apply"]["status"] == "complete"
    writes = [(path, method) for path, _, method in calls if method == "POST"]
    assert writes == [("repos/ll7/robot_sf_ll7/issues/2/dependencies/blocked_by", "POST")]

    ops = report["apply"]["operations"]
    assert len(ops) == 2
    assert ops[0]["kind"] == "blocking"
    assert ops[0]["status"] == "applied"
    assert ops[1]["kind"] == "blocked_by"
    assert ops[1]["status"] == "already_applied"
    assert "reciprocal native relationship already added" in ops[1]["reason"]


def test_write_operation_reconciles_422_when_readback_confirms() -> None:
    calls: list[tuple[str, object | None, str | None]] = []
    write_attempted = False

    def api(
        path: str, payload: object | None, method: str | None
    ) -> subprocess.CompletedProcess[str]:
        nonlocal write_attempted
        calls.append((path, payload, method))
        if "issues?state=open" in path:
            return _result(
                [
                    {
                        "number": 2,
                        "id": 102,
                        "title": "blocked",
                        "body": "## Relationships\n- Parent issue: none\n- Blocked by: #1\n- Blocking: none\n- Relates to: none\n",
                        "html_url": "https://github.com/ll7/robot_sf_ll7/issues/2",
                    },
                ]
            )
        if "issues/1" in path and method != "POST":
            return _result({"number": 1, "id": 101, "title": "blocker", "body": ""})
        if path.endswith("/parent"):
            return _result({})
        if path.endswith("/dependencies/blocking"):
            return _result([])
        if path.endswith("/issues/2/dependencies/blocked_by") and method == "POST":
            write_attempted = True
            return _result(
                {},
                returncode=1,
                stderr="gh: Target issue has already been taken (HTTP 422)",
            )
        if path.endswith("/issues/2/dependencies/blocked_by") and method != "POST":
            return _result([{"number": 1}] if write_attempted else [])
        raise AssertionError(f"unexpected call: {path} (method={method})")

    report = audit_relationships(
        api=api,
        per_page=10,
        max_pages=1,
        apply=True,
        confirmation="RELATIONSHIP_MIGRATION",
    )

    assert report["apply"]["status"] == "complete"
    ops = report["apply"]["operations"]
    assert len(ops) == 1
    assert ops[0]["status"] == "already_applied"
    assert "verified with read-back" in ops[0]["reason"]


def test_write_operation_fails_closed_on_server_error() -> None:
    def api_500(
        path: str, payload: object | None, method: str | None
    ) -> subprocess.CompletedProcess[str]:
        if "issues?state=open" in path:
            return _result(
                [
                    {
                        "number": 2,
                        "id": 102,
                        "title": "blocked",
                        "body": "## Relationships\n- Parent issue: none\n- Blocked by: #1\n- Blocking: none\n- Relates to: none\n",
                        "html_url": "https://github.com/ll7/robot_sf_ll7/issues/2",
                    },
                ]
            )
        if "issues/1" in path and method != "POST":
            return _result({"number": 1, "id": 101, "title": "blocker", "body": ""})
        if path.endswith("/parent") or path.endswith("/dependencies/blocking"):
            return _result([])
        if path.endswith("/issues/2/dependencies/blocked_by") and method == "POST":
            return _result({}, returncode=1, stderr="HTTP 500 Internal Server Error")
        if path.endswith("/issues/2/dependencies/blocked_by") and method != "POST":
            return _result([])
        raise AssertionError(path)

    report = audit_relationships(
        api=api_500,
        apply=True,
        confirmation="RELATIONSHIP_MIGRATION",
        max_pages=1,
    )
    assert report["apply"]["status"] == "failed"
    assert report["apply"]["operations"][0]["status"] == "failed"


def test_write_operation_fails_closed_on_unverified_422() -> None:
    def api_unverified_422(
        path: str, payload: object | None, method: str | None
    ) -> subprocess.CompletedProcess[str]:
        if "issues?state=open" in path:
            return _result(
                [
                    {
                        "number": 2,
                        "id": 102,
                        "title": "blocked",
                        "body": "## Relationships\n- Parent issue: none\n- Blocked by: #1\n- Blocking: none\n- Relates to: none\n",
                        "html_url": "https://github.com/ll7/robot_sf_ll7/issues/2",
                    },
                ]
            )
        if "issues/1" in path and method != "POST":
            return _result({"number": 1, "id": 101, "title": "blocker", "body": ""})
        if path.endswith("/parent") or path.endswith("/dependencies/blocking"):
            return _result([])
        if path.endswith("/issues/2/dependencies/blocked_by") and method == "POST":
            return _result(
                {},
                returncode=1,
                stderr="gh: Target issue has already been taken (HTTP 422)",
            )
        if path.endswith("/issues/2/dependencies/blocked_by") and method != "POST":
            return _result([])  # Does NOT contain target #1
        raise AssertionError(path)

    report = audit_relationships(
        api=api_unverified_422,
        apply=True,
        confirmation="RELATIONSHIP_MIGRATION",
        max_pages=1,
    )
    assert report["apply"]["status"] == "failed"
    assert report["apply"]["operations"][0]["status"] == "failed"
