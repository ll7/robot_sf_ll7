"""Tests for the report-only open-issue implementability audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.dev.audit_open_issue_contracts import (
    NEXT_ACTIONS,
    _build_report,
    _fixture_evaluator,
    _render_markdown,
    _validate_fixture,
)

COMPLETE_BODY = """## Objective
Repair one bounded workflow defect.

## Scope
Change one helper. Do not change scientific semantics.

## Inputs
- `scripts/dev/example.py`

## Acceptance criteria
- [ ] The regression passes.

## Validation
```bash
uv run pytest -q tests/dev/test_example.py
```
"""


def _raw_issue(
    number: int,
    *,
    labels: list[str] | None = None,
    assignees: list[str] | None = None,
    title: str | None = None,
) -> dict[str, object]:
    """Return one REST-listing fixture row."""
    return {
        "number": number,
        "title": title or f"issue {number}",
        "state": "open",
        "html_url": f"https://github.test/issues/{number}",
        "labels": [{"name": label} for label in (labels or ["state:ready"])],
        "assignees": [{"login": login} for login in (assignees or [])],
    }


def _exact_issue(
    number: int,
    *,
    labels: list[str] | None = None,
    assignees: list[str] | None = None,
    body: str = COMPLETE_BODY,
    title: str | None = None,
    state: str = "OPEN",
) -> dict[str, object]:
    """Return one normalized exact-read fixture issue."""
    return {
        "number": number,
        "title": title or f"issue {number}",
        "body": body,
        "state": state,
        "url": f"https://github.test/issues/{number}",
        "labels": labels or ["state:ready"],
        "assignees": assignees or [],
    }


def _claim(*, claimed: bool = False, ok: bool = True) -> dict[str, object]:
    """Return one normalized claim fixture."""
    return {
        "ok": ok,
        "claimed": claimed if ok else None,
        "claim_ref": "agent-claims/issue-1" if claimed else None,
        "sha": "abc" if claimed else None,
    }


def _fixture(
    rows: list[dict[str, object]],
    *,
    exact: dict[int, dict[str, object]] | None = None,
    claims: dict[int, dict[str, object]] | None = None,
    dependencies: dict[int, dict[str, object]] | None = None,
    trailing_empty_page: bool = True,
) -> dict[str, object]:
    """Return one valid fixture envelope."""
    exact_rows = exact or {row["number"]: _exact_issue(row["number"]) for row in rows}
    claim_rows = claims or {row["number"]: _claim() for row in rows}
    pages: list[list[dict[str, object]]] = [rows]
    if trailing_empty_page:
        pages.append([])
    return {
        "pages": pages,
        "exact_issues": {str(number): value for number, value in exact_rows.items()},
        "claims": {str(number): value for number, value in claim_rows.items()},
        "dependencies": {
            str(number): value for number, value in (dependencies or {}).items()
        },
    }


def _report(
    fixture: dict[str, object], *, page_size: int = 100, max_pages: int = 20
) -> dict[str, object]:
    """Build one deterministic report from fixture data."""
    normalized = _validate_fixture(fixture)
    pages = normalized["pages"][:max_pages]
    complete = bool(pages) and len(pages[-1]) < page_size
    errors = [] if complete else ["fixture pagination incomplete"]
    return _build_report(
        repo="ll7/robot_sf_ll7",
        source="fixture",
        pages=pages,
        pagination={
            "complete": complete,
            "errors": errors,
            "page_size": page_size,
            "max_pages": max_pages,
        },
        evaluator=_fixture_evaluator(normalized),
        input_sha256="f" * 64,
    )


def test_ready_issue_uses_canonical_classifier_and_dispatch_handoff() -> None:
    """A complete ready leaf should be listed only for guarded admission."""
    report = _report(_fixture([_raw_issue(1)]))

    item = report["items"][0]
    assert item["classification"] == "ready"
    assert item["dispatch_eligible"] is True
    assert item["next_action"] == "dispatch_via_goal_issue_admission"
    assert report["summary"]["executable_leaf_numbers"] == [1]
    assert report["mutation_authorized"] is False


def test_ready_label_is_not_sufficient_when_contract_field_is_missing() -> None:
    """The audit must preserve the canonical needs-spec fail-closed result."""
    body = COMPLETE_BODY.replace("## Validation", "## Notes")
    fixture = _fixture(
        [_raw_issue(1)],
        exact={1: _exact_issue(1, body=body)},
    )

    item = _report(fixture)["items"][0]
    assert item["classification"] == "needs_spec"
    assert item["missing_fields"] == ["verification"]
    assert item["dispatch_eligible"] is False


def test_pull_requests_are_excluded_and_counted() -> None:
    """The REST issues endpoint may return PR rows, which are not issue supply."""
    pr = _raw_issue(2)
    pr["pull_request"] = {"url": "https://api.github.test/pulls/2"}
    report = _report(_fixture([_raw_issue(1), pr], exact={1: _exact_issue(1)}))

    assert report["pagination"]["raw_rows"] == 2
    assert report["pagination"]["issue_rows"] == 1
    assert report["pagination"]["excluded_pull_requests"] == 1
    assert [item["number"] for item in report["items"]] == [1]


def test_duplicate_issue_numbers_fail_closed() -> None:
    """Duplicate rows across pages must make the report non-applicable."""
    fixture = _fixture([_raw_issue(1)], trailing_empty_page=False)
    fixture["pages"] = [[_raw_issue(1)], [_raw_issue(1)], []]

    report = _report(fixture, page_size=1)
    assert report["applicable"] is False
    assert any("duplicate issue number 1" in error for error in report["errors"])


def test_full_final_page_at_limit_is_truncated() -> None:
    """A full final page cannot prove that all open issues were read."""
    fixture = _fixture([_raw_issue(1)], trailing_empty_page=False)

    report = _report(fixture, page_size=1, max_pages=1)
    assert report["complete"] is False
    assert report["applicable"] is False
    assert report["content_sha256"]


def test_listing_to_exact_read_drift_cannot_authorize_dispatch() -> None:
    """Changed labels between listing and exact read remain explicit and non-applicable."""
    fixture = _fixture(
        [_raw_issue(1, labels=["state:ready"])],
        exact={1: _exact_issue(1, labels=["state:ready", "type:workflow"])},
    )

    report = _report(fixture)
    item = report["items"][0]
    assert item["observed_classification"] == "ready"
    assert item["classification"] == "error"
    assert item["listing_drift"][0]["field"] == "labels"
    assert item["dispatch_eligible"] is False
    assert report["applicable"] is False


def test_active_parent_decision_compute_and_blocked_states_remain_distinct() -> None:
    """Repository routing states must not collapse to one generic not-ready class."""
    rows = [
        _raw_issue(1, labels=["parent"]),
        _raw_issue(2, labels=["decision-required"]),
        _raw_issue(3, labels=["resource:slurm"]),
        _raw_issue(4, labels=["state:blocked"]),
        _raw_issue(5, labels=["state:working"]),
        _raw_issue(6, labels=["state:review"]),
    ]
    exact = {
        row["number"]: _exact_issue(
            row["number"], labels=[label["name"] for label in row["labels"]]
        )
        for row in rows
    }

    report = _report(_fixture(rows, exact=exact))
    assert [item["classification"] for item in report["items"]] == [
        "parent",
        "human_decision",
        "needs_compute",
        "blocked",
        "working",
        "review",
    ]
    assert report["summary"]["executable_leaf_numbers"] == []


def test_claimed_assigned_and_unavailable_claim_states_are_distinct() -> None:
    """Claim ownership and claim-read failure must remain separate outcomes."""
    rows = [
        _raw_issue(1),
        _raw_issue(2, assignees=["worker"]),
        _raw_issue(3),
    ]
    exact = {
        1: _exact_issue(1),
        2: _exact_issue(2, assignees=["worker"]),
        3: _exact_issue(3),
    }
    claims = {1: _claim(claimed=True), 2: _claim(), 3: _claim(ok=False)}

    report = _report(_fixture(rows, exact=exact, claims=claims))
    assert [item["classification"] for item in report["items"]] == [
        "already_claimed",
        "assigned",
        "error",
    ]
    assert report["applicable"] is False


def test_dependency_evaluation_is_delegated_to_canonical_gate() -> None:
    """A mandatory dependency failure should surface as needs_dependency."""
    dependency = {
        "schema": "issue_dependency_evaluation.v1",
        "ok": False,
        "verdict": "blocked",
        "packet_digest": "d" * 64,
        "mandatory_failures": [
            {
                "id": "required-pr",
                "reason": "required PR is not merged",
                "unblock_condition": "merge PR #42",
            }
        ],
        "advisory_failures": [],
    }
    report = _report(_fixture([_raw_issue(1)], dependencies={1: dependency}))

    item = report["items"][0]
    assert item["classification"] == "needs_dependency"
    assert item["next_action"] == NEXT_ACTIONS["needs_dependency"]
    assert item["dependency_gate"]["mandatory_failures"][0]["id"] == "required-pr"


def test_report_is_byte_stable_for_fixed_fixture() -> None:
    """A fixed fixture must produce identical content and digest."""
    fixture = _fixture([_raw_issue(1), _raw_issue(2)])

    first = _report(fixture)
    second = _report(json.loads(json.dumps(fixture)))
    assert first == second
    assert first["content_sha256"] == second["content_sha256"]


def test_markdown_summary_is_bounded_and_omits_issue_bodies() -> None:
    """The context-capsule rendering should link the full JSON instead of copying bodies."""
    body = COMPLETE_BODY.replace("## Validation", "## Notes")
    fixture = _fixture(
        [_raw_issue(1), _raw_issue(2)],
        exact={1: _exact_issue(1, body=body), 2: _exact_issue(2, body=body)},
    )
    report = _report(fixture)

    rendered = _render_markdown(report, item_limit=1, json_report="output/audit.json")
    assert "Full JSON report: `output/audit.json`" in rendered
    assert rendered.count("| #") == 1
    assert "Repair one bounded workflow defect" not in rendered


def test_fixture_cli_process_boundary(tmp_path: Path) -> None:
    """The documented fixture CLI must emit JSON and preserve exit-code semantics."""
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(_fixture([_raw_issue(1)])), encoding="utf-8")
    root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/dev/audit_open_issue_contracts.py",
            "--fixture",
            str(fixture_path),
            "--page-size",
            "100",
            "--check",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["schema"] == "open_issue_contract_audit.v1"
    assert report["mutation_authorized"] is False
    assert report["summary"]["executable_leaf_numbers"] == [1]
