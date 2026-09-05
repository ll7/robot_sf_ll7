"""Tests for the report-only open-issue implementability audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.dev.audit_open_issue_contracts import (
    DEFAULT_ITEM_LIMIT,
    DEFAULT_MAX_PAGES,
    DEFAULT_PAGE_SIZE,
    NEXT_ACTIONS,
    _build_report,
    _fixture_evaluator,
    _fixture_pagination,
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
) -> dict[str, Any]:
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
) -> dict[str, Any]:
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


def _claim(*, claimed: bool = False, ok: bool = True) -> dict[str, Any]:
    """Return one normalized claim fixture."""
    return {
        "ok": ok,
        "claimed": claimed if ok else None,
        "claim_ref": "agent-claims/issue-1" if claimed else None,
        "sha": "abc" if claimed else None,
    }


def _fixture(
    rows: list[dict[str, Any]],
    *,
    exact: dict[int, dict[str, Any]] | None = None,
    claims: dict[int, dict[str, Any]] | None = None,
    dependencies: dict[int, dict[str, Any]] | None = None,
    trailing_empty_page: bool = True,
) -> dict[str, Any]:
    """Return one valid fixture envelope."""
    exact_rows = exact or {row["number"]: _exact_issue(row["number"]) for row in rows}
    claim_rows = claims or {row["number"]: _claim() for row in rows}
    pages: list[list[dict[str, Any]]] = [rows]
    if trailing_empty_page:
        pages.append([])
    return {
        "pages": pages,
        "exact_issues": {str(number): value for number, value in exact_rows.items()},
        "claims": {str(number): value for number, value in claim_rows.items()},
        "dependencies": {str(number): value for number, value in (dependencies or {}).items()},
    }


def _report(
    fixture: dict[str, Any], *, page_size: int = 100, max_pages: int = 20
) -> dict[str, Any]:
    """Build one deterministic report from fixture data."""
    normalized = _validate_fixture(fixture)
    pages, complete, errors = _fixture_pagination(
        normalized,
        page_size=page_size,
        max_pages=max_pages,
    )
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
    assert item["claim"] == {
        "ok": True,
        "claimed": False,
        "claim_ref": None,
        "sha": None,
    }
    assert item["dispatch_eligible"] is True
    assert item["next_action"] == "dispatch_via_goal_issue_admission"
    assert report["summary"]["claim_states"] == {"unclaimed": 1}
    assert report["summary"]["admission_reason_histogram"] == {"claimable": 1}
    assert report["summary"]["not_admitted"] == {}
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


def test_wrong_owner_repo_is_reported_without_dispatch_authority() -> None:
    """The repository audit must preserve an explicit cross-repository owner blocker."""
    body = (
        COMPLETE_BODY
        + """
## Execution
```yaml
execution:
  owning_repo: ll7/codex-orchestrator
  mutation_repos:
    - ll7/codex-orchestrator
  route_required: multi_repository
  external_inputs: []
```
"""
    )
    report = _report(
        _fixture(
            [_raw_issue(1)],
            exact={1: _exact_issue(1, body=body)},
        )
    )

    item = report["items"][0]
    assert item["classification"] == "wrong_owner_repo"
    assert item["admission_reason"] == "wrong_owner_repo"
    assert item["next_action"] == "move_or_split_cross_repository_issue"
    assert report["summary"]["not_admitted"] == {"wrong_owner_repo": 1}


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
    assert report["pagination"]["resume_hint"]


def test_default_live_audit_bounds_are_explicit_and_conservative() -> None:
    """Default developer scans stay bounded and expose fresh-run guidance when partial."""
    assert DEFAULT_PAGE_SIZE == 20
    assert DEFAULT_MAX_PAGES == 1
    assert DEFAULT_ITEM_LIMIT == 20


def test_complete_report_has_no_resume_hint() -> None:
    """A complete fixture does not advertise a continuation path."""
    report = _report(_fixture([_raw_issue(1)]))
    assert report["complete"] is True
    assert report["pagination"]["resume_hint"] is None


def test_partial_markdown_report_exposes_resume_hint() -> None:
    """Markdown output carries the same fresh-run guidance as the JSON report."""
    report = _report(
        _fixture([_raw_issue(1)], trailing_empty_page=False),
        page_size=1,
        max_pages=1,
    )

    rendered = _render_markdown(report, item_limit=1, json_report=None)
    assert "Pagination guidance:" in rendered
    assert "fresh audit" in rendered


def test_hidden_extra_fixture_page_is_truncated() -> None:
    """Extra source pages beyond max-pages remain incomplete even after a short selected page."""
    fixture = _fixture([_raw_issue(1)])
    fixture["pages"] = [[_raw_issue(1)], [], []]

    report = _report(fixture, page_size=100, max_pages=2)
    assert report["complete"] is False
    assert report["applicable"] is False
    assert any("fixture pagination is incomplete" in error for error in report["errors"])


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
        _raw_issue(1, labels=["parent", "state:blocked"]),
        _raw_issue(2, labels=["decision-required", "state:blocked"]),
        _raw_issue(3, labels=["resource:slurm", "state:ready"]),
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
    assert report["summary"]["admission_reason_histogram"] == {
        "active_work": 1,
        "blocked": 1,
        "covering_pr_open": 1,
        "human_decision": 1,
        "needs_compute": 1,
        "parent_not_leaf": 1,
    }


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
    assert report["summary"]["claim_states"] == {
        "claimed": 1,
        "unavailable": 1,
        "unclaimed": 1,
    }
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


def test_unknown_classifier_output_fails_closed() -> None:
    """A future unknown classification must not become implied dispatch authority."""
    fixture = _validate_fixture(_fixture([_raw_issue(1)]))

    def unknown(_number: int) -> dict[str, Any]:
        report = _fixture_evaluator(fixture)(1)
        report["classification"] = "future_state"
        return report

    report = _build_report(
        repo="ll7/robot_sf_ll7",
        source="fixture",
        pages=fixture["pages"],
        pagination={"complete": True, "errors": [], "page_size": 100, "max_pages": 20},
        evaluator=unknown,
        input_sha256="f" * 64,
    )
    assert report["items"][0]["classification"] == "error"
    assert report["applicable"] is False


def test_unexpected_per_issue_exception_is_bounded_and_scan_continues() -> None:
    """One unexpected evaluator exception must not hide later issue packets."""
    fixture = _validate_fixture(_fixture([_raw_issue(1), _raw_issue(2)]))
    stable_evaluator = _fixture_evaluator(fixture)

    def unstable(number: int) -> dict[str, Any]:
        if number == 1:
            raise ZeroDivisionError("unexpected per-issue failure")
        return stable_evaluator(number)

    report = _build_report(
        repo="ll7/robot_sf_ll7",
        source="fixture",
        pages=fixture["pages"],
        pagination={"complete": True, "errors": [], "page_size": 100, "max_pages": 20},
        evaluator=unstable,
        input_sha256="f" * 64,
    )

    assert [item["classification"] for item in report["items"]] == ["error", "ready"]
    assert report["items"][1]["dispatch_eligible"] is True
    assert report["applicable"] is False
    assert any("ZeroDivisionError" in error for error in report["errors"])


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


def test_module_has_no_github_mutation_surface() -> None:
    """The audit owner must not import or call repository mutation helpers."""
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/dev/audit_open_issue_contracts.py").read_text(encoding="utf-8")
    forbidden = (
        "add_issue_labels",
        "remove_issue_label",
        "update_issue",
        "create_issue",
        "add_comment_to_issue",
        "create_pull_request",
        "update_ref",
    )
    assert not any(name in source for name in forbidden)


def test_fixture_cli_process_boundary(tmp_path: Path) -> None:
    """The documented fixture CLI must emit JSON and preserve exit-code semantics."""
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(_fixture([_raw_issue(1)])), encoding="utf-8")
    root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.dev.audit_open_issue_contracts",
            "--fixture",
            str(fixture_path),
            "--page-size",
            "100",
            "--max-pages",
            "2",
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


def test_fixture_cli_partial_check_returns_two_after_writing_report(tmp_path: Path) -> None:
    """A bounded partial audit remains diagnostic and fails closed at the CLI boundary."""
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps(_fixture([_raw_issue(1)], trailing_empty_page=False)), encoding="utf-8"
    )
    root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.dev.audit_open_issue_contracts",
            "--fixture",
            str(fixture_path),
            "--page-size",
            "1",
            "--max-pages",
            "1",
            "--check",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["complete"] is False
    assert report["applicable"] is False
    assert report["pagination"]["resume_hint"]
