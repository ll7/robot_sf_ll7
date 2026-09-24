"""Offline contract tests for the bounded GitHub REST audit transport."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import pytest

from robot_sf.analysis_workbench.audit_github import GitHubIssueMissing
from robot_sf.analysis_workbench.audit_github_rest import (
    GitHubCommentWrite,
    GitHubRestAmbiguousError,
    GitHubRestDuplicateMarkerError,
    GitHubRestHTTPError,
    GitHubRestLimitError,
    GitHubRestPaginationError,
    GitHubRESTProvider,
    GitHubRestUnsupportedError,
    GitHubRestValidationError,
    HttpResponse,
    auditor_request_marker,
    parse_auditor_request_marker,
)

REPOSITORY = "ll7/robot_sf_ll7"
API_ORIGIN = "https://api.github.test"


def _issue(
    number: int,
    *,
    body: str = "human body",
    title: str = "audit issue",
    repository: str = REPOSITORY,
) -> dict[str, Any]:
    return {
        "repository": repository,
        "number": number,
        "html_url": f"https://github.com/{repository}/issues/{number}",
        "title": title,
        "body": body,
        "comments": 0,
        "labels": [{"name": "benchmark-audit"}],
        "state": "open",
        "updated_at": f"2026-09-20T00:00:{number:02d}Z",
    }


def _comment(comment_id: int, body: str) -> dict[str, Any]:
    return {
        "id": comment_id,
        "body": body,
        "html_url": f"https://github.com/{REPOSITORY}/issues/7#issuecomment-{comment_id}",
    }


def _link(path: str, page: int, *, query: dict[str, str] | None = None) -> str:
    values = dict(query or {})
    values["page"] = str(page)
    return f'<{API_ORIGIN}{path}?{urlencode(values)}>; rel="next"'


@dataclass
class FakeHTTP:
    """Route-level fake that records requests and can raise after mutation."""

    routes: dict[tuple[str, str, int], HttpResponse | BaseException]

    def __init__(self) -> None:
        """Initialize empty routes, comments, and request history."""

        self.routes = {}
        self.calls: list[tuple[str, str, bytes | None]] = []
        self.comment_store: list[dict[str, Any]] = []
        self.timeout_after_comment = False
        self.next_comment_id = 100

    def add(self, method: str, path: str, response: HttpResponse, *, page: int = 1) -> None:
        self.routes[(method, path, page)] = response

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> HttpResponse:
        del headers, timeout
        parsed = urlparse(url)
        page = int(parse_qs(parsed.query).get("page", ["1"])[0])
        self.calls.append((method, url, body))
        path = parsed.path
        route = self.routes.get((method, path, page))
        if method == "GET" and route is not None:
            if isinstance(route, BaseException):
                raise route
            return route
        if method == "POST" and path.endswith("/comments"):
            payload = json.loads(body.decode("utf-8")) if body is not None else {}
            comment = _comment(self.next_comment_id, payload.get("body", ""))
            self.next_comment_id += 1
            self.comment_store.append(comment)
            response = HttpResponse(201, comment)
            if self.timeout_after_comment:
                self.timeout_after_comment = False
                raise TimeoutError("response lost after accepted comment")
            return response
        if path.endswith("/comments") and method == "GET":
            return HttpResponse(200, self.comment_store)
        if isinstance(route, BaseException):
            raise route
        if route is None:
            raise AssertionError(f"unexpected fake HTTP request: {method} {url}")
        return route


def _provider(fake: FakeHTTP, **kwargs: Any) -> GitHubRESTProvider:
    return GitHubRESTProvider(
        fake,
        allowed_repositories=(REPOSITORY,),
        api_base_url=API_ORIGIN,
        **kwargs,
    )


def test_search_follows_all_pages_and_filters_exact_marker() -> None:
    fake = FakeHTTP()
    marker = "<!-- robot_sf_audit_finding:v1 repository=ll7/robot_sf_ll7 finding_id=f-1 -->"
    first = [_issue(1, body="unrelated")]
    second = [_issue(2, body=f"{marker}\n\nHuman follow-up")]
    fake.add(
        "GET",
        "/search/issues",
        HttpResponse(
            200,
            {"total_count": 2, "incomplete_results": False, "items": first},
            {
                "Link": _link(
                    "/search/issues",
                    2,
                    query={
                        "q": f'repo:{REPOSITORY} "{marker}"',
                        "per_page": "1",
                    },
                )
            },
        ),
    )
    fake.add(
        "GET",
        "/search/issues",
        HttpResponse(200, {"total_count": 2, "incomplete_results": False, "items": second}),
        page=2,
    )

    result = _provider(fake, per_page=1).search_issues(REPOSITORY, marker=marker)

    assert result.complete
    assert len(result.issues) == 1
    assert result.issues[0].number == 2
    assert result.issues[0].body.endswith("Human follow-up")
    assert [parse_qs(urlparse(url).query)["page"][0] for _, url, _ in fake.calls] == ["1", "2"]


def test_search_rejects_duplicate_exact_marker_and_incomplete_results() -> None:
    marker = "<!-- robot_sf_audit_finding:v1 repository=ll7/robot_sf_ll7 finding_id=f-2 -->"
    fake = FakeHTTP()
    fake.add(
        "GET",
        "/search/issues",
        HttpResponse(
            200,
            {
                "total_count": 2,
                "incomplete_results": False,
                "items": [_issue(1, body=marker), _issue(2, body=marker)],
            },
        ),
    )
    with pytest.raises(GitHubRestDuplicateMarkerError):
        _provider(fake).search_issues(REPOSITORY, marker=marker)

    incomplete = FakeHTTP()
    incomplete.add(
        "GET",
        "/search/issues",
        HttpResponse(
            200,
            {"total_count": 1, "incomplete_results": True, "items": [_issue(1)]},
        ),
    )
    with pytest.raises(GitHubRestPaginationError):
        _provider(incomplete).search_issues(REPOSITORY, marker=marker)


@pytest.mark.parametrize("incomplete_value", [None, "false"])
def test_search_requires_boolean_incomplete_results_metadata(
    incomplete_value: Any,
) -> None:
    fake = FakeHTTP()
    payload: dict[str, Any] = {
        "total_count": 0,
        "items": [],
    }
    if incomplete_value is not None:
        payload["incomplete_results"] = incomplete_value
    fake.add("GET", "/search/issues", HttpResponse(200, payload))

    with pytest.raises((GitHubRestPaginationError, GitHubRestValidationError)):
        _provider(fake).search_issues(REPOSITORY, marker="stable-marker")


def test_search_rejects_skipped_link_page_before_marker_selection() -> None:
    marker = "<!-- robot_sf_audit_finding:v1 repository=ll7/robot_sf_ll7 finding_id=skipped -->"
    fake = FakeHTTP()
    fake.add(
        "GET",
        "/search/issues",
        HttpResponse(
            200,
            {"total_count": 2, "incomplete_results": False, "items": [_issue(1)]},
            {
                "Link": _link(
                    "/search/issues",
                    3,
                    query={
                        "q": f'repo:{REPOSITORY} "{marker}"',
                        "per_page": "100",
                    },
                )
            },
        ),
    )
    fake.add(
        "GET",
        "/search/issues",
        HttpResponse(
            200,
            {"total_count": 2, "incomplete_results": False, "items": [_issue(3, body=marker)]},
        ),
        page=3,
    )

    with pytest.raises(GitHubRestPaginationError):
        _provider(fake).search_issues(REPOSITORY, marker=marker)
    assert len(fake.calls) == 1


def test_issue_list_can_apply_the_same_exact_marker_contract() -> None:
    marker = "<!-- robot_sf_audit_finding:v1 repository=ll7/robot_sf_ll7 finding_id=f-list -->"
    fake = FakeHTTP()
    fake.add(
        "GET",
        "/repos/ll7/robot_sf_ll7/issues",
        HttpResponse(200, [_issue(4, body=f"{marker}\n\nHuman body"), _issue(5)]),
    )

    matches = _provider(fake, per_page=3).list_issues(REPOSITORY, marker)

    assert [issue.number for issue in matches] == [4]
    assert matches[0].body.endswith("Human body")


def test_search_rejects_1000_plus_result_cap_before_following_pages() -> None:
    fake = FakeHTTP()
    fake.add(
        "GET",
        "/search/issues",
        HttpResponse(
            200,
            {"total_count": 1001, "incomplete_results": False, "items": []},
        ),
    )

    with pytest.raises(GitHubRestLimitError):
        _provider(fake).search_issues(REPOSITORY, marker="stable-marker")
    assert len(fake.calls) == 1


def test_full_sized_page_without_next_link_is_incomplete() -> None:
    fake = FakeHTTP()
    fake.add("GET", "/search/issues", HttpResponse(200, {"total_count": 2, "items": [_issue(1)]}))

    with pytest.raises(GitHubRestPaginationError):
        _provider(fake, per_page=1).search_issues(REPOSITORY, marker="absent")


def test_get_issue_reads_all_comment_pages_and_preserves_human_body() -> None:
    fake = FakeHTTP()
    fake.add("GET", "/repos/ll7/robot_sf_ll7/issues/7", HttpResponse(200, _issue(7)))
    fake.add(
        "GET",
        "/repos/ll7/robot_sf_ll7/issues/7/comments",
        HttpResponse(
            200,
            [_comment(1, "Human note")],
            {
                "Link": _link(
                    "/repos/ll7/robot_sf_ll7/issues/7/comments",
                    2,
                    query={"per_page": "2"},
                )
            },
        ),
    )
    fake.add(
        "GET",
        "/repos/ll7/robot_sf_ll7/issues/7/comments",
        HttpResponse(200, [_comment(2, "Second human note")]),
        page=2,
    )

    issue = _provider(fake, per_page=2).get_issue(REPOSITORY, 7)

    assert issue.body == "human body"
    assert [comment["body"] for comment in issue.comments] == [
        "Human note",
        "Second human note",
    ]


def test_create_issue_posts_exact_human_body_and_does_not_patch() -> None:
    fake = FakeHTTP()
    body = "<!-- robot_sf_audit_finding:v1 repository=ll7/robot_sf_ll7 finding_id=f-3 -->\n\nHuman text"
    fake.add("POST", "/repos/ll7/robot_sf_ll7/issues", HttpResponse(201, _issue(9, body=body)))

    issue = _provider(fake).create_issue(
        REPOSITORY,
        title="audit issue",
        body=body,
        labels=("benchmark-audit",),
    )

    assert issue.body == body
    assert len(fake.calls) == 1
    method, _url, request_body = fake.calls[0]
    assert method == "POST"
    assert json.loads(request_body.decode("utf-8"))["body"] == body
    with pytest.raises(GitHubRestUnsupportedError):
        _provider(fake).update_issue(REPOSITORY, 9, body="unsafe")


def test_canonical_revision_create_posts_immutable_issue_without_patch() -> None:
    fake = FakeHTTP()
    body = "immutable body"
    fake.add(
        "POST",
        "/repos/ll7/robot_sf_ll7/issues",
        HttpResponse(201, _issue(9, body=body)),
    )

    issue = _provider(fake).create_issue_with_finding_revision(
        REPOSITORY,
        finding_id="f-rest-create",
        expected_finding_revision=0,
        title="audit issue",
        body=body,
        labels=("benchmark-audit",),
    )

    assert issue.number == 9
    assert issue.body == body
    assert [method for method, _url, _body in fake.calls] == ["POST"]


@pytest.mark.parametrize(
    ("finding_id", "expected_finding_revision"),
    [
        ("", 0),
        ("f-invalid", -1),
        ("f-invalid", True),
    ],
)
def test_canonical_revision_create_rejects_malformed_reservation_before_http(
    finding_id: str,
    expected_finding_revision: Any,
) -> None:
    fake = FakeHTTP()

    with pytest.raises(GitHubRestValidationError):
        _provider(fake).create_issue_with_finding_revision(
            REPOSITORY,
            finding_id=finding_id,
            expected_finding_revision=expected_finding_revision,
            title="audit issue",
            body="body",
            labels=("benchmark-audit",),
        )

    assert fake.calls == []


def test_append_comment_is_idempotent_and_preserves_human_comments() -> None:
    fake = FakeHTTP()
    human = _comment(1, "Human comment remains unchanged")
    fake.comment_store.append(human)
    provider = _provider(fake)
    digest = "a" * 64

    first = provider.append_auditor_comment(REPOSITORY, 7, request_digest=digest, body="Finding")
    second = provider.append_auditor_comment(REPOSITORY, 7, request_digest=digest, body="Finding")

    marker = auditor_request_marker(digest)
    assert first.status == "created"
    assert second.status == "unchanged"
    assert first.body.startswith(marker)
    assert fake.comment_store[0]["body"] == human["body"]
    assert len([call for call in fake.calls if call[0] == "POST"]) == 1


def test_append_comment_reconciles_timeout_after_success_without_retry() -> None:
    fake = FakeHTTP()
    fake.timeout_after_comment = True
    provider = _provider(fake)

    result = provider.append_auditor_comment(
        REPOSITORY,
        7,
        request_digest="b" * 64,
        body="auditor finding",
    )

    assert result.status == "reconciled"
    assert result.reconciled
    assert len([call for call in fake.calls if call[0] == "POST"]) == 1
    assert len([call for call in fake.calls if call[0] == "GET"]) == 2


def test_append_comment_timeout_without_marker_is_ambiguous_and_not_retried() -> None:
    fake = FakeHTTP()
    fake.add("GET", "/repos/ll7/robot_sf_ll7/issues/7/comments", HttpResponse(200, []))
    fake.timeout_after_comment = False

    class TimeoutWithoutCommit(FakeHTTP):
        def request(self, method: str, url: str, **kwargs: Any) -> HttpResponse:
            if method == "POST":
                self.calls.append((method, url, kwargs.get("body")))
                raise TimeoutError("request outcome unknown")
            return super().request(method, url, **kwargs)

    timeout_fake = TimeoutWithoutCommit()
    timeout_fake.add("GET", "/repos/ll7/robot_sf_ll7/issues/7/comments", HttpResponse(200, []))
    timeout_provider = _provider(timeout_fake)
    with pytest.raises(GitHubRestAmbiguousError):
        timeout_provider.append_auditor_comment(
            REPOSITORY,
            7,
            request_digest="c" * 64,
            body="auditor finding",
        )
    assert len([call for call in timeout_fake.calls if call[0] == "POST"]) == 1


def test_allowlist_and_malformed_comment_response_fail_closed() -> None:
    fake = FakeHTTP()
    provider = _provider(fake)
    with pytest.raises(GitHubRestValidationError):
        provider.list_comments("other/repository", 1)

    malformed = FakeHTTP()
    malformed.add("GET", "/repos/ll7/robot_sf_ll7/issues/1/comments", HttpResponse(200, [{}]))
    with pytest.raises(GitHubRestValidationError):
        _provider(malformed).list_comments(REPOSITORY, 1)


def test_marker_validation_rejects_duplicate_and_malformed_input() -> None:
    with pytest.raises(GitHubRestValidationError):
        auditor_request_marker("contains spaces")
    fake = FakeHTTP()
    fake.add("GET", "/repos/ll7/robot_sf_ll7/issues/7/comments", HttpResponse(200, []))
    with pytest.raises(GitHubRestDuplicateMarkerError):
        _provider(fake).append_auditor_comment(
            REPOSITORY,
            7,
            request_digest="d" * 64,
            body=auditor_request_marker("d" * 64) * 2,
        )


def test_marker_and_comment_write_value_contracts_are_strict() -> None:
    marker = auditor_request_marker("request-17")
    assert parse_auditor_request_marker(marker) == "request-17"
    result = GitHubCommentWrite(_comment(7, marker), marker, "created")
    assert result.id == 7
    assert result.body == marker
    assert result.to_dict()["reconciled"] is False

    with pytest.raises(GitHubRestValidationError):
        auditor_request_marker(None)  # type: ignore[arg-type]
    with pytest.raises(GitHubRestValidationError):
        parse_auditor_request_marker(None)  # type: ignore[arg-type]
    with pytest.raises(GitHubRestValidationError):
        parse_auditor_request_marker("<!-- malformed -->")
    with pytest.raises(GitHubRestValidationError):
        GitHubCommentWrite(_comment(7, marker), marker, "failed")
    with pytest.raises(GitHubRestValidationError):
        GitHubCommentWrite([], marker, "created")  # type: ignore[arg-type]
    with pytest.raises(GitHubRestValidationError):
        GitHubCommentWrite(_comment(7, marker), "", "created")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allowed_repositories": "ll7/robot_sf_ll7"},
        {"allowed_repositories": ()},
        {"allowed_repositories": ("../repo",)},
        {"allowed_repositories": (REPOSITORY,), "api_base_url": "ftp://api.github.test"},
        {
            "allowed_repositories": (REPOSITORY,),
            "api_base_url": "https://user:pass@api.github.test",
        },
        {"allowed_repositories": (REPOSITORY,), "api_base_url": "https://api.github.test/base"},
        {"allowed_repositories": (REPOSITORY,), "timeout_s": 0},
        {"allowed_repositories": (REPOSITORY,), "per_page": 101},
        {"allowed_repositories": (REPOSITORY,), "max_pages": 0},
        {"allowed_repositories": (REPOSITORY,), "max_response_bytes": 0},
        {"allowed_repositories": (REPOSITORY,), "max_total_bytes": 0},
        {"allowed_repositories": (REPOSITORY,), "max_search_results": 1001},
        {"allowed_repositories": (REPOSITORY,), "token": 1},
    ],
)
def test_constructor_rejects_unsafe_transport_configuration(kwargs: dict[str, Any]) -> None:
    fake = FakeHTTP()
    kwargs.setdefault("api_base_url", API_ORIGIN)
    with pytest.raises(GitHubRestValidationError):
        GitHubRESTProvider(fake, **kwargs)


def test_response_coercion_accepts_injected_response_shapes_and_rejects_bad_bytes() -> None:
    provider = _provider(FakeHTTP())

    class ReadWithSize:
        status_code = 200
        headers = [("X-Test", "yes")]

        def read(self, size: int) -> bytes:
            assert size > 0
            return b"{}"

    class ReadWithoutSize:
        status = 200
        headers = {}

        def read(self) -> str:
            return "{}"

    class JsonResponse:
        status_code = 200
        headers = {}

        def json(self) -> dict[str, Any]:
            return {}

    assert (
        provider._coerce_response({"status": 200, "headers": [("X", "y")], "body": {}}).payload
        == {}
    )
    assert provider._coerce_response(ReadWithSize()).payload == {}
    assert provider._coerce_response(ReadWithoutSize()).payload == {}
    assert provider._coerce_response(JsonResponse()).payload == {}

    with pytest.raises(GitHubRestValidationError):
        provider._coerce_response({"status": True, "body": {}})
    with pytest.raises(GitHubRestValidationError):
        provider._coerce_response({"status": 200, "headers": "bad", "body": {}})
    with pytest.raises(GitHubRestValidationError):
        provider._coerce_response({"status": 200, "body": b"not-json"})
    with pytest.raises(GitHubRestValidationError):
        provider._coerce_response({"status": 200, "body": None})
    with pytest.raises(GitHubRestLimitError):
        _provider(FakeHTTP(), max_response_bytes=2)._coerce_response(
            {"status": 200, "body": b"{}x"}
        )


def test_response_and_request_status_errors_are_fail_closed() -> None:
    missing = FakeHTTP()
    missing.add("GET", "/repos/ll7/robot_sf_ll7/issues/7", HttpResponse(404, {"message": "gone"}))
    with pytest.raises(GitHubIssueMissing) as missing_error:
        _provider(missing).get_issue(REPOSITORY, 7)
    assert "gone" not in str(missing_error.value)

    server_error = FakeHTTP()
    server_error.add(
        "GET", "/repos/ll7/robot_sf_ll7/issues/7", HttpResponse(500, {"message": "bad"})
    )
    with pytest.raises(GitHubRestHTTPError):
        _provider(server_error).get_issue(REPOSITORY, 7)

    with pytest.raises(GitHubRestUnsupportedError):
        _provider(FakeHTTP())._request("PATCH", "/repos/ll7/robot_sf_ll7/issues/7")


def test_malformed_issue_and_create_response_fields_are_rejected() -> None:
    malformed = FakeHTTP()
    malformed.add(
        "GET",
        "/search/issues",
        HttpResponse(
            200,
            {"total_count": 1, "incomplete_results": False, "items": [{"comments": -1}]},
        ),
    )
    with pytest.raises(GitHubRestValidationError):
        _provider(malformed).search_issues(REPOSITORY, marker="stable")

    mismatch = FakeHTTP()
    mismatch.add(
        "POST", "/repos/ll7/robot_sf_ll7/issues", HttpResponse(201, _issue(8, body="other"))
    )
    with pytest.raises(GitHubRestValidationError):
        _provider(mismatch).create_issue(
            REPOSITORY,
            title="audit issue",
            body="requested",
            labels=("benchmark-audit",),
        )


def test_pagination_integrity_rejects_bad_links_changed_totals_and_limits() -> None:
    malformed_link = FakeHTTP()
    malformed_link.add(
        "GET",
        "/search/issues",
        HttpResponse(
            200,
            {"total_count": 0, "incomplete_results": False, "items": []},
            {"Link": "not-a-link"},
        ),
    )
    with pytest.raises(GitHubRestPaginationError):
        _provider(malformed_link).search_issues(REPOSITORY, marker="stable")

    duplicate_next = FakeHTTP()
    duplicate_next.add(
        "GET",
        "/search/issues",
        HttpResponse(
            200,
            {"total_count": 1, "incomplete_results": False, "items": []},
            {
                "Link": (
                    f'<{API_ORIGIN}/search/issues?page=2>; rel="next", '
                    f'<{API_ORIGIN}/search/issues?page=2>; rel="next"'
                )
            },
        ),
    )
    with pytest.raises(GitHubRestPaginationError):
        _provider(duplicate_next).search_issues(REPOSITORY, marker="stable")

    changed_total = FakeHTTP()
    changed_total.add(
        "GET",
        "/search/issues",
        HttpResponse(
            200,
            {"total_count": 2, "incomplete_results": False, "items": []},
            {
                "Link": _link(
                    "/search/issues",
                    2,
                    query={"q": 'repo:ll7/robot_sf_ll7 "stable"', "per_page": "100"},
                )
            },
        ),
    )
    changed_total.add(
        "GET",
        "/search/issues",
        HttpResponse(200, {"total_count": 1, "incomplete_results": False, "items": []}),
        page=2,
    )
    with pytest.raises(GitHubRestPaginationError):
        _provider(changed_total).search_issues(REPOSITORY, marker="stable")

    changed_query = FakeHTTP()
    changed_query.add(
        "GET",
        "/search/issues",
        HttpResponse(
            200,
            {"total_count": 2, "incomplete_results": False, "items": []},
            {"Link": f'<{API_ORIGIN}/search/issues?page=2&q=other>; rel="next"'},
        ),
    )
    with pytest.raises(GitHubRestPaginationError):
        _provider(changed_query).search_issues(REPOSITORY, marker="stable")

    max_pages = FakeHTTP()
    max_pages.add(
        "GET",
        "/repos/ll7/robot_sf_ll7/issues",
        HttpResponse(
            200,
            [_issue(1)],
            {
                "Link": _link(
                    "/repos/ll7/robot_sf_ll7/issues",
                    2,
                    query={"state": "all", "per_page": "2"},
                )
            },
        ),
    )
    with pytest.raises(GitHubRestLimitError):
        _provider(max_pages, per_page=2, max_pages=1).list_issues(REPOSITORY)

    too_many_items = FakeHTTP()
    too_many_items.add(
        "GET",
        "/repos/ll7/robot_sf_ll7/issues",
        HttpResponse(200, [_issue(1), _issue(2)]),
    )
    with pytest.raises(GitHubRestPaginationError):
        _provider(too_many_items, per_page=1).list_issues(REPOSITORY)


def test_comment_post_response_marker_loss_and_duplicate_history_fail_closed() -> None:
    class MarkerLoss(FakeHTTP):
        def request(self, method: str, url: str, **kwargs: Any) -> HttpResponse:
            if method == "POST":
                self.calls.append((method, url, kwargs.get("body")))
                return HttpResponse(201, _comment(50, "server removed marker"))
            return super().request(method, url, **kwargs)

    marker_loss = MarkerLoss()
    with pytest.raises(GitHubRestValidationError):
        _provider(marker_loss).append_auditor_comment(
            REPOSITORY,
            7,
            request_digest="e" * 64,
            body="auditor body",
        )

    marker = auditor_request_marker("f" * 64)
    duplicate = FakeHTTP()
    duplicate.comment_store.extend([_comment(1, marker), _comment(2, marker)])
    with pytest.raises(GitHubRestDuplicateMarkerError):
        _provider(duplicate).append_auditor_comment(
            REPOSITORY,
            7,
            request_digest="f" * 64,
            body="auditor body",
        )


def test_comment_timeout_reconciliation_rejects_truncated_or_duplicate_history() -> None:
    class DuplicateAfterTimeout(FakeHTTP):
        def request(self, method: str, url: str, **kwargs: Any) -> HttpResponse:
            if method == "POST":
                payload = json.loads(kwargs["body"].decode("utf-8"))
                marker = payload["body"].split("\n", 1)[0]
                self.comment_store.extend([_comment(20, marker), _comment(21, marker)])
                self.calls.append((method, url, kwargs.get("body")))
                raise TimeoutError("accepted but response lost")
            return super().request(method, url, **kwargs)

    duplicate = DuplicateAfterTimeout()
    with pytest.raises(GitHubRestAmbiguousError):
        _provider(duplicate).append_auditor_comment(
            REPOSITORY,
            7,
            request_digest="1" * 64,
            body="auditor body",
        )

    truncated = FakeHTTP()
    truncated.timeout_after_comment = True
    with pytest.raises(GitHubRestAmbiguousError):
        _provider(truncated, per_page=1).append_auditor_comment(
            REPOSITORY,
            7,
            request_digest="2" * 64,
            body="auditor body",
        )


def test_transport_errors_are_wrapped_and_comment_history_validated() -> None:
    class BrokenHTTP(FakeHTTP):
        def request(self, method: str, url: str, **kwargs: Any) -> HttpResponse:
            self.calls.append((method, url, kwargs.get("body")))
            raise ConnectionError("offline")

    with pytest.raises(TimeoutError):
        _provider(BrokenHTTP()).list_comments(REPOSITORY, 7)

    malformed = FakeHTTP()
    malformed.add(
        "GET",
        "/repos/ll7/robot_sf_ll7/issues/7/comments",
        HttpResponse(200, [{"id": 1, "body": 3}]),
    )
    with pytest.raises(GitHubRestValidationError):
        _provider(malformed).list_comments(REPOSITORY, 7)

    duplicate_ids = FakeHTTP()
    duplicate_ids.add(
        "GET",
        "/repos/ll7/robot_sf_ll7/issues/7/comments",
        HttpResponse(200, [_comment(1, "one"), _comment(1, "same id")]),
    )
    with pytest.raises(GitHubRestPaginationError):
        _provider(duplicate_ids).list_comments(REPOSITORY, 7)


def test_pagination_byte_bound_is_enforced() -> None:
    fake = FakeHTTP()
    fake.add(
        "GET", "/repos/ll7/robot_sf_ll7/issues/7/comments", HttpResponse(200, [_comment(1, "x")])
    )

    with pytest.raises(GitHubRestLimitError):
        _provider(fake, max_response_bytes=8).list_comments(REPOSITORY, 7)
