"""Authenticated BA-05 GitHub service boundary tests."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

from robot_sf.analysis_workbench.audit_contracts import Finding
from robot_sf.analysis_workbench.audit_findings import add_candidate, new_finding
from robot_sf.analysis_workbench.audit_github import (
    AUDITOR_BLOCK_START,
    GitHubCapabilityUnavailable,
    GitHubConflictError,
    GitHubIssue,
    SearchResult,
    render_finding_issue,
)
from robot_sf.analysis_workbench.audit_github_rest import (
    GitHubRESTProvider,
    HttpResponse,
    auditor_request_marker,
)
from robot_sf.analysis_workbench.audit_mcp import (
    AuditMCPDispatcher,
    AuditMCPRequest,
)
from robot_sf.analysis_workbench.audit_mcp_stdio import AuditMCPStdioServer
from robot_sf.analysis_workbench.audit_service import (
    AuditSelectionContext,
    AuditService,
    AuditValidationError,
    ServiceResult,
    SessionPolicy,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "audit_campaign_v1"
    / "campaign.json"
)
REPOSITORY = "ll7/robot_sf_ll7"


class ServiceFakeProvider:
    """Provider fake with complete search and explicit body CAS."""

    def __init__(self, *, issues: list[GitHubIssue] | None = None) -> None:
        """Initialize remote issue state and mutation call logs."""

        self.issues = list(issues or [])
        self.create_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []
        self.search_complete = True
        self.search_incomplete_after_create = False
        self.create_timeout_after_success = False
        self.next_number = 100
        self.finding_revision_lookup: Any = None

    def _check_finding_revision(self, finding_id: str, expected_finding_revision: int) -> None:
        """Model the provider's canonical finding-revision CAS seam."""

        if self.finding_revision_lookup is None:
            return
        actual = self.finding_revision_lookup(finding_id)
        if actual != expected_finding_revision:
            raise GitHubConflictError("canonical finding revision changed before provider mutation")

    def search_issues(self, repository: str, *, marker: str) -> SearchResult:
        del marker
        return SearchResult(tuple(self.issues), complete=self.search_complete)

    def create_issue(
        self,
        repository: str,
        *,
        title: str,
        body: str,
        labels: tuple[str, ...],
    ) -> GitHubIssue:
        self.create_calls.append(
            {"repository": repository, "title": title, "body": body, "labels": labels}
        )
        issue = GitHubIssue(
            repository=repository,
            number=self.next_number,
            url=f"https://github.com/{repository}/issues/{self.next_number}",
            title=title,
            body=body,
            labels=labels,
            updated_at=str(self.next_number),
        )
        self.next_number += 1
        self.issues.append(issue)
        if self.create_timeout_after_success:
            if self.search_incomplete_after_create:
                self.search_complete = False
            raise TimeoutError("response lost after accepted create")
        return issue

    def create_issue_with_finding_revision(
        self,
        repository: str,
        *,
        finding_id: str,
        expected_finding_revision: int,
        title: str,
        body: str,
        labels: tuple[str, ...],
    ) -> GitHubIssue:
        self._check_finding_revision(finding_id, expected_finding_revision)
        return self.create_issue(repository, title=title, body=body, labels=labels)

    def get_issue(self, repository: str, number: int) -> GitHubIssue:
        for issue in self.issues:
            if issue.repository == repository and issue.number == number:
                return issue
        raise LookupError(number)

    def update_issue_if_unchanged(
        self,
        repository: str,
        number: int,
        *,
        body: str,
        expected_body_digest: str,
        expected_updated_at: str,
    ) -> GitHubIssue:
        current = self.get_issue(repository, number)
        if hashlib.sha256(current.body.encode("utf-8")).hexdigest() != expected_body_digest:
            raise GitHubConflictError("body CAS failed")
        if current.updated_at != expected_updated_at:
            raise GitHubConflictError("issue revision CAS failed")
        updated = replace(current, body=body, updated_at=f"{number}:updated")
        self.update_calls.append({"repository": repository, "number": number, "body": body})
        self.issues = [updated if item.number == number else item for item in self.issues]
        return updated

    def update_issue_with_finding_revision(
        self,
        repository: str,
        number: int,
        *,
        finding_id: str,
        expected_finding_revision: int,
        body: str,
        expected_body_digest: str,
        expected_updated_at: str,
    ) -> GitHubIssue:
        self._check_finding_revision(finding_id, expected_finding_revision)
        return self.update_issue_if_unchanged(
            repository,
            number,
            body=body,
            expected_body_digest=expected_body_digest,
            expected_updated_at=expected_updated_at,
        )


class AppendOnlyServiceFakeProvider(ServiceFakeProvider):
    """Service fake exposing only immutable issue plus append-comment writes."""

    def __init__(self, *, issues: list[GitHubIssue] | None = None) -> None:
        """Initialize append-only comments and their deterministic receipts."""

        super().__init__(issues=issues)
        self.comment_calls: list[dict[str, Any]] = []
        self.reconciled_comment_result = False

    def append_auditor_comment(
        self,
        repository: str,
        number: int,
        *,
        body: str,
        request_digest: str,
    ) -> dict[str, Any]:
        """Append one immutable comment, optionally returning reconciliation."""

        issue = self.get_issue(repository, number)
        marker = auditor_request_marker(request_digest)
        for comment in issue.comments:
            if str(comment.get("body", "")).count(marker) == 1:
                return {"status": "unchanged", "comment": comment}
        comment = {
            "id": 1000 + len(issue.comments),
            "body": f"{marker}\n{body}",
        }
        self.comment_calls.append({"number": number, "body": body})
        self.issues = [
            replace(issue, comments=(*issue.comments, comment)) if item.number == number else item
            for item in self.issues
        ]
        status = "reconciled" if self.reconciled_comment_result else "created"
        self.reconciled_comment_result = False
        return {"status": status, "comment": comment}

    def update_issue_if_unchanged(self, *_args: Any, **_kwargs: Any) -> GitHubIssue:
        """Reject the legacy body-CAS path so this fake cannot mask routing."""

        raise AssertionError("append-only service fake attempted an issue-body update")


def _setup(
    tmp_path: Path,
    provider: ServiceFakeProvider | None = None,
    *,
    issue_write_budget: int = 2,
) -> tuple[AuditService, Any, Finding, int, ServiceFakeProvider]:
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    fake = provider or ServiceFakeProvider()
    service = AuditService(
        tmp_path / "store",
        campaign_source=source,
        source_root=root,
        github_provider=fake,
    )
    context = AuditSelectionContext(campaign_id="audit-fixture-campaign")
    session = service.open_session(
        context,
        actor="agent",
        actor_id="github-service-test",
        policy=SessionPolicy(
            allowed_roots=(str(root),),
            allowed_repositories=(REPOSITORY,),
            issue_write_budget=issue_write_budget,
        ),
    )
    finding = add_candidate(new_finding("service-finding", "turning symptom"), "episode-1")
    commit = service.finding_store.create(
        finding,
        operation_id="create-service-finding",
        actor="human",
    )
    fake.finding_revision_lookup = lambda finding_id: (
        service.store.get(finding_id).revision
        if service.store.get(finding_id) is not None
        else None
    )
    return service, session, finding, commit.revision, fake


def _setup_known_source(
    tmp_path: Path,
    provider: ServiceFakeProvider | None = None,
) -> tuple[AuditService, Any, Finding, int, ServiceFakeProvider]:
    """Build a service whose source and canonical finding carry a revision token."""

    fake = provider or ServiceFakeProvider()
    source = {
        "campaign_id": "audit-fixture-campaign",
        "source_revision": "source-v1",
    }
    service = AuditService(tmp_path / "store", campaign_source=source, github_provider=fake)
    session = service.open_session(
        AuditSelectionContext(campaign_id="audit-fixture-campaign"),
        actor="agent",
        actor_id="github-source-test",
        policy=SessionPolicy(
            allowed_repositories=(REPOSITORY,),
            issue_write_budget=4,
        ),
    )
    finding = replace(
        add_candidate(new_finding("source-finding", "source-bound symptom"), "episode-1"),
        source_revision="source-v1",
    )
    commit = service.finding_store.create(
        finding,
        operation_id="create-source-finding",
        actor="human",
    )
    fake.finding_revision_lookup = lambda finding_id: (
        service.store.get(finding_id).revision
        if service.store.get(finding_id) is not None
        else None
    )
    return service, session, finding, commit.revision, fake


def test_service_sync_uses_canonical_finding_and_durable_replay(tmp_path: Path) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:
        first = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            evidence={"campaign_id": "audit-fixture-campaign"},
            operation_id="service-sync",
        )
        assert first.status == "committed"
        assert first.value is not None and first.value.status == "created"
        assert len(provider.create_calls) == 1
        assert service.get_session(session).usage.issue_writes == 1

        replay = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            evidence={"campaign_id": "audit-fixture-campaign"},
            operation_id="service-sync",
        )
        assert replay.status == "committed"
        assert replay.operation is not None and replay.operation.replayed
        assert len(provider.create_calls) == 1
    finally:
        service.close()


def test_service_append_only_reconciled_comment_consumes_reserved_write(
    tmp_path: Path,
) -> None:
    provider = AppendOnlyServiceFakeProvider()
    service, session, finding, revision, provider = _setup(
        tmp_path,
        provider=provider,
        issue_write_budget=2,
    )
    try:
        initial = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="append-service-initial",
        )
        assert initial.status == "committed"
        assert initial.value is not None and initial.value.status == "created"
        assert service.get_session(session).usage.issue_writes == 1

        stored_record = service.store.get(finding.finding_id)
        assert stored_record is not None and isinstance(stored_record.record, Finding)
        revised = replace(stored_record.record, observations=("service append revision",))
        revised_commit = service.finding_store.update(
            revised,
            operation_id="append-service-revision",
            expected_revision=stored_record.revision,
            actor="human",
        )
        provider.reconciled_comment_result = True
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revised_commit.revision,
            operation_id="append-service-comment",
        )

        assert result.status == "committed"
        assert result.value is not None and result.value.status == "reconciled"
        assert result.value.remote_write == "applied"
        assert len(provider.comment_calls) == 1
        assert service.get_session(session).usage.issue_writes == 2
    finally:
        service.close()


def test_service_rest_initial_create_uses_append_only_provider_without_patch(
    tmp_path: Path,
) -> None:
    class InitialCreateHTTP:
        """Injected REST transport proving service-backed initial publication."""

        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []
            self.issue: dict[str, Any] | None = None

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
            path = urlparse(url).path
            self.calls.append((method, path))
            if method == "POST":
                assert body is not None
                payload = json.loads(body.decode("utf-8"))
                number = 17
                self.issue = {
                    "repository": REPOSITORY,
                    "number": number,
                    "html_url": f"https://github.com/{REPOSITORY}/issues/{number}",
                    "title": payload["title"],
                    "body": payload["body"],
                    "labels": [{"name": label} for label in payload["labels"]],
                    "state": "open",
                    "comments": 0,
                    "updated_at": "2026-09-23T00:00:00Z",
                }
                return HttpResponse(201, self.issue)
            assert method == "GET"
            if path == "/search/issues":
                items = [self.issue] if self.issue is not None else []
                return HttpResponse(
                    200,
                    {"total_count": len(items), "incomplete_results": False, "items": items},
                )
            if self.issue is not None:
                issue_path = f"/repos/{REPOSITORY}/issues/{self.issue['number']}"
                if path == issue_path:
                    return HttpResponse(200, self.issue)
                if path == f"{issue_path}/comments":
                    return HttpResponse(200, [])
            raise AssertionError(f"unexpected injected HTTP request: {method} {url}")

    http = InitialCreateHTTP()
    provider = GitHubRESTProvider(
        http,
        allowed_repositories=(REPOSITORY,),
        api_base_url="https://api.github.test",
    )
    service, session, finding, revision, _provider = _setup(tmp_path, provider=provider)
    try:
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="rest-initial-create",
        )

        assert result.status == "committed"
        assert result.value is not None and result.value.status == "created"
        assert result.value.remote_write == "applied"
        assert result.value.outbox is not None and result.value.outbox.state == "succeeded"
        assert [method for method, _path in http.calls].count("POST") == 1
        assert [method for method, _path in http.calls].count("PATCH") == 0
        assert service.get_session(session).usage.issue_writes == 1
        assert not service.authority.snapshot()["reservations"]
        assert not service._reservations

    finally:
        service.close()


def test_service_rest_accepts_canonical_long_finding_id(tmp_path: Path) -> None:
    class LongFindingHTTP:
        """Transport proving canonical marker-safe IDs reach the REST POST."""

        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []
            self.issue: dict[str, Any] | None = None

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
            path = urlparse(url).path
            self.calls.append((method, path))
            if method == "POST":
                assert body is not None
                payload = json.loads(body.decode("utf-8"))
                number = 24
                self.issue = {
                    "repository": REPOSITORY,
                    "number": number,
                    "html_url": f"https://github.com/{REPOSITORY}/issues/{number}",
                    "title": payload["title"],
                    "body": payload["body"],
                    "labels": [{"name": label} for label in payload["labels"]],
                    "state": "open",
                    "comments": 0,
                    "updated_at": "2026-09-23T00:00:00Z",
                }
                return HttpResponse(201, self.issue)
            assert method == "GET"
            if path == "/search/issues":
                items = [self.issue] if self.issue is not None else []
                return HttpResponse(
                    200,
                    {"total_count": len(items), "incomplete_results": False, "items": items},
                )
            if self.issue is not None:
                issue_path = f"/repos/{REPOSITORY}/issues/{self.issue['number']}"
                if path == issue_path:
                    return HttpResponse(200, self.issue)
                if path == f"{issue_path}/comments":
                    return HttpResponse(200, [])
            raise AssertionError(f"unexpected injected HTTP request: {method} {url}")

    http = LongFindingHTTP()
    provider = GitHubRESTProvider(
        http,
        allowed_repositories=(REPOSITORY,),
        api_base_url="https://api.github.test",
    )
    service, session, _finding, _revision, _provider = _setup(tmp_path, provider=provider)
    long_finding = add_candidate(new_finding("f" * 129, "long finding"), "episode-long")
    commit = service.finding_store.create(
        long_finding,
        operation_id="create-long-finding",
        actor="human",
    )
    try:
        result = service.sync_finding(
            session,
            finding_id=long_finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=commit.revision,
            operation_id="rest-long-finding-id",
        )

        assert result.status == "committed"
        assert result.value is not None and result.value.status == "created"
        assert result.value.remote_write == "applied"
        assert http.issue is not None and ("finding_id=" + "f" * 129) in http.issue["body"]
        assert [method for method, _path in http.calls].count("POST") == 1
        assert [method for method, _path in http.calls].count("PATCH") == 0
        assert service.get_session(session).usage.issue_writes == 1
        assert not service.authority.snapshot()["reservations"]
        assert not service._reservations

    finally:
        service.close()


def test_service_rest_create_payload_conflict_charges_post_and_stops_retry(
    tmp_path: Path,
) -> None:
    class MismatchedCreateHTTP:
        """Transport returning a marker-compatible but immutable-mismatched issue."""

        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []
            self.issue: dict[str, Any] | None = None

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
            path = urlparse(url).path
            self.calls.append((method, path))
            if method == "POST":
                assert body is not None
                payload = json.loads(body.decode("utf-8"))
                number = 23
                self.issue = {
                    "repository": REPOSITORY,
                    "number": number,
                    "html_url": f"https://github.com/{REPOSITORY}/issues/{number}",
                    "title": f"{payload['title']} (provider mismatch)",
                    "body": payload["body"],
                    "labels": [{"name": label} for label in payload["labels"]],
                    "state": "open",
                    "comments": 0,
                    "updated_at": "2026-09-23T00:00:00Z",
                }
                return HttpResponse(201, self.issue)
            assert method == "GET"
            if path == "/search/issues":
                items = [self.issue] if self.issue is not None else []
                return HttpResponse(
                    200,
                    {"total_count": len(items), "incomplete_results": False, "items": items},
                )
            if self.issue is not None:
                issue_path = f"/repos/{REPOSITORY}/issues/{self.issue['number']}"
                if path == issue_path:
                    return HttpResponse(200, self.issue)
                if path == f"{issue_path}/comments":
                    return HttpResponse(200, [])
            raise AssertionError(f"unexpected injected HTTP request: {method} {url}")

    http = MismatchedCreateHTTP()
    provider = GitHubRESTProvider(
        http,
        allowed_repositories=(REPOSITORY,),
        api_base_url="https://api.github.test",
    )
    service, session, finding, revision, _provider = _setup(tmp_path, provider=provider)
    try:
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="rest-create-payload-conflict",
        )

        assert result.status == "conflict"
        assert result.value is not None and result.value.status == "conflict"
        assert result.value.remote_write == "ambiguous"
        assert result.value.outbox is not None and result.value.outbox.state == "conflict"
        assert service._github_outbox is not None
        claim = service._github_outbox.get_claim(REPOSITORY, finding.finding_id)
        assert claim is not None and claim.state == "conflict"
        assert [method for method, _path in http.calls].count("POST") == 1
        assert [method for method, _path in http.calls].count("PATCH") == 0
        assert service.get_session(session).usage.issue_writes == 1
        assert not service.authority.snapshot()["reservations"]
        assert not service._reservations

        replay = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="rest-create-payload-conflict",
        )
        assert replay.status == "conflict"
        assert replay.operation is not None and replay.operation.replayed
        assert [method for method, _path in http.calls].count("POST") == 1
        assert service.get_session(session).usage.issue_writes == 1
        replay_claim = service._github_outbox.get_claim(REPOSITORY, finding.finding_id)
        assert replay_claim is not None and replay_claim.state == "conflict"
        replay_entry = service._github_outbox.find_publication_kind(
            REPOSITORY,
            finding.finding_id,
            "initial_issue",
        )
        assert replay_entry is not None and replay_entry.state == "conflict"
    finally:
        service.close()


def _respond_with_posted_issue(issue: dict[str, Any]) -> HttpResponse:
    return HttpResponse(200, issue)


def _timeout_posted_issue_readback(_issue: dict[str, Any]) -> HttpResponse:
    raise TimeoutError("issue reread timed out after marker recovery")


@pytest.mark.parametrize(
    (
        "persist_post",
        "timeout_issue_get",
        "expected_result_status",
        "expected_value_status",
        "expected_remote_write",
        "expected_outbox_state",
    ),
    [
        (False, False, "unavailable", "ambiguous", "ambiguous", "ambiguous"),
        (True, False, "committed", "reconciled", "applied", "succeeded"),
        (True, True, "unavailable", "ambiguous", "ambiguous", "ambiguous"),
    ],
)
def test_service_rest_post_timeout_is_accounted_after_readback(
    tmp_path: Path,
    persist_post: bool,
    timeout_issue_get: bool,
    expected_result_status: str,
    expected_value_status: str,
    expected_remote_write: str,
    expected_outbox_state: str,
) -> None:
    issue_get_handler = {
        False: _respond_with_posted_issue,
        True: _timeout_posted_issue_readback,
    }[timeout_issue_get]

    class PostTimeoutHTTP:
        """Inject a lost POST response and an optional issue-readback timeout."""

        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []
            self.issue: dict[str, Any] | None = None
            self.issue_get_handler = issue_get_handler

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
            self.calls.append((method, url))
            path = urlparse(url).path
            if method == "POST":
                assert body is not None
                payload = json.loads(body.decode("utf-8"))
                if persist_post:
                    number = 17
                    self.issue = {
                        "number": number,
                        "html_url": f"https://github.com/{REPOSITORY}/issues/{number}",
                        "title": payload["title"],
                        "body": payload["body"],
                        "labels": [{"name": label} for label in payload["labels"]],
                        "state": "open",
                        "comments": 0,
                        "updated_at": "2026-09-23T00:00:00Z",
                    }
                raise TimeoutError("response lost after POST began")
            assert method == "GET"
            if path == "/search/issues":
                items = [self.issue] if self.issue is not None else []
                return HttpResponse(
                    200,
                    {"total_count": len(items), "incomplete_results": False, "items": items},
                )
            if self.issue is not None:
                issue_path = f"/repos/{REPOSITORY}/issues/{self.issue['number']}"
                if path == issue_path:
                    return self.issue_get_handler(self.issue)
                if path == f"{issue_path}/comments":
                    return HttpResponse(200, [])
            raise AssertionError(f"unexpected injected HTTP request: {method} {url}")

    http = PostTimeoutHTTP()
    provider = GitHubRESTProvider(
        http,
        allowed_repositories=(REPOSITORY,),
        api_base_url="https://api.github.test",
    )
    service, session, finding, revision, _provider = _setup(tmp_path, provider=provider)
    try:
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="rest-post-timeout",
        )

        assert result.status == expected_result_status
        assert result.value is not None and result.value.status == expected_value_status
        assert result.value.remote_write == expected_remote_write
        assert result.value.outbox is not None
        assert result.value.outbox.state == expected_outbox_state
        assert service._github_outbox is not None
        claim = service._github_outbox.get_claim(REPOSITORY, finding.finding_id)
        assert claim is not None and claim.state == expected_outbox_state
        assert [method for method, _url in http.calls].count("POST") == 1
        assert service.get_session(session).usage.issue_writes == 1
        assert not service.authority.snapshot()["reservations"]
        assert not service._reservations
    finally:
        service.close()


def test_service_denies_wrong_repository_or_exhausted_budget_before_provider(
    tmp_path: Path,
) -> None:
    service, session, finding, revision, provider = _setup(tmp_path, issue_write_budget=0)
    try:
        denied_budget = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="budget-denied",
        )
        assert denied_budget.status == "denied"
        assert provider.create_calls == []

        wrong_repo = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository="other/repository",
            expected_finding_revision=revision,
            operation_id="repo-denied",
        )
        assert wrong_repo.status == "denied"
        assert provider.create_calls == []
    finally:
        service.close()


def test_service_rejects_stale_context_source_and_finding_revision_without_provider_call(
    tmp_path: Path,
) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:
        stale_context = session.context.next_revision(episode_id="other-episode")
        stale = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            context=stale_context,
            expected_finding_revision=revision,
            operation_id="stale-context",
        )
        assert stale.status == "conflict"
        assert provider.create_calls == []

        stale_source = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_source_revision="changed-source",
            expected_finding_revision=revision,
            operation_id="stale-source",
        )
        assert stale_source.status == "conflict"
        assert provider.create_calls == []

        changed = replace(finding, title="new title")
        service.finding_store.update(
            changed,
            operation_id="change-finding",
            expected_revision=revision,
            actor="human",
        )
        stale_finding = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="stale-finding",
        )
        assert stale_finding.status == "conflict"
        assert provider.create_calls == []
    finally:
        service.close()


def test_service_enforces_source_bound_evidence_before_provider_work(tmp_path: Path) -> None:
    service, session, finding, revision, provider = _setup_known_source(tmp_path)
    try:
        good_evidence = {
            "campaign_id": session.context.campaign_id,
            "source_identity": session.source_digest,
            "source_revision": session.source_revision,
            "source_digest": session.source_digest,
        }
        good = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            evidence=good_evidence,
            expected_finding_revision=revision,
            operation_id="source-evidence-good",
        )
        assert good.status == "committed"
        assert provider.create_calls

        for operation_id, field, value in (
            ("source-evidence-campaign", "campaign_id", "other-campaign"),
            ("source-evidence-identity", "source_identity", "other-source"),
            ("source-evidence-revision", "source_revision", "source-v2"),
            ("source-evidence-digest", "source_digest", "other-digest"),
        ):
            stale = dict(good_evidence)
            stale[field] = value
            result = service.sync_finding(
                session,
                finding_id=finding.finding_id,
                repository=REPOSITORY,
                evidence=stale,
                expected_finding_revision=revision + 1,
                operation_id=operation_id,
            )
            assert result.status == "conflict"
        assert len(provider.create_calls) == 1
    finally:
        service.close()


def test_service_rejects_canonical_source_revision_mismatch(tmp_path: Path) -> None:
    service, session, finding, revision, provider = _setup_known_source(tmp_path)
    try:
        changed = replace(finding, source_revision="source-v2")
        updated = service.finding_store.update(
            changed,
            operation_id="change-source-revision",
            expected_revision=revision,
            actor="human",
        )
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=updated.revision,
            operation_id="canonical-source-mismatch",
        )
        assert result.status == "conflict"
        assert provider.create_calls == []
    finally:
        service.close()


def test_service_argument_and_provider_failures_are_receipted_without_remote_work(
    tmp_path: Path,
) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:
        missing_revision = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            operation_id="missing-finding-revision",
        )
        assert missing_revision.status == "failed"

        bad_retry = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            retry_ambiguous="yes",  # type: ignore[arg-type]
            operation_id="bad-retry-flag",
        )
        assert bad_retry.status == "failed"

        bad_worker = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            worker_id=7,  # type: ignore[arg-type]
            operation_id="bad-worker-id",
        )
        assert bad_worker.status == "failed"

        missing = service.sync_finding(
            session,
            finding_id="missing-finding",
            repository=REPOSITORY,
            expected_finding_revision=0,
            operation_id="missing-finding",
        )
        assert missing.status == "unavailable"

        service.github_provider = None
        no_provider = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="no-provider",
        )
        assert no_provider.status == "unavailable"
        assert provider.create_calls == []
    finally:
        service.close()


def test_service_kill_switch_blocks_github_sync_before_provider_work(tmp_path: Path) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:
        stopped = service.kill_switch(session, reason="operator stop", operation_id="kill-sync")
        assert stopped.status == "cancelled"
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="killed-sync",
        )
        assert result.status == "cancelled"
        assert provider.create_calls == []
    finally:
        service.close()


def test_service_kill_switch_wins_after_reservation_without_provider_call(
    tmp_path: Path, monkeypatch: Any
) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:
        reserve = service.reserve_budget

        def reserve_then_cancel(*args: Any, **kwargs: Any) -> ServiceResult[Any]:
            result = reserve(*args, **kwargs)
            stopped = service.kill_switch(session, reason="race stop", operation_id="race-stop")
            assert stopped.status == "cancelled"
            return result

        monkeypatch.setattr(service, "reserve_budget", reserve_then_cancel)
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="cancel-after-reserve",
        )
        assert result.status == "cancelled"
        assert provider.create_calls == []
        assert service.get_session(session).usage.issue_writes == 0
    finally:
        service.close()


def test_service_source_change_after_reservation_is_rejected_before_provider(
    tmp_path: Path, monkeypatch: Any
) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    source = tmp_path / "allowed" / "campaign.json"
    try:
        reserve = service.reserve_budget

        def reserve_then_change_source(*args: Any, **kwargs: Any) -> ServiceResult[Any]:
            result = reserve(*args, **kwargs)
            source.write_bytes(source.read_bytes() + b"\n")
            return result

        monkeypatch.setattr(service, "reserve_budget", reserve_then_change_source)
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="source-after-reserve",
        )
        assert result.status == "conflict"
        assert provider.create_calls == []
        assert service.get_session(session).usage.issue_writes == 0
    finally:
        service.close()


def test_service_context_change_after_reservation_is_rejected_before_provider(
    tmp_path: Path, monkeypatch: Any
) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:
        reserve = service.reserve_budget

        def reserve_then_change_context(*args: Any, **kwargs: Any) -> ServiceResult[Any]:
            result = reserve(*args, **kwargs)
            changed = session.context.next_revision(episode_id="episode-2")
            updated = service.update_context(
                session,
                changed,
                expected_context_revision=session.context.context_revision,
                operation_id="context-after-reserve",
            )
            assert updated.status == "committed"
            return result

        monkeypatch.setattr(service, "reserve_budget", reserve_then_change_context)
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="context-after-reserve-sync",
        )
        assert result.status == "conflict"
        assert provider.create_calls == []
        assert service.get_session(session).usage.issue_writes == 0
    finally:
        service.close()


def test_service_preflight_conflict_after_reservation_is_durable_and_uncharged(
    tmp_path: Path, monkeypatch: Any
) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:
        build_sync = service._github_sync

        def build_racing_sync(*args: Any, **kwargs: Any) -> Any:
            delegate = build_sync(*args, **kwargs)

            class RacingSync:
                def sync(self, *sync_args: Any, **sync_kwargs: Any) -> Any:
                    stored = service.store.get(finding.finding_id)
                    assert stored is not None and isinstance(stored.record, Finding)
                    service.finding_store.update(
                        replace(stored.record, title="canonical finding raced preflight"),
                        operation_id="canonical-preflight-race",
                        expected_revision=stored.revision,
                        actor="human",
                    )
                    return delegate.sync(*sync_args, **sync_kwargs)

            return RacingSync()

        monkeypatch.setattr(service, "_github_sync", build_racing_sync)
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="preflight-conflict-after-reservation",
        )

        assert result.status == "conflict"
        assert result.operation is not None and result.operation.status == "conflict"
        assert provider.create_calls == []
        assert service.get_session(session).usage.issue_writes == 0
        assert not service.authority.snapshot()["reservations"]
        assert not service._reservations
    finally:
        service.close()


def test_service_local_link_conflict_after_reservation_is_durable_and_uncharged(
    tmp_path: Path, monkeypatch: Any
) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:
        build_sync = service._github_sync

        def build_link_conflict_sync(*args: Any, **kwargs: Any) -> Any:
            delegate = build_sync(*args, **kwargs)

            class LinkConflictSync:
                def sync(self, repository: str, value: Finding, **sync_kwargs: Any) -> Any:
                    linked = replace(
                        value,
                        github_issue={"repository": "other/repo", "number": 17},
                    )
                    return delegate.sync(repository, linked, **sync_kwargs)

            return LinkConflictSync()

        monkeypatch.setattr(service, "_github_sync", build_link_conflict_sync)
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="link-conflict-after-reservation",
        )

        assert result.status == "conflict"
        assert result.operation is not None and result.operation.status == "conflict"
        assert provider.create_calls == []
        assert service.get_session(session).usage.issue_writes == 0
        assert not service.authority.snapshot()["reservations"]
        assert not service._reservations
    finally:
        service.close()


def test_service_cancellation_after_provider_send_keeps_paid_metering(
    tmp_path: Path, monkeypatch: Any
) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:
        create = provider.create_issue_with_finding_revision

        def create_then_cancel(*args: Any, **kwargs: Any) -> GitHubIssue:
            issue = create(*args, **kwargs)
            stopped = service.kill_switch(
                session, reason="post-send stop", operation_id="post-stop"
            )
            assert stopped.status == "cancelled"
            return issue

        monkeypatch.setattr(provider, "create_issue_with_finding_revision", create_then_cancel)
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="cancel-after-send",
        )
        assert result.status == "committed"
        assert result.value is not None and result.value.status == "created"
        assert len(provider.create_calls) == 1
        assert service.get_session(session).cancelled
        assert service.get_session(session).usage.issue_writes == 1
    finally:
        service.close()


def test_service_source_change_after_provider_send_is_conflict_and_metered(
    tmp_path: Path, monkeypatch: Any
) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    source = tmp_path / "allowed" / "campaign.json"
    try:
        create = provider.create_issue_with_finding_revision

        def create_then_change_source(*args: Any, **kwargs: Any) -> GitHubIssue:
            issue = create(*args, **kwargs)
            source.write_bytes(source.read_bytes() + b"\n")
            return issue

        monkeypatch.setattr(
            provider, "create_issue_with_finding_revision", create_then_change_source
        )
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="source-after-send",
        )
        assert result.status == "conflict"
        assert result.value is not None and result.value.status == "conflict"
        assert len(provider.create_calls) == 1
        assert service.get_session(session).usage.issue_writes == 1
    finally:
        service.close()


def test_service_provider_guard_rechecks_canonical_finding_revision(
    tmp_path: Path, monkeypatch: Any
) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:
        create = provider.create_issue_with_finding_revision

        def human_update_then_create(*args: Any, **kwargs: Any) -> GitHubIssue:
            stored = service.store.get(finding.finding_id)
            assert stored is not None
            service.finding_store.update(
                replace(stored.record, title="human raced the provider"),
                operation_id="human-race-before-provider",
                expected_revision=stored.revision,
                actor="human",
            )
            return create(*args, **kwargs)

        monkeypatch.setattr(
            provider, "create_issue_with_finding_revision", human_update_then_create
        )
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="finding-cas-race",
        )
        assert result.status == "conflict"
        assert provider.create_calls == []
        assert service.get_session(session).usage.issue_writes == 0
    finally:
        service.close()


@pytest.mark.parametrize(
    ("lease_status", "expected_status"),
    (
        ("cancelled", "cancelled"),
        ("conflict", "conflict"),
        ("denied", "denied"),
        ("unavailable", "unavailable"),
        ("failed", "failed"),
    ),
)
def test_service_fails_closed_when_send_lease_is_not_granted(
    tmp_path: Path,
    monkeypatch: Any,
    lease_status: str,
    expected_status: str,
) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:
        monkeypatch.setattr(
            service,
            "_acquire_github_send_lease",
            lambda *_args, **_kwargs: ServiceResult(
                lease_status, reason=f"test lease {lease_status}", context=session.context
            ),
        )
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id=f"lease-{lease_status}",
        )
        assert result.status == expected_status
        assert provider.create_calls == []
        assert service.get_session(session).usage.issue_writes == 0
    finally:
        service.close()


def test_service_blocks_context_cas_while_send_lease_is_active(tmp_path: Path) -> None:
    service, session, _finding, _revision, _provider = _setup(tmp_path)
    try:
        reserved = service.reserve_budget(
            session, issue_writes=1, operation_id="manual-send-budget"
        )
        assert reserved.status == "committed" and reserved.value is not None
        lease = service._acquire_github_send_lease(
            session,
            reservation_id=reserved.value["reservation_id"],
            reservation_operation_id="manual-send-budget",
            context=session.context,
            source_digest=session.source_digest,
            source_revision=session.source_revision,
            operation_id="manual-send-budget:send",
        )
        assert lease.status == "committed"
        changed = session.context.next_revision(episode_id="episode-2")
        blocked = service.update_context(
            session,
            changed,
            expected_context_revision=session.context.context_revision,
            operation_id="context-while-send",
        )
        assert blocked.status == "conflict"
        settled = service.settle_budget(
            session,
            reserved_tokens=0,
            reserved_compute=0.0,
            reserved_issue_writes=1,
            actual_tokens=0,
            actual_compute=0.0,
            actual_issue_writes=0,
            reservation_id=reserved.value["reservation_id"],
            reservation_operation_id="manual-send-budget",
            operation_id="manual-send-budget:settle",
        )
        assert settled.status == "committed"
    finally:
        service.close()


@pytest.mark.parametrize(
    "lease",
    (
        {"operation_id": "send", "context_revision": 0},
        {
            "operation_id": "send",
            "context_revision": "bad",
            "source_revision": 0,
            "source_digest": "",
        },
        {
            "operation_id": "send",
            "context_revision": 0,
            "source_revision": True,
            "source_digest": "",
        },
        {
            "operation_id": "send",
            "context_revision": 0,
            "source_revision": 0,
            "source_digest": 7,
        },
    ),
)
def test_service_rejects_malformed_persisted_send_lease(lease: dict[str, Any]) -> None:
    with pytest.raises(AuditValidationError):
        AuditService._reservation_send_lease({"send_lease": lease})


def test_service_meters_unexpected_provider_failure_as_ambiguous_write(
    tmp_path: Path, monkeypatch: Any
) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:

        class RaisingSync:
            def sync(self, *_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError("provider process failed after send")

        monkeypatch.setattr(service, "_github_sync", lambda *_args: RaisingSync())
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="provider-failure",
        )
        assert result.status == "failed"
        assert provider.create_calls == []
        assert service.get_session(session).usage.issue_writes == 1
    finally:
        service.close()


def test_service_classifies_provider_failure_with_post_send_drift_as_unavailable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:

        class RaisingSync:
            def sync(self, *_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError("provider response lost")

        monkeypatch.setattr(service, "_github_sync", lambda *_args: RaisingSync())
        monkeypatch.setattr(
            service,
            "_github_post_send_conflict",
            lambda *_args, **_kwargs: "campaign source changed after provider send",
        )
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="provider-ambiguous",
        )
        assert result.status == "unavailable"
        assert provider.create_calls == []
        assert service.get_session(session).usage.issue_writes == 1
    finally:
        service.close()


def test_service_fails_closed_on_malformed_reservation_and_settlement(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:
        monkeypatch.setattr(
            service,
            "reserve_budget",
            lambda *_args, **_kwargs: ServiceResult("committed", {}, context=session.context),
        )
        malformed = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="malformed-reservation-id",
        )
        assert malformed.status == "failed"
        assert provider.create_calls == []

        monkeypatch.setattr(
            service,
            "reserve_budget",
            lambda *_args, **_kwargs: ServiceResult(
                "committed",
                {
                    "reservation_id": "fake-reservation",
                    "reserved": 0,
                },
                context=session.context,
            ),
        )
        settled = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="malformed-settlement",
        )
        assert settled.status == "failed"
        assert provider.create_calls == []

        monkeypatch.setattr(
            service,
            "reserve_budget",
            lambda *_args, **_kwargs: ServiceResult(
                "committed",
                {
                    "reservation_id": "fake-reservation",
                    "reserved": {"tokens": 0, "compute": 0.0, "issue_writes": 1},
                },
                context=session.context,
            ),
        )
        monkeypatch.setattr(
            service,
            "settle_budget",
            lambda *_args, **_kwargs: ServiceResult(
                "denied", reason="test settlement rejection", context=session.context
            ),
        )
        rejected = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=service.store.get(finding.finding_id).revision,
            operation_id="settlement-rejected",
        )
        assert rejected.status == "denied"
    finally:
        service.close()


def test_service_reconciles_timeout_after_create_without_duplicate_issue(tmp_path: Path) -> None:
    provider = ServiceFakeProvider()
    provider.create_timeout_after_success = True
    provider.search_incomplete_after_create = True
    service, session, finding, revision, provider = _setup(tmp_path, provider)
    try:
        first = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="timeout-sync",
        )
        assert first.status == "unavailable"
        assert first.value is not None and first.value.status == "ambiguous"
        assert first.value.remote_write == "ambiguous"
        assert first.value.outbox is not None and first.value.outbox.state == "ambiguous"
        assert service.get_session(session).usage.issue_writes == 1
        assert len(provider.create_calls) == 1

        provider.search_complete = True
        retried = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            retry_ambiguous=True,
            operation_id="timeout-sync",
        )
        assert retried.status == "committed"
        assert retried.value is not None and retried.value.status in {"reconciled", "unchanged"}
        assert len(provider.create_calls) == 1
    finally:
        service.close()


def test_service_charges_create_returned_before_auditor_block_conflict(
    tmp_path: Path, monkeypatch: Any
) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:
        create = provider.create_issue_with_finding_revision

        def create_malformed(*args: Any, **kwargs: Any) -> GitHubIssue:
            issue = create(*args, **kwargs)
            block = issue.body[issue.body.index(AUDITOR_BLOCK_START) :]
            malformed = replace(issue, body=f"{issue.body}\n{block}")
            provider.issues = [
                malformed if item.number == issue.number else item for item in provider.issues
            ]
            return malformed

        monkeypatch.setattr(provider, "create_issue_with_finding_revision", create_malformed)
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="create-returned-invalid",
        )
        assert result.status == "conflict"
        assert result.value is not None and result.value.status == "conflict"
        assert result.value.remote_write == "applied"
        assert len(provider.create_calls) == 1
        assert service.get_session(session).usage.issue_writes == 1
    finally:
        service.close()


def test_service_charges_create_applied_before_provider_conflict(
    tmp_path: Path, monkeypatch: Any
) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:
        create = provider.create_issue_with_finding_revision

        def create_then_conflict(*args: Any, **kwargs: Any) -> GitHubIssue:
            issue = create(*args, **kwargs)
            del issue
            raise GitHubConflictError("provider reported conflict after applying create")

        monkeypatch.setattr(provider, "create_issue_with_finding_revision", create_then_conflict)
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="create-conflict-after-send",
        )
        assert result.status == "conflict"
        assert result.value is not None and result.value.status == "conflict"
        assert result.value.remote_write == "applied"
        assert len(provider.create_calls) == 1
        assert service.get_session(session).usage.issue_writes == 1
    finally:
        service.close()


def test_service_charges_update_applied_before_provider_conflict(
    tmp_path: Path, monkeypatch: Any
) -> None:
    seed_finding = add_candidate(new_finding("service-finding", "turning symptom"), "episode-1")
    rendered = render_finding_issue(seed_finding, repository=REPOSITORY)
    provider = ServiceFakeProvider(
        issues=[
            GitHubIssue(
                repository=REPOSITORY,
                number=100,
                url=f"https://github.com/{REPOSITORY}/issues/100",
                title=rendered.title,
                body=rendered.body,
                updated_at="100",
            )
        ]
    )
    service, session, finding, _revision, provider = _setup(tmp_path, provider)
    try:
        stored = service.store.get(finding.finding_id)
        assert stored is not None and isinstance(stored.record, Finding)
        changed = service.finding_store.update(
            replace(stored.record, title="updated before provider conflict"),
            operation_id="update-conflict-finding",
            expected_revision=stored.revision,
            actor="human",
        )
        update = provider.update_issue_with_finding_revision

        def update_then_conflict(*args: Any, **kwargs: Any) -> GitHubIssue:
            updated = update(*args, **kwargs)
            del updated
            raise GitHubConflictError("provider reported conflict after applying update")

        monkeypatch.setattr(provider, "update_issue_with_finding_revision", update_then_conflict)
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=changed.revision,
            operation_id="update-conflict-after-send",
        )
        assert result.status == "conflict"
        assert result.value is not None and result.value.status == "conflict"
        assert result.value.remote_write == "applied"
        assert len(provider.update_calls) == 1
        assert service.get_session(session).usage.issue_writes == 1
    finally:
        service.close()


def test_service_does_not_charge_update_conflict_rejected_before_provider_send(
    tmp_path: Path, monkeypatch: Any
) -> None:
    seed_finding = add_candidate(new_finding("service-finding", "turning symptom"), "episode-1")
    rendered = render_finding_issue(seed_finding, repository=REPOSITORY)
    provider = ServiceFakeProvider(
        issues=[
            GitHubIssue(
                repository=REPOSITORY,
                number=100,
                url=f"https://github.com/{REPOSITORY}/issues/100",
                title=rendered.title,
                body=rendered.body,
                updated_at="100",
            )
        ]
    )
    service, session, finding, _revision, provider = _setup(tmp_path, provider)
    try:
        stored = service.store.get(finding.finding_id)
        assert stored is not None and isinstance(stored.record, Finding)
        changed = service.finding_store.update(
            replace(stored.record, title="update rejected before send"),
            operation_id="update-rejected-finding",
            expected_revision=stored.revision,
            actor="human",
        )

        def reject_before_send(*_args: Any, **_kwargs: Any) -> GitHubIssue:
            raise GitHubConflictError("provider rejected CAS before update")

        monkeypatch.setattr(provider, "update_issue_with_finding_revision", reject_before_send)
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=changed.revision,
            operation_id="update-rejected-before-send",
        )
        assert result.status == "conflict"
        assert result.value is not None and result.value.status == "conflict"
        assert result.value.remote_write == "none"
        assert provider.update_calls == []
        assert service.get_session(session).usage.issue_writes == 0
    finally:
        service.close()


def test_service_reports_unsupported_update_without_remote_write_or_charge(
    tmp_path: Path, monkeypatch: Any
) -> None:
    seed_finding = add_candidate(new_finding("service-finding", "turning symptom"), "episode-1")
    rendered = render_finding_issue(seed_finding, repository=REPOSITORY)
    provider = ServiceFakeProvider(
        issues=[
            GitHubIssue(
                repository=REPOSITORY,
                number=100,
                url=f"https://github.com/{REPOSITORY}/issues/100",
                title=rendered.title,
                body=rendered.body,
                updated_at="100",
            )
        ]
    )
    service, session, finding, _revision, provider = _setup(tmp_path, provider)
    try:
        stored = service.store.get(finding.finding_id)
        assert stored is not None and isinstance(stored.record, Finding)
        changed = service.finding_store.update(
            replace(stored.record, title="update unavailable before provider send"),
            operation_id="update-capability-finding",
            expected_revision=stored.revision,
            actor="human",
        )

        def reject_before_send(*_args: Any, **_kwargs: Any) -> GitHubIssue:
            raise GitHubCapabilityUnavailable("issue-body update is unsupported")

        monkeypatch.setattr(provider, "update_issue_with_finding_revision", reject_before_send)
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=changed.revision,
            operation_id="update-capability-unavailable",
        )

        assert result.status == "unavailable"
        assert result.value is not None and result.value.status == "unavailable"
        assert result.value.remote_write == "none"
        assert result.value.outbox is not None and result.value.outbox.state == "failed"
        assert provider.update_calls == []
        assert service.get_session(session).usage.issue_writes == 0
        assert not service.authority.snapshot()["reservations"]
        assert not service._reservations
    finally:
        service.close()


def test_service_update_timeout_retry_does_not_double_charge_reconciled_readback(
    tmp_path: Path, monkeypatch: Any
) -> None:
    seed_finding = add_candidate(new_finding("service-finding", "turning symptom"), "episode-1")
    rendered = render_finding_issue(seed_finding, repository=REPOSITORY)
    provider = ServiceFakeProvider(
        issues=[
            GitHubIssue(
                repository=REPOSITORY,
                number=100,
                url=f"https://github.com/{REPOSITORY}/issues/100",
                title=rendered.title,
                body=rendered.body,
                updated_at="100",
            )
        ]
    )
    service, session, finding, _revision, provider = _setup(
        tmp_path, provider, issue_write_budget=4
    )
    try:
        stored = service.store.get(finding.finding_id)
        assert stored is not None and isinstance(stored.record, Finding)
        changed = service.finding_store.update(
            replace(stored.record, title="update timeout title"),
            operation_id="update-timeout-finding",
            expected_revision=stored.revision,
            actor="human",
        )
        update = provider.update_issue_with_finding_revision
        timeout_once = True

        def update_then_timeout(*args: Any, **kwargs: Any) -> GitHubIssue:
            nonlocal timeout_once
            updated = update(*args, **kwargs)
            if timeout_once:
                timeout_once = False
                raise TimeoutError("response lost after accepted update")
            return updated

        monkeypatch.setattr(provider, "update_issue_with_finding_revision", update_then_timeout)
        first = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=changed.revision,
            operation_id="update-timeout-service",
        )
        assert first.status == "unavailable"
        assert first.value is not None and first.value.status == "ambiguous"
        assert first.value.remote_write == "applied"
        assert service.get_session(session).usage.issue_writes == 1

        retried = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=changed.revision,
            retry_ambiguous=True,
            operation_id="update-timeout-service",
        )
        assert retried.status == "committed"
        assert retried.value is not None and retried.value.status == "reconciled"
        assert retried.value.remote_write == "none"
        assert len(provider.update_calls) == 1
        assert service.get_session(session).usage.issue_writes == 1
    finally:
        service.close()


def test_mcp_dispatches_sync_finding_through_same_authenticated_service(tmp_path: Path) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:
        dispatcher = AuditMCPDispatcher(service)
        response = dispatcher.dispatch(
            AuditMCPRequest(
                request_id="mcp-sync",
                session_id=session.session_id,
                session_token=session.session_token,
                origin="http://127.0.0.1",
                operation="sync_finding",
                payload={
                    "finding_id": finding.finding_id,
                    "repository": REPOSITORY,
                    "expected_finding_revision": revision,
                },
            )
        )
        assert response.status == "committed"
        assert response.result["value"]["status"] == "created"
        assert len(provider.create_calls) == 1
    finally:
        service.close()


def test_stdio_registers_and_dispatches_sync_finding_tool(tmp_path: Path) -> None:
    service, session, finding, revision, provider = _setup(tmp_path)
    try:
        server = AuditMCPStdioServer(
            AuditMCPDispatcher(service),
            session_id=session.session_id,
            session_token=session.session_token,
        )
        initialized = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "initialize",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "clientInfo": {"name": "test", "version": "1"},
                },
            }
        )
        assert initialized is not None and "error" not in initialized
        listed = server.handle_message(
            {"jsonrpc": "2.0", "id": "tools", "method": "tools/list", "params": {}}
        )
        assert listed is not None
        descriptor = next(
            item for item in listed["result"]["tools"] if item["name"] == "sync_finding"
        )
        assert descriptor["inputSchema"]["required"] == [
            "finding_id",
            "repository",
            "expected_finding_revision",
        ]
        assert descriptor["inputSchema"]["properties"]["repository"] == {
            "type": "string",
            "minLength": 1,
        }
        assert descriptor["inputSchema"]["properties"]["expected_finding_revision"] == {
            "type": "integer",
            "minimum": 0,
        }
        called = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "sync",
                "method": "tools/call",
                "params": {
                    "name": "sync_finding",
                    "arguments": {
                        "finding_id": finding.finding_id,
                        "repository": REPOSITORY,
                        "expected_finding_revision": revision,
                    },
                },
            }
        )
        assert called is not None
        assert called["result"]["structuredContent"]["status"] == "committed"
        assert len(provider.create_calls) == 1
    finally:
        service.close()


def test_duplicate_remote_markers_remain_conflict_and_do_not_create(tmp_path: Path) -> None:
    finding = add_candidate(new_finding("service-finding", "turning symptom"), "episode-1")
    rendered = render_finding_issue(finding, repository=REPOSITORY)
    issue = GitHubIssue(
        repository=REPOSITORY,
        number=1,
        url=f"https://github.com/{REPOSITORY}/issues/1",
        title=rendered.title,
        body=rendered.body,
    )
    duplicate = replace(issue, number=2, url=f"https://github.com/{REPOSITORY}/issues/2")
    provider = ServiceFakeProvider(issues=[issue, duplicate])
    service, session, _finding, revision, provider = _setup(tmp_path, provider)
    try:
        result = service.sync_finding(
            session,
            finding_id=finding.finding_id,
            repository=REPOSITORY,
            expected_finding_revision=revision,
            operation_id="duplicate-marker",
        )
        assert result.status == "failed"
        assert result.value is None
        assert provider.create_calls == []
    finally:
        service.close()
