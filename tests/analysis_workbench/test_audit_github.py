"""Adversarial offline tests for finding-level GitHub synchronization."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Event
from typing import TYPE_CHECKING, Any

import pytest

from robot_sf.analysis_workbench.audit_contracts import Finding
from robot_sf.analysis_workbench.audit_findings import (
    FindingStore,
    add_candidate,
    confirm_member,
    new_finding,
)
from robot_sf.analysis_workbench.audit_github import (
    AUDITOR_BLOCK_END,
    AUDITOR_BLOCK_START,
    FindingEvidence,
    GitHubCapabilityUnavailable,
    GitHubConflictError,
    GitHubFindingClaim,
    GitHubIssue,
    GitHubOutbox,
    GitHubOutboxEntry,
    GitHubOutboxError,
    GitHubPrivacyError,
    GitHubSync,
    GitHubSyncError,
    GitHubValidationError,
    PublicDataFilter,
    SearchResult,
    finding_marker,
    parse_finding_marker,
    render_finding_issue,
    replace_auditor_block,
    sync_finding,
)
from robot_sf.analysis_workbench.audit_store import AuditStore

REPOSITORY = "ll7/robot_sf_ll7"


if TYPE_CHECKING:
    from pathlib import Path


class FakeProvider:
    """Provider fake that records every attempted mutation."""

    def __init__(self, *, issues: list[GitHubIssue] | None = None) -> None:
        """Initialize a fake provider with optional remote issues."""

        self.issues = list(issues or [])
        self.create_calls: list[dict[str, object]] = []
        self.update_calls: list[dict[str, object]] = []
        self.search_calls = 0
        self.create_timeout_after_success = False
        self.search_complete = True
        self.next_number = 100

    def search_issues(self, repository: str, *, marker: str) -> SearchResult:
        self.search_calls += 1
        return SearchResult(tuple(self.issues), complete=self.search_complete)

    def create_issue(self, repository: str, *, title: str, body: str, labels: tuple[str, ...]):
        payload = {"repository": repository, "title": title, "body": body, "labels": labels}
        self.create_calls.append(payload)
        issue = GitHubIssue(
            repository=repository,
            number=self.next_number,
            url=f"https://github.com/{repository}/issues/{self.next_number}",
            title=title,
            body=body,
            labels=tuple(labels),
        )
        self.next_number += 1
        if self.create_timeout_after_success:
            self.issues.append(issue)
            raise TimeoutError("response lost after GitHub accepted create")
        self.issues.append(issue)
        return issue

    def get_issue(self, repository: str, number: int) -> GitHubIssue:
        for issue in self.issues:
            if issue.repository == repository and issue.number == number:
                return issue
        raise LookupError(number)

    def update_issue(self, repository: str, number: int, *, body: str) -> GitHubIssue:
        self.update_calls.append({"repository": repository, "number": number, "body": body})
        current = self.get_issue(repository, number)
        updated = replace(current, body=body)
        self.issues = [updated if item.number == number else item for item in self.issues]
        return updated

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
            raise GitHubConflictError("fake provider body CAS failed")
        if current.updated_at != expected_updated_at:
            raise GitHubConflictError("fake provider issue version CAS failed")
        return self.update_issue(repository, number, body=body)

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
        del finding_id, expected_finding_revision
        return self.create_issue(repository, title=title, body=body, labels=labels)

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
        del finding_id, expected_finding_revision
        return self.update_issue_if_unchanged(
            repository,
            number,
            body=body,
            expected_body_digest=expected_body_digest,
            expected_updated_at=expected_updated_at,
        )


def _finding(finding_id: str = "finding-1", *, title: str = "turning symptom") -> Finding:
    finding = new_finding(finding_id, title, source_revision="commit-1")
    finding = add_candidate(finding, "episode-candidate")
    return confirm_member(finding, "episode-confirmed")


def _issue_from_rendered(rendered, *, body_prefix: str = "") -> GitHubIssue:
    return GitHubIssue(
        repository=REPOSITORY,
        number=7,
        url=f"https://github.com/{REPOSITORY}/issues/7",
        title=rendered.title,
        body=f"{rendered.marker}\n\n{body_prefix}{rendered.auditor_block}\n\nHuman follow-up",
        labels=("human-label", "benchmark-audit"),
    )


def _entry_for_validation() -> GitHubOutboxEntry:
    rendered = render_finding_issue(_finding("validation-entry"), repository=REPOSITORY)
    return GitHubOutboxEntry(
        repository=REPOSITORY,
        finding_id=rendered.finding_id,
        operation_id="validation-entry-op",
        request_digest=rendered.request_digest,
        title=rendered.title,
        body=rendered.body,
        marker=rendered.marker,
        labels=rendered.labels,
        auditor_block_digest=rendered.auditor_block_digest,
    )


def _claim_for_validation() -> GitHubFindingClaim:
    return GitHubFindingClaim(
        repository=REPOSITORY,
        finding_id="validation-claim",
        operation_id="validation-claim-op",
        request_digest="a" * 64,
        marker=finding_marker(REPOSITORY, "validation-claim"),
    )


def test_marker_and_render_keep_candidate_and_confirmed_counts_distinct() -> None:
    finding = _finding()
    finding = replace(
        finding,
        evidence=({"source_note": "source-bound evidence"},),
        negative_evidence=({"control_note": "negative evidence"},),
        diagnostic_results=({"diagnostic": "supported"},),
    )
    rendered = render_finding_issue(
        finding,
        repository=REPOSITORY,
        evidence={
            "campaign_id": "camera-ready-campaign",
            "source_digest": "source-digest",
            "reproduction_commands": ["python evaluate.py --campaign camera-ready-campaign"],
        },
    )

    assert parse_finding_marker(rendered.body) == (REPOSITORY, finding.finding_id)
    assert "Candidate episodes (1)" in rendered.body
    assert "Confirmed episodes (1)" in rendered.body
    assert "Negative controls (0)" in rendered.body
    assert "source-bound evidence" in rendered.body
    assert "negative evidence" in rendered.body
    assert "supported" in rendered.body
    assert rendered.body.count(AUDITOR_BLOCK_START) == 1
    assert rendered.body.count(AUDITOR_BLOCK_END) == 1


@pytest.mark.parametrize(
    "value",
    [
        "<!-- robot_sf_audit_finding:v1 repository=wrong -->",
        "<!-- robot_sf_audit_finding:v1 repository=ll7/robot_sf_ll7 finding_id=x -->\n"
        "<!-- robot_sf_audit_finding:v1 repository=ll7/robot_sf_ll7 finding_id=x -->",
    ],
)
def test_malformed_or_duplicate_marker_fails_closed(value: str) -> None:
    with pytest.raises(GitHubValidationError):
        parse_finding_marker(value)


def test_success_then_timeout_reconciles_without_duplicate_create(tmp_path: Path) -> None:
    provider = FakeProvider()
    provider.create_timeout_after_success = True
    finding = _finding()
    with GitHubOutbox(tmp_path) as outbox:
        first = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="sync-timeout",
            evidence={"campaign_id": "campaign", "source_digest": "digest"},
        )
        assert first.status == "reconciled"
        assert first.ok
        assert len(provider.create_calls) == 1
        second = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="sync-timeout",
            evidence={"campaign_id": "campaign", "source_digest": "digest"},
        )
        assert second.replayed
        assert len(provider.create_calls) == 1
        assert outbox.get(REPOSITORY, finding.finding_id, "sync-timeout").state == "succeeded"


def test_unsupported_issue_create_returns_unavailable_without_remote_write(
    tmp_path: Path,
) -> None:
    class UnsupportedCreateProvider(FakeProvider):
        def create_issue(
            self,
            repository: str,
            *,
            title: str,
            body: str,
            labels: tuple[str, ...],
        ) -> GitHubIssue:
            del repository, title, body, labels
            raise GitHubCapabilityUnavailable("direct issue creation is unsupported")

    provider = UnsupportedCreateProvider()
    finding = _finding("unavailable-direct-create")
    with GitHubOutbox(tmp_path) as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="unavailable-direct-create",
        )
        entry = outbox.get(REPOSITORY, finding.finding_id, "unavailable-direct-create")
        claim = outbox.get_claim(REPOSITORY, finding.finding_id)

    assert result.status == "unavailable"
    assert result.remote_write == "none"
    assert result.outbox is not None and result.outbox.state == "failed"
    assert entry is not None and entry.state == "failed"
    assert claim is not None and claim.state == "failed"
    assert provider.create_calls == []


def test_replay_repairs_claim_after_crash_between_outbox_and_claim_success(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    finding = _finding("finding-claim-replay")
    with GitHubOutbox(tmp_path) as outbox:
        sync = GitHubSync(provider, outbox)
        original_complete = outbox.complete_success

        def crash_after_outbox_success(entry: GitHubOutboxEntry, claim: GitHubFindingClaim):
            current = outbox._get_with_revision(*entry.key)
            assert current is not None
            outbox.update(
                replace(entry, state="succeeded", issue=claim.issue),
                expected_revision=current[1],
            )
            raise RuntimeError("simulated crash after outbox success")

        outbox.complete_success = crash_after_outbox_success  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="simulated crash"):
            sync.sync(REPOSITORY, finding, operation_id="claim-replay")
        entry = outbox.get(REPOSITORY, finding.finding_id, "claim-replay")
        claim = outbox.get_claim(REPOSITORY, finding.finding_id)
        assert entry is not None and entry.state == "succeeded"
        assert claim is not None and claim.state == "in_flight"

        outbox.complete_success = original_complete  # type: ignore[method-assign]
        replay = sync.sync(REPOSITORY, finding, operation_id="claim-replay")
        claim = outbox.get_claim(REPOSITORY, finding.finding_id)

    assert replay.replayed
    assert claim is not None and claim.state == "succeeded"


def test_concurrent_workers_do_not_create_two_marker_issues(tmp_path: Path) -> None:
    provider = FakeProvider()
    create_started = Event()
    release_create = Event()
    original_create = provider.create_issue

    def blocking_create(*args, **kwargs):
        create_started.set()
        assert release_create.wait(timeout=30)
        return original_create(*args, **kwargs)

    provider.create_issue = blocking_create  # type: ignore[method-assign]
    finding = _finding("finding-concurrent")
    results: list[object] = []
    with GitHubOutbox(tmp_path) as outbox:
        first_sync = GitHubSync(provider, outbox)
        second_sync = GitHubSync(provider, outbox)

        def first_worker() -> None:
            results.append(
                first_sync.sync(
                    REPOSITORY,
                    finding,
                    operation_id="sync-concurrent",
                    evidence={"campaign_id": "campaign", "source_digest": "digest"},
                )
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(first_worker)
            try:
                assert create_started.wait(timeout=30)
                second = second_sync.sync(
                    REPOSITORY,
                    finding,
                    operation_id="sync-concurrent",
                    evidence={"campaign_id": "campaign", "source_digest": "digest"},
                )
            finally:
                release_create.set()
            future.result()

    assert second.status == "ambiguous"
    assert len(provider.create_calls) == 1
    assert results and results[0].status == "created"


def test_different_operations_share_one_finding_wide_claim(tmp_path: Path) -> None:
    provider = FakeProvider()
    search_barrier = Barrier(2)
    original_search = provider.search_issues

    def synchronized_search(repository: str, *, marker: str) -> SearchResult:
        result = original_search(repository, marker=marker)
        search_barrier.wait(timeout=30)
        return result

    provider.search_issues = synchronized_search  # type: ignore[method-assign]
    finding = _finding("finding-wide-claim")
    results: dict[str, object] = {}

    def run(operation_id: str) -> None:
        with GitHubOutbox(tmp_path) as outbox:
            sync = GitHubSync(provider, outbox)
            results[operation_id] = sync.sync(
                REPOSITORY,
                finding,
                operation_id=operation_id,
                evidence={"campaign_id": "campaign", "source_digest": "digest"},
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run, "wide-claim-0"),
            executor.submit(run, "wide-claim-1"),
        ]
        for future in futures:
            future.result()
    with GitHubOutbox(tmp_path) as outbox:
        claim = outbox.get_claim(REPOSITORY, finding.finding_id)

    assert len(provider.create_calls) == 1
    assert sorted(result.status for result in results.values()) == ["created", "pending"]
    assert claim is not None
    assert claim.state == "succeeded"


def test_stale_expected_finding_revision_fails_before_remote_mutation(tmp_path: Path) -> None:
    finding = _finding("finding-stale-revision")
    provider = FakeProvider()
    with AuditStore(tmp_path / "audit") as store:
        finding_store = FindingStore(store)
        finding_store.create(finding, operation_id="create-stale-finding")
        with GitHubOutbox(store) as outbox:
            with pytest.raises(GitHubConflictError, match="before remote mutation"):
                GitHubSync(provider, outbox, finding_store=finding_store).sync(
                    REPOSITORY,
                    finding,
                    operation_id="stale-revision-sync",
                    expected_finding_revision=2,
                    evidence={"campaign_id": "campaign", "source_digest": "digest"},
                )

    assert provider.search_calls == 0
    assert provider.create_calls == []


def test_incomplete_matching_search_never_updates_visible_issue(tmp_path: Path) -> None:
    old = _finding("finding-incomplete-search", title="old finding title")
    old_rendered = render_finding_issue(
        old,
        repository=REPOSITORY,
        evidence={"campaign_id": "campaign", "source_digest": "digest"},
    )
    provider = FakeProvider(issues=[_issue_from_rendered(old_rendered)])
    provider.search_complete = False
    with GitHubOutbox(tmp_path) as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            _finding("finding-incomplete-search", title="new finding title"),
            operation_id="incomplete-search-sync",
            evidence={"campaign_id": "campaign", "source_digest": "digest"},
        )

        entry = outbox.get(REPOSITORY, "finding-incomplete-search", "incomplete-search-sync")

    assert result.status == "ambiguous"
    assert provider.update_calls == []
    assert entry is not None and entry.state == "ambiguous"


def test_marker_shaped_finding_text_is_escaped_before_create(tmp_path: Path) -> None:
    marker_text = finding_marker(REPOSITORY, "other-finding")
    finding = _finding("finding-marker-text", title=f"user text {marker_text}")
    rendered = render_finding_issue(finding, repository=REPOSITORY)
    assert rendered.body.count("<!-- robot_sf_audit_finding:v1") == 1
    assert "&lt;!-- robot_sf_audit_finding:v1" in rendered.body

    provider = FakeProvider()
    with GitHubOutbox(tmp_path) as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="marker-text-sync",
        )

    assert result.status == "created"
    assert len(provider.create_calls) == 1


def test_update_success_then_timeout_reconciles_on_reopen(tmp_path: Path) -> None:
    old = _finding("finding-update-timeout", title="old finding title")
    old_rendered = render_finding_issue(
        old,
        repository=REPOSITORY,
        evidence={"campaign_id": "campaign", "source_digest": "digest"},
    )
    provider = FakeProvider(issues=[_issue_from_rendered(old_rendered)])
    original_update = provider.update_issue
    timeout_once = True

    def update_then_timeout(repository: str, number: int, *, body: str) -> GitHubIssue:
        nonlocal timeout_once
        updated = original_update(repository, number, body=body)
        if timeout_once:
            timeout_once = False
            raise TimeoutError("response lost after update applied")
        return updated

    provider.update_issue = update_then_timeout  # type: ignore[method-assign]
    new_finding = _finding("finding-update-timeout", title="new finding title")
    new_rendered = render_finding_issue(
        new_finding,
        repository=REPOSITORY,
        evidence={"campaign_id": "campaign", "source_digest": "digest"},
    )
    new_finding = replace(
        new_finding,
        github_issue={
            "schema_version": "github-link.v1",
            "repository": REPOSITORY,
            "number": 7,
            "marker": old_rendered.marker,
            "auditor_block_digest": old_rendered.auditor_block_digest,
        },
    )
    with GitHubOutbox(tmp_path) as outbox:
        first = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            new_finding,
            operation_id="update-timeout-sync",
            evidence={"campaign_id": "campaign", "source_digest": "digest"},
        )
        second = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            new_finding,
            operation_id="update-timeout-sync",
            retry_ambiguous=True,
            evidence={"campaign_id": "campaign", "source_digest": "digest"},
        )
        entry = outbox.get(REPOSITORY, new_finding.finding_id, "update-timeout-sync")

    assert first.status == "ambiguous"
    assert second.status == "reconciled"
    assert len(provider.update_calls) == 1
    assert entry is not None and entry.state == "succeeded"
    assert new_rendered.auditor_block in provider.issues[0].body


def test_ambiguous_update_conflicts_with_human_auditor_block_edit_without_link(
    tmp_path: Path,
) -> None:
    old = _finding("finding-ambiguous-human-block", title="old finding title")
    old_rendered = render_finding_issue(old, repository=REPOSITORY)
    provider = FakeProvider(issues=[_issue_from_rendered(old_rendered)])
    original_update = provider.update_issue
    timeout_once = True

    def update_then_timeout(repository: str, number: int, *, body: str) -> GitHubIssue:
        nonlocal timeout_once
        updated = original_update(repository, number, body=body)
        if timeout_once:
            timeout_once = False
            raise TimeoutError("response lost after update applied")
        return updated

    provider.update_issue = update_then_timeout  # type: ignore[method-assign]
    updated = _finding("finding-ambiguous-human-block", title="new finding title")
    human_block = old_rendered.auditor_block.replace(
        "## Benchmark audit finding", "## Human rewrite"
    )
    with GitHubOutbox(tmp_path) as outbox:
        first = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            updated,
            operation_id="ambiguous-human-block",
        )
        provider.issues[0] = replace(
            provider.issues[0],
            body=replace_auditor_block(provider.issues[0].body, human_block),
        )
        second = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            updated,
            operation_id="ambiguous-human-block",
            retry_ambiguous=True,
        )
        entry = outbox.get(REPOSITORY, updated.finding_id, "ambiguous-human-block")
        claim = outbox.get_claim(REPOSITORY, updated.finding_id)

    assert first.status == "ambiguous"
    assert second.status == "conflict"
    assert len(provider.update_calls) == 1
    assert "## Human rewrite" in provider.issues[0].body
    assert entry is not None and entry.state == "conflict"
    assert claim is not None and claim.state == "conflict"


def test_ambiguous_create_reconciliation_conflicts_with_human_block_edit(
    tmp_path: Path,
) -> None:
    finding = _finding("finding-ambiguous-create-human-block")
    rendered = render_finding_issue(finding, repository=REPOSITORY)

    class HumanEditDuringReconciliation(FakeProvider):
        def search_issues(self, repository: str, *, marker: str) -> SearchResult:
            result = super().search_issues(repository, marker=marker)
            if self.search_calls > 1 and self.issues:
                human_block = rendered.auditor_block.replace(
                    "## Benchmark audit finding", "## Human rewrite"
                )
                self.issues[0] = replace(
                    self.issues[0],
                    body=replace_auditor_block(self.issues[0].body, human_block),
                )
                return SearchResult(tuple(self.issues), complete=result.complete)
            return result

    provider = HumanEditDuringReconciliation()
    provider.create_timeout_after_success = True
    with GitHubOutbox(tmp_path) as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="ambiguous-create-human-block",
        )
        entry = outbox.get(REPOSITORY, finding.finding_id, "ambiguous-create-human-block")
        claim = outbox.get_claim(REPOSITORY, finding.finding_id)

    assert result.status == "conflict"
    assert len(provider.create_calls) == 1
    assert provider.update_calls == []
    assert "## Human rewrite" in provider.issues[0].body
    assert entry is not None and entry.state == "conflict"
    assert claim is not None and claim.state == "conflict"


def test_ambiguous_create_retry_conflicts_with_late_human_block_edit_without_digest(
    tmp_path: Path,
) -> None:
    finding = _finding("finding-ambiguous-create-late-human-block")
    rendered = render_finding_issue(finding, repository=REPOSITORY)
    provider = FakeProvider()

    def lost_before_apply(*args: Any, **kwargs: Any) -> GitHubIssue:
        raise TimeoutError("connection failed before response")

    provider.create_issue = lost_before_apply  # type: ignore[method-assign]
    with GitHubOutbox(tmp_path) as outbox:
        first = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="ambiguous-create-late-human-block",
        )
        human_block = rendered.auditor_block.replace(
            "## Benchmark audit finding", "## Human rewrite"
        )
        provider.issues.append(
            replace(
                _issue_from_rendered(rendered),
                body=replace_auditor_block(rendered.body, human_block),
            )
        )
        second = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="ambiguous-create-late-human-block",
            retry_ambiguous=True,
        )
        entry = outbox.get(REPOSITORY, finding.finding_id, "ambiguous-create-late-human-block")
        claim = outbox.get_claim(REPOSITORY, finding.finding_id)

    assert first.status == "ambiguous"
    assert second.status == "conflict"
    assert provider.create_calls == []
    assert provider.update_calls == []
    assert "## Human rewrite" in provider.issues[0].body
    assert entry is not None and entry.state == "conflict"
    assert claim is not None and claim.state == "conflict"


def test_generic_credential_shaped_observations_are_redacted() -> None:
    finding = replace(
        _finding("finding-generic-credentials"),
        observations=(
            "token=plainsecret",
            "credential: plainsecret",
            "oauth_token=plainsecret",
        ),
        evidence=(
            {
                "token": "nestedsecret",
                "nested": {"oauth_token": "nested-oauth-secret"},
                "api_key": "nested-api-secret",
                "accessToken": "camel-access-secret",
                "apiKey": "camel-api-secret",
                "APIKey": "acronym-api-secret",
                "oauthToken": "camel-oauth-secret",
            },
        ),
        negative_evidence=({"credential": "negative-secret"},),
        diagnostic_results=({"authorization": "diagnostic-secret"},),
    )
    rendered = render_finding_issue(finding, repository=REPOSITORY)
    assert "plainsecret" not in rendered.body
    assert "nestedsecret" not in rendered.body
    assert "nested-oauth-secret" not in rendered.body
    assert "nested-api-secret" not in rendered.body
    assert "camel-access-secret" not in rendered.body
    assert "camel-api-secret" not in rendered.body
    assert "acronym-api-secret" not in rendered.body
    assert "camel-oauth-secret" not in rendered.body
    assert "negative-secret" not in rendered.body
    assert "diagnostic-secret" not in rendered.body
    assert rendered.body.count("[REDACTED_SECRET]") >= 3


def test_ambiguous_create_survives_reopen_without_blind_retry(tmp_path: Path) -> None:
    provider = FakeProvider()
    provider.create_timeout_after_success = False
    original_create = provider.create_issue

    def lost_before_apply(*args, **kwargs):
        raise TimeoutError("connection failed before response")

    provider.create_issue = lost_before_apply  # type: ignore[method-assign]
    finding = _finding("finding-reopen")
    with GitHubOutbox(tmp_path) as outbox:
        first = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="sync-reopen",
            evidence={"campaign_id": "campaign", "source_digest": "digest"},
        )
        assert first.status == "ambiguous"
        assert len(provider.create_calls) == 0

    provider.create_issue = original_create  # type: ignore[method-assign]
    with GitHubOutbox(tmp_path) as reopened:
        second = GitHubSync(provider, reopened).sync(
            REPOSITORY,
            finding,
            operation_id="sync-reopen",
            evidence={"campaign_id": "campaign", "source_digest": "digest"},
        )
        assert second.status == "ambiguous"
        assert len(provider.create_calls) == 0
        retried = GitHubSync(provider, reopened).sync(
            REPOSITORY,
            finding,
            operation_id="sync-reopen",
            evidence={"campaign_id": "campaign", "source_digest": "digest"},
            retry_ambiguous=True,
        )
        assert retried.status == "created"
        assert len(provider.create_calls) == 1


def test_ambiguous_create_requires_complete_reconciliation_search(tmp_path: Path) -> None:
    class IncompleteReconciliationProvider(FakeProvider):
        def search_issues(self, repository: str, *, marker: str) -> SearchResult:
            result = super().search_issues(repository, marker=marker)
            if self.search_calls > 1:
                return SearchResult(result.issues, complete=False, reason="later page unavailable")
            return result

    provider = IncompleteReconciliationProvider()
    provider.create_timeout_after_success = True
    finding = _finding("finding-incomplete-reconciliation")
    with GitHubOutbox(tmp_path) as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="incomplete-reconciliation",
        )
        entry = outbox.get(REPOSITORY, finding.finding_id, "incomplete-reconciliation")
        claim = outbox.get_claim(REPOSITORY, finding.finding_id)

    assert result.status == "ambiguous"
    assert "later page unavailable" in result.reason
    assert entry is not None and entry.state == "ambiguous"
    assert claim is not None and claim.state == "ambiguous"


def test_deleted_claimed_issue_terminalizes_claim_and_recovers(tmp_path: Path) -> None:
    provider = FakeProvider()
    finding = _finding("finding-deleted-claim")
    with GitHubOutbox(tmp_path) as outbox:
        first = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="deleted-claim-first",
        )
        assert first.status == "created"
        provider.issues.clear()
        second = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="deleted-claim-recovery",
        )
        claim = outbox.get_claim(REPOSITORY, finding.finding_id)

    assert second.status == "created"
    assert len(provider.create_calls) == 2
    assert claim is not None
    assert claim.state == "succeeded"
    assert claim.operation_id == "deleted-claim-recovery"


def test_older_ambiguous_operation_cannot_rollback_newer_success(tmp_path: Path) -> None:
    provider = FakeProvider()
    original_create = provider.create_issue
    fail_once = True

    def fail_first_create(*args: Any, **kwargs: Any) -> GitHubIssue:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise TimeoutError("response lost before first create was applied")
        return original_create(*args, **kwargs)

    provider.create_issue = fail_first_create  # type: ignore[method-assign]
    finding = _finding("finding-newer-success")
    newer = _finding("finding-newer-success", title="newer finding title")
    with GitHubOutbox(tmp_path) as outbox:
        first = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="older-ambiguous",
        )
        second = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            newer,
            operation_id="newer-success",
            retry_ambiguous=True,
        )
        replay = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="older-ambiguous",
            retry_ambiguous=True,
        )
        claim = outbox.get_claim(REPOSITORY, finding.finding_id)

    assert first.status == "ambiguous"
    assert second.status == "created"
    assert replay.status == "conflict"
    assert len(provider.create_calls) == 1
    assert provider.issues[0].body == second.issue.body
    assert "newer finding title" in provider.issues[0].body
    assert claim is not None and claim.state == "succeeded"
    assert claim.operation_id == "newer-success"


def test_update_replaces_only_auditor_block_and_preserves_human_text_and_labels(
    tmp_path: Path,
) -> None:
    old = _finding(title="old finding title")
    old_rendered = render_finding_issue(
        old,
        repository=REPOSITORY,
        evidence={"campaign_id": "campaign", "source_digest": "digest"},
    )
    provider = FakeProvider(
        issues=[_issue_from_rendered(old_rendered, body_prefix="Human preface\n\n")]
    )
    updated = _finding(title="new finding title")
    with GitHubOutbox(tmp_path) as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            updated,
            operation_id="sync-human-text",
            evidence={"campaign_id": "campaign", "source_digest": "digest"},
        )

    assert result.status == "updated"
    assert len(provider.update_calls) == 1
    call = provider.update_calls[0]
    assert set(call) == {"repository", "number", "body"}
    remote = provider.issues[0]
    assert "Human preface" in remote.body
    assert "Human follow-up" in remote.body
    assert remote.labels == ("human-label", "benchmark-audit")
    assert "new finding title" in remote.body


def test_human_edit_after_search_is_a_conflict_without_provider_cas_snapshot(
    tmp_path: Path,
) -> None:
    original = _finding(title="old finding title")
    original_rendered = render_finding_issue(
        original,
        repository=REPOSITORY,
        evidence={"campaign_id": "campaign", "source_digest": "digest"},
    )

    class HumanEditAfterSearch(FakeProvider):
        def search_issues(self, repository: str, *, marker: str) -> SearchResult:
            result = super().search_issues(repository, marker=marker)
            current = self.issues[0]
            self.issues[0] = replace(current, body=current.body + "\n\nHuman edit after search")
            return result

    provider = HumanEditAfterSearch(issues=[_issue_from_rendered(original_rendered)])
    with GitHubOutbox(tmp_path) as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            _finding(title="new finding title"),
            operation_id="human-edit-after-search",
            evidence={"campaign_id": "campaign", "source_digest": "digest"},
        )

    assert result.status == "conflict"
    assert provider.update_calls == []
    assert "Human edit after search" in provider.issues[0].body


def test_existing_update_without_provider_cas_fails_closed(tmp_path: Path) -> None:
    original = _finding(title="old finding title")
    rendered = render_finding_issue(
        original,
        repository=REPOSITORY,
        evidence={"campaign_id": "campaign", "source_digest": "digest"},
    )

    class LegacyProvider:
        def __init__(self) -> None:
            self.delegate = FakeProvider(issues=[_issue_from_rendered(rendered)])

        def search_issues(self, repository: str, *, marker: str) -> SearchResult:
            return self.delegate.search_issues(repository, marker=marker)

        def get_issue(self, repository: str, number: int) -> GitHubIssue:
            return self.delegate.get_issue(repository, number)

        def update_issue(self, repository: str, number: int, *, body: str) -> GitHubIssue:
            return self.delegate.update_issue(repository, number, body=body)

    provider = LegacyProvider()
    with GitHubOutbox(tmp_path) as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            _finding(title="new finding title"),
            operation_id="missing-provider-cas",
            evidence={"campaign_id": "campaign", "source_digest": "digest"},
        )

    assert result.status == "conflict"
    assert "CAS/ETag" in result.reason
    assert provider.delegate.update_calls == []


def test_immediate_reread_transport_failure_is_durable_ambiguous(tmp_path: Path) -> None:
    original = _finding(title="old finding title")
    rendered = render_finding_issue(original, repository=REPOSITORY)

    class UnavailableReread(FakeProvider):
        def get_issue(self, repository: str, number: int) -> GitHubIssue:
            raise ConnectionError("read unavailable")

    provider = UnavailableReread(issues=[_issue_from_rendered(rendered)])
    with GitHubOutbox(tmp_path) as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            _finding(title="new finding title"),
            operation_id="reread-transport-failure",
        )
        entry = outbox.get(REPOSITORY, original.finding_id, "reread-transport-failure")
        claim = outbox.get_claim(REPOSITORY, original.finding_id)

    assert result.status == "ambiguous"
    assert "read unavailable" in result.reason
    assert entry is not None and entry.state == "ambiguous"
    assert claim is not None and claim.state == "ambiguous"


def test_immediate_reread_wrong_marker_is_durable_conflict(tmp_path: Path) -> None:
    original = _finding("finding-reread-marker", title="old finding title")
    rendered = render_finding_issue(original, repository=REPOSITORY)
    other = render_finding_issue(_finding("other-reread-marker"), repository=REPOSITORY)

    class WrongMarkerReread(FakeProvider):
        def get_issue(self, repository: str, number: int) -> GitHubIssue:
            current = super().get_issue(repository, number)
            return replace(current, body=other.body)

    provider = WrongMarkerReread(issues=[_issue_from_rendered(rendered)])
    with GitHubOutbox(tmp_path) as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            _finding("finding-reread-marker", title="new finding title"),
            operation_id="reread-marker-conflict",
        )
        entry = outbox.get(REPOSITORY, original.finding_id, "reread-marker-conflict")
        claim = outbox.get_claim(REPOSITORY, original.finding_id)

    assert result.status == "conflict"
    assert entry is not None and entry.state == "conflict"
    assert claim is not None and claim.state == "conflict"


@pytest.mark.parametrize("snapshot_kind", ["mismatched", "malformed"])
def test_claim_refresh_invalid_snapshot_is_durable_conflict(
    tmp_path: Path,
    snapshot_kind: str,
) -> None:
    finding = _finding(f"finding-claim-refresh-{snapshot_kind}")
    rendered = render_finding_issue(finding, repository=REPOSITORY)
    claimed_issue = _issue_from_rendered(rendered)
    other = render_finding_issue(
        _finding(f"other-claim-refresh-{snapshot_kind}"),
        repository=REPOSITORY,
    )

    class InvalidClaimRefresh(FakeProvider):
        def search_issues(self, repository: str, *, marker: str) -> SearchResult:
            return SearchResult((), complete=True)

        def get_issue(self, repository: str, number: int) -> GitHubIssue | dict[str, object]:
            if snapshot_kind == "mismatched":
                return replace(claimed_issue, body=other.body)
            return {"repository": repository}

    provider = InvalidClaimRefresh()
    operation_id = f"claim-refresh-{snapshot_kind}"
    claim = GitHubFindingClaim(
        repository=REPOSITORY,
        finding_id=finding.finding_id,
        operation_id=operation_id,
        request_digest=rendered.request_digest,
        marker=rendered.marker,
        issue=claimed_issue.to_dict(),
    )
    with GitHubOutbox(tmp_path) as outbox:
        outbox.enqueue_claim(claim)
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id=operation_id,
        )
        entry = outbox.get(REPOSITORY, finding.finding_id, operation_id)
        stored_claim = outbox.get_claim(REPOSITORY, finding.finding_id)

    assert result.status == "conflict"
    assert entry is not None and entry.state == "conflict"
    assert stored_claim is not None and stored_claim.state == "conflict"
    assert provider.create_calls == []
    assert provider.update_calls == []


def test_update_rejects_provider_changes_to_unrelated_issue_fields(tmp_path: Path) -> None:
    original = _finding(title="old finding title")
    original_rendered = render_finding_issue(
        original,
        repository=REPOSITORY,
        evidence={"campaign_id": "campaign", "source_digest": "digest"},
    )

    class MutatingProvider(FakeProvider):
        def update_issue(self, repository: str, number: int, *, body: str) -> GitHubIssue:
            current = self.get_issue(repository, number)
            return replace(current, body=body, title="provider-mutated-title")

    provider = MutatingProvider(issues=[_issue_from_rendered(original_rendered)])
    with GitHubOutbox(tmp_path) as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            _finding(title="new finding title"),
            operation_id="sync-unrelated-fields",
            evidence={"campaign_id": "campaign", "source_digest": "digest"},
        )

    assert result.status == "ambiguous"
    assert provider.issues[0].title == original_rendered.title


def test_remote_auditor_block_conflict_is_retained(tmp_path: Path) -> None:
    original = _finding()
    original_rendered = render_finding_issue(
        original,
        repository=REPOSITORY,
        evidence={"campaign_id": "campaign", "source_digest": "digest"},
    )
    issue = _issue_from_rendered(original_rendered).to_dict()
    issue["body"] = issue["body"].replace("## Benchmark audit finding", "## Human rewrite")
    provider = FakeProvider(issues=[GitHubIssue.from_mapping(issue)])
    finding = replace(
        _finding(title="new title"),
        github_issue={
            "schema_version": "github-link.v1",
            "repository": REPOSITORY,
            "number": 7,
            "marker": original_rendered.marker,
            "auditor_block_digest": original_rendered.auditor_block_digest,
        },
    )
    with GitHubOutbox(tmp_path) as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="sync-conflict",
            evidence={"campaign_id": "campaign", "source_digest": "digest"},
        )

    assert result.status == "conflict"
    assert provider.update_calls == []
    assert "Human rewrite" in provider.issues[0].body


def test_conflicting_search_markers_are_not_ignored(tmp_path: Path) -> None:
    rendered = render_finding_issue(
        _finding(),
        repository=REPOSITORY,
        evidence={"campaign_id": "campaign", "source_digest": "digest"},
    )
    conflicting = replace(
        _issue_from_rendered(rendered),
        number=8,
        body=rendered.body.replace("finding-1", "other-finding"),
    )
    provider = FakeProvider(issues=[_issue_from_rendered(rendered), conflicting])
    with GitHubOutbox(tmp_path) as outbox:
        with pytest.raises(GitHubConflictError):
            GitHubSync(provider, outbox).sync(
                REPOSITORY,
                _finding(),
                operation_id="sync-conflicting-marker",
                evidence={"campaign_id": "campaign", "source_digest": "digest"},
            )


def test_private_paths_secrets_and_local_media_are_filtered(tmp_path: Path) -> None:
    finding = _finding()
    rendered = render_finding_issue(
        finding,
        repository=REPOSITORY,
        private_roots=[tmp_path],
        evidence={
            "campaign_id": "campaign",
            "source_digest": "digest",
            "metadata": {
                "private_path": str(tmp_path / "trace.json"),
                "token": "ghp_12345678901234567890",
            },
        },
    )
    assert str(tmp_path) not in rendered.body
    assert "ghp_12345678901234567890" not in rendered.body
    with pytest.raises(GitHubPrivacyError):
        render_finding_issue(
            finding,
            repository=REPOSITORY,
            evidence={"campaign_id": "campaign", "media_urls": ["file:///tmp/video.mp4"]},
        )


def test_malformed_provider_response_fails_closed(tmp_path: Path) -> None:
    class MalformedProvider(FakeProvider):
        def search_issues(self, repository: str, *, marker: str):
            return [{"repository": repository, "number": "not-an-int"}]

    with GitHubOutbox(tmp_path) as outbox:
        with pytest.raises(GitHubValidationError):
            GitHubSync(MalformedProvider(), outbox).sync(
                REPOSITORY,
                _finding(),
                operation_id="sync-malformed",
                evidence={"campaign_id": "campaign", "source_digest": "digest"},
            )


def test_outbox_identity_rejects_reuse_for_different_request(tmp_path: Path) -> None:
    entry = GitHubOutboxEntry(
        repository=REPOSITORY,
        finding_id="finding-1",
        operation_id="same-operation",
        request_digest="a" * 64,
        title="title",
        body="body",
        marker=finding_marker(REPOSITORY, "finding-1"),
        labels=("benchmark-audit",),
        auditor_block_digest="b" * 64,
    )
    changed = replace(entry, request_digest="c" * 64)
    with GitHubOutbox(tmp_path) as outbox:
        outbox.enqueue(entry)
        with pytest.raises(GitHubConflictError):
            outbox.enqueue(changed)


def test_optional_finding_store_binds_link_and_replays_caller_snapshot(tmp_path: Path) -> None:
    finding = _finding("finding-linked")
    provider = FakeProvider()
    with AuditStore(tmp_path / "audit") as store:
        finding_store = FindingStore(store)
        finding_store.create(finding, operation_id="create-finding")
        with GitHubOutbox(store) as outbox:
            sync = GitHubSync(provider, outbox, finding_store=finding_store)
            first = sync.sync(
                REPOSITORY,
                finding,
                operation_id="sync-linked",
                evidence={"campaign_id": "campaign", "source_digest": "digest"},
            )
            assert first.finding is not None
            assert first.finding.github_issue is not None
            replay = sync.sync(
                REPOSITORY,
                finding,
                operation_id="sync-linked",
                evidence={"campaign_id": "campaign", "source_digest": "digest"},
            )
            assert replay.replayed
            assert replay.finding is not None
            assert replay.finding.github_issue == first.finding.github_issue


def test_canonical_revision_write_without_reservation_capability_fails_closed(
    tmp_path: Path,
) -> None:
    class LegacyProvider:
        def __init__(self) -> None:
            self.delegate = FakeProvider()

        def search_issues(self, repository: str, *, marker: str) -> SearchResult:
            return self.delegate.search_issues(repository, marker=marker)

        def create_issue(self, repository: str, *, title: str, body: str, labels: tuple[str, ...]):
            return self.delegate.create_issue(repository, title=title, body=body, labels=labels)

        def get_issue(self, repository: str, number: int) -> GitHubIssue:
            return self.delegate.get_issue(repository, number)

        def update_issue(self, repository: str, number: int, *, body: str) -> GitHubIssue:
            return self.delegate.update_issue(repository, number, body=body)

    finding = _finding("finding-canonical-guard")
    provider = LegacyProvider()
    with AuditStore(tmp_path / "audit") as store:
        finding_store = FindingStore(store)
        finding_store.create(finding, operation_id="create-canonical-guard")
        with GitHubOutbox(store) as outbox:
            result = GitHubSync(provider, outbox, finding_store=finding_store).sync(
                REPOSITORY,
                finding,
                operation_id="canonical-guard-sync",
            )

    assert result.status == "conflict"
    assert "reservation" in result.reason
    assert provider.delegate.create_calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("number", True),
        ("number", 0),
        ("url", "not-a-url"),
        ("title", ""),
        ("body", None),
        ("state", "pending"),
        ("labels", (1,)),
        ("comments", (1,)),
        ("updated_at", 1),
    ],
)
def test_issue_contract_rejects_malformed_fields(field: str, value: Any) -> None:
    payload: dict[str, Any] = {
        "repository": REPOSITORY,
        "number": 1,
        "url": f"https://github.com/{REPOSITORY}/issues/1",
        "title": "title",
        "body": "body",
    }
    payload[field] = value
    with pytest.raises(GitHubValidationError):
        GitHubIssue.from_mapping(payload)


def test_issue_and_search_response_shapes_fail_closed() -> None:
    with pytest.raises(GitHubValidationError):
        GitHubIssue.from_mapping(None)  # type: ignore[arg-type]
    with pytest.raises(GitHubValidationError):
        GitHubIssue.from_mapping({"repository": REPOSITORY})
    with pytest.raises(GitHubValidationError):
        GitHubIssue.from_mapping(
            {
                "repository": REPOSITORY,
                "number": 1,
                "url": f"https://github.com/{REPOSITORY}/issues/1",
                "title": "title",
                "body": "body",
                "labels": "label",
            }
        )
    with pytest.raises(GitHubValidationError):
        GitHubIssue.from_mapping(
            {
                "repository": REPOSITORY,
                "number": 1,
                "url": f"https://github.com/{REPOSITORY}/issues/1",
                "title": "title",
                "body": "body",
                "comments": "comment",
            }
        )
    with pytest.raises(GitHubValidationError):
        SearchResult((object(),))  # type: ignore[arg-type]
    with pytest.raises(GitHubValidationError):
        SearchResult(complete=1)  # type: ignore[arg-type]
    with pytest.raises(GitHubValidationError):
        SearchResult(reason=1)  # type: ignore[arg-type]
    with pytest.raises(GitHubValidationError):
        SearchResult.from_value({"issues": "not-a-list"})
    with pytest.raises(GitHubValidationError):
        SearchResult.from_value("not-a-search")
    valid = _issue_from_rendered(render_finding_issue(_finding(), repository=REPOSITORY))
    assert SearchResult.from_value({"issues": [valid.to_dict()]}).issues == (valid,)
    assert SearchResult.from_value([valid.to_dict()]).issues == (valid,)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("campaign_id", 1),
        ("representative_cases", (1,)),
        ("diagnostics", (1,)),
        ("metadata", ("not-a-map",)),
    ],
)
def test_evidence_contract_rejects_malformed_fields(field: str, value: Any) -> None:
    payload: dict[str, Any] = {field: value}
    with pytest.raises(GitHubValidationError):
        FindingEvidence(**payload)  # type: ignore[arg-type]


def test_evidence_mapping_and_filter_boundaries_fail_closed(tmp_path: Path) -> None:
    assert FindingEvidence.from_value(None) == FindingEvidence()
    original = FindingEvidence(campaign_id="campaign")
    assert FindingEvidence.from_value(original) is original
    with pytest.raises(GitHubValidationError):
        FindingEvidence.from_value("not-a-map")  # type: ignore[arg-type]
    with pytest.raises(GitHubValidationError):
        FindingEvidence.from_value({"commands": 1})
    with pytest.raises(GitHubValidationError):
        FindingEvidence.from_value({"campaign_id": object()})
    with pytest.raises(GitHubValidationError):
        FindingEvidence.from_value({"diagnostics": "not-a-list"})

    data_filter = PublicDataFilter(private_roots=[tmp_path], allowed_media_hosts=["example.com"])
    assert data_filter.text(tmp_path / "trace.json") == "[REDACTED_PRIVATE_PATH]/trace.json"
    assert data_filter.text(123) == "123"
    assert data_filter.value({"token=plainsecret": "credential: plainsecret"}) == {
        "[REDACTED_SECRET]": "[REDACTED_SECRET]"
    }
    assert data_filter.value(1) == 1
    assert data_filter.value(None) is None
    with pytest.raises(GitHubPrivacyError):
        data_filter.value(object())
    with pytest.raises(GitHubPrivacyError):
        data_filter.value({"media_url": "data:video/mp4;base64,abc"})
    with pytest.raises(GitHubPrivacyError):
        data_filter.value({"media_url": "http://localhost/video.mp4"})
    with pytest.raises(GitHubPrivacyError):
        data_filter.value({"media_url": "http://127.0.0.1/video.mp4"})
    with pytest.raises(GitHubPrivacyError):
        data_filter.value({"media_url": "http://other.example/video.mp4"})
    with pytest.raises(GitHubPrivacyError):
        data_filter.value({"media_url": "ftp://example.com/video.mp4"})
    with pytest.raises(GitHubPrivacyError):
        data_filter.value({"media_url": "../video.mp4"})
    assert data_filter.value({"media_url": "https://example.com/video.mp4"}) == {
        "media_url": "https://example.com/video.mp4"
    }


def test_outbox_and_finding_claim_contract_boundaries() -> None:
    entry = _entry_for_validation()
    assert GitHubOutboxEntry.from_dict(entry.to_dict()) == entry
    with pytest.raises(GitHubOutboxError):
        GitHubOutboxEntry.from_dict(None)  # type: ignore[arg-type]
    with pytest.raises(GitHubOutboxError):
        GitHubOutboxEntry.from_dict({"labels": "not-a-list"})
    for field, value in (
        ("schema_version", "other"),
        ("state", "other"),
        ("attempts", -1),
        ("finding_revision", -1),
        ("title", 1),
        ("labels", (1,)),
        ("issue", "not-a-map"),
        ("request_digest", "bad"),
        ("auditor_block_digest", "bad"),
        ("marker", "wrong"),
    ):
        with pytest.raises(GitHubValidationError):
            replace(entry, **{field: value})  # type: ignore[call-overload]

    claim = _claim_for_validation()
    assert GitHubFindingClaim.from_dict(claim.to_dict()) == claim
    with pytest.raises(GitHubOutboxError):
        GitHubFindingClaim.from_dict(None)  # type: ignore[arg-type]
    for field, value in (
        ("schema_version", "other"),
        ("state", "other"),
        ("request_digest", "bad"),
        ("marker", "wrong"),
        ("finding_revision", -1),
        ("worker_id", 1),
        ("issue", "not-a-map"),
    ):
        with pytest.raises(GitHubValidationError):
            replace(claim, **{field: value})  # type: ignore[call-overload]


def test_search_failure_and_linked_read_failure_remain_durable(tmp_path: Path) -> None:
    class SearchFailureProvider(FakeProvider):
        def search_issues(self, repository: str, *, marker: str) -> SearchResult:
            raise TimeoutError("search unavailable")

    finding = _finding("finding-search-failure")
    with GitHubOutbox(tmp_path / "search") as outbox:
        result = GitHubSync(SearchFailureProvider(), outbox).sync(
            REPOSITORY,
            finding,
            operation_id="search-failure-sync",
        )
        entry = outbox.get(REPOSITORY, finding.finding_id, "search-failure-sync")
    assert result.status == "failed"
    assert entry is not None and entry.state == "failed"

    linked = render_finding_issue(_finding("finding-linked-read"), repository=REPOSITORY)
    linked_finding = replace(
        _finding("finding-linked-read"),
        github_issue={
            "schema_version": "github-link.v1",
            "repository": REPOSITORY,
            "number": 77,
            "marker": linked.marker,
        },
    )
    with GitHubOutbox(tmp_path / "linked") as outbox:
        result = GitHubSync(FakeProvider(), outbox).sync(
            REPOSITORY,
            linked_finding,
            operation_id="linked-read-failure-sync",
        )
    assert result.status == "failed"


def test_marker_and_link_contract_boundaries(tmp_path: Path) -> None:
    assert parse_finding_marker("plain text") is None
    with pytest.raises(GitHubValidationError):
        parse_finding_marker(None)  # type: ignore[arg-type]
    with pytest.raises(GitHubValidationError):
        parse_finding_marker("robot_sf_audit_finding:v1")
    with pytest.raises(GitHubValidationError):
        finding_marker("not-a-repository", "finding")
    with pytest.raises(GitHubValidationError):
        finding_marker(REPOSITORY, "not safe")

    finding = _finding("finding-link-boundaries")
    rendered = render_finding_issue(finding, repository=REPOSITORY)
    bad_links = (
        {
            "schema_version": "other",
            "repository": REPOSITORY,
            "number": 1,
            "marker": rendered.marker,
        },
        {
            "schema_version": "github-link.v1",
            "repository": "other/repo",
            "number": 1,
            "marker": rendered.marker,
        },
        {
            "schema_version": "github-link.v1",
            "repository": REPOSITORY,
            "number": 0,
            "marker": rendered.marker,
        },
        {
            "schema_version": "github-link.v1",
            "repository": REPOSITORY,
            "number": 1,
            "marker": "wrong",
        },
    )
    for index, link in enumerate(bad_links):
        with GitHubOutbox(tmp_path / f"link-boundary-{index}") as outbox:
            with pytest.raises(GitHubSyncError):
                GitHubSync(FakeProvider(), outbox).sync(
                    REPOSITORY,
                    replace(finding, github_issue=link),
                    operation_id=f"bad-link-{len(link)}-{link.get('number', 'x')}",
                )


def test_replace_auditor_block_rejects_malformed_delimiters() -> None:
    rendered = render_finding_issue(_finding("finding-block-boundary"), repository=REPOSITORY)
    with pytest.raises(GitHubValidationError):
        replace_auditor_block(None, rendered.auditor_block)  # type: ignore[arg-type]
    with pytest.raises(GitHubValidationError):
        replace_auditor_block(rendered.body, "not-a-block")
    with pytest.raises(GitHubConflictError):
        replace_auditor_block("human text", rendered.auditor_block)
    with pytest.raises(GitHubConflictError):
        replace_auditor_block(f"{AUDITOR_BLOCK_END}{AUDITOR_BLOCK_START}", rendered.auditor_block)
    with pytest.raises(GitHubConflictError):
        replace_auditor_block(
            f"{AUDITOR_BLOCK_START}one{AUDITOR_BLOCK_END}{AUDITOR_BLOCK_START}two{AUDITOR_BLOCK_END}",
            rendered.auditor_block,
        )
    assert replace_auditor_block(rendered.body, rendered.auditor_block) == rendered.body
    with pytest.raises(GitHubConflictError):
        replace_auditor_block(
            rendered.body,
            rendered.auditor_block,
            expected_block_digest="b" * 64,
        )


def test_sync_wrapper_and_public_labels_are_deterministic(tmp_path: Path) -> None:
    finding = replace(_finding("finding-wrapper"), tags=("benchmark-audit", "extra", "extra"))
    provider = FakeProvider()
    with GitHubOutbox(tmp_path) as outbox:
        result = sync_finding(
            provider,
            outbox,
            REPOSITORY,
            finding,
            operation_id="wrapper-sync",
        )
    assert result.ok
    assert provider.create_calls[0]["labels"] == ("benchmark-audit", "extra")
