"""Offline proof for the BA05 immutable-issue/append-only-comment slice."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from robot_sf.analysis_workbench.audit_contracts import Finding
from robot_sf.analysis_workbench.audit_findings import (
    add_candidate,
    confirm_member,
    new_finding,
)
from robot_sf.analysis_workbench.audit_github import (
    GITHUB_PUBLICATION_SCHEMA_VERSION,
    GitHubFindingClaim,
    GitHubIssue,
    GitHubOutbox,
    GitHubSync,
    GitHubValidationError,
    SearchResult,
    render_finding_issue,
)
from robot_sf.analysis_workbench.audit_github_rest import auditor_request_marker

if TYPE_CHECKING:
    from pathlib import Path


REPOSITORY = "ll7/robot_sf_ll7"


def _finding(finding_id: str = "append-only-finding") -> Finding:
    finding = new_finding(finding_id, "immutable issue", source_revision="commit-1")
    return confirm_member(add_candidate(finding, "candidate-1"), "confirmed-1")


class AppendOnlyProvider:
    """Small provider fake that rejects every issue-body update attempt."""

    def __init__(self) -> None:
        """Initialize an empty immutable-issue provider."""

        self.issues: list[GitHubIssue] = []
        self.create_calls: list[dict[str, Any]] = []
        self.comment_calls: list[dict[str, Any]] = []
        self.update_calls = 0
        self.timeout_after_comment = False
        self.search_error: Exception | None = None
        self.search_complete = True
        self.create_error: Exception | None = None
        self.create_error_after_success = False
        self.comment_error: Exception | None = None
        self.comment_error_after_success = False
        self.next_number = 10

    def search_issues(self, repository: str, *, marker: str) -> SearchResult:
        del marker
        if self.search_error is not None:
            raise self.search_error
        return SearchResult(
            tuple(self.issues),
            complete=self.search_complete,
            reason="test search incomplete" if not self.search_complete else "",
        )

    def create_issue(
        self,
        repository: str,
        *,
        title: str,
        body: str,
        labels: tuple[str, ...],
    ) -> GitHubIssue:
        issue = GitHubIssue(
            repository=repository,
            number=self.next_number,
            url=f"https://github.com/{repository}/issues/{self.next_number}",
            title=title,
            body=body,
            labels=tuple(labels),
        )
        self.next_number += 1
        self.create_calls.append({"title": title, "body": body})
        self.issues.append(issue)
        if self.create_error_after_success:
            raise self.create_error or TimeoutError("response lost after accepted create")
        if self.create_error is not None:
            raise self.create_error
        return issue

    def get_issue(self, repository: str, number: int) -> GitHubIssue:
        for issue in self.issues:
            if issue.repository == repository and issue.number == number:
                return issue
        raise LookupError(number)

    def append_auditor_comment(
        self,
        repository: str,
        number: int,
        *,
        body: str,
        request_digest: str,
    ) -> dict[str, Any]:
        issue = self.get_issue(repository, number)
        marker = auditor_request_marker(request_digest)
        comment_body = f"{marker}\n\n{body}"
        for comment in issue.comments:
            if marker in str(comment.get("body", "")):
                return {"status": "unchanged", "comment": comment}
        if self.comment_error is not None and not self.comment_error_after_success:
            raise self.comment_error
        comment = {"id": 100 + len(issue.comments), "body": comment_body}
        self.comment_calls.append({"number": number, "body": comment_body})
        updated = replace(issue, comments=(*issue.comments, comment))
        self.issues = [updated if item.number == number else item for item in self.issues]
        if self.comment_error_after_success:
            raise self.comment_error or TimeoutError("comment response lost after GitHub accepted POST")
        if self.timeout_after_comment:
            raise TimeoutError("comment response lost after GitHub accepted POST")
        return {"status": "created", "comment": comment}

    def update_issue(self, *_args: Any, **_kwargs: Any) -> None:
        self.update_calls += 1
        raise AssertionError("BA05 append-only path attempted an issue-body update")


def test_initial_issue_is_immutable_and_revision_is_a_comment(tmp_path: Path) -> None:
    provider = AppendOnlyProvider()
    original = _finding()
    with GitHubOutbox(tmp_path) as outbox:
        first = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            original,
            operation_id="initial-operation",
            evidence={"campaign_id": "campaign", "source_digest": "digest"},
        )
        issue_body = provider.issues[0].body
        revised = replace(original, observations=("new observation",))
        second = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            revised,
            operation_id="revision-operation",
            evidence={"campaign_id": "campaign", "source_digest": "digest"},
        )

        assert first.status == "created"
        assert second.status == "commented"
        assert provider.issues[0].body == issue_body
        assert len(provider.create_calls) == 1
        assert len(provider.comment_calls) == 1
        assert provider.update_calls == 0
        assert second.comment is not None
        assert second.outbox is not None
        assert second.outbox.publication_schema_version == GITHUB_PUBLICATION_SCHEMA_VERSION
        assert second.outbox.publication_kind == "revision_comment"
        assert second.finding is not None
        assert second.finding.github_issue is not None
        assert second.finding.github_issue["sync_mode"] == "append_only"
        assert second.finding.github_issue["comment_id"] == second.comment["id"]


def test_semantic_replay_deduplicates_comment_across_operation_ids(tmp_path: Path) -> None:
    provider = AppendOnlyProvider()
    finding = _finding("semantic-replay")
    with GitHubOutbox(tmp_path) as outbox:
        sync = GitHubSync(provider, outbox)
        sync.sync(REPOSITORY, finding, operation_id="create")
        revised = replace(finding, observations=("same revision payload",))
        first = sync.sync(REPOSITORY, revised, operation_id="comment-a")
        replay = sync.sync(REPOSITORY, revised, operation_id="comment-b")

        assert first.status == "commented"
        assert replay.status == "unchanged"
        assert replay.replayed
        assert len(provider.comment_calls) == 1


def test_timeout_after_comment_is_reconciled_without_duplicate(tmp_path: Path) -> None:
    provider = AppendOnlyProvider()
    finding = _finding("comment-timeout")
    with GitHubOutbox(tmp_path) as outbox:
        sync = GitHubSync(provider, outbox)
        sync.sync(REPOSITORY, finding, operation_id="create")
        revised = replace(finding, observations=("timeout revision",))
        provider.timeout_after_comment = True
        result = sync.sync(REPOSITORY, revised, operation_id="comment-timeout")
        replay = sync.sync(REPOSITORY, revised, operation_id="comment-timeout-replay")

    assert result.status == "reconciled"
    assert replay.status == "unchanged"
    assert len(provider.comment_calls) == 1
    assert provider.update_calls == 0


def test_initial_publication_body_carries_immutable_revision_identity() -> None:
    rendered = render_finding_issue(_finding("body-marker"), repository=REPOSITORY)
    assert "robot_sf_audit_finding:v1" in rendered.body


def test_append_only_search_failures_and_incomplete_reads_fail_closed(tmp_path: Path) -> None:
    finding = _finding("append-search-boundary")
    failed = AppendOnlyProvider()
    failed.search_error = ConnectionError("search unavailable")
    with GitHubOutbox(tmp_path / "failed") as outbox:
        failure = GitHubSync(failed, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="search-failed",
        )
    assert failure.status == "failed"
    assert "search unavailable" in failure.reason
    assert failed.create_calls == []

    incomplete = AppendOnlyProvider()
    incomplete.search_complete = False
    with GitHubOutbox(tmp_path / "incomplete") as outbox:
        blocked = GitHubSync(incomplete, outbox).sync(
            REPOSITORY,
            replace(finding, finding_id="append-search-incomplete"),
            operation_id="search-incomplete",
        )
    assert blocked.status == "ambiguous"
    assert "incomplete" in blocked.reason
    assert incomplete.create_calls == []


def test_append_only_create_timeout_reconciles_or_preserves_ambiguity(tmp_path: Path) -> None:
    finding = _finding("append-create-reconcile")
    accepted = AppendOnlyProvider()
    accepted.create_error_after_success = True
    with GitHubOutbox(tmp_path / "accepted") as outbox:
        reconciled = GitHubSync(accepted, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="create-timeout-reconciled",
        )
    assert reconciled.status == "reconciled"
    assert reconciled.remote_write == "none"
    assert len(accepted.create_calls) == 1

    class LostCreateProvider(AppendOnlyProvider):
        def create_issue(self, *args: Any, **kwargs: Any) -> GitHubIssue:
            del args
            self.create_calls.append(dict(kwargs))
            raise TimeoutError("create response lost before apply")

    lost = LostCreateProvider()
    with GitHubOutbox(tmp_path / "lost") as outbox:
        ambiguous = GitHubSync(lost, outbox).sync(
            REPOSITORY,
            replace(finding, finding_id="append-create-lost"),
            operation_id="create-timeout-lost",
        )
    assert ambiguous.status == "ambiguous"
    assert ambiguous.remote_write == "ambiguous"
    assert "create outcome is ambiguous" in ambiguous.reason


def test_append_only_preexisting_issue_is_migrated_without_body_rewrite(tmp_path: Path) -> None:
    finding = _finding("append-preexisting")
    seed = AppendOnlyProvider()
    with GitHubOutbox(tmp_path / "seed") as outbox:
        created = GitHubSync(seed, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="seed-issue",
        )
    provider = AppendOnlyProvider()
    provider.issues = list(seed.issues)
    original_body = provider.issues[0].body
    with GitHubOutbox(tmp_path / "migrate") as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="migrate-existing-issue",
        )
    assert created.status == "created"
    assert result.status == "commented"
    assert provider.issues[0].body == original_body
    assert len(provider.create_calls) == 0
    assert len(provider.comment_calls) == 1


def test_append_only_comment_failure_without_marker_is_ambiguous(tmp_path: Path) -> None:
    provider = AppendOnlyProvider()
    finding = _finding("append-comment-ambiguous")
    with GitHubOutbox(tmp_path) as outbox:
        first = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="comment-seed",
        )
        assert first.status == "created"
        provider.comment_error = TimeoutError("comment response lost before apply")
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            replace(finding, observations=("comment failed",)),
            operation_id="comment-ambiguous",
        )
    assert result.status == "ambiguous"
    assert result.remote_write == "ambiguous"
    assert "comment outcome is ambiguous" in result.reason


def test_append_only_publication_identity_validation_and_lookup(tmp_path: Path) -> None:
    finding = _finding("append-publication-validation")
    provider = AppendOnlyProvider()
    with GitHubOutbox(tmp_path) as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="publication-validation",
        )
        assert result.outbox is not None
        publication = result.outbox
        assert outbox.find_publication(
            REPOSITORY,
            finding.finding_id,
            publication.publication_key,
        ) is not None
        with pytest.raises(GitHubValidationError):
            outbox.find_publication(REPOSITORY, finding.finding_id, "bad")

    for field, value in (
        ("publication_schema_version", "unknown"),
        ("publication_kind", "unknown"),
        ("publication_revision", -1),
        ("comment", "not-a-map"),
        ("publication_digest", "bad"),
        ("publication_key", "bad"),
    ):
        with pytest.raises(GitHubValidationError):
            replace(publication, **{field: value})

    with pytest.raises(GitHubValidationError):
        replace(
            publication,
            publication_schema_version=GITHUB_PUBLICATION_SCHEMA_VERSION,
            publication_kind="legacy_body",
        )
    with pytest.raises(GitHubValidationError):
        replace(
            publication,
            publication_schema_version=GITHUB_PUBLICATION_SCHEMA_VERSION,
            publication_revision=None,
        )


def test_append_only_claim_repair_settles_crash_left_initial_claim(tmp_path: Path) -> None:
    finding = _finding("append-claim-repair")
    seed = AppendOnlyProvider()
    with GitHubOutbox(tmp_path / "seed") as outbox:
        seeded = GitHubSync(seed, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="claim-seed",
        )
    provider = AppendOnlyProvider()
    provider.issues = list(seed.issues)
    assert seeded.issue is not None
    rendered = render_finding_issue(finding, repository=REPOSITORY)
    claim = GitHubFindingClaim(
        repository=REPOSITORY,
        finding_id=finding.finding_id,
        operation_id=seeded.outbox.operation_id if seeded.outbox is not None else "missing",
        request_digest=rendered.request_digest,
        marker=rendered.marker,
        worker_id="worker-crashed",
        issue=seeded.issue.to_dict(),
    )
    with GitHubOutbox(tmp_path / "repair") as outbox:
        outbox.enqueue_claim(claim)
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="publication-crash-left",
        )
        repaired = outbox.get_claim(REPOSITORY, finding.finding_id)
    assert result.status == "commented"
    assert repaired is not None and repaired.state == "succeeded"
