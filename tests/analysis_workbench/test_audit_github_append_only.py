"""Offline proof for the BA05 immutable-issue/append-only-comment slice."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from robot_sf.analysis_workbench.audit_contracts import Finding
from robot_sf.analysis_workbench.audit_findings import (
    add_candidate,
    confirm_member,
    new_finding,
)
from robot_sf.analysis_workbench.audit_github import (
    GITHUB_PUBLICATION_SCHEMA_VERSION,
    GitHubIssue,
    GitHubOutbox,
    GitHubSync,
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
        self.next_number = 10

    def search_issues(self, repository: str, *, marker: str) -> SearchResult:
        del marker
        return SearchResult(tuple(self.issues), complete=True)

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
        comment = {"id": 100 + len(issue.comments), "body": comment_body}
        self.comment_calls.append({"number": number, "body": comment_body})
        updated = replace(issue, comments=(*issue.comments, comment))
        self.issues = [updated if item.number == number else item for item in self.issues]
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
