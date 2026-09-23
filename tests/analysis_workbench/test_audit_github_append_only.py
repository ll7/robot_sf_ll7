"""Offline proof for the BA05 immutable-issue/append-only-comment slice."""

from __future__ import annotations

from dataclasses import replace
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
    GITHUB_PUBLICATION_SCHEMA_VERSION,
    GitHubConflictError,
    GitHubFindingClaim,
    GitHubIssue,
    GitHubIssueMissing,
    GitHubOutbox,
    GitHubSync,
    GitHubTransportError,
    GitHubValidationError,
    SearchResult,
    _adopt_initial_publication_snapshot,
    _coerce_comment_write,
    _comment_payload_without_request_marker,
    _find_exact_publication_comment,
    _linked_issue_identity,
    _parse_publication_marker,
    _publication_request_marker,
    _select_marker_issue,
    _validate_initial_entry_identity,
    _validate_publication_entry_payload,
    render_finding_issue,
)
from robot_sf.analysis_workbench.audit_github_rest import auditor_request_marker
from robot_sf.analysis_workbench.audit_store import AuditStore

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
        self.get_issue_calls: list[tuple[str, int]] = []
        self.update_calls = 0
        self.timeout_after_comment = False
        self.search_error: Exception | None = None
        self.search_complete = True
        self.hide_issues_from_search = False
        self.create_error: Exception | None = None
        self.create_error_after_success = False
        self.comment_error: Exception | None = None
        self.comment_error_after_success = False
        self.tamper_comment_after_success = False
        self.next_number = 10

    def search_issues(self, repository: str, *, marker: str) -> SearchResult:
        del marker
        if self.search_error is not None:
            raise self.search_error
        if self.hide_issues_from_search:
            return SearchResult((), complete=True)
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
        self.get_issue_calls.append((repository, number))
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
        if self.tamper_comment_after_success:
            tampered = dict(comment)
            tampered["body"] = comment_body.replace(
                "robot_sf_audit_publication:v1", "robot_sf_audit_publication:forged", 1
            )
            updated = replace(issue, comments=(*issue.comments, tampered))
        self.issues = [updated if item.number == number else item for item in self.issues]
        if self.comment_error_after_success:
            raise self.comment_error or TimeoutError(
                "comment response lost after GitHub accepted POST"
            )
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
    assert result.remote_write == "ambiguous"
    assert replay.status == "unchanged"
    assert len(provider.comment_calls) == 1
    assert provider.update_calls == 0


def test_append_only_conflicting_comment_readback_is_retained_not_raised(
    tmp_path: Path,
) -> None:
    provider = AppendOnlyProvider()
    finding = _finding("comment-conflicting-readback")
    with GitHubOutbox(tmp_path) as outbox:
        sync = GitHubSync(provider, outbox)
        sync.sync(REPOSITORY, finding, operation_id="create")
        revised = replace(finding, observations=("conflicting comment",))
        provider.comment_error_after_success = True
        provider.tamper_comment_after_success = True
        result = sync.sync(REPOSITORY, revised, operation_id="comment-conflicting-readback")

    assert result.status in {"conflict", "ambiguous"}
    assert result.outbox is not None
    assert result.outbox.state in {"conflict", "ambiguous"}
    assert "publication" in result.reason or "comment" in result.reason
    assert len(provider.comment_calls) == 1


def test_initial_publication_body_carries_immutable_revision_identity() -> None:
    rendered = render_finding_issue(_finding("body-marker"), repository=REPOSITORY)
    assert "robot_sf_audit_finding:v1" in rendered.body


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "source must be text"),
        ("<!-- robot_sf_audit_publication:v1 malformed -->", "malformed"),
        (
            "<!-- robot_sf_audit_publication:v1 finding_id=x revision=0 -->\n"
            "<!-- robot_sf_audit_publication:v1 finding_id=x revision=0 -->",
            "duplicate",
        ),
        ("<!-- robot_sf_audit_publication:v1 finding_id=x revision=0 extra=x -->", "malformed"),
    ],
)
def test_publication_marker_parser_rejects_malformed_or_duplicate_tokens(
    value: Any,
    message: str,
) -> None:
    with pytest.raises(GitHubValidationError, match=message):
        _parse_publication_marker(value)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "body must be text"),
        ("no request marker", "one exact request marker"),
        ("forged prefix\n{marker}\n\nbody", "leading marker"),
        ("{marker}\n\n", "payload is empty"),
    ],
)
def test_comment_request_marker_parser_rejects_invalid_envelopes(
    value: Any,
    message: str,
) -> None:
    marker = _publication_request_marker("a" * 64)
    body = value if value is None else value.format(marker=marker)
    with pytest.raises(GitHubValidationError, match=message):
        _comment_payload_without_request_marker(body, publication_digest="a" * 64)


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


def test_append_only_preexisting_issue_is_adopted_without_body_rewrite(tmp_path: Path) -> None:
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
    # A marker-discovered issue with no local receipt is adopted as the
    # immutable initial publication.  Adoption must not append a synthetic
    # revision comment for the initial finding.
    assert result.status == "unchanged"
    assert result.replayed
    assert provider.issues[0].body == original_body
    assert len(provider.create_calls) == 0
    assert len(provider.comment_calls) == 0


def test_append_only_comment_failure_requires_explicit_retry_opt_in(tmp_path: Path) -> None:
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
        sync = GitHubSync(provider, outbox)
        result = sync.sync(
            REPOSITORY,
            replace(finding, observations=("comment failed",)),
            operation_id="comment-ambiguous",
        )
        refused = sync.sync(
            REPOSITORY,
            replace(finding, observations=("comment failed",)),
            operation_id="comment-ambiguous-replay",
        )
        provider.comment_error = None
        retried = sync.sync(
            REPOSITORY,
            replace(finding, observations=("comment failed",)),
            operation_id="comment-ambiguous-authorized-replay",
            retry_ambiguous=True,
        )
    assert result.status == "ambiguous"
    assert result.remote_write == "ambiguous"
    assert "comment outcome is ambiguous" in result.reason
    assert refused.status == "ambiguous"
    assert refused.remote_write == "none"
    assert "retry_ambiguous=True" in refused.reason
    assert retried.status == "commented"
    assert len(provider.comment_calls) == 1


def test_append_only_canonical_link_is_read_before_create(tmp_path: Path) -> None:
    finding = _finding("append-link-readback")
    seed = AppendOnlyProvider()
    with GitHubOutbox(tmp_path / "seed") as outbox:
        seeded = GitHubSync(seed, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="link-seed",
        )
    assert seeded.finding is not None and seeded.finding.github_issue is not None
    provider = AppendOnlyProvider()
    provider.issues = list(seed.issues)
    provider.hide_issues_from_search = True
    linked = replace(finding, github_issue=seeded.finding.github_issue)
    with GitHubOutbox(tmp_path / "recovery") as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            linked,
            operation_id="link-readback",
        )

    assert result.status == "unchanged"
    assert result.replayed
    assert provider.create_calls == []
    assert provider.get_issue_calls
    assert provider.get_issue_calls[0] == (REPOSITORY, seeded.issue.number)


def test_append_only_canonical_link_conflict_with_marker_search_is_explicit(
    tmp_path: Path,
) -> None:
    finding = _finding("append-link-conflict")
    seed = AppendOnlyProvider()
    with GitHubOutbox(tmp_path / "seed") as outbox:
        seeded = GitHubSync(seed, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="link-conflict-seed",
        )
    assert seeded.finding is not None and seeded.finding.github_issue is not None
    provider = AppendOnlyProvider()
    provider.issues = list(seed.issues)
    linked = dict(seeded.finding.github_issue)
    linked["number"] = seeded.issue.number + 100
    linked["url"] = f"https://github.com/{REPOSITORY}/issues/{linked['number']}"
    with GitHubOutbox(tmp_path / "recovery") as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            replace(finding, github_issue=linked),
            operation_id="link-conflict",
        )

    assert result.status == "conflict"
    assert "canonical finding link" in result.reason
    assert provider.create_calls == []
    assert result.outbox is not None and result.outbox.state == "conflict"


def test_append_only_same_revision_outbox_loss_without_link_reconciles_existing_comment(
    tmp_path: Path,
) -> None:
    finding = _finding("append-outbox-loss")
    provider = AppendOnlyProvider()
    revised = replace(finding, observations=("same semantic revision",))
    with GitHubOutbox(tmp_path / "original") as outbox:
        initial = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="outbox-loss-initial",
        )
        published = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            revised,
            operation_id="outbox-loss-comment",
        )
    assert initial.status == "created"
    assert published.status == "commented"
    assert published.finding is not None and published.finding.github_issue is not None
    assert published.outbox is not None and published.outbox.publication_revision == 0
    assert published.finding.github_issue["publication_kind"] == "revision_comment"
    assert len(provider.comment_calls) == 1
    with GitHubOutbox(tmp_path / "recovered") as outbox:
        replay = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            replace(published.finding, github_issue=None),
            operation_id="outbox-loss-replay",
        )

    assert replay.status == "reconciled"
    assert replay.replayed
    assert replay.comment == published.comment
    assert len(provider.create_calls) == 1
    assert len(provider.comment_calls) == 1

    with GitHubOutbox(tmp_path / "recovered") as outbox:
        second_replay = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            replay.finding or published.finding,
            operation_id="outbox-loss-second-replay",
        )

    assert second_replay.status == "unchanged"
    assert second_replay.replayed
    assert len(provider.create_calls) == 1
    assert len(provider.comment_calls) == 1


def test_append_only_finding_store_requires_create_reservation_before_mutation(
    tmp_path: Path,
) -> None:
    finding = _finding("append-reservation-required")
    provider = AppendOnlyProvider()
    with AuditStore(tmp_path / "audit") as store:
        finding_store = FindingStore(store)
        finding_store.create(finding, operation_id="create-canonical-finding")
        with GitHubOutbox(store) as outbox:
            result = GitHubSync(provider, outbox, finding_store=finding_store).sync(
                REPOSITORY,
                finding,
                operation_id="append-create-needs-reservation",
            )

    assert result.status == "conflict"
    assert "reservation" in result.reason
    assert provider.create_calls == []
    assert provider.comment_calls == []


def test_append_only_changed_finding_does_not_reuse_ambiguous_initial_payload(
    tmp_path: Path,
) -> None:
    class LostCreateProvider(AppendOnlyProvider):
        def create_issue(
            self,
            repository: str,
            *,
            title: str,
            body: str,
            labels: tuple[str, ...],
        ) -> GitHubIssue:
            self.create_calls.append({"title": title, "body": body})
            raise TimeoutError("create response was lost before apply")

    finding = _finding("append-changed-after-ambiguous-create")
    provider = LostCreateProvider()
    with GitHubOutbox(tmp_path) as outbox:
        sync = GitHubSync(provider, outbox)
        first = sync.sync(
            REPOSITORY,
            finding,
            operation_id="ambiguous-initial-create",
        )
        changed = replace(finding, observations=("changed while create was ambiguous",))
        retry = sync.sync(
            REPOSITORY,
            changed,
            operation_id="changed-finding-retry",
            retry_ambiguous=True,
        )
        receipt = outbox.find_publication_kind(
            REPOSITORY,
            finding.finding_id,
            "initial_issue",
        )

    assert first.status == "ambiguous"
    assert retry.status == "ambiguous"
    assert "different immutable finding snapshot" in retry.reason
    assert len(provider.create_calls) == 1
    assert provider.issues == []
    assert receipt is not None
    assert receipt.body != render_finding_issue(changed, repository=REPOSITORY).body
    assert retry.finding is not None and retry.finding.github_issue is None


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        ("missing", "conflict"),
        ("transport", "ambiguous"),
        ("identity", "conflict"),
    ],
)
def test_append_only_create_receipt_survives_failed_post_create_readback(
    tmp_path: Path,
    failure: str,
    expected_status: str,
) -> None:
    class FailedCreateReadbackProvider(AppendOnlyProvider):
        def get_issue(self, repository: str, number: int) -> GitHubIssue:
            if self.create_calls:
                self.get_issue_calls.append((repository, number))
                if failure == "missing":
                    raise GitHubIssueMissing("issue disappeared after create")
                if failure == "transport":
                    raise GitHubTransportError("post-create reread is unavailable")
                issue = super().get_issue(repository, number)
                return replace(issue, body="remote issue identity changed")
            return super().get_issue(repository, number)

    provider = FailedCreateReadbackProvider()
    with GitHubOutbox(tmp_path) as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            _finding(f"create-readback-{failure}"),
            operation_id=f"create-readback-{failure}",
        )

    assert result.status == expected_status
    assert result.remote_write == "applied"
    assert result.outbox is not None
    assert result.outbox.state == expected_status
    assert result.outbox.issue is not None
    assert len(provider.create_calls) == 1


def test_append_only_conflicted_initial_receipt_does_not_recreate_missing_issue(
    tmp_path: Path,
) -> None:
    class MissingAfterCreateReadbackProvider(AppendOnlyProvider):
        def get_issue(self, repository: str, number: int) -> GitHubIssue:
            if self.create_calls:
                self.get_issue_calls.append((repository, number))
                raise GitHubIssueMissing("issue disappeared after create")
            return super().get_issue(repository, number)

    finding = _finding("append-conflict-does-not-recreate")
    provider = MissingAfterCreateReadbackProvider()
    with GitHubOutbox(tmp_path) as outbox:
        sync = GitHubSync(provider, outbox)
        first = sync.sync(
            REPOSITORY,
            finding,
            operation_id="append-conflict-does-not-recreate",
        )
        assert first.status == "conflict"
        assert first.outbox is not None and first.outbox.issue is not None
        first_issue = first.outbox.issue
        provider.issues = []

        replay = sync.sync(
            REPOSITORY,
            finding,
            operation_id="append-conflict-does-not-recreate",
        )
        retained = outbox.find_publication_kind(
            REPOSITORY,
            finding.finding_id,
            "initial_issue",
        )
        claim = outbox.get_claim(REPOSITORY, finding.finding_id)

    assert replay.status == "conflict"
    assert replay.outbox is not None and replay.outbox.state == "conflict"
    assert replay.outbox.issue == first_issue
    assert retained is not None and retained.state == "conflict"
    assert retained.issue == first_issue
    assert claim is not None and claim.state == "conflict"
    assert provider.issues == []
    assert len(provider.create_calls) == 1


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        ("missing", "conflict"),
        ("transport", "ambiguous"),
        ("missing_comment", "conflict"),
        ("forged_comment", "conflict"),
    ],
)
def test_append_only_comment_receipt_survives_failed_final_readback(
    tmp_path: Path,
    failure: str,
    expected_status: str,
) -> None:
    class FailedCommentReadbackProvider(AppendOnlyProvider):
        def get_issue(self, repository: str, number: int) -> GitHubIssue:
            if self.comment_calls:
                self.get_issue_calls.append((repository, number))
                if failure == "missing":
                    raise GitHubIssueMissing("issue disappeared after comment")
                if failure == "transport":
                    raise GitHubTransportError("comment reread is unavailable")
                issue = super().get_issue(repository, number)
                if failure == "missing_comment":
                    return replace(issue, comments=issue.comments[:-1])
                comments = list(issue.comments)
                forged = dict(comments[-1])
                forged["body"] = str(forged["body"]).replace(
                    "robot_sf_audit_publication:v1",
                    "robot_sf_audit_publication:forged",
                    1,
                )
                comments[-1] = forged
                return replace(issue, comments=tuple(comments))
            return super().get_issue(repository, number)

    provider = FailedCommentReadbackProvider()
    finding = _finding(f"comment-readback-{failure}")
    revised = replace(finding, observations=("new observation",))
    with GitHubOutbox(tmp_path) as outbox:
        initial = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id=f"initial-{failure}",
        )
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            revised,
            operation_id=f"comment-{failure}",
        )

    assert initial.status == "created"
    assert result.status == expected_status
    assert result.remote_write == "applied"
    assert result.outbox is not None
    assert result.outbox.state == expected_status
    assert len(provider.comment_calls) == 1


def test_append_only_current_comment_link_id_mismatch_is_conflict(tmp_path: Path) -> None:
    finding = _finding("append-link-comment-id-conflict")
    provider = AppendOnlyProvider()
    revised = replace(finding, observations=("same semantic revision",))
    with GitHubOutbox(tmp_path / "original") as outbox:
        GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="comment-id-initial",
        )
        published = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            revised,
            operation_id="comment-id-revision",
        )
    assert published.finding is not None and published.finding.github_issue is not None
    linked = dict(published.finding.github_issue)
    linked["comment_id"] = int(linked["comment_id"]) + 1
    create_count = len(provider.create_calls)
    comment_count = len(provider.comment_calls)

    with GitHubOutbox(tmp_path / "recovery") as outbox:
        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            replace(published.finding, github_issue=linked),
            operation_id="comment-id-recovery",
        )

    assert result.status == "conflict"
    assert "exact remote publication comment" in result.reason
    assert len(provider.create_calls) == create_count
    assert len(provider.comment_calls) == comment_count


@pytest.mark.parametrize("tamper", ("provenance", "body"))
def test_append_only_comment_readback_rejects_forged_provenance_or_body(
    tmp_path: Path,
    tamper: str,
) -> None:
    finding = _finding(f"append-comment-{tamper}")
    revised = replace(finding, observations=("immutable comment payload",))
    provider = AppendOnlyProvider()
    with GitHubOutbox(tmp_path) as outbox:
        GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id=f"{tamper}-initial",
        )
        published = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            revised,
            operation_id=f"{tamper}-comment",
        )
        assert published.outbox is not None
        assert published.outbox.comment is not None
        issue = provider.issues[0]
        comment = dict(issue.comments[0])
        body = str(comment["body"])
        request_marker, payload = body.split("\n", 1)
        if tamper == "provenance":
            expected = f"revision=0 digest={published.outbox.publication_digest}"
            assert expected in payload
            payload = payload.replace(expected, f"revision=0 digest={'0' * 64}", 1)
        else:
            assert "**Candidate episodes:**" in payload
            payload = payload.replace("**Candidate episodes:**", "**Tampered episodes:**", 1)
        comment["body"] = f"{request_marker}\n{payload}"
        provider.issues = [replace(issue, comments=(comment,))]

        result = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            revised,
            operation_id=f"{tamper}-replay",
        )

    assert result.status == "conflict"
    assert "publication" in result.reason or "comment" in result.reason
    assert len(provider.comment_calls) == 1


def test_append_only_deleted_remote_issue_preserves_local_receipt_as_conflict(
    tmp_path: Path,
) -> None:
    finding = _finding("append-issue-deleted")
    provider = AppendOnlyProvider()
    with GitHubOutbox(tmp_path) as outbox:
        created = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="issue-deleted-create",
        )
        assert created.outbox is not None and created.outbox.issue is not None
        provider.issues = []
        replay = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="issue-deleted-replay",
        )

    assert replay.status == "conflict"
    assert "missing" in replay.reason
    assert replay.outbox is not None and replay.outbox.state == "conflict"
    assert replay.outbox.issue == created.outbox.issue
    assert replay.issue is not None and replay.issue.number == created.issue.number
    assert len(provider.create_calls) == 1


def test_append_only_legacy_issue_publication_marker_is_a_conflict(tmp_path: Path) -> None:
    finding = _finding("append-legacy-marker")
    provider = AppendOnlyProvider()
    with GitHubOutbox(tmp_path) as outbox:
        created = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="legacy-marker-create",
        )
        assert created.outbox is not None and created.outbox.issue is not None
        issue = provider.issues[0]
        legacy_body = "\n".join(
            line for line in issue.body.splitlines() if "robot_sf_audit_publication:v1" not in line
        )
        provider.issues = [replace(issue, body=legacy_body)]
        replay = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="legacy-marker-replay",
        )

    assert replay.status == "conflict"
    assert "publication" in replay.reason
    assert replay.outbox is not None and replay.outbox.state == "conflict"
    assert replay.outbox.issue == created.outbox.issue
    assert provider.create_calls and len(provider.create_calls) == 1


def test_append_only_deleted_remote_comment_preserves_local_receipt_as_conflict(
    tmp_path: Path,
) -> None:
    finding = _finding("append-comment-deleted")
    revised = replace(finding, observations=("comment to delete",))
    provider = AppendOnlyProvider()
    with GitHubOutbox(tmp_path) as outbox:
        GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="comment-deleted-initial",
        )
        published = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            revised,
            operation_id="comment-deleted-create",
        )
        assert published.outbox is not None and published.outbox.comment is not None
        issue = provider.issues[0]
        provider.issues = [replace(issue, comments=())]
        replay = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            revised,
            operation_id="comment-deleted-replay",
        )

    assert replay.status == "conflict"
    assert "comment" in replay.reason
    assert replay.outbox is not None and replay.outbox.state == "conflict"
    assert replay.outbox.comment == published.outbox.comment
    assert len(provider.comment_calls) == 1


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
        assert (
            outbox.find_publication(
                REPOSITORY,
                finding.finding_id,
                publication.publication_key,
            )
            is not None
        )
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
    assert result.status == "unchanged"
    assert result.replayed
    assert provider.comment_calls == []
    assert repaired is not None and repaired.state == "succeeded"


@pytest.mark.parametrize(
    ("scenario", "error_type"),
    (
        ("unknown-status", GitHubValidationError),
        ("missing-comment", GitHubValidationError),
        ("boolean-comment-id", GitHubValidationError),
        ("non-text-body", GitHubValidationError),
        ("missing-request-marker", GitHubValidationError),
        ("duplicate-request-marker", GitHubValidationError),
        ("misplaced-request-marker", GitHubValidationError),
        ("empty-payload", GitHubValidationError),
        ("missing-publication-marker", GitHubValidationError),
        ("wrong-payload", GitHubConflictError),
    ),
)
def test_append_only_comment_write_rejects_unverifiable_provider_responses(
    scenario: str,
    error_type: type[Exception],
) -> None:
    finding_id = "comment-response-finding"
    publication_digest = "b" * 64
    request_marker = _publication_request_marker(publication_digest)
    payload = (
        f"<!-- robot_sf_audit_publication:v1 finding_id={finding_id} "
        f"revision=2 digest={publication_digest} -->\nAuditor comment"
    )
    exact_body = f"{request_marker}\n{payload}"
    comment: dict[str, Any] = {"id": 7, "body": exact_body}
    expected_body: str | None = payload

    if scenario == "unknown-status":
        response: Any = {"status": "pending", "comment": comment}
    elif scenario == "missing-comment":
        response = {"status": "created", "comment": None}
    elif scenario == "boolean-comment-id":
        response = {"status": "created", "comment": {**comment, "id": True}}
    elif scenario == "non-text-body":
        response = {"status": "created", "comment": {**comment, "body": None}}
    elif scenario == "missing-request-marker":
        response = {"status": "created", "comment": {**comment, "body": payload}}
    elif scenario == "duplicate-request-marker":
        response = {
            "status": "created",
            "comment": {**comment, "body": f"{request_marker}\n{request_marker}\n{payload}"},
        }
    elif scenario == "misplaced-request-marker":
        response = {
            "status": "created",
            "comment": {**comment, "body": f"prefix\n{request_marker}\n{payload}"},
        }
    elif scenario == "empty-payload":
        response = {"status": "created", "comment": {**comment, "body": request_marker}}
    elif scenario == "missing-publication-marker":
        response = {
            "status": "created",
            "comment": {**comment, "body": f"{request_marker}\nplain text"},
        }
        expected_body = None
    else:
        response = {"status": "created", "comment": comment}
        expected_body = "different request body"

    with pytest.raises(error_type):
        _coerce_comment_write(
            response,
            publication_digest=publication_digest,
            expected_body=expected_body,
        )


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    (
        ("schema_version", "unknown", GitHubValidationError),
        ("repository", "other/repository", GitHubConflictError),
        ("number", True, GitHubValidationError),
        ("marker", "forged marker", GitHubConflictError),
        ("url", "file:///private/path", GitHubValidationError),
        ("publication_digest", "not-a-digest", GitHubValidationError),
        ("publication_kind", "future-kind", GitHubValidationError),
        ("publication_revision", True, GitHubValidationError),
        ("comment_id", True, GitHubValidationError),
    ),
)
def test_append_only_canonical_link_rejects_malformed_identity_fields(
    field: str,
    value: Any,
    error_type: type[Exception],
) -> None:
    finding = _finding("malformed-link-finding")
    rendered = render_finding_issue(finding, repository=REPOSITORY)
    link = {
        "repository": REPOSITORY,
        "number": 10,
        "marker": rendered.marker,
    }
    link[field] = value

    with pytest.raises(error_type):
        _linked_issue_identity(replace(finding, github_issue=link), REPOSITORY, rendered.marker)


def test_append_only_marker_search_rejects_wrong_repository_and_conflicting_markers() -> None:
    finding = _finding("marker-search-finding")
    rendered = render_finding_issue(finding, repository=REPOSITORY)
    matching_issue = GitHubIssue(
        repository=REPOSITORY,
        number=10,
        url=f"https://github.com/{REPOSITORY}/issues/10",
        title=rendered.title,
        body=rendered.body,
        labels=rendered.labels,
    )
    wrong_repository = replace(matching_issue, repository="other/repository")
    with pytest.raises(GitHubValidationError, match="wrong repository"):
        _select_marker_issue(
            SearchResult((wrong_repository,), complete=True),
            repository=REPOSITORY,
            finding_id=finding.finding_id,
        )

    other = render_finding_issue(_finding("other-finding"), repository=REPOSITORY)
    conflicting_issue = replace(matching_issue, number=11, body=other.body)
    with pytest.raises(GitHubConflictError, match="conflicting finding marker"):
        _select_marker_issue(
            SearchResult((conflicting_issue,), complete=True),
            repository=REPOSITORY,
            finding_id=finding.finding_id,
        )

    duplicate_issue = replace(matching_issue, number=12)
    with pytest.raises(GitHubConflictError, match="more than one"):
        _select_marker_issue(
            SearchResult((matching_issue, duplicate_issue), complete=True),
            repository=REPOSITORY,
            finding_id=finding.finding_id,
        )


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    (
        ("missing-request-marker", GitHubConflictError),
        ("newer-revision", GitHubConflictError),
        ("same-revision-digest", GitHubConflictError),
        ("duplicate-request-marker", GitHubConflictError),
    ),
)
def test_append_only_exact_comment_lookup_rejects_forged_or_newer_comments(
    tmp_path: Path,
    tamper: str,
    expected_error: type[Exception],
) -> None:
    finding = _finding(f"exact-comment-{tamper}")
    revised = replace(finding, observations=("published comment",))
    provider = AppendOnlyProvider()
    with GitHubOutbox(tmp_path) as outbox:
        GitHubSync(provider, outbox).sync(REPOSITORY, finding, operation_id="initial")
        published = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            revised,
            operation_id="revision",
        )
    assert published.outbox is not None and published.outbox.comment is not None
    issue = provider.issues[0]
    comment = dict(issue.comments[0])
    body = str(comment["body"])
    request_marker, payload = body.split("\n", 1)
    digest = published.outbox.publication_digest
    if tamper == "missing-request-marker":
        comment["body"] = payload
    elif tamper == "newer-revision":
        comment["body"] = f"{request_marker}\n{payload.replace('revision=0', 'revision=1', 1)}"
    elif tamper == "same-revision-digest":
        comment["body"] = f"{request_marker}\n{payload.replace(digest, '0' * 64, 1)}"
    else:
        comment["body"] = f"{request_marker}\n{request_marker}\n{payload}"
    issue_with_tampered_comment = replace(issue, comments=(comment,))

    with pytest.raises(expected_error):
        _find_exact_publication_comment(
            issue_with_tampered_comment,
            finding_id=finding.finding_id,
            finding_revision=0,
            publication_digest=digest,
            expected_body=payload.lstrip("\n"),
        )


@pytest.mark.parametrize("tamper", ("expected-block", "provenance"))
def test_append_only_publication_payload_validator_rejects_receipt_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    finding = _finding(f"payload-validator-{tamper}")
    provider = AppendOnlyProvider()
    with GitHubOutbox(tmp_path) as outbox:
        published = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="payload-validator-initial",
        )
    assert published.outbox is not None
    entry = published.outbox
    rendered = render_finding_issue(finding, repository=REPOSITORY)
    arguments = {
        "finding_revision": entry.publication_revision,
        "publication_kind": entry.publication_kind,
        "publication_digest": entry.publication_digest,
        "publication_key": entry.publication_key,
        "body": entry.body,
    }
    _validate_publication_entry_payload(entry, rendered, **arguments)

    if tamper == "expected-block":
        entry = replace(entry, expected_block_digest="0" * 64)
    else:
        provenance = dict(entry.publication_provenance or {})
        provenance["repository"] = "other/repository"
        entry = replace(entry, publication_provenance=provenance)

    with pytest.raises(GitHubConflictError, match="payload/provenance mismatch"):
        _validate_publication_entry_payload(entry, rendered, **arguments)


@pytest.mark.parametrize(
    "tamper",
    (
        "different-rendered-identity",
        "expected-block",
        "duplicate-publication-marker",
        "publication-marker-identity",
        "auditor-block",
        "provenance",
        "adoption-mode",
        "publication-key",
        "request-digest",
    ),
)
def test_append_only_initial_receipt_identity_rejects_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    finding = _finding(f"initial-identity-{tamper}")
    provider = AppendOnlyProvider()
    with GitHubOutbox(tmp_path) as outbox:
        published = GitHubSync(provider, outbox).sync(
            REPOSITORY,
            finding,
            operation_id="identity-validator-initial",
        )
    assert published.outbox is not None and published.issue is not None
    entry = published.outbox
    rendered = render_finding_issue(finding, repository=REPOSITORY)

    if tamper == "different-rendered-identity":
        rendered = render_finding_issue(finding, repository="other/repository")
    elif tamper == "expected-block":
        entry = replace(entry, expected_block_digest="0" * 64)
    elif tamper == "duplicate-publication-marker":
        marker_line = entry.body.splitlines()[0]
        entry = replace(entry, body=f"{entry.body}\n{marker_line}")
    elif tamper == "publication-marker-identity":
        entry = replace(
            entry,
            body=entry.body.replace(
                f"finding_id={finding.finding_id}", "finding_id=forged-finding", 1
            ),
        )
    elif tamper == "auditor-block":
        entry = replace(
            entry,
            body=entry.body.replace("Confirmed episodes (", "Tampered episodes (", 1),
        )
    elif tamper == "provenance":
        provenance = dict(entry.publication_provenance or {})
        provenance["repository"] = "other/repository"
        entry = replace(entry, publication_provenance=provenance)
    elif tamper == "adoption-mode":
        entry = _adopt_initial_publication_snapshot(entry, published.issue)
        provenance = dict(entry.publication_provenance or {})
        provenance["adoption_mode"] = "unrecognized"
        entry = replace(entry, publication_provenance=provenance)
    elif tamper == "publication-key":
        entry = replace(entry, publication_key="0" * 64)
    else:
        entry = replace(entry, request_digest="0" * 64)

    with pytest.raises(GitHubConflictError, match="identity mismatch|malformed"):
        _validate_initial_entry_identity(entry, rendered)
