"""Unit tests for scripts/ci/pr_contract_check.py."""

# evidence-writer-exempt: these tests intentionally write temporary evidence-path fixtures,
# including malformed files, to exercise the PR contract and writer-guard behavior.

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.ci import pr_contract_check
from tests.support.environment_guards import configure_git_identity

ROOT = Path(__file__).resolve().parents[2]

# PR #8440 is the known pre-guard regression: its merge reference closed the
# incident in #8414 before the two-green reconciler criterion was established.
KNOWN_HISTORICAL_MAIN_CI_CLOSING_GUARD_HITS = {8440: {"8414"}}

# PRs #9565/#9569 are the known pre-guard regressions for issue #9566: their
# bodies contain negated prose ("does not close #9489") that GitHub parsed as
# a closing reference and auto-closed parent #9489 on merge. The parity guard
# must keep flagging these historical bodies while future PRs stay clean.
KNOWN_HISTORICAL_GITHUB_PARITY_HITS = {9565: {"9489"}, 9569: {"9489"}}


# Keep this mapping deliberately narrow: entries must identify one merged PR,
# one linked issue, and the immutable GitHub file-stat totals that prove the
# historical breach. The current last-20 inventory has no such breach: PR
# #9122 is 631 additions/1 deletion against issue #8851's 800-line cap, and
# PR #9118 is 750 additions/1 deletion against issue #8856's 750-line cap.
@dataclass(frozen=True)
class HistoricalPREvidence:
    """File statistics bound to the immutable revisions of one merged PR."""

    pr_number: int
    base_sha: str
    head_sha: str
    merge_commit_sha: str
    merge_parent_shas: tuple[str, ...]
    merge_base_sha: str
    changed_files: tuple[str, ...]
    numstat: pr_contract_check.HistoricalNumstatEvidence

    def __post_init__(self) -> None:
        """Reject malformed or internally inconsistent compatibility evidence."""
        if (
            not isinstance(self.pr_number, int)
            or isinstance(self.pr_number, bool)
            or self.pr_number <= 0
        ):
            raise ValueError("historical evidence PR number must be positive")
        if (
            not isinstance(self.merge_parent_shas, tuple)
            or not self.merge_parent_shas
            or any(
                not isinstance(parent_sha, str)
                or re.fullmatch(r"[0-9a-fA-F]{40}", parent_sha) is None
                for parent_sha in self.merge_parent_shas
            )
            or len(self.merge_parent_shas) != len(set(self.merge_parent_shas))
        ):
            raise ValueError("historical evidence merge parents must be unique full commit IDs")
        sha_values = (
            self.base_sha,
            self.head_sha,
            self.merge_commit_sha,
            self.merge_base_sha,
        )
        if any(
            not isinstance(sha, str) or re.fullmatch(r"[0-9a-fA-F]{40}", sha) is None
            for sha in sha_values
        ):
            raise ValueError("historical evidence revisions must be full commit IDs")
        if self.merge_base_sha != self.base_sha:
            raise ValueError("historical evidence merge base must equal the comparison base")
        if not isinstance(self.changed_files, tuple):
            raise ValueError("historical evidence changed files must be immutable")
        if not isinstance(self.numstat, pr_contract_check.HistoricalNumstatEvidence):
            raise ValueError("historical evidence must carry validated numstat")
        if self.changed_files != self.numstat.changed_files:
            raise ValueError("historical evidence files do not match validated numstat")


# This is a test-only compatibility fixture. It deliberately cannot authorize a live
# regression sweep; production exceptions remain empty until a real merged breach is
# reviewed and recorded with its exact immutable identity and totals.
HISTORICAL_BUDGET_EXCEPTION_FIXTURES: dict[int, dict[int, dict[str, object]]] = {
    9000: {
        8000: {
            "identity": {
                "base_sha": "a" * 40,
                "head_sha": "b" * 40,
                "merge_commit_sha": "c" * 40,
                "merge_parent_shas": ["a" * 40],
                "merge_base_sha": "a" * 40,
            },
            "stats": {"files": 3, "added": 825, "deleted": 0, "net": 825},
        }
    }
}

# No live compatibility exceptions are currently authorized.
KNOWN_HISTORICAL_BUDGET_EXCEPTIONS: dict[int, dict[int, dict[str, object]]] = {}


def _matches_historical_budget_exception(
    evidence: HistoricalPREvidence, issue_number: int, record: dict[str, object]
) -> bool:
    """Match only when PR identity and every exact immutable statistic agree."""
    identity = record.get("identity")
    stats = record.get("stats")
    if not isinstance(identity, dict) or not isinstance(stats, dict):
        return False
    expected_identity = {
        "base_sha": evidence.base_sha,
        "head_sha": evidence.head_sha,
        "merge_commit_sha": evidence.merge_commit_sha,
        "merge_parent_shas": list(evidence.merge_parent_shas),
        "merge_base_sha": evidence.merge_base_sha,
    }
    if (
        identity != expected_identity
        or evidence.pr_number not in KNOWN_HISTORICAL_BUDGET_EXCEPTIONS
    ):
        return False
    if set(stats) != {"files", "added", "deleted", "net"}:
        return False
    if (
        evidence.merge_base_sha != evidence.base_sha
        or not evidence.merge_parent_shas
        or any(
            not isinstance(parent_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", parent_sha)
            for parent_sha in evidence.merge_parent_shas
        )
    ):
        return False
    expected = KNOWN_HISTORICAL_BUDGET_EXCEPTIONS[evidence.pr_number].get(issue_number)
    measured = {
        "files": evidence.numstat.files,
        "added": evidence.numstat.added,
        "deleted": evidence.numstat.deleted,
        "net": evidence.numstat.net,
    }
    return expected == record and all(
        isinstance(value, int) and not isinstance(value, bool) and measured.get(key) == value
        for key, value in stats.items()
    )


def _is_expected_historical_budget_blocker(
    evidence: HistoricalPREvidence, body: str, blocker: str
) -> bool:
    """Return whether a budget blocker is covered by an exact exception record."""
    match = re.match(
        r"^BLOCKER: PR exceeds the budget declared in issue #(?P<issue>[0-9]+)\b",
        blocker,
    )
    if match is None:
        return False
    issue_number = int(match.group("issue"))
    closed_issue_numbers = {
        int(issue) for issue in pr_contract_check.find_closed_issues(body, "ll7/robot_sf_ll7")
    }
    if issue_number not in closed_issue_numbers:
        return False
    return _matches_historical_budget_exception(
        evidence,
        issue_number,
        KNOWN_HISTORICAL_BUDGET_EXCEPTIONS.get(evidence.pr_number, {}).get(issue_number, {}),
    )


def _fetch_historical_pr_identity(number: int, repo: str) -> tuple[str, str, str] | None:
    """Read and validate the immutable revision identity of a merged PR."""
    try:
        metadata_response = subprocess.run(
            ["gh", "api", f"repos/{repo}/pulls/{number}"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        metadata = json.loads(metadata_response.stdout)
        if not isinstance(metadata, dict) or not _is_valid_github_merged_at(
            metadata.get("merged_at")
        ):
            return None
        if metadata.get("state") != "closed" or metadata.get("merged") is not True:
            return None
        base = metadata.get("base")
        head = metadata.get("head")
        base_sha = base.get("sha") if isinstance(base, dict) else None
        head_sha = head.get("sha") if isinstance(head, dict) else None
        merge_commit_sha = metadata.get("merge_commit_sha")
        sha_values = (base_sha, head_sha, merge_commit_sha)
        if any(
            not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", sha)
            for sha in sha_values
        ):
            return None
        return tuple(sha.lower() for sha in sha_values)  # type: ignore[return-value]
    except (subprocess.SubprocessError, OSError, ValueError, TypeError):
        return None


_GITHUB_MERGED_AT_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")


def _is_valid_github_merged_at(value: object) -> bool:
    """Return whether *value* has GitHub's UTC timestamp shape and valid date."""
    if not isinstance(value, str) or _GITHUB_MERGED_AT_PATTERN.fullmatch(value) is None:
        return False
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return True


def _historical_compare_reaches_target(
    comparison: object,
    base_sha: str,
    target_sha: str,
    *,
    expected_merge_base_sha: str | None = None,
) -> bool:
    """Validate immutable endpoint identities in a GitHub compare response."""
    if not isinstance(comparison, dict):
        return False
    compare_base = comparison.get("base_commit")
    compare_merge_base = comparison.get("merge_base_commit")
    compare_commits = comparison.get("commits")
    total_commits = comparison.get("total_commits")
    if (
        not isinstance(compare_base, dict)
        or compare_base.get("sha") != base_sha
        or not isinstance(compare_merge_base, dict)
        or not isinstance(compare_merge_base.get("sha"), str)
        or not re.fullmatch(r"[0-9a-fA-F]{40}", compare_merge_base["sha"])
        or not isinstance(compare_commits, list)
        or not compare_commits
        or not isinstance(total_commits, int)
        or isinstance(total_commits, bool)
        or total_commits != len(compare_commits)
    ):
        return False
    if expected_merge_base_sha is not None and compare_merge_base["sha"] != expected_merge_base_sha:
        return False
    commit_shas: set[str] = set()
    for commit in compare_commits:
        if (
            not isinstance(commit, dict)
            or not isinstance(commit.get("sha"), str)
            or not re.fullmatch(r"[0-9a-fA-F]{40}", commit["sha"])
            or commit["sha"] in commit_shas
        ):
            return False
        commit_shas.add(commit["sha"])
    last_compare_commit = compare_commits[-1]
    return isinstance(last_compare_commit, dict) and last_compare_commit.get("sha") == target_sha


def _fetch_historical_compare(
    repo: str, base_sha: str, head_sha: str
) -> tuple[tuple[str, ...], pr_contract_check.HistoricalNumstatEvidence] | None:
    """Read file statistics from a compare addressed by full commit IDs."""
    try:
        compare_response = subprocess.run(
            ["gh", "api", f"repos/{repo}/compare/{base_sha}...{head_sha}?per_page=100"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        comparison = json.loads(compare_response.stdout)
        if not isinstance(comparison, dict):
            return None
        if not _historical_compare_reaches_target(comparison, base_sha, head_sha):
            return None
        files = comparison.get("files")
        if not isinstance(files, list) or not files or len(files) >= 100:
            return None
        rows: list[tuple[str, int, int]] = []
        seen_filenames: set[str] = set()
        for item in files:
            if not isinstance(item, dict):
                return None
            filename = item.get("filename")
            additions = item.get("additions")
            deletions = item.get("deletions")
            if (
                not isinstance(filename, str)
                or not filename
                or any(unicodedata.category(character) == "Cc" for character in filename)
                or isinstance(additions, bool)
                or not isinstance(additions, int)
                or additions < 0
                or isinstance(deletions, bool)
                or not isinstance(deletions, int)
                or deletions < 0
                or filename in seen_filenames
            ):
                return None
            seen_filenames.add(filename)
            rows.append((filename, additions, deletions))
        changed_files = tuple(filename for filename, _, _ in rows)
        numstat = "".join(
            f"{additions}\t{deletions}\t{filename}\n" for filename, additions, deletions in rows
        )
        validated_numstat = pr_contract_check.HistoricalNumstatEvidence.from_numstat(numstat)
        if changed_files != validated_numstat.changed_files:
            return None
        return changed_files, validated_numstat
    except (subprocess.SubprocessError, OSError, ValueError, TypeError):
        return None


def _fetch_historical_merge_binding(
    repo: str, base_sha: str, merge_commit_sha: str
) -> tuple[tuple[str, ...], str] | None:
    """Bind the exact merge commit to the immutable base revision."""
    try:
        merge_response = subprocess.run(
            ["gh", "api", f"repos/{repo}/commits/{merge_commit_sha}"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        merge_commit = json.loads(merge_response.stdout)
        if not isinstance(merge_commit, dict):
            return None
        if merge_commit.get("sha") != merge_commit_sha:
            return None
        parents = merge_commit.get("parents")
        if not isinstance(parents, list) or not parents:
            return None
        parent_shas: list[str] = []
        for parent in parents:
            parent_sha = parent.get("sha") if isinstance(parent, dict) else None
            if not isinstance(parent_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", parent_sha):
                return None
            parent_shas.append(parent_sha.lower())
        if len(parent_shas) != len(set(parent_shas)):
            return None
        # The PR diff compare binds base -> head; this second immutable compare
        # binds the same base -> merge target. Squash/rebase merges may have a
        # different direct parent, so the compare's merge-base is the relation
        # checked here instead of assuming base is a direct merge parent.
        merge_compare_response = subprocess.run(
            ["gh", "api", f"repos/{repo}/compare/{base_sha}...{merge_commit_sha}?per_page=100"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        merge_comparison = json.loads(merge_compare_response.stdout)
        if not _historical_compare_reaches_target(
            merge_comparison,
            base_sha,
            merge_commit_sha,
            expected_merge_base_sha=base_sha,
        ):
            return None
        return tuple(parent_shas), base_sha
    except (subprocess.SubprocessError, OSError, ValueError, TypeError):
        return None


def _fetch_historical_pr_evidence(
    number: int, repo: str = "ll7/robot_sf_ll7"
) -> HistoricalPREvidence | None:
    """Fetch file stats from a SHA-bound compare for one merged PR."""
    identity = _fetch_historical_pr_identity(number, repo)
    if identity is None:
        return None
    base_sha, head_sha, merge_commit_sha = identity
    comparison = _fetch_historical_compare(repo, base_sha, head_sha)
    if comparison is None:
        return None
    merge_binding = _fetch_historical_merge_binding(repo, base_sha, merge_commit_sha)
    if merge_binding is None:
        return None
    merge_parent_shas, merge_base_sha = merge_binding
    changed_files, numstat = comparison
    return HistoricalPREvidence(
        pr_number=number,
        base_sha=base_sha,
        head_sha=head_sha,
        merge_commit_sha=merge_commit_sha,
        merge_parent_shas=merge_parent_shas,
        merge_base_sha=merge_base_sha,
        changed_files=tuple(changed_files),
        numstat=numstat,
    )


def _valid_review_sidecar(artifact: Path, artifact_path: str) -> dict[str, object]:
    """Build a valid immutable-evidence review sidecar payload for a fixture artifact."""
    return {
        "schema_version": "evidence-review-marker.v1",
        "artifact_path": artifact_path,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "review_marker": "AI-GENERATED NEEDS-REVIEW",
        "preserved_exact_bytes": True,
    }


def test_find_closed_issues() -> None:
    """Test find_closed_issues matches closing keywords."""
    body = (
        "This fixes #123, closes: #456, and resolves ll7/robot_sf_ll7#789. "
        "Fixes https://github.com/ll7/robot_sf_ll7/issues/1011."
    )
    closed = pr_contract_check.find_closed_issues(body)
    assert closed == ["123", "456", "789", "1011"]


def test_find_closed_issues_keeps_cross_repository_references_parseable() -> None:
    """Qualified references are parsed so the discipline rule can ignore other repos."""
    body = "Closes other-org/other-repo#123 and closes ll7/robot_sf_ll7#456"
    assert pr_contract_check.find_closed_issues(body) == ["123", "456"]
    assert pr_contract_check.find_closed_issues(body, "ll7/robot_sf_ll7") == ["456"]


def test_find_title_issues() -> None:
    """Test find_title_issues parses issue numbers from title."""
    assert pr_contract_check.find_title_issues("Issue #4735: some title") == ["4735"]
    assert pr_contract_check.find_title_issues("Refs #123, issue 456") == ["123", "456"]


def test_has_declaration_for_issue() -> None:
    """Test has_declaration_for_issue checks body matches."""
    body = (
        "We reference Refs #123 here, close: #456, and fix ll7/robot_sf_ll7#789. "
        "Resolves https://github.com/ll7/robot_sf_ll7/issues/1011."
    )
    assert pr_contract_check.has_declaration_for_issue("123", body) is True
    assert pr_contract_check.has_declaration_for_issue("456", body) is True
    assert pr_contract_check.has_declaration_for_issue("789", body) is True
    assert pr_contract_check.has_declaration_for_issue("1011", body) is True
    assert pr_contract_check.has_declaration_for_issue("999", body) is False


@patch("subprocess.run")
def test_check_closes_discipline(mock_run: MagicMock) -> None:
    """Test check_closes_discipline protects special issue lifecycles."""
    # Test case 1: Issue has no epic label
    mock_run.return_value = MagicMock(
        returncode=0, stdout='{"labels": [{"name": "bug"}], "body": ""}'
    )
    blockers = pr_contract_check.check_closes_discipline("Closes #123", "ll7/robot_sf_ll7")
    assert not blockers

    # Test case 2: Issue has epic label
    mock_run.return_value = MagicMock(
        returncode=0, stdout='{"labels": [{"name": "epic"}], "body": ""}'
    )
    blockers = pr_contract_check.check_closes_discipline("Closes #123", "ll7/robot_sf_ll7")
    assert len(blockers) == 1
    assert "epic" in blockers[0]

    # A canonical marker blocks all semantic closing keywords, including a repair PR.
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps(
            {"labels": [], "body": "<!-- ll7-main-red-incident:v1 -->\nAutomated incident."}
        ),
    )
    blockers = pr_contract_check.check_closes_discipline("Fixes #8414", "ll7/robot_sf_ll7")
    assert len(blockers) == 1
    assert "main continuous-integration (CI) incident" in blockers[0]
    assert "Refs #8414" in blockers[0]
    assert "two consecutive decisive green runs" in blockers[0]

    # The compatibility label protects marker-less incidents as well.
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps({"labels": [{"name": "ll7-main-red-incident:v1"}], "body": ""}),
    )
    blockers = pr_contract_check.check_closes_discipline("Resolves #8441", "ll7/robot_sf_ll7")
    assert len(blockers) == 1
    assert "main continuous-integration (CI) incident" in blockers[0]

    # A failed metadata read is unknown, not evidence that semantic closure is safe.
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="API unavailable")
    blockers = pr_contract_check.check_closes_discipline("Closes #999", "ll7/robot_sf_ll7")
    assert len(blockers) == 1
    assert "fails closed" in blockers[0]


@patch("scripts.ci.pr_contract_check.get_issue_metadata")
def test_check_closes_discipline_scans_commit_messages(mock_metadata: MagicMock) -> None:
    """Commit-message closing keywords receive the same lifecycle protection as body keywords."""
    mock_metadata.return_value = (
        [],
        "<!-- ll7-main-red-incident:v1 -->\nAutomated incident.",
    )

    blockers = pr_contract_check.check_closes_discipline(
        "Adds a repair without a body closing keyword.",
        "ll7/robot_sf_ll7",
        commit_messages="Implement repair\n\nCloses: #8414\n",
        commit_messages_checked=True,
    )

    assert len(blockers) == 1
    assert "PR commit message" in blockers[0]
    mock_metadata.assert_called_once_with("8414", "ll7/robot_sf_ll7")


@patch("scripts.ci.pr_contract_check.get_issue_metadata")
def test_check_closes_discipline_ignores_other_repository(mock_metadata: MagicMock) -> None:
    """A qualified close for another repository is not a local lifecycle mutation."""
    blockers = pr_contract_check.check_closes_discipline(
        "Closes other-org/other-repo#8414",
        "ll7/robot_sf_ll7",
    )

    assert not blockers
    mock_metadata.assert_not_called()


@patch("scripts.ci.pr_contract_check.subprocess.run")
def test_get_issue_metadata_requires_complete_payload(mock_run: MagicMock) -> None:
    """Partial issue responses cannot be treated as evidence that closure is safe."""
    mock_run.return_value = MagicMock(returncode=0, stdout='{"labels": []}')
    assert pr_contract_check.get_issue_metadata("8414", "ll7/robot_sf_ll7") is None

    mock_run.return_value = MagicMock(returncode=0, stdout='{"body": ""}')
    assert pr_contract_check.get_issue_metadata("8414", "ll7/robot_sf_ll7") is None


@pytest.mark.parametrize(
    "stdout, expected",
    (
        ('{"head_sha":"' + "a" * 40 + '","base_sha":"' + "b" * 40 + '"}', ("a" * 40, "b" * 40)),
        ('{"head_sha":"short","base_sha":"' + "b" * 40 + '"}', None),
        ('{"head_sha":"' + "a" * 40 + '","base_sha":null}', None),
    ),
)
@patch("scripts.ci.pr_contract_check.subprocess.run")
def test_get_pr_label_guard_shas_validates_exact_pair(
    mock_run: MagicMock, stdout: str, expected: tuple[str, str] | None
) -> None:
    """PR label callers require a complete full-SHA head/base pair."""
    mock_run.return_value = MagicMock(returncode=0, stdout=stdout)

    assert pr_contract_check.get_pr_label_guard_shas("8451", "ll7/robot_sf_ll7") == expected


@patch("scripts.ci.pr_contract_check.subprocess.run")
def test_get_pr_label_guard_shas_uses_pull_request_rest_endpoint(mock_run: MagicMock) -> None:
    """The guard pair comes from the exact PR REST object rather than a branch name."""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout='{"head_sha":"' + "a" * 40 + '","base_sha":"' + "b" * 40 + '"}',
    )

    assert pr_contract_check.get_pr_label_guard_shas("8451", "ll7/robot_sf_ll7") == (
        "a" * 40,
        "b" * 40,
    )
    mock_run.assert_called_once_with(
        [
            "gh",
            "api",
            "repos/ll7/robot_sf_ll7/pulls/8451",
            "--jq",
            "{head_sha: .head.sha, base_sha: .base.sha}",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@patch("scripts.ci.pr_contract_check.subprocess.run")
def test_get_pr_label_guard_shas_rejects_non_ascii_number(mock_run: MagicMock) -> None:
    """Only ASCII decimal PR numbers may reach the REST endpoint."""
    assert pr_contract_check.get_pr_label_guard_shas("１２３", "ll7/robot_sf_ll7") is None
    mock_run.assert_not_called()


@patch("scripts.ci.pr_contract_check.subprocess.run")
def test_get_pr_commit_messages_uses_paginated_commit_api(mock_run: MagicMock) -> None:
    """The commit source is fetched through the paginated PR commits endpoint."""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="\n".join(
            base64.b64encode(message.encode("utf-8")).decode("ascii")
            for message in ("first line\nsecond line — café", "second")
        )
        + "\n",
    )

    assert (
        pr_contract_check.get_pr_commit_messages("8451", "ll7/robot_sf_ll7")
        == "first line\nsecond line — café\nsecond\n"
    )
    mock_run.assert_called_once_with(
        [
            "gh",
            "api",
            "--paginate",
            "repos/ll7/robot_sf_ll7/pulls/8451/commits?per_page=100",
            "--jq",
            '.[] | if (.commit? | type) == "object" and (.commit.message? | type) == "string" '
            "then .commit.message | @base64 "
            'else error("invalid commit metadata") end',
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@patch("scripts.ci.pr_contract_check.subprocess.run")
def test_get_pr_commit_messages_rejects_empty_success(mock_run: MagicMock) -> None:
    """A successful empty response is unavailable evidence, not a verified commit list."""
    mock_run.return_value = MagicMock(returncode=0, stdout=" \n")

    assert pr_contract_check.get_pr_commit_messages("8451", "ll7/robot_sf_ll7") is None


@patch("scripts.ci.pr_contract_check.subprocess.run")
def test_get_pr_commit_messages_rejects_nonempty_malformed_output(mock_run: MagicMock) -> None:
    """A malformed jq value cannot masquerade as available commit evidence."""
    mock_run.return_value = MagicMock(returncode=0, stdout="null\n")

    assert pr_contract_check.get_pr_commit_messages("8451", "ll7/robot_sf_ll7") is None


@pytest.mark.parametrize("separator", ("\n", " \t\n"))
@patch("scripts.ci.pr_contract_check.subprocess.run")
def test_get_pr_commit_messages_rejects_mixed_blank_or_whitespace_lines(
    mock_run: MagicMock, separator: str
) -> None:
    """Blank or whitespace-only records cannot be dropped from commit evidence."""
    encoded = base64.b64encode(b"first").decode("ascii")
    mock_run.return_value = MagicMock(returncode=0, stdout=f"{encoded}\n{separator}{encoded}\n")

    assert pr_contract_check.get_pr_commit_messages("8451", "ll7/robot_sf_ll7") is None


@pytest.mark.parametrize("output", (" {encoded}\n", "{encoded} \n", "\t{encoded}\n"))
@patch("scripts.ci.pr_contract_check.subprocess.run")
def test_get_pr_commit_messages_rejects_surrounding_whitespace(
    mock_run: MagicMock, output: str
) -> None:
    """Base64 records must contain no surrounding whitespace before strict decoding."""
    encoded = base64.b64encode(b"first").decode("ascii")
    mock_run.return_value = MagicMock(returncode=0, stdout=output.format(encoded=encoded))

    assert pr_contract_check.get_pr_commit_messages("8451", "ll7/robot_sf_ll7") is None


@pytest.mark.parametrize(
    "encoded",
    (
        "not-base64",
        base64.b64encode(b"\xff\xfe").decode("ascii"),
    ),
)
@patch("scripts.ci.pr_contract_check.subprocess.run")
def test_get_pr_commit_messages_rejects_invalid_base64_or_utf8(
    mock_run: MagicMock, encoded: str
) -> None:
    """Malformed base64 and non-UTF-8 decoded bytes are unavailable evidence."""
    mock_run.return_value = MagicMock(returncode=0, stdout=f"{encoded}\n")

    assert pr_contract_check.get_pr_commit_messages("8451", "ll7/robot_sf_ll7") is None


@patch("scripts.ci.pr_contract_check.get_issue_metadata")
def test_check_closes_discipline_fails_closed_when_commit_source_unavailable(
    mock_metadata: MagicMock,
) -> None:
    """A live PR check cannot silently skip commit-message closure references."""
    for commit_messages in (None, "", " \n"):
        blockers = pr_contract_check.check_closes_discipline(
            "No semantic closing reference in the body.",
            "ll7/robot_sf_ll7",
            commit_messages=commit_messages,
            commit_messages_checked=True,
        )

        assert len(blockers) == 1
        assert "commit messages" in blockers[0]
    mock_metadata.assert_not_called()


def test_check_closes_discipline_allows_non_closing_reference() -> None:
    """``Refs`` keeps GitHub from closing an incident before reconciliation."""
    assert not pr_contract_check.check_closes_discipline("Refs #8414", "ll7/robot_sf_ll7")


def test_github_closing_parity_flags_negated_prose_mention() -> None:
    """Regression for issue #9566: negated prose still auto-closes on GitHub."""
    body = (
        "The result is diagnostic integration evidence only; it does not close #9489 or epic #9483."
    )
    # The negation-aware repo parser excuses the mention ...
    assert pr_contract_check._find_closed_references(body) == []
    assert not pr_contract_check.check_closes_discipline(body, "ll7/robot_sf_ll7")
    # ... but GitHub honors it, so the parity guard must fail closed.
    blockers = pr_contract_check.check_github_closing_parity(body, "ll7/robot_sf_ll7")
    assert len(blockers) == 1
    assert "#9489" in blockers[0]
    assert "leaves #9489 open" in blockers[0]


def test_github_closing_parity_allows_refs_and_explicit_closes() -> None:
    """``Refs`` never closes; intentional ``Closes`` stays with closes-discipline."""
    assert pr_contract_check.check_github_closing_parity("Refs #9489", "ll7/robot_sf_ll7") == []
    assert pr_contract_check.check_github_closing_parity("Closes #9489", "ll7/robot_sf_ll7") == []
    assert (
        pr_contract_check.check_github_closing_parity("leaves #9489 open", "ll7/robot_sf_ll7") == []
    )


@pytest.mark.parametrize(
    "body",
    (
        "This does not affect runtime. Closes #9566",
        "This does not affect runtime, closes #9566",
        "This does not affect runtime; resolves #9566",
        "No changes - Closes #9566",
        "No changes — Closes #9566",
    ),
)
def test_github_closing_parity_allows_explicit_close_after_unrelated_negation(
    body: str,
) -> None:
    """An earlier prose clause cannot negate an intentional closing declaration."""
    assert pr_contract_check._find_closed_references(body) == [(None, "9566")]
    assert pr_contract_check.check_github_closing_parity(body, "ll7/robot_sf_ll7") == []


@pytest.mark.parametrize(
    "body",
    (
        "It does-not-close #9489",
        "It don't-close #9489",
        "It never-close #9489",
    ),
)
def test_github_closing_parity_preserves_hyphenated_negation(body: str) -> None:
    """A hyphen inside a negated phrase is not a clause boundary."""
    blockers = pr_contract_check.check_github_closing_parity(body, "ll7/robot_sf_ll7")

    assert len(blockers) == 1
    assert "#9489" in blockers[0]


@pytest.mark.parametrize(
    ("body", "target"),
    (
        ("it does not close other-org/other-repo#9489", "other-org/other-repo#9489"),
        ("it does not close ll7/robot_sf_ll7#9489", "ll7/robot_sf_ll7#9489"),
        (
            "it does not close https://github.com/other-org/other-repo/issues/9489",
            "other-org/other-repo#9489",
        ),
        (
            "it does not close https://github.com/ll7/robot_sf_ll7/issues/9489",
            "ll7/robot_sf_ll7#9489",
        ),
    ),
)
def test_github_closing_parity_flags_qualified_and_url_mentions(body: str, target: str) -> None:
    """Negated qualified and URL forms remain GitHub-closing parity blockers."""
    blockers = pr_contract_check.check_github_closing_parity(body, "ll7/robot_sf_ll7")

    assert len(blockers) == 1
    assert target in blockers[0]


@pytest.mark.parametrize(
    "body",
    (
        "Closes other-org/other-repo#9489",
        "Closes ll7/robot_sf_ll7#9489",
        "Closes https://github.com/other-org/other-repo/issues/9489",
        "Closes https://github.com/ll7/robot_sf_ll7/issues/9489",
    ),
)
def test_github_closing_parity_allows_explicit_qualified_and_url_closes(body: str) -> None:
    """Explicit local and cross-repository closes remain intentional references."""
    assert pr_contract_check.check_github_closing_parity(body, "ll7/robot_sf_ll7") == []


def test_github_closing_parity_normalizes_local_qualified_target() -> None:
    """An explicit local close excuses an equivalent qualified prose mention."""
    body = "Closes #9489\nThe note does not close ll7/robot_sf_ll7#9489."

    assert pr_contract_check.check_github_closing_parity(body, "ll7/robot_sf_ll7") == []


def test_github_closing_parity_scans_commit_messages() -> None:
    """Squash-merge commit prose receives the same parity protection as the body."""
    blockers = pr_contract_check.check_github_closing_parity(
        "Adds a repair without a body closing keyword.",
        "ll7/robot_sf_ll7",
        commit_messages="Implement repair\n\nit does not close #9489\n",
        commit_messages_checked=True,
    )
    assert len(blockers) == 1
    assert "PR commit message" in blockers[0]


_OVERRIDE_NUMSTAT = "\n".join(f"300\t0\tscripts/dev/file_{index}.py" for index in range(5)) + "\n"
_CAPPED_ISSUE_BODY = "Reviewability budget: Maximum 10 files and 800 net new lines.\n"
_BINARY_NUMSTAT = "10\t2\tscripts/dev/a.py\n-\t-\texamples/fixtures/synthetic.zip\n"


@patch("scripts.ci.pr_contract_check._diff_numstat", return_value=_OVERRIDE_NUMSTAT)
@patch("scripts.ci.pr_contract_check.get_issue_metadata")
def test_check_line_budget_discipline_blocks_over_budget(
    mock_metadata: MagicMock, mock_numstat: MagicMock
) -> None:
    """An over-cap PR without an override is blocked with the issue number."""
    mock_metadata.return_value = (["technical-debt"], _CAPPED_ISSUE_BODY)

    blockers = pr_contract_check.check_line_budget_discipline(
        "Closes #9094\n", "origin/main", "ll7/robot_sf_ll7"
    )

    assert len(blockers) == 1
    assert "#9094" in blockers[0]
    assert "1500 net new lines > 800-line cap" in blockers[0]
    mock_numstat.assert_called_once_with("origin/main")


@patch("scripts.ci.pr_contract_check._diff_numstat", return_value=_OVERRIDE_NUMSTAT)
@patch("scripts.ci.pr_contract_check.get_issue_metadata")
def test_check_line_budget_discipline_honors_reasoned_override(
    mock_metadata: MagicMock, _mock_numstat: MagicMock
) -> None:
    """A reasoned budget-override line clears the breach."""
    mock_metadata.return_value = (["technical-debt"], _CAPPED_ISSUE_BODY)

    blockers = pr_contract_check.check_line_budget_discipline(
        "Closes #9094\nbudget-override: split agreed with review; follow-up filed\n",
        "origin/main",
        "ll7/robot_sf_ll7",
    )

    assert blockers == []


@patch("scripts.ci.pr_contract_check.get_issue_metadata")
def test_check_line_budget_discipline_is_inert_without_cap(
    mock_metadata: MagicMock,
) -> None:
    """Issues without a declared cap never produce blockers."""
    mock_metadata.return_value = (["technical-debt"], "No budget declared here.\n")

    assert (
        pr_contract_check.check_line_budget_discipline(
            "Closes #9094\n", "origin/main", "ll7/robot_sf_ll7"
        )
        == []
    )


@patch("scripts.ci.pr_contract_check.get_issue_metadata")
def test_check_line_budget_discipline_ignores_other_repository(
    mock_metadata: MagicMock,
) -> None:
    """A qualified close for another repository cannot select a local budget issue."""
    assert (
        pr_contract_check.check_line_budget_discipline(
            "Closes other-org/other-repo#9094\n", "origin/main", "ll7/robot_sf_ll7"
        )
        == []
    )
    mock_metadata.assert_not_called()


@patch("scripts.ci.pr_contract_check.get_issue_metadata", return_value=None)
def test_check_line_budget_discipline_skips_unreadable_issue(
    _mock_metadata: MagicMock,
) -> None:
    """An unreadable linked issue is left to the closes-discipline check."""
    assert (
        pr_contract_check.check_line_budget_discipline(
            "Closes #9094\n", "origin/main", "ll7/robot_sf_ll7"
        )
        == []
    )


@patch("scripts.ci.pr_contract_check._diff_numstat", return_value=None)
@patch("scripts.ci.pr_contract_check.get_issue_metadata")
def test_check_line_budget_discipline_fails_closed_when_diff_unavailable(
    mock_metadata: MagicMock, mock_numstat: MagicMock
) -> None:
    """An unresolvable base cannot masquerade as an empty, within-budget diff."""
    mock_metadata.return_value = (["technical-debt"], _CAPPED_ISSUE_BODY)

    blockers = pr_contract_check.check_line_budget_discipline(
        "Closes #9094\n", "missing-base", "ll7/robot_sf_ll7"
    )

    assert len(blockers) == 1
    assert "cannot measure the PR diff" in blockers[0]
    assert "fail-closed" in blockers[0]
    mock_numstat.assert_called_once_with("missing-base")


@patch("scripts.ci.pr_contract_check.get_issue_metadata")
def test_check_line_budget_discipline_accepts_canonical_binary_numstat_row(
    mock_metadata: MagicMock,
) -> None:
    """A Git binary row counts as a file without inventing text-line totals."""
    mock_metadata.return_value = (["technical-debt"], _CAPPED_ISSUE_BODY)

    blockers = pr_contract_check.check_line_budget_discipline(
        "Closes #9094\n",
        "origin/main",
        "ll7/robot_sf_ll7",
        numstat_text=_BINARY_NUMSTAT,
    )

    assert blockers == []
    evidence = pr_contract_check.HistoricalNumstatEvidence.from_numstat(_BINARY_NUMSTAT)
    assert evidence.changed_files == (
        "scripts/dev/a.py",
        "examples/fixtures/synthetic.zip",
    )
    assert evidence.files == 2
    assert evidence.added == 10
    assert evidence.deleted == 2
    assert evidence.net == 8


@patch("scripts.ci.pr_contract_check._diff_numstat")
@patch("scripts.ci.pr_contract_check.get_issue_metadata")
def test_check_line_budget_discipline_uses_supplied_historical_numstat(
    mock_metadata: MagicMock, mock_numstat: MagicMock
) -> None:
    """Historical evidence is selected per call and never reads the candidate HEAD diff."""
    mock_metadata.return_value = (["technical-debt"], _CAPPED_ISSUE_BODY)

    blockers = pr_contract_check.check_line_budget_discipline(
        "Closes #9094\n",
        "origin/main",
        "ll7/robot_sf_ll7",
        numstat_text="10\t0\thistorical.py\n",
    )

    assert blockers == []
    mock_numstat.assert_not_called()


@patch("scripts.ci.pr_contract_check.get_issue_metadata")
def test_run_all_checks_injects_historical_numstat_only_for_budget_check(
    mock_metadata: MagicMock,
) -> None:
    """The regression harness can bind budget measurement to one historical PR."""
    mock_metadata.return_value = (["technical-debt"], _CAPPED_ISSUE_BODY)

    blockers, _, _ = pr_contract_check.run_all_checks(
        "historical",
        "Closes #9094\n",
        [],
        "ll7/robot_sf_ll7",
        "missing-base",
        None,
        historical_numstat="10\t0\thistorical.py\n",
    )

    assert not any("cannot measure the PR diff" in blocker for blocker in blockers)


@patch("scripts.ci.pr_contract_check.get_issue_metadata")
def test_supplied_unavailable_historical_numstat_fails_closed(
    mock_metadata: MagicMock,
) -> None:
    """An unavailable historical payload remains a blocker, not an empty diff."""
    mock_metadata.return_value = (["technical-debt"], _CAPPED_ISSUE_BODY)

    blockers = pr_contract_check.check_line_budget_discipline(
        "Closes #9094\n",
        "origin/main",
        "ll7/robot_sf_ll7",
        numstat_text=None,
    )

    assert len(blockers) == 1
    assert "fail-closed" in blockers[0]
    assert "origin/main" in blockers[0]


@pytest.mark.parametrize(
    "numstat_text",
    [
        "",
        "   \n",
        "malformed payload\n",
        "4\tbad\ta.py\n",
        "4\t0\ta.py\textra\n",
        "4\t0\ta.py\n5\t0\ta.py\n",
        "-\t0\tbinary.zip\n",
        "0\t-\tbinary.zip\n",
    ],
)
@patch("scripts.ci.pr_contract_check.get_issue_metadata")
def test_supplied_historical_numstat_rejects_malformed_or_ambiguous_payload(
    mock_metadata: MagicMock, numstat_text: str
) -> None:
    """Every malformed injected stats payload remains unavailable evidence."""
    mock_metadata.return_value = (["technical-debt"], _CAPPED_ISSUE_BODY)

    blockers = pr_contract_check.check_line_budget_discipline(
        "Closes #9094\n",
        "origin/main",
        "ll7/robot_sf_ll7",
        numstat_text=numstat_text,
    )

    assert len(blockers) == 1
    assert "fail-closed" in blockers[0]


def test_historical_compare_requires_merge_base_and_unique_commits() -> None:
    """Compare evidence must carry the exact base relation and unique commits."""
    base_sha = "a" * 40
    target_sha = "b" * 40
    valid = {
        "base_commit": {"sha": base_sha},
        "merge_base_commit": {"sha": base_sha},
        "total_commits": 1,
        "commits": [{"sha": target_sha}],
    }
    missing_merge_base = {key: value for key, value in valid.items() if key != "merge_base_commit"}
    wrong_merge_base = {**valid, "merge_base_commit": {"sha": "c" * 40}}
    duplicate_commits = {
        **valid,
        "total_commits": 2,
        "commits": [{"sha": target_sha}, {"sha": target_sha}],
    }

    assert _historical_compare_reaches_target(valid, base_sha, target_sha)
    assert _historical_compare_reaches_target(
        wrong_merge_base,
        base_sha,
        target_sha,
    )
    assert not _historical_compare_reaches_target(
        wrong_merge_base,
        base_sha,
        target_sha,
        expected_merge_base_sha=base_sha,
    )
    assert not _historical_compare_reaches_target(missing_merge_base, base_sha, target_sha)
    assert not _historical_compare_reaches_target(duplicate_commits, base_sha, target_sha)


@patch("subprocess.run")
def test_fetch_historical_compare_rejects_duplicate_files(mock_run: MagicMock) -> None:
    """Repeated filenames cannot become repeated numstat rows."""
    base_sha = "a" * 40
    head_sha = "b" * 40
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps(
            {
                "base_commit": {"sha": base_sha},
                "merge_base_commit": {"sha": base_sha},
                "total_commits": 1,
                "commits": [{"sha": head_sha}],
                "files": [
                    {"filename": "a.py", "additions": 4, "deletions": 1},
                    {"filename": "a.py", "additions": 5, "deletions": 0},
                ],
            }
        ),
    )

    assert _fetch_historical_compare("ll7/robot_sf_ll7", base_sha, head_sha) is None


@pytest.mark.parametrize(
    "merged_at",
    [None, {}, "", "not-a-timestamp", "2026-02-30T00:00:00Z"],
)
@patch("subprocess.run")
def test_fetch_historical_pr_identity_rejects_malformed_merged_at(
    mock_run: MagicMock, merged_at: object
) -> None:
    """Merged timestamps must remain a valid GitHub UTC timestamp string."""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps(
            {
                "state": "closed",
                "merged": True,
                "merged_at": merged_at,
                "base": {"sha": "a" * 40},
                "head": {"sha": "b" * 40},
                "merge_commit_sha": "c" * 40,
            }
        ),
    )

    assert _fetch_historical_pr_identity(9122, "ll7/robot_sf_ll7") is None


def test_historical_budget_exception_requires_exact_evidence() -> None:
    """A bounded compatibility record cannot match another PR or changed stats."""
    record = HISTORICAL_BUDGET_EXCEPTION_FIXTURES[9000][8000]
    evidence = HistoricalPREvidence(
        pr_number=9000,
        base_sha="a" * 40,
        head_sha="b" * 40,
        merge_commit_sha="c" * 40,
        merge_parent_shas=("a" * 40,),
        merge_base_sha="a" * 40,
        changed_files=("a.py", "b.py", "c.py"),
        numstat=pr_contract_check.HistoricalNumstatEvidence.from_numstat(
            "275\t0\ta.py\n275\t0\tb.py\n275\t0\tc.py\n"
        ),
    )
    KNOWN_HISTORICAL_BUDGET_EXCEPTIONS[9000] = {80: record, 8000: record}
    try:
        assert _matches_historical_budget_exception(evidence, 8000, record)
        mutated_stats = {**record, "stats": {**record["stats"], "net": 824}}
        assert not _matches_historical_budget_exception(evidence, 8000, mutated_stats)
        wrong_identity = {**record, "identity": {**record["identity"], "head_sha": "d" * 40}}
        assert not _matches_historical_budget_exception(evidence, 8000, wrong_identity)
        assert not _matches_historical_budget_exception(
            HistoricalPREvidence(**{**evidence.__dict__, "pr_number": 9001}),
            8000,
            record,
        )
        assert not _matches_historical_budget_exception(evidence, 8000, {"stats": {}})
        assert _is_expected_historical_budget_blocker(
            evidence,
            "Closes #8000",
            "BLOCKER: PR exceeds the budget declared in issue #8000 "
            "(825 net new lines > 800-line cap).",
        )
        assert not _is_expected_historical_budget_blocker(
            evidence,
            "Closes other-org/other-repo#8000",
            "BLOCKER: PR exceeds the budget declared in issue #8000 "
            "(825 net new lines > 800-line cap).",
        )
        assert _is_expected_historical_budget_blocker(
            evidence,
            "Closes ll7/robot_sf_ll7#8000",
            "BLOCKER: PR exceeds the budget declared in issue #8000 "
            "(825 net new lines > 800-line cap).",
        )
        assert not _is_expected_historical_budget_blocker(
            evidence,
            "Closes #8000",
            "BLOCKER: independent incident issue #8000 requires review.",
        )
        assert not _is_expected_historical_budget_blocker(
            evidence,
            "Closes #80",
            "BLOCKER: PR exceeds the budget declared in issue #8000 "
            "(825 net new lines > 800-line cap).",
        )
    finally:
        KNOWN_HISTORICAL_BUDGET_EXCEPTIONS.clear()


def test_historical_pr_evidence_rejects_malformed_identity_and_file_binding() -> None:
    """Compatibility evidence cannot be constructed with forged identity or files."""
    numstat = pr_contract_check.HistoricalNumstatEvidence.from_numstat("4\t1\ta.py\n")
    valid = {
        "pr_number": 9000,
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "merge_commit_sha": "c" * 40,
        "merge_parent_shas": ("a" * 40,),
        "merge_base_sha": "a" * 40,
        "changed_files": ("a.py",),
        "numstat": numstat,
    }

    with pytest.raises(ValueError, match="full commit IDs"):
        HistoricalPREvidence(**{**valid, "head_sha": "b"})
    with pytest.raises(ValueError, match="unique full commit IDs"):
        HistoricalPREvidence(**{**valid, "merge_parent_shas": ("a" * 40, "a" * 40)})
    with pytest.raises(ValueError, match="files do not match"):
        HistoricalPREvidence(**{**valid, "changed_files": ("forged.py",)})


@pytest.mark.parametrize(
    "filename", ["large.py\n0\t10000\tfake.py", "unsafe\x00.py", "unsafe\u0085.py"]
)
@patch("subprocess.run")
def test_fetch_historical_compare_rejects_control_character_filenames(
    mock_run: MagicMock, filename: str
) -> None:
    """GitHub filenames must not alter the row boundaries used for budget evidence."""
    base_sha = "a" * 40
    head_sha = "b" * 40
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps(
            {
                "base_commit": {"sha": base_sha},
                "merge_base_commit": {"sha": base_sha},
                "total_commits": 1,
                "commits": [{"sha": head_sha}],
                "files": [{"filename": filename, "additions": 1000, "deletions": 0}],
            }
        ),
    )

    assert _fetch_historical_compare("ll7/robot_sf_ll7", base_sha, head_sha) is None


@patch("subprocess.run")
def test_fetch_historical_pr_evidence_renders_authoritative_numstat(
    mock_run: MagicMock,
) -> None:
    """The historical adapter preserves GitHub additions/deletions per filename."""
    mock_run.side_effect = [
        MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "state": "closed",
                    "merged": True,
                    "merged_at": "2026-09-01T00:00:00Z",
                    "base": {"sha": "a" * 40},
                    "head": {"sha": "b" * 40},
                    "merge_commit_sha": "c" * 40,
                }
            ),
        ),
        MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "base_commit": {"sha": "a" * 40},
                    "merge_base_commit": {"sha": "a" * 40},
                    "total_commits": 1,
                    "commits": [{"sha": "b" * 40}],
                    "files": [{"filename": "a.py", "additions": 4, "deletions": 1}],
                }
            ),
        ),
        MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "sha": "c" * 40,
                    "parents": [{"sha": "a" * 40}],
                }
            ),
        ),
        MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "base_commit": {"sha": "a" * 40},
                    "merge_base_commit": {"sha": "a" * 40},
                    "total_commits": 1,
                    "commits": [{"sha": "c" * 40}],
                }
            ),
        ),
    ]

    evidence = _fetch_historical_pr_evidence(9122)
    assert evidence is not None
    assert evidence.changed_files == ("a.py",)
    assert evidence.numstat.numstat == "4\t1\ta.py\n"
    assert evidence.base_sha == "a" * 40
    assert evidence.merge_parent_shas == ("a" * 40,)
    assert evidence.merge_base_sha == "a" * 40
    assert "pulls/9122" in mock_run.call_args_list[0].args[0][2]
    assert "compare/" in mock_run.call_args_list[1].args[0][2]
    assert f"commits/{'c' * 40}" in mock_run.call_args_list[2].args[0][2]
    assert f"compare/{'a' * 40}...{'c' * 40}" in mock_run.call_args_list[3].args[0][2]


@patch("subprocess.run")
def test_fetch_historical_pr_evidence_returns_none_for_malformed_stats(
    mock_run: MagicMock,
) -> None:
    """Malformed or unavailable historical stats cannot be treated as zero changes."""
    mock_run.side_effect = [
        MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "state": "closed",
                    "merged": True,
                    "merged_at": "2026-09-01T00:00:00Z",
                    "base": {"sha": "a" * 40},
                    "head": {"sha": "b" * 40},
                    "merge_commit_sha": "c" * 40,
                }
            ),
        ),
        MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "base_commit": {"sha": "a" * 40},
                    "merge_base_commit": {"sha": "a" * 40},
                    "total_commits": 1,
                    "commits": [{"sha": "b" * 40}],
                    "files": [{"filename": "a.py", "additions": "4", "deletions": 1}],
                }
            ),
        ),
    ]

    assert _fetch_historical_pr_evidence(9122) is None


@pytest.mark.parametrize(
    "metadata",
    [
        {"state": "closed", "merged": True, "merged_at": "2026-09-01T00:00:00Z"},
        {
            "state": "closed",
            "merged": True,
            "merged_at": "2026-09-01T00:00:00Z",
            "base": {"sha": "a" * 40},
            "head": {"sha": "b" * 40},
            "merge_commit_sha": "not-a-sha",
        },
    ],
)
@patch("subprocess.run")
def test_fetch_historical_pr_evidence_rejects_unbound_revision_identity(
    mock_run: MagicMock, metadata: dict[str, object]
) -> None:
    """A merged-PR record without a complete immutable identity is unavailable."""
    mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(metadata))

    assert _fetch_historical_pr_evidence(9122) is None


@patch("subprocess.run")
def test_fetch_historical_pr_evidence_rejects_mutated_compare_identity(
    mock_run: MagicMock,
) -> None:
    """Compare results must echo the requested immutable base revision."""
    mock_run.side_effect = [
        MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "state": "closed",
                    "merged": True,
                    "merged_at": "2026-09-01T00:00:00Z",
                    "base": {"sha": "a" * 40},
                    "head": {"sha": "b" * 40},
                    "merge_commit_sha": "c" * 40,
                }
            ),
        ),
        MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "base_commit": {"sha": "d" * 40},
                    "files": [{"filename": "a.py", "additions": 4, "deletions": 1}],
                }
            ),
        ),
    ]

    assert _fetch_historical_pr_evidence(9122) is None


@pytest.mark.parametrize(
    "merge_payload",
    [
        {
            "sha": "d" * 40,
            "parents": [{"sha": "a" * 40}],
        },
        {
            "sha": "c" * 40,
            "parents": [],
        },
        {
            "sha": "c" * 40,
            "parents": [{"sha": "not-a-sha"}],
        },
        {
            "sha": "c" * 40,
            "parents": [{"sha": "a" * 40}, {"sha": "a" * 40}],
        },
    ],
)
@patch("subprocess.run")
def test_fetch_historical_pr_evidence_rejects_unbound_merge_commit(
    mock_run: MagicMock, merge_payload: dict[str, object]
) -> None:
    """Merge SHA and malformed parent identities remain unavailable evidence."""
    mock_run.side_effect = [
        MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "state": "closed",
                    "merged": True,
                    "merged_at": "2026-09-01T00:00:00Z",
                    "base": {"sha": "a" * 40},
                    "head": {"sha": "b" * 40},
                    "merge_commit_sha": "c" * 40,
                }
            ),
        ),
        MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "base_commit": {"sha": "a" * 40},
                    "merge_base_commit": {"sha": "a" * 40},
                    "total_commits": 1,
                    "commits": [{"sha": "b" * 40}],
                    "files": [{"filename": "a.py", "additions": 4, "deletions": 1}],
                }
            ),
        ),
        MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "sha": "c" * 40,
                    "parents": [{"sha": "a" * 40}],
                }
            ),
        ),
        MagicMock(returncode=0, stdout=json.dumps(merge_payload)),
    ]

    assert _fetch_historical_pr_evidence(9122) is None


@pytest.mark.parametrize(
    "merge_comparison",
    [
        {
            "base_commit": {"sha": "d" * 40},
            "merge_base_commit": {"sha": "a" * 40},
            "total_commits": 1,
            "commits": [{"sha": "c" * 40}],
        },
        {
            "base_commit": {"sha": "a" * 40},
            "merge_base_commit": {"sha": "d" * 40},
            "total_commits": 1,
            "commits": [{"sha": "c" * 40}],
        },
        {
            "base_commit": {"sha": "a" * 40},
            "merge_base_commit": {"sha": "a" * 40},
            "total_commits": 1,
            "commits": [{"sha": "d" * 40}],
        },
    ],
)
@patch("subprocess.run")
def test_fetch_historical_pr_evidence_rejects_mutated_merge_compare_identity(
    mock_run: MagicMock, merge_comparison: dict[str, object]
) -> None:
    """The PR and merge compares must bind the immutable base/head/merge identity."""
    mock_run.side_effect = [
        MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "state": "closed",
                    "merged": True,
                    "merged_at": "2026-09-01T00:00:00Z",
                    "base": {"sha": "a" * 40},
                    "head": {"sha": "b" * 40},
                    "merge_commit_sha": "c" * 40,
                }
            ),
        ),
        MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "base_commit": {"sha": "a" * 40},
                    "merge_base_commit": {"sha": "a" * 40},
                    "total_commits": 1,
                    "commits": [{"sha": "b" * 40}],
                    "files": [{"filename": "a.py", "additions": 4, "deletions": 1}],
                }
            ),
        ),
        MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "sha": "c" * 40,
                    "parents": [{"sha": "a" * 40}],
                }
            ),
        ),
        MagicMock(returncode=0, stdout=json.dumps(merge_comparison)),
    ]

    assert _fetch_historical_pr_evidence(9122) is None


@patch("subprocess.run")
def test_diff_numstat_falls_back_to_two_dot_without_merge_base(
    mock_run: MagicMock,
) -> None:
    """A shallow checkout without a merge base measures via the two-dot tree diff."""
    mock_run.side_effect = [
        MagicMock(returncode=1, stdout=""),
        MagicMock(returncode=0, stdout="10\t2\tscripts/dev/a.py\n"),
    ]

    assert pr_contract_check._diff_numstat("origin/main") == "10\t2\tscripts/dev/a.py\n"
    assert mock_run.call_count == 2
    assert mock_run.call_args_list[0].args[0][-1] == "origin/main...HEAD"
    assert mock_run.call_args_list[1].args[0][-1] == "origin/main..HEAD"


@patch("subprocess.run")
def test_diff_numstat_returns_none_when_both_forms_fail(mock_run: MagicMock) -> None:
    """Unavailable measurement stays None so budget enforcement fails closed."""
    mock_run.side_effect = [
        MagicMock(returncode=1, stdout=""),
        MagicMock(returncode=1, stdout=""),
    ]

    assert pr_contract_check._diff_numstat("origin/main") is None


def test_build_comment_body_marks_main_ci_closing_guard_failure() -> None:
    """The summary row reports incident-closure blockers as failed."""
    blocker = (
        f"BLOCKER: {pr_contract_check.CLOSES_DISCIPLINE_TAG} PR body attempts to close "
        "a canonical main-CI incident."
    )
    comment = pr_contract_check.build_comment_body([blocker], [], [], "🔴 FAILED")
    assert "| 1. Closes-discipline | ❌ FAILED |" in comment


@pytest.mark.parametrize(
    ("blockers", "expected_status"),
    (
        ([], "✅ PASSED"),
        (
            [
                f"BLOCKER: {pr_contract_check.GITHUB_CLOSING_PARITY_TAG} "
                "PR body contains a prose closing mention."
            ],
            "❌ FAILED",
        ),
    ),
)
def test_build_comment_body_renders_github_closing_parity_row(
    blockers: list[str], expected_status: str
) -> None:
    """The summary exposes parity status independently and keeps ten rows numbered."""
    comment = pr_contract_check.build_comment_body(blockers, [], [], "🔴 FAILED")

    assert f"| 2. GitHub closing-keyword parity | {expected_status} |" in comment
    assert "| 3. Closure declaration | ✅ PASSED |" in comment
    assert "| 10. Issue line/file budget | ✅ PASSED |" in comment


def test_check_closure_declaration() -> None:
    """Test check_closure_declaration warns on missing declarations."""
    title = "Issue #123: fix bug"
    body_ok = "Closes #123"
    body_bad = "some description without refs"

    assert not pr_contract_check.check_closure_declaration(title, body_ok)
    warnings = pr_contract_check.check_closure_declaration(title, body_bad)
    assert len(warnings) == 1
    assert "closure declaration" in warnings[0]


def test_check_state_refresh_only() -> None:
    """Test check_state_refresh_only blocks state-only updates."""
    title = "State Update"
    body = "closure-audit refresh"
    changed_state_only = ["docs/context/issue_123_state.yaml"]
    changed_code = ["docs/context/issue_123_state.yaml", "robot_sf/sim/core.py"]

    # Blocked: only state files and matching patterns
    blockers = pr_contract_check.check_state_refresh_only(changed_state_only, title, body)
    assert len(blockers) == 1
    assert "touches ONLY docs/context/**" in blockers[0]

    # OK: touches code as well
    blockers = pr_contract_check.check_state_refresh_only(changed_code, title, body)
    assert not blockers


@patch("scripts.ci.pr_contract_check.is_file_new")
@patch("scripts.ci.pr_contract_check.get_new_files")
def test_check_evidence_tree_hygiene(
    mock_new_files: MagicMock, mock_is_new: MagicMock, tmp_path: Path
) -> None:
    """Test check_evidence_tree_hygiene checks new file markers and README claims."""
    mock_new_files.return_value = set()
    mock_is_new.return_value = True

    # Case 1: New file without marker
    f1 = tmp_path / "docs/context/evidence/test_report.md"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f1.write_text("Some random contents", encoding="utf-8")

    blockers = pr_contract_check.check_evidence_tree_hygiene([str(f1)], "origin/main")
    assert len(blockers) == 1
    assert "marker convention" in blockers[0]

    # Case 2: New file with marker
    f2 = tmp_path / "docs/context/evidence/test_report2.md"
    f2.write_text("<!-- AI-GENERATED — NEEDS-REVIEW -->\nSome contents", encoding="utf-8")
    blockers = pr_contract_check.check_evidence_tree_hygiene([str(f2)], "origin/main")
    assert not blockers

    # Case 3: README claim without provenance
    f3 = tmp_path / "docs/context/evidence/README.md"
    f3.write_text(
        "<!-- AI-GENERATED — NEEDS-REVIEW -->\nThis proves that the model is stable.",
        encoding="utf-8",
    )
    blockers = pr_contract_check.check_evidence_tree_hygiene([str(f3)], "origin/main")
    assert len(blockers) == 1
    assert "provenance fields" in blockers[0]

    # Case 4: README claim with provenance
    f4 = tmp_path / "docs/context/evidence/README2.md"
    f4.write_text(
        "<!-- AI-GENERATED — NEEDS-REVIEW -->\nThis proves stability. seeds: 1, config: ppo, hash: abc",
        encoding="utf-8",
    )
    blockers = pr_contract_check.check_evidence_tree_hygiene([str(f4)], "origin/main")
    assert not blockers


def test_markerless_new_evidence_accepts_valid_same_pr_review_sidecar(tmp_path: Path) -> None:
    """Issue #5752: exact sidecar metadata authorizes immutable marker-less evidence."""
    artifact = tmp_path / "docs/context/evidence/immutable_report.md"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"Historical evidence bytes\n")
    artifact_path = "docs/context/evidence/immutable_report.md"
    sidecar = Path(f"{artifact}.review.json")
    sidecar.write_text(json.dumps(_valid_review_sidecar(artifact, artifact_path)), encoding="utf-8")
    added_files = {str(artifact), str(sidecar)}

    blockers = pr_contract_check.check_evidence_tree_hygiene(
        [str(artifact), str(sidecar)], "origin/main", added_files
    )

    assert not blockers


def test_markerless_json_evidence_is_rejected(tmp_path: Path) -> None:
    """Issue #7812: a marker-less new JSON evidence artifact is a hosted-policy blocker."""
    artifact = tmp_path / "docs/context/evidence/result_interpretation_review.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    added_files = {str(artifact)}

    blockers = pr_contract_check.check_evidence_tree_hygiene(
        [str(artifact)], "origin/main", added_files
    )

    assert any("marker convention" in b for b in blockers)


def test_markerless_json_evidence_passes_with_valid_sidecar(tmp_path: Path) -> None:
    """Issue #7812: an exact-hash review sidecar repairs a marker-less JSON artifact."""
    artifact = tmp_path / "docs/context/evidence/result_interpretation_review.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    artifact_path = "docs/context/evidence/result_interpretation_review.json"
    sidecar = Path(f"{artifact}.review.json")
    sidecar.write_text(json.dumps(_valid_review_sidecar(artifact, artifact_path)), encoding="utf-8")
    added_files = {str(artifact), str(sidecar)}

    blockers = pr_contract_check.check_evidence_tree_hygiene(
        [str(artifact), str(sidecar)], "origin/main", added_files
    )

    assert not blockers


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("malformed_json", "not valid JSON"),
        ("path_traversal", "artifact_path"),
        ("windows_drive", "artifact_path"),
        ("missing_hash", "artifact_sha256"),
        ("mismatched_hash", "does not match"),
        ("uppercase_hash", "lowercase"),
        ("wrong_artifact_path", "artifact_path"),
        ("missing_markers", "marker values"),
        ("wrong_schema", "schema_version"),
        ("unpreserved_bytes", "preserved_exact_bytes"),
    ],
)
def test_markerless_new_evidence_rejects_invalid_review_sidecars(
    tmp_path: Path, case: str, expected_message: str
) -> None:
    """Issue #5752: malformed, unbound, or incomplete sidecars remain blockers."""
    artifact = tmp_path / "docs/context/evidence/immutable_report.md"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"Historical evidence bytes\n")
    artifact_path = "docs/context/evidence/immutable_report.md"
    sidecar = Path(f"{artifact}.review.json")

    if case == "malformed_json":
        sidecar.write_text('{"review_marker": "AI-GENERATED NEEDS-REVIEW"', encoding="utf-8")
    else:
        payload = _valid_review_sidecar(artifact, artifact_path)
        mutation = {
            "path_traversal": ("artifact_path", "../immutable_report.md"),
            "windows_drive": ("artifact_path", "C:/immutable_report.md"),
            "missing_hash": None,
            "mismatched_hash": ("artifact_sha256", "0" * 64),
            "uppercase_hash": ("artifact_sha256", str(payload["artifact_sha256"]).upper()),
            "wrong_artifact_path": (
                "artifact_path",
                "docs/context/evidence/other_report.md",
            ),
            "missing_markers": ("review_marker", "AI-GENERATED"),
            "wrong_schema": ("schema_version", "evidence-review-marker.v2"),
            "unpreserved_bytes": ("preserved_exact_bytes", False),
        }[case]
        if mutation is None:
            payload.pop("artifact_sha256")
        else:
            field, value = mutation
            payload[field] = value
        sidecar.write_text(json.dumps(payload), encoding="utf-8")

    blockers = pr_contract_check.check_evidence_tree_hygiene(
        [str(artifact), str(sidecar)], "origin/main", {str(artifact), str(sidecar)}
    )

    assert any(expected_message in blocker for blocker in blockers)


def test_markerless_new_evidence_rejects_sidecar_not_added_to_same_pr(tmp_path: Path) -> None:
    """Issue #5752: an existing or modified sidecar cannot waive a new artifact."""
    artifact = tmp_path / "docs/context/evidence/immutable_report.md"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"Historical evidence bytes\n")
    artifact_path = "docs/context/evidence/immutable_report.md"
    sidecar = Path(f"{artifact}.review.json")
    sidecar.write_text(json.dumps(_valid_review_sidecar(artifact, artifact_path)), encoding="utf-8")

    blockers = pr_contract_check.check_evidence_tree_hygiene(
        [str(artifact), str(sidecar)], "origin/main", {str(artifact)}
    )

    assert any("same-PR added review sidecar" in blocker for blocker in blockers)


@patch("scripts.ci.pr_contract_check.is_file_new")
@patch("scripts.ci.pr_contract_check.get_new_files")
def test_check_evidence_tree_hygiene_distance_convention_missing(
    mock_new_files: MagicMock, mock_is_new: MagicMock, tmp_path: Path
) -> None:
    """Issue #5141: a new distance-like series without distance_convention is blocked."""
    mock_new_files.return_value = set()
    mock_is_new.return_value = True

    # New distance-series CSV with a marker but NO convention declaration.
    f = tmp_path / "docs/context/evidence/min_distance_series.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(
        "# AI-GENERATED NEEDS-REVIEW\nstep,min_robot_ped_distance_m\n0,1.37\n",
        encoding="utf-8",
    )
    blockers = pr_contract_check.check_evidence_tree_hygiene([str(f)], "origin/main")
    distance_blockers = [b for b in blockers if "distance_convention" in b]
    assert len(distance_blockers) == 1
    assert "distance-like series" in distance_blockers[0]


@patch("scripts.ci.pr_contract_check.is_file_new")
@patch("scripts.ci.pr_contract_check.get_new_files")
def test_check_evidence_tree_hygiene_distance_convention_present_in_file(
    mock_new_files: MagicMock, mock_is_new: MagicMock, tmp_path: Path
) -> None:
    """Issue #5141: an in-file `# distance_convention:` header satisfies the lint."""
    mock_new_files.return_value = set()
    mock_is_new.return_value = True

    f = tmp_path / "docs/context/evidence/min_distance_series.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(
        "# AI-GENERATED NEEDS-REVIEW\n"
        "# distance_convention: center_center\n"
        "step,min_robot_ped_distance_m\n0,1.37\n",
        encoding="utf-8",
    )
    blockers = pr_contract_check.check_evidence_tree_hygiene([str(f)], "origin/main")
    assert not blockers


@patch("scripts.ci.pr_contract_check.is_file_new")
@patch("scripts.ci.pr_contract_check.get_new_files")
def test_check_evidence_tree_hygiene_distance_convention_present_in_sibling_metadata(
    mock_new_files: MagicMock, mock_is_new: MagicMock, tmp_path: Path
) -> None:
    """Issue #5141: a sibling metadata.json carrying the field satisfies the lint."""
    mock_new_files.return_value = set()
    mock_is_new.return_value = True

    bundle = tmp_path / "docs/context/evidence/bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    # Distance CSV has no in-file declaration...
    csv_path = bundle / "min_distance_series.csv"
    csv_path.write_text(
        "# AI-GENERATED NEEDS-REVIEW\nstep,min_robot_ped_distance_m\n0,1.37\n",
        encoding="utf-8",
    )
    # ...but the sibling metadata.json declares it.
    (bundle / "metadata.json").write_text(
        '{"distance_convention": "center_center"}\n', encoding="utf-8"
    )
    blockers = pr_contract_check.check_evidence_tree_hygiene([str(csv_path)], "origin/main")
    assert not blockers


@patch("scripts.ci.pr_contract_check.is_file_new")
@patch("scripts.ci.pr_contract_check.get_new_files")
def test_check_evidence_tree_hygiene_distance_convention_not_retroactive(
    mock_new_files: MagicMock, mock_is_new: MagicMock, tmp_path: Path
) -> None:
    """Issue #5141: the lint only applies to NEW evidence files."""
    mock_new_files.return_value = set()
    mock_is_new.return_value = False  # pre-existing file

    f = tmp_path / "docs/context/evidence/old_min_distance_series.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(
        "# AI-GENERATED NEEDS-REVIEW\nstep,min_robot_ped_distance_m\n0,1.37\n",
        encoding="utf-8",
    )
    blockers = pr_contract_check.check_evidence_tree_hygiene([str(f)], "origin/main")
    assert not any("distance_convention" in b for b in blockers)


@patch("scripts.ci.pr_contract_check.is_file_new")
@patch("scripts.ci.pr_contract_check.get_new_files")
def test_check_evidence_tree_hygiene_non_distance_series_unaffected(
    mock_new_files: MagicMock, mock_is_new: MagicMock, tmp_path: Path
) -> None:
    """Issue #5141: files that are not distance-like are not flagged."""
    mock_new_files.return_value = set()
    mock_is_new.return_value = True

    f = tmp_path / "docs/context/evidence/README.md"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(
        "<!-- AI-GENERATED NEEDS-REVIEW -->\nSummary text without distance data.\n",
        encoding="utf-8",
    )
    blockers = pr_contract_check.check_evidence_tree_hygiene([str(f)], "origin/main")
    assert not any("distance_convention" in b for b in blockers)


@patch("subprocess.run")
def test_base_ref_is_resolvable(mock_run: MagicMock) -> None:
    """Issue #5464: base_ref_is_resolvable reflects git rev-parse success/failure."""
    mock_run.return_value = MagicMock(returncode=0)
    assert pr_contract_check.base_ref_is_resolvable("origin/main") is True

    mock_run.return_value = MagicMock(returncode=128)
    assert pr_contract_check.base_ref_is_resolvable("origin/main") is False


@patch("scripts.ci.pr_contract_check.base_ref_is_resolvable", return_value=False)
def test_is_file_new_unresolvable_base_returns_false(
    _mock_resolvable: MagicMock, tmp_path: Path
) -> None:
    """Issue #5464: an existing file is NOT reported new when the base ref is unresolvable.

    This is the exact false-positive path: on a shallow CI checkout ``origin/main`` is
    absent, and the old code returned True for every on-disk file. It must return False.
    """
    f = tmp_path / "some_evidence.json"
    f.write_text("{}", encoding="utf-8")
    assert pr_contract_check.is_file_new(str(f), "origin/main") is False


def test_get_added_files(tmp_path: Path) -> None:
    """Issue #5464: get_added_files parses the added-files list, else returns None."""
    assert pr_contract_check.get_added_files(None) is None
    assert pr_contract_check.get_added_files(tmp_path / "missing.txt") is None

    added = tmp_path / "pr_added_files.txt"
    added.write_text(
        "docs/context/evidence/new_a.json\n\ndocs/context/evidence/new_b.svg\n",
        encoding="utf-8",
    )
    assert pr_contract_check.get_added_files(added) == {
        "docs/context/evidence/new_a.json",
        "docs/context/evidence/new_b.svg",
    }


@patch("scripts.ci.pr_contract_check.subprocess.run")
@patch("scripts.ci.pr_contract_check.base_ref_is_resolvable", return_value=True)
def test_get_changed_files_prefers_current_base_diff_over_stale_api_list(
    _mock_resolvable: MagicMock, mock_run: MagicMock, tmp_path: Path
) -> None:
    """Issue #7668: stale PR API files cannot override a resolvable current-base diff."""
    api_files = tmp_path / "pr_changed_files.txt"
    api_files.write_text("stale-base-only.py\n", encoding="utf-8")
    mock_run.return_value = MagicMock(returncode=0, stdout="current-base.py\n", stderr="")

    assert pr_contract_check.get_changed_files(api_files, "origin/main") == ["current-base.py"]
    mock_run.assert_called_once_with(
        ["git", "diff", "--name-only", "origin/main...HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )


@patch("scripts.ci.pr_contract_check.base_ref_is_resolvable", return_value=False)
def test_get_changed_files_uses_api_fallback_when_base_is_unavailable(
    _mock_resolvable: MagicMock, tmp_path: Path
) -> None:
    """Issue #7668: shallow/local runs retain the API changed-file fallback."""
    api_files = tmp_path / "pr_changed_files.txt"
    api_files.write_text("api-fallback.py\n\n", encoding="utf-8")

    assert pr_contract_check.get_changed_files(api_files, "origin/main") == ["api-fallback.py"]


def test_check_evidence_tree_hygiene_authoritative_added_files(tmp_path: Path) -> None:
    """Issue #5464: with an authoritative added set, only added files get marker blockers.

    A marker-less evidence file that is *modified* (not in the added set) must not be
    flagged, while a marker-less *added* file still is. No git heuristic is consulted.
    """
    evidence_dir = tmp_path / "docs/context/evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    modified = evidence_dir / "packet.json"
    modified.write_text('{"note": "predates marker convention"}', encoding="utf-8")
    added = evidence_dir / "brand_new.json"
    added.write_text('{"note": "no marker"}', encoding="utf-8")

    # Only ``brand_new.json`` is authoritatively added.
    added_set = {str(added).replace("\\", "/")}
    blockers = pr_contract_check.check_evidence_tree_hygiene(
        [str(modified), str(added)], "origin/main", added_set
    )
    marker_blockers = [b for b in blockers if "marker convention" in b]
    assert len(marker_blockers) == 1
    assert str(added) in marker_blockers[0]
    assert str(modified) not in marker_blockers[0]

    # Empty added set (PR that only modifies evidence) → no marker blockers at all.
    assert not pr_contract_check.check_evidence_tree_hygiene([str(modified)], "origin/main", set())


@patch("subprocess.run")
def test_check_successor_discipline(mock_run: MagicMock) -> None:
    """Test check_successor_discipline warns on lack of successor statement."""
    # Issue in title has merged PRs, but body lacks successor statement
    title = "Issue #123: title"
    body_no_stmt = "some description"
    body_ok = "This is a successor slice; does not duplicate PR #12"

    # Merge exists and canonically references the issue
    mock_run.return_value = MagicMock(
        returncode=0, stdout='[{"number": 12, "title": "Fix", "body": "Closes #123."}]'
    )

    warnings = pr_contract_check.check_successor_discipline(title, body_no_stmt, "ll7/robot_sf_ll7")
    assert len(warnings) == 1
    assert "successor statement" in warnings[0]

    warnings = pr_contract_check.check_successor_discipline(title, body_ok, "ll7/robot_sf_ll7")
    assert not warnings


@patch("subprocess.run")
def test_check_successor_discipline_requires_canonical_issue_reference(
    mock_run: MagicMock,
) -> None:
    """A broad numeric search hit without a canonical reference is not a successor."""
    title = "Issue #8818: title"

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout='[{"number": 3364, "title": "planner policy-builder refactor", "body": "unrelated"}]',
    )

    warnings = pr_contract_check.check_successor_discipline(
        title, "some description", "ll7/robot_sf_ll7"
    )

    assert not warnings


@pytest.mark.parametrize(
    ("candidate_text", "expected"),
    [
        ("Closes #123.", True),
        ("ll7/robot_sf_ll7#123", True),
        ("https://github.com/ll7/robot_sf_ll7/issues/123", True),
        ("issue 123 was discussed", False),
        ("Fixes #123.5 rounding", False),
        ("hash fragment #123abc", False),
        ("other/repo#123", False),
        ("cross-reference 18818", False),
    ],
)
@patch("subprocess.run")
def test_successor_discipline_reference_forms(
    mock_run: MagicMock, candidate_text: str, expected: bool
) -> None:
    """Only canonical references to the specific repository issue count."""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps([{"number": 42, "title": "candidate", "body": candidate_text}]),
    )

    warnings = pr_contract_check.check_successor_discipline(
        "Issue #123: title", "some description", "ll7/robot_sf_ll7"
    )

    assert bool(warnings) is expected
    if expected:
        assert "referenced in 1 merged PR(s)" in warnings[0]


@patch("subprocess.run")
def test_successor_discipline_counts_only_confirmed_references(mock_run: MagicMock) -> None:
    """The warning count reflects only canonically confirmed merged PRs."""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps(
            [
                {"number": 1, "title": "Fix", "body": "Closes #123."},
                {"number": 2, "title": "Other", "body": "plain 123 mention"},
                {
                    "number": 3,
                    "title": "Again",
                    "body": "https://github.com/ll7/robot_sf_ll7/issues/123",
                },
            ]
        ),
    )

    warnings = pr_contract_check.check_successor_discipline(
        "Issue #123: title", "some description", "ll7/robot_sf_ll7"
    )

    assert len(warnings) == 1
    assert "referenced in 2 merged PR(s)" in warnings[0]


@patch("scripts.ci.pr_contract_check.get_pr_label_guard_shas")
@patch("scripts.ci.pr_contract_check.add_label")
def test_check_worker_lane_provenance(mock_add_label: MagicMock, mock_get_shas: MagicMock) -> None:
    """Test check_worker_lane_provenance detects cheap lane and labels PR."""
    body_lane = "This PR was produced by the agy/Gemini-3.5-Flash cheap implementation lane"
    body_normal = "Some normal PR"

    # Lane provenance with PR number
    mock_add_label.return_value = {
        "status": "ok",
        "number": 123,
        "label": "cheap-lane",
        "action": "add",
    }
    mock_get_shas.return_value = ("a" * 40, "b" * 40)
    info, labeled = pr_contract_check.check_worker_lane_provenance(
        body_lane, "123", "ll7/robot_sf_ll7"
    )
    assert labeled is True
    assert "Automatically added" in info
    mock_get_shas.assert_called_once_with("123", "ll7/robot_sf_ll7")
    mock_add_label.assert_called_once_with(
        123,
        "cheap-lane",
        repo="ll7/robot_sf_ll7",
        target="pr",
        expected_head_sha="a" * 40,
        expected_base_sha="b" * 40,
    )

    info, labeled = pr_contract_check.check_worker_lane_provenance(
        body_normal, "123", "ll7/robot_sf_ll7"
    )
    assert labeled is False


@patch("scripts.ci.pr_contract_check.get_pr_label_guard_shas", return_value=None)
@patch("scripts.ci.pr_contract_check.add_label")
def test_check_worker_lane_provenance_skips_label_when_pr_shas_unavailable(
    mock_add_label: MagicMock, mock_get_shas: MagicMock
) -> None:
    """A PR label write is withheld when the exact live head/base pair is unavailable."""
    info, labeled = pr_contract_check.check_worker_lane_provenance(
        "cheap implementation lane", "123", "ll7/robot_sf_ll7"
    )

    assert labeled is True
    assert "exact PR head/base SHAs" in info
    mock_get_shas.assert_called_once_with("123", "ll7/robot_sf_ll7")
    mock_add_label.assert_not_called()


_RECENT_MERGED_PR_LIMIT = 20


def _validate_recent_merged_pr_inventory(
    payload: object,
) -> list[dict[str, object]]:
    """Validate the exact, typed inventory required by the live regression sweep."""
    if not isinstance(payload, list) or len(payload) != _RECENT_MERGED_PR_LIMIT:
        raise ValueError(
            f"recent merged PR inventory must contain exactly {_RECENT_MERGED_PR_LIMIT} rows"
        )

    normalized: list[dict[str, object]] = []
    seen_numbers: set[int] = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict) or set(item) != {"number", "title", "body"}:
            raise ValueError(f"recent merged PR inventory row {index} has the wrong shape")
        number = item["number"]
        title = item["title"]
        body = item["body"]
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or number <= 0
            or number in seen_numbers
            or not isinstance(title, str)
            or not title.strip()
            or (body is not None and not isinstance(body, str))
        ):
            raise ValueError(f"recent merged PR inventory row {index} has invalid field types")
        seen_numbers.add(number)
        normalized.append({"number": number, "title": title, "body": body})
    return normalized


def _fetch_recent_merged_pr_inventory(repo: str) -> list[dict[str, object]]:
    """Fetch and validate the complete recent merged-PR inventory, failing closed."""
    try:
        response = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                repo,
                "--state",
                "merged",
                "--limit",
                str(_RECENT_MERGED_PR_LIMIT),
                "--json",
                "number,title,body",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return _validate_recent_merged_pr_inventory(json.loads(response.stdout))
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("recent merged PR inventory query timed out") from error
    except (subprocess.SubprocessError, OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"recent merged PR inventory is unavailable: {error}") from error


def _valid_recent_merged_pr_inventory() -> list[dict[str, object]]:
    """Build a GitHub-compatible exact-size merged-PR inventory fixture."""
    return [
        {"number": number, "title": f"PR {number}", "body": None if number == 1 else ""}
        for number in range(1, 21)
    ]


@pytest.mark.parametrize(
    "payload_factory",
    [
        lambda: None,
        lambda: {},
        lambda: [],
        lambda: _valid_recent_merged_pr_inventory()[:19],
        lambda: _valid_recent_merged_pr_inventory()[:-1] + [_valid_recent_merged_pr_inventory()[0]],
        lambda: _valid_recent_merged_pr_inventory()[:-1] + [{}],
        lambda: (
            _valid_recent_merged_pr_inventory()[:-1]
            + [{"number": "20", "title": "PR 20", "body": ""}]
        ),
        lambda: (
            _valid_recent_merged_pr_inventory()[:-1] + [{"number": 20, "title": None, "body": ""}]
        ),
        lambda: (
            _valid_recent_merged_pr_inventory()[:-1]
            + [{"number": 20, "title": "PR 20", "body": 20}]
        ),
    ],
)
def test_recent_merged_pr_inventory_rejects_malformed_or_incomplete_payload(
    payload_factory: object,
) -> None:
    """Inventory validation rejects every malformed, empty, or incomplete response."""
    payload = payload_factory()  # type: ignore[operator]

    with pytest.raises(ValueError):
        _validate_recent_merged_pr_inventory(payload)


@patch("subprocess.run")
def test_fetch_recent_merged_pr_inventory_uses_explicit_repo_and_timeout(
    mock_run: MagicMock,
) -> None:
    """The live inventory query is repository-bound and time-bounded."""
    payload = _valid_recent_merged_pr_inventory()
    mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(payload))

    assert _fetch_recent_merged_pr_inventory("ll7/robot_sf_ll7") == payload
    mock_run.assert_called_once_with(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            "ll7/robot_sf_ll7",
            "--state",
            "merged",
            "--limit",
            "20",
            "--json",
            "number,title,body",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )


@patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["gh", "pr", "list"], 15))
def test_fetch_recent_merged_pr_inventory_fails_closed_on_timeout(
    _mock_run: MagicMock,
) -> None:
    """A hung inventory query is an error, never a passing or skipped sweep."""
    with pytest.raises(RuntimeError, match="timed out"):
        _fetch_recent_merged_pr_inventory("ll7/robot_sf_ll7")


def test_regression_last_20_merged_prs() -> None:
    """Run regression test on the last 20 merged PRs to ensure zero false blockers."""
    try:
        prs = _fetch_recent_merged_pr_inventory("ll7/robot_sf_ll7")
    except RuntimeError as error:
        pytest.fail(f"Cannot prove the recent merged PR regression sweep: {error}")

    for pr in prs:
        title = pr["title"]
        body = pr["body"] or ""
        number = pr["number"]
        assert isinstance(title, str)
        assert isinstance(body, str)
        assert isinstance(number, int)
        historical_evidence = _fetch_historical_pr_evidence(number)
        if historical_evidence is None:
            pytest.fail(
                "Cannot prove the live PR regression sweep: immutable diff evidence unavailable "
                f"for PR #{number}"
            )
        changed_files = list(historical_evidence.changed_files)

        # Pass pr_number=None: this regression test only asserts on blockers, and
        # supplying a real PR number would make Rule 6 (worker-lane provenance) run a
        # live `gh pr edit --add-label cheap-lane` against real merged PRs as a test
        # side-effect. None exercises the same blocker paths without mutating GitHub.
        blockers, _, _ = pr_contract_check.run_all_checks(
            title,
            body,
            changed_files,
            "ll7/robot_sf_ll7",
            "origin/main",
            None,
            historical_numstat=historical_evidence.numstat,
        )
        metadata_unavailable = next(
            (blocker for blocker in blockers if "Could not verify issue" in blocker), None
        )
        if metadata_unavailable is not None:
            pytest.fail(f"Cannot prove the live PR regression sweep: {metadata_unavailable}")
        expected_incident_issues = KNOWN_HISTORICAL_MAIN_CI_CLOSING_GUARD_HITS.get(number, set())
        for issue in expected_incident_issues:
            assert any(f"incident issue #{issue}" in blocker for blocker in blockers), (
                f"PR #{number} no longer exposes its known historical guard hit"
            )
        expected_parity_issues = KNOWN_HISTORICAL_GITHUB_PARITY_HITS.get(number, set())
        for issue in expected_parity_issues:
            assert any(
                "github-closing-parity" in blocker and f"#{issue}" in blocker
                for blocker in blockers
            ), f"PR #{number} no longer exposes its known historical parity hit"
        unexpected_blockers = [
            blocker
            for blocker in blockers
            if not any(f"incident issue #{issue}" in blocker for issue in expected_incident_issues)
            and not any(
                "github-closing-parity" in blocker and f"#{issue}" in blocker
                for issue in expected_parity_issues
            )
            and not _is_expected_historical_budget_blocker(historical_evidence, body, blocker)
        ]
        assert not unexpected_blockers, (
            f"PR #{number} ('{title}') triggered unexpected blockers: {unexpected_blockers}"
        )


class TestPlaceholderDocstringRatchet:
    """Issue #5856: reject NEW placeholder docstrings added in the PR diff.

    These tests exercise the git-diff-backed ratchet in an isolated throwaway git
    repository so they do not depend on the surrounding robot_sf_ll7 history.
    """

    def _init_repo(self, tmp_path: Path) -> Path:
        """Create and initialize a throwaway git repo rooted at ``tmp_path``."""
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        configure_git_identity(repo, name="CI", email="ci@example.com")
        # A base commit with a pre-existing (grandfathered) stub.
        legacy = repo / "tool.py"
        legacy.write_text(
            'def legacy():\n    """TODO docstring. Document this function."""\n    return 1\n',
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
        return repo

    def _commit_change(self, repo: Path, filename: str, content: str) -> None:
        """Write a tracked file at ``repo`` and commit it as a new HEAD."""
        (repo / filename).write_text(content, encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "change"], cwd=repo, check=True)

    def test_adds_placeholder_docstring_fails(self, tmp_path: Path) -> None:
        """A PR that adds a placeholder stub docstring is blocked with file:line."""
        repo = self._init_repo(tmp_path)
        self._commit_change(
            repo,
            "new.py",
            'def do_thing():\n    """TODO docstring. Document this function."""\n    return 0\n',
        )
        blockers = pr_contract_check.check_placeholder_docstrings("HEAD~1", repo_root=str(repo))
        blocker = next((b for b in blockers if "new.py" in b), None)
        assert blocker is not None, f"expected blocker for new.py, got: {blockers}"
        assert "new.py:2" in blocker

    def test_adds_empty_docstring_fails(self, tmp_path: Path) -> None:
        """A PR that ADDS a trivially-empty '\"\"\".\"\"\"' line is blocked."""
        repo = self._init_repo(tmp_path)
        self._commit_change(
            repo,
            "new.py",
            'def do_thing():\n    """."""\n    return 0\n',
        )
        blockers = pr_contract_check.check_placeholder_docstrings("HEAD~1", repo_root=str(repo))
        assert any("new.py:2" in b for b in blockers)

    def test_adds_single_quoted_empty_docstring_fails(self, tmp_path: Path) -> None:
        """A PR that ADDS a trivially-empty triple-single-quoted line is blocked."""
        repo = self._init_repo(tmp_path)
        self._commit_change(
            repo,
            "new.py",
            "def do_thing():\n    '''.'''\n    return 0\n",
        )
        blockers = pr_contract_check.check_placeholder_docstrings("HEAD~1", repo_root=str(repo))
        assert any("new.py:2" in b for b in blockers)

    def test_noprefix_config_cannot_bypass_ratchet(self, tmp_path: Path) -> None:
        """Explicit diff prefixes keep the ratchet active with ``diff.noprefix``."""
        repo = self._init_repo(tmp_path)
        subprocess.run(["git", "config", "diff.noprefix", "true"], cwd=repo, check=True)
        self._commit_change(
            repo,
            "new.py",
            'def do_thing():\n    """TODO docstring."""\n    return 0\n',
        )
        blockers = pr_contract_check.check_placeholder_docstrings("HEAD~1", repo_root=str(repo))
        assert any("new.py:2" in b for b in blockers)

    def test_placeholder_text_inside_fixture_string_passes(self, tmp_path: Path) -> None:
        """Placeholder examples inside non-docstring source strings are allowed."""
        repo = self._init_repo(tmp_path)
        self._commit_change(
            repo,
            "test_example.py",
            'SOURCE = \'def f():\\n    """TODO docstring."""\\n\'\n',
        )
        blockers = pr_contract_check.check_placeholder_docstrings("HEAD~1", repo_root=str(repo))
        assert not blockers

    def test_touching_legacy_stub_passes(self, tmp_path: Path) -> None:
        """Touching a file that already has a stub (without adding new ones) passes."""
        repo = self._init_repo(tmp_path)
        # Only modify a non-docstring line; the old stub remains but is not ADDED.
        legacy = repo / "tool.py"
        legacy.write_text(
            'def legacy():\n    """TODO docstring. Document this function."""\n    return 2\n',
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "touch"], cwd=repo, check=True)
        blockers = pr_contract_check.check_placeholder_docstrings("HEAD~1", repo_root=str(repo))
        assert not blockers

    def test_real_docstring_passes(self, tmp_path: Path) -> None:
        """Adding a genuine one-line docstring is accepted."""
        repo = self._init_repo(tmp_path)
        self._commit_change(
            repo,
            "new.py",
            'def do_thing():\n    """Compute the thing and return a result."""\n    return 0\n',
        )
        assert not pr_contract_check.check_placeholder_docstrings("HEAD~1", repo_root=str(repo))

    def test_diff_added_line_parser(self, tmp_path: Path) -> None:
        """_diff_added_python_lines maps added line numbers per file."""
        repo = self._init_repo(tmp_path)
        self._commit_change(
            repo,
            "new.py",
            'def a():\n    """real."""\n    return 0\n\ndef b():\n    """TODO docstring."""\n    return 1\n',
        )
        added = pr_contract_check._diff_added_python_lines("HEAD~1", repo_root=str(repo))
        # The fixture file has a blank line between the two functions, so 7 lines
        # are added; the parser must enumerate every added line number.
        assert added.get("new.py") == [1, 2, 3, 4, 5, 6, 7]


class TestWorkflowFetchFallback:
    """Validate the PR contract-check workflow tolerates fetch failure.

    See issue #5558: the git fetch step must fall back gracefully instead of
    hard-stopping the entire contract check job.
    """

    def test_workflow_contains_fetch_fallback(self) -> None:
        """The workflow must include a fallback for the git fetch step."""
        workflow_path = ROOT / ".github" / "workflows" / "pr-contract-check.yml"
        content = workflow_path.read_text(encoding="utf-8")

        # The fetch step must include a fallback pattern: `git fetch ... || echo`
        # that prevents the job from stopping on fetch failure.
        fallback_pattern = re.compile(r"git fetch.*\|\|.*echo.*::warning::", re.DOTALL)
        match = fallback_pattern.search(content)
        assert match is not None, (
            "The 'Fetch base ref' step in pr-contract-check.yml must tolerate "
            "fetch failure with a fallback pattern (git fetch ... || echo). "
            "Without this, a network error or deleted base branch hard-stops the "
            "entire contract check job. See issue #5558."
        )
