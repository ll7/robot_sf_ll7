"""Unit tests for scripts/ci/pr_contract_check.py."""

# evidence-writer-exempt: these tests intentionally write temporary evidence-path fixtures,
# including malformed files, to exercise the PR contract and writer-guard behavior.

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.ci import pr_contract_check
from tests.support.environment_guards import configure_git_identity

ROOT = Path(__file__).resolve().parents[2]

# PR #8440 is the known pre-guard regression: its merge reference closed the
# incident in #8414 before the two-green reconciler criterion was established.
KNOWN_HISTORICAL_MAIN_CI_CLOSING_GUARD_HITS = {8440: {"8414"}}


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


@patch("scripts.ci.pr_contract_check.subprocess.run")
def test_get_pr_commit_messages_uses_paginated_commit_api(mock_run: MagicMock) -> None:
    """The commit source is fetched through the paginated PR commits endpoint."""
    mock_run.return_value = MagicMock(returncode=0, stdout="first\nsecond\n")

    assert pr_contract_check.get_pr_commit_messages("8451", "ll7/robot_sf_ll7") == "first\nsecond\n"
    mock_run.assert_called_once_with(
        [
            "gh",
            "api",
            "--paginate",
            "repos/ll7/robot_sf_ll7/pulls/8451/commits?per_page=100",
            "--jq",
            ".[] | .commit.message",
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


def test_build_comment_body_marks_main_ci_closing_guard_failure() -> None:
    """The summary row reports incident-closure blockers as failed."""
    blocker = (
        f"BLOCKER: {pr_contract_check.CLOSES_DISCIPLINE_TAG} PR body attempts to close "
        "a canonical main-CI incident."
    )
    comment = pr_contract_check.build_comment_body([blocker], [], [], "🔴 FAILED")
    assert "| 1. Closes-discipline | ❌ FAILED |" in comment


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

    # Merge exists
    mock_run.return_value = MagicMock(returncode=0, stdout='[{"number": 12}]')

    warnings = pr_contract_check.check_successor_discipline(title, body_no_stmt, "ll7/robot_sf_ll7")
    assert len(warnings) == 1
    assert "successor statement" in warnings[0]

    warnings = pr_contract_check.check_successor_discipline(title, body_ok, "ll7/robot_sf_ll7")
    assert not warnings


@patch("scripts.ci.pr_contract_check.add_label")
def test_check_worker_lane_provenance(mock_add_label: MagicMock) -> None:
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
    info, labeled = pr_contract_check.check_worker_lane_provenance(
        body_lane, "123", "ll7/robot_sf_ll7"
    )
    assert labeled is True
    assert "Automatically added" in info
    mock_add_label.assert_called_once_with(123, "cheap-lane", repo="ll7/robot_sf_ll7")

    info, labeled = pr_contract_check.check_worker_lane_provenance(
        body_normal, "123", "ll7/robot_sf_ll7"
    )
    assert labeled is False


def test_regression_last_20_merged_prs() -> None:
    """Run regression test on the last 20 merged PRs to ensure zero false blockers."""
    try:
        res = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--state",
                "merged",
                "--limit",
                "20",
                "--json",
                "number,title,body",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        prs = json.loads(res.stdout)
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        # Narrow (not broad) except so the repo's except->skip policy is satisfied:
        # OSError = gh not installed, SubprocessError = non-zero exit (check=True),
        # ValueError = json.JSONDecodeError from unparseable stdout.
        pytest.skip(f"Skipping regression test because gh CLI query failed: {e}")
        return

    for pr in prs:
        title = pr.get("title", "")
        body = pr.get("body", "") or ""
        number = pr.get("number")

        try:
            res_files = subprocess.run(
                [
                    "gh",
                    "api",
                    f"repos/ll7/robot_sf_ll7/pulls/{number}/files?per_page=100",
                    "--jq",
                    ".[].filename",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            changed_files = [line.strip() for line in res_files.stdout.splitlines() if line.strip()]
        except Exception:
            changed_files = []

        # Pass pr_number=None: this regression test only asserts on blockers, and
        # supplying a real PR number would make Rule 6 (worker-lane provenance) run a
        # live `gh pr edit --add-label cheap-lane` against real merged PRs as a test
        # side-effect. None exercises the same blocker paths without mutating GitHub.
        blockers, _, _ = pr_contract_check.run_all_checks(
            title, body, changed_files, "ll7/robot_sf_ll7", "origin/main", None
        )
        metadata_unavailable = next(
            (blocker for blocker in blockers if "Could not verify issue" in blocker), None
        )
        if metadata_unavailable is not None:
            # The production rule intentionally fails closed. A local regression sweep must not
            # turn a temporary GitHub/API rate limit into a false code failure.
            pytest.skip(f"Skipping live PR regression sweep: {metadata_unavailable}")
        expected_incident_issues = KNOWN_HISTORICAL_MAIN_CI_CLOSING_GUARD_HITS.get(number, set())
        for issue in expected_incident_issues:
            assert any(f"incident issue #{issue}" in blocker for blocker in blockers), (
                f"PR #{number} no longer exposes its known historical guard hit"
            )
        unexpected_blockers = [
            blocker
            for blocker in blockers
            if not any(f"incident issue #{issue}" in blocker for issue in expected_incident_issues)
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
