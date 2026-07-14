"""Unit tests for scripts/ci/pr_contract_check.py."""

# evidence-writer-exempt: these tests intentionally write temporary evidence-path fixtures,
# including malformed files, to exercise the PR contract and writer-guard behavior.

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.ci import pr_contract_check

ROOT = Path(__file__).resolve().parents[2]


def test_find_closed_issues() -> None:
    """Test find_closed_issues matches closing keywords."""
    body = (
        "This fixes #123 and closes #456. Resolves https://github.com/ll7/robot_sf_ll7/issues/789."
    )
    closed = pr_contract_check.find_closed_issues(body)
    assert closed == ["123", "456", "789"]


def test_find_title_issues() -> None:
    """Test find_title_issues parses issue numbers from title."""
    assert pr_contract_check.find_title_issues("Issue #4735: some title") == ["4735"]
    assert pr_contract_check.find_title_issues("Refs #123, issue 456") == ["123", "456"]


def test_has_declaration_for_issue() -> None:
    """Test has_declaration_for_issue checks body matches."""
    body = "We reference Refs #123 here."
    assert pr_contract_check.has_declaration_for_issue("123", body) is True
    assert pr_contract_check.has_declaration_for_issue("456", body) is False


@patch("subprocess.run")
def test_check_closes_discipline(mock_run: MagicMock) -> None:
    """Test check_closes_discipline blocks closing epic issues."""
    # Test case 1: Issue has no epic label
    mock_run.return_value = MagicMock(returncode=0, stdout='{"labels": [{"name": "bug"}]}')
    blockers = pr_contract_check.check_closes_discipline("Closes #123", "ll7/robot_sf_ll7")
    assert not blockers

    # Test case 2: Issue has epic label
    mock_run.return_value = MagicMock(returncode=0, stdout='{"labels": [{"name": "epic"}]}')
    blockers = pr_contract_check.check_closes_discipline("Closes #123", "ll7/robot_sf_ll7")
    assert len(blockers) == 1
    assert "epic" in blockers[0]


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


@patch("subprocess.run")
def test_check_worker_lane_provenance(mock_run: MagicMock) -> None:
    """Test check_worker_lane_provenance detects cheap lane and labels PR."""
    body_lane = "This PR was produced by the agy/Gemini-3.5-Flash cheap implementation lane"
    body_normal = "Some normal PR"

    # Lane provenance with PR number
    mock_run.return_value = MagicMock(returncode=0)
    info, labeled = pr_contract_check.check_worker_lane_provenance(
        body_lane, "123", "ll7/robot_sf_ll7"
    )
    assert labeled is True
    assert "Automatically added" in info

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
        assert not blockers, f"PR #{number} ('{title}') triggered false blockers: {blockers}"


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
