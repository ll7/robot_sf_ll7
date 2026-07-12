"""Tests for the resume-plan preflight module (issue #5392).

Verifies that the resume plan:
- counts existing episodes correctly
- emits correct verdicts (skip-complete, continue-from-N, fresh)
- fails closed on campaign-id or config-hash mismatch
- writes a valid resume_plan.json with correct totals
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from robot_sf.benchmark.camera_ready._resume_plan import (
    ResumeMismatchError,
    _build_verdict_str,
    _count_jsonl_episodes,
    _expected_jobs,
    build_resume_plan,
    emit_resume_plan_log,
    resume_plan_summary,
    verify_resume_context,
    write_resume_plan,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_manifest(
    campaign_root: Path,
    *,
    campaign_id: str = "test-campaign",
    config_hash: str = "abc123",
) -> None:
    """Write a minimal campaign manifest for context checks."""
    campaign_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "campaign_id": campaign_id,
        "config_hash": config_hash,
    }
    (campaign_root / "campaign_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_episodes_jsonl(path: Path, count: int) -> None:
    """Write a JSONL file with *count* fake episode records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(count):
            record: dict[str, Any] = {
                "success": True,
                "episode": i,
                "seed": i,
                "scenario_params": {"algo": "test"},
            }
            f.write(json.dumps(record) + "\n")


def _write_arm(
    runs_dir: Path,
    arm_name: str,
    episodes: int,
    *,
    write_summary: bool = True,
) -> None:
    """Create an arm directory with *episodes* JSONL records."""
    arm_dir = runs_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = arm_dir / "episodes.jsonl"
    if episodes > 0:
        _write_episodes_jsonl(episodes_path, episodes)
    elif episodes_path.exists():
        pass
    if write_summary:
        summary: dict[str, Any] = {"written": episodes, "status": "ok"}
        (arm_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


# --- _count_jsonl_episodes ---


class TestCountJsonlEpisodes:
    """Tests for _count_jsonl_episodes."""

    def test_counts_valid_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "episodes.jsonl"
        _write_episodes_jsonl(p, 7)
        assert _count_jsonl_episodes(p) == 7

    def test_returns_zero_for_missing(self, tmp_path: Path) -> None:
        p = tmp_path / "nope.jsonl"
        assert _count_jsonl_episodes(p) == 0

    def test_returns_zero_for_corrupt(self, tmp_path: Path) -> None:
        p = tmp_path / "corrupt.jsonl"
        p.write_text("not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="line 1"):
            _count_jsonl_episodes(p)


# --- _expected_jobs ---


class TestExpectedJobs:
    """Tests for _expected_jobs."""

    def test_no_repeats_defaults_to_one(self) -> None:
        scenarios = [{"name": "s1"}, {"name": "s2"}]
        assert _expected_jobs(scenarios) == 2

    def test_respects_repeats_field(self) -> None:
        scenarios = [{"name": "s1", "repeats": 3}, {"name": "s2", "repeats": 2}]
        assert _expected_jobs(scenarios) == 5

    def test_empty_scenarios(self) -> None:
        assert _expected_jobs([]) == 0

    def test_rejects_negative_repeats(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _expected_jobs([{"name": "s1", "repeats": -1}])

    def test_rejects_non_integer_repeats(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _expected_jobs([{"name": "s1", "repeats": "3"}])


# --- _build_verdict_str ---


class TestBuildVerdictStr:
    """Tests for _build_verdict_str."""

    def test_skip_complete(self) -> None:
        assert _build_verdict_str(10, 10) == "skip-complete"

    def test_skip_when_more_than_expected(self) -> None:
        assert _build_verdict_str(15, 10) == "skip-complete"

    def test_continue_from(self) -> None:
        assert _build_verdict_str(3, 10) == "continue-from-3"

    def test_fresh_when_zero_expected(self) -> None:
        assert _build_verdict_str(0, 0) == "fresh"

    def test_fresh_when_partial(self) -> None:
        assert _build_verdict_str(0, 5) == "continue-from-0"


# --- verify_resume_context ---


class TestVerifyResumeContext:
    """Tests for verify_resume_context."""

    def test_no_error_when_match(self, tmp_path: Path) -> None:
        root = tmp_path / "campaign"
        _write_manifest(root, campaign_id="my-run", config_hash="h1")
        # Should not raise
        verify_resume_context(root, campaign_id="my-run", config_hash="h1")

    def test_raises_when_no_manifest(self, tmp_path: Path) -> None:
        root = tmp_path / "campaign"
        root.mkdir()
        with pytest.raises(FileNotFoundError, match="Campaign manifest missing"):
            verify_resume_context(root, campaign_id="any", config_hash="any")

    def test_raises_when_manifest_is_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "campaign"
        root.mkdir()
        (root / "campaign_manifest.json").mkdir()
        with pytest.raises(FileNotFoundError, match="Campaign manifest missing"):
            verify_resume_context(root, campaign_id="any", config_hash="any")

    def test_raises_when_manifest_fields_are_missing(self, tmp_path: Path) -> None:
        root = tmp_path / "campaign"
        root.mkdir()
        (root / "campaign_manifest.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="campaign_id"):
            verify_resume_context(root, campaign_id="any", config_hash="any")

    def test_raises_on_campaign_id_mismatch(self, tmp_path: Path) -> None:
        root = tmp_path / "campaign"
        _write_manifest(root, campaign_id="old-run", config_hash="h1")
        with pytest.raises(ResumeMismatchError, match="campaign-id mismatch"):
            verify_resume_context(root, campaign_id="new-run", config_hash="h1")

    def test_raises_on_config_hash_mismatch(self, tmp_path: Path) -> None:
        root = tmp_path / "campaign"
        _write_manifest(root, campaign_id="my-run", config_hash="old-hash")
        with pytest.raises(ResumeMismatchError, match="config-hash mismatch"):
            verify_resume_context(root, campaign_id="my-run", config_hash="new-hash")

    def test_raises_on_both_mismatches(self, tmp_path: Path) -> None:
        root = tmp_path / "campaign"
        _write_manifest(root, campaign_id="old", config_hash="old")
        with pytest.raises(ResumeMismatchError, match="campaign-id mismatch"):
            verify_resume_context(root, campaign_id="new", config_hash="new")


# --- build_resume_plan ---


class TestBuildResumePlan:
    """Tests for build_resume_plan."""

    def test_fresh_arm_when_no_directory(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        planners = [{"key": "sf", "enabled": True}]
        scenarios = [{"name": "s1", "repeats": 3}]
        verdicts = build_resume_plan(
            runs_dir,
            planners=planners,
            kinematics_matrix=["differential_drive"],
            scenarios=scenarios,
        )
        assert len(verdicts) == 1
        v = verdicts[0]
        assert v.verdict == "continue-from-0"
        assert v.episodes_found == 0
        assert v.expected_total == 3
        assert v.episodes_remaining == 3

    def test_skip_complete_arm(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        arm_name = "sf__differential_drive"
        _write_arm(runs_dir, arm_name, episodes=3)
        planners = [{"key": "sf", "enabled": True}]
        scenarios = [{"name": "s1", "repeats": 3}]
        verdicts = build_resume_plan(
            runs_dir,
            planners=planners,
            kinematics_matrix=["differential_drive"],
            scenarios=scenarios,
        )
        v = verdicts[0]
        assert v.verdict == "skip-complete"
        assert v.episodes_found == 3
        assert v.episodes_remaining == 0

    def test_continue_from_partial(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        arm_name = "sf__differential_drive"
        _write_arm(runs_dir, arm_name, episodes=2)
        planners = [{"key": "sf", "enabled": True}]
        scenarios = [{"name": "s1", "repeats": 5}]
        verdicts = build_resume_plan(
            runs_dir,
            planners=planners,
            kinematics_matrix=["differential_drive"],
            scenarios=scenarios,
        )
        v = verdicts[0]
        assert v.verdict == "continue-from-2"
        assert v.episodes_found == 2
        assert v.episodes_remaining == 3

    def test_skips_disabled_planners(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        planners = [
            {"key": "sf", "enabled": True},
            {"key": "rl", "enabled": False},
        ]
        scenarios = [{"name": "s1"}]
        verdicts = build_resume_plan(
            runs_dir,
            planners=planners,
            kinematics_matrix=["differential_drive"],
            scenarios=scenarios,
        )
        keys = [v.planner_key for v in verdicts]
        assert keys == ["sf"]

    def test_multiple_kinematics(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        planners = [{"key": "sf", "enabled": True}]
        scenarios = [{"name": "s1"}]

        # Pre-populate one arm with 1 episode, leave other empty
        _write_arm(runs_dir, "sf__differential_drive", episodes=1)
        # holonomic_arm has no directory

        verdicts = build_resume_plan(
            runs_dir,
            planners=planners,
            kinematics_matrix=["differential_drive", "holonomic"],
            scenarios=scenarios,
        )
        assert len(verdicts) == 2
        dd = next(v for v in verdicts if v.kinematics == "differential_drive")
        ho = next(v for v in verdicts if v.kinematics == "holonomic")
        assert dd.verdict == "skip-complete"
        assert ho.verdict == "continue-from-0"

    def test_uses_precomputed_expected_jobs(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        planners = [{"key": "sf", "enabled": True}]
        scenarios = [{"name": "s1", "repeats": 10}]

        # Override expected_jobs to 5
        verdicts = build_resume_plan(
            runs_dir,
            planners=planners,
            kinematics_matrix=["differential_drive"],
            scenarios=scenarios,
            expected_jobs=5,
        )
        v = verdicts[0]
        assert v.expected_total == 5

    def test_rejects_file_at_arm_directory_path(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        (runs_dir / "sf__differential_drive").write_text("not a directory", encoding="utf-8")
        with pytest.raises(NotADirectoryError, match="not a directory"):
            build_resume_plan(
                runs_dir,
                planners=[{"key": "sf", "enabled": True}],
                kinematics_matrix=["differential_drive"],
                scenarios=[{"name": "s1"}],
            )


# --- resume_plan_summary ---


class TestResumePlanSummary:
    """Tests for resume_plan_summary."""

    def test_totals(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        _write_arm(runs_dir, "sf__differential_drive", episodes=10)  # complete
        _write_arm(runs_dir, "rl__differential_drive", episodes=3)  # partial
        # ho__differential_drive not created -> continue-from-0

        planners = [
            {"key": "sf", "enabled": True},
            {"key": "rl", "enabled": True},
            {"key": "ho", "enabled": True},
        ]
        scenarios = [{"name": "s1", "repeats": 10}]
        verdicts = build_resume_plan(
            runs_dir,
            planners=planners,
            kinematics_matrix=["differential_drive"],
            scenarios=scenarios,
        )
        summary = resume_plan_summary(verdicts)
        assert summary["total_arms"] == 3
        assert summary["arms_skip_complete"] == 1
        # rl (3/10 -> continue-from-3) and ho (0/10 -> continue-from-0) are both continues
        assert summary["arms_continue"] == 2
        assert summary["arms_fresh"] == 0
        assert summary["episodes_banked"] == 13
        assert summary["arms"][0]["verdict"] == "skip-complete"


# --- write_resume_plan ---


class TestWriteResumePlan:
    """Tests for write_resume_plan."""

    def test_writes_plan_json(self, tmp_path: Path) -> None:
        campaign_root = tmp_path / "campaign"
        campaign_root.mkdir()
        runs_dir = campaign_root / "runs"
        runs_dir.mkdir()

        planners = [{"key": "sf", "enabled": True}]
        scenarios = [{"name": "s1", "repeats": 5}]
        _write_arm(runs_dir, "sf__differential_drive", episodes=3)

        verdicts = build_resume_plan(
            runs_dir,
            planners=planners,
            kinematics_matrix=["differential_drive"],
            scenarios=scenarios,
        )
        plan_path = write_resume_plan(
            campaign_root,
            config_hash="h1",
            campaign_id="test",
            verdicts=verdicts,
        )
        assert plan_path.exists()
        data = json.loads(plan_path.read_text(encoding="utf-8"))
        assert data["schema_version"] == "benchmark-resume-plan.v1"
        assert data["campaign_id"] == "test"
        assert data["config_hash"] == "h1"
        assert data["context_check"]["config_hash_match"] is True
        assert data["context_check"]["campaign_id_match"] is True
        assert data["episodes_banked"] == 3
        assert data["total_arms"] == 1
        assert data["generated_at_utc"] is not None

    def test_summary_totals_in_fixture(self, tmp_path: Path) -> None:
        """Simulate the job-13376 fixture: 5 complete arms + 1 partial with 562 episodes."""
        campaign_root = tmp_path / "campaign"
        campaign_root.mkdir()
        runs_dir = campaign_root / "runs"
        runs_dir.mkdir()

        # 5 complete arms (expect 1000 episodes each)
        for i in range(5):
            arm_name = f"arm{i}__differential_drive"
            _write_arm(runs_dir, arm_name, episodes=1000)

        # 1 partial arm
        _write_arm(runs_dir, "arm5__differential_drive", episodes=562)

        planners = [{"key": f"arm{i}", "enabled": True} for i in range(6)]
        scenarios = [{"name": "s1", "repeats": 1000}]
        verdicts = build_resume_plan(
            runs_dir,
            planners=planners,
            kinematics_matrix=["differential_drive"],
            scenarios=scenarios,
        )

        summary = resume_plan_summary(verdicts)
        assert summary["arms_skip_complete"] == 5
        assert summary["arms_continue"] == 1
        assert summary["arms_fresh"] == 0
        assert summary["episodes_banked"] == 5562

        # Partial arm should show continue-from-562
        partial = next(v for v in verdicts if v.verdict.startswith("continue-from-"))
        assert partial.episodes_found == 562
        assert partial.episodes_remaining == 438


# --- emit_resume_plan_log (smoke test) ---


class TestEmitResumePlanLog:
    """Smoke test for emit_resume_plan_log."""

    def test_does_not_raise(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        planners = [{"key": "sf", "enabled": True}]
        scenarios = [{"name": "s1"}]
        verdicts = build_resume_plan(
            runs_dir,
            planners=planners,
            kinematics_matrix=["differential_drive"],
            scenarios=scenarios,
        )
        # Should not raise
        emit_resume_plan_log(verdicts)


# --- Integration: full verify + build + write flow ---


class TestFullResumePlanFlow:
    """Integration tests for full verify + build + write flow."""

    def test_verify_and_emit_end_to_end(self, tmp_path: Path) -> None:
        campaign_root = tmp_path / "campaign"
        campaign_root.mkdir()
        _write_manifest(campaign_root, campaign_id="resume-test", config_hash="chk-001")
        runs_dir = campaign_root / "runs"
        runs_dir.mkdir()

        _write_arm(runs_dir, "sf__differential_drive", episodes=5)
        _write_arm(runs_dir, "rl__differential_drive", episodes=2)

        planners = [
            {"key": "sf", "enabled": True},
            {"key": "rl", "enabled": True},
        ]
        scenarios = [{"name": "s1", "repeats": 5}]

        # Verify should pass
        verify_resume_context(campaign_root, campaign_id="resume-test", config_hash="chk-001")

        verdicts = build_resume_plan(
            runs_dir,
            planners=planners,
            kinematics_matrix=["differential_drive"],
            scenarios=scenarios,
        )
        plan_path = write_resume_plan(
            campaign_root,
            config_hash="chk-001",
            campaign_id="resume-test",
            verdicts=verdicts,
        )
        assert plan_path.exists()
        data = json.loads(plan_path.read_text(encoding="utf-8"))
        assert data["episodes_banked"] == 7
        assert data["arms_skip_complete"] == 1
        assert data["arms_continue"] == 1

    def test_context_mismatch_fails(self, tmp_path: Path) -> None:
        campaign_root = tmp_path / "campaign"
        _write_manifest(campaign_root, campaign_id="other", config_hash="old")
        with pytest.raises(ResumeMismatchError, match="campaign-id mismatch"):
            verify_resume_context(campaign_root, campaign_id="this", config_hash="current")
