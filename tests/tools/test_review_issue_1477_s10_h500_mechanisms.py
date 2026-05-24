"""Tests for issue #1477 S10/h500 mechanism-review selection."""

from __future__ import annotations

import hashlib
import json
import sys
from typing import TYPE_CHECKING

from scripts.tools import review_issue_1477_s10_h500_mechanisms

if TYPE_CHECKING:
    from pathlib import Path


def _episode(
    *,
    scenario_id: str,
    seed: int,
    success: bool,
    near_misses: int,
    termination_reason: str,
) -> dict[str, object]:
    """Build a minimal episode-summary fixture."""
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "episode_id": f"{scenario_id}--{seed}",
        "status": "success" if success else "failure",
        "termination_reason": termination_reason,
        "metrics": {
            "success": success,
            "collisions": 0,
            "near_misses": near_misses,
            "clearing_distance_min": 1.25,
            "time_to_goal_norm": 0.75 if success else 1.0,
        },
    }


def test_review_selects_candidate_cells_and_marks_summary_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The review helper should select candidate cells and fail closed on mechanisms."""
    raw_dir = tmp_path / "raw"
    run_dir = raw_dir / "runs" / "hybrid_rule_v3_fast_progress__differential_drive"
    run_dir.mkdir(parents=True)
    (run_dir / "episodes.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _episode(
                    scenario_id="francis2023_robot_crowding",
                    seed=111,
                    success=True,
                    near_misses=12,
                    termination_reason="success",
                ),
                _episode(
                    scenario_id="francis2023_robot_crowding",
                    seed=112,
                    success=True,
                    near_misses=40,
                    termination_reason="success",
                ),
                _episode(
                    scenario_id="francis2023_narrow_doorway",
                    seed=111,
                    success=False,
                    near_misses=0,
                    termination_reason="terminated",
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    core_dir = raw_dir / "runs" / "orca__differential_drive"
    core_dir.mkdir(parents=True)
    (core_dir / "episodes.jsonl").write_text(
        json.dumps(
            _episode(
                scenario_id="francis2023_robot_crowding",
                seed=111,
                success=False,
                near_misses=99,
                termination_reason="collision",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    archive = tmp_path / "archive.tar.zst"
    archive.write_bytes(b"fixture archive")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_issue_1477_s10_h500_mechanisms.py",
            "--raw-campaign-dir",
            str(raw_dir),
            "--archive",
            str(archive),
            "--expected-archive-sha256",
            digest,
            "--output-dir",
            str(output_dir),
            "--per-scenario",
            "1",
        ],
    )

    assert review_issue_1477_s10_h500_mechanisms.main() == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "summary_only_no_trace_or_video"
    assert summary["mechanism_claim_result"] == "unsupported_without_step_trace_or_video"
    assert summary["selected_cell_count"] == 2

    rows = (output_dir / "reviewed_cells.csv").read_text(encoding="utf-8")
    assert "francis2023_robot_crowding" in rows
    assert "francis2023_narrow_doorway" in rows
    assert "orca" not in rows
    assert "not_resolved_summary_only_no_trace_or_video" in rows


def test_review_skips_incomplete_episode_records(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Incomplete rows should not crash selection or leak None-like strings."""
    raw_dir = tmp_path / "raw"
    run_dir = raw_dir / "runs" / "hybrid_rule_v3_fast_progress__differential_drive"
    run_dir.mkdir(parents=True)
    (run_dir / "episodes.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"scenario_id": "francis2023_robot_crowding", "metrics": {"near_misses": 99}},
                {"seed": 112, "episode_id": "missing-scenario"},
                _episode(
                    scenario_id="francis2023_robot_crowding",
                    seed=113,
                    success=True,
                    near_misses=40,
                    termination_reason="success",
                )
                | {"status": None, "termination_reason": None},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    archive = tmp_path / "archive.tar.zst"
    archive.write_bytes(b"fixture archive")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_issue_1477_s10_h500_mechanisms.py",
            "--raw-campaign-dir",
            str(raw_dir),
            "--archive",
            str(archive),
            "--expected-archive-sha256",
            digest,
            "--output-dir",
            str(output_dir),
            "--per-scenario",
            "2",
        ],
    )

    assert review_issue_1477_s10_h500_mechanisms.main() == 0

    rows = (output_dir / "reviewed_cells.csv").read_text(encoding="utf-8")
    assert "francis2023_robot_crowding" in rows
    assert "None" not in rows
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["selected_cell_count"] == 1


def test_review_fails_closed_when_raw_campaign_dir_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Missing raw campaign input should fail before writing misleading empty evidence."""
    archive = tmp_path / "archive.tar.zst"
    archive.write_bytes(b"fixture archive")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_issue_1477_s10_h500_mechanisms.py",
            "--raw-campaign-dir",
            str(tmp_path / "missing"),
            "--archive",
            str(archive),
            "--expected-archive-sha256",
            digest,
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    try:
        review_issue_1477_s10_h500_mechanisms.main()
    except SystemExit as exc:
        assert "raw campaign directory not found" in str(exc)
    else:
        raise AssertionError("missing raw campaign directory did not fail closed")
