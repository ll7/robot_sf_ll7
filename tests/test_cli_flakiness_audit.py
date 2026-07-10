"""CLI tests for the ``flakiness-audit`` benchmark subcommand (issue #4978)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path


def _write_episodes(path: Path) -> None:
    """Write a small episode JSONL with one stable and one knife-edge cell."""
    records = [
        # s1/ppo: unanimous success -> stable
        {"scenario_id": "s1", "algo": "ppo", "seed": 0, "metrics": {"success": 1}},
        {"scenario_id": "s1", "algo": "ppo", "seed": 1, "metrics": {"success": 1}},
        # s2/ppo: 50/50 split across seeds -> knife-edge
        {"scenario_id": "s2", "algo": "ppo", "seed": 0, "metrics": {"success": 1}},
        {"scenario_id": "s2", "algo": "ppo", "seed": 1, "metrics": {"success": 0}},
    ]
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_cli_flakiness_audit_writes_report(tmp_path: Path, capsys):
    """The subcommand loads JSONL, scores cells, and writes a JSON report."""
    episodes = tmp_path / "episodes.jsonl"
    _write_episodes(episodes)
    out_json = tmp_path / "flakiness.json"

    rc = cli_main(
        [
            "flakiness-audit",
            "--in",
            str(episodes),
            "--out",
            str(out_json),
            "--group-by",
            "algo",
        ],
    )
    cap = capsys.readouterr()
    assert rc == 0, f"flakiness-audit failed: {cap.err}"

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["schema_version"] == "scenario_flakiness.v1"
    assert report["summary"]["n_cells"] == 2
    assert report["summary"]["n_knife_edge_cells"] == 1
    # No exact-repeat data present -> determinism unknown, fail closed.
    assert report["exact_repeat"]["is_deterministic"] is None

    by_key = {c["cell_key"]: c for c in report["cells"]}
    assert by_key["s1::ppo"]["knife_edge"] is False
    assert by_key["s2::ppo"]["knife_edge"] is True


def test_cli_flakiness_audit_missing_input_fails_closed(tmp_path: Path):
    """A missing input path fails closed with a non-zero exit code."""
    out_json = tmp_path / "flakiness.json"
    rc = cli_main(
        [
            "flakiness-audit",
            "--in",
            str(tmp_path / "does_not_exist.jsonl"),
            "--out",
            str(out_json),
        ],
    )
    assert rc == 2
    assert not out_json.exists()


def test_cli_flakiness_audit_rejects_nonpositive_min_seeds(tmp_path: Path):
    """The CLI rejects a semantic configuration that removes the evidence floor."""
    episodes = tmp_path / "episodes.jsonl"
    _write_episodes(episodes)
    out_json = tmp_path / "flakiness.json"

    rc = cli_main(
        [
            "flakiness-audit",
            "--in",
            str(episodes),
            "--out",
            str(out_json),
            "--min-seeds",
            "0",
        ],
    )

    assert rc == 2
    assert not out_json.exists()


def _write_mixed_track_episodes(path: Path) -> None:
    """Write episodes that mix two ``benchmark_track`` observation contracts on one cell."""
    records = [
        # lidar_2d_v1 track on s1/ppo: unanimous success
        {
            "scenario_id": "s1",
            "algo": "ppo",
            "seed": 0,
            "benchmark_track": "lidar_2d_v1",
            "metrics": {"success": 1},
        },
        {
            "scenario_id": "s1",
            "algo": "ppo",
            "seed": 1,
            "benchmark_track": "lidar_2d_v1",
            "metrics": {"success": 1},
        },
        # grid_socnav_v1 track on s1/ppo: unanimous failure
        {
            "scenario_id": "s1",
            "algo": "ppo",
            "seed": 2,
            "benchmark_track": "grid_socnav_v1",
            "metrics": {"success": 0},
        },
        {
            "scenario_id": "s1",
            "algo": "ppo",
            "seed": 3,
            "benchmark_track": "grid_socnav_v1",
            "metrics": {"success": 0},
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_cli_flakiness_audit_strict_fails_closed_for_mixed_tracks(tmp_path: Path, capsys):
    """Issue #5072: mixed observation tracks fail closed by default (exit code 2)."""
    episodes = tmp_path / "mixed.jsonl"
    _write_mixed_track_episodes(episodes)
    out_json = tmp_path / "flakiness.json"

    rc = cli_main(
        [
            "flakiness-audit",
            "--in",
            str(episodes),
            "--out",
            str(out_json),
            "--group-by",
            "algo",
        ],
    )
    assert rc == 2
    assert not out_json.exists()


def test_cli_flakiness_audit_diagnostic_cross_track_partitions_per_track(tmp_path: Path, capsys):
    """Issue #5072: opt-in diagnostic mode separates mixed tracks into per-track cells."""
    episodes = tmp_path / "mixed.jsonl"
    _write_mixed_track_episodes(episodes)
    out_json = tmp_path / "flakiness.json"

    rc = cli_main(
        [
            "flakiness-audit",
            "--in",
            str(episodes),
            "--out",
            str(out_json),
            "--group-by",
            "algo",
            "--observation-track-mode",
            "diagnostic-cross-track",
        ],
    )
    cap = capsys.readouterr()
    assert rc == 0, f"flakiness-audit failed: {cap.err}"

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["observation_track_mode"] == "diagnostic_cross_track"
    assert report["observation_tracks"]["mixed_tracks"] is True
    keys = {c["cell_key"] for c in report["cells"]}
    assert keys == {"lidar_2d_v1::s1::ppo", "grid_socnav_v1::s1::ppo"}
    # The two tracks are never pooled into one false knife-edge cell.
    assert all(c["knife_edge"] is False for c in report["cells"])


def test_cli_flakiness_audit_rejects_unknown_track_mode(tmp_path: Path):
    """Issue #5072: an invalid track mode is rejected before any report is written.

    argparse rejects unknown ``choices`` at parse time with a non-zero exit, so no
    report file is produced. We assert the process fails closed rather than emit.
    """
    episodes = tmp_path / "mixed.jsonl"
    _write_mixed_track_episodes(episodes)
    out_json = tmp_path / "flakiness.json"

    with pytest.raises(SystemExit):
        cli_main(
            [
                "flakiness-audit",
                "--in",
                str(episodes),
                "--out",
                str(out_json),
                "--group-by",
                "algo",
                "--observation-track-mode",
                "bogus",
            ],
        )
    assert not out_json.exists()


def test_cli_flakiness_audit_help_documents_track_policy(capsys):
    """Issue #5072: the ``flakiness-audit`` help makes the observation-track policy reproducible."""
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["flakiness-audit", "--help"])
    # ``--help`` exits 0 after printing usage; never a failure.
    assert excinfo.value.code in (0, None)
    help_text = capsys.readouterr().out
    assert "--observation-track-mode" in help_text
    assert "diagnostic-cross-track" in help_text
    assert "benchmark_track" in help_text
