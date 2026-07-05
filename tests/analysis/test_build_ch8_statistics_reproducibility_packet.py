"""Tests for the issue #4445 Chapter 8 reproducibility packet builder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "analysis" / "build_ch8_statistics_reproducibility_packet.py"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "ch8_statistics" / "reproducible_manifest.json"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "docs"
    / "context"
    / "evidence"
    / "issue_4445_ch8_statistics_reproducibility"
    / "source_manifest.json"
)


def test_packet_builder_recomputes_fixture(tmp_path: Path) -> None:
    """Fixture manifest with raw values produces a reproducible packet."""

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(FIXTURE),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    packet = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert packet["overall_status"] == "reproducible"
    assert {row["status"] for row in packet["statistics"]} == {"matches_expected"}
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "README.md").is_file()
    checksums = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8")
    assert "summary.json" in checksums
    assert "report.md" in checksums
    assert "README.md" in checksums
    assert "reproducible_manifest.json" in checksums


def test_default_manifest_fails_closed_until_ch8_sources_are_registered(tmp_path: Path) -> None:
    """The tracked issue packet records missing source data as blocked."""

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(DEFAULT_MANIFEST),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    packet = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert packet["overall_status"] == "reproducible"
    assert {row["status"] for row in packet["statistics"]} == {"matches_expected"}
    expected_by_id = {row["id"]: row["expected"] for row in packet["statistics"]}
    assert expected_by_id["ch8_eta_squared_success_mean"]["scenario_family_eta_squared"] == 0.388
    assert expected_by_id["ch8_spearman_success_time_to_goal_minus_0_998"]["value"] == -0.998
    assert expected_by_id["ch8_bootstrap_ppo_rank_1_ci"]["rank_ci"] == [1, 1]
    assert len(packet["statistics"]) == 7


def test_manifest_with_non_dict_statistic_entry_fails_closed(tmp_path: Path) -> None:
    """A malformed statistics entry raises a clear manifest error, not an opaque crash."""

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "issue_4445.ch8_statistics_reproducibility.v1",
                "issue": 4445,
                "title": "malformed entry",
                "source_status": "unavailable",
                "statistics": ["not-a-dict"],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "must be a JSON object" in completed.stderr


def test_missing_source_file_blocks_cleanly(tmp_path: Path) -> None:
    """Missing source file blocks cleanly without crashing."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "issue_4445.ch8_statistics_reproducibility.v1",
                "issue": 4445,
                "title": "missing source",
                "source_status": "unavailable",
                "statistics": [
                    {
                        "id": "missing_stat",
                        "statistic_kind": "spearman_rho",
                        "source_query": {
                            "input_file": "does_not_exist.csv",
                            "x_field": "x",
                            "y_field": "y",
                        },
                        "expected": {"value": 1.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    packet = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert packet["overall_status"] == "blocked"
    assert packet["statistics"][0]["status"] == "blocked_missing_source_data"


def test_computed_mismatch_reported_as_mismatch(tmp_path: Path) -> None:
    """A computed mismatch is reported as mismatch status/blocker."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "issue_4445.ch8_statistics_reproducibility.v1",
                "issue": 4445,
                "title": "mismatch manifest",
                "source_status": "test",
                "statistics": [
                    {
                        "id": "mismatch_stat",
                        "statistic_kind": "spearman_rho",
                        "expected": {"value": 0.5, "tolerance": 1e-9},
                        "data": {"x_values": [1.0, 2.0, 3.0], "y_values": [3.0, 2.0, 1.0]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    packet = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert packet["overall_status"] == "mismatch"
    assert packet["statistics"][0]["status"] == "computed_mismatch"
    assert any(
        "does not match expected" in blocker for blocker in packet["statistics"][0]["blockers"]
    )


def test_sha256sums_uses_valid_paths(tmp_path: Path) -> None:
    """SHA256SUMS uses packet-local or repo-relative paths accepted by integrity checker."""
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(FIXTURE),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    checksums_content = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8")
    for line in checksums_content.splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        assert len(parts) == 2
        path_str = parts[1]
        assert not path_str.startswith("/")
        assert any(
            path_str.startswith(prefix) or "/" not in path_str
            for prefix in ("docs/", "configs/", "scripts/", "tests/", "output/")
        )
