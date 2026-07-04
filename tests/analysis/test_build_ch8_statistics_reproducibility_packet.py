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
    assert packet["overall_status"] == "blocked"
    assert {row["status"] for row in packet["statistics"]} == {"blocked_missing_source_data"}
