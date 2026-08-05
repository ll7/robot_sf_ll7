"""Deterministic Robot SF ecosystem handoff fixture tests."""

from __future__ import annotations

import shutil
from pathlib import Path

from scripts.tools import build_ecosystem_handoff_fixture as builder
from scripts.tools import validate_ecosystem_handoff_fixture as validator

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / builder.DEFAULT_OUTPUT


def _tree_bytes(root: Path) -> dict[str, bytes]:
    """Return all packet files keyed by packet-relative path."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_checked_in_packet_and_negative_variants_validate() -> None:
    """The checked-in packet passes the standalone validator."""
    assert validator.validate_packet(PACKET) == 0


def test_generation_is_byte_identical_in_two_clean_directories(tmp_path: Path) -> None:
    """Two fresh generations must have identical bytes and file paths."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    builder.generate(first)
    builder.generate(second)

    assert _tree_bytes(first) == _tree_bytes(second)


def test_validator_fails_closed_on_outer_checksum_drift(tmp_path: Path) -> None:
    """Changing a packet file without updating SHA256SUMS must fail closed."""
    candidate = tmp_path / "packet"
    shutil.copytree(PACKET, candidate)
    episode_path = candidate / "episodes.jsonl"
    episode_path.write_text(episode_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    assert validator.validate_packet(candidate) == 1


def test_standalone_validator_has_no_robot_sf_import() -> None:
    """The distributed validator must not depend on Robot SF internals."""
    source = (ROOT / "scripts/tools/validate_ecosystem_handoff_fixture.py").read_text(
        encoding="utf-8"
    )

    assert "from robot_sf" not in source
    assert "import robot_sf" not in source
