"""Exercise the packet refresh CLI without changing immutable source identities."""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "scripts/dev/refresh_packet_working_tree_pins.py"


def run(root, packet, *args):
    # An absent command is an explicit feature failure, never an import/fixture error.
    assert TOOL.is_file(), "the packet working-tree pin refresh command is missing"
    return subprocess.run(
        [sys.executable, str(TOOL), "--root", str(root), str(packet), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def packet_fixture(root, name="packet.yaml"):
    (root / "implementation.py").write_bytes(b"new implementation\n")
    packet = root / name
    packet.parent.mkdir(parents=True, exist_ok=True)
    immutable = "a" * 64
    old = "b" * 64
    contents = (
        "# Preserve this comment and flow formatting.\n"
        "base_commit: historical-commit\n"
        f"inputs: {{runner: {{path: implementation.py, sha256: {immutable}, "
        f"working_tree_sha256: '{old}'}}, other: {{path: implementation.py, sha256: {immutable}}}}}\n"
    ).encode()
    packet.write_bytes(contents)
    return packet, contents, old


def test_check_and_refresh_change_only_existing_current_tree_pin(tmp_path):
    packet, original, old = packet_fixture(tmp_path)
    dirty = run(tmp_path, packet, "--check")
    assert dirty.returncode == 1, dirty.stderr
    assert "implementation.py" in dirty.stdout
    assert packet.read_bytes() == original
    refreshed = run(tmp_path, packet)
    assert refreshed.returncode == 0, refreshed.stderr
    actual = hashlib.sha256(b"new implementation\n").hexdigest()
    assert packet.read_bytes() == original.replace(old.encode(), actual.encode())
    assert run(tmp_path, packet, "--check").returncode == 0
    assert run(tmp_path, packet).returncode == 0
    assert packet.read_bytes() == original.replace(old.encode(), actual.encode())


@pytest.mark.parametrize(
    "name",
    [
        "packet_frozen.yaml",
        "docs/context/evidence/packet.yaml",
        "releases/0.0.8/packet.yaml",
        "0.0.2/packet.yaml",
        "0.0.7/packet.yaml",
    ],
)
def test_protected_packets_are_refused_without_writing(tmp_path, name):
    packet, original, _ = packet_fixture(tmp_path, name)
    result = run(tmp_path, packet)
    assert result.returncode == 2
    assert "protected packet" in result.stderr
    assert packet.read_bytes() == original


def test_packet_without_current_tree_pin_is_refused(tmp_path):
    packet = tmp_path / "packet.yaml"
    original = b"base_commit: old\ninputs: {runner: {path: missing.py, sha256: immutable}}\n"
    packet.write_bytes(original)
    result = run(tmp_path, packet)
    assert result.returncode == 2
    assert "no working_tree_sha256" in result.stderr
    assert packet.read_bytes() == original


def test_root_option_cannot_hide_a_release_tree(tmp_path):
    root = tmp_path / "releases" / "0.0.8"
    root.mkdir(parents=True)
    packet, original, _ = packet_fixture(root)
    result = run(root, packet)
    assert result.returncode == 2
    assert "protected packet" in result.stderr
    assert packet.read_bytes() == original


def test_shared_immutable_scalar_is_not_rewritten(tmp_path):
    packet, original, old = packet_fixture(tmp_path)
    original = original.replace(
        ("sha256: " + "a" * 64).encode(), ("sha256: &immutable " + old).encode(), 1
    )
    original = original.replace(
        ("working_tree_sha256: '" + old + "'").encode(), b"working_tree_sha256: *immutable"
    )
    packet.write_bytes(original)
    result = run(tmp_path, packet)
    assert result.returncode == 2
    assert "anchored or tagged" in result.stderr
    assert packet.read_bytes() == original
