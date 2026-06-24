"""Tests generic launch-packet preflight helper."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path  # noqa: TC003

import yaml

from scripts.dev.preflight_launch_packet import main, preflight_launch_packet


def _sha256(path: Path) -> str:
    """Return SHA-256 for fixture file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preflight_launch_packet_accepts_checked_paths(tmp_path: Path) -> None:
    """A concrete packet with valid paths, SHA, seed, command, and gate is ready."""

    repo = tmp_path
    config = repo / "configs" / "benchmarks" / "example.yaml"
    script = repo / "scripts" / "tools" / "run_example.py"
    packet = repo / "configs" / "benchmarks" / "issue_999_launch_packet.yaml"
    config.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    config.write_text("ok: true\n", encoding="utf-8")
    script.write_text("print('ok')\n", encoding="utf-8")
    packet.write_text(
        yaml.safe_dump(
            {
                "schema_version": "example.v1",
                "issue": 999,
                "claim_gate": {"target_claim": "diagnostic only"},
                "seed_budget": {"primary": {"seed_set": "tiny"}},
                "campaign_config": "configs/benchmarks/example.yaml",
                "campaign_config_sha256": _sha256(config),
                "campaign_command": (
                    "uv run python scripts/tools/run_example.py "
                    "--config configs/benchmarks/example.yaml"
                ),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    report = preflight_launch_packet(packet, repo_root=repo)

    assert report["ready"] is True
    assert report["issue"] == 999
    assert report["kind"] == "benchmark"
    assert report["reasons"] == []
    assert any(item["sha256_matches"] is True for item in report["configs"])


def test_preflight_launch_packet_fails_closed_on_placeholder_and_sha_drift(
    tmp_path: Path,
) -> None:
    """Placeholders and checksum drift should make readiness explicit false."""

    repo = tmp_path
    config = repo / "configs" / "training" / "example.yaml"
    packet = repo / "configs" / "training" / "issue_1000_launch_packet.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("changed: true\n", encoding="utf-8")
    packet.write_text(
        yaml.safe_dump(
            {
                "schema_version": "example.v1",
                "issue": 1000,
                "execution_boundary": "launch-packet only",
                "seed_policy": {"seed_set": "<seed-set>"},
                "training_config": "configs/training/example.yaml",
                "training_config_sha256": "0" * 64,
                "validation_command": "uv run python <script>",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    report = preflight_launch_packet(packet, repo_root=repo)

    assert report["ready"] is False
    assert any("sha256 mismatch" in reason for reason in report["reasons"])
    assert any("placeholder values remain" in reason for reason in report["reasons"])


def test_preflight_launch_packet_rejects_path_escape_with_sha(tmp_path: Path) -> None:
    """Checksum preflight must not hash paths outside the repository root."""

    repo = tmp_path / "repo"
    outside = tmp_path / "outside.txt"
    packet = repo / "configs" / "benchmarks" / "issue_1001_launch_packet.yaml"
    packet.parent.mkdir(parents=True)
    outside.write_text("secret-ish\n", encoding="utf-8")
    packet.write_text(
        yaml.safe_dump(
            {
                "schema_version": "example.v1",
                "issue": 1001,
                "claim_gate": {"target_claim": "none"},
                "campaign_command": "echo no-op",
                "checksums": {"../outside.txt": _sha256(outside)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    report = preflight_launch_packet(packet, repo_root=repo)
    checked = report["configs"][0]

    assert report["ready"] is False
    assert checked["sha256_observed"] is None
    assert any("escapes repository root" in reason for reason in report["reasons"])


def test_preflight_launch_packet_strips_command_path_punctuation(tmp_path: Path) -> None:
    """Command token paths may appear in prose with trailing punctuation."""

    repo = tmp_path
    config = repo / "configs" / "training" / "example.yaml"
    packet = repo / "configs" / "training" / "issue_1002_launch_packet.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("ok: true\n", encoding="utf-8")
    packet.write_text(
        yaml.safe_dump(
            {
                "schema_version": "example.v1",
                "issue": 1002,
                "execution_boundary": "launch-packet only",
                "seed_policy": {"seed_set": "tiny"},
                "slurm_command_shape": (
                    "submit configs/training/example.yaml, then write output elsewhere."
                ),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    report = preflight_launch_packet(packet, repo_root=repo)

    assert report["ready"] is True
    assert any(item["path"] == "configs/training/example.yaml" for item in report["configs"])


def test_preflight_launch_packet_cli_writes_json(tmp_path: Path, capsys) -> None:
    """CLI should emit and optionally write JSON preflight reports."""

    packet = tmp_path / "configs" / "benchmarks" / "issue_5_launch_packet.yaml"
    output = tmp_path / "preflight.json"
    packet.parent.mkdir(parents=True)
    packet.write_text(
        yaml.safe_dump(
            {
                "schema_version": "example.v1",
                "issue": 5,
                "claim_gate": {"target_claim": "none"},
                "campaign_command": "echo no-op",
            }
        ),
        encoding="utf-8",
    )

    rc = main(["--packet", str(packet), "--repo-root", str(tmp_path), "--output", str(output)])
    stdout = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))

    assert rc == 0
    assert stdout["ready"] is True
    assert written["issue"] == 5
