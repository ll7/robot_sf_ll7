"""Tests for the launch-packet preflight helper (#3549)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from scripts.dev.preflight_launch_packet import (
    find_sha_pairs,
    preflight_packet,
    sha256_of,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_repo(tmp_path: Path) -> Path:
    """Create a minimal repo root with configs/ and scripts/ dirs."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "scripts").mkdir()
    return tmp_path


def _write_config(repo: Path, rel: str, body: str = "a: 1\n") -> tuple[str, str]:
    """Write a config file and return (rel_path, sha256)."""
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return rel, sha256_of(path)


def _write_packet(repo: Path, name: str, packet: dict) -> Path:
    """Write a launch packet YAML under configs/ and return its path."""
    path = repo / "configs" / name
    path.write_text(yaml.safe_dump(packet))
    return path


def _ready_packet(repo: Path, name: str = "x_issue_4242_launch_packet.yaml") -> Path:
    """A fully run-ready packet referencing a real, sha-matched config."""
    rel, sha = _write_config(repo, "configs/real_config.yaml")
    packet = {
        "schema_version": "test-packet.v1",
        "runnable_config": {"config": rel, "config_sha256": sha},
        "campaign_command": "uv run python run.py",
        "claim_gate": {"no_claim_until_run": True},
        "seed_budget": {"primary": {"seed_set": "paper_eval_s20"}},
    }
    return _write_packet(repo, name, packet)


def test_clean_packet_is_ready(tmp_path: Path):
    """A packet with matched sha, command, and claim gate is run-ready."""
    repo = _make_repo(tmp_path)
    report = preflight_packet(_ready_packet(repo), repo)
    assert report["ready"] is True
    assert report["drift_guarded"] is True
    assert report["issue"] == 4242
    assert report["reasons"] == []
    assert report["command_keys"] == ["campaign_command"]


def test_sha_drift_fails_closed(tmp_path: Path):
    """A drifted sha256 makes the packet not run-ready."""
    repo = _make_repo(tmp_path)
    rel, _ = _write_config(repo, "configs/real_config.yaml")
    packet = {
        "runnable_config": {"config": rel, "config_sha256": "0" * 64},
        "campaign_command": "run",
        "claim_gate": "x",
    }
    report = preflight_packet(_write_packet(repo, "p_issue_1_launch_packet.yaml", packet), repo)
    assert report["ready"] is False
    assert report["drift_guarded"] is False
    assert any("sha256 drift" in r for r in report["reasons"])


def test_missing_config_fails_closed(tmp_path: Path):
    """A sha-pinned config that does not exist fails closed."""
    repo = _make_repo(tmp_path)
    packet = {
        "runnable_config": {"config": "configs/does_not_exist.yaml", "config_sha256": "a" * 64},
        "campaign_command": "run",
        "claim_gate": "x",
    }
    report = preflight_packet(_write_packet(repo, "p_issue_2_launch_packet.yaml", packet), repo)
    assert report["ready"] is False
    assert any("missing" in r for r in report["reasons"])


def test_missing_command_fails_closed(tmp_path: Path):
    """A packet with no runnable command key is not ready."""
    repo = _make_repo(tmp_path)
    rel, sha = _write_config(repo, "configs/real_config.yaml")
    packet = {"runnable_config": {"config": rel, "config_sha256": sha}, "claim_gate": "x"}
    report = preflight_packet(_write_packet(repo, "p_issue_3_launch_packet.yaml", packet), repo)
    assert report["ready"] is False
    assert any("command" in r for r in report["reasons"])


def test_missing_claim_gate_fails_closed(tmp_path: Path):
    """A packet without a claim/evidence boundary is not ready."""
    repo = _make_repo(tmp_path)
    rel, sha = _write_config(repo, "configs/real_config.yaml")
    packet = {"runnable_config": {"config": rel, "config_sha256": sha}, "campaign_command": "run"}
    report = preflight_packet(_write_packet(repo, "p_issue_4_launch_packet.yaml", packet), repo)
    assert report["ready"] is False
    assert any("claim" in r or "evidence" in r for r in report["reasons"])


def test_unparseable_packet_fails_closed(tmp_path: Path):
    """Invalid YAML yields not-ready with an unparseable reason."""
    repo = _make_repo(tmp_path)
    path = repo / "configs" / "broken_issue_9_launch_packet.yaml"
    path.write_text("a: [unbalanced\n")
    report = preflight_packet(path, repo)
    assert report["ready"] is False
    assert any("unparseable" in r for r in report["reasons"])


def test_no_config_reference_fails_closed(tmp_path: Path):
    """A packet referencing no in-repo config is not ready."""
    repo = _make_repo(tmp_path)
    packet = {"campaign_command": "run", "claim_gate": "x", "schema_version": "v"}
    report = preflight_packet(_write_packet(repo, "p_issue_5_launch_packet.yaml", packet), repo)
    assert report["ready"] is False
    assert any("no in-repo config paths" in r for r in report["reasons"])


def test_find_sha_pairs_sibling_rule():
    """Only keys with a sibling <key>_sha256 form sha pairs."""
    packet = {
        "runnable_config": {
            "config": "configs/a.yaml",
            "config_sha256": "aa",
            "matrix": "configs/b.yaml",
            "matrix_sha256": "bb",
            "kinematics": "differential_drive",  # no sibling sha -> not a pair
        }
    }
    pairs = find_sha_pairs(packet)
    paths = {p["path"]: p["expected_sha256"] for p in pairs}
    assert paths == {"configs/a.yaml": "aa", "configs/b.yaml": "bb"}


def test_seed_count_resolution(tmp_path: Path):
    """A declared seed-set token resolves to its concrete seed count."""
    repo = _make_repo(tmp_path)
    (repo / "configs" / "benchmarks").mkdir(parents=True)
    (repo / "configs" / "benchmarks" / "seed_sets_v1.yaml").write_text(
        yaml.safe_dump({"paper_eval_s20": {"seeds": list(range(111, 131))}})
    )
    report = preflight_packet(_ready_packet(repo), repo)
    assert report["seed_set"] == "paper_eval_s20"
    assert report["seed_count"] == 20
