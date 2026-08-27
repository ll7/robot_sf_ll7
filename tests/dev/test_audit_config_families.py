"""Tests for the config-family inventory tool (issue #7901)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.dev.audit_config_families import (
    MIN_FAMILY_SIZE,
    SCHEMA,
    _common_resolved_paths,
    _family_key,
    run_inventory,
    scan_config,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "audit_config_families.py"


def _write(root: Path, name: str, content: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


LEAF = """policy: ppo
learning_rate: 0.0003
gamma: 0.99
seed: 1
"""


def test_scan_config_records_digest_and_lines(tmp_path: Path) -> None:
    path = _write(tmp_path, "leaf.yaml", LEAF)
    record = scan_config(path)
    assert record["error"] is None
    assert record["resolved_digest"]
    assert record["key_count"] == 4
    assert record["line_count"] == 4
    assert record["inheritance_chain"] == [str(path)]


def test_scan_config_resolves_base_config(tmp_path: Path) -> None:
    _write(tmp_path, "base.yaml", "gamma: 0.99\nlearning_rate: 0.001\n")
    path = _write(
        tmp_path,
        "leaf.yaml",
        "base_config: base.yaml\nlearning_rate: 0.0003\nseed: 1\n",
    )
    record = scan_config(path)
    assert record["error"] is None
    assert record["resolved"]["gamma"] == 0.99
    assert record["resolved"]["learning_rate"] == 0.0003
    assert len(record["inheritance_chain"]) == 2


def test_scan_config_detects_missing_base(tmp_path: Path) -> None:
    path = _write(tmp_path, "leaf.yaml", "base_config: missing.yaml\nseed: 1\n")
    record = scan_config(path)
    assert record["error"] is not None
    assert "does not exist" in record["error"]


def test_scan_config_detects_cycle(tmp_path: Path) -> None:
    _write(tmp_path, "a.yaml", "base_config: b.yaml\n")
    _write(tmp_path, "b.yaml", "base_config: a.yaml\n")
    record = scan_config(_write(tmp_path, "c.yaml", "base_config: a.yaml\nseed: 1\n"))
    assert record["error"] is not None
    assert "cycle" in record["error"].lower() or "cycle" in str(record["error"]).lower()


def test_scan_config_marks_unsupported_category(tmp_path: Path) -> None:
    path = _write(tmp_path, "carla_thing.yaml", "category: carla_scenario\nseed: 1\n")
    record = scan_config(path)
    assert record["error"] is not None
    assert "unsupported" in record["error"]


def test_family_key_strips_seed_suffixes() -> None:
    assert _family_key("configs/algos/ppo_seed1.yaml") == "ppo"
    assert _family_key("configs/algos/ppo_camera_ready.yaml") == "ppo"
    assert _family_key("configs/algos/other.yaml") == "other"


def test_common_resolved_paths(tmp_path: Path) -> None:
    _write(tmp_path, "base.yaml", "gamma: 0.99\nlearning_rate: 0.001\n")
    members = [
        scan_config(_write(tmp_path, f"ppo_seed{i}.yaml", f"base_config: base.yaml\nseed: {i}\n"))
        for i in range(3)
    ]
    common = _common_resolved_paths(members)
    assert any(path == ("gamma", "0.99") for path in common)
    assert not any(path == ("seed",) for path in common if len(path) == 2)


def test_run_inventory_groups_family(tmp_path: Path) -> None:
    root = tmp_path / "configs"
    # Identical standalone leaves (no base_config) form a candidate family.
    for i in range(3):
        _write(root, f"ppo_seed{i}.yaml", f"gamma: 0.99\nlearning_rate: 0.001\nseed: {i}\n")
    report = run_inventory([root])
    assert report["schema"] == SCHEMA
    assert report["scan"]["resolved_count"] == 3
    assert len(report["candidate_families"]) >= 1
    family = report["candidate_families"][0]
    assert family["family"] == "ppo"
    assert family["member_count"] >= MIN_FAMILY_SIZE


def test_run_inventory_excludes_already_migrated_families(tmp_path: Path) -> None:
    """Families whose members already declare a base_config are not recommended."""
    root = tmp_path / "configs"
    _write(root, "base.yaml", "gamma: 0.99\nlearning_rate: 0.001\n")
    for i in range(3):
        _write(root, f"ppo_seed{i}.yaml", f"base_config: base.yaml\nseed: {i}\n")
    report = run_inventory([root])
    assert report["candidate_families"] == []
    assert report["disposition"] == "no_safe_family"


def test_repeated_runs_byte_stable(tmp_path: Path) -> None:
    root = tmp_path / "configs"
    _write(root, "base.yaml", "gamma: 0.99\n")
    for i in range(3):
        _write(root, f"ppo_seed{i}.yaml", f"base_config: base.yaml\nseed: {i}\n")
    first = json.dumps(run_inventory([root]), sort_keys=True)
    second = json.dumps(run_inventory([root]), sort_keys=True)
    assert first == second


def test_cli_emits_deterministic_json(tmp_path: Path) -> None:
    root = tmp_path / "configs"
    _write(root, "base.yaml", "gamma: 0.99\n")
    for i in range(3):
        _write(root, f"ppo_seed{i}.yaml", f"base_config: base.yaml\nseed: {i}\n")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--roots", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    report = json.loads(proc.stdout)
    assert report["schema"] == SCHEMA


def test_cli_markdown_report(tmp_path: Path) -> None:
    root = tmp_path / "configs"
    _write(root, "base.yaml", "gamma: 0.99\n")
    for i in range(3):
        _write(root, f"ppo_seed{i}.yaml", f"base_config: base.yaml\nseed: {i}\n")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--roots", str(root), "--markdown"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "Config-family inventory" in proc.stdout
