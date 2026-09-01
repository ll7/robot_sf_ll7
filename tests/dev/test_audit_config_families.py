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
    assert family["estimated_after_lines"] == family["estimated_base_lines"] + sum(
        family["estimated_leaf_lines"]
    )
    assert len(family["estimated_leaf_lines"]) == 3


def test_reduction_math_decomposition_and_regression(tmp_path: Path) -> None:
    """Estimated after lines must equal base plus sum of all leaf lines."""
    root = tmp_path / "configs"
    _write(
        root,
        "ppo_run_seed1.yaml",
        "gamma: 0.99\nlearning_rate: 0.001\nentropy_coef: 0.01\nclip_range: 0.2\nseed: 1\n",
    )
    _write(
        root,
        "ppo_run_seed2.yaml",
        "gamma: 0.99\nlearning_rate: 0.001\nentropy_coef: 0.01\nclip_range: 0.2\nseed: 2\n",
    )
    _write(
        root,
        "ppo_run_seed3.yaml",
        "gamma: 0.99\nlearning_rate: 0.001\nentropy_coef: 0.01\nclip_range: 0.2\nseed: 3\n",
    )
    report = run_inventory([root])
    family = report["candidate_families"][0]
    assert family["before_lines"] == 15
    assert family["estimated_base_lines"] == 4  # gamma, lr, entropy, clip
    # Each leaf has base_config (1 line) + seed (1 line) = 2 lines
    assert family["estimated_leaf_lines"] == [2, 2, 2]
    assert family["estimated_after_lines"] == 4 + (2 + 2 + 2)  # 10 lines
    expected_reduction = round(1.0 - (10 / 15), 3)
    assert family["estimated_reduction"] == expected_reduction
    assert family["estimated_reduction"] == 0.333
    assert len(report["ready_families"]) == 1


def test_nested_common_and_differing_projection(tmp_path: Path) -> None:
    """Nested mappings retain common values in base and differing values in leaves."""
    root = tmp_path / "configs"
    for i in range(3):
        content = (
            "algo:\n"
            "  gamma: 0.99\n"
            f"  learning_rate: 0.00{i}\n"
            "  policy:\n"
            "    net_arch: [64, 64]\n"
            "env:\n"
            "  num_agents: 5\n"
        )
        _write(root, f"ppo_nested_seed{i}.yaml", content)
    report = run_inventory([root])
    family = report["candidate_families"][0]
    assert family["estimated_after_lines"] == family["estimated_base_lines"] + sum(
        family["estimated_leaf_lines"]
    )
    assert family["estimated_base_lines"] > 0
    assert all(leaf_cost > 0 for leaf_cost in family["estimated_leaf_lines"])


def test_threshold_classification_boundary(tmp_path: Path) -> None:
    """Families above and below 20% threshold are classified correctly."""
    root = tmp_path / "configs"
    # Low reduction family: only 1 common key out of 5, 3 members
    # Before = 18 lines. Base = 1 line. Leaves = 5 lines each. After = 1 + 15 = 16 lines.
    # Reduction = 1 - 16/18 = 11.1% (< 20%) -> not ready!
    for i in range(3):
        content = (
            f"common_key: constant\nk1: {i}\nk2: {i * 2}\nk3: {i * 3}\nk4: {i * 4}\nk5: {i * 5}\n"
        )
        _write(root, f"low_family_seed{i}.yaml", content)
    report = run_inventory([root])
    assert len(report["candidate_families"]) == 1
    assert report["candidate_families"][0]["estimated_reduction"] < 0.20
    assert len(report["ready_families"]) == 0
    assert report["disposition"] == "no_safe_family"


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
