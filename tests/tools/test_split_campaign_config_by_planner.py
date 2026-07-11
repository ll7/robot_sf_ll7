"""Tests for the deterministic per-planner camera-ready config splitter."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from robot_sf.benchmark.camera_ready._config import load_campaign_config
from scripts.tools import split_campaign_config_by_planner as splitter


def _fixture_parent() -> dict[str, object]:
    return {
        "name": "fixture_campaign",
        "scenario_matrix": "configs/scenarios/classic_interactions_francis2023.yaml",
        "seed_policy": {"mode": "scenario-default"},
        "custom_top_level": {"preserve": ["this", {"nested": True}]},
        "planners": [
            {"key": "goal", "algo": "goal", "planner_group": "core"},
            {"key": "orca", "algo": "orca", "planner_group": "core"},
            {
                "key": "disabled",
                "algo": "goal",
                "planner_group": "experimental",
                "enabled": False,
            },
            {"key": "hybrid", "algo": "hybrid_rule_local_planner", "planner_group": "experimental"},
        ],
    }


def _write_parent(path: Path) -> dict[str, object]:
    payload = _fixture_parent()
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload


def test_split_per_planner_preserves_parent_payload_and_is_idempotent(
    tmp_path: Path, capsys
) -> None:
    """Each enabled planner gets an unchanged payload except for split metadata."""

    parent_path = tmp_path / "parent.yaml"
    parent = _write_parent(parent_path)
    parent_bytes = parent_path.read_bytes()
    out_dir = tmp_path / "splits"

    assert splitter.main(["--config", str(parent_path), "--out-dir", str(out_dir)]) == 0
    assert "Skipping disabled planner arm: disabled" in capsys.readouterr().err

    children = sorted(out_dir.glob("*.yaml"))
    assert [path.name for path in children] == [
        "fixture_campaign__arm_goal.yaml",
        "fixture_campaign__arm_hybrid.yaml",
        "fixture_campaign__arm_orca.yaml",
    ]
    assert parent_path.read_bytes() == parent_bytes

    manifest = json.loads((out_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_sha256"] == hashlib.sha256(parent_bytes).hexdigest()
    assert [child["planner_keys"] for child in manifest["children"]] == [
        ["goal"],
        ["orca"],
        ["hybrid"],
    ]

    child_payloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in children]
    assert {entry["planners"][0]["key"] for entry in child_payloads} == {"goal", "orca", "hybrid"}
    for payload in child_payloads:
        key = payload["planners"][0]["key"]
        expected = copy.deepcopy(parent)
        expected["name"] = f"fixture_campaign__arm_{key}"
        expected["planners"] = [next(item for item in parent["planners"] if item["key"] == key)]
        expected["split_provenance"] = {
            "source_config": str(parent_path),
            "source_sha256": hashlib.sha256(parent_bytes).hexdigest(),
            "split_mode": "per_planner",
            "arm_key": key,
            "arm_index": ["goal", "orca", "hybrid"].index(key),
            "arm_total": 3,
            "tool": "split_campaign_config_by_planner.py",
        }
        assert payload == expected

    first_bytes = {path.name: path.read_bytes() for path in out_dir.iterdir()}
    assert splitter.main(["--config", str(parent_path), "--out-dir", str(out_dir)]) == 0
    assert {path.name: path.read_bytes() for path in out_dir.iterdir()} == first_bytes


def test_split_by_planner_group_emits_one_child_per_group(tmp_path: Path) -> None:
    """The optional group mode keeps all enabled arms in each planner group."""

    parent_path = tmp_path / "parent.yaml"
    _write_parent(parent_path)
    out_dir = tmp_path / "group-splits"

    result = splitter.split_campaign_config(parent_path, out_dir, group_by="planner_group")

    assert [child.output_path.name for child in result.children] == [
        "fixture_campaign__group_core.yaml",
        "fixture_campaign__group_experimental.yaml",
    ]
    core_payload = yaml.safe_load((out_dir / "fixture_campaign__group_core.yaml").read_text())
    assert [planner["key"] for planner in core_payload["planners"]] == ["goal", "orca"]
    assert core_payload["split_provenance"]["arm_key"] == "core"
    assert core_payload["split_provenance"]["split_mode"] == "per_group"


def test_real_h500_s20_children_load_without_error(tmp_path: Path) -> None:
    """Nested emitted configs remain directly runnable by the existing config loader."""

    parent_path = Path(
        "configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml"
    )
    result = splitter.split_campaign_config(parent_path, tmp_path / "s20")

    assert len(result.children) == 9
    for child in result.children:
        config = load_campaign_config(child.output_path)
        assert len(config.planners) == 1
