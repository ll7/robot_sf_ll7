"""Dry-run provenance tests for the issue #4017 constrained-RL training entry point."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import yaml

from scripts.training.train_constrained_rl import (
    load_constrained_rl_config,
    train_constrained_rl,
)


def test_dry_run_writes_manifest_resolved_config_and_empty_trace(tmp_path: Path) -> None:
    """Dry-run validates the config and writes reviewable smoke provenance."""

    source_config = Path("configs/training/ppo/issue_4017_constrained_smoke.yaml")
    config = load_constrained_rl_config(source_config)
    config = replace(config, output_dir=tmp_path / "ppo_lagrangian_issue_4017_smoke")

    outputs = train_constrained_rl(config, config_path=source_config, dry_run=True)

    manifest_path = outputs["manifest_path"]
    resolved_config_path = outputs["resolved_config_path"]
    trace_path = outputs["trace_path"]
    assert isinstance(manifest_path, Path)
    assert isinstance(resolved_config_path, Path)
    assert isinstance(trace_path, Path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))

    assert manifest["policy_id"] == "ppo_lagrangian_issue_4017_smoke"
    assert manifest["algorithm"] == "ppo_lagrangian"
    assert manifest["evidence_tier"] == "smoke"
    assert manifest["dry_run"] is True
    assert manifest["fallback_or_degraded"] is False
    assert manifest["checkpoint_path"] is None
    assert len(manifest["constraints"]) == 3
    assert resolved["safety_constraints"]["enabled"] is True
    assert trace_path.read_text(encoding="utf-8") == ""
