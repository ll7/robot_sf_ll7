"""Parity tests for the historical SocNav ORCA analysis helper configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.tools import policy_analysis_run
from scripts.training import train_ppo

CONFIG_DIR = Path("configs/training")
BASELINE_PATH = Path("tests/training/_baseline_issue_7325_resolved.json")
VARIANTS = (
    "benchmark_orca_classic_cross_trap_subset.yaml",
    "benchmark_orca_classic_crossing_subset.yaml",
)


def _resolved_mapping(config_path: Path) -> dict[str, Any]:
    """Return the loader-resolved mapping with portable repository-relative paths."""
    resolved = train_ppo._load_expert_training_config_mapping(config_path)
    scenario_path = Path(str(resolved["scenario_config"]))
    if not scenario_path.is_absolute():
        scenario_path = config_path.parent / scenario_path
    resolved["scenario_config"] = scenario_path.resolve().relative_to(Path.cwd()).as_posix()
    return resolved


def test_orca_helper_configs_preserve_prechange_resolved_values() -> None:
    """Shared inheritance must preserve both historical effective mappings exactly."""
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    for variant in VARIANTS:
        config_path = (CONFIG_DIR / variant).resolve()
        assert _resolved_mapping(config_path) == baseline["variants"][variant]


def test_policy_analysis_loads_both_historical_paths_without_running_episodes() -> None:
    """The real policy-analysis config path resolves both wrappers without execution."""
    for variant in VARIANTS:
        args = policy_analysis_run._build_parser().parse_args(
            ["--training-config", str(CONFIG_DIR / variant), "--policy", "socnav_orca"]
        )
        context = policy_analysis_run._load_training_context(args)
        assert context.training_config is not None
        assert context.training_config.policy_id.startswith("socnav_orca_classic_")
        assert context.training_config.scenario_config.is_file()
