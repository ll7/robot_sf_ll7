"""Tests for the issue #3207 fixed-scope run-plan consumption path.

These cover the runner-side counterpart to the fixed-scope preflight: the
campaign runner consumes the pre-registered plan and enumerates the concrete
planner_group x axis-variant x seed run cells, while keeping actual launch
fail-closed behind unmet launch prerequisites (ORCA/rvo2, hybrid opt-in, and the
post-run rank-identifiability recheck). No benchmark episodes are run.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark.fidelity_fixed_scope_preflight import build_fixed_scope_preflight

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "research" / "fidelity_sensitivity_v1.yaml"


def _load_campaign_runner() -> ModuleType:
    module_path = REPO_ROOT / "scripts" / "benchmark" / "run_fidelity_sensitivity_campaign.py"
    spec = importlib.util.spec_from_file_location(
        "fidelity_sensitivity_campaign_runner", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load campaign runner module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fidelity_sensitivity_campaign_runner"] = module
    spec.loader.exec_module(module)
    return module


campaign_runner = _load_campaign_runner()


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _plan() -> dict:
    return campaign_runner.build_fixed_scope_run_plan(
        _config(),
        config_path="configs/research/fidelity_sensitivity_v1.yaml",
        git_head="test-head",
    )


def test_run_plan_cell_count_matches_preflight_materialization() -> None:
    """Enumerated run cells must equal preflight run_cells_per_scenario (3x12x3=108)."""
    config = _config()
    preflight = build_fixed_scope_preflight(
        config, config_path="configs/research/fidelity_sensitivity_v1.yaml", git_head="test-head"
    )
    plan = _plan()

    expected = preflight["materialized_scope"]["run_cells_per_scenario"]
    assert plan["run_cell_count"] == expected
    assert len(plan["run_cells"]) == expected
    assert plan["run_cells_per_scenario_expected"] == expected


def test_run_cells_carry_resolved_algorithm_and_scope_axes() -> None:
    """Each cell exposes the resolved catalog algorithm, axis, variant, and seed."""
    plan = _plan()
    groups = {cell["planner_group"] for cell in plan["run_cells"]}
    axes = {cell["axis"] for cell in plan["run_cells"]}
    seeds = {cell["seed"] for cell in plan["run_cells"]}
    algorithms = {cell["planner_group"]: cell["algorithm"] for cell in plan["run_cells"]}

    assert groups == {"orca", "default_social_force", "hybrid_rule_v0_minimal"}
    assert {
        "integration_timestep",
        "social_force_speed_archetypes",
        "observation_noise",
        "clearance_radius",
    } <= axes
    assert seeds == {111, 112, 113}
    # default_social_force resolves to the canonical social_force algorithm.
    assert algorithms["default_social_force"] == "social_force"
    # Baseline variants are included in the full fixed scope: one baseline per
    # axis (4 axes) x every planner group (3) x every seed (3) => 36 cells.
    assert sum(1 for cell in plan["run_cells"] if cell["baseline_variant"]) == 4 * len(
        groups
    ) * len(seeds)


def test_shipped_config_plan_is_not_launchable_and_records_gates() -> None:
    """The shipped config is preflight_ready but fails closed on launch prerequisites."""
    plan = _plan()

    assert plan["preflight_decision"] == "preflight_ready"
    assert plan["preflight_ready"] is True
    # Fail-closed: unmet launch prerequisites keep the plan non-executable / unlaunched.
    assert plan["executable"] is False
    assert plan["launched"] is False
    assert plan["gate_reasons"], "expected residual launch gates"

    joined = " ".join(plan["gate_reasons"])
    assert "rvo2" in joined  # ORCA runtime dependency
    assert "explicit_opt_in:hybrid_rule_v0_minimal" in joined  # hybrid opt-in
    assert "runtime_rank_identifiability_recheck_required" in joined  # post-run recheck
    assert "campaign_runner_not_wired_for_full_fixed_scope" in joined


def test_ensure_launchable_raises_when_gated() -> None:
    """ensure_fixed_scope_launchable must fail closed while gates remain."""
    plan = _plan()
    with pytest.raises(campaign_runner.FixedScopeNotLaunchableError):
        campaign_runner.ensure_fixed_scope_launchable(plan)


def test_ensure_launchable_passes_only_when_all_gates_cleared() -> None:
    """A fully cleared plan is the only launchable state."""
    plan = _plan()
    plan["executable"] = True
    plan["gate_reasons"] = []
    # Should not raise.
    campaign_runner.ensure_fixed_scope_launchable(plan)


def test_unresolved_planner_group_blocks_plan_fail_closed() -> None:
    """Removing the planner_algorithms binding must surface a fail-closed blocker."""
    config = copy.deepcopy(_config())
    # Without the explicit binding, the ``default_social_force`` label no longer
    # resolves to a catalog algorithm and must fail closed as a blocker.
    del config["fixed_scope"]["planner_algorithms"]
    plan = campaign_runner.build_fixed_scope_run_plan(
        config,
        config_path="configs/research/fidelity_sensitivity_v1.yaml",
        git_head="test-head",
    )

    assert plan["preflight_decision"] == "blocked"
    assert plan["executable"] is False
    assert any(reason.startswith("planner_unavailable:") for reason in plan["blockers"])
    with pytest.raises(campaign_runner.FixedScopeNotLaunchableError):
        campaign_runner.ensure_fixed_scope_launchable(plan)


def test_plan_only_cli_writes_plan_without_running_episodes(tmp_path: Path) -> None:
    """--fixed-scope-plan-only emits the plan JSON and never runs an episode."""
    plan_dir = tmp_path / "plan"
    exit_code = campaign_runner.main(
        [
            "--fixed-scope-plan-only",
            "--plan-out",
            str(plan_dir),
        ]
    )
    assert exit_code == 0
    plan_path = plan_dir / "fidelity_fixed_scope_run_plan.json"
    assert plan_path.exists()
    import json

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["schema_version"] == campaign_runner.FIXED_SCOPE_PLAN_SCHEMA_VERSION
    assert plan["launched"] is False
    assert plan["run_cell_count"] == 108


def test_plan_only_cli_require_launchable_fails_closed(tmp_path: Path) -> None:
    """--require-launchable exits non-zero because launch gates remain unmet."""
    exit_code = campaign_runner.main(
        [
            "--fixed-scope-plan-only",
            "--plan-out",
            str(tmp_path / "plan"),
            "--require-launchable",
        ]
    )
    assert exit_code == 1
