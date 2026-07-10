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
    """Enumerated run cells must equal preflight run_cells_per_scenario (3x14x3=126)."""
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
    # axis (5 axes) x every planner group (3) x every seed (3) => 45 cells.
    assert sum(1 for cell in plan["run_cells"] if cell["baseline_variant"]) == 5 * len(
        groups
    ) * len(seeds)


def test_shipped_config_plan_is_launchable_when_prerequisites_are_bound() -> None:
    """The shipped config is launchable when rvo2 and hybrid opt-in are present."""
    plan = _plan()

    assert plan["preflight_decision"] == "preflight_ready"
    assert plan["preflight_ready"] is True
    assert plan["executable"] is True
    assert plan["launched"] is False
    assert plan["gate_reasons"] == []
    hybrid_resolution = next(
        record
        for record in plan["planner_resolution"]
        if record["planner_group"] == "hybrid_rule_v0_minimal"
    )
    assert hybrid_resolution["explicit_opt_in_satisfied"] is True
    assert hybrid_resolution["requires_explicit_opt_in"] is False
    assert "runtime_rank_identifiability_recheck_required" in " ".join(plan["post_run_contracts"])
    rank_contract = plan["post_run_contract_specs"][0]
    assert rank_contract == {
        "id": "runtime_rank_identifiability_recheck",
        "report": "fidelity_rank_stability_report.json",
        "builder": "robot_sf/benchmark/fidelity_rank_stability.py",
        "metric": "snqi",
        "threshold": "non_zero_variance_and_rank_identifiable",
        "output_path": "output/fidelity_sensitivity/<campaign>/rank_identifiability.json",
        "blocks_claims_when_failed": True,
    }


def test_shipped_config_plan_fails_closed_without_rvo2(monkeypatch: pytest.MonkeyPatch) -> None:
    """ORCA cells keep fixed-scope launch non-executable when rvo2 is unavailable."""
    monkeypatch.setattr(
        "robot_sf.benchmark.fidelity_fixed_scope_preflight._rvo2_importable",
        lambda: False,
    )
    plan = _plan()
    assert plan["executable"] is False
    assert plan["gate_reasons"] == [
        "planner_requires_rvo2:orca — install orca extra `uv sync --all-extras`; "
        "if stale local CMake build remains, remove `third_party/python-rvo2/build` "
        "first, retry."
    ]


def test_ensure_launchable_raises_when_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    """ensure_fixed_scope_launchable must fail closed while gates remain."""
    monkeypatch.setattr(
        "robot_sf.benchmark.fidelity_fixed_scope_preflight._rvo2_importable",
        lambda: False,
    )
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
    assert plan["run_cell_count"] == 126


def test_plan_only_cli_require_launchable_passes_when_prerequisites_bound(tmp_path: Path) -> None:
    """--require-launchable exits zero when runtime prerequisites are satisfied."""
    exit_code = campaign_runner.main(
        [
            "--fixed-scope-plan-only",
            "--plan-out",
            str(tmp_path / "plan"),
            "--require-launchable",
        ]
    )
    assert exit_code == 0


# ---------------------------------------------------------------------------
# Per-cell episode wiring: mapping fixed-scope plan cells to concrete runner
# inputs, fail-closed, with no silent fallback (issue #3207 next launch gate).
# ---------------------------------------------------------------------------


def test_variant_index_covers_every_plan_cell() -> None:
    """Every plan cell's (axis, variant) resolves to exactly one materialized variant."""
    config = _config()
    index = campaign_runner.build_fixed_scope_variant_index(config)
    plan = _plan()

    # Four three-variant axes plus the two-variant integration-scheme axis = 14 variants.
    assert len(index) == 14
    for cell in plan["run_cells"]:
        assert (cell["axis"], cell["variant"]) in index
    # Every variant is runtime-bound (no "unsupported"), so no cell is stranded.
    assert all(spec.runtime_binding != "unsupported" for spec in index.values())


def test_bind_social_force_cell_maps_to_runner_planner() -> None:
    """A default_social_force cell binds to the runner's baseline_social_force planner."""
    config = _config()
    index = campaign_runner.build_fixed_scope_variant_index(config)
    cells = campaign_runner._run_cells_from_plan(_plan())
    cell = next(c for c in cells if c.planner_group == "default_social_force")

    binding = campaign_runner.bind_fixed_scope_run_cell(cell, index)
    assert binding.runner_bound is True
    assert binding.planner_name == "baseline_social_force"
    assert binding.unbound_reason is None
    # The bound variant carries the cell's own axis/variant runtime binding.
    assert binding.variant.axis == cell.axis
    assert binding.variant.source_key == cell.variant


def test_bind_orca_cell_is_unbound_no_silent_fallback() -> None:
    """An ORCA cell has no native runner planner and never inherits a substitute."""
    config = _config()
    index = campaign_runner.build_fixed_scope_variant_index(config)
    cells = campaign_runner._run_cells_from_plan(_plan())
    cell = next(c for c in cells if c.planner_group == "orca")

    binding = campaign_runner.bind_fixed_scope_run_cell(cell, index)
    assert binding.runner_bound is True
    assert binding.planner_name == "orca"
    assert binding.unbound_reason is None


def test_bind_hybrid_cell_requires_opt_in_unbound() -> None:
    """The experimental hybrid-rule cell stays unbound pending explicit opt-in."""
    config = _config()
    index = campaign_runner.build_fixed_scope_variant_index(config)
    cells = campaign_runner._run_cells_from_plan(_plan())
    cell = next(c for c in cells if c.planner_group == "hybrid_rule_v0_minimal")

    binding = campaign_runner.bind_fixed_scope_run_cell(cell, index)
    assert binding.runner_bound is True
    assert binding.planner_name == "hybrid_rule_v0_minimal"
    assert binding.unbound_reason is None


def test_bind_plan_preserves_cell_count_and_order() -> None:
    """bind_fixed_scope_run_plan yields one binding per plan cell, in plan order."""
    config = _config()
    plan = _plan()
    bindings = campaign_runner.bind_fixed_scope_run_plan(plan, config)

    assert len(bindings) == plan["run_cell_count"] == 126
    for binding, cell in zip(bindings, plan["run_cells"], strict=True):
        assert binding.cell.planner_group == cell["planner_group"]
        assert binding.cell.axis == cell["axis"]
        assert binding.cell.variant == cell["variant"]
        assert binding.cell.seed == cell["seed"]
    # Only the social-force group is runner-bound today; ORCA/hybrid stay unbound.
    bound_groups = {b.cell.planner_group for b in bindings if b.runner_bound}
    assert bound_groups == {"orca", "default_social_force", "hybrid_rule_v0_minimal"}


def test_bind_missing_variant_raises_not_silently_dropped() -> None:
    """A plan/variant-index disagreement is a hard error, not a silent skip."""
    config = _config()
    index = campaign_runner.build_fixed_scope_variant_index(config)
    cells = campaign_runner._run_cells_from_plan(_plan())
    ghost = cells[0].__class__(
        **{**cells[0].__dict__, "axis": "no_such_axis", "variant": "no_such_variant"}
    )
    with pytest.raises(KeyError):
        campaign_runner.bind_fixed_scope_run_cell(ghost, index)


def test_execute_raises_on_unbound_cells_without_running_any() -> None:
    """execute_fixed_scope_cells fails closed before running a single unbound cell."""
    config = _config()
    bindings = campaign_runner.bind_fixed_scope_run_plan(_plan(), config)
    unbound = bindings[0].__class__(
        cell=bindings[0].cell,
        variant=bindings[0].variant,
        planner_name=None,
        runner_bound=False,
        unbound_reason="unit_test_unbound",
    )
    bindings = [unbound, *bindings[1:]]
    calls: list[object] = []

    def _recording_runner(binding: object) -> list[dict]:
        calls.append(binding)
        return [{"marker": True}]

    with pytest.raises(campaign_runner.FixedScopeNotLaunchableError):
        campaign_runner.execute_fixed_scope_cells(bindings, cell_runner=_recording_runner)
    # No silent fallback: the runner is never invoked when any cell is unbound.
    assert calls == []


def test_execute_runs_one_batch_per_bound_cell_with_correct_inputs() -> None:
    """Each bound cell drives the injected runner with its planner, variant, and seed."""
    config = _config()
    bindings = campaign_runner.bind_fixed_scope_run_plan(_plan(), config)
    bound = [b for b in bindings if b.runner_bound]
    # Sanity: 14 variants x 3 seeds x 3 runner-bound planner groups.
    assert len(bound) == 126

    seen: list[tuple] = []

    def _fake_runner(binding: object) -> list[dict]:
        seen.append((binding.planner_name, binding.variant.source_key, binding.cell.seed))
        return [{"planner": binding.planner_name}]

    rows = campaign_runner.execute_fixed_scope_cells(bound, cell_runner=_fake_runner)
    assert len(rows) == len(bound)
    assert {name for name, _variant, _seed in seen} == {
        "orca",
        "baseline_social_force",
        "hybrid_rule_v0_minimal",
    }
    # Every (variant, seed) pairing in the bound scope is driven exactly once.
    assert len(set(seen)) == len(bound)


def test_fixed_scope_execute_cli_fails_closed_and_writes_no_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--fixed-scope-execute exits non-zero and runs zero episodes when rvo2 is missing."""
    monkeypatch.setattr(
        "robot_sf.benchmark.fidelity_fixed_scope_preflight._rvo2_importable",
        lambda: False,
    )
    raw_root = tmp_path / "raw"
    exit_code = campaign_runner.main(
        [
            "--fixed-scope-execute",
            "--raw-root",
            str(raw_root),
        ]
    )
    assert exit_code == 1
    assert not (raw_root / "episode_rows.jsonl").exists()


# ---------------------------------------------------------------------------
# Post-run contract: rank-identifiability report and checker wiring
# ---------------------------------------------------------------------------


def test_plan_carries_rank_identifiability_contract_spec() -> None:
    """The plan's post_run_contract_specs includes the rank-identifiability recheck."""
    from robot_sf.benchmark.fidelity_rank_stability import (
        PostRunContractResult,
        check_rank_identifiability_contract,
    )

    plan = _plan()
    specs = plan.get("post_run_contract_specs") or []
    rank_spec = next(
        (s for s in specs if s.get("id") == "runtime_rank_identifiability_recheck"),
        None,
    )
    assert rank_spec is not None, "rank-identifiability contract spec missing from plan"
    assert rank_spec["threshold"] == "non_zero_variance_and_rank_identifiable"
    assert rank_spec["blocks_claims_when_failed"] is True
    assert rank_spec["builder"] == "robot_sf/benchmark/fidelity_rank_stability.py"

    # The contract checker is importable and callable on a well-formed report.
    identifiable_report = {"rank_identifiable": True, "rank_identifiability_reason": None}
    result = check_rank_identifiability_contract(identifiable_report, rank_spec)
    assert isinstance(result, PostRunContractResult)
    assert result.satisfied is True

    non_identifiable_report = {
        "rank_identifiable": False,
        "rank_identifiability_reason": "primary_metric_zero_variance",
    }
    result_fail = check_rank_identifiability_contract(non_identifiable_report, rank_spec)
    assert result_fail.satisfied is False
    assert "rank not identifiable" in (result_fail.reason or "")


def test_campaign_runner_imports_contract_checker() -> None:
    """The campaign runner module exposes the post-run contract checker."""
    assert hasattr(campaign_runner, "check_rank_identifiability_contract")
    assert hasattr(campaign_runner, "write_rank_identifiability_report")
    assert callable(campaign_runner.check_rank_identifiability_contract)
    assert callable(campaign_runner.write_rank_identifiability_report)
