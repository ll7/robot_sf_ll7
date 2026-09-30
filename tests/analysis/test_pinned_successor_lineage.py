"""Actual frozen release configurations must resolve without rewriting their bytes."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.analysis.compare_release_0_0_7_to_0_0_8 import V4_SLOT_REPLACEMENTS

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
ORCA_HANDOFF_KEYS = (
    "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4",
    "scenario_adaptive_hybrid_orca_v2_collision_guard_v4",
)


def _resolve_real_handoff(key: str, mutation: str | None = None) -> subprocess.CompletedProcess:
    """Resolve real files, optionally mutating only their parsed in-memory config.

    The subprocess replaces the existing config loader, leaving all frozen bytes
    and production runtime resolution intact. No planner or environment steps run.
    """
    scenario = (
        "classic_bottleneck_low" if mutation == "other-scenario" else "francis2023_leave_group"
    )
    request = {
        "config_path": TEMPLATE,
        "versioned_keys": sorted(V4_SLOT_REPLACEMENTS.values()),
        "rows": [{"slot": [key, "differential_drive", scenario, 111, ""], "scenario_params": {}}],
    }
    script = """
import copy
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

root = Path.cwd()
sys.path[:0] = [str(root), str(root / 'fast-pysf')]
from robot_sf.benchmark.map_runner_policies import map_runner_policy_resolution as policies

key, mutation = sys.argv[1:]
original = policies._parse_algo_config

def load_config(path):
    config = original(path)
    if mutation and Path(path).name == key + '_s30_h600_release_0_0_8_frozen.yaml':
        config = copy.deepcopy(config)
        override = config['scenario_algo_overrides']['francis2023_leave_group']
        if mutation == 'other-scenario':
            config['scenario_algo_overrides']['classic_bottleneck_low'] = override
        elif mutation == 'other-config':
            override['base_config_path'] = 'configs/algos/social_navigation_pyenvs_orca_probe.yaml'
        elif mutation == 'goal':
            override['algo'] = 'goal'
        elif mutation == 'hybrid-base':
            override['algo'] = 'hybrid_rule_local_planner'
            override['base_config_path'] = 'configs/algos/hybrid_rule_v4_clearance_braking.yaml'
        elif mutation == 'non-v4':
            config.setdefault('scenario_overrides', {})['francis2023_leave_group'] = {
                'planner_variant': 'hybrid_rule_v3_teb_like_rollout'
            }
            del config['scenario_algo_overrides']['francis2023_leave_group']
        else:
            raise AssertionError(mutation)
    return config

with patch.object(policies, '_parse_algo_config', load_config):
    runpy.run_path(str(root / 'scripts/analysis/_pinned_successor_runtime.py'), run_name='__main__')
"""
    return subprocess.run(
        [sys.executable, "-I", "-c", script, key, mutation or ""],
        cwd=ROOT,
        input=json.dumps(request),
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_real_orca_handoff(key: str) -> None:
    """Check an independently specified identity and effective frozen parameters."""
    result = _resolve_real_handoff(key)
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["slot"] == [key, "differential_drive", "francis2023_leave_group", 111, ""]
    assert row["algo"] == "orca"
    assert "planner_variant" not in row["config"]
    assert row["config"]["max_linear_speed"] == 1.15
    assert row["config"]["orca_time_horizon"] == 5.0
    assert row["config"]["orca_obstacle_margin"] == 0.14
    assert row["config"]["provenance"]["issue"] == 707
    assert row["path"] == (
        f"configs/policy_search/candidates/{key}_s30_h600_release_0_0_8_frozen.yaml"
    )


@pytest.mark.parametrize("key", ORCA_HANDOFF_KEYS)
def test_real_frozen_orca_handoff_resolves(key: str):
    """Both approved scenario-adaptive slots must execute their real ORCA handoff."""
    _assert_real_orca_handoff(key)


@pytest.mark.parametrize("key", ORCA_HANDOFF_KEYS)
@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("other-scenario", "successor v4 slot resolves wrong algorithm"),
        ("other-config", "successor v4 slot resolves wrong algorithm"),
        ("goal", "successor v4 slot resolves wrong algorithm"),
        ("hybrid-base", "successor v4 slot has unapproved algorithm/base override"),
        ("non-v4", "successor planner row resolves non-v4 config"),
    ],
)
def test_real_orca_handoff_acceptance_keeps_substitutions_rejected(
    key: str, mutation: str, error: str
):
    """Accept the real handoff first, then reject one unauthorized runtime change.

    The positive precondition makes this acceptance/rejection boundary regression
    fail on the broken base; the exact negative reason guards permissive fixes.
    """
    _assert_real_orca_handoff(key)
    result = _resolve_real_handoff(key, mutation)
    assert result.returncode != 0
    assert error in result.stderr, result.stderr


def test_real_frozen_0_0_8_template_resolves_all_four_v4_lineages():
    """Exercise production pinned resolution with the real release matrix and frozen v4 files."""
    slots = [
        (key, "differential_drive", "classic_bottleneck_low", 111, "")
        for key in V4_SLOT_REPLACEMENTS.values()
    ]  # Seed 111 is a static join key; this resolver never steps an environment or planner.
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts/analysis/_pinned_successor_runtime.py")],
        cwd=ROOT,
        input=json.dumps(
            {
                "config_path": TEMPLATE,
                "versioned_keys": sorted(V4_SLOT_REPLACEMENTS.values()),
                "rows": [{"slot": slot, "scenario_params": {}} for slot in slots],
            }
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)
    assert len(resolved["expected_slots"]) == 20160  # 14 arms * 48 scenarios * 30 seeds.
    assert {tuple(row["slot"]) for row in resolved["rows"]} == set(slots)
    for row in resolved["rows"]:
        assert row["algo"] == "hybrid_rule_local_planner"
        assert row["config"]["planner_variant"] == "hybrid_rule_v4_clearance_braking"
        assert row["path"].endswith("_release_0_0_8_frozen.yaml")


@pytest.mark.parametrize(
    "foreign_module", ["robot_sf.baselines.ppo", "robot_sf.benchmark.map_runner.map_runner"]
)
def test_pinned_module_guard_covers_ppo_and_map_runner(tmp_path, monkeypatch, foreign_module):
    """An otherwise pinned resolver must reject either newly imported foreign module."""
    from types import SimpleNamespace

    from scripts.analysis._pinned_successor_runtime import _assert_pinned_modules

    # The complete independently specified import boundary, including the two
    # missing leaves. Fake origins isolate this guard without model/env setup.
    names = (
        "robot_sf.benchmark.camera_ready._util",
        "robot_sf.benchmark.camera_ready._config",
        "robot_sf.benchmark.camera_ready._preflight",
        "robot_sf.benchmark.runner",
        "robot_sf.benchmark.map_runner.map_runner_identity",
        "robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution",
        "robot_sf.benchmark.utils",
        "robot_sf.benchmark.algorithm_metadata",
        "robot_sf.benchmark.observation_noise",
        "robot_sf.benchmark.release_candidate",
        "robot_sf.benchmark.release_parameter_freeze",
        "pysocialforce",
        "robot_sf.baselines.ppo",
        "robot_sf.benchmark.map_runner.map_runner",
    )
    checkout = tmp_path / "pinned"
    for name in names:
        monkeypatch.setitem(
            sys.modules, name, SimpleNamespace(__file__=str(checkout / "module.py"))
        )
    _assert_pinned_modules(checkout)
    monkeypatch.setitem(
        sys.modules, foreign_module, SimpleNamespace(__file__=str(tmp_path / "foreign.py"))
    )
    with pytest.raises(
        ValueError, match=f"runtime module {foreign_module} did not load from pinned"
    ):
        _assert_pinned_modules(checkout)
