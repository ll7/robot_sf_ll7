"""Behavior lock for in-process vs subprocess arm parity (issue #5299).

This is the REAL-subprocess end-to-end companion to the mocked-subprocess tests
in ``test_camera_ready_subprocess_isolation.py``. PR #5270 fixed the third silent
divergence between the in-process and subprocess execution paths: the subprocess
worker used to re-load the raw scenario matrix from disk, losing the campaign
loader's ``map_file`` normalization (relative paths, seed overrides, candidate
filtering, horizon schedules, holonomic_command_mode). On Slurm jobs 13372/13373
that silently dropped 147/147 episodes. Earlier fixes (#5239, #4957) shipped
without any test that compares the two paths end-to-end.

The structural contract now exists (parent serializes ``scoped_scenarios.json``,
worker consumes verbatim), but nothing locked it behaviorally: a future transform
added to only one path would regress silently again. This test is that
behavioral lock. It runs a minimal 1-scenario / 1-seed / 2-arm campaign via the
production ``run_campaign`` entry point twice — once with
``arm_isolation="in_process"`` and once with ``arm_isolation="subprocess"`` —
using a stub ``goal`` planner (no checkpoints, no GPU), and asserts per arm:

* identical *ordered* ``(scenario_id, seed)`` pairs in ``runs/*/episodes.jsonl``
* identical episode counts and identical ``resolved_seeds`` in
  ``matrix_summary.json``
* both campaign summaries report ``total_episodes > 0`` (guards the
  0-episode failure class that the #5270 regression caused)

The scenario fixture deliberately uses a *relative* ``map_file`` resolved
relative to the scenario matrix file — that relativity is the exact regression
this test locks: the in-process path normalizes it via the campaign loader, and
the subprocess path must execute the same normalized scenario rather than
re-reading the raw matrix.

Runs CPU-only in well under 120s; not marked ``slow`` so it stays under the
``-m "not slow"`` gate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.camera_ready._config import load_campaign_config
from robot_sf.benchmark.camera_ready.campaign import run_campaign
from robot_sf.common.artifact_paths import get_repository_root

# A tiny stub planner needs no checkpoint and no GPU; the ``goal`` baseline is
# baseline-ready in the readiness registry. ``horizon`` / ``max_episode_steps``
# are kept minimal so the whole two-campaign run stays well under the timeout.
_STUB_ALGO = "goal"
_FIXED_SEED = 1080
_MIN_HORIZON = 30
_MIN_MAX_EPISODE_STEPS = 30

# Real map used by the planner-sanity scenarios; relative ``map_file`` paths
# resolved from the scenario matrix file must land on this file.
_RELATIVE_MAP_FILE = "maps/svg_maps/planner_sanity_open.svg"


def _scenario_matrix_yaml(*, relative_map_file: str) -> str:
    """Return a minimal 1-scenario matrix YAML exercising a relative map_file."""
    return (
        "# Minimal scenario matrix for the arm-isolation parity lock (issue #5299).\n"
        "# The map_file is intentionally a relative path resolved from this\n"
        "# matrix file's location; that relativity is the regression this locks.\n"
        "scenarios:\n"
        "  - name: arm_parity_single\n"
        f"    map_file: {relative_map_file}\n"
        "    simulation_config:\n"
        f"      max_episode_steps: {_MIN_MAX_EPISODE_STEPS}\n"
        "      ped_density: 0.0\n"
        "    single_pedestrians: []\n"
        "    robot_config: {}\n"
        "    metadata:\n"
        "      archetype: arm_parity\n"
        "      flow: none\n"
        "      behavior: none\n"
        "      purpose: arm_isolation_parity_lock\n"
        "    seeds:\n"
        f"      - {_FIXED_SEED}\n"
    )


def _campaign_config_yaml(*, scenario_matrix_rel: str) -> str:
    """Return a minimal 2-arm campaign config YAML using the stub goal planner."""
    return (
        "name: arm_isolation_parity_smoke\n"
        "paper_facing: false\n"
        f"scenario_matrix: {scenario_matrix_rel}\n"
        "\n"
        "seed_policy:\n"
        "  mode: fixed-list\n"
        f"  seeds: [{_FIXED_SEED}]\n"
        "\n"
        "workers: 1\n"
        f"horizon: {_MIN_HORIZON}\n"
        "dt: 0.1\n"
        "record_forces: false\n"
        "resume: true\n"
        "stop_on_failure: false\n"
        "bootstrap_samples: 50\n"
        "bootstrap_confidence: 0.95\n"
        "bootstrap_seed: 1080\n"
        "\n"
        "kinematics_matrix:\n"
        "  - differential_drive\n"
        "\n"
        "planners:\n"
        "  - key: goal_a\n"
        f"    algo: {_STUB_ALGO}\n"
        "    planner_group: core\n"
        "    benchmark_profile: baseline-safe\n"
        "\n"
        "  - key: goal_b\n"
        f"    algo: {_STUB_ALGO}\n"
        "    planner_group: core\n"
        "    benchmark_profile: baseline-safe\n"
    )


def _write_parity_campaign_config(tmp_path: Path) -> Path:
    """Write a minimal scenario matrix + campaign config and return the config path.

    The scenario matrix's ``map_file`` is a path relative to the matrix file so
    the campaign loader's relative-map_file normalization is exercised by both
    execution paths (the exact code the #5270 regression bypassed in the worker).
    """
    repo_root = Path(get_repository_root()).resolve()
    map_abs = repo_root / _RELATIVE_MAP_FILE
    assert map_abs.exists(), (
        f"Arm-isolation parity fixture requires map at {_RELATIVE_MAP_FILE} "
        f"(resolved to {map_abs}); it must exist in the repository."
    )
    matrix_path = tmp_path / "arm_parity_matrix.yaml"
    # Path relative to the matrix file's location, to exercise relative resolution.
    relative_map_file = os.path.relpath(map_abs, matrix_path.parent)
    matrix_path.write_text(_scenario_matrix_yaml(relative_map_file=relative_map_file), "utf-8")

    config_path = tmp_path / "arm_parity_campaign.yaml"
    config_path.write_text(
        _campaign_config_yaml(scenario_matrix_rel=matrix_path.name),
        "utf-8",
    )
    return config_path


def _arm_dir_name(planner_key: str, kinematics: str = "differential_drive") -> str:
    """Return the run directory name for a planner/kinematics arm.

    Mirrors ``_prepare_campaign_planner_variant_run`` in campaign.py, which builds
    ``{_sanitize_name(planner.key)}__{_sanitize_name(kinematics)}``.
    """
    from robot_sf.benchmark.camera_ready._config import _sanitize_name

    return f"{_sanitize_name(planner_key)}__{_sanitize_name(kinematics)}"


def _read_episode_pairs(episodes_path: Path) -> list[tuple[str, int]]:
    """Return ordered ``(scenario_id, seed)`` pairs from an episodes.jsonl file."""
    if not episodes_path.exists():
        return []
    pairs: list[tuple[str, int]] = []
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        scenario_id = str(record["scenario_id"])
        seed = int(record["seed"])
        pairs.append((scenario_id, seed))
    return pairs


def _load_matrix_summary_rows(campaign_root: Path) -> dict[str, dict[str, Any]]:
    """Return per-arm rows from matrix_summary.json keyed by planner_key."""
    summary_path = campaign_root / "reports" / "matrix_summary.json"
    assert summary_path.exists(), f"matrix_summary.json missing at {summary_path}"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    by_arm: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['planner_key']}__{row['kinematics']}"
        by_arm[key] = row
    return by_arm


def _arm_keys_from_config(cfg) -> list[str]:
    """Return the expected arm dir names for the campaign config's enabled planners."""
    return [
        _arm_dir_name(planner.key, kinematics)
        for planner in cfg.planners
        if planner.enabled
        for kinematics in cfg.kinematics_matrix
    ]


def _run_campaign_for_arm_isolation(cfg, output_root: Path, arm_isolation: str) -> dict[str, Any]:
    """Run one campaign in the given isolation mode and return its result dict.

    The result carries ``campaign_root`` and ``total_episodes``; callers derive
    the campaign root and assert the episode count from this single source.
    """
    result = run_campaign(
        cfg,
        output_root=output_root,
        arm_isolation=arm_isolation,
        skip_publication_bundle=True,
    )
    campaign_root = Path(result["campaign_root"])
    assert campaign_root.exists(), f"campaign_root missing: {campaign_root}"
    return result


@pytest.mark.timeout(120)
def test_in_process_and_subprocess_arms_execute_identical_episode_set(tmp_path: Path):
    """Both isolation modes must execute exactly the same (scenario_id, seed) set.

    This is the behavior lock for the parent->worker scenario handoff fixed in
    #5270 (and #5239 / #4957 before it). Runs the same minimal 2-arm campaign
    through the production ``run_campaign`` in both ``in_process`` and
    ``subprocess`` isolation modes and asserts per-arm parity of the ordered
    episode identity pairs, the episode counts, and ``resolved_seeds`` in
    ``matrix_summary.json``. Fails — rather than silently passing — if either
    mode yields 0 episodes (the exact failure class the #5270 regression caused
    on Slurm jobs 13372/13373).
    """
    config_path = _write_parity_campaign_config(tmp_path)
    cfg_in_process = load_campaign_config(config_path)
    expected_arms = _arm_keys_from_config(cfg_in_process)
    assert len(expected_arms) == 2, (
        f"Parity test expects a 2-arm campaign, got arms={expected_arms}"
    )

    in_process_result = _run_campaign_for_arm_isolation(
        cfg_in_process, output_root=tmp_path / "in_process", arm_isolation="in_process"
    )
    cfg_subprocess = load_campaign_config(config_path)
    subprocess_result = _run_campaign_for_arm_isolation(
        cfg_subprocess, output_root=tmp_path / "subprocess", arm_isolation="subprocess"
    )
    in_process_root = Path(in_process_result["campaign_root"])
    subprocess_root = Path(subprocess_result["campaign_root"])

    in_matrix = _load_matrix_summary_rows(in_process_root)
    sub_matrix = _load_matrix_summary_rows(subprocess_root)

    # Per-arm parity assertions.
    for arm in expected_arms:
        in_episodes = _read_episode_pairs(in_process_root / "runs" / arm / "episodes.jsonl")
        sub_episodes = _read_episode_pairs(subprocess_root / "runs" / arm / "episodes.jsonl")

        # Guard the 0-episode failure class specifically (issue #5279: 147/147
        # episodes dropped). Both modes must produce a non-empty, identical set.
        assert in_episodes, (
            f"arm '{arm}' produced 0 episodes in_process (expected >0); "
            "this is the 0-episode failure class the parity lock must catch"
        )
        assert sub_episodes, (
            f"arm '{arm}' produced 0 episodes in subprocess (expected >0); "
            "this is the 0-episode failure class the parity lock must catch"
        )

        # Identical *ordered* (scenario_id, seed) pairs.
        assert in_episodes == sub_episodes, (
            f"arm '{arm}' episode (scenario_id, seed) pairs differ between "
            f"in_process and subprocess:\n"
            f"  in_process={in_episodes}\n"
            f"  subprocess={sub_episodes}"
        )

        # Identical episode counts and resolved_seeds in matrix_summary.json.
        assert arm in in_matrix, (
            f"arm '{arm}' missing from in_process matrix summary; available={list(in_matrix)}"
        )
        assert arm in sub_matrix, (
            f"arm '{arm}' missing from subprocess matrix summary; available={list(sub_matrix)}"
        )
        in_row = in_matrix[arm]
        sub_row = sub_matrix[arm]
        assert in_row["resolved_seeds"] == sub_row["resolved_seeds"], (
            f"arm '{arm}' resolved_seeds differ: "
            f"in_process={in_row['resolved_seeds']} subprocess={sub_row['resolved_seeds']}"
        )
        assert in_row["scenario_count"] == sub_row["scenario_count"], (
            f"arm '{arm}' scenario_count differs: "
            f"in_process={in_row['scenario_count']} subprocess={sub_row['scenario_count']}"
        )
        assert in_row["seed_policy.mode"] == sub_row["seed_policy.mode"], (
            f"arm '{arm}' seed_policy.mode differs: "
            f"in_process={in_row['seed_policy.mode']} "
            f"subprocess={sub_row['seed_policy.mode']}"
        )

        # The normalized map_file must be identical and repo-relative in both
        # paths (the #5270 regression left the worker with an unresolvable
        # relative map_file). The relative fixture path resolves to the same
        # repo-relative canonical path under both modes.
        assert in_row["scenario_matrix"] == sub_row["scenario_matrix"], (
            f"arm '{arm}' scenario_matrix differs between modes"
        )

    # Both campaign-level summaries report total_episodes > 0. This is the
    # campaign-result counter; the 0-episode failure class (#5270 dropped
    # 147/147 episodes) is guarded both here and per-arm above.
    assert int(in_process_result.get("total_episodes", 0)) > 0, (
        "in_process campaign total_episodes must be >0, got "
        f"{in_process_result.get('total_episodes')}"
    )
    assert int(subprocess_result.get("total_episodes", 0)) > 0, (
        "subprocess campaign total_episodes must be >0, got "
        f"{subprocess_result.get('total_episodes')}"
    )

    # Sanity: the two arms are genuinely distinct run dirs and each ran the
    # stub planner against the single expected (scenario_id, seed) pair.
    expected_pair = ("arm_parity_single", _FIXED_SEED)
    for arm in expected_arms:
        in_episodes = _read_episode_pairs(in_process_root / "runs" / arm / "episodes.jsonl")
        sub_episodes = _read_episode_pairs(subprocess_root / "runs" / arm / "episodes.jsonl")
        assert in_episodes == [expected_pair] == sub_episodes, (
            f"arm '{arm}' did not execute exactly the expected single pair "
            f"{expected_pair}: in_process={in_episodes}, subprocess={sub_episodes}"
        )
