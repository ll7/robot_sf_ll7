"""Tests for the issue #5303 slice 1 cross-planner transfer matrix (cheap lane)."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.adversarial.transfer_matrix import (
    DEFAULT_TRANSFER_ROSTER,
    PlannerEval,
    build_transfer_matrix,
    render_transfer_report,
    select_certified_configs,
    write_transfer_artifact,
)


def _certified_candidate(start_x: float, *, seed: int, objective: float) -> dict:
    """Build a candidate payload using the real certification_status schema."""
    return {
        "candidate": {
            "start": {"x": start_x, "y": 2.0, "theta": 0.0},
            "goal": {"x": 5.0, "y": 2.0, "theta": 0.0},
            "spawn_time_s": 0.0,
            "pedestrian_speed_mps": 1.0,
            "pedestrian_delay_s": 0.0,
            "scenario_seed": seed,
        },
        "objective_value": objective,
        "bundle_path": f"output/adversarial/run/cand_{seed}",
        "scenario_yaml_path": "output/adversarial/run/cand_{seed}/scenario.yaml",
        "certification_status": {
            "schema_version": "scenario_cert.v1",
            "status": "passed",
            "details": {
                "certificates": [
                    {
                        "benchmark_eligibility": "eligible",
                        "classification": "hard_but_solvable",
                    }
                ]
            },
        },
    }


def _stress_only_candidate(start_x: float, *, seed: int, objective: float) -> dict:
    """Build a knife-edge (stress_only) certified candidate."""
    payload = _certified_candidate(start_x, seed=seed, objective=objective)
    payload["certification_status"]["details"]["certificates"][0]["benchmark_eligibility"] = (
        "stress_only"
    )
    payload["certification_status"]["details"]["certificates"][0]["classification"] = "knife_edge"
    return payload


def _uncertified_candidate(start_x: float, *, seed: int, objective: float) -> dict:
    """Build an excluded/infeasible candidate that must NOT be selected."""
    payload = _certified_candidate(start_x, seed=seed, objective=objective)
    payload["certification_status"]["details"]["certificates"][0]["benchmark_eligibility"] = (
        "excluded"
    )
    payload["certification_status"]["details"]["certificates"][0]["classification"] = (
        "geometrically_infeasible"
    )
    return payload


def _manifest(tmp_path: Path, *, name: str, candidates: list[dict]) -> Path:
    """Write a synthetic adversarial search manifest (real schema)."""
    payload = {
        "schema_version": "adversarial-search-manifest.v1",
        "config": {
            "policy": "scenario_adaptive_hybrid_orca_v1",
            "scenario_template": "configs/scenarios/templates/crossing_ttc.yaml",
            "search_space": {
                "variables": {
                    "start_x": {"min": 0.0, "max": 4.0},
                    "scenario_seed": {"min": 700, "max": 800},
                }
            },
        },
        "candidates": candidates,
    }
    path = tmp_path / name
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    return path


def _evals_for_configs(configs, *, planner_robustness, failed):
    """Build per-planner eval results for every config/planner pair."""
    evals = []
    for cfg in configs:
        for planner in DEFAULT_TRANSFER_ROSTER:
            if planner == cfg.target_planner:
                continue
            evals.append(
                PlannerEval(
                    config_id=cfg.config_id,
                    planner=planner,
                    robustness=planner_robustness,
                    failed=failed,
                    seed=cfg.scenario_seed,
                )
            )
    return evals


def test_select_certified_configs_keeps_only_certified(tmp_path):
    m = _manifest(
        tmp_path,
        name="m1.json",
        candidates=[
            _uncertified_candidate(0.1, seed=701, objective=20.0),  # must be excluded
            _certified_candidate(0.2, seed=702, objective=9.0),
            _stress_only_candidate(0.3, seed=703, objective=15.0),
        ],
    )
    configs = select_certified_configs([m], target_planner="orca_v1", K=10)
    assert len(configs) == 2
    # Sorted worst-first: objective 15 before 9.
    assert configs[0].objective_value == 15.0
    assert configs[0].certification_tier == "stress_only"
    assert all(c.target_planner == "orca_v1" for c in configs)


def test_select_certified_configs_respects_k(tmp_path):
    candidates = [
        _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(8)
    ]
    m = _manifest(tmp_path, name="m.json", candidates=candidates)
    configs = select_certified_configs([m], target_planner="orca_v1", K=5)
    assert len(configs) == 5
    # Largest objectives kept (worst-first).
    assert [c.objective_value for c in configs] == [7.0, 6.0, 5.0, 4.0, 3.0]


def test_select_requires_real_manifest_schema(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(__import__("json").dumps({"schema_version": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError):
        select_certified_configs([bad], target_planner="orca_v1", K=5)


def test_build_transfer_matrix_structure_and_ranking(tmp_path):
    m = _manifest(
        tmp_path,
        name="m.json",
        candidates=[
            _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(6)
        ],
    )
    configs = select_certified_configs([m], target_planner="orca_v1", K=6)
    # All non-target planners fail => full transfer.
    evals = _evals_for_configs(configs, planner_robustness=-1.0, failed=True)
    matrix = build_transfer_matrix(configs, evals)
    assert matrix.target_planner == "orca_v1"
    assert len(matrix.config_ids) == 6
    assert len(matrix.planners) == 3
    assert len(matrix.cells) == 6 * 3
    assert matrix.overall_transfer_rate == 1.0
    assert matrix.transfer_rate_ci == (1.0, 1.0)
    # Worst-case robustness equal across planners => stable deterministic ranking.
    assert matrix.ranking[0].worst_case_robustness == -1.0


def test_build_transfer_matrix_no_transfer(tmp_path):
    m = _manifest(
        tmp_path,
        name="m.json",
        candidates=[
            _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(5)
        ],
    )
    configs = select_certified_configs([m], target_planner="orca_v1", K=5)
    # Non-target planners succeed => weak points are policy-specific.
    evals = _evals_for_configs(configs, planner_robustness=2.0, failed=False)
    matrix = build_transfer_matrix(configs, evals)
    assert matrix.overall_transfer_rate == 0.0
    assert matrix.transfer_rate_ci == (0.0, 0.0)
    assert all(not c.transferred for c in matrix.cells if c.planner != "orca_v1")


def test_build_requires_k_at_least_5(tmp_path):
    m = _manifest(
        tmp_path,
        name="m.json",
        candidates=[_certified_candidate(0.1, seed=700, objective=1.0)],
    )
    configs = select_certified_configs([m], target_planner="orca_v1", K=1)
    evals = _evals_for_configs(configs, planner_robustness=2.0, failed=False)
    with pytest.raises(ValueError):
        build_transfer_matrix(configs, evals)


def test_bootstrap_ci_covers_point_estimate(tmp_path):
    m = _manifest(
        tmp_path,
        name="m.json",
        candidates=[
            _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(8)
        ],
    )
    configs = select_certified_configs([m], target_planner="orca_v1", K=8)
    # Mixed: half transfer, half not.
    evals = []
    for idx, cfg in enumerate(configs):
        for planner in DEFAULT_TRANSFER_ROSTER:
            if planner == cfg.target_planner:
                continue
            failed = (idx % 2) == 0
            evals.append(
                PlannerEval(
                    config_id=cfg.config_id,
                    planner=planner,
                    robustness=-1.0 if failed else 2.0,
                    failed=failed,
                    seed=cfg.scenario_seed,
                )
            )
    matrix = build_transfer_matrix(configs, evals, bootstrap_n=500, bootstrap_seed=7)
    lo, hi = matrix.transfer_rate_ci
    assert lo <= matrix.overall_transfer_rate <= hi
    assert matrix.transfer_rate_bootstrap_n == 500


def test_render_report_contains_markers_and_ranking(tmp_path):
    m = _manifest(
        tmp_path,
        name="m.json",
        candidates=[
            _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(6)
        ],
    )
    configs = select_certified_configs([m], target_planner="orca_v1", K=6)
    evals = _evals_for_configs(configs, planner_robustness=-1.0, failed=True)
    matrix = build_transfer_matrix(configs, evals)
    report = render_transfer_report(matrix, configs=configs)
    assert "capability-only" in report
    assert "minimax" in report.lower() or "Minimax" in report
    assert "X" in report  # transferred failure marker
    assert "Transfer matrix" in report


def test_write_artifact_roundtrip(tmp_path):
    m = _manifest(
        tmp_path,
        name="m.json",
        candidates=[
            _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(6)
        ],
    )
    configs = select_certified_configs([m], target_planner="orca_v1", K=6)
    evals = _evals_for_configs(configs, planner_robustness=-1.0, failed=True)
    matrix = build_transfer_matrix(configs, evals)
    out = tmp_path / "out"
    path = write_transfer_artifact(matrix, out_dir=out)
    assert path.exists()
    reloaded = __import__("json").loads(path.read_text())
    assert reloaded["schema_version"] == "adversarial_transfer_matrix.v1"
    assert len(reloaded["cells"]) == 18
    assert (out / "transfer_report.md").exists()


def test_default_roster_has_three_planners():
    assert len(DEFAULT_TRANSFER_ROSTER) == 3
    assert DEFAULT_TRANSFER_ROSTER[0] == "scenario_adaptive_hybrid_orca_v1"


def test_transfer_matrix_is_frozen_and_jsonable(tmp_path):
    m = _manifest(
        tmp_path,
        name="m.json",
        candidates=[
            _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(5)
        ],
    )
    configs = select_certified_configs([m], target_planner="orca_v1", K=5)
    evals = _evals_for_configs(configs, planner_robustness=-1.0, failed=True)
    matrix = build_transfer_matrix(configs, evals)
    payload = matrix.to_json()
    assert isinstance(payload, dict)
    assert payload["schema_version"] == "adversarial_transfer_matrix.v1"
