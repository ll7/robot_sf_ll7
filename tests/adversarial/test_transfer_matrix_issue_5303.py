"""Tests for the issue #5303 slice 1 cross-planner transfer matrix (cheap lane)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from robot_sf.adversarial.transfer_matrix import (
    DEFAULT_TRANSFER_ROSTER,
    PlannerEval,
    build_gate_a_transfer_matrix,
    build_transfer_matrix,
    check_issue_6145_activation,
    render_transfer_report,
    select_certified_configs,
    write_transfer_artifact,
)

_TARGET_PLANNER = DEFAULT_TRANSFER_ROSTER[0]


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
            "primary_mechanism": "collision",
        },
        "objective_value": objective,
        "bundle_path": f"output/adversarial/run/cand_{seed}",
        "scenario_yaml_path": f"output/adversarial/run/cand_{seed}/scenario.yaml",
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


def _manifest(
    tmp_path: Path,
    *,
    name: str,
    candidates: list[dict],
    policy: str = _TARGET_PLANNER,
) -> Path:
    """Write a synthetic adversarial search manifest (real schema)."""
    payload = {
        "schema_version": "adversarial-search-manifest.v1",
        "config": {
            "policy": policy,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _evals_for_configs(configs, *, planner_robustness, failed, mechanism="collision"):
    """Build per-planner eval results for every config/planner pair."""
    evals = []
    for cfg in configs:
        for planner in DEFAULT_TRANSFER_ROSTER:
            evals.append(
                PlannerEval(
                    config_id=cfg.config_id,
                    planner=planner,
                    robustness=planner_robustness,
                    failed=failed,
                    seed=cfg.scenario_seed,
                    eval_seed=cfg.scenario_seed,
                    mechanism=mechanism,
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
    configs = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=10)
    assert len(configs) == 2
    # Sorted worst-first: objective 15 before 9.
    assert configs[0].objective_value == 15.0
    assert configs[0].certification_tier == "stress_only"
    assert all(c.target_planner == _TARGET_PLANNER for c in configs)


def test_select_certified_configs_respects_k(tmp_path):
    candidates = [
        _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(8)
    ]
    m = _manifest(tmp_path, name="m.json", candidates=candidates)
    configs = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=5)
    assert len(configs) == 5
    # Largest objectives kept (worst-first).
    assert [c.objective_value for c in configs] == [7.0, 6.0, 5.0, 4.0, 3.0]


def test_select_requires_real_manifest_schema(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError):
        select_certified_configs([bad], target_planner=_TARGET_PLANNER, K=5)


def test_build_transfer_matrix_structure_and_ranking(tmp_path):
    m = _manifest(
        tmp_path,
        name="m.json",
        candidates=[
            _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(6)
        ],
    )
    configs = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=6)
    # All non-target planners fail => full transfer.
    evals = _evals_for_configs(configs, planner_robustness=-1.0, failed=True)
    matrix = build_transfer_matrix(configs, evals)
    assert matrix.target_planner == _TARGET_PLANNER
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
    configs = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=5)
    # Non-target planners succeed => weak points are policy-specific.
    evals = _evals_for_configs(configs, planner_robustness=2.0, failed=False)
    matrix = build_transfer_matrix(configs, evals)
    assert matrix.overall_transfer_rate == 0.0
    assert matrix.transfer_rate_ci == (0.0, 0.0)
    assert all(not c.transferred for c in matrix.cells if c.planner != _TARGET_PLANNER)


def test_build_requires_k_at_least_5(tmp_path):
    m = _manifest(
        tmp_path,
        name="m.json",
        candidates=[_certified_candidate(0.1, seed=700, objective=1.0)],
    )
    configs = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=1)
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
    configs = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=8)
    # Mixed: half transfer, half not.
    evals = []
    for idx, cfg in enumerate(configs):
        for planner in DEFAULT_TRANSFER_ROSTER:
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
    configs = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=6)
    evals = _evals_for_configs(configs, planner_robustness=-1.0, failed=True)
    matrix = build_transfer_matrix(configs, evals)
    report = render_transfer_report(matrix, configs=configs)
    assert "capability-only" in report
    assert "Capability-only ranking" in report
    assert "minimax" not in report.lower()
    assert "regret" not in report.lower()
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
    configs = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=6)
    evals = _evals_for_configs(configs, planner_robustness=-1.0, failed=True)
    matrix = build_transfer_matrix(configs, evals)
    out = tmp_path / "out"
    path = write_transfer_artifact(matrix, out_dir=out)
    assert path.exists()
    reloaded = json.loads(path.read_text())
    assert reloaded["schema_version"] == "adversarial_transfer_matrix.v2"
    assert reloaded["capability_only"] is True
    assert len(reloaded["configs"]) == 6
    assert all(config["scenario_seed"] is not None for config in reloaded["configs"])
    assert len(reloaded["cells"]) == 18
    assert (out / "transfer_report.md").exists()
    report = (out / "transfer_report.md").read_text(encoding="utf-8")
    assert "Certified config provenance" in report
    assert configs[0].source_manifest in report


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
    configs = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=5)
    evals = _evals_for_configs(configs, planner_robustness=-1.0, failed=True)
    matrix = build_transfer_matrix(configs, evals)
    payload = matrix.to_json()
    assert isinstance(payload, dict)
    assert payload["schema_version"] == "adversarial_transfer_matrix.v2"
    assert payload["capability_only"] is True


def test_ranking_places_best_worst_case_robustness_first(tmp_path: Path) -> None:
    """Minimax rank 1 must mean the strongest, not the most brittle, planner."""
    manifest = _manifest(
        tmp_path,
        name="ranking.json",
        candidates=[
            _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(5)
        ],
    )
    configs = select_certified_configs([manifest], target_planner=_TARGET_PLANNER, K=5)
    robustness_by_planner = {
        _TARGET_PLANNER: -2.0,
        DEFAULT_TRANSFER_ROSTER[1]: -0.5,
        DEFAULT_TRANSFER_ROSTER[2]: 1.0,
    }
    evaluations = [
        PlannerEval(
            config_id=config.config_id,
            planner=planner,
            robustness=robustness,
            failed=robustness < 0.0,
            seed=config.scenario_seed,
        )
        for config in configs
        for planner, robustness in robustness_by_planner.items()
    ]
    matrix = build_transfer_matrix(configs, evaluations)
    assert [row.planner for row in matrix.ranking] == [
        DEFAULT_TRANSFER_ROSTER[2],
        DEFAULT_TRANSFER_ROSTER[1],
        _TARGET_PLANNER,
    ]


def test_matrix_rejects_incomplete_duplicate_and_nonfinite_evaluations(tmp_path: Path) -> None:
    """Bad evaluation tables must fail rather than dilute missing runs into successes."""
    manifest = _manifest(
        tmp_path,
        name="matrix.json",
        candidates=[
            _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(5)
        ],
    )
    configs = select_certified_configs([manifest], target_planner=_TARGET_PLANNER, K=5)
    evaluations = _evals_for_configs(configs, planner_robustness=-1.0, failed=True)
    with pytest.raises(ValueError, match="incomplete"):
        build_transfer_matrix(configs, evaluations[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        build_transfer_matrix(configs, [*evaluations, evaluations[0]])
    nonfinite = [*evaluations]
    nonfinite[0] = replace(nonfinite[0], robustness=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        build_transfer_matrix(configs, nonfinite)
    inconsistent = [*evaluations]
    inconsistent[0] = replace(inconsistent[0], failed=False)
    with pytest.raises(ValueError, match="disagrees"):
        build_transfer_matrix(configs, inconsistent)


def test_selection_preserves_unique_lineage_and_uses_conservative_certification(
    tmp_path: Path,
) -> None:
    """Common manifest names stay unique and any excluded certificate blocks selection."""
    eligible = _certified_candidate(0.1, seed=701, objective=2.0)
    mixed = _certified_candidate(0.2, seed=702, objective=9.0)
    mixed["certification_status"]["details"]["certificates"].append(
        {"benchmark_eligibility": "excluded", "classification": "invalid"}
    )
    nonfinite = _certified_candidate(0.3, seed=703, objective=float("nan"))
    first = _manifest(tmp_path, name="run-a/manifest.json", candidates=[eligible, mixed, nonfinite])
    second = _manifest(
        tmp_path,
        name="run-b/manifest.json",
        candidates=[_certified_candidate(0.4, seed=704, objective=3.0)],
    )
    configs = select_certified_configs([first, second], target_planner=_TARGET_PLANNER, K=10)
    assert len(configs) == 2
    assert len({config.config_id for config in configs}) == 2
    assert all(config.source_manifest in config.config_id for config in configs)


def test_lineage_and_matrix_configuration_fail_closed(tmp_path: Path) -> None:
    """Target, roster, seed, and bootstrap settings must preserve the slice contract."""
    manifest = _manifest(
        tmp_path,
        name="config.json",
        candidates=[
            _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(5)
        ],
    )
    with pytest.raises(ValueError, match="Target planner mismatch"):
        select_certified_configs([manifest], target_planner="wrong-planner", K=5)
    configs = select_certified_configs([manifest], target_planner=_TARGET_PLANNER, K=5)
    evaluations = _evals_for_configs(configs, planner_robustness=-1.0, failed=True)
    with pytest.raises(ValueError, match="target planner plus 2"):
        build_transfer_matrix(configs, evaluations, planners=DEFAULT_TRANSFER_ROSTER[:2])
    with pytest.raises(ValueError, match="unique"):
        build_transfer_matrix(
            configs,
            evaluations,
            planners=(_TARGET_PLANNER, DEFAULT_TRANSFER_ROSTER[1], DEFAULT_TRANSFER_ROSTER[1]),
        )
    with pytest.raises(ValueError, match="bootstrap_n"):
        build_transfer_matrix(configs, evaluations, bootstrap_n=-1)


def _gate_a_manifest(tmp_path: Path, *, name: str, candidates: list[dict]) -> Path:
    """Write a synthetic manifest and return its path for Gate A tests."""
    return _manifest(tmp_path, name=name, candidates=candidates)


def _gate_a_evals(configs, *, robustness_by_planner=None, mechanism="collision"):
    """Build Gate A eval rows with explicit eval_seed and mechanism."""
    if robustness_by_planner is None:
        robustness_by_planner = dict.fromkeys(DEFAULT_TRANSFER_ROSTER, -1.0)
    evals = []
    for cfg in configs:
        for planner, robustness in robustness_by_planner.items():
            evals.append(
                PlannerEval(
                    config_id=cfg.config_id,
                    planner=planner,
                    robustness=robustness,
                    failed=robustness < 0.0,
                    seed=cfg.scenario_seed,
                    eval_seed=cfg.scenario_seed,
                    mechanism=mechanism,
                )
            )
    return evals


def test_gate_a_select_rejects_stress_only(tmp_path):
    """Gate A selection must drop stress_only candidates from the matrix."""
    m = _manifest(
        tmp_path,
        name="m.json",
        candidates=[
            _certified_candidate(0.1, seed=701, objective=10.0),
            _stress_only_candidate(0.2, seed=702, objective=20.0),
        ],
    )
    legacy = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=10)
    assert any(c.certification_tier == "stress_only" for c in legacy)
    gate_a = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=10, eligible_only=True)
    assert len(gate_a) == 1
    assert gate_a[0].certification_tier == "eligible"


def test_gate_a_select_rejects_excluded_row_classes(tmp_path):
    """Gate A must reject fallback/degraded/unavailable/duplicate/pre-correction rows."""

    def _excluded_candidate(start_x, *, seed, objective, classification):
        payload = _certified_candidate(start_x, seed=seed, objective=objective)
        payload["certification_status"]["details"]["certificates"][0]["classification"] = (
            classification
        )
        return payload

    m = _manifest(
        tmp_path,
        name="m.json",
        candidates=[
            _certified_candidate(0.1, seed=701, objective=10.0),
            _excluded_candidate(0.2, seed=702, objective=20.0, classification="fallback"),
            _excluded_candidate(0.3, seed=703, objective=30.0, classification="degraded"),
            _excluded_candidate(0.4, seed=704, objective=40.0, classification="duplicate"),
            _excluded_candidate(0.5, seed=705, objective=50.0, classification="pre_correction"),
        ],
    )
    gate_a = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=10, eligible_only=True)
    assert len(gate_a) == 1


def test_gate_a_builds_immutable_rows_and_clusters(tmp_path):
    """Gate A matrix must contain one row per config x planner x fresh seed and per-candidate clusters."""
    m = _manifest(
        tmp_path,
        name="m.json",
        candidates=[
            _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(5)
        ],
    )
    configs = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=5, eligible_only=True)
    evals = _gate_a_evals(configs)
    matrix = build_gate_a_transfer_matrix(configs, evals)
    assert matrix.capability_only is True
    assert len(matrix.rows) == 5 * 3
    assert len(matrix.clusters) == 5
    for row in matrix.rows:
        assert row.lineage_complete is True
        assert row.mechanism_retained is True
    for cluster in matrix.clusters:
        assert cluster.n_evaluated_seeds == 3
        assert cluster.n_failed == 3
        assert cluster.n_transferred == 2


def test_gate_a_rejects_opposite_mechanism(tmp_path):
    """A row whose observed mechanism differs from the predeclared mechanism must fail retention."""
    m = _manifest(
        tmp_path,
        name="m.json",
        candidates=[
            _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(5)
        ],
    )
    for candidate in m.read_text():  # type: ignore[union-attr]
        pass
    configs = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=5, eligible_only=True)
    configs[0] = replace(configs[0], primary_mechanism="collision")
    evals = _gate_a_evals(configs, mechanism="collision")
    # Change only the first eval's observed mechanism to the opposite.
    evals[0] = replace(evals[0], mechanism="opposite_mechanism")
    matrix = build_gate_a_transfer_matrix(configs, evals)
    opposite_rows = [r for r in matrix.rows if r.observed_mechanism == "opposite_mechanism"]
    assert opposite_rows
    assert not opposite_rows[0].mechanism_retained
    cluster = next(c for c in matrix.clusters if c.config_id == configs[0].config_id)
    assert cluster.mechanism_retained is False


def test_gate_a_rejects_repeated_eval_seed(tmp_path):
    """Two rows with the same config/planner/eval_seed must fail closed."""
    m = _manifest(
        tmp_path,
        name="m.json",
        candidates=[
            _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(5)
        ],
    )
    configs = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=5, eligible_only=True)
    evals = _gate_a_evals(configs)
    # Duplicate the first eval row to simulate a repeated seed.
    evals.append(evals[0])
    with pytest.raises(ValueError, match="Duplicate evaluation"):
        build_gate_a_transfer_matrix(configs, evals)


def test_gate_a_rejects_missing_eval_seed(tmp_path):
    """A config/planner pair without a fresh-seed row must fail closed."""
    m = _manifest(
        tmp_path,
        name="m.json",
        candidates=[
            _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(5)
        ],
    )
    configs = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=5, eligible_only=True)
    evals = _gate_a_evals(configs)
    # Drop one eval to create a missing seed / incomplete lineage.
    evals = evals[1:]
    with pytest.raises(ValueError, match="incomplete"):
        build_gate_a_transfer_matrix(configs, evals)


def test_gate_a_misleading_ranking_is_capability_only(tmp_path):
    """A high transfer-failure rate must not be reported as a minimax regret claim."""
    m = _manifest(
        tmp_path,
        name="m.json",
        candidates=[
            _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(5)
        ],
    )
    configs = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=5, eligible_only=True)
    evals = _gate_a_evals(configs)
    matrix = build_gate_a_transfer_matrix(configs, evals)
    report = render_transfer_report(matrix, configs=configs)
    assert "Capability-only ranking" in report
    assert "minimax" not in report.lower()
    assert "regret" not in report.lower()


def test_gate_a_rejects_missing_lineage(tmp_path):
    """A config without a pinned scenario seed must be rejected before matrix build."""
    bad = _certified_candidate(0.1, seed=700, objective=1.0)
    bad["candidate"].pop("scenario_seed")
    m = _manifest(tmp_path, name="m.json", candidates=[bad])
    configs = select_certified_configs(
        [m], target_planner=_TARGET_PLANNER, K=10, eligible_only=True
    )
    assert len(configs) == 0


def test_check_issue_6145_activation_passes_for_promote_with_five():
    """A valid promote result with >= 5 admitted candidates activates downstream work."""
    payload = {
        "schema_version": "issue_5303_search_promotion_result.v2",
        "decision": "promote",
        "contract_sha256": "a" * 64,
        "execution_commit": "2b3e3c199f1f0d283ffeed0e0bac55710d8efccc",
        "admitted_candidate_count": 5,
        "candidate_manifest_sha256": "b" * 64,
        "evidence_packet_sha256": "c" * 64,
    }
    assert check_issue_6145_activation(payload) == []


def test_check_issue_6145_activation_rejects_closure_without_promote():
    """Issue closure alone, or any non-promote decision, must never activate downstream work."""
    base = {
        "schema_version": "issue_5303_search_promotion_result.v2",
        "decision": "promote",
        "contract_sha256": "a" * 64,
        "execution_commit": "2b3e3c199f1f0d283ffeed0e0bac55710d8efccc",
        "admitted_candidate_count": 5,
        "candidate_manifest_sha256": "b" * 64,
        "evidence_packet_sha256": "c" * 64,
    }
    closure = {**base, "decision": "closed"}
    assert any("promote" in error for error in check_issue_6145_activation(closure))
    stop = {**base, "decision": "stop"}
    assert any("promote" in error for error in check_issue_6145_activation(stop))
    inconclusive = {**base, "decision": "inconclusive"}
    assert any("promote" in error for error in check_issue_6145_activation(inconclusive))


def test_check_issue_6145_activation_rejects_fewer_than_five():
    """Fewer than five admitted candidates must fail closed."""
    payload = {
        "schema_version": "issue_5303_search_promotion_result.v2",
        "decision": "promote",
        "contract_sha256": "a" * 64,
        "execution_commit": "2b3e3c199f1f0d283ffeed0e0bac55710d8efccc",
        "admitted_candidate_count": 4,
        "candidate_manifest_sha256": "b" * 64,
        "evidence_packet_sha256": "c" * 64,
    }
    errors = check_issue_6145_activation(payload)
    assert any("admitted_candidate_count" in error for error in errors)


def test_check_issue_6145_activation_rejects_missing_hashes():
    """Missing or malformed hashes must fail closed."""
    payload = {
        "schema_version": "issue_5303_search_promotion_result.v2",
        "decision": "promote",
        "execution_commit": "2b3e3c199f1f0d283ffeed0e0bac55710d8efccc",
        "admitted_candidate_count": 5,
    }
    errors = check_issue_6145_activation(payload)
    assert any("contract_sha256" in error for error in errors)
    assert any("candidate_manifest_sha256" in error for error in errors)
    assert any("evidence_packet_sha256" in error for error in errors)


def test_check_issue_6145_activation_rejects_bad_contract_hash():
    """A contract hash that does not match the frozen hash must fail closed."""
    payload = {
        "schema_version": "issue_5303_search_promotion_result.v2",
        "decision": "promote",
        "contract_sha256": "a" * 64,
        "execution_commit": "2b3e3c199f1f0d283ffeed0e0bac55710d8efccc",
        "admitted_candidate_count": 5,
        "candidate_manifest_sha256": "b" * 64,
        "evidence_packet_sha256": "c" * 64,
    }
    errors = check_issue_6145_activation(payload, expected_contract_sha256="d" * 64)
    assert any("contract_sha256" in error for error in errors)
