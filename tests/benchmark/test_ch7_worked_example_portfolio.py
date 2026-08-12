"""Tests for the Chapter 7 worked-example portfolio contract (issue #6789)."""

# evidence-writer-exempt: this test writes only a temporary malformed YAML input fixture

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from robot_sf.benchmark import case_portfolio
from robot_sf.benchmark.case_portfolio import (
    SCHEMA_VERSION,
    build_ch7_worked_example_portfolio,
    finalize_manifest,
    read_json_or_gzip,
    validate_ch7_worked_example_portfolio,
    write_deterministic_json,
)
from scripts.analysis.build_ch7_worked_example_portfolio import main as portfolio_cli_main

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "robot_sf/benchmark/schemas/ch7_case_portfolio.schema.v2.json"
DEFAULT_CONFIG = REPO_ROOT / "configs/analysis/ch7_worked_example_portfolio.v1.yaml"
DEFAULT_CANDIDATES = (
    REPO_ROOT / "docs/context/evidence/issue_5446_release_0_0_3_candidates/"
    "seed_flip_inversion_candidates.v1.json.gz"
)
FIXTURE_RELEASE_REF = "configs/scenarios/single/francis2023_narrow_doorway.yaml"
FIXTURE_TRACE_REF = "robot_sf/benchmark/trace_reexport_packaging.py"
FIXTURE_RELEASE_SHA256 = hashlib.sha256((REPO_ROOT / FIXTURE_RELEASE_REF).read_bytes()).hexdigest()
FIXTURE_TRACE_SHA256 = hashlib.sha256((REPO_ROOT / FIXTURE_TRACE_REF).read_bytes()).hexdigest()


def _pass_checks(grain: str) -> dict[str, str]:
    by_grain = {
        "cross_cell": (
            "release_campaign_identity",
            "source_hashes",
            "exact_digest_human_review_admission",
            "durable_source_status",
            "typed_outcome_collision_semantics",
            "execution_status",
            "scenario_config_seed_provenance",
            "telemetry_sufficiency",
        ),
        "cell": (
            "release_campaign_identity",
            "source_hashes",
            "exact_digest_human_review_admission",
            "durable_source_status",
            "typed_outcome_collision_semantics",
            "execution_status",
            "scenario_config_seed_provenance",
            "route_feasibility",
            "release_vs_rerun_outcome_agreement",
            "telemetry_sufficiency",
        ),
        "matched_planner_pair": (
            "release_campaign_identity",
            "source_hashes",
            "exact_digest_human_review_admission",
            "durable_source_status",
            "typed_outcome_collision_semantics",
            "execution_status",
            "scenario_config_seed_provenance",
            "matched_initial_state_or_shared_prefix",
            "release_vs_rerun_outcome_agreement",
            "exact_repeat_or_context_sensitivity",
            "telemetry_sufficiency",
        ),
        "matched_seed_pair": (
            "release_campaign_identity",
            "source_hashes",
            "exact_digest_human_review_admission",
            "durable_source_status",
            "typed_outcome_collision_semantics",
            "execution_status",
            "scenario_config_seed_provenance",
            "matched_initial_state_or_shared_prefix",
            "release_vs_rerun_outcome_agreement",
            "exact_repeat_or_context_sensitivity",
            "telemetry_sufficiency",
        ),
        "episode": (
            "release_campaign_identity",
            "source_hashes",
            "exact_digest_human_review_admission",
            "durable_source_status",
            "typed_outcome_collision_semantics",
            "execution_status",
            "scenario_config_seed_provenance",
            "route_feasibility",
            "trace_resolution",
            "trace_schema",
            "release_vs_rerun_outcome_agreement",
            "visualization_only_status",
            "telemetry_sufficiency",
        ),
    }
    checks = dict.fromkeys(by_grain[grain], "pass")
    checks["execution_status"] = {
        "status": "pass",
        "execution_mode": "native",
        "stop_condition": "terminal",
    }
    return checks


def _unit(  # noqa: PLR0913
    case_id: str,
    role: str,
    grain: str,
    topology: str,
    *,
    conceptual_grain: str,
    severity: float = 1.0,
    page_cost: float = 1.0,
    eligibility: dict[str, Any] | None = None,
    conceptual_coverage: list[str] | None = None,
    failure_class: str | None = None,
    process_class: str | None = None,
    telemetry_grade: str = "geometry",
) -> dict[str, Any]:
    coverage_tags = conceptual_coverage or [conceptual_grain]
    trace_case = "trace" in coverage_tags
    eligibility_checks = eligibility if eligibility is not None else _pass_checks(grain)
    if trace_case:
        eligibility_checks = dict(eligibility_checks)
        eligibility_checks.setdefault("trace_resolution", "pass")
        eligibility_checks.setdefault("trace_schema", "pass")
        eligibility_checks.setdefault("visualization_only_status", "pass")
    eligibility_checks = dict(eligibility_checks)
    telemetry = eligibility_checks.get("telemetry_sufficiency")
    if telemetry == "pass":
        eligibility_checks["telemetry_sufficiency"] = {
            "status": "pass",
            "telemetry_grade": telemetry_grade,
        }
    elif isinstance(telemetry, dict) and telemetry.get("status") == "pass":
        telemetry.setdefault("telemetry_grade", telemetry_grade)
    required_views = (
        ["world_xy", "route_sn", "time_space", "event_timeline", "cell_context"]
        if trace_case
        else ["world_xy", "cell_context"]
    )
    return {
        "case_id": case_id,
        "grain": grain,
        "conceptual_grain": conceptual_grain,
        "conceptual_coverage": coverage_tags,
        "primary_role": role,
        "claim_grade": "descriptive",
        "secondary_descriptors": [],
        "allowed_claim": f"bounded claim for {case_id}",
        "forbidden_claims": ["unsupported causal claim"],
        "event_anchor": {
            "type": "terminal",
            "source": "unit_test",
            "time_s": 1.0,
            "source_field": "fixture.event",
            "shared_between_cases": False,
            "observed_at": "2026-08-05T00:00:00Z",
            "shared_semantics": {"case_id": case_id},
        },
        "presentation": {
            "required_views": required_views,
            "conceptual_coverage": coverage_tags,
            "shared_axes": {"status": "not_required"},
            "keyframes": {"status": "not_required"},
            "shared_axis_contract": "fixture_world_time_axes",
            "semantic_keyframes": [],
        },
        "source_boundary": {
            "synthetic_fixture": True,
            "release_id": "fixture_release",
            "release_rows_sha256": FIXTURE_RELEASE_SHA256,
            "expected_release_rows_sha256": FIXTURE_RELEASE_SHA256,
            "trace_package_sha256": FIXTURE_TRACE_SHA256 if trace_case else "unavailable",
            "expected_trace_package_sha256": FIXTURE_TRACE_SHA256 if trace_case else "unavailable",
            "visualization_only_reexecution": trace_case,
            "telemetry_grade": telemetry_grade,
        },
        "source_refs": [FIXTURE_RELEASE_REF, FIXTURE_TRACE_REF]
        if trace_case
        else [FIXTURE_RELEASE_REF],
        "coverage": {
            "topology": topology,
            "mechanism": role,
            "failure_class": failure_class or role,
            "process_class": process_class or role,
        },
        "eligibility": eligibility_checks,
        "dimensions": {
            "evidence_grade": 3,
            "provenance_completeness": 3,
            "topology_mechanism": 2,
            "terminal_outcome": 2,
            "criticality_persistence": severity,
            "entropy_bimodality": severity,
            "paired_divergence": severity,
            "metric_disagreement": severity,
            "representativeness_or_outlier": severity,
            "telemetry_visualizability": 3,
            "page_cost": page_cost,
        },
    }


def _complete_config(extra_units: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    units = [
        _unit(
            "campaign_metric",
            "metric_disagreement",
            "cross_cell",
            "cross_trap",
            conceptual_grain="campaign",
            failure_class="metric_failure",
            process_class="aggregate_table",
        ),
        _unit(
            "cell_feasibility",
            "feasibility_criticism",
            "cell",
            "narrow_doorway",
            conceptual_grain="cell",
            failure_class="feasibility_failure",
            process_class="geometry_table",
        ),
        _unit(
            "contrast_upset",
            "planner_upset",
            "matched_planner_pair",
            "double_bottleneck",
            conceptual_grain="matched_contrast",
            failure_class="planner_failure",
            process_class="matched_planner_table",
        ),
        _unit(
            "trace_seed",
            "seed_sensitivity",
            "matched_seed_pair",
            "doorway",
            conceptual_grain="matched_contrast",
            conceptual_coverage=["matched_contrast", "trace"],
            failure_class="seed_failure",
            process_class="matched_seed_trace",
        ),
    ]
    if extra_units:
        units.extend(extra_units)
    return {
        "schema_version": SCHEMA_VERSION,
        "selection": {
            "target_size": 4,
            "max_size": 4,
            "required_roles": [
                "planner_upset",
                "seed_sensitivity",
                "feasibility_criticism",
                "metric_disagreement",
            ],
            "required_grains": [
                "cross_cell",
                "cell",
                "matched_planner_pair",
                "matched_seed_pair",
            ],
            "required_conceptual_grains": [
                "campaign",
                "cell",
                "matched_contrast",
                "trace",
            ],
            "required_topologies": [
                "cross_trap",
                "narrow_doorway",
                "double_bottleneck",
                "doorway",
            ],
            "required_failure_classes": [
                "metric_failure",
                "feasibility_failure",
                "planner_failure",
                "seed_failure",
            ],
            "required_process_classes": [
                "aggregate_table",
                "geometry_table",
                "matched_planner_table",
                "matched_seed_trace",
            ],
        },
        "evidence_units": units,
    }


def test_chapter7_portfolio_schema_validates_complete_manifest() -> None:
    """A complete four-grain synthetic manifest validates structurally and by JSON schema."""
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(_complete_config()))
    result = validate_ch7_worked_example_portfolio(manifest)
    assert result.ok, result.structural_violations
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(manifest)
    assert manifest["status"] == "complete"
    assert manifest["schema_version"] == "ch7_case_portfolio.v2"
    assert {tag for item in manifest["selected"] for tag in item["conceptual_coverage"]} == {
        "campaign",
        "cell",
        "matched_contrast",
        "trace",
    }
    assert {item["grain"] for item in manifest["selected"]} == {
        "cross_cell",
        "cell",
        "matched_planner_pair",
        "matched_seed_pair",
    }
    assert len({item["primary_role"] for item in manifest["selected"]}) == 4
    for item in manifest["selected"]:
        assert item["event_anchor"]["type"] in {
            "collision",
            "min_clearance",
            "first_gate_breach",
            "stall_onset",
            "terminal",
        }
        assert (
            isinstance(item["event_anchor"]["time_s"], int | float)
            or item["event_anchor"]["time_s"] == "unavailable"
        )
        assert set(item["presentation"]["required_views"]).issubset(
            {"world_xy", "route_sn", "time_space", "event_timeline", "cell_context"}
        )
        assert isinstance(item["presentation"]["shared_axis_contract"], str)
        assert item["presentation"]["shared_axis_contract"]
        assert item["selection"]["pareto_status"] == "nondominated"
        assert isinstance(item["selection"]["vector"], dict)
        assert item["selection"]["included_reason"] != "unavailable"
        assert item["selection"]["excluded_reason"] == "unavailable"
        assert item["eligibility"]["status"] == "admitted"
        assert item["primary_role"] in {
            "prototype",
            "criticism",
            "boundary",
            "planner_upset",
            "seed_sensitivity",
            "metric_disagreement",
            "process_contrast",
            "feasibility_criticism",
            "negative_control",
            "causal_abstention",
        }
        assert item["eligibility"]["initial_state_match"] in {"pass", "fail", "unavailable"}
        assert item["eligibility"]["outcome_match"] in {"pass", "fail", "unavailable"}
        assert item["eligibility"]["telemetry_grade"] in {
            "geometry",
            "kinematics",
            "controller",
            "counterfactual",
        }


def test_finalized_portfolio_remains_valid_after_json_round_trip(tmp_path: Path) -> None:
    """The stable case-ID tie break must preserve its JSON representation on replay."""

    output = tmp_path / "portfolio.json"
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(_complete_config()))

    write_deterministic_json(manifest, output)
    reloaded = json.loads(output.read_text(encoding="utf-8"))

    validation = validate_ch7_worked_example_portfolio(reloaded)
    assert validation.ok, validation.structural_violations
    assert (
        reloaded["exact_enumeration"]["best_score"] == manifest["exact_enumeration"]["best_score"]
    )


def test_literal_role_aliases_are_mapped_before_public_v2_emission() -> None:
    """Legacy aliases can be accepted internally but never emitted as public v2 roles."""
    alias = _unit(
        "aaa_alias_boundary",
        "boundary_case",
        "cell",
        "alias_topology",
        conceptual_grain="cell",
        failure_class="alias_failure",
        process_class="alias_process",
    )
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(_complete_config([alias])))
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    assert ledger["aaa_alias_boundary"]["primary_role"] == "boundary"
    assert ledger["aaa_alias_boundary"]["coverage"]["role"] == "boundary"
    assert "boundary_case" not in json.dumps(manifest, sort_keys=True)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    role_schema = schema["$defs"]["selectedCase"]["properties"]["primary_role"]
    assert "boundary_case" not in role_schema["enum"]
    assert {"prototype", "criticism", "boundary", "negative_control"}.issubset(
        set(role_schema["enum"])
    )


def test_exact_v2_scalar_fields_never_emit_internal_sentinel_values() -> None:
    """Public scalar fields use the authoritative v2 literals even when checks fail."""
    config = _complete_config()
    cell = next(
        unit for unit in config["evidence_units"] if unit["primary_role"] == "feasibility_criticism"
    )
    cell["eligibility"]["release_vs_rerun_outcome_agreement"] = {
        "status": "not_applicable",
        "reason": "allowed cell-table exception",
    }
    cell["eligibility"]["telemetry_sufficiency"] = {
        "status": "fail",
        "reason": "geometry-only table lacks controller telemetry",
    }
    manifest = build_ch7_worked_example_portfolio(config)
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    report = ledger["cell_feasibility"]["eligibility"]
    assert report["initial_state_match"] == "unavailable"
    assert report["outcome_match"] == "unavailable"
    assert report["telemetry_grade"] == "geometry"
    assert report["checks"]["release_vs_rerun_outcome_agreement"]["status"] == "not_applicable"
    assert ledger["cell_feasibility"]["disposition"] == "excluded"


@pytest.mark.parametrize("grade", ["geometry", "kinematics", "controller", "counterfactual"])
def test_telemetry_grade_preserves_canonical_declared_grade(grade: str) -> None:
    """Telemetry grade is a replayed contract value, not a hardcoded geometry default."""
    config = _complete_config()
    planner = next(unit for unit in config["evidence_units"] if unit["case_id"] == "contrast_upset")
    planner["eligibility"]["telemetry_sufficiency"] = {
        "status": "pass",
        "telemetry_grade": grade,
    }
    planner["source_boundary"]["telemetry_grade"] = grade
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(config))
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    report = ledger["contrast_upset"]["eligibility"]
    assert report["telemetry_grade"] == grade
    assert report["checks"]["telemetry_sufficiency"]["telemetry_grade"] == grade
    assert validate_ch7_worked_example_portfolio(manifest).ok


def test_boolean_eligibility_runs_before_scientific_interest() -> None:
    """An ineligible high-severity case is excluded before Pareto/coverage selection."""
    bad_eligibility = _pass_checks("episode")
    bad_eligibility["trace_schema"] = {"status": "fail", "reason": "malformed trace rows"}
    severe_bad = _unit(
        "trace_seed_bad_severe",
        "seed_sensitivity",
        "episode",
        "doorway",
        conceptual_grain="trace",
        severity=99,
        eligibility=bad_eligibility,
    )
    manifest = build_ch7_worked_example_portfolio(_complete_config([severe_bad]))
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    assert ledger["trace_seed_bad_severe"]["disposition"] == "excluded"
    assert ledger["trace_seed_bad_severe"]["pareto_member"] is False
    assert any(
        "eligibility_trace_schema:fail" in r
        for r in ledger["trace_seed_bad_severe"]["exclusion_reasons"]
    )


def test_all_required_checks_not_applicable_cannot_bypass_eligibility() -> None:
    """not_applicable is only allowed for explicit role/grain exceptions."""
    config = _complete_config()
    unit = config["evidence_units"][0]
    unit["eligibility"] = {
        check: {"status": "not_applicable", "reason": "attempted bypass"}
        for check in _pass_checks(unit["grain"])
    }
    manifest = build_ch7_worked_example_portfolio(config)
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    assert ledger[unit["case_id"]]["disposition"] == "excluded"
    assert any(
        "eligibility_release_campaign_identity:fail" in reason
        for reason in ledger[unit["case_id"]]["exclusion_reasons"]
    )


def test_nested_pass_with_fallback_execution_mode_fails_eligibility() -> None:
    """A nested pass cannot hide fallback/degraded execution mode."""
    config = _complete_config()
    unit = config["evidence_units"][0]
    unit["eligibility"]["execution_status"] = {
        "status": "pass",
        "execution_mode": "fallback",
        "stop_condition": "terminal",
    }
    manifest = build_ch7_worked_example_portfolio(config)
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    report = ledger[unit["case_id"]]["eligibility"]
    assert report["execution_mode"] == "fallback"
    assert report["checks"]["execution_status"]["status"] == "fail"
    assert "eligibility_execution_status:fail" in ledger[unit["case_id"]]["exclusion_reasons"]


def test_trace_presentation_requires_trace_gates_for_matched_seed_pair() -> None:
    """A matched-seed case that presents trace evidence must carry trace gates."""
    config = _complete_config()
    seed = next(
        unit for unit in config["evidence_units"] if unit["primary_role"] == "seed_sensitivity"
    )
    del seed["eligibility"]["trace_resolution"]
    manifest = build_ch7_worked_example_portfolio(config)
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    assert ledger["trace_seed"]["disposition"] == "excluded"
    assert "eligibility_trace_resolution:unavailable" in ledger["trace_seed"]["exclusion_reasons"]


def test_cell_trace_presentation_cannot_use_trace_na_allowlist() -> None:
    """Cell feasibility can use NA for non-trace gates, but not for declared trace gates."""
    config = _complete_config()
    cell = next(
        unit for unit in config["evidence_units"] if unit["primary_role"] == "feasibility_criticism"
    )
    cell["conceptual_coverage"] = ["cell", "trace"]
    cell["presentation"]["required_views"] = ["world_xy", "event_timeline"]
    cell["presentation"]["conceptual_coverage"] = ["cell", "trace"]
    cell["eligibility"]["trace_resolution"] = {
        "status": "not_applicable",
        "reason": "invalid NA once trace is declared",
    }
    cell["eligibility"]["trace_schema"] = {
        "status": "not_applicable",
        "reason": "invalid NA once trace is declared",
    }
    cell["eligibility"]["visualization_only_status"] = "pass"
    cell["source_boundary"]["trace_package_sha256"] = "b" * 64
    manifest = build_ch7_worked_example_portfolio(config)
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    assert ledger["cell_feasibility"]["disposition"] == "excluded"
    assert "eligibility_trace_resolution:fail" in ledger["cell_feasibility"]["exclusion_reasons"]


def test_cell_canonical_trace_view_requires_trace_gates_without_substring_match() -> None:
    """Canonical trace views require trace gates, while arbitrary substrings do not."""
    config = _complete_config()
    cell = next(
        unit for unit in config["evidence_units"] if unit["primary_role"] == "feasibility_criticism"
    )
    cell["presentation"]["required_views"] = ["world_xy", "route_sn", "cell_context"]
    manifest = build_ch7_worked_example_portfolio(config)
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    assert ledger["cell_feasibility"]["disposition"] == "excluded"
    assert (
        "eligibility_trace_resolution:unavailable"
        in ledger["cell_feasibility"]["exclusion_reasons"]
    )

    config = _complete_config()
    cell = next(
        unit for unit in config["evidence_units"] if unit["primary_role"] == "feasibility_criticism"
    )
    cell["presentation"]["required_views"] = ["world_xy", "traceability_summary", "cell_context"]
    manifest = build_ch7_worked_example_portfolio(config)
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    assert "trace_resolution" not in ledger["cell_feasibility"]["eligibility"]["required_checks"]
    result = validate_ch7_worked_example_portfolio(finalize_manifest(manifest))
    assert not result.ok
    assert any("malformed presentation contract" in v for v in result.structural_violations)


def test_redundant_severe_case_loses_to_role_covering_case() -> None:
    """A redundant severe case must not displace a less-severe required-role case."""
    redundant = _unit(
        "trace_boundary_redundant_more_severe",
        "boundary_case",
        "episode",
        "doorway",
        conceptual_grain="trace",
        severity=10,
    )
    needed = _unit(
        "campaign_metric",
        "metric_disagreement",
        "cross_cell",
        "cross_trap",
        conceptual_grain="campaign",
        severity=1,
    )
    config = _complete_config([redundant])
    config["evidence_units"] = [
        unit for unit in config["evidence_units"] if unit["case_id"] != "campaign_metric"
    ] + [needed]
    manifest = build_ch7_worked_example_portfolio(config)
    selected_ids = {item["case_id"] for item in manifest["selected"]}
    assert "campaign_metric" in selected_ids
    assert "trace_boundary_redundant_more_severe" not in selected_ids


def test_complete_exclusion_ledger_and_stable_tie_breaks() -> None:
    """Every inventory row is retained with a disposition and stable case-ID tie break."""
    extra = _unit(
        "aaa_extra_boundary",
        "boundary_case",
        "cell",
        "doorway",
        conceptual_grain="cell",
    )
    manifest = build_ch7_worked_example_portfolio(_complete_config([extra]))
    ledger_ids = [entry["case_id"] for entry in manifest["ledger"]]
    assert ledger_ids == sorted(ledger_ids)
    assert len(ledger_ids) == len(set(ledger_ids)) == 5
    excluded = [entry for entry in manifest["ledger"] if entry["disposition"] == "excluded"]
    assert excluded
    assert all(entry["exclusion_reasons"] for entry in excluded)
    assert all(entry["stable_tie_break"] == entry["case_id"] for entry in manifest["ledger"])


def test_input_order_invariance_and_deterministic_output(tmp_path) -> None:
    """Permuting input order does not change selected cases or byte output."""
    config_a = _complete_config()
    config_b = _complete_config()
    config_b["evidence_units"] = list(reversed(config_b["evidence_units"]))
    manifest_a = finalize_manifest(build_ch7_worked_example_portfolio(config_a))
    manifest_b = finalize_manifest(build_ch7_worked_example_portfolio(config_b))
    assert [item["case_id"] for item in manifest_a["selected"]] == [
        item["case_id"] for item in manifest_b["selected"]
    ]
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    write_deterministic_json(manifest_a, path_a)
    write_deterministic_json(manifest_b, path_b)
    assert (
        hashlib.sha256(path_a.read_bytes()).hexdigest()
        == hashlib.sha256(path_b.read_bytes()).hexdigest()
    )


def test_default_production_contract_returns_honest_partial(tmp_path) -> None:
    """The first production config returns selected frozen targets or explicit blockers."""
    out = tmp_path / "portfolio.json"
    code = portfolio_cli_main(
        [
            "--config",
            str(DEFAULT_CONFIG),
            "--json",
            str(out),
            "--validate",
        ]
    )
    assert code == 0
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "ch7_case_portfolio.v2"
    assert manifest["status"] == "partial"
    assert manifest["selected"] == []
    assert manifest["summary"]["n_source_inventory"] == 50
    assert manifest["summary"]["n_ledger"] == 54
    assert manifest["exact_enumeration"]["enumeration_count"] == 1
    assert manifest["pareto_analysis"]["adapter"] == (
        "role_local_adapter_over_issue_5601_compute_pareto_front"
    )
    assert set(manifest["summary"]["uncovered_roles"]) == {
        "planner_upset",
        "seed_sensitivity",
        "feasibility_criticism",
        "metric_disagreement",
    }
    ledger = {entry["primary_role"]: entry for entry in manifest["ledger"]}
    assert "exact_digest_human_review_admission" in json.dumps(ledger["planner_upset"])
    assert "durable_source_status" in json.dumps(ledger["seed_sensitivity"])
    assert ledger["feasibility_criticism"]["exclusion_reasons"]
    assert ledger["metric_disagreement"]["exclusion_reasons"]
    assert ledger["metric_disagreement"]["case_id"] == (
        "ch7-role-cross-cell-inversion--hybrid-vs-ppo--double-bottleneck-vs-blind-corner"
    )
    seed = ledger["seed_sensitivity"]
    shared_prefix = seed["eligibility"]["checks"]["matched_initial_state_or_shared_prefix"]
    assert shared_prefix["status"] == "fail"
    assert shared_prefix["applicable"] is False
    assert "shared_prefix=false" in shared_prefix["reason"]
    assert "eligibility_matched_initial_state_or_shared_prefix" not in " ".join(
        seed["exclusion_reasons"]
    )
    narrow = ledger["feasibility_criticism"]
    assert narrow["eligibility"]["checks"]["release_vs_rerun_outcome_agreement"]["status"] == (
        "not_applicable"
    )
    assert "eligibility_release_vs_rerun_outcome_agreement" not in " ".join(
        narrow["exclusion_reasons"]
    )
    upstream = [entry for entry in manifest["ledger"] if entry["case_id"].startswith("upstream::")]
    assert len(upstream) == 50
    consumed = [entry for entry in upstream if entry["consumed_by_case_id"]]
    unconfigured = [entry for entry in upstream if not entry["consumed_by_case_id"]]
    assert len(consumed) == 2
    assert len(unconfigured) == 48
    assert all(
        "upstream_not_configured_for_ch7_case_contract" in entry["exclusion_reasons"]
        for entry in unconfigured
    )
    assert all(
        entry["exclusion_reasons"][0].startswith("upstream_consumed_by_configured_case:")
        for entry in consumed
    )


def test_cli_rejects_source_digest_mismatch(tmp_path) -> None:
    """Configured source digest is verified before portfolio construction."""
    config = json.loads(json.dumps(_complete_config()))
    config["pinned_inputs"] = {
        "candidate_manifest_rel": str(DEFAULT_CANDIDATES),
        "candidate_manifest_gz_sha256": "0" * 64,
    }
    config_path = tmp_path / "bad_digest.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert portfolio_cli_main(["--config", str(config_path)]) == 2


def test_missing_configured_candidate_target_fails() -> None:
    """A configured candidate_manifest_id must resolve in the source inventory."""
    config = _complete_config()
    config["evidence_units"][0]["candidate_manifest_id"] = "missing::candidate"
    with pytest.raises(Exception, match="configured candidate_manifest_id not found"):
        build_ch7_worked_example_portfolio(config, candidate_manifest={"candidates": []})


def test_missing_scientific_dimension_is_not_fabricated() -> None:
    """Missing dimensions stay unavailable and exclude the row before selection."""
    config = _complete_config()
    del config["evidence_units"][0]["dimensions"]["metric_disagreement"]
    manifest = build_ch7_worked_example_portfolio(config)
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    assert (
        "scientific_dimension_unavailable:metric_disagreement"
        in ledger["campaign_metric"]["exclusion_reasons"]
    )


def test_source_hashes_pass_requires_concrete_required_hashes() -> None:
    """A source_hashes pass cannot coexist with unavailable required source hashes."""
    config = _complete_config()
    trace = next(unit for unit in config["evidence_units"] if unit["case_id"] == "trace_seed")
    trace["source_boundary"]["trace_package_sha256"] = "unavailable"
    manifest = build_ch7_worked_example_portfolio(config)
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    assert ledger["trace_seed"]["eligibility"]["checks"]["source_hashes"]["status"] == "fail"
    assert "eligibility_source_hashes:fail" in ledger["trace_seed"]["exclusion_reasons"]


def test_source_hashes_pass_rejects_all_zero_or_mismatched_expected_digest() -> None:
    """A syntactic 64-hex digest is not enough without matching referenced bytes."""
    config = _complete_config()
    planner = next(unit for unit in config["evidence_units"] if unit["case_id"] == "contrast_upset")
    planner["source_boundary"]["release_rows_sha256"] = "0" * 64
    manifest = build_ch7_worked_example_portfolio(config)
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    assert ledger["contrast_upset"]["eligibility"]["checks"]["source_hashes"]["status"] == "fail"

    config = _complete_config()
    planner = next(unit for unit in config["evidence_units"] if unit["case_id"] == "contrast_upset")
    planner["source_boundary"]["release_rows_sha256"] = "c" * 64
    manifest = build_ch7_worked_example_portfolio(config)
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    assert ledger["contrast_upset"]["eligibility"]["checks"]["source_hashes"]["status"] == "fail"
    reason = ledger["contrast_upset"]["eligibility"]["checks"]["source_hashes"]["reason"]
    assert "release_rows_sha256!=observed_release_rows_sha256" in reason


def test_source_hashes_pass_requires_safe_readable_in_repo_source_ref() -> None:
    """Missing, external, or traversal source refs cannot back a source_hashes pass."""
    config = _complete_config()
    planner = next(unit for unit in config["evidence_units"] if unit["case_id"] == "contrast_upset")
    planner["source_refs"] = ["../outside.json"]
    manifest = build_ch7_worked_example_portfolio(config)
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    source_check = ledger["contrast_upset"]["eligibility"]["checks"]["source_hashes"]
    assert source_check["status"] == "fail"
    assert "observed_release_rows_sha256" in source_check["reason"]


def test_source_hashes_pass_handles_source_ref_oserror_fail_closed(monkeypatch) -> None:
    """Public builder/validator report source-ref errors instead of raising OS failures."""
    config = _complete_config()

    def raise_oserror(_path: Path) -> str:
        raise OSError("simulated unreadable source")

    monkeypatch.setattr(case_portfolio, "file_sha256", raise_oserror)
    manifest = build_ch7_worked_example_portfolio(config)
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    source = ledger["contrast_upset"]["source"]
    assert source["observed_release_rows_sha256"] == "unavailable"
    assert "source ref error" in source["source_ref_errors"]["release_rows"]
    source_check = ledger["contrast_upset"]["eligibility"]["checks"]["source_hashes"]
    assert source_check["status"] == "fail"
    assert "observed_release_rows_sha256" in source_check["reason"]
    result = validate_ch7_worked_example_portfolio(finalize_manifest(manifest))
    assert result.ok


def test_role_specific_dimension_applicability_does_not_exclude_planner_upset() -> None:
    """Planner-upset rows do not require the inapplicable entropy-bimodality dimension."""
    config = _complete_config()
    planner = next(
        unit for unit in config["evidence_units"] if unit["primary_role"] == "planner_upset"
    )
    del planner["dimensions"]["entropy_bimodality"]
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(config))
    assert manifest["status"] == "complete"
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    assert ledger["contrast_upset"]["disposition"] == "selected"
    assert "contrast_upset" not in manifest["pareto_analysis"]["dimension_unavailable"]


def test_seed_shared_prefix_false_can_be_observed_without_excluding_seed_sensitivity() -> None:
    """A false shared-prefix observation is retained but is not a seed-sensitivity blocker."""
    config = _complete_config()
    seed = next(
        unit for unit in config["evidence_units"] if unit["primary_role"] == "seed_sensitivity"
    )
    seed["eligibility"]["matched_initial_state_or_shared_prefix"] = {
        "status": "fail",
        "reason": "shared_prefix=false; seed pair intentionally compares distinct realizations",
    }
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(config))
    assert manifest["status"] == "complete"
    ledger = {entry["case_id"]: entry for entry in manifest["ledger"]}
    check = ledger["trace_seed"]["eligibility"]["checks"]["matched_initial_state_or_shared_prefix"]
    assert check["status"] == "fail"
    assert check["applicable"] is False
    assert ledger["trace_seed"]["disposition"] == "selected"


def test_empty_selection_writes_strict_rfc_json_without_infinity(tmp_path) -> None:
    """Empty exact selections must not emit Infinity/NaN into deterministic JSON."""
    config = _complete_config()
    for unit in config["evidence_units"]:
        unit["eligibility"]["durable_source_status"] = {"status": "unavailable", "reason": "test"}
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(config))
    assert manifest["selected"] == []
    out = tmp_path / "manifest.json"
    write_deterministic_json(manifest, out)

    def reject_constant(value: str) -> None:
        raise AssertionError(f"non-RFC JSON constant emitted: {value}")

    json.loads(out.read_text(encoding="utf-8"), parse_constant=reject_constant)


def test_topology_failure_and_process_are_non_relaxable_constraints() -> None:
    """A same-topology/process portfolio cannot become complete through other strengths."""
    config = _complete_config()
    for unit in config["evidence_units"]:
        unit["coverage"]["topology"] = "same_topology"
        unit["coverage"]["failure_class"] = "same_failure"
        unit["coverage"]["process_class"] = "same_process"
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(config))
    assert manifest["status"] == "partial"
    assert manifest["summary"]["uncovered_topologies"]
    assert manifest["summary"]["uncovered_failure_classes"]
    assert manifest["summary"]["uncovered_process_classes"]

    tampered = json.loads(json.dumps(manifest))
    tampered["status"] = "complete"
    tampered = finalize_manifest(tampered)
    result = validate_ch7_worked_example_portfolio(tampered)
    assert not result.ok
    assert any(
        "complete status missing non-relaxable coverage" in v for v in result.structural_violations
    )


def test_uniqueness_violating_fallback_is_never_complete_or_validated() -> None:
    """A duplicate-role fallback stays partial even if relaxable tie-breakers look good."""
    config = _complete_config()
    metric = next(
        unit for unit in config["evidence_units"] if unit["primary_role"] == "metric_disagreement"
    )
    metric["primary_role"] = "planner_upset"
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(config))
    assert manifest["status"] == "partial"
    assert "metric_disagreement" in manifest["summary"]["uncovered_roles"]
    assert manifest["exact_enumeration"]["nonrelaxable_valid_subset"] is True
    assert len(manifest["selected"]) == 3
    assert len({item["primary_role"] for item in manifest["selected"]}) == 3

    tampered = json.loads(json.dumps(manifest))
    tampered["status"] = "complete"
    tampered = finalize_manifest(tampered)
    result = validate_ch7_worked_example_portfolio(tampered)
    assert not result.ok
    assert any(
        "complete status missing non-relaxable coverage" in v for v in result.structural_violations
    )


def test_validator_rejects_blank_hash_and_stale_eligibility() -> None:
    """The validator treats content hash and eligibility as recomputable facts."""
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(_complete_config()))
    blank_hash = json.loads(json.dumps(manifest))
    blank_hash["content_sha256"] = ""
    result = validate_ch7_worked_example_portfolio(blank_hash)
    assert not result.ok
    assert "content_sha256 mismatch" in result.structural_violations

    stale = json.loads(json.dumps(manifest))
    selected_id = stale["selected"][0]["case_id"]
    entry = next(item for item in stale["ledger"] if item["case_id"] == selected_id)
    check = next(iter(entry["eligibility"]["checks"].values()))
    check["status"] = "fail"
    stale = finalize_manifest(stale)
    result = validate_ch7_worked_example_portfolio(stale)
    assert not result.ok
    assert any("stale eligibility" in v for v in result.structural_violations)


def test_validator_rejects_deleted_required_trace_gates_from_both_copies() -> None:
    """Required checks are derived from ledger grain/coverage/views, not trusted lists."""
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(_complete_config()))
    tampered = json.loads(json.dumps(manifest))
    selected = next(item for item in tampered["selected"] if item["case_id"] == "trace_seed")
    ledger = next(item for item in tampered["ledger"] if item["case_id"] == "trace_seed")
    for container in (selected["eligibility"], ledger["eligibility"]):
        container["required_checks"] = [
            check
            for check in container["required_checks"]
            if check not in {"trace_resolution", "trace_schema", "visualization_only_status"}
        ]
        for check in ("trace_resolution", "trace_schema", "visualization_only_status"):
            container["checks"].pop(check, None)
        container["blockers"] = [
            blocker
            for blocker in container["blockers"]
            if blocker.get("check")
            not in {"trace_resolution", "trace_schema", "visualization_only_status"}
        ]
    result = validate_ch7_worked_example_portfolio(finalize_manifest(tampered))
    assert not result.ok
    assert any("stale eligibility required_checks" in v for v in result.structural_violations)
    assert any("stale eligibility checks" in v for v in result.structural_violations)


def test_validator_rejects_source_boundary_and_source_cotamper_against_actual_bytes() -> None:
    """Co-tampering source and boundary cannot override the digest of pinned source bytes."""
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(_complete_config()))
    tampered = json.loads(json.dumps(manifest))
    selected = next(item for item in tampered["selected"] if item["case_id"] == "contrast_upset")
    ledger = next(item for item in tampered["ledger"] if item["case_id"] == "contrast_upset")
    for container in (selected, ledger):
        container["source_boundary"]["release_rows_sha256"] = "c" * 64
        container["source"]["release_rows_sha256"] = "c" * 64
        container["source"]["observed_release_rows_sha256"] = FIXTURE_RELEASE_SHA256
        container["eligibility"]["checks"]["source_hashes"] = {
            "status": "pass",
            "reason": "",
            "applicable": True,
        }
    result = validate_ch7_worked_example_portfolio(finalize_manifest(tampered))
    assert not result.ok
    assert any("stale source block" in v for v in result.structural_violations)
    assert any("stale source_hashes pass" in v for v in result.structural_violations)


def test_validator_rejects_fabricated_status_and_reasons_on_eligible_row() -> None:
    """Eligibility status and reasons are replayed from checks/blockers."""
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(_complete_config()))
    tampered = json.loads(json.dumps(manifest))
    selected = tampered["selected"][0]
    ledger = next(item for item in tampered["ledger"] if item["case_id"] == selected["case_id"])
    for container in (selected["eligibility"], ledger["eligibility"]):
        container["eligible"] = True
        container["status"] = "excluded"
        container["reasons"] = ["fabricated:fail"]
    result = validate_ch7_worked_example_portfolio(finalize_manifest(tampered))
    assert not result.ok
    assert any("stale eligibility status" in v for v in result.structural_violations)
    assert any("stale eligibility reasons" in v for v in result.structural_violations)


def test_validator_rejects_telemetry_grade_cotamper() -> None:
    """Telemetry grade is replayed from canonical check/source metadata."""
    config = _complete_config()
    planner = next(unit for unit in config["evidence_units"] if unit["case_id"] == "contrast_upset")
    planner["eligibility"]["telemetry_sufficiency"] = {
        "status": "pass",
        "telemetry_grade": "controller",
    }
    planner["source_boundary"]["telemetry_grade"] = "controller"
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(config))
    tampered = json.loads(json.dumps(manifest))
    selected = next(item for item in tampered["selected"] if item["case_id"] == "contrast_upset")
    ledger = next(item for item in tampered["ledger"] if item["case_id"] == "contrast_upset")
    for container in (selected["eligibility"], ledger["eligibility"]):
        container["telemetry_grade"] = "geometry"
    result = validate_ch7_worked_example_portfolio(finalize_manifest(tampered))
    assert not result.ok
    assert any("stale eligibility telemetry_grade" in v for v in result.structural_violations)


def test_validator_recomputes_pareto_summary_exact_and_selected_eligibility() -> None:
    """Rehashed tampering of derived fields is still rejected."""
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(_complete_config()))

    tampered = json.loads(json.dumps(manifest))
    tampered["pareto_analysis"]["front"] = []
    result = validate_ch7_worked_example_portfolio(finalize_manifest(tampered))
    assert not result.ok
    assert "pareto_analysis.front is stale" in result.structural_violations

    tampered = json.loads(json.dumps(manifest))
    tampered["summary"]["uncovered_roles"] = ["bogus"]
    result = validate_ch7_worked_example_portfolio(finalize_manifest(tampered))
    assert not result.ok
    assert "summary.uncovered_roles is stale" in result.structural_violations

    tampered = json.loads(json.dumps(manifest))
    tampered["exact_enumeration"]["best_score"] = []
    result = validate_ch7_worked_example_portfolio(finalize_manifest(tampered))
    assert not result.ok
    assert "exact_enumeration.best_score is stale" in result.structural_violations

    tampered = json.loads(json.dumps(manifest))
    tampered["selected"][0]["eligibility"]["eligible"] = False
    result = validate_ch7_worked_example_portfolio(finalize_manifest(tampered))
    assert not result.ok
    assert any(
        "selected embedded eligibility differs from ledger" in violation
        for violation in result.structural_violations
    )


def test_validator_rejects_selected_projection_tampering_after_rehash() -> None:
    """Selected source/selection/event/presentation/claim fields are recomputed from ledger facts."""
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(_complete_config()))

    tampered = json.loads(json.dumps(manifest))
    tampered["selected"][0]["source"]["visualization_only_reexecution"] = True
    selected_id = tampered["selected"][0]["case_id"]
    entry = next(item for item in tampered["ledger"] if item["case_id"] == selected_id)
    entry["selected_projection"] = tampered["selected"][0]
    result = validate_ch7_worked_example_portfolio(finalize_manifest(tampered))
    assert not result.ok
    assert any(
        "selected record differs from recomputed ledger source" in v
        for v in result.structural_violations
    )

    tampered = json.loads(json.dumps(manifest))
    del tampered["selected"][0]["presentation"]["shared_axis_contract"]
    result = validate_ch7_worked_example_portfolio(finalize_manifest(tampered))
    assert not result.ok
    assert any("malformed presentation contract" in v for v in result.structural_violations)

    tampered = json.loads(json.dumps(manifest))
    tampered["selected"][0]["claim"]["grade"] = "descriptive"
    tampered["selected"][0]["selection"]["included_reason"] = "fabricated"
    selected_id = tampered["selected"][0]["case_id"]
    entry = next(item for item in tampered["ledger"] if item["case_id"] == selected_id)
    entry["selected_projection"] = tampered["selected"][0]
    result = validate_ch7_worked_example_portfolio(finalize_manifest(tampered))
    assert not result.ok
    assert any(
        "selected record differs from recomputed ledger source" in v
        for v in result.structural_violations
    )


def test_validator_recomputes_pareto_directions_and_dominated_metadata() -> None:
    """Pareto audit directions and dominated records are recomputed, not trusted."""
    dominated = _unit(
        "campaign_metric_dominated",
        "metric_disagreement",
        "cross_cell",
        "cross_trap",
        conceptual_grain="campaign",
        severity=0.1,
        failure_class="metric_failure",
        process_class="aggregate_table",
    )
    config = _complete_config([dominated])
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(config))
    assert manifest["pareto_analysis"]["dominated"]

    tampered = json.loads(json.dumps(manifest))
    tampered["pareto_analysis"]["directions"]["evidence_grade"] = "minimize"
    result = validate_ch7_worked_example_portfolio(finalize_manifest(tampered))
    assert not result.ok
    assert "pareto_analysis.directions is stale" in result.structural_violations

    tampered = json.loads(json.dumps(manifest))
    tampered["pareto_analysis"]["dominated"] = {}
    result = validate_ch7_worked_example_portfolio(finalize_manifest(tampered))
    assert not result.ok
    assert "pareto_analysis.dominated is stale" in result.structural_violations


def test_pareto_adapter_semantics_and_pairwise_diversity_scale_are_declared() -> None:
    """The manifest declares the #5601 adapter semantics and normalized diversity scale."""
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(_complete_config()))
    assert "issue #5601" in manifest["pareto_analysis"]["adapter_semantics"]
    score = manifest["exact_enumeration"]["best_score"]
    diversity_component = -score[6]
    assert 0.0 <= diversity_component <= 1.0


def test_validator_rejects_ineligible_or_non_pareto_selected_record() -> None:
    """Structural validation rejects malformed selected records."""
    manifest = finalize_manifest(build_ch7_worked_example_portfolio(_complete_config()))
    selected_id = manifest["selected"][0]["case_id"]
    for entry in manifest["ledger"]:
        if entry["case_id"] == selected_id:
            entry["eligibility"]["eligible"] = False
            break
    result = validate_ch7_worked_example_portfolio(manifest)
    assert not result.ok
    assert any("selected record is not eligible" in v for v in result.structural_violations)


def test_candidate_manifest_gzip_reader_resolves_frozen_targets() -> None:
    """The production candidate manifest can be read through the reusable gzip helper."""
    payload = read_json_or_gzip(DEFAULT_CANDIDATES)
    assert payload["schema_version"] == "seed_flip_inversion_candidates.v1"
    ids = {candidate["candidate_id"] for candidate in payload["candidates"]}
    assert "planner_upset::classic_realworld_double_bottleneck_high::goal>ppo" in ids
    assert "seed_flip::classic_doorway_medium::ppo" in ids


def test_portfolio_cli_help(capsys) -> None:
    """The CLI exposes help without requiring inputs."""
    with pytest.raises(SystemExit) as exc:
        portfolio_cli_main(["--help"])
    assert exc.value.code == 0
    assert "worked-example portfolio" in capsys.readouterr().out
