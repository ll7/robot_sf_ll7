"""Contract tests for deterministic Chapter 7 case dossiers (issue #6791)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from robot_sf.analysis_workbench.interaction_coordinates import (
    build_worked_example_process_trace_from_export,
    load_registered_conflict_zone_spec,
    load_registered_route_spec,
)
from robot_sf.analysis_workbench.simulation_trace_export import simulation_trace_export_from_dict
from robot_sf.benchmark.artifact_catalog import load_artifact_catalog
from robot_sf.benchmark.case_dossier_figure import (
    SYNTHETIC_FIXTURE_LABEL,
    render_case_dossier,
    validate_case_dossier_manifest,
)
from robot_sf.benchmark.case_portfolio import (
    build_ch7_worked_example_portfolio,
    finalize_manifest,
)
from robot_sf.benchmark.figure_qa import check_figure_file, validate_figures_in_catalog
from scripts.analysis.build_ch7_worked_example_portfolio import main as portfolio_cli_main
from scripts.analysis.render_case_dossier import main as dossier_cli_main

REPO_ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_REGISTRY = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "analysis_workbench"
    / "process_trace_geometry_registry_v1"
    / "fixture_registry.json"
)
DOSSIER_FIXTURES = REPO_ROOT / "tests/fixtures/benchmark/case_dossier_v1"


def test_benchmark_package_lazily_exports_case_dossier_renderer() -> None:
    """The renderer is discoverable through the benchmark public API."""

    from robot_sf import benchmark

    assert benchmark.render_case_dossier is render_case_dossier
    assert benchmark.validate_case_dossier_manifest is validate_case_dossier_manifest


@pytest.mark.parametrize(
    ("fixture_dir", "grammar"),
    (
        ("matched_seed118", "matched_start_planner"),
        ("doorway_seeds113_114", "same_cell_seed_sensitivity"),
    ),
)
def test_committed_production_shaped_fixture_packages_render_without_simulation(
    tmp_path: Path,
    fixture_dir: str,
    grammar: str,
) -> None:
    """Committed source-bound records render both requested templates directly."""

    bundle = render_case_dossier(
        DOSSIER_FIXTURES / fixture_dir / "input.json",
        tmp_path / fixture_dir,
    )

    assert bundle.manifest["comparison_grammar"] == grammar
    assert bundle.manifest["mode"] == "synthetic_fixture"
    assert bundle.manifest["scientific_admission"] is False


@pytest.mark.parametrize(
    ("fixture_path", "error_code"),
    (
        ("matched_seed118/bad_source_hash.input.json", "source_sha256_mismatch"),
        (
            "doorway_seeds113_114/bad_difference_curve.input.json",
            "no_shared_prefix_forbidden_mode",
        ),
    ),
)
def test_committed_known_bad_fixture_packages_fail_closed(
    tmp_path: Path,
    fixture_path: str,
    error_code: str,
) -> None:
    """Known-bad packages pin the source-hash and no-divergence stop conditions."""

    with pytest.raises(Exception, match=error_code):
        render_case_dossier(DOSSIER_FIXTURES / fixture_path, tmp_path / "bad")


def test_matched_start_public_renderer_writes_fixture_only_bundle(tmp_path: Path) -> None:
    """The public renderer emits a source-bound, visibly synthetic matched dossier."""

    input_path = _write_matched_input(tmp_path)

    bundle = render_case_dossier(input_path, tmp_path / "rendered")

    assert bundle.svg_path.is_file()
    assert bundle.pdf_path.is_file()
    assert bundle.caption_path.is_file()
    assert bundle.sidecar_path.is_file()
    assert bundle.manifest_path.is_file()
    assert SYNTHETIC_FIXTURE_LABEL in bundle.svg_path.read_text(encoding="utf-8")
    assert bundle.manifest["schema_version"] == "case_dossier_manifest.v1"
    assert bundle.manifest["comparison_grammar"] == "matched_start_planner"
    assert bundle.manifest["scientific_admission"] is False
    assert bundle.manifest["selection"]["case_id"] == "fixture-seed-118-planner-upset"
    assert bundle.manifest["selection"]["selected"] is True
    assert bundle.manifest["source_bindings"]["process_traces"][0]["sha256"] == _sha256(
        tmp_path / "goal-process.json"
    )
    assert validate_case_dossier_manifest(bundle.manifest) == []


def test_same_cell_seed_sensitivity_records_no_shared_prefix_without_difference_curve(
    tmp_path: Path,
) -> None:
    """Doorway seeds expose start separation and prohibit divergence semantics."""

    input_path = _write_doorway_input(tmp_path)

    bundle = render_case_dossier(input_path, tmp_path / "rendered-doorway")

    sidecar = json.loads(bundle.sidecar_path.read_text(encoding="utf-8"))
    comparison = sidecar["comparison"]
    assert comparison["shared_prefix"]["shared_prefix"] is False
    assert comparison["recorded_start_separation"]["initial_robot_separation_m"] == 0.05
    assert comparison["divergence_interpretation"] == {
        "allowed": False,
        "reason": "no_shared_prefix_reject_divergence_output",
    }
    assert set(comparison["prohibited_semantics"]) == {
        "adjacent_seed_significance",
        "causal_hinge",
        "difference_curve",
        "divergence_point",
        "pivot_time",
    }
    svg = bundle.svg_path.read_text(encoding="utf-8")
    assert "shared_prefix=false" in svg
    assert "recorded start separation = 0.050 m" in svg
    assert "difference curve" not in svg.lower()


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("difference_curve", True),
        ("pivot_time_s", 0.2),
        ("causal_hinge", True),
        ("adjacent_seed_significance", True),
    ),
)
def test_no_shared_prefix_rejects_forbidden_interpretation_modes(
    tmp_path: Path,
    option: str,
    value: object,
) -> None:
    """Doorway inputs fail closed before rendering forbidden comparison semantics."""

    input_path = _write_doorway_input(tmp_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload["comparison_options"][option] = value
    _write_json(input_path, payload)

    with pytest.raises(Exception, match="no_shared_prefix_forbidden_mode"):
        render_case_dossier(input_path, tmp_path / "forbidden")


def test_current_production_portfolio_fails_when_requested_case_is_not_selected(
    tmp_path: Path,
) -> None:
    """The current honest selected=[] production result stops before trace rendering."""

    input_path = _write_matched_input(tmp_path)
    production_portfolio = tmp_path / "production-portfolio.json"
    assert (
        portfolio_cli_main(
            [
                "--config",
                str(REPO_ROOT / "configs/analysis/ch7_worked_example_portfolio.v1.yaml"),
                "--json",
                str(production_portfolio),
                "--validate",
            ]
        )
        == 0
    )
    manifest = json.loads(production_portfolio.read_text(encoding="utf-8"))
    assert manifest["selected"] == []
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload["mode"] = "production"
    payload["case_id"] = (
        "ch7-role-planner-upset--classic-realworld-double-bottleneck-high--goal-vs-ppo--seed-118"
    )
    payload["sources"]["portfolio"] = _file_ref(production_portfolio)
    _write_json(input_path, payload)

    with pytest.raises(Exception, match="production_case_not_selected"):
        render_case_dossier(input_path, tmp_path / "production-stop")


def test_missing_optional_signals_render_explicit_unavailable_panels(tmp_path: Path) -> None:
    """Missing route, velocity, controller, and ensemble inputs are never synthesized."""

    input_path = _write_input(
        tmp_path,
        grammar="same_cell_seed_sensitivity",
        with_geometry=False,
        missing_velocity=True,
    )

    bundle = render_case_dossier(input_path, tmp_path / "unavailable-render")

    status = bundle.manifest["panel_status"]
    assert status["time_space"] == {
        "status": "unavailable",
        "reason": "route_and_conflict_projection_unavailable",
    }
    assert status["radial_closing_speed"] == {
        "status": "unavailable",
        "reason": "relative_velocity_unavailable",
    }
    assert status["controller_state"] == {
        "status": "unavailable",
        "reason": "controller_state_signal_absent",
    }
    assert status["ensemble_context"] == {
        "status": "unavailable",
        "reason": "synthetic_fixture_no_ensemble_inventory",
    }
    svg = bundle.svg_path.read_text(encoding="utf-8")
    assert "ROUTE / CONFLICT TIME–SPACE" in svg
    assert "UNAVAILABLE" in svg


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    (
        ("source_hash", "source_sha256_mismatch"),
        ("process_schema", "process_trace_invalid"),
        ("reciprocal_pair", "process_trace_pair_binding_mismatch"),
        ("atlas_selection", "atlas_selection_hash_mismatch"),
        ("shared_time_range", "shared_time_range_excludes_trace"),
    ),
)
def test_invalid_source_bindings_fail_before_figure_output(
    tmp_path: Path,
    mutation: str,
    error_code: str,
) -> None:
    """Hashes, schemas, reciprocal pair identity, and atlas selection bind before draw."""

    input_path = _write_matched_input(tmp_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    refs = {item["role"]: item for item in payload["sources"]["process_traces"]}
    if mutation == "source_hash":
        payload["sources"]["portfolio"]["sha256"] = "0" * 64
    elif mutation == "process_schema":
        trace_path = Path(refs["left"]["path"])
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        trace["schema_version"] = "worked_example_process_trace.v0"
        _write_json(trace_path, trace)
        refs["left"]["sha256"] = _sha256(trace_path)
    elif mutation == "reciprocal_pair":
        refs["right"]["path"] = refs["left"]["path"]
        refs["right"]["sha256"] = refs["left"]["sha256"]
    elif mutation == "atlas_selection":
        atlas_path = Path(payload["sources"]["campaign_atlas"]["path"])
        atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
        atlas["selection_manifest_hash"] = "0" * 64
        _write_json(atlas_path, atlas)
        payload["sources"]["campaign_atlas"]["sha256"] = _sha256(atlas_path)
    elif mutation == "shared_time_range":
        payload["layout"]["time_range_s"] = [0.0, 0.2]
    _write_json(input_path, payload)
    out_dir = tmp_path / "must-stay-empty"

    with pytest.raises(Exception, match=error_code):
        render_case_dossier(input_path, out_dir)
    assert not list(out_dir.glob("*.svg"))
    assert not list(out_dir.glob("*.pdf"))


@pytest.mark.parametrize("grammar", ("matched_start_planner", "same_cell_seed_sensitivity"))
def test_dossier_outputs_are_byte_deterministic_and_pass_all_figure_qa(
    tmp_path: Path,
    grammar: str,
) -> None:
    """Both template grammars produce identical fixed-size bytes and zero-error QA."""

    input_path = _write_input(tmp_path, grammar=grammar)
    first = render_case_dossier(input_path, tmp_path / "first")
    second = render_case_dossier(input_path, tmp_path / "second")

    for name in (
        "svg_path",
        "pdf_path",
        "caption_path",
        "sidecar_path",
        "manifest_path",
        "catalog_path",
    ):
        assert getattr(first, name).read_bytes() == getattr(second, name).read_bytes(), name
    assert (
        check_figure_file(
            first.svg_path,
            artifact_id="case_dossier",
            expected_format="svg",
            caption_path=first.caption_path,
        )
        == []
    )
    assert (
        check_figure_file(
            first.pdf_path,
            artifact_id="case_dossier",
            expected_format="pdf",
            caption_path=first.caption_path,
        )
        == []
    )
    catalog = load_artifact_catalog(first.catalog_path)
    assert catalog.artifacts[0].generation_commit != "0000000"
    issues = validate_figures_in_catalog(
        catalog,
        catalog_path=first.catalog_path,
        required_formats=frozenset({"svg", "pdf"}),
    )
    assert [issue for issue in issues if issue.severity == "error"] == []
    svg = first.svg_path.read_text(encoding="utf-8")
    assert "<dc:date>" not in svg
    width_match = re.search(r'<svg[^>]+width="([0-9.]+)pt"', svg)
    assert width_match is not None
    assert float(width_match.group(1)) / 72.0 == pytest.approx(426.79135 / 72.27)
    font_sizes = [float(value) for value in re.findall(r"font-size: ([0-9.]+)px", svg)]
    assert font_sizes
    assert min(font_sizes) >= 8.25


def test_case_dossier_cli_renders_and_checks_determinism(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public CLI renders a bundle and compares every emitted artifact."""

    input_path = _write_matched_input(tmp_path)

    assert (
        dossier_cli_main(
            [
                "--check-determinism",
                "--input",
                str(input_path),
                "--out-dir",
                str(tmp_path / "cli-output"),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["deterministic"] is True
    assert report["schema_version"] == "case_dossier_cli_report.v1"
    assert set(report["output_sha256"]) == {
        "artifact_catalog",
        "caption",
        "manifest",
        "pdf",
        "sidecar",
        "svg",
    }


def _write_matched_input(tmp_path: Path) -> Path:
    return _write_input(tmp_path, grammar="matched_start_planner")


def _write_doorway_input(tmp_path: Path) -> Path:
    return _write_input(tmp_path, grammar="same_cell_seed_sensitivity")


def _write_input(
    tmp_path: Path,
    *,
    grammar: str,
    with_geometry: bool = True,
    missing_velocity: bool = False,
) -> Path:
    matched = grammar == "matched_start_planner"
    scenario_id = (
        "classic_realworld_double_bottleneck_high" if matched else "classic_doorway_medium"
    )
    case_id = "fixture-seed-118-planner-upset" if matched else "fixture-doorway-seeds-113-114"
    left_planner = "goal" if matched else "ppo"
    right_planner = "ppo"
    left_seed = 118 if matched else 113
    right_seed = 118 if matched else 114
    right_offset = (0.0, 0.0) if matched else (0.05, 0.0)
    left = _trace_payload(
        trace_id=f"fixture-{scenario_id}-{left_planner}-seed-{left_seed}",
        planner_id=left_planner,
        seed=left_seed,
        scenario_id=scenario_id,
        final_position=(0.8, 0.0),
        missing_velocity=missing_velocity,
    )
    right = _trace_payload(
        trace_id=f"fixture-{scenario_id}-{right_planner}-seed-{right_seed}",
        planner_id=right_planner,
        seed=right_seed,
        scenario_id=scenario_id,
        start_offset=right_offset,
        final_position=(0.6, 0.2),
        missing_velocity=missing_velocity,
    )
    left_trace = simulation_trace_export_from_dict(left)
    right_trace = simulation_trace_export_from_dict(right)
    route = (
        load_registered_route_spec(GEOMETRY_REGISTRY, "fixture-route") if with_geometry else None
    )
    conflict = (
        load_registered_conflict_zone_spec(GEOMETRY_REGISTRY, "fixture-zone")
        if with_geometry
        else None
    )
    left_process = build_worked_example_process_trace_from_export(
        left_trace,
        route=route,
        conflict_zone=conflict,
        pair_trace=right_trace,
        pair_comparison_grain="matched_planner_pair" if matched else "matched_realization_pair",
    )
    right_process = build_worked_example_process_trace_from_export(
        right_trace,
        route=route,
        conflict_zone=conflict,
        pair_trace=left_trace,
        pair_comparison_grain="matched_planner_pair" if matched else "matched_realization_pair",
    )
    left_path = _write_json(tmp_path / f"{left_planner}-process.json", left_process)
    right_path = _write_json(tmp_path / f"{right_planner}-right-process.json", right_process)

    portfolio = finalize_manifest(
        build_ch7_worked_example_portfolio(_portfolio_config(case_id, grammar=grammar))
    )
    portfolio_path = _write_json(tmp_path / "portfolio.json", portfolio)
    atlas = {
        "schema_version": "campaign_atlas.v2",
        "campaign_id": "synthetic-fixture-release",
        "scenario_families": [scenario_id],
        "planners": sorted({left_planner, right_planner}),
        "event_anchor": "minimum_clearance",
        "selection_manifest_hash": portfolio["content_sha256"],
        "metric_definitions": {"outcome": "fixture labels consumed verbatim"},
        "cells": [
            _atlas_cell(scenario_id, left_planner, {"success": 9, "collision": 6}),
            *(
                [_atlas_cell(scenario_id, right_planner, {"success": 6, "collision": 9})]
                if right_planner != left_planner
                else []
            ),
        ],
    }
    atlas_path = _write_json(tmp_path / "atlas.json", atlas)
    payload = {
        "schema_version": "case_dossier_input.v1",
        "dossier_id": f"{case_id}-dossier",
        "case_id": case_id,
        "mode": "synthetic_fixture",
        "scientific_admission": False,
        "comparison_grammar": grammar,
        "comparison_options": {
            "difference_curve": False,
            "pivot_time_s": None,
            "causal_hinge": False,
            "adjacent_seed_significance": False,
        },
        "ensemble_context": {
            "status": "unavailable",
            "reason": "synthetic_fixture_no_ensemble_inventory",
            "missing_trace_ids": [],
            "ineligible_trace_ids": [],
            "excluded_trace_ids": [],
        },
        "sources": {
            "portfolio": _file_ref(portfolio_path),
            "process_traces": [
                {
                    "role": "left",
                    "label": f"{left_planner} seed {left_seed}",
                    "recorded_outcome": "success",
                    "source_class": "visualization_only_rerun_diagnostics",
                    **_file_ref(left_path),
                },
                {
                    "role": "right",
                    "label": f"{right_planner} seed {right_seed}",
                    "recorded_outcome": "collision",
                    "source_class": "visualization_only_rerun_diagnostics",
                    **_file_ref(right_path),
                },
            ],
            "campaign_atlas": {"source_class": "release_statistics", **_file_ref(atlas_path)},
        },
        "layout": {
            "final_width_in": 426.79135 / 72.27,
            "final_height_in": 9.8,
            "minimum_font_pt": 8.25,
            "world_crop_m": [-0.2, 1.4, -0.55, 0.55],
            "metre_scale_m": 0.5,
            "time_range_s": [0.0, 0.4],
            "clearance_range_m": [-0.2, 1.2],
            "speed_range_mps": [-0.2, 1.2],
            "palette_id": "case_dossier_colorblind.v1",
            "threshold_profile": "worked_example_threshold_profile.diagnostic.v1",
        },
        "narrative": {
            "observed_signature": "The fixture traces show different recorded paths and outcomes.",
            "competing_explanation": "Fixture geometry and planner commands both differ after start.",
            "causal_status": "observational_only",
            "generalization_limit": "Synthetic renderer proof; no scientific generalization.",
        },
    }
    return _write_json(tmp_path / "matched-input.json", payload)


def _trace_payload(
    *,
    trace_id: str,
    planner_id: str,
    seed: int,
    scenario_id: str,
    final_position: tuple[float, float],
    start_offset: tuple[float, float] = (0.0, 0.0),
    missing_velocity: bool = False,
) -> dict[str, Any]:
    positions = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.6, 0.0), final_position]
    positions = [
        (position[0] + start_offset[0], position[1] + start_offset[1]) for position in positions
    ]
    frames: list[dict[str, Any]] = []
    for step, position in enumerate(positions):
        frames.append(
            {
                "step": step,
                "time_s": step * 0.1,
                "robot": {
                    "position": list(position),
                    "heading": 0.0,
                    "velocity": (
                        [float("nan"), 0.0] if missing_velocity else [1.0 if step < 3 else 0.5, 0.0]
                    ),
                    "radius": 0.25,
                },
                "pedestrians": [
                    {
                        "id": "ped-a",
                        "position": [1.0, 0.15 - 0.05 * step],
                        "velocity": [float("nan"), 0.0] if missing_velocity else [0.0, -0.5],
                        "radius": 0.25,
                    },
                    {
                        "id": "ped-context",
                        "position": [1.3, 0.45],
                        "velocity": [0.0, 0.0],
                        "radius": 0.2,
                    },
                ],
                "planner": {
                    "selected_action": {
                        "linear_velocity": 1.0 if step < 3 else 0.5,
                        "angular_velocity": 0.0,
                    },
                    "encounter": {
                        "actor_id": "ped-a",
                        "encounter_id": "ped-a:encounter-0001",
                    },
                    "event": "step",
                    "run_config": {
                        "map_id": "fixture-double-bottleneck",
                        "horizon": 4,
                        "config_digest": ("a" if planner_id == "goal" else "b") * 64,
                        "time_step_s": 0.1,
                    },
                },
            }
        )
    return {
        "schema_version": "simulation_trace_export.v1",
        "trace_id": trace_id,
        "source": {
            "scenario_id": scenario_id,
            "seed": seed,
            "planner_id": planner_id,
            "episode_id": f"{trace_id}-episode",
            "generated_by": "issue-6791 synthetic fixture",
        },
        "evidence_boundary": "analysis_workbench_only",
        "coordinate_frame": "world",
        "units": {"position": "m", "heading": "rad", "time": "s", "velocity": "m/s"},
        "frames": frames,
    }


def _portfolio_config(case_id: str, *, grammar: str) -> dict[str, Any]:
    matched = grammar == "matched_start_planner"
    role = "planner_upset" if matched else "seed_sensitivity"
    grain = "matched_planner_pair" if matched else "matched_seed_pair"
    topology = "double_bottleneck" if matched else "doorway"
    process_class = "matched_planner_process" if matched else "matched_seed_process"
    release_ref = "configs/scenarios/single/francis2023_narrow_doorway.yaml"
    trace_ref = "robot_sf/benchmark/trace_reexport_packaging.py"
    release_sha = _sha256(REPO_ROOT / release_ref)
    trace_sha = _sha256(REPO_ROOT / trace_ref)
    checks = dict.fromkeys(
        (
            "release_campaign_identity",
            "source_hashes",
            "exact_digest_human_review_admission",
            "durable_source_status",
            "typed_outcome_collision_semantics",
            "scenario_config_seed_provenance",
            "matched_initial_state_or_shared_prefix",
            "release_vs_rerun_outcome_agreement",
            "exact_repeat_or_context_sensitivity",
            "trace_resolution",
            "trace_schema",
            "visualization_only_status",
        ),
        "pass",
    )
    checks["execution_status"] = {
        "status": "pass",
        "execution_mode": "native",
        "stop_condition": "terminal",
    }
    checks["telemetry_sufficiency"] = {
        "status": "pass",
        "telemetry_grade": "controller",
    }
    unit = {
        "case_id": case_id,
        "grain": grain,
        "conceptual_grain": "matched_contrast",
        "conceptual_coverage": ["matched_contrast", "trace"],
        "primary_role": role,
        "claim_grade": "descriptive",
        "secondary_descriptors": ["synthetic_fixture"],
        "allowed_claim": "Different executed fixture planners show different observed processes.",
        "forbidden_claims": ["causal planner superiority"],
        "event_anchor": {
            "type": "min_clearance",
            "time_s": 0.4,
            "source_field": "fixture.minimum_clearance",
            "shared_between_cases": True,
        },
        "presentation": {
            "required_views": [
                "world_xy",
                "route_sn",
                "time_space",
                "event_timeline",
                "cell_context",
            ],
            "shared_axis_contract": "case_dossier_shared_axes.v1",
            "semantic_keyframes": ["minimum_clearance"],
        },
        "source_boundary": {
            "synthetic_fixture": True,
            "release_id": "synthetic-fixture-release",
            "release_rows_sha256": release_sha,
            "expected_release_rows_sha256": release_sha,
            "trace_package_sha256": trace_sha,
            "expected_trace_package_sha256": trace_sha,
            "visualization_only_reexecution": True,
            "telemetry_grade": "controller",
        },
        "source_refs": [release_ref, trace_ref],
        "coverage": {
            "topology": topology,
            "mechanism": role,
            "failure_class": role,
            "process_class": process_class,
        },
        "eligibility": checks,
        "dimensions": {
            "evidence_grade": 3,
            "provenance_completeness": 3,
            "topology_mechanism": 3,
            "terminal_outcome": 3,
            "criticality_persistence": 2,
            "entropy_bimodality": 2,
            "paired_divergence": 3,
            "metric_disagreement": 1,
            "representativeness_or_outlier": 2,
            "telemetry_visualizability": 3,
            "page_cost": 1,
        },
    }
    return {
        "schema_version": "ch7_case_portfolio.v2",
        "selection": {
            "target_size": 1,
            "max_size": 1,
            "required_roles": [role],
            "required_grains": [grain],
            "required_conceptual_grains": ["matched_contrast", "trace"],
            "required_topologies": [topology],
            "required_failure_classes": [role],
            "required_process_classes": [process_class],
        },
        "evidence_units": [unit],
    }


def _atlas_cell(scenario_id: str, planner: str, counts: dict[str, int]) -> dict[str, Any]:
    denominator = sum(counts.values())
    return {
        "scenario_family": scenario_id,
        "planner": planner,
        "release_arm_id": None,
        "eligible": True,
        "ineligible_reason": None,
        "n_total": denominator,
        "outcome_counts": counts,
        "outcome_ci": {
            outcome: [count / denominator, 0.0, 1.0] for outcome, count in counts.items()
        },
        "exemplar_episode_ids": [],
    }


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _file_ref(path: Path) -> dict[str, str]:
    return {"path": path.as_posix(), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
