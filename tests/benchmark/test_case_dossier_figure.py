"""Contract tests for deterministic Chapter 7 case dossiers (issue #6791)."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from robot_sf.analysis_workbench.interaction_coordinates import (
    build_worked_example_process_trace_from_export,
    load_registered_conflict_zone_spec,
    load_registered_route_spec,
)
from robot_sf.analysis_workbench.simulation_trace_export import simulation_trace_export_from_dict
from robot_sf.benchmark import case_dossier_figure as dossier_module
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
    assert bundle.manifest["claim_fields"]["observed_signature"] == (
        "Different executed planner stacks show different observed processes and "
        "terminal outcomes under the matched recorded start."
    )
    assert bundle.manifest["claim_fields"]["competing_explanation"] == (
        "Recorded command and encounter-geometry differences are documented; "
        "attribution is not estimated."
    )
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
    assert ">case_id=fixture-doorway-seeds-113-114</text>" in svg
    assert "minimum clearance" in svg
    assert "safety breach" in svg
    assert "difference curve" not in svg.lower()
    assert bundle.manifest["renderer"]["canvas_text_bounds_checked"] is True
    assert bundle.manifest["renderer"]["cross_axes_text_overlap_checked"] is True
    assert bundle.manifest["panel_status"]["time_space"]["occupancy_ribbon"] == {
        "status": "available",
        "reason": "recorded_proxy_radius_envelope",
    }
    assert bundle.manifest["panel_status"]["time_space"]["tick_label_overlap_checked"] is True
    closest = bundle.manifest["panel_status"]["radial_closing_speed"]["closest_approach"]
    assert closest["left"]["status"] == "available"
    assert closest["left"]["model"] == "local_constant_velocity"
    assert closest["left"]["time_to_closest_approach_s"] == pytest.approx(0.15)
    for role in ("left", "right"):
        assert bundle.manifest["panel_status"][f"world_{role}"]["semantic_event_anchor_count"] > 1


@pytest.mark.parametrize(
    ("mode", "grammar", "expected_first_line"),
    (
        ("synthetic_fixture", "matched_start_planner", SYNTHETIC_FIXTURE_LABEL),
        (
            "synthetic_fixture",
            "same_cell_seed_sensitivity",
            f"{SYNTHETIC_FIXTURE_LABEL} · shared_prefix=false",
        ),
        (
            "production",
            "matched_start_planner",
            "RENDERING DOES NOT ADMIT SCIENTIFIC EVIDENCE",
        ),
        (
            "production",
            "same_cell_seed_sensitivity",
            "RENDERING DOES NOT ADMIT SCIENTIFIC EVIDENCE · shared_prefix=false",
        ),
    ),
)
def test_header_boundary_is_mode_and_grammar_owned(
    tmp_path: Path,
    mode: str,
    grammar: str,
    expected_first_line: str,
) -> None:
    """All mode×grammar headers expose the right controlled evidence boundary."""

    input_path = _write_input(
        tmp_path,
        grammar=grammar,
        mode=mode,
        terminal_outcomes=("success", "collision") if mode == "production" else (None, None),
    )

    boundary = dossier_module._dossier_header_boundary(dossier_module._load_bound_input(input_path))

    assert boundary.splitlines()[0] == expected_first_line
    assert (SYNTHETIC_FIXTURE_LABEL in boundary) is (mode == "synthetic_fixture")
    if grammar == "same_cell_seed_sensitivity":
        assert "shared_prefix=false" in boundary
        assert "recorded start separation = 0.050 m" in boundary


def test_production_same_cell_outputs_never_claim_synthetic(tmp_path: Path) -> None:
    """No production same-cell output surface may inherit synthetic fixture wording."""

    input_path = _write_input(
        tmp_path,
        grammar="same_cell_seed_sensitivity",
        mode="production",
        terminal_outcomes=("success", "collision"),
    )

    bundle = render_case_dossier(input_path, tmp_path / "production-same-cell")

    for path in (
        bundle.svg_path,
        bundle.caption_path,
        bundle.sidecar_path,
        bundle.manifest_path,
        bundle.catalog_path,
    ):
        output = path.read_text(encoding="utf-8").lower()
        assert "synthetic fixture" not in output
        assert "synthetic-fixture" not in output
    svg = bundle.svg_path.read_text(encoding="utf-8")
    assert "RENDERING DOES NOT ADMIT SCIENTIFIC EVIDENCE · shared_prefix=false" in svg
    assert "production ensemble inventory unavailable" in svg


@pytest.mark.parametrize(
    ("mode", "expected_reason", "forbidden_reason"),
    (
        (
            "synthetic_fixture",
            "synthetic_fixture_no_ensemble_inventory",
            "production_ensemble_inventory_unavailable",
        ),
        (
            "production",
            "production_ensemble_inventory_unavailable",
            "synthetic_fixture_no_ensemble_inventory",
        ),
    ),
)
def test_ensemble_reason_is_closed_and_mode_consistent(
    tmp_path: Path,
    mode: str,
    expected_reason: str,
    forbidden_reason: str,
) -> None:
    """Fixture builders and semantic validation cannot cross production boundaries."""

    input_path = _write_input(
        tmp_path,
        grammar="matched_start_planner",
        mode=mode,
        terminal_outcomes=("success", "collision") if mode == "production" else (None, None),
    )
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    assert payload["ensemble_context"] == {
        "status": "unavailable",
        "reason": expected_reason,
        "missing_trace_ids": [],
        "ineligible_trace_ids": [],
        "excluded_trace_ids": [],
    }

    payload["ensemble_context"]["reason"] = forbidden_reason
    _write_json(input_path, payload)
    with pytest.raises(Exception, match="ensemble_context_mode_reason_mismatch"):
        render_case_dossier(input_path, tmp_path / "forbidden-ensemble-reason")


def test_cross_axes_structural_text_overlap_fails_closed() -> None:
    """Adjacent panel labels and titles may not occupy the same rendered pixels."""

    figure = dossier_module.plt.figure(figsize=(2.0, 2.0))
    upper = figure.add_axes((0.1, 0.5, 0.8, 0.4))
    lower = figure.add_axes((0.1, 0.1, 0.8, 0.38))
    upper.set_xlabel("world x label")
    upper.xaxis.set_label_coords(0.5, -0.05)
    lower.set_title("route title", y=1.0, pad=0.0)
    figure.canvas.draw()
    try:
        with pytest.raises(
            dossier_module.CaseDossierError,
            match="cross_axes_text_overlap",
        ):
            dossier_module._assert_cross_axes_text_separation(figure)
    finally:
        dossier_module.plt.close(figure)


def test_structural_panel_body_text_must_remain_inside_its_axes() -> None:
    """Turn notes and context bodies may not consume neighboring panel space."""

    figure = dossier_module.plt.figure(figsize=(2.0, 2.0))
    panel = figure.add_axes((0.1, 0.1, 0.8, 0.8))
    panel.text(0.5, -0.1, "escaped body", transform=panel.transAxes)
    figure.canvas.draw()
    try:
        with pytest.raises(
            dossier_module.CaseDossierError,
            match="structural_panel_text_outside_axes",
        ):
            dossier_module._assert_structural_panel_text_containment({"fixture_panel": panel})
    finally:
        dossier_module.plt.close(figure)


def test_structural_panel_tick_label_overlap_fails_closed() -> None:
    """Visible x/y tick labels participate in final-width bbox overlap checks."""

    figure = dossier_module.plt.figure(figsize=(2.0, 0.5))
    panel = figure.add_axes((0.25, 0.15, 0.65, 0.70))
    panel.set_yticks((0.49, 0.51), ("lower", "upper"))
    figure.canvas.draw()
    try:
        with pytest.raises(
            dossier_module.CaseDossierError,
            match="structural_panel_text_overlap",
        ):
            dossier_module._assert_panel_text_nonoverlap({"fixture_panel": panel})
    finally:
        dossier_module.plt.close(figure)


def test_input_tree_copies_produce_portable_byte_identical_bundles(tmp_path: Path) -> None:
    """Durable outputs use input-relative refs and do not bind checkout paths."""

    packages = []
    for name in ("copy-a", "copy-b"):
        package = tmp_path / name / "matched"
        shutil.copytree(DOSSIER_FIXTURES / "matched_seed118", package)
        packages.append(package)

    first = render_case_dossier(packages[0] / "input.json", tmp_path / "render-a")
    second = render_case_dossier(packages[1] / "input.json", tmp_path / "render-b")

    for attribute in (
        "svg_path",
        "pdf_path",
        "caption_path",
        "sidecar_path",
        "manifest_path",
        "catalog_path",
    ):
        assert getattr(first, attribute).read_bytes() == getattr(second, attribute).read_bytes()
    source_bindings = first.manifest["source_bindings"]
    declared_paths = [
        source_bindings["portfolio"]["path"],
        source_bindings["campaign_atlas"]["path"],
        *(item["path"] for item in source_bindings["process_traces"]),
    ]
    assert all(not Path(path).is_absolute() for path in declared_paths)
    assert str(tmp_path) not in first.manifest_path.read_text(encoding="utf-8")


def test_reciprocal_pair_contracts_must_agree_on_admission_semantics() -> None:
    """A contradictory right-hand prefix contract cannot pass the left-hand gate."""

    fixture = DOSSIER_FIXTURES / "matched_seed118"
    left = json.loads((fixture / "process_left.json").read_text(encoding="utf-8"))
    right = json.loads((fixture / "process_right.json").read_text(encoding="utf-8"))
    contradictory = copy.deepcopy(right)
    contradictory["pair_compatibility"]["shared_prefix"]["shared_prefix"] = False

    with pytest.raises(Exception, match="reciprocal_pair_contract_disagreement"):
        dossier_module._validate_pair(
            "matched_start_planner",
            {"left": left, "right": contradictory},
        )


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


@pytest.mark.parametrize(
    "text",
    (
        "A Difference-Curve appears after the start.",
        "The plotted DIFFERENCE_CURVES separate.",
        "The PIVOT/TIME is 0.2 s.",
        "This proposes a causal—hinge.",
        "Adjacent seed statistical significance is claimed.",
        "Adjacent-seed significance is claimed.",
        "Adjacent seeds are statistically significant.",
        "A divergence_point is identified.",
        "A localized divergence is identified.",
        "The paths peel apart near the doorway.",
        "The split is statistically reliable.",
        "Seed 113 materially outperforms seed 114.",
        "The planner change explains the later collision.",
    ),
)
@pytest.mark.parametrize(
    "grammar",
    ("matched_start_planner", "same_cell_seed_sensitivity"),
)
def test_controlled_narrative_rejects_all_free_form_reviewer_examples(
    tmp_path: Path,
    text: str,
    grammar: str,
) -> None:
    """Neither grammar admits caller-authored claim text on any output surface."""

    input_path = _write_input(tmp_path, grammar=grammar)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload["narrative"]["free_form"] = text
    _write_json(input_path, payload)

    with pytest.raises(Exception, match="case_dossier_input_invalid"):
        render_case_dossier(input_path, tmp_path / "forbidden-narrative")


def test_process_trace_display_labels_are_renderer_owned(tmp_path: Path) -> None:
    """Caller-authored trace prose is not part of the dossier input contract."""

    input_path = _write_matched_input(tmp_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload["sources"]["process_traces"][0]["label"] = "Reviewer says this planner decisively wins."
    _write_json(input_path, payload)

    with pytest.raises(Exception, match="case_dossier_input_invalid"):
        render_case_dossier(input_path, tmp_path / "forbidden-trace-label")


@pytest.mark.parametrize(
    ("identity_overrides", "controller_signals", "error_code"),
    (
        (
            {"case_id": "reviewer says this case proves superiority"},
            False,
            "case_dossier_input_invalid",
        ),
        ({"scenario_id": "reviewer supplied scenario sentence"}, False, "identity_token_invalid"),
        ({"planner_id": "planner wins decisively"}, False, "identity_token_invalid"),
        (
            {"episode_id": "episode selected because it proves the claim"},
            False,
            "identity_token_invalid",
        ),
        (
            {"controller_state": "controller proves the planner is safer"},
            True,
            "identity_token_invalid",
        ),
    ),
)
def test_sentence_like_identity_values_fail_before_rendering(
    tmp_path: Path,
    identity_overrides: dict[str, str],
    controller_signals: bool,
    error_code: str,
) -> None:
    """Identity and categorical slots admit bounded tokens, never prose."""

    input_path = _write_input(
        tmp_path,
        grammar="matched_start_planner",
        controller_signals=controller_signals,
        identity_overrides=identity_overrides,
    )

    with pytest.raises(Exception, match=error_code):
        render_case_dossier(input_path, tmp_path / "forbidden-identity")


@pytest.mark.parametrize(
    ("slot", "value"),
    (
        ("recorded_outcome", "reviewer_declares_a_win"),
        ("missing_trace_ids", "trace missing because this method is unsafe"),
    ),
)
def test_input_vocabularies_reject_semantic_injection(
    tmp_path: Path,
    slot: str,
    value: str,
) -> None:
    """Outcome vocabularies and inventory identity slots are schema closed."""

    input_path = _write_matched_input(tmp_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if slot == "recorded_outcome":
        payload["sources"]["process_traces"][0][slot] = value
    else:
        payload["ensemble_context"][slot] = [value]
    _write_json(input_path, payload)

    with pytest.raises(Exception, match="case_dossier_input_invalid"):
        render_case_dossier(input_path, tmp_path / "forbidden-vocabulary")


def test_campaign_atlas_outcome_keys_use_closed_vocabulary(tmp_path: Path) -> None:
    """Atlas outcome names cannot become caller-authored prose in the context panel."""

    input_path = _write_matched_input(tmp_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    atlas_path = Path(payload["sources"]["campaign_atlas"]["path"])
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    atlas["cells"][0]["outcome_counts"]["reviewer_claim"] = atlas["cells"][0]["outcome_counts"].pop(
        "success"
    )
    atlas["cells"][0]["outcome_ci"]["reviewer_claim"] = atlas["cells"][0]["outcome_ci"].pop(
        "success"
    )
    _write_json(atlas_path, atlas)
    payload["sources"]["campaign_atlas"]["sha256"] = _sha256(atlas_path)
    _write_json(input_path, payload)

    with pytest.raises(Exception, match="campaign_atlas_outcome_key_invalid"):
        render_case_dossier(input_path, tmp_path / "forbidden-atlas-outcome")


def test_narrative_rejects_arbitrary_semantic_mutation(tmp_path: Path) -> None:
    """Only the controlled grammar template may reach dossier claim surfaces."""

    input_path = _write_doorway_input(tmp_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload["narrative"]["template_id"] = "same_cell_small_neighbourhood_split.v1"
    _write_json(input_path, payload)

    with pytest.raises(Exception, match="narrative_template_invalid"):
        render_case_dossier(input_path, tmp_path / "mutated-narrative")


def test_same_cell_claim_surfaces_use_exact_controlled_abstention_template(
    tmp_path: Path,
) -> None:
    """The same-cell grammar emits only its versioned descriptive abstention."""

    input_path = _write_doorway_input(tmp_path)
    bundle = render_case_dossier(input_path, tmp_path / "causal-abstention")

    expected = {
        "observed_signature": (
            "The recorded traces show distinct observed paths and terminal outcomes "
            "from different recorded starts."
        ),
        "competing_explanation": (
            "The different recorded starts preclude attribution to seed choice."
        ),
        "causal_status": "causal_abstention",
        "generalization_limit": (
            "This selected comparison is descriptive only and supports no mechanistic "
            "or population inference."
        ),
    }
    sidecar = json.loads(bundle.sidecar_path.read_text(encoding="utf-8"))
    caption = bundle.caption_path.read_text(encoding="utf-8")
    assert bundle.manifest["claim_template_id"] == "same_cell_distinct_start_abstention.v1"
    assert bundle.manifest["claim_fields"] == expected
    assert sidecar["claim_template_id"] == "same_cell_distinct_start_abstention.v1"
    assert sidecar["claim_fields"] == expected
    assert all(value in caption for value in expected.values())


@pytest.mark.parametrize(
    ("grammar", "expected_allowed_claim"),
    (
        (
            "matched_start_planner",
            "Different executed planner stacks show different observed processes and "
            "terminal outcomes under the matched recorded start.",
        ),
        (
            "same_cell_seed_sensitivity",
            "The recorded traces show distinct observed paths and terminal outcomes "
            "from different recorded starts.",
        ),
    ),
)
def test_selected_portfolio_claim_is_the_grammar_owned_controlled_claim(
    tmp_path: Path,
    grammar: str,
    expected_allowed_claim: str,
) -> None:
    """Selection prose must be byte-equal to the renderer's controlled claim contract."""

    input_path = _write_input(tmp_path, grammar=grammar)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    portfolio = json.loads(Path(payload["sources"]["portfolio"]["path"]).read_text("utf-8"))
    selected = portfolio["selected"][0]

    assert selected["allowed_claim"] == expected_allowed_claim
    assert selected["claim"]["allowed"] == [expected_allowed_claim]


@pytest.mark.parametrize(
    "mutated_claim",
    (
        "Reviewer-authored superiority claim.",
        "This selected planner is safer than the other planner.",
    ),
)
def test_selected_portfolio_cannot_override_the_controlled_claim(
    tmp_path: Path,
    mutated_claim: str,
) -> None:
    """A validly hashed portfolio still cannot inject selection prose."""

    input_path = _write_matched_input(tmp_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    portfolio_path = Path(payload["sources"]["portfolio"]["path"])
    portfolio = json.loads(portfolio_path.read_text(encoding="utf-8"))
    portfolio["selected"][0]["allowed_claim"] = mutated_claim
    portfolio["selected"][0]["claim"]["allowed"] = [mutated_claim]
    ledger_record = next(
        item
        for item in portfolio["ledger"]
        if item["case_id"] == portfolio["selected"][0]["case_id"]
    )
    ledger_record["allowed_claim"] = mutated_claim
    portfolio = finalize_manifest(portfolio)
    _write_json(portfolio_path, portfolio)
    payload["sources"]["portfolio"]["sha256"] = _sha256(portfolio_path)

    atlas_path = Path(payload["sources"]["campaign_atlas"]["path"])
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    atlas["selection_manifest_hash"] = portfolio["content_sha256"]
    _write_json(atlas_path, atlas)
    payload["sources"]["campaign_atlas"]["sha256"] = _sha256(atlas_path)
    _write_json(input_path, payload)

    with pytest.raises(Exception, match="selected_claim_contract_mismatch"):
        render_case_dossier(input_path, tmp_path / "forbidden-selected-claim")


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
    payload["ensemble_context"]["reason"] = "production_ensemble_inventory_unavailable"
    payload["case_id"] = (
        "ch7-role-planner-upset--classic-realworld-double-bottleneck-high--goal-vs-ppo--seed-118"
    )
    payload["sources"]["portfolio"] = _file_ref(production_portfolio)
    _write_json(input_path, payload)

    with pytest.raises(Exception, match="production_case_not_selected"):
        render_case_dossier(input_path, tmp_path / "production-stop")


def test_synthetic_recorded_outcomes_are_atlas_checked_and_non_authoritative(
    tmp_path: Path,
) -> None:
    """Fixture declarations stay visibly non-authoritative on every durable surface."""

    input_path = _write_matched_input(tmp_path)

    bundle = render_case_dossier(input_path, tmp_path / "synthetic-outcomes")

    sidecar = json.loads(bundle.sidecar_path.read_text(encoding="utf-8"))
    atlas_bindings = {
        role: {
            "status": "resolved",
            "scenario_family": "classic_realworld_double_bottleneck_high",
            "planner": planner,
            "release_arm_id": None,
            "resolution": "unique_scenario_planner_cell",
            "authority_source": None,
        }
        for role, planner in (("left", "goal"), ("right", "ppo"))
    }
    expected = {
        "left": {
            "status": "available",
            "value": "success",
            "source": "case_dossier_input.sources.process_traces[role=left].recorded_outcome",
            "authority": "non_authoritative_synthetic_fixture_declaration",
            "atlas_outcome_key_validated": True,
            "atlas_cell_binding": atlas_bindings["left"],
        },
        "right": {
            "status": "available",
            "value": "collision",
            "source": "case_dossier_input.sources.process_traces[role=right].recorded_outcome",
            "authority": "non_authoritative_synthetic_fixture_declaration",
            "atlas_outcome_key_validated": True,
            "atlas_cell_binding": atlas_bindings["right"],
        },
    }
    assert sidecar["recorded_outcomes"] == expected
    assert {
        item["role"]: item["recorded_outcome"]
        for item in bundle.manifest["source_bindings"]["process_traces"]
    } == expected
    svg = bundle.svg_path.read_text(encoding="utf-8")
    assert "NON-AUTHORITATIVE synthetic declaration" in svg
    assert "atlas-key checked" in svg


def test_recorded_outcome_must_be_a_key_in_its_bound_atlas_cell(tmp_path: Path) -> None:
    """A free-form outcome label cannot acquire authority through rendering."""

    input_path = _write_matched_input(tmp_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload["sources"]["process_traces"][0]["recorded_outcome"] = "near_miss"
    _write_json(input_path, payload)

    with pytest.raises(Exception, match="case_dossier_input_invalid"):
        render_case_dossier(input_path, tmp_path / "unknown-outcome")


def test_production_recorded_outcomes_bind_matching_typed_terminal_evidence(
    tmp_path: Path,
) -> None:
    """Production labels are accepted only from a typed terminal trace outcome."""

    input_path = _write_input(
        tmp_path,
        grammar="matched_start_planner",
        mode="production",
        terminal_outcomes=("success", "collision"),
    )

    bundle = render_case_dossier(input_path, tmp_path / "production-outcomes")

    records = {
        item["role"]: item["recorded_outcome"]
        for item in bundle.manifest["source_bindings"]["process_traces"]
    }
    assert records["left"] == {
        "status": "available",
        "value": "success",
        "source": ("source_trace.content_receipt.content_contract.frames[-1].planner.outcome"),
        "authority": "typed_terminal_trace_evidence",
        "atlas_outcome_key_validated": True,
        "atlas_cell_binding": {
            "status": "resolved",
            "scenario_family": "classic_realworld_double_bottleneck_high",
            "planner": "goal",
            "release_arm_id": None,
            "resolution": "unique_scenario_planner_cell",
            "authority_source": None,
        },
    }
    assert records["right"]["value"] == "collision"
    assert records["right"]["authority"] == "typed_terminal_trace_evidence"


@pytest.mark.parametrize(
    ("terminal_outcomes", "error_code"),
    (
        ((None, None), "production_typed_terminal_outcome_unavailable"),
        (("collision", "collision"), "production_recorded_outcome_mismatch"),
        (("ambiguous", "collision"), "production_typed_terminal_outcome_ambiguous"),
    ),
)
def test_production_recorded_outcomes_fail_closed_without_matching_typed_evidence(
    tmp_path: Path,
    terminal_outcomes: tuple[str | None, str | None],
    error_code: str,
) -> None:
    """Missing or contradictory production terminal evidence stops before rendering."""

    input_path = _write_input(
        tmp_path,
        grammar="matched_start_planner",
        mode="production",
        terminal_outcomes=terminal_outcomes,
    )

    with pytest.raises(Exception, match=error_code):
        render_case_dossier(input_path, tmp_path / "production-outcome-stop")


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
    assert status["time_space"]["status"] == "unavailable"
    assert status["time_space"]["reason"] == "route_and_conflict_projection_unavailable"
    assert status["time_space"]["occupancy_ribbon"]["status"] == "unavailable"
    assert status["radial_closing_speed"]["status"] == "unavailable"
    assert status["radial_closing_speed"]["reason"] == "relative_velocity_unavailable"
    assert status["radial_closing_speed"]["closest_approach"]["left"]["status"] == "unavailable"
    controller = status["controller_state"]
    assert controller["status"] == "unavailable"
    assert controller["reason"] == "controller_state_signal_absent"
    assert controller["artist_count"] == 0
    assert set(controller["signals"]) == {
        "command_source",
        "controller_state",
        "fallback_state",
        "guard_state",
    }
    assert all(record["status"] == "unavailable" for record in controller["signals"].values())
    assert status["ensemble_context"] == {
        "status": "unavailable",
        "reason": "synthetic_fixture_no_ensemble_inventory",
    }
    svg = bundle.svg_path.read_text(encoding="utf-8")
    assert "ROUTE / CONFLICT TIME–SPACE" in svg
    assert "UNAVAILABLE" in svg


def test_nonzero_commanded_turn_renders_a_separate_aligned_panel(tmp_path: Path) -> None:
    """Recorded commanded angular velocity is visible as its own process view."""

    input_path = _write_input(
        tmp_path,
        grammar="matched_start_planner",
        nonzero_turn=True,
    )

    bundle = render_case_dossier(input_path, tmp_path / "turn-render")

    status = bundle.manifest["panel_status"]["turn_rate"]
    assert status["tick_label_overlap_checked"] is True
    assert status["status"] == "available"
    assert status["commanded"]["left"]["nonzero_observed"] is True
    assert status["commanded"]["right"]["nonzero_observed"] is True
    assert "Commanded / executed turn rate" in bundle.svg_path.read_text(encoding="utf-8")


def test_negative_commanded_turn_keeps_execution_status_out_of_the_data_region(
    tmp_path: Path,
) -> None:
    """A negative turn cannot collide with the execution-availability annotation."""

    input_path = _write_input(
        tmp_path,
        grammar="matched_start_planner",
        nonzero_turn=True,
        negative_turn=True,
    )

    bundle = render_case_dossier(input_path, tmp_path / "negative-turn-render")

    status = bundle.manifest["panel_status"]["turn_rate"]
    assert status["commanded"]["left"]["nonzero_observed"] is True
    assert status["commanded"]["right"]["nonzero_observed"] is True
    assert "EXECUTED UNAVAILABLE — L/R" in bundle.svg_path.read_text(encoding="utf-8")


def test_missing_executed_turn_is_explicitly_unavailable(tmp_path: Path) -> None:
    """The renderer never derives executed turn from headings or commands."""

    input_path = _write_input(
        tmp_path,
        grammar="matched_start_planner",
        nonzero_turn=True,
    )

    bundle = render_case_dossier(input_path, tmp_path / "missing-executed-turn")

    status = bundle.manifest["panel_status"]["turn_rate"]
    for role in ("left", "right"):
        assert status["executed"][role] == {
            "status": "unavailable",
            "reason": "explicit_executed_angular_velocity_unavailable",
            "source": (
                "source_trace.content_receipt.content_contract.frames[].planner."
                "executed_action.angular_velocity"
            ),
            "artist_count": 0,
            "nonzero_observed": False,
        }
    svg = bundle.svg_path.read_text(encoding="utf-8")
    assert "Commanded / executed turn rate" in svg
    assert "EXECUTED UNAVAILABLE — L/R" in svg
    assert status["executed_unavailable_note"]["artist_count"] == 1
    assert bundle.manifest["renderer"]["structural_panel_text_bounds_checked"] is True


def test_recorded_executed_angular_velocity_renders_as_available(tmp_path: Path) -> None:
    """Recorded executed angular velocity exercises the available turn-rate branch."""

    input_path = _write_input(
        tmp_path,
        grammar="matched_start_planner",
        nonzero_turn=True,
        executed_turn=True,
        controller_signals=True,
    )

    bundle = render_case_dossier(input_path, tmp_path / "available-executed-turn")

    status = bundle.manifest["panel_status"]["turn_rate"]
    for role in ("left", "right"):
        assert status["executed"][role] == {
            "status": "available",
            "reason": "explicit_executed_angular_velocity",
            "source": (
                "source_trace.content_receipt.content_contract.frames[].planner."
                "executed_action.angular_velocity"
            ),
            "artist_count": 1,
            "nonzero_observed": True,
        }
    assert status["executed_unavailable_note"] == {
        "status": "not_applicable",
        "reason": "executed_angular_velocity_available",
        "roles": [],
        "artist_count": 0,
    }
    svg = bundle.svg_path.read_text(encoding="utf-8")
    assert "Commanded / executed turn rate" in svg
    assert "EXECUTED UNAVAILABLE" not in svg
    assert bundle.manifest["panel_status"]["controller_state"]["status"] == "available"
    assert bundle.manifest["renderer"]["panel_tick_label_overlap_checked"] is True


def test_source_controller_signals_render_categorical_strip(tmp_path: Path) -> None:
    """Recorded controller, guard, fallback, and command-source values become artists."""

    input_path = _write_input(
        tmp_path,
        grammar="matched_start_planner",
        controller_signals=True,
    )

    bundle = render_case_dossier(input_path, tmp_path / "controller-strip")

    status = bundle.manifest["panel_status"]["controller_state"]
    assert status["status"] == "available"
    assert status["artist_count"] > 0
    assert status["semantic_event_cursor_count"] > 0
    assert set(status["signals"]) == {
        "command_source",
        "controller_state",
        "fallback_state",
        "guard_state",
    }
    assert all(record["status"] == "available" for record in status["signals"].values())
    svg = bundle.svg_path.read_text(encoding="utf-8")
    assert "Controller signals · directly labelled L/R sublanes" in svg
    assert "tracking" in svg
    assert "planner" in svg


def test_controller_strip_is_decodable_source_bound_and_geometry_checked(
    tmp_path: Path,
) -> None:
    """Four signal rows expose L/R sublanes and directly label every color."""

    input_path = _write_input(
        tmp_path,
        grammar="matched_start_planner",
        controller_signals=True,
    )

    bundle = render_case_dossier(input_path, tmp_path / "controller-strip-contract")

    status = bundle.manifest["panel_status"]["controller_state"]
    assert status["layout"] == "four_signal_rows_with_left_right_sublanes.v1"
    assert status["decoding"] == {
        "method": "direct_segment_labels",
        "role_encoding": "labelled_left_right_sublanes",
        "all_values_labelled": True,
    }
    assert status["text_bounds_checked"] is True
    assert status["text_overlap_checked"] is True
    assert status["tick_label_overlap_checked"] is True
    assert status["signal_row_count"] == 4
    for signal, record in status["signals"].items():
        assert record["source"] == (
            f"source_trace.content_receipt.content_contract.frames[].planner.{signal}"
        )
        assert record["row_index"] in range(4)
        assert all(style["label_rendered"] is True for style in record["value_styles"].values())
        assert record["roles"]["left"]["sublane"] == "left"
        assert record["roles"]["right"]["sublane"] == "right"

    catalog = json.loads(bundle.catalog_path.read_text(encoding="utf-8"))
    legend = catalog["artifacts"][0]["figure_semantics"]["legend_series"]
    assert "controller states (direct L/R labels)" in legend
    svg = bundle.svg_path.read_text(encoding="utf-8")
    for value in ("tracking", "braking", "planner", "clear", "active", "inactive"):
        assert value in svg


def test_semantic_style_key_is_complete_and_matches_catalog_legend(tmp_path: Path) -> None:
    """Every visual encoding has a compact visible key and truthful catalog entry."""

    input_path = _write_matched_input(tmp_path)

    bundle = render_case_dossier(input_path, tmp_path / "semantic-key")

    catalog = json.loads(bundle.catalog_path.read_text(encoding="utf-8"))
    legend_series = catalog["artifacts"][0]["figure_semantics"]["legend_series"]
    expected_semantics = {
        "robot trajectory",
        "focal actor trajectory",
        "primary surface clearance",
        "secondary centre distance",
        "commanded speed / turn",
        "executed speed / turn (when recorded)",
        "diagnostic threshold",
        "semantic event cursor",
        "recorded occupancy ribbon",
        "controller states (direct L/R labels)",
    }
    assert expected_semantics <= set(legend_series)
    assert {"L ID · goal/118", "R ID · ppo/118"} <= set(legend_series)
    assert catalog["artifacts"][0]["figure_semantics"]["legend_complete"] is True
    svg = bundle.svg_path.read_text(encoding="utf-8")
    for label in expected_semantics - {"controller states (direct L/R labels)"}:
        assert label in svg
    assert "controller states" in svg
    assert "(direct L/R labels)" in svg


def test_validated_atlas_intervals_are_bound_displayed_and_catalogued(tmp_path: Path) -> None:
    """Outcome intervals are consumed visibly and bound to every durable metadata surface."""

    input_path = _write_matched_input(tmp_path)

    bundle = render_case_dossier(input_path, tmp_path / "atlas-intervals")

    uncertainty = bundle.manifest["source_bindings"]["campaign_atlas"]["uncertainty"]
    assert uncertainty["status"] == "available"
    assert uncertainty["source"] == "campaign_atlas.v2.cells[].outcome_ci"
    assert uncertainty["method"] == "campaign_atlas_outcome_ci_validated_and_consumed"
    assert uncertainty["cells"][0]["outcomes"]["collision"] == {
        "count": 6,
        "estimate": 0.4,
        "interval": [0.0, 1.0],
    }
    sidecar = json.loads(bundle.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["uncertainty"] == uncertainty
    assert bundle.manifest["panel_status"]["cell_context"]["uncertainty"] == uncertainty
    catalog = json.loads(bundle.catalog_path.read_text(encoding="utf-8"))
    semantics = next(
        artifact["figure_semantics"]
        for artifact in catalog["artifacts"]
        if artifact["artifact_id"] == "release_context"
    )
    assert semantics["uncertainty_declared"] is True
    assert semantics["uncertainty_method"] == ("campaign_atlas_outcome_ci_validated_and_consumed")
    svg = bundle.svg_path.read_text(encoding="utf-8")
    assert "collision 6/15 CI[0.00,1.00]" in svg
    assert "success 9/15 CI[0.00,1.00]" in svg


def test_durable_outputs_whitelist_selection_provenance_and_atlas_cells(
    tmp_path: Path,
) -> None:
    """Unknown source/atlas prose never crosses a durable or visible output boundary."""

    forbidden = "Reviewer claim_note: this planner is scientifically superior."
    input_path = _write_matched_input(tmp_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    portfolio_path = Path(payload["sources"]["portfolio"]["path"])
    portfolio = json.loads(portfolio_path.read_text(encoding="utf-8"))
    selected = portfolio["selected"][0]
    ledger_record = next(
        item for item in portfolio["ledger"] if item["case_id"] == selected["case_id"]
    )
    for record in (selected, ledger_record):
        record["source_boundary"]["claim_note"] = forbidden
        record["source"]["boundary"]["claim_note"] = forbidden
    portfolio = finalize_manifest(portfolio)
    _write_json(portfolio_path, portfolio)
    payload["sources"]["portfolio"]["sha256"] = _sha256(portfolio_path)

    atlas_path = Path(payload["sources"]["campaign_atlas"]["path"])
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    atlas["selection_manifest_hash"] = portfolio["content_sha256"]
    atlas["cells"][0]["claim_note"] = forbidden
    _write_json(atlas_path, atlas)
    payload["sources"]["campaign_atlas"]["sha256"] = _sha256(atlas_path)
    _write_json(input_path, payload)

    bundle = render_case_dossier(input_path, tmp_path / "projection-whitelist")

    for path in (
        bundle.svg_path,
        bundle.pdf_path,
        bundle.caption_path,
        bundle.sidecar_path,
        bundle.manifest_path,
        bundle.catalog_path,
    ):
        assert forbidden.encode("utf-8") not in path.read_bytes()
    selection = bundle.manifest["selection"]
    assert set(selection["eligibility"]) == {
        "eligible",
        "status",
        "execution_mode",
        "telemetry_grade",
        "typed_outcome_semantics",
        "initial_state_match",
        "outcome_match",
    }
    assert "source_boundary" not in selection
    assert set(selection["source_provenance"]) == {
        "synthetic_fixture",
        "visualization_only_reexecution",
        "release_id",
        "telemetry_grade",
        "hashes",
        "release_arm_bindings",
    }
    for cell in bundle.manifest["source_bindings"]["campaign_atlas"]["release_cells"]:
        assert set(cell) == {
            "scenario_family",
            "planner",
            "release_arm_id",
            "eligible",
            "n_total",
            "outcome_counts",
            "outcome_ci",
        }


def test_catalog_separates_two_trace_diagnostic_support_from_release_context(
    tmp_path: Path,
) -> None:
    """Release confidence counts never masquerade as clearance-trace support."""

    input_path = _write_matched_input(tmp_path)

    bundle = render_case_dossier(input_path, tmp_path / "support-semantics")

    catalog = json.loads(bundle.catalog_path.read_text(encoding="utf-8"))
    artifacts = {artifact["artifact_id"]: artifact for artifact in catalog["artifacts"]}
    clearance = artifacts["case_dossier"]["figure_semantics"]
    assert clearance["metric_id"] == "proxy_envelope_surface_clearance_diagnostic"
    assert clearance["support"] == 2
    assert clearance["denominator"] == 2
    assert clearance["uncertainty_declared"] is True
    assert clearance["uncertainty_method"] == (
        "release_context_only:campaign_atlas_outcome_ci_validated_and_consumed"
    )

    release = artifacts["release_context"]["figure_semantics"]
    assert release["metric_id"] == "campaign_atlas_outcome_release_context"
    assert release["support"] == 30
    assert release["denominator"] == 30
    assert release["uncertainty_declared"] is True
    assert release["uncertainty_method"] == ("campaign_atlas_outcome_ci_validated_and_consumed")
    assert "Release-context confidence only" in artifacts["release_context"]["claim_boundary"]


def test_atlas_interval_estimates_accept_the_producers_six_decimal_rounding(
    tmp_path: Path,
) -> None:
    """A valid 1/3 release cell survives the atlas producer's six-decimal projection."""

    input_path = _write_matched_input(tmp_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    atlas_path = Path(payload["sources"]["campaign_atlas"]["path"])
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    cell = atlas["cells"][0]
    cell["n_total"] = 3
    cell["outcome_counts"] = {"collision": 1, "success": 2}
    cell["outcome_ci"] = {
        "collision": [0.333333, 0.0, 1.0],
        "success": [0.666667, 0.0, 1.0],
    }
    _write_json(atlas_path, atlas)
    payload["sources"]["campaign_atlas"]["sha256"] = _sha256(atlas_path)
    _write_json(input_path, payload)

    bundle = render_case_dossier(input_path, tmp_path / "rounded-atlas-ci")

    uncertainty = bundle.manifest["source_bindings"]["campaign_atlas"]["uncertainty"]
    assert uncertainty["cells"][0]["outcomes"]["collision"]["estimate"] == 0.333333


def test_atlas_duplicate_release_arms_without_provenance_fail_closed(
    tmp_path: Path,
) -> None:
    """Planner identity alone may not choose among multiple release arms."""

    input_path = _write_matched_input(tmp_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    atlas_path = Path(payload["sources"]["campaign_atlas"]["path"])
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    goal_cell = atlas["cells"][0]
    goal_cell["release_arm_id"] = "goal__native"
    alternate = copy.deepcopy(goal_cell)
    alternate["release_arm_id"] = "goal__alternate"
    atlas["cells"].append(alternate)
    _write_json(atlas_path, atlas)
    payload["sources"]["campaign_atlas"]["sha256"] = _sha256(atlas_path)
    _write_json(input_path, payload)

    with pytest.raises(Exception, match="campaign_atlas_cell_ambiguous"):
        render_case_dossier(input_path, tmp_path / "ambiguous-atlas-arm")


@pytest.mark.parametrize(
    "release_arm_id",
    ("", {"id": "goal__native"}, "arm selected because it proves the claim"),
)
def test_atlas_release_arm_id_has_stable_identity_grammar(
    tmp_path: Path,
    release_arm_id: object,
) -> None:
    """Malformed arm values fail before candidate keys or durable bindings are built."""

    input_path = _write_matched_input(tmp_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    atlas_path = Path(payload["sources"]["campaign_atlas"]["path"])
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    atlas["cells"][0]["release_arm_id"] = release_arm_id
    _write_json(atlas_path, atlas)
    payload["sources"]["campaign_atlas"]["sha256"] = _sha256(atlas_path)
    _write_json(input_path, payload)

    with pytest.raises(Exception, match="release_arm_id_invalid"):
        render_case_dossier(input_path, tmp_path / "invalid-atlas-arm")


def test_authoritative_selection_release_arm_resolves_and_is_bound_durably(
    tmp_path: Path,
) -> None:
    """An explicit selection arm selects exactly one atlas cell and records its authority."""

    input_path = _write_input(
        tmp_path,
        grammar="matched_start_planner",
        release_arm_bindings={"left": "goal__native"},
    )
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    atlas_path = Path(payload["sources"]["campaign_atlas"]["path"])
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    goal_cell = atlas["cells"][0]
    goal_cell["release_arm_id"] = "goal__native"
    alternate = copy.deepcopy(goal_cell)
    alternate["release_arm_id"] = "goal__alternate"
    atlas["cells"].append(alternate)
    _write_json(atlas_path, atlas)
    payload["sources"]["campaign_atlas"]["sha256"] = _sha256(atlas_path)
    _write_json(input_path, payload)

    bundle = render_case_dossier(input_path, tmp_path / "resolved-atlas-arm")

    campaign = bundle.manifest["source_bindings"]["campaign_atlas"]
    binding = campaign["resolved_cell_bindings"]["left"]
    assert binding == {
        "status": "resolved",
        "scenario_family": "classic_realworld_double_bottleneck_high",
        "planner": "goal",
        "release_arm_id": "goal__native",
        "resolution": "authoritative_release_arm_id",
        "authority_source": "selection.source_boundary.release_arm_bindings.left",
    }
    assert [
        cell["release_arm_id"] for cell in campaign["release_cells"] if cell["planner"] == "goal"
    ] == ["goal__native"]
    left_outcome = bundle.manifest["source_bindings"]["process_traces"][0]["recorded_outcome"]
    assert left_outcome["atlas_cell_binding"] == binding
    sidecar = json.loads(bundle.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["atlas_cell_bindings"]["left"] == binding


@pytest.mark.parametrize("release_arm_id", ("", "arm chosen because this planner is safer"))
def test_authoritative_release_arm_uses_the_same_identity_grammar(
    tmp_path: Path,
    release_arm_id: str,
) -> None:
    """Selection-side release-arm provenance cannot carry caller prose."""

    input_path = _write_input(
        tmp_path,
        grammar="matched_start_planner",
        release_arm_bindings={"left": release_arm_id},
    )

    with pytest.raises(Exception, match="release_arm_id_invalid"):
        render_case_dossier(input_path, tmp_path / "invalid-authoritative-arm")


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    (
        ("ineligible", "campaign_atlas_cell_ineligible"),
        ("key_mismatch", "campaign_atlas_outcome_ci_invalid"),
        ("triple_shape", "campaign_atlas_outcome_ci_invalid"),
        ("nonfinite", "campaign_atlas_outcome_ci_invalid"),
        ("ordering", "campaign_atlas_outcome_ci_invalid"),
        ("range", "campaign_atlas_outcome_ci_invalid"),
        ("estimate_mismatch", "campaign_atlas_outcome_ci_invalid"),
    ),
)
def test_invalid_atlas_outcome_intervals_fail_closed_before_render(
    tmp_path: Path,
    mutation: str,
    error_code: str,
) -> None:
    """Malformed, non-finite, or count-inconsistent atlas intervals never become claims."""

    input_path = _write_matched_input(tmp_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    atlas_path = Path(payload["sources"]["campaign_atlas"]["path"])
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    cell = atlas["cells"][0]
    if mutation == "ineligible":
        cell["eligible"] = False
        cell["ineligible_reason"] = "fixture_negative"
    elif mutation == "key_mismatch":
        cell["outcome_ci"].pop("collision")
    elif mutation == "triple_shape":
        cell["outcome_ci"]["collision"] = [0.4, 0.0]
    elif mutation == "nonfinite":
        cell["outcome_ci"]["collision"] = [float("nan"), 0.0, 1.0]
    elif mutation == "ordering":
        cell["outcome_ci"]["collision"] = [0.4, 0.8, 0.2]
    elif mutation == "range":
        cell["outcome_ci"]["collision"] = [0.4, -0.1, 1.0]
    elif mutation == "estimate_mismatch":
        cell["outcome_ci"]["collision"] = [0.5, 0.0, 1.0]
    _write_json(atlas_path, atlas)
    payload["sources"]["campaign_atlas"]["sha256"] = _sha256(atlas_path)
    _write_json(input_path, payload)

    with pytest.raises(Exception, match=error_code):
        render_case_dossier(input_path, tmp_path / "invalid-atlas-ci")


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


def _write_input(  # noqa: PLR0913 - fixture controls independently pin render contracts
    tmp_path: Path,
    *,
    grammar: str,
    with_geometry: bool = True,
    missing_velocity: bool = False,
    nonzero_turn: bool = False,
    negative_turn: bool = False,
    executed_turn: bool = False,
    controller_signals: bool = False,
    mode: str = "synthetic_fixture",
    terminal_outcomes: tuple[str | None, str | None] = (None, None),
    release_arm_bindings: dict[str, str] | None = None,
    identity_overrides: dict[str, str] | None = None,
) -> Path:
    identity_overrides = identity_overrides or {}
    matched = grammar == "matched_start_planner"
    scenario_id = identity_overrides.get(
        "scenario_id",
        "classic_realworld_double_bottleneck_high" if matched else "classic_doorway_medium",
    )
    case_id = identity_overrides.get(
        "case_id",
        "fixture-seed-118-planner-upset" if matched else "fixture-doorway-seeds-113-114",
    )
    left_planner = identity_overrides.get("planner_id", "goal" if matched else "ppo")
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
        nonzero_turn=nonzero_turn,
        negative_turn=negative_turn,
        executed_turn=executed_turn,
        controller_signals=controller_signals,
        terminal_outcome=terminal_outcomes[0],
        episode_id=identity_overrides.get("episode_id"),
        controller_state=identity_overrides.get("controller_state"),
    )
    right = _trace_payload(
        trace_id=f"fixture-{scenario_id}-{right_planner}-seed-{right_seed}",
        planner_id=right_planner,
        seed=right_seed,
        scenario_id=scenario_id,
        start_offset=right_offset,
        final_position=(0.6, 0.2),
        missing_velocity=missing_velocity,
        nonzero_turn=nonzero_turn,
        negative_turn=negative_turn,
        executed_turn=executed_turn,
        controller_signals=controller_signals,
        terminal_outcome=terminal_outcomes[1],
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
        build_ch7_worked_example_portfolio(
            _portfolio_config(
                case_id,
                grammar=grammar,
                synthetic_fixture=mode == "synthetic_fixture",
                release_arm_bindings=release_arm_bindings,
            )
        )
    )
    portfolio_path = _write_json(tmp_path / "portfolio.json", portfolio)
    atlas = {
        "schema_version": "campaign_atlas.v2",
        "campaign_id": (
            "synthetic-fixture-release"
            if mode == "synthetic_fixture"
            else "production-controlled-release"
        ),
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
        "dossier_id": "fixture-matched-dossier" if matched else "fixture-doorway-dossier",
        "case_id": case_id,
        "mode": mode,
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
            "reason": (
                "synthetic_fixture_no_ensemble_inventory"
                if mode == "synthetic_fixture"
                else "production_ensemble_inventory_unavailable"
            ),
            "missing_trace_ids": [],
            "ineligible_trace_ids": [],
            "excluded_trace_ids": [],
        },
        "sources": {
            "portfolio": _file_ref(portfolio_path),
            "process_traces": [
                {
                    "role": "left",
                    "recorded_outcome": "success",
                    "source_class": "visualization_only_rerun_diagnostics",
                    **_file_ref(left_path),
                },
                {
                    "role": "right",
                    "recorded_outcome": "collision",
                    "source_class": "visualization_only_rerun_diagnostics",
                    **_file_ref(right_path),
                },
            ],
            "campaign_atlas": {"source_class": "release_statistics", **_file_ref(atlas_path)},
        },
        "layout": {
            "final_width_in": 426.79135 / 72.27,
            "final_height_in": 10.8,
            "minimum_font_pt": 8.25,
            "world_crop_m": [-0.2, 1.4, -0.55, 0.55],
            "metre_scale_m": 0.5,
            "time_range_s": [0.0, 0.4],
            "clearance_range_m": [-0.2, 1.2],
            "speed_range_mps": [-0.2, 1.2],
            "turn_rate_range_rad_s": [-0.5, 0.5],
            "palette_id": "case_dossier_colorblind.v1",
            "threshold_profile": "worked_example_threshold_profile.diagnostic.v1",
        },
        "narrative": {
            "template_id": (
                "matched_start_descriptive.v1"
                if matched
                else "same_cell_distinct_start_abstention.v1"
            )
        },
    }
    return _write_json(tmp_path / "matched-input.json", payload)


def _trace_payload(  # noqa: PLR0913 - fixture controls independently pin signal contracts
    *,
    trace_id: str,
    planner_id: str,
    seed: int,
    scenario_id: str,
    final_position: tuple[float, float],
    start_offset: tuple[float, float] = (0.0, 0.0),
    missing_velocity: bool = False,
    nonzero_turn: bool = False,
    negative_turn: bool = False,
    executed_turn: bool = False,
    controller_signals: bool = False,
    terminal_outcome: str | None = None,
    episode_id: str | None = None,
    controller_state: str | None = None,
) -> dict[str, Any]:
    positions = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.6, 0.0), final_position]
    positions = [
        (position[0] + start_offset[0], position[1] + start_offset[1]) for position in positions
    ]
    frames: list[dict[str, Any]] = []
    for step, position in enumerate(positions):
        planner = {
            "selected_action": {
                "linear_velocity": 1.0 if step < 3 else 0.5,
                "angular_velocity": (-0.3 if negative_turn else 0.3)
                if nonzero_turn and step in {1, 2}
                else 0.0,
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
        }
        if controller_signals:
            planner.update(
                {
                    "controller_state": (
                        controller_state
                        if controller_state is not None
                        else "tracking"
                        if step < 3
                        else "braking"
                    ),
                    "command_source": "planner",
                    "guard_state": "clear" if step < 3 else "active",
                    "fallback_state": "inactive",
                }
            )
        if executed_turn:
            planner["executed_action"] = {
                "linear_velocity": 0.9 if step < 3 else 0.4,
                "angular_velocity": 0.2 if step in {1, 2} else 0.0,
            }
        if step == len(positions) - 1 and terminal_outcome is not None:
            planner["outcome"] = {
                "collision_event": terminal_outcome in {"ambiguous", "collision"},
                "route_complete": terminal_outcome in {"ambiguous", "success"},
                "timeout_event": terminal_outcome == "timeout",
            }
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
                "planner": planner,
            }
        )
    return {
        "schema_version": "simulation_trace_export.v1",
        "trace_id": trace_id,
        "source": {
            "scenario_id": scenario_id,
            "seed": seed,
            "planner_id": planner_id,
            "episode_id": episode_id or f"{trace_id}-episode",
            "generated_by": "issue-6791 synthetic fixture",
        },
        "evidence_boundary": "analysis_workbench_only",
        "coordinate_frame": "world",
        "units": {"position": "m", "heading": "rad", "time": "s", "velocity": "m/s"},
        "frames": frames,
    }


def _portfolio_config(
    case_id: str,
    *,
    grammar: str,
    synthetic_fixture: bool = True,
    release_arm_bindings: dict[str, str] | None = None,
) -> dict[str, Any]:
    matched = grammar == "matched_start_planner"
    role = "planner_upset" if matched else "seed_sensitivity"
    grain = "matched_planner_pair" if matched else "matched_seed_pair"
    topology = "double_bottleneck" if matched else "doorway"
    process_class = "matched_planner_process" if matched else "matched_seed_process"
    release_ref = "configs/scenarios/single/francis2023_narrow_doorway.yaml"
    trace_ref = "robot_sf/benchmark/trace_reexport_packaging.py"
    release_id = (
        "synthetic-fixture-release" if synthetic_fixture else "production-controlled-release"
    )
    release_sha = _sha256(REPO_ROOT / release_ref)
    trace_sha = _sha256(REPO_ROOT / trace_ref)
    allowed_claim = (
        "Different executed planner stacks show different observed processes and "
        "terminal outcomes under the matched recorded start."
        if matched
        else "The recorded traces show distinct observed paths and terminal outcomes "
        "from different recorded starts."
    )
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
    source_boundary = {
        "synthetic_fixture": synthetic_fixture,
        "release_id": release_id,
        "release_rows_sha256": release_sha,
        "expected_release_rows_sha256": release_sha,
        "trace_package_sha256": trace_sha,
        "expected_trace_package_sha256": trace_sha,
        "visualization_only_reexecution": True,
        "telemetry_grade": "controller",
    }
    if release_arm_bindings is not None:
        source_boundary["release_arm_bindings"] = release_arm_bindings
    unit = {
        "case_id": case_id,
        "grain": grain,
        "conceptual_grain": "matched_contrast",
        "conceptual_coverage": ["matched_contrast", "trace"],
        "primary_role": role,
        "claim_grade": "descriptive",
        "secondary_descriptors": ["synthetic_fixture"],
        "allowed_claim": allowed_claim,
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
        "source_boundary": source_boundary,
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
