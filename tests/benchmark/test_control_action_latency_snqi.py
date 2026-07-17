"""Focused tests for the control-action-latency SNQI analyzer (issue #5912).

Covers the Definition-of-Done contract: input checksum / fixed-scope coverage /
execution-mode preservation fail closed (malformed provenance, missing rows,
fallback/degraded exclusion), the canonical command reproduces the registered
packet under the reproducibility tolerance contract, and CSV / raw-row inputs
round-trip consistently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.control_action_latency_preflight import AXIS_KEY
from robot_sf.benchmark.control_action_latency_snqi import (
    ANALYSIS_SCHEMA_VERSION,
    BASELINE_SHA256,
    EXPECTED_SCENARIO_IDS,
    INTERVAL_TOL,
    PAIRWISE_DIFF_TOL,
    POINT_TOL,
    PROBABILITY_TOL,
    WEIGHTS_SHA256,
    SnqiLatencyAnalysisError,
    build_snqi_analysis,
    classify_input_row,
    derive_inputs_from_raw_rows,
    load_input_provenance,
    load_input_rows,
    validate_file_checksum,
    validate_fixed_scope,
    validate_input_checksum,
    validate_raw_rows_checksum,
    verify_against_reference,
    write_input_provenance,
    write_input_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "docs/context/evidence/issue_5034_control_action_latency_sweep"
DURABLE_INPUT = EVIDENCE_DIR / "snqi_latency_inputs.csv"
DURABLE_INPUT_PROVENANCE = EVIDENCE_DIR / "snqi_latency_inputs.csv.provenance.json"
REGISTERED_REFERENCE = EVIDENCE_DIR / "snqi_analysis.json"
WEIGHTS_PATH = REPO_ROOT / "configs/benchmarks/snqi_weights_camera_ready_v3.json"
BASELINE_PATH = REPO_ROOT / "configs/benchmarks/snqi_baseline_camera_ready_v3.json"

#: Minimal synthetic SNQI-v0 weights / baseline for fixture-driven unit tests.
WEIGHTS = {
    "w_success": 0.2,
    "w_time": 0.1,
    "w_collisions": 0.1,
    "w_near": 0.3,
    "w_comfort": 0.2,
    "w_force_exceed": 0.0,
    "w_jerk": 0.0,
}
BASELINE = {
    "collisions": {"med": 0.0, "p95": 1.0},
    "near_misses": {"med": 0.0, "p95": 10.0},
}

PLANNER_GROUPS = ("default_social_force", "hybrid_rule_v0_minimal", "orca")
PLANNER_BY_GROUP = {
    "default_social_force": "baseline_social_force",
    "hybrid_rule_v0_minimal": "hybrid_rule_v0_minimal",
    "orca": "orca",
}
MODE_BY_GROUP = {
    "default_social_force": "native",
    "hybrid_rule_v0_minimal": "adapter",
    "orca": "adapter",
}
SCENARIOS = list(EXPECTED_SCENARIO_IDS)
SEEDS = (111, 112, 113)
STEPS = (0, 1, 3)


def _input_row(  # noqa: PLR0913
    *,
    planner_group: str,
    step: int,
    seed: int,
    scenario_id: str,
    success: bool = True,
    collision: bool = False,
    time_to_goal_norm: float = 0.7,
    near_miss_rate: float = 0.0,
    steps: int = 200,
    comfort_exposure_mean: float = 0.01,
    execution_mode: str | None = None,
    availability_status: str = "available",
) -> dict[str, Any]:
    """Build one durable sufficient-input row (analyzer shape)."""
    return {
        "planner_group": planner_group,
        "planner": PLANNER_BY_GROUP[planner_group],
        "latency_step": step,
        "latency_ms": float(step * 100),
        "seed": seed,
        "scenario_id": scenario_id,
        "execution_mode": execution_mode
        if execution_mode is not None
        else MODE_BY_GROUP[planner_group],
        "availability_status": availability_status,
        "success": success,
        "collision": collision,
        "time_to_goal_norm": time_to_goal_norm,
        "near_miss_rate": near_miss_rate,
        "steps": steps,
        "comfort_exposure_mean": comfort_exposure_mean,
    }


def _full_input_set() -> list[dict[str, Any]]:
    """Return the complete 1,296-row fixed-scope input cross-product."""
    rows: list[dict[str, Any]] = []
    for planner_group in PLANNER_GROUPS:
        for step in STEPS:
            for seed in SEEDS:
                for scenario_id in SCENARIOS:
                    rows.append(
                        _input_row(
                            planner_group=planner_group,
                            step=step,
                            seed=seed,
                            scenario_id=scenario_id,
                        )
                    )
    return rows


# --- classify_input_row: execution-mode preservation + exclusions ----------


def test_classify_result_row_preserves_native_execution_mode() -> None:
    """A native row with all SNQI inputs classifies as a result and keeps its mode."""
    cell = classify_input_row(
        _input_row(
            planner_group="default_social_force", step=0, seed=111, scenario_id="scenario_00"
        )
    )
    assert cell.classification == "result"
    assert cell.exclusion_reason is None
    assert cell.execution_mode == "native"
    assert cell.success is True
    assert cell.collision is False


def test_classify_adapter_row_preserves_adapter_execution_mode() -> None:
    """An adapter row is a result but its execution_mode stays 'adapter'."""
    cell = classify_input_row(
        _input_row(planner_group="orca", step=1, seed=112, scenario_id="scenario_01")
    )
    assert cell.classification == "result"
    assert cell.execution_mode == "adapter"


def test_classify_fallback_execution_mode_is_exclusion() -> None:
    """A fallback execution-mode row is an exclusion, never a result (#691 policy)."""
    row = _input_row(
        planner_group="default_social_force", step=0, seed=111, scenario_id="scenario_00"
    )
    row["execution_mode"] = "fallback"
    cell = classify_input_row(row)
    assert cell.classification == "exclusion"
    assert "non_native_execution_mode:fallback" in (cell.exclusion_reason or "")


def test_classify_degraded_unavailable_row_is_exclusion() -> None:
    """A degraded/unavailable row is excluded and never contributes."""
    row = _input_row(planner_group="orca", step=0, seed=111, scenario_id="scenario_00")
    row["availability_status"] = "degraded"
    cell = classify_input_row(row)
    assert cell.classification == "exclusion"
    assert "unavailable:degraded" in (cell.exclusion_reason or "")


def test_classify_missing_snqi_metric_is_exclusion() -> None:
    """A row missing a required SNQI input metric is excluded."""
    row = _input_row(planner_group="orca", step=0, seed=111, scenario_id="scenario_00")
    row["time_to_goal_norm"] = None
    cell = classify_input_row(row)
    assert cell.classification == "exclusion"
    assert "missing_or_invalid_metric:time_to_goal_norm" in (cell.exclusion_reason or "")


def test_classify_inconsistent_latency_milliseconds_is_exclusion() -> None:
    """A latency step and millisecond marker must describe the same delay."""
    row = _input_row(planner_group="orca", step=1, seed=111, scenario_id=SCENARIOS[0])
    row["latency_ms"] = 300.0
    cell = classify_input_row(row)
    assert cell.classification == "exclusion"
    assert "latency_ms_does_not_match_latency_step" in (cell.exclusion_reason or "")


def test_classify_non_boolean_success_fails_closed() -> None:
    """A non-boolean success value is rejected rather than silently coerced."""
    row = _input_row(planner_group="orca", step=0, seed=111, scenario_id="scenario_00")
    row["success"] = "yes"
    with pytest.raises(SnqiLatencyAnalysisError, match="non-boolean"):
        classify_input_row(row)


# --- validate_fixed_scope: missing rows fail closed ------------------------


def test_validate_fixed_scope_accepts_complete_cross_product() -> None:
    """The full 1,296-row cross-product verifies cleanly."""
    inputs = [classify_input_row(row) for row in _full_input_set()]
    coverage = validate_fixed_scope(inputs)
    assert coverage["status"] == "verified"
    assert coverage["latency_row_count"] == 1296
    assert coverage["expected_row_count"] == 1296
    assert coverage["missing_latency_cells"] == 0


def test_validate_fixed_scope_rejects_missing_rows() -> None:
    """Dropping one cell fails closed with a cross-product error."""
    rows = _full_input_set()
    del rows[0]  # remove one (planner_group, step, seed, scenario) cell
    inputs = [classify_input_row(row) for row in rows]
    with pytest.raises(SnqiLatencyAnalysisError, match="cross-product"):
        validate_fixed_scope(inputs)


def test_validate_fixed_scope_rejects_duplicate_rows() -> None:
    """A duplicate cell fails closed."""
    rows = _full_input_set()
    rows.append(rows[0])
    inputs = [classify_input_row(row) for row in rows]
    with pytest.raises(SnqiLatencyAnalysisError, match="duplicate_cells"):
        validate_fixed_scope(inputs)


def test_validate_fixed_scope_rejects_fallback_row_in_set() -> None:
    """A fallback row surviving into the input set fails closed."""
    rows = _full_input_set()
    rows[0] = {**rows[0], "execution_mode": "fallback"}
    inputs = [classify_input_row(row) for row in rows]
    with pytest.raises(SnqiLatencyAnalysisError, match="non_native_or_unavailable"):
        validate_fixed_scope(inputs)


def test_validate_fixed_scope_rejects_wrong_seed_roster() -> None:
    """An unexpected seed value fails closed."""
    rows = _full_input_set()
    rows[0] = {**rows[0], "seed": 999}
    inputs = [classify_input_row(row) for row in rows]
    with pytest.raises(SnqiLatencyAnalysisError, match="seeds"):
        validate_fixed_scope(inputs)


def test_validate_fixed_scope_rejects_wrong_scenario_roster() -> None:
    """A same-sized replacement scenario set must not pass fixed-scope validation."""
    rows = [
        {**row, "scenario_id": "unregistered_scenario"}
        if row["scenario_id"] == SCENARIOS[0]
        else row
        for row in _full_input_set()
    ]
    inputs = [classify_input_row(row) for row in rows]
    with pytest.raises(SnqiLatencyAnalysisError, match="scenario roster"):
        validate_fixed_scope(inputs)


# --- validate_input_checksum: malformed provenance -------------------------


def test_checksum_validation_rejects_mismatch(tmp_path: Path) -> None:
    """A durable input whose SHA-256 disagrees with its provenance fails closed."""
    inputs = [classify_input_row(row) for row in _full_input_set()]
    input_path = tmp_path / "inputs.csv"
    write_input_rows(inputs, input_path)
    provenance = {"input_sha256": "0" * 64, "source": {"raw_rows_sha256": "6b34e690" + "0" * 56}}
    with pytest.raises(SnqiLatencyAnalysisError, match="checksum mismatch"):
        validate_input_checksum(input_path, provenance)


def test_checksum_validation_rejects_missing_sha(tmp_path: Path) -> None:
    """A provenance sidecar without a valid input_sha256 fails closed."""
    inputs = [classify_input_row(row) for row in _full_input_set()]
    input_path = tmp_path / "inputs.csv"
    write_input_rows(inputs, input_path)
    with pytest.raises(SnqiLatencyAnalysisError, match="no valid input_sha256"):
        validate_input_checksum(input_path, {"input_sha256": "short"})


def test_checksum_validation_rejects_wrong_raw_anchor(tmp_path: Path) -> None:
    """A provenance not anchored to the registered raw-row SHA-256 fails closed."""
    inputs = [classify_input_row(row) for row in _full_input_set()]
    input_path = tmp_path / "inputs.csv"
    write_input_rows(inputs, input_path)
    from robot_sf.benchmark.identity.hash_utils import sha256_file

    provenance = {
        "input_sha256": sha256_file(input_path),
        "source": {"raw_rows_sha256": "deadbeef" + "0" * 56},
    }
    with pytest.raises(SnqiLatencyAnalysisError, match="raw-row SHA-256"):
        validate_input_checksum(input_path, provenance)


def test_raw_row_checksum_rejects_unregistered_file(tmp_path: Path) -> None:
    """Raw-row mode must not accept a file with an unregistered digest."""
    raw_path = tmp_path / "episode_rows.jsonl"
    raw_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SnqiLatencyAnalysisError, match="raw campaign rows checksum mismatch"):
        validate_raw_rows_checksum(raw_path)


def test_registered_config_checksum_rejects_override(tmp_path: Path) -> None:
    """Canonical config provenance must reject an arbitrary override file."""
    override = tmp_path / "weights.json"
    override.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SnqiLatencyAnalysisError, match="SNQI weights checksum mismatch"):
        validate_file_checksum(override, WEIGHTS_SHA256, label="SNQI weights")
    with pytest.raises(SnqiLatencyAnalysisError, match="SNQI baseline checksum mismatch"):
        validate_file_checksum(override, BASELINE_SHA256, label="SNQI baseline")


def test_provenance_sidecar_round_trips(tmp_path: Path) -> None:
    """write_input_provenance + load_input_provenance round-trips the anchor."""
    inputs = [classify_input_row(row) for row in _full_input_set()]
    input_path = tmp_path / "inputs.csv"
    provenance_path = tmp_path / "inputs.provenance.json"
    write_input_rows(inputs, input_path)
    write_input_provenance(
        input_path,
        provenance_path,
        raw_rows_path="private/raw.jsonl",
        promoter_git_head="abc12345",
        date="2026-07-17",
    )
    payload = load_input_provenance(provenance_path)
    assert payload["input_sha256"] != "0" * 64
    assert payload["source"]["raw_rows_sha256"].startswith("6b34e690")
    # The written input must validate against its own provenance.
    validate_input_checksum(input_path, payload)


# --- CSV round-trip --------------------------------------------------------


def test_csv_round_trip_preserves_inputs(tmp_path: Path) -> None:
    """write_input_rows then load_input_rows preserves every classified field."""
    original = [classify_input_row(row) for row in _full_input_set()]
    input_path = tmp_path / "inputs.csv"
    write_input_rows(original, input_path)
    reloaded = load_input_rows(input_path)
    assert len(reloaded) == len(original)
    first = next(entry for entry in reloaded if entry.planner_group == "default_social_force")
    assert first.execution_mode == "native"
    assert first.latency_step == 0 or first.latency_step in STEPS
    # Every reloaded row is a valid result.
    assert all(entry.classification == "result" for entry in reloaded)


def test_load_input_rows_rejects_missing_columns(tmp_path: Path) -> None:
    """A durable input missing a required column fails closed."""
    input_path = tmp_path / "inputs.csv"
    input_path.write_text(
        "# AI-GENERATED NEEDS-REVIEW\nplanner_group,planner,latency_step\nx,y,0\n", encoding="utf-8"
    )
    with pytest.raises(SnqiLatencyAnalysisError, match="missing required columns"):
        load_input_rows(input_path)


# --- raw-row derivation ----------------------------------------------------


def _raw_row(*, planner_group: str, step: int, seed: int, scenario_id: str) -> dict[str, Any]:
    """Build one raw campaign episode row (runner shape) on the latency axis."""
    return {
        "axis": AXIS_KEY,
        "variant": f"{AXIS_KEY}__step_{step}",
        "variant_source_key": f"step_{step}",
        "baseline_variant": step == 0,
        "action_latency": {
            "configured_steps": step,
            "configured_ms": None,
            "effective_steps": step,
            "effective_ms": float(step * 100),
        },
        "planner": PLANNER_BY_GROUP[planner_group],
        "planner_group": planner_group,
        "scenario_id": scenario_id,
        "seed": seed,
        "success": True,
        "collision": False,
        "execution_mode": MODE_BY_GROUP[planner_group],
        "availability_status": "available",
        "steps": 200,
        "metrics": {
            "success_rate": 1.0,
            "collision_rate": 0.0,
            "time_to_goal_norm": 0.7,
            "near_miss_rate": 0.0,
            "comfort_exposure_mean": 0.01,
        },
    }


def test_derive_inputs_from_raw_rows_isolates_latency_axis() -> None:
    """Only control_action_latency rows are promoted; other axes are ignored."""
    raw = [_raw_row(planner_group="orca", step=0, seed=111, scenario_id="scenario_00")]
    # Add a non-latency axis row that must be ignored.
    noise = dict(raw[0])
    noise["axis"] = "clearance_radius"
    raw.append(noise)
    derived = derive_inputs_from_raw_rows(raw)
    assert len(derived) == 1
    assert derived[0].planner_group == "orca"
    assert derived[0].execution_mode == "adapter"


def test_derive_inputs_preserves_execution_modes() -> None:
    """Native and adapter labels survive the raw-to-input derivation."""
    raw = [
        _raw_row(planner_group="default_social_force", step=0, seed=111, scenario_id="s0"),
        _raw_row(planner_group="orca", step=0, seed=111, scenario_id="s0"),
        _raw_row(planner_group="hybrid_rule_v0_minimal", step=0, seed=111, scenario_id="s0"),
    ]
    derived = {
        entry.planner_group: entry.execution_mode for entry in derive_inputs_from_raw_rows(raw)
    }
    assert derived == {
        "default_social_force": "native",
        "orca": "adapter",
        "hybrid_rule_v0_minimal": "adapter",
    }


# --- end-to-end build_snqi_analysis ---------------------------------------


def test_build_snqi_analysis_has_registered_schema_and_structure() -> None:
    """The generated packet carries the registered schema and required blocks."""
    inputs = [classify_input_row(row) for row in _full_input_set()]
    packet = build_snqi_analysis(
        inputs, weights=WEIGHTS, baseline_stats=BASELINE, date="2026-07-17"
    )
    assert packet["schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert packet["evidence_status"] == "diagnostic-only"
    assert packet["claim_boundary"]  # non-empty
    for block in (
        "provenance",
        "scope_verification",
        "snqi_method",
        "point_estimate_robustness_ranking",
        "pairwise_slope_uncertainty",
        "verdict",
        "caveats",
        "reproducibility_contract",
    ):
        assert block in packet, f"missing block {block!r}"
    assert packet["scope_verification"]["latency_episode_row_count"] == 1296
    assert packet["scope_verification"]["missing_latency_cells"] == 0


def test_build_snqi_analysis_preserves_execution_modes_in_ranking() -> None:
    """Native / adapter labels are retained in the ranking and execution_modes."""
    inputs = [classify_input_row(row) for row in _full_input_set()]
    packet = build_snqi_analysis(
        inputs, weights=WEIGHTS, baseline_stats=BASELINE, date="2026-07-17"
    )
    modes = {
        row["planner_group"]: row["execution_mode"]
        for row in packet["point_estimate_robustness_ranking"]
    }
    assert modes == {
        "default_social_force": "native",
        "orca": "adapter",
        "hybrid_rule_v0_minimal": "adapter",
    }
    summary_modes = {
        entry["planner_group"]: entry["execution_mode"]
        for entry in packet["scope_verification"]["execution_modes"]
    }
    assert summary_modes == modes


def test_build_snqi_analysis_deterministic() -> None:
    """Two builds from the same input produce byte-identical packets."""
    inputs = [classify_input_row(row) for row in _full_input_set()]
    first = build_snqi_analysis(inputs, weights=WEIGHTS, baseline_stats=BASELINE, date="2026-07-17")
    second = build_snqi_analysis(
        inputs, weights=WEIGHTS, baseline_stats=BASELINE, date="2026-07-17"
    )
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# --- verify_against_reference ---------------------------------------------


def test_verify_against_reference_rejects_schema_mismatch() -> None:
    """A reference with the wrong schema_version fails closed."""
    inputs = [classify_input_row(row) for row in _full_input_set()]
    packet = build_snqi_analysis(
        inputs, weights=WEIGHTS, baseline_stats=BASELINE, date="2026-07-17"
    )
    with pytest.raises(SnqiLatencyAnalysisError, match="schema_version mismatch"):
        verify_against_reference(packet, {"schema_version": "something.else.v9"})


def test_verify_against_self_passes() -> None:
    """A packet verifies against itself with zero deviation."""
    inputs = [classify_input_row(row) for row in _full_input_set()]
    packet = build_snqi_analysis(
        inputs, weights=WEIGHTS, baseline_stats=BASELINE, date="2026-07-17"
    )
    report = verify_against_reference(packet, packet)
    assert report["status"] == "verified"
    assert report["max_point_estimate_deviation"] == 0.0


def test_verify_against_reference_fails_on_tampered_slope() -> None:
    """A reference whose slope exceeds the point tolerance fails closed."""
    inputs = [classify_input_row(row) for row in _full_input_set()]
    packet = build_snqi_analysis(
        inputs, weights=WEIGHTS, baseline_stats=BASELINE, date="2026-07-17"
    )
    tampered = json.loads(json.dumps(packet))
    tampered["point_estimate_robustness_ranking"][0]["snqi_slope_per_100ms"] += 1.0
    with pytest.raises(SnqiLatencyAnalysisError, match="deviates"):
        verify_against_reference(packet, tampered)


# --- registered-artifact integration (skipped when inputs absent) ----------


@pytest.mark.skipif(
    not DURABLE_INPUT.exists() or not REGISTERED_REFERENCE.exists(),
    reason="committed durable input / registered reference not present",
)
def test_committed_durable_input_reproduces_registered_packet() -> None:
    """The committed durable input reproduces the registered packet under the contract."""
    inputs = load_input_rows(DURABLE_INPUT)
    weights = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    packet = build_snqi_analysis(
        inputs, weights=weights, baseline_stats=baseline, date="2026-07-16"
    )
    reference = json.loads(REGISTERED_REFERENCE.read_text(encoding="utf-8"))
    report = verify_against_reference(packet, reference)
    assert report["status"] == "verified"
    # Point estimates reproduce essentially exactly.
    assert report["max_point_estimate_deviation"] <= POINT_TOL
    assert report["max_pairwise_slope_difference_deviation"] <= PAIRWISE_DIFF_TOL
    assert report["max_bootstrap_interval_deviation"] <= INTERVAL_TOL
    assert report["max_probability_deviation"] <= PROBABILITY_TOL


@pytest.mark.skipif(
    not DURABLE_INPUT.exists() or not DURABLE_INPUT_PROVENANCE.exists(),
    reason="committed durable input / provenance not present",
)
def test_committed_durable_input_checksum_matches_provenance() -> None:
    """The committed durable input validates against its provenance sidecar."""
    provenance = load_input_provenance(DURABLE_INPUT_PROVENANCE)
    validate_input_checksum(DURABLE_INPUT, provenance)


@pytest.mark.skipif(not DURABLE_INPUT.exists(), reason="committed durable input not present")
def test_committed_durable_input_has_full_fixed_scope() -> None:
    """The committed durable input covers the complete 1,296-row fixed scope."""
    inputs = load_input_rows(DURABLE_INPUT)
    coverage = validate_fixed_scope(inputs)
    assert coverage["latency_row_count"] == 1296
    assert coverage["fallback_row_count"] == 0
    assert coverage["degraded_row_count"] == 0
    assert coverage["unavailable_row_count"] == 0
