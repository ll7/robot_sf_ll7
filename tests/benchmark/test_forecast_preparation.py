"""Regression tests for the issue #7602 forecast-preparation contract."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from robot_sf.benchmark.forecast.forecast_preparation import (
    ForecastPreparationSourceSpec,
    _actor_for_frame,
    _load_json_object,
    _validate_split_policy,
    build_forecast_preparation_packet,
    validate_forecast_preparation_packet,
)
from robot_sf.benchmark.identity.hash_utils import stable_hash

REPO_ROOT = Path(__file__).resolve().parents[2]


def _source_specs() -> tuple[ForecastPreparationSourceSpec, ...]:
    return (
        ForecastPreparationSourceSpec(
            path="tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/"
            "bottleneck_motion_rich_fixture.json",
            scenario_family="bottleneck",
            cutoff_frame_step=5,
        ),
        ForecastPreparationSourceSpec(
            path="docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/"
            "inputs/synthetic_crossing_proxy_orca_111_trace_export.json",
            scenario_family="crossing_proxy",
            cutoff_frame_step=2,
        ),
        ForecastPreparationSourceSpec(
            path="docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/"
            "ammv_social_force_trace_export.json",
            scenario_family="head_on_corridor",
            cutoff_frame_step=5,
        ),
    )


def _packet() -> dict:
    return build_forecast_preparation_packet(
        _source_specs(),
        repo_root=REPO_ROOT,
        horizons_s=(1.0,),
    )


def _validate(payload: dict) -> None:
    validate_forecast_preparation_packet(
        payload,
        repo_root=REPO_ROOT,
        verify_checksums=False,
    )


def test_packet_emits_matched_rows_and_explicit_ego_unavailability() -> None:
    """The packet has one identity-matched oracle/ego pair per selected source."""
    payload = _packet()

    assert payload["pair_count"] == 3
    assert payload["row_count"] == 6
    assert set(payload["coverage"]["scenario_families"]) == {
        "bottleneck",
        "crossing_proxy",
        "head_on_corridor",
    }
    assert set(payload["coverage"]["planners"]) == {
        "ammv_social_force",
        "hybrid_rule_v0_minimal",
        "orca",
    }
    assert payload["coverage"]["ego_observation_status"] == "not_available"

    rows_by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in payload["rows"]:
        rows_by_pair[row["pair_id"]].append(row)
    assert len(rows_by_pair) == 3
    for pair_rows in rows_by_pair.values():
        assert {row["observation_tier"] for row in pair_rows} == {
            "oracle_full_state",
            "ego_observation",
        }
        assert pair_rows[0]["identity"] == pair_rows[1]["identity"]
        assert pair_rows[0]["lineage"] == pair_rows[1]["lineage"]
        assert pair_rows[0]["target"] == pair_rows[1]["target"]
        ego_row = next(row for row in pair_rows if row["observation_tier"] == "ego_observation")
        assert ego_row["availability_status"] == "not_available"
        assert "pedestrian_position_m" not in ego_row["input"]
        assert "pedestrian_velocity_mps" not in ego_row["input"]
        assert all(
            not entry["future_target"]
            for entry in ego_row["field_leakage_ledger"]
            if entry["field"].startswith("input.")
        )
        assert any(
            entry["field"] == "target.future_position_m" and entry["future_target"]
            for entry in ego_row["field_leakage_ledger"]
        )

    _validate(payload)


def test_tracked_packet_checksum_manifest_is_valid() -> None:
    """The committed packet validates against its complete SHA-256 manifest."""
    packet_path = (
        REPO_ROOT
        / "docs/context/evidence/issue_7399_forecast_preparation/forecast_preparation_packet.json"
    )
    payload = json.loads(packet_path.read_text(encoding="utf-8"))

    summary = validate_forecast_preparation_packet(payload, repo_root=REPO_ROOT)

    assert summary["status"] == "passed"


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("not_mapping", "packet must be a mapping"),
        ("schema_version", "schema_version must be"),
        ("issue", "issue must be 7602"),
        ("evidence_status", "evidence_status must be diagnostic-only"),
        ("observation_contract", "observation_contract_changed must be false"),
        ("empty_rows", "rows must be a non-empty list"),
        ("empty_sources", "source_artifacts must be a non-empty list"),
        ("row_count", "row_count does not match rows"),
        ("pair_count", "pair_count does not match rows"),
        ("odd_rows", "rows must contain two rows per pair"),
    ),
)
def test_packet_shape_contract_fails_closed(case: str, match: str) -> None:
    """Top-level packet shape and identity fields cannot be bypassed."""
    payload = _packet()
    invalid_payload = {
        "not_mapping": lambda packet: [],
        "schema_version": lambda packet: packet | {"schema_version": "unknown"},
        "issue": lambda packet: packet | {"issue": 7603},
        "evidence_status": lambda packet: packet | {"evidence_status": "success"},
        "observation_contract": lambda packet: packet | {"observation_contract_changed": True},
        "empty_rows": lambda packet: packet | {"rows": []},
        "empty_sources": lambda packet: packet | {"source_artifacts": []},
        "row_count": lambda packet: packet | {"row_count": packet["row_count"] + 1},
        "pair_count": lambda packet: packet | {"pair_count": packet["pair_count"] + 1},
        "odd_rows": lambda packet: (
            packet | {"rows": packet["rows"][:-1], "row_count": 5, "pair_count": 2}
        ),
    }[case](payload)

    with pytest.raises(ValueError, match=match):
        validate_forecast_preparation_packet(
            invalid_payload,  # type: ignore[arg-type]
            repo_root=REPO_ROOT,
            verify_checksums=False,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("claim_boundary", "benchmark success", "claim_boundary"),
        ("source_owner", "other.py", "source_owner"),
        ("source_schema", "other.v1", "source_schema"),
        ("row_schema_version", "other.v1", "row_schema_version"),
        ("horizons_s", [2.0], "declared horizons"),
        ("evidence_references", [], "evidence_references"),
    ),
)
def test_packet_top_level_contract_is_bound(field: str, value: object, match: str) -> None:
    """Top-level declarations cannot drift from the packet owner contract."""
    payload = _packet()
    payload[field] = value

    with pytest.raises(ValueError, match=match):
        _validate(payload)


def test_packet_rejects_unknown_top_level_claim_fields() -> None:
    """Unvalidated result or claim fields cannot travel in a preparation packet."""
    payload = _packet()
    payload["unsupported_claim"] = {"status": "benchmark_success", "score": 1.0}

    with pytest.raises(ValueError, match="packet top-level fields are not canonical"):
        _validate(payload)


@pytest.mark.parametrize("container", ("coverage", "field_leakage_ledger", "split_policy"))
def test_packet_rejects_unknown_declaration_fields(container: str) -> None:
    """Declaration maps reject fields that the validator does not interpret."""
    payload = _packet()
    payload[container]["unsupported_claim"] = "not validated"

    with pytest.raises(ValueError, match="not canonical"):
        _validate(payload)


@pytest.mark.parametrize(
    ("contents", "match"),
    (
        ('{"key": 1, "key": 2}', "duplicate JSON object key"),
        ('{"key": NaN}', "non-standard JSON constant"),
    ),
)
def test_source_json_parser_rejects_noncanonical_values(
    tmp_path: Path, contents: str, match: str
) -> None:
    """Source input rejects parser recovery that can hide ambiguous values."""
    path = tmp_path / "source.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _load_json_object(path)


def test_duplicate_actor_ids_are_not_selected_implicitly() -> None:
    """An ambiguous actor identity cannot determine a forecast target."""
    actors = [
        {"id": "pedestrian-1", "position": [0.0, 0.0], "velocity": [0.0, 0.0]},
        {"id": "pedestrian-1", "position": [1.0, 1.0], "velocity": [0.0, 0.0]},
    ]

    with pytest.raises(ValueError, match="actor ids must be unique"):
        _actor_for_frame(actors, None)


@pytest.mark.parametrize("field", ("trace_id", "episode_id", "scenario_id", "seed", "planner_id"))
def test_trace_metadata_is_bound_to_source_export(field: str) -> None:
    """Packet identity fields cannot be rewritten independently of the trace export."""
    payload = _packet()
    payload["source_artifacts"][0][field] = "fabricated"

    with pytest.raises(ValueError, match="source metadata drift"):
        _validate(payload)


@pytest.mark.parametrize("field", ("lineage_group_id", "frame_count", "size_bytes"))
def test_derived_source_metadata_is_bound_to_source_export(field: str) -> None:
    """Derived source metadata cannot be replaced with packet-authored values."""
    payload = _packet()
    artifact = payload["source_artifacts"][0]
    artifact[field] = "fabricated" if field == "lineage_group_id" else artifact[field] + 1

    with pytest.raises(ValueError, match="source metadata drift"):
        _validate(payload)


def test_ego_source_key_is_bound_to_source_artifacts() -> None:
    """A packet cannot relabel the source key after source metadata is emitted."""
    payload = _packet()
    payload["source_artifacts"][0]["ego_observation_source_key"] = "other_agents"

    with pytest.raises(ValueError, match="ego_observation_source_key"):
        _validate(payload)


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("coverage_mapping", "coverage must be a mapping"),
        ("families", "scenario family coverage drift"),
        ("planners", "planner coverage drift"),
        ("tiers", "observation tier coverage drift"),
        ("ego_status", "ego observation availability must remain explicit"),
        ("unavailable", "coverage.unavailable_strata must be a list"),
    ),
)
def test_coverage_contract_fails_closed(case: str, match: str) -> None:
    """Coverage declarations must describe the emitted source and observation strata."""
    payload = _packet()
    if case == "coverage_mapping":
        payload["coverage"] = []
    elif case == "families":
        payload["coverage"]["scenario_families"] = ["fabricated"]
    elif case == "planners":
        payload["coverage"]["planners"] = ["fabricated"]
    elif case == "tiers":
        payload["coverage"]["observation_tiers"] = ["fabricated"]
    elif case == "ego_status":
        payload["coverage"]["ego_observation_status"] = "available"
    elif case == "unavailable":
        payload["coverage"]["unavailable_strata"] = None

    with pytest.raises(ValueError, match=match):
        _validate(payload)


def test_cross_partition_group_leakage_fails_closed() -> None:
    """A lineage group assigned to two splits cannot pass validation."""
    payload = _packet()
    first, second = payload["source_artifacts"][:2]
    second["lineage_group_id"] = first["lineage_group_id"]
    second["split"] = "test" if first["split"] != "test" else "train"
    for row in payload["rows"]:
        if row["lineage"]["source_path"] == second["path"]:
            row["lineage"]["lineage_group_id"] = second["lineage_group_id"]
            row["lineage"]["split"] = second["split"]

    with pytest.raises(ValueError, match="group leakage across splits"):
        _validate_split_policy(payload, [first, second])


def test_cross_partition_near_duplicate_fails_closed() -> None:
    """An exact normalized trajectory fingerprint cannot cross split boundaries."""
    payload = _packet()
    first, second = payload["source_artifacts"][:2]
    second["near_duplicate_fingerprint"] = first["near_duplicate_fingerprint"]

    with pytest.raises(ValueError, match="near-duplicate trajectory leakage"):
        _validate_split_policy(payload, [first, second])


def test_split_assignments_are_recomputed_from_source_groups() -> None:
    """A packet cannot move a group while keeping source and assignment fields self-consistent."""
    payload = _packet()
    artifact = payload["source_artifacts"][0]
    group_id = artifact["lineage_group_id"]
    expected_split = artifact["split"]
    replacement_split = next(
        split for split in ("train", "validation", "test") if split != expected_split
    )
    artifact["split"] = replacement_split
    payload["split_policy"]["assignments"][group_id] = replacement_split
    for row in payload["rows"]:
        if row["lineage"]["lineage_group_id"] == group_id:
            row["lineage"]["split"] = replacement_split

    with pytest.raises(ValueError, match="split assignments drift"):
        _validate(payload)


def test_source_fingerprint_is_bound_to_trace_bytes() -> None:
    """A fabricated near-duplicate fingerprint cannot weaken split validation."""
    payload = _packet()
    payload["source_artifacts"][0]["near_duplicate_fingerprint"] = "fabricated"

    with pytest.raises(ValueError, match="near_duplicate_fingerprint"):
        _validate(payload)


def test_mismatched_pair_identity_fails_closed() -> None:
    """Changing one row's cutoff identity invalidates the pair contract."""
    payload = _packet()
    ego_row = next(row for row in payload["rows"] if row["observation_tier"] == "ego_observation")
    ego_row["identity"]["cutoff_time_s"] += 0.25

    with pytest.raises(ValueError, match="pair_id does not match identity"):
        _validate(payload)


def test_pair_identity_and_target_are_bound_to_source_trace() -> None:
    """A self-consistent but source-invented cutoff cannot pass validation."""
    payload = _packet()
    original_pair_id = payload["rows"][0]["pair_id"]
    pair_rows = [row for row in payload["rows"] if row["pair_id"] == original_pair_id]
    for row in pair_rows:
        row["identity"].update(
            {
                "frame_step": 0,
                "cutoff_time_s": 0.0,
                "target_frame_step": 1,
                "target_time_s": 1.0,
            }
        )
        row["pair_id"] = f"pair-{stable_hash(row['identity'])[:24]}"

    with pytest.raises(ValueError, match="does not match source trace"):
        _validate(payload)


def test_input_values_and_keys_are_bound_to_source_trace() -> None:
    """Unknown privileged ego fields and fabricated robot values are rejected."""
    payload = _packet()
    ego_row = next(row for row in payload["rows"] if row["observation_tier"] == "ego_observation")
    ego_row["input"]["pedestrian_heading_rad"] = 0.0

    with pytest.raises(ValueError, match="input fields do not match"):
        _validate(payload)


def test_field_leakage_ledger_semantics_are_bound_to_row_tier() -> None:
    """A packet cannot relabel privileged or future fields as robot-available."""
    payload = _packet()
    ego_row = next(row for row in payload["rows"] if row["observation_tier"] == "ego_observation")
    robot_entry = next(
        entry
        for entry in ego_row["field_leakage_ledger"]
        if entry["field"] == "input.robot_position_m"
    )
    robot_entry["robot_available"] = False

    with pytest.raises(ValueError, match="ledger semantics drift"):
        _validate(payload)


def test_source_declaring_ego_field_cannot_claim_unavailability() -> None:
    """The unavailable ego stratum is checked against the pinned source bytes."""
    payload = _packet()
    payload["ego_observation_source_key"] = "pedestrians"
    for artifact in payload["source_artifacts"]:
        artifact["ego_observation_source_key"] = "pedestrians"

    with pytest.raises(ValueError, match="source ego observation field is present"):
        _validate(payload)


def test_split_policy_declaration_is_bound_to_validator_contract() -> None:
    """A packet cannot rewrite split names while retaining matching assignments."""
    payload = _packet()
    payload["split_policy"]["split_names"] = ["train", "validation", "production"]

    with pytest.raises(ValueError, match="split_names are not canonical"):
        _validate(payload)


def test_future_field_in_ego_input_fails_closed() -> None:
    """Future or target-labelled fields are forbidden in ego inputs."""
    payload = _packet()
    ego_row = next(row for row in payload["rows"] if row["observation_tier"] == "ego_observation")
    ego_row["input"]["future_target_label"] = "forbidden"

    with pytest.raises(ValueError, match="future/target field leaked into ego input"):
        _validate(payload)


def test_unavailable_source_status_cannot_be_promoted() -> None:
    """The current trace sample cannot silently claim an ego source is available."""
    payload = _packet()
    payload["source_artifacts"][0]["ego_observation_status"] = "available"

    with pytest.raises(ValueError, match="source ego_observation_status must be not_available"):
        _validate(payload)


def test_row_lineage_metadata_must_match_source_artifact() -> None:
    """A valid source hash cannot excuse contradictory row provenance metadata."""
    payload = _packet()
    payload["rows"][0]["lineage"]["planner_id"] = "fabricated_planner"

    with pytest.raises(ValueError, match="lineage metadata does not match source: planner_id"):
        _validate(payload)


def test_row_lineage_rejects_unowned_fields() -> None:
    """Lineage rows cannot carry unvalidated future or privileged metadata."""
    payload = _packet()
    payload["rows"][0]["lineage"]["future_target"] = True

    with pytest.raises(ValueError, match="lineage fields are not canonical"):
        _validate(payload)


def test_source_artifact_rejects_unowned_fields() -> None:
    """Source artifacts cannot carry metadata the validator does not recompute."""
    payload = _packet()
    payload["source_artifacts"][0]["future_target"] = True

    with pytest.raises(ValueError, match="fields are not canonical"):
        _validate(payload)


def test_absolute_source_paths_are_rejected() -> None:
    """Preparation manifests must not retain machine-specific absolute source paths."""
    payload = _packet()
    relative_path = payload["source_artifacts"][0]["path"]
    payload["source_artifacts"][0]["path"] = str((REPO_ROOT / relative_path).resolve())

    with pytest.raises(
        ValueError, match=r"source_artifacts\[0\]\.path must be repository-relative"
    ):
        _validate(payload)


def test_false_reassurance_case_is_trace_backed_and_not_a_performance_claim() -> None:
    """The analytic counterexample records zero ADE/FDE with close robot clearance."""
    case = _packet()["ade_fde_false_reassurance_case"]

    assert case["status"] == "analytic_trace_backed_diagnostic_only"
    assert case["ade_m"] == 0.0
    assert case["fde_m"] == 0.0
    assert case["robot_pedestrian_clearance_m"] < case["risk_reference_m"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("cutoff_time_s", -1.0, "false case cutoff_time_s"),
        ("target_time_s", -1.0, "false case target_time_s"),
        ("horizon_s", 999.0, "false case horizon_s"),
        ("stationary_prediction_m", [999.0, 999.0], "false case stationary_prediction_m"),
        ("target_position_m", [999.0, 999.0], "false case target_position_m"),
        ("robot_position_m", [999.0, 999.0], "false case robot_position_m"),
        ("risk_reference_m", 999.0, "false case risk reference"),
    ),
)
def test_false_reassurance_metadata_is_trace_bound(field: str, value: object, match: str) -> None:
    """The analytic counterexample cannot carry fabricated coordinates or timing."""
    payload = _packet()
    payload["ade_fde_false_reassurance_case"][field] = value

    with pytest.raises(ValueError, match=match):
        _validate(payload)


def test_baseline_estimates_record_sample_and_hardware_assumptions() -> None:
    """Analytic baseline estimates must state their preparation assumptions."""
    payload = _packet()

    for estimate in payload["runtime_memory_estimates"]:
        assert estimate["sample_size_assumption"]
        assert estimate["hardware_assumption"]


def test_baseline_estimate_values_are_bound_to_contract() -> None:
    """Analytic estimate values cannot be changed without changing the owner contract."""
    payload = _packet()
    payload["runtime_memory_estimates"][0]["runtime_estimate"][
        "estimated_scalar_operations_per_actor_horizon"
    ] = 999

    with pytest.raises(ValueError, match="baseline estimate contract drift"):
        _validate(payload)


def test_dependency_license_metadata_is_bound_to_contract() -> None:
    """Dependency and rights dispositions cannot drift in the tracked packet."""
    payload = _packet()
    payload["dependency_license_comparison"][0]["decision"] = "adopt_external_dependency"

    with pytest.raises(ValueError, match="dependency/license comparison contract drift"):
        _validate(payload)
