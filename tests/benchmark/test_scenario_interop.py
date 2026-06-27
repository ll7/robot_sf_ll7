"""Tests for the dry-run Robot SF -> external-benchmark scenario converter (issue #3285).

Covers the deterministic conversion contract, schema validation of the emitted
intermediate representation (IR), and explicit unsupported-field diagnostics.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.scenario_interop import (
    CONVERTER_NAME,
    IR_SCHEMA_VERSION,
    convert_scenario_to_ir,
    dump_ir,
    load_interop_ir_schema,
    validate_interop_ir,
)


def _axis_scenario() -> dict:
    """Return an axis-style fixture scenario (matches scenarios.schema.json)."""

    return {
        "id": "demo-uni-low-open",
        "density": "low",
        "flow": "uni",
        "obstacle": "open",
        "groups": 0.0,
        "speed_var": "low",
        "goal_topology": "point",
        "robot_context": "embedded",
        "repeats": 2,
        "seeds": [1, 2, 3],
    }


def _explicit_map_scenario() -> dict:
    """Return an explicit-map fixture scenario with agents and unsupported fields."""

    return {
        "name": "corridor-handoff",
        "map_file": "maps/svg_maps/corridor.svg",
        "simulation_config": {"step_time_s": 0.1},
        "robot_config": {"radius": 0.3},
        "route_overrides_file": "configs/routes/corridor.yaml",
        "supported": True,
        "amv": {"use_case": "delivery"},
        "single_pedestrians": [
            {
                "id": "ped-1",
                "start_poi": "north",
                "goal_poi": "south",
                "speed_m_s": 1.2,
                "role": "leader",
                "wait_at": [[1.0, 2.0]],
            },
            {
                "start_poi": "east",
                "goal_poi": "west",
            },
        ],
        "metadata": {"author": "fixture"},
    }


def test_axis_scenario_ir_is_schema_valid_and_complete() -> None:
    """An axis scenario converts to a schema-valid IR with mapped semantics."""

    result = convert_scenario_to_ir(_axis_scenario(), source_file="example.yaml")

    assert result.is_valid, result.schema_errors
    assert result.ir["schema_version"] == IR_SCHEMA_VERSION
    assert result.ir["provenance"]["source_scenario_id"] == "demo-uni-low-open"
    assert result.ir["provenance"]["source_kind"] == "axis"
    assert result.ir["provenance"]["source_file"] == "example.yaml"
    assert result.ir["provenance"]["converter"] == CONVERTER_NAME
    assert result.ir["geometry"]["obstacle_topology"] == "open"
    assert result.ir["geometry"]["environment_type"] == "open_space"
    assert result.ir["environment"]["density"] == "low"
    assert result.ir["environment"]["groups_fraction"] == 0.0
    assert result.ir["timing"]["repeats"] == 2
    assert result.ir["timing"]["seeds"] == [1, 2, 3]
    assert result.unsupported_fields == []


def test_conversion_is_deterministic() -> None:
    """Identical input yields byte-identical serialized IR across runs."""

    scenario = _explicit_map_scenario()
    first = dump_ir(convert_scenario_to_ir(scenario).ir)
    second = dump_ir(convert_scenario_to_ir(dict(scenario)).ir)
    assert first == second


def test_unsupported_fields_reported_not_dropped() -> None:
    """Recognized simulator-specific and unknown fields are reported explicitly."""

    scenario = _explicit_map_scenario()
    scenario["totally_unknown_field"] = 7
    result = convert_scenario_to_ir(scenario)

    assert result.is_valid, result.schema_errors
    reported = {item["field"]: item["reason"] for item in result.unsupported_fields}

    # Known simulator-specific fields are flagged, not silently dropped.
    for known in ("simulation_config", "robot_config", "route_overrides_file", "amv", "supported"):
        assert known in reported, f"expected {known} to be reported as unsupported"

    # Unknown fields get the generic no-mapping reason.
    assert reported["totally_unknown_field"] == (
        "no intermediate-representation mapping is defined for this field"
    )

    # Reports are sorted by field name for determinism.
    fields = [item["field"] for item in result.unsupported_fields]
    assert fields == sorted(fields)

    # Mapped fields never appear in the unsupported report.
    assert "single_pedestrians" not in reported
    assert "metadata" not in reported
    assert "map_file" not in reported


def test_agents_mapped_with_positional_fallback_id() -> None:
    """Agents preserve order; specs without an id get a deterministic positional id."""

    result = convert_scenario_to_ir(_explicit_map_scenario())
    agents = result.ir["agents"]

    assert [a["id"] for a in agents] == ["ped-1", "agent_1"]
    assert agents[0]["start"] == "north"
    assert agents[0]["goal"] == "south"
    assert agents[0]["preferred_speed_mps"] == 1.2
    assert agents[0]["role"] == "leader"
    assert agents[0]["wait_points"] == [[1.0, 2.0]]
    # Absent optional fields normalize to None.
    assert agents[1]["preferred_speed_mps"] is None
    assert agents[1]["role"] is None


def test_metadata_preserved_in_provenance() -> None:
    """Source metadata is preserved through provenance, not dropped."""

    result = convert_scenario_to_ir(_explicit_map_scenario())
    assert result.ir["provenance"]["source_metadata"] == {"author": "fixture"}
    assert result.ir["provenance"]["source_kind"] == "explicit_map"


def test_source_fields_recorded_sorted() -> None:
    """Provenance records every top-level source key, sorted for determinism."""

    scenario = _axis_scenario()
    result = convert_scenario_to_ir(scenario)
    recorded = result.ir["provenance"]["source_fields"]
    assert recorded == sorted(str(k) for k in scenario)


def test_missing_identifier_raises() -> None:
    """A scenario without any usable identifier is a hard error."""

    with pytest.raises(ValueError, match="usable identifier"):
        convert_scenario_to_ir({"density": "low"})


def test_non_mapping_input_raises() -> None:
    """Non-mapping input is rejected with a clear error."""

    with pytest.raises(TypeError, match="must be a mapping"):
        convert_scenario_to_ir(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_emitted_ir_validates_against_loaded_schema() -> None:
    """The emitted IR validates against the on-disk schema directly."""

    load_interop_ir_schema()  # exercises the cached loader
    result = convert_scenario_to_ir(_axis_scenario())
    assert validate_interop_ir(result.ir) == []


def test_tampered_ir_fails_validation() -> None:
    """Removing a required section makes the IR fail schema validation."""

    result = convert_scenario_to_ir(_axis_scenario())
    broken = dict(result.ir)
    del broken["geometry"]
    assert validate_interop_ir(broken) != []
