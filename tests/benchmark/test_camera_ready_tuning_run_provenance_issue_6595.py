"""Prospective tuning-run provenance tests for issue #6595."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from robot_sf.benchmark.camera_ready._config import _validate_campaign_config
from robot_sf.benchmark.camera_ready._config_types import (
    CampaignConfig,
    PlannerSpec,
    ScenarioCandidateSelection,
    SeedPolicy,
    TuningSpec,
)
from robot_sf.benchmark.camera_ready._preflight import prepare_campaign_preflight
from robot_sf.benchmark.camera_ready._util import _repo_relative
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.tuning_run_provenance import (
    TUNING_LEDGER_SCHEMA,
    TUNING_RUN_CLASS_DEBUG,
    TUNING_RUN_CLASS_EVIDENCE,
    TUNING_RUN_CLASS_TUNING,
    TUNING_RUN_RECORD_SCHEMA,
    TuningRunRecord,
    TuningRunSpec,
    aggregate_tuning_records,
    build_launch_records,
    load_tuning_records,
    parse_tuning_run_spec,
    record_from_mapping,
    validate_tuning_run_spec,
)


def test_parse_tuning_run_spec_preserves_unknowns_and_run_class() -> None:
    """The parser keeps absent counters unknown instead of turning them into zero."""
    spec = parse_tuning_run_spec(
        {
            "run_class": "tuning",
            "run_id": "tune-001",
            "objective": "minimize collision rate",
            "development_split": "development-v1",
            "eval_set_disjoint": True,
            "stopping_rule": "stop after 20 trials",
        }
    )
    assert spec is not None
    assert spec.run_class == TUNING_RUN_CLASS_TUNING
    assert spec.attempted_configurations is None
    assert spec.person_hours is None


def test_strict_validation_is_narrow_and_fail_closed() -> None:
    """Strict validation rejects debug or incomplete publication provenance only."""
    validate_tuning_run_spec(None)
    with pytest.raises(ValueError, match="cannot use run_class='debug'"):
        validate_tuning_run_spec(TuningRunSpec(), strict=True)
    with pytest.raises(ValueError, match="missing:.*objective"):
        validate_tuning_run_spec(
            TuningRunSpec(run_class=TUNING_RUN_CLASS_TUNING, run_id="run-1"),
            strict=True,
        )


def test_run_class_aggregation_excludes_debug_and_evidence() -> None:
    """Only tuning records contribute counters; other classes remain visible but excluded."""
    records = (
        TuningRunRecord(
            run_id="tune-1",
            run_class=TUNING_RUN_CLASS_TUNING,
            planner_id="planner-a",
            attempted_configurations=4,
            simulator_episodes=20,
            simulator_calls=20,
            wall_clock_seconds=12.5,
        ),
        TuningRunRecord(
            run_id="debug-1",
            run_class=TUNING_RUN_CLASS_DEBUG,
            planner_id="planner-a",
            attempted_configurations=999,
            simulator_episodes=999,
        ),
        TuningRunRecord(
            run_id="evidence-1",
            run_class=TUNING_RUN_CLASS_EVIDENCE,
            planner_id="planner-a",
            attempted_configurations=888,
            simulator_episodes=888,
        ),
    )
    ledger = aggregate_tuning_records(records)
    planner = ledger["by_planner"]["planner-a"]
    assert ledger["schema_version"] == TUNING_LEDGER_SCHEMA
    assert planner["record_count"] == 3
    assert planner["tuning_record_count"] == 1
    assert planner["debug_record_count"] == 1
    assert planner["evidence_record_count"] == 1
    assert planner["attempted_configurations"] == 4
    assert planner["simulator_episodes"] == 20
    assert planner["counts_toward_tuning"] is True


def test_aggregation_is_deterministic_and_unknown_stays_null() -> None:
    """Record order does not affect the ledger digest and partial counters stay unknown."""
    records = (
        TuningRunRecord(
            run_id="tune-a",
            run_class=TUNING_RUN_CLASS_TUNING,
            planner_id="planner-a",
            attempted_configurations=2,
        ),
        TuningRunRecord(
            run_id="tune-b",
            run_class=TUNING_RUN_CLASS_TUNING,
            planner_id="planner-a",
            attempted_configurations=None,
        ),
    )
    first = aggregate_tuning_records(records)
    second = aggregate_tuning_records(reversed(records))
    assert first["ledger_sha256"] == second["ledger_sha256"]
    assert first["by_planner"]["planner-a"]["attempted_configurations"] is None
    assert first["by_planner"]["planner-a"]["unknown_attempted_configurations_count"] == 1
    assert first["records"][0]["person_hours"] is None


def test_build_launch_records_captures_per_arm_provenance() -> None:
    """Automatic launch capture emits stable, planner-scoped records."""
    spec = TuningRunSpec(
        run_class=TUNING_RUN_CLASS_TUNING,
        run_id="job-42",
        objective="maximize validation success",
        development_split="dev-v2",
        eval_set_disjoint=True,
        stopping_rule="best validation score after 10 trials",
        attempted_configurations=10,
    )
    records = build_launch_records(
        spec,
        campaign_id="campaign-42",
        source_commit="a" * 40,
        config_hash="b" * 64,
        planner_parameters={"planner-a": ("learning_rate",), "planner-b": None},
    )
    assert [record.run_id for record in records] == ["job-42:planner-a", "job-42:planner-b"]
    assert records[0].parameters_changed == ("learning_rate",)
    assert records[1].parameters_changed is None
    assert records[0].to_mapping()["counts_toward_tuning"] is True
    assert records[0].to_mapping()["person_hours"] is None


def test_record_ingestion_rejects_wrong_schema() -> None:
    """Ingestion fails closed when a record is not the versioned contract."""
    with pytest.raises(ValueError, match="unsupported tuning-run record schema"):
        record_from_mapping({"schema_version": "tuning-run-record.v0"})
    record = record_from_mapping(
        {
            "schema_version": TUNING_RUN_RECORD_SCHEMA,
            "run_id": "debug-1",
            "run_class": "debug",
            "planner_id": "planner-a",
            "counts_toward_tuning": False,
            "person_hours": None,
        }
    )
    assert record.person_hours is None


def test_generated_record_matches_versioned_json_schema() -> None:
    """The Python record emitter stays aligned with the checked-in schema."""
    schema_path = Path(__file__).parents[2] / "robot_sf/benchmark/schemas/tuning_run_record.v1.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    record = TuningRunRecord(
        run_id="debug-1",
        run_class=TUNING_RUN_CLASS_DEBUG,
        planner_id="planner-a",
    )
    jsonschema.validate(record.to_mapping(), schema)


def test_generated_ledger_matches_versioned_json_schema() -> None:
    """The deterministic ledger envelope stays aligned with its checked-in schema."""
    schema_path = Path(__file__).parents[2] / "robot_sf/benchmark/schemas/tuning_ledger.v1.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    ledger = aggregate_tuning_records(
        [
            TuningRunRecord(
                run_id="tuning-1",
                run_class=TUNING_RUN_CLASS_TUNING,
                planner_id="planner-a",
                attempted_configurations=2,
            )
        ]
    )
    jsonschema.validate(ledger, schema)


@pytest.mark.parametrize(
    ("run_class", "invalid_counts_toward_tuning"),
    [
        (TUNING_RUN_CLASS_DEBUG, True),
        (TUNING_RUN_CLASS_TUNING, False),
        (TUNING_RUN_CLASS_EVIDENCE, True),
    ],
)
def test_versioned_schemas_reject_run_class_counting_mismatches(
    run_class: str, invalid_counts_toward_tuning: bool
) -> None:
    """Both public schemas enforce the fixed run-class counting policy."""
    schema_root = Path(__file__).parents[2] / "robot_sf/benchmark/schemas"
    record = TuningRunRecord(
        run_id=f"{run_class}-1",
        run_class=run_class,
        planner_id="planner-a",
    )
    invalid_record = record.to_mapping()
    invalid_record["counts_toward_tuning"] = invalid_counts_toward_tuning
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            invalid_record,
            json.loads((schema_root / "tuning_run_record.v1.json").read_text(encoding="utf-8")),
        )

    invalid_ledger = aggregate_tuning_records([record])
    invalid_ledger["records"][0]["counts_toward_tuning"] = invalid_counts_toward_tuning
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            invalid_ledger,
            json.loads((schema_root / "tuning_ledger.v1.json").read_text(encoding="utf-8")),
        )


def _write_campaign(tmp_path: Path, *, strict: bool = True, with_provenance: bool = True) -> Path:
    """Write a compact camera-ready config exercising automatic ledger emission."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_path = tmp_path / scenario_rel
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    strict_block = "tuning_effort_enforcement: error\n" if strict else ""
    tuning_block = (
        """
tuning_run_provenance:
  run_class: tuning
  run_id: tuning-job-6595
  objective: minimize validation collision rate
  development_split: development-v1
  eval_set_disjoint: true
  attempted_configurations: 4
  simulator_episodes: 80
  simulator_calls: 80
  stopping_rule: stop after four valid configurations
  compute_resource: local-cpu
"""
        if with_provenance
        else ""
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        """name: issue_6595_smoke
seed_policy:
  mode: fixed-list
  seeds: [111]
scenario_matrix: configs/scenarios/single/francis2023_blind_corner.yaml
"""
        + strict_block
        + tuning_block
        + """planners:
  - key: planner_a
    algo: goal
    planner_group: core
    tuning:
      parameters_touched: [v_max]
      source: declared
""",
        encoding="utf-8",
    )
    return config_path


def test_campaign_preflight_emits_ledger_and_manifest_link(tmp_path: Path) -> None:
    """Strict camera-ready preflight writes a linked, populated tuning ledger."""
    cfg = load_campaign_config(_write_campaign(tmp_path))
    prepared = prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="issue6595")
    ledger_path = Path(prepared["tuning_ledger_path"])
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (Path(prepared["campaign_root"]) / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    assert ledger["records"][0]["objective"] == "minimize validation collision rate"
    assert ledger["records"][0]["development_split"] == "development-v1"
    assert ledger["records"][0]["source_commit"]
    assert ledger["records"][0]["config_hash"]
    assert ledger["records"][0]["stopping_rule"] == "stop after four valid configurations"
    assert ledger["by_planner"]["planner_a"]["attempted_configurations"] == 4
    assert ledger["records"][0]["person_hours"] is None
    assert load_tuning_records(ledger_path)[0].run_id == ledger["records"][0]["run_id"]
    assert manifest["artifacts"]["tuning_ledger"] == _repo_relative(ledger_path)
    assert manifest["tuning_run_provenance"]["ledger_sha256"] == ledger["ledger_sha256"]


def test_legacy_tuning_run_alias_loads_provenance_block(tmp_path: Path) -> None:
    """The short legacy key remains accepted while preserving the typed provenance block."""
    config_path = _write_campaign(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("tuning_run_provenance:", "tuning_run:", 1),
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)
    assert cfg.tuning_run_provenance is not None
    assert cfg.tuning_run_provenance.run_id == "tuning-job-6595"


def test_strict_campaign_rejects_missing_prospective_provenance(tmp_path: Path) -> None:
    """The publication-style gate rejects an otherwise declared arm without a run record block."""
    config_path = _write_campaign(tmp_path, with_provenance=False)
    with pytest.raises(ValueError, match="complete 'tuning_run_provenance' block"):
        load_campaign_config(config_path)


def test_historical_campaign_config_dataclass_remains_compatible() -> None:
    """Programmatic legacy configs keep the optional provenance block unset."""
    cfg = CampaignConfig(
        name="legacy",
        scenario_matrix_path=Path("does-not-matter.yaml"),
        planners=(
            PlannerSpec(
                key="planner-a",
                algo="goal",
                tuning=TuningSpec(source="declared"),
            ),
        ),
        seed_policy=SeedPolicy(),
        scenario_candidates=ScenarioCandidateSelection(),
    )
    _validate_campaign_config(cfg)
    assert cfg.tuning_run_provenance is None
