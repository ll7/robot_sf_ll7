"""Tests for per-arm tuning-effort metadata in campaign manifests (issue #5143).

Covers the declared ``tuning`` block parsing, the best-effort ``backfill_pending`` synthesis that
makes cross-arm tuning asymmetry visible, the manifest emission of the per-arm block plus a
tuning-effort summary, and the fail-closed ``tuning_effort_enforcement='error'`` gate (mirroring
#4970's checkpoint-provenance fail-closed spirit).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.camera_ready._config import _parse_tuning_spec, _validate_campaign_config
from robot_sf.benchmark.camera_ready._config_types import (
    TUNING_SOURCE_BACKFILLED,
    TUNING_SOURCE_DECLARED,
    CampaignConfig,
    PlannerSpec,
    ScenarioCandidateSelection,
    SeedPolicy,
    TuningSpec,
)
from robot_sf.benchmark.camera_ready._preflight import (
    _tuning_effort_block,
    _tuning_effort_summary,
    prepare_campaign_preflight,
)
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config


def _parse_tuning(raw: object) -> TuningSpec | None:
    return _parse_tuning_spec(raw, planner_key="arm")


def test_parse_tuning_spec_returns_none_when_absent() -> None:
    """An absent tuning block parses to None so the manifest can backfill later."""
    assert _parse_tuning(None) is None


def test_parse_tuning_spec_reads_declared_block() -> None:
    """A fully-declared tuning block round-trips into a TuningSpec with source='declared'."""
    tuning = _parse_tuning(
        {
            "parameters_touched": ["v_max", "time_horizon"],
            "tuning_scenario_ids": ["tune_a", "tune_b"],
            "eval_set_disjoint": True,
            "budget_runs": 50,
            "budget_hours": 2.5,
            "tuned_by": "j.doe",
            "tuned_at_utc": "2026-07-10T00:00:00Z",
            "source": "declared",
        }
    )
    assert tuning is not None
    assert tuning.parameters_touched == ("v_max", "time_horizon")
    assert tuning.tuning_scenario_ids == ("tune_a", "tune_b")
    assert tuning.eval_set_disjoint is True
    assert tuning.budget_runs == 50
    assert tuning.budget_hours == pytest.approx(2.5)
    assert tuning.tuned_by == "j.doe"
    assert tuning.tuned_at_utc == "2026-07-10T00:00:00Z"
    assert tuning.source == TUNING_SOURCE_DECLARED


def test_parse_tuning_spec_defaults_source_to_declared() -> None:
    """Omitting source defaults to 'declared' (the author declared it)."""
    tuning = _parse_tuning({"parameters_touched": ["v_max"]})
    assert tuning is not None
    assert tuning.source == TUNING_SOURCE_DECLARED


def test_parse_tuning_spec_marks_backfilled_source() -> None:
    """A best-effort reconstruction records source='backfilled' honestly."""
    tuning = _parse_tuning({"parameters_touched": ["v_max"], "source": "backfilled"})
    assert tuning is not None
    assert tuning.source == TUNING_SOURCE_BACKFILLED


@pytest.mark.parametrize("field", ["parameters_touched", "tuning_scenario_ids"])
def test_parse_tuning_spec_rejects_non_list_field(field: str) -> None:
    """A dict for a string-list field is rejected; scalars are coerced (codebase pattern)."""
    with pytest.raises(TypeError, match="must be a list of strings"):
        _parse_tuning({field: {"not": "a list"}})


@pytest.mark.parametrize("field", ["parameters_touched", "tuning_scenario_ids"])
def test_parse_tuning_spec_coerces_scalar_to_single_entry(field: str) -> None:
    """Scalar entries are coerced to a single-element tuple (matches scenario_candidates pattern)."""
    tuning = _parse_tuning({field: "v_max"})
    assert tuning is not None
    assert getattr(tuning, field) == ("v_max",)


def test_parse_tuning_spec_rejects_invalid_source_vocabulary() -> None:
    """An unknown tuning.source value is rejected to keep the provenance vocabulary honest."""
    with pytest.raises(ValueError, match="is not one of"):
        _parse_tuning({"parameters_touched": ["v_max"], "source": "guessed"})


def test_parse_tuning_spec_rejects_non_mapping_block() -> None:
    """A non-mapping tuning block is rejected with a clear planner-scoped message."""
    with pytest.raises(ValueError, match="must be a mapping"):
        _parse_tuning_spec(["not", "a", "mapping"], planner_key="arm")  # type: ignore[arg-type]


def test_parse_tuning_spec_rejects_negative_budget() -> None:
    """Negative tuning budgets are rejected since budget is an approximate but non-negative cost."""
    with pytest.raises(ValueError, match="non-negative"):
        _parse_tuning({"budget_runs": -1})
    with pytest.raises(ValueError, match="non-negative"):
        _parse_tuning({"budget_hours": -0.5})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_parse_tuning_spec_rejects_non_finite_budget_hours(value: float) -> None:
    """Non-finite values must not reach JSON campaign-manifest output."""
    with pytest.raises(ValueError, match="budget_hours must be non-negative"):
        _parse_tuning({"budget_hours": value})


def test_parse_tuning_spec_rejects_non_bool_disjoint_flag() -> None:
    """The eval-set disjointness flag must be a boolean when provided."""
    with pytest.raises(TypeError, match="eval_set_disjoint must be a boolean"):
        _parse_tuning({"eval_set_disjoint": "yes"})


def test_tuning_effort_block_synthesizes_backfill_pending_when_undeclared() -> None:
    """An undeclared arm still gets a manifest block so asymmetry is visible (not silent)."""
    planner = PlannerSpec(key="classical_a", algo="goal")
    block = _tuning_effort_block(planner)
    assert block["source"] == "backfill_pending"
    assert block["parameters_touched"] == []
    assert block["tuning_scenario_ids"] == []
    assert "note" in block
    assert planner.tuning is None


def test_tuning_effort_block_emits_declared_block_verbatim() -> None:
    """A declared tuning block is emitted verbatim with no caveat note."""
    planner = PlannerSpec(
        key="learned_a",
        algo="ppo",
        tuning=TuningSpec(parameters_touched=("lr",), source=TUNING_SOURCE_DECLARED),
    )
    block = _tuning_effort_block(planner)
    assert block["source"] == TUNING_SOURCE_DECLARED
    assert block["parameters_touched"] == ["lr"]
    assert "note" not in block


def test_tuning_effort_block_flags_non_declared_source_with_note() -> None:
    """A backfilled (non-declared) source is still emitted but flagged with a caveat note."""
    planner = PlannerSpec(
        key="mpc_a",
        algo="mpc",
        tuning=TuningSpec(parameters_touched=("horizon",), source=TUNING_SOURCE_BACKFILLED),
    )
    block = _tuning_effort_block(planner)
    assert block["source"] == TUNING_SOURCE_BACKFILLED
    assert "note" in block


def test_tuning_effort_summary_counts_declared_vs_backfill_pending() -> None:
    """The summary exposes how many arms are declared vs backfill-pending and which are missing."""
    planners = (
        PlannerSpec(
            key="a",
            algo="goal",
            tuning=TuningSpec(source=TUNING_SOURCE_DECLARED),
        ),
        PlannerSpec(key="b", algo="orca"),
        PlannerSpec(
            key="c",
            algo="mpc",
            enabled=False,
            tuning=TuningSpec(source=TUNING_SOURCE_BACKFILLED),
        ),
    )
    summary = _tuning_effort_summary(planners)
    assert summary["enabled_arm_count"] == 2
    assert summary["declared_count"] == 1
    assert summary["backfill_pending_count"] == 1
    assert summary["arms_missing_tuning"] == ["b"]
    assert summary["by_source"] == {TUNING_SOURCE_DECLARED: 1, "backfill_pending": 1}


def _write_minimal_campaign(
    tmp_path: Path,
    *,
    planners_yaml: str,
    extra_top_level: str = "",
) -> Path:
    """Write a minimal valid campaign config with the given planners block."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: tuning_effort_cfg",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                extra_top_level,
                "planners:",
                planners_yaml,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def test_load_campaign_config_parses_tuning_block(tmp_path: Path) -> None:
    """A declared per-arm tuning block round-trips through the YAML config loader."""
    config_path = _write_minimal_campaign(
        tmp_path,
        planners_yaml=(
            "  - key: learned_a\n"
            "    algo: ppo\n"
            "    planner_group: core\n"
            "    tuning:\n"
            "      parameters_touched: [lr, clip_range]\n"
            "      tuning_scenario_ids: [tune_a, tune_b]\n"
            "      eval_set_disjoint: true\n"
            "      budget_runs: 40\n"
            "      budget_hours: 3.0\n"
            "      tuned_by: j.doe\n"
            "      tuned_at_utc: '2026-07-10T00:00:00Z'\n"
            "      source: declared\n"
            "  - key: classical_a\n"
            "    algo: goal\n"
            "    planner_group: core\n"
        ),
    )
    cfg = load_campaign_config(config_path)
    learned, classical = cfg.planners
    assert learned.tuning is not None
    assert learned.tuning.parameters_touched == ("lr", "clip_range")
    assert learned.tuning.tuning_scenario_ids == ("tune_a", "tune_b")
    assert learned.tuning.eval_set_disjoint is True
    assert learned.tuning.budget_runs == 40
    assert learned.tuning.budget_hours == pytest.approx(3.0)
    assert learned.tuning.tuned_by == "j.doe"
    assert learned.tuning.source == TUNING_SOURCE_DECLARED
    assert classical.tuning is None


def test_load_campaign_config_defaults_tuning_effort_enforcement_off(tmp_path: Path) -> None:
    """Backwards compatibility: configs without the new field load with enforcement='off'."""
    config_path = _write_minimal_campaign(
        tmp_path,
        planners_yaml="  - key: a\n    algo: goal\n    planner_group: core\n",
    )
    cfg = load_campaign_config(config_path)
    assert cfg.tuning_effort_enforcement == "off"


def test_validate_campaign_config_error_gate_fails_when_arm_missing_tuning() -> None:
    """tuning_effort_enforcement='error' fails closed when any enabled arm lacks a tuning block."""
    cfg = CampaignConfig(
        name="gate",
        scenario_matrix_path=Path("does-not-matter.yaml"),
        planners=(
            PlannerSpec(
                key="a",
                algo="goal",
                tuning=TuningSpec(source=TUNING_SOURCE_DECLARED),
            ),
            PlannerSpec(key="b", algo="orca"),
        ),
        seed_policy=SeedPolicy(),
        scenario_candidates=ScenarioCandidateSelection(),
        tuning_effort_enforcement="error",
    )
    with pytest.raises(ValueError, match="missing declared tuning block for: b"):
        _validate_campaign_config(cfg)


@pytest.mark.parametrize("source", [TUNING_SOURCE_BACKFILLED, "unknown"])
def test_validate_campaign_config_error_gate_rejects_non_declared_tuning(source: str) -> None:
    """Strict enforcement cannot accept a backfilled or unknown record as a new-arm declaration."""
    cfg = CampaignConfig(
        name="gate",
        scenario_matrix_path=Path("does-not-matter.yaml"),
        planners=(
            PlannerSpec(
                key="a",
                algo="goal",
                tuning=TuningSpec(source=source),
            ),
        ),
        seed_policy=SeedPolicy(),
        scenario_candidates=ScenarioCandidateSelection(),
        tuning_effort_enforcement="error",
    )
    with pytest.raises(ValueError, match="missing declared tuning block for: a"):
        _validate_campaign_config(cfg)


def test_validate_campaign_config_error_gate_passes_when_all_enabled_arms_declared() -> None:
    """When every enabled arm declares tuning, the error gate passes; disabled arms are exempt."""
    cfg = CampaignConfig(
        name="gate",
        scenario_matrix_path=Path("does-not-matter.yaml"),
        planners=(
            PlannerSpec(
                key="a",
                algo="goal",
                tuning=TuningSpec(source=TUNING_SOURCE_DECLARED),
            ),
            PlannerSpec(
                key="b",
                algo="orca",
                enabled=False,
            ),
        ),
        seed_policy=SeedPolicy(),
        scenario_candidates=ScenarioCandidateSelection(),
        tuning_effort_enforcement="error",
    )
    # A disabled arm is exempt; should not raise.
    _validate_campaign_config(cfg)


def test_validate_campaign_config_error_gate_ignores_disabled_arms() -> None:
    """A disabled arm missing a tuning block does not trigger the error gate."""
    cfg = CampaignConfig(
        name="gate",
        scenario_matrix_path=Path("does-not-matter.yaml"),
        planners=(
            PlannerSpec(key="a", algo="goal", enabled=False),
            PlannerSpec(
                key="b",
                algo="orca",
                tuning=TuningSpec(source=TUNING_SOURCE_DECLARED),
            ),
        ),
        seed_policy=SeedPolicy(),
        scenario_candidates=ScenarioCandidateSelection(),
        tuning_effort_enforcement="error",
    )
    _validate_campaign_config(cfg)


def test_validate_campaign_config_rejects_unknown_enforcement_value() -> None:
    """An unrecognized enforcement vocabulary value is rejected at validation time."""
    cfg = CampaignConfig(
        name="gate",
        scenario_matrix_path=Path("does-not-matter.yaml"),
        planners=(PlannerSpec(key="a", algo="goal"),),
        seed_policy=SeedPolicy(),
        scenario_candidates=ScenarioCandidateSelection(),
        tuning_effort_enforcement="strict",
    )
    with pytest.raises(ValueError, match="Unsupported tuning_effort_enforcement 'strict'"):
        _validate_campaign_config(cfg)


def test_load_campaign_config_error_gate_rejects_missing_tuning(tmp_path: Path) -> None:
    """The error gate fires during config load, before any campaign artifact exists."""
    config_path = _write_minimal_campaign(
        tmp_path,
        planners_yaml="  - key: a\n    algo: goal\n    planner_group: core\n",
        extra_top_level="tuning_effort_enforcement: error",
    )
    with pytest.raises(ValueError, match="missing declared tuning block for: a"):
        load_campaign_config(config_path)


def test_prepare_campaign_preflight_writes_tuning_blocks_and_summary(tmp_path: Path) -> None:
    """The campaign manifest must carry a per-arm tuning block for every arm plus a coverage summary."""
    config_path = _write_minimal_campaign(
        tmp_path,
        planners_yaml=(
            "  - key: learned_a\n"
            "    algo: ppo\n"
            "    planner_group: core\n"
            "    tuning:\n"
            "      parameters_touched: [lr]\n"
            "      tuning_scenario_ids: [tune_a]\n"
            "      eval_set_disjoint: true\n"
            "      budget_runs: 40\n"
            "      budget_hours: 3.0\n"
            "      tuned_by: j.doe\n"
            "      tuned_at_utc: '2026-07-10T00:00:00Z'\n"
            "      source: declared\n"
            "  - key: classical_a\n"
            "    algo: goal\n"
            "    planner_group: core\n"
        ),
    )
    cfg = load_campaign_config(config_path)
    prepared = prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="issue5143")
    manifest = json.loads(
        (Path(prepared["campaign_root"]) / "campaign_manifest.json").read_text(encoding="utf-8")
    )

    planners = {planner["key"]: planner for planner in manifest["planners"]}
    # Declared arm emits its block verbatim.
    learned = planners["learned_a"]
    assert learned["tuning"]["source"] == TUNING_SOURCE_DECLARED
    assert learned["tuning"]["parameters_touched"] == ["lr"]
    assert learned["tuning"]["tuning_scenario_ids"] == ["tune_a"]
    assert learned["tuning"]["eval_set_disjoint"] is True
    assert learned["tuning"]["budget_runs"] == 40
    assert learned["tuning"]["budget_hours"] == pytest.approx(3.0)
    assert learned["tuning"]["tuned_by"] == "j.doe"
    assert learned["tuning"]["tuned_at_utc"] == "2026-07-10T00:00:00Z"
    # Undeclared classical arm still carries a backfill_pending block (asymmetry visible).
    classical = planners["classical_a"]
    assert classical["tuning"]["source"] == "backfill_pending"
    assert classical["tuning"]["parameters_touched"] == []
    # Top-level summary surfaces coverage.
    assert manifest["tuning_effort_enforcement"] == "off"
    summary = manifest["tuning_effort_summary"]
    assert summary["enabled_arm_count"] == 2
    assert summary["declared_count"] == 1
    assert summary["backfill_pending_count"] == 1
    assert summary["arms_missing_tuning"] == ["classical_a"]


def test_prepare_campaign_preflight_error_gate_aborts_before_artifacts(tmp_path: Path) -> None:
    """Fail-closed enforcement must abort preflight before producing artifacts (issue #5143)."""
    # Construct a config directly (bypassing load_campaign_config) with enforcement='error' and an
    # enabled arm missing its tuning block. Preflight injects _validate_campaign_config, so the
    # fail-closed gate must fire before any campaign artifact is written.
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    cfg = CampaignConfig(
        name="gate_campaign",
        scenario_matrix_path=scenario_abs,
        planners=(PlannerSpec(key="a", algo="goal", planner_group="core"),),
        seed_policy=SeedPolicy(mode="fixed-list", seeds=(111,)),
        scenario_candidates=ScenarioCandidateSelection(),
        tuning_effort_enforcement="error",
    )
    with pytest.raises(ValueError, match="missing declared tuning block for: a"):
        prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="issue5143")
    # No campaign root should have been created on the fail-closed path.
    assert not (tmp_path / "out").exists() or not list(
        (tmp_path / "out").rglob("campaign_manifest.json")
    )
