"""Tests for AMMV-aware Social Force paired diagnostic tooling."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.tools.run_ammv_social_force_pair_diagnostic import (
    _infer_scenario_id,
    _run_mechanism_probe,
)

AMMV_CONFIG = Path("configs/baselines/social_force_ammv_aware.yaml")


def test_issue_3202_anticipatory_crossing_probe_activates_ammv_term() -> None:
    """The #3202 direct probe should expose a same-seed AMMV behavior delta."""
    probe = _run_mechanism_probe(
        AMMV_CONFIG,
        probe_name="issue_3202_anticipatory_crossing_probe",
    )

    ammv_trace = probe["traces"]["ammv_social_force"]

    assert probe["name"] == "issue_3202_anticipatory_crossing_probe"
    assert probe["seed"] == 3202
    assert probe["verdict"] == "behavioral_delta_found"
    assert ammv_trace["max_ammv_force_magnitude"] > 0.0
    assert ammv_trace["max_intrusion_count"] >= 1
    assert probe["paired_delta"]["final_robot_lateral_offset_m"] != pytest.approx(0.0)


def test_scenario_id_inferred_from_paired_records() -> None:
    """Evidence packs should not stamp the old #2168 scenario onto new rows."""
    scenario_id = _infer_scenario_id(
        default_records=[{"scenario_id": "issue_3202_anticipatory_conflict"}],
        ammv_records=[{"scenario_id": "issue_3202_anticipatory_conflict"}],
        fallback=Path("configs/scenarios/sets/issue_3202_ammv_anticipatory_conflict.yaml"),
    )

    assert scenario_id == "issue_3202_anticipatory_conflict"


def test_scenario_id_falls_back_to_config_stem_when_records_lack_ids() -> None:
    """Malformed or synthetic inputs should still produce a stable identifier."""
    scenario_id = _infer_scenario_id(
        default_records=[{"metrics": {}}],
        ammv_records=[{"metrics": {}}],
        fallback=Path("configs/scenarios/sets/issue_3202_ammv_anticipatory_conflict.yaml"),
    )

    assert scenario_id == "issue_3202_ammv_anticipatory_conflict"
