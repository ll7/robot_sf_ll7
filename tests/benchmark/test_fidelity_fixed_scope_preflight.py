"""Tests for the full fixed-scope fidelity-sensitivity preflight (issue #3207)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.fidelity_fixed_scope_preflight import (
    DECISION_BLOCKED,
    DECISION_READY,
    REQUIRED_CLAIM_BOUNDARY_PHRASES,
    SCHEMA_VERSION,
    build_fixed_scope_preflight,
    write_fixed_scope_preflight,
)
from robot_sf.benchmark.fidelity_sensitivity import load_fidelity_sensitivity_config

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = REPO_ROOT / "configs/research/fidelity_sensitivity_v1.yaml"


def _base_config() -> dict[str, Any]:
    """Return a minimal valid fidelity-sensitivity config with resolvable planners."""
    return {
        "schema_version": "fidelity-sensitivity.v1",
        "issue": 3207,
        "study_id": "unit_test_fixed_scope",
        "fixed_scope": {
            "scenario_set": "configs/benchmarks/paper_experiment_matrix_v1.yaml",
            "seeds": [111, 112, 113],
            "planner_groups": ["orca", "default_social_force"],
            "planner_algorithms": {
                "orca": "orca",
                "default_social_force": "social_force",
            },
        },
        "ranking": {"metric": "snqi", "higher_is_better": True},
        "metrics": [
            {"name": "snqi", "direction": "higher_is_better"},
            {"name": "success", "direction": "higher_is_better"},
        ],
        "axes": [
            {
                "key": f"axis_{index}",
                "variants": [
                    {"key": f"axis_{index}_nominal", "baseline": True},
                    {"key": f"axis_{index}_alt"},
                ],
            }
            for index in range(3)
        ],
    }


def _build(config: dict[str, Any]) -> dict[str, Any]:
    return build_fixed_scope_preflight(config, config_path="configs/x.yaml", git_head="deadbeef")


def test_ready_path_materializes_full_scope() -> None:
    """A resolvable config is preflight_ready with an explicit materialized scope."""
    packet = _build(_base_config())

    assert packet["schema_version"] == SCHEMA_VERSION
    assert packet["decision"] == DECISION_READY
    assert packet["preflight_ready"] is True
    assert packet["blockers"] == []

    scope = packet["materialized_scope"]
    # 2 planner groups x (3 axes x 2 variants = 6 variants) x 3 seeds = 36 cells.
    assert scope["planner_group_count"] == 2
    assert scope["axis_count"] == 3
    assert scope["variant_count"] == 6
    assert scope["seed_count"] == 3
    assert scope["run_cells_per_scenario"] == 36


def test_ready_path_records_claim_boundary_phrases() -> None:
    """The preflight packet carries the no-claim boundary phrases."""
    packet = _build(_base_config())
    for phrase in REQUIRED_CLAIM_BOUNDARY_PHRASES:
        assert phrase in packet["claim_boundary"]
    assert packet["evidence_status"].endswith("not_benchmark_evidence")


def test_primary_metric_identifiable_by_contract() -> None:
    """The declared ranking metric is reported identifiable by contract."""
    packet = _build(_base_config())
    primary = packet["primary_metric"]
    assert primary["metric"] == "snqi"
    assert primary["present_in_metrics"] is True
    assert primary["identifiable_by_contract"] is True
    assert primary["non_identifiable_reason"] is None


def test_unresolved_planner_fails_closed() -> None:
    """An unknown planner group with no explicit binding blocks the preflight."""
    config = _base_config()
    config["fixed_scope"]["planner_groups"].append("nonexistent_planner")
    packet = _build(config)

    assert packet["decision"] == DECISION_BLOCKED
    assert packet["preflight_ready"] is False
    assert "planner_unavailable:nonexistent_planner" in packet["blockers"]


def test_unbound_default_social_force_fails_closed() -> None:
    """Without the explicit algorithm binding, the label does not resolve."""
    config = _base_config()
    # Drop the explicit binding so the label is resolved directly (and fails).
    config["fixed_scope"].pop("planner_algorithms")
    packet = _build(config)

    assert packet["decision"] == DECISION_BLOCKED
    assert "planner_unavailable:default_social_force" in packet["blockers"]


def test_placeholder_planner_fails_closed() -> None:
    """A placeholder-tier catalog planner is treated as unavailable."""
    config = _base_config()
    config["fixed_scope"]["planner_groups"] = ["orca", "rvo"]
    config["fixed_scope"]["planner_algorithms"] = {"orca": "orca", "rvo": "rvo"}
    packet = _build(config)

    assert packet["decision"] == DECISION_BLOCKED
    assert "planner_placeholder:rvo" in packet["blockers"]


def test_non_identifiable_primary_metric_fails_closed() -> None:
    """A primary metric flagged non-identifiable blocks the preflight (issue #3299)."""
    config = _base_config()
    config["metrics"][0]["identifiable"] = False
    config["metrics"][0]["non_identifiable_reason"] = "primary_metric_zero_variance"
    packet = _build(config)

    assert packet["decision"] == DECISION_BLOCKED
    assert "primary_metric_non_identifiable:primary_metric_zero_variance" in packet["blockers"]
    assert packet["primary_metric"]["identifiable_by_contract"] is False


def test_experimental_planner_recorded_as_launch_prerequisite() -> None:
    """Opt-in planners resolve but surface an explicit launch prerequisite."""
    config = _base_config()
    config["fixed_scope"]["planner_groups"].append("hybrid_rule_v0_minimal")
    config["fixed_scope"]["planner_algorithms"]["hybrid_rule_v0_minimal"] = "hybrid_rule_v0_minimal"
    packet = _build(config)

    assert packet["decision"] == DECISION_READY
    prereqs = "\n".join(packet["launch_prerequisites"])
    assert "planner_requires_explicit_opt_in:hybrid_rule_v0_minimal" in prereqs


def test_experimental_planner_opt_in_satisfies_launch_prerequisite() -> None:
    """Fixed-scope opt-in binds hybrid-rule planner without leaving a stale gate."""
    config = _base_config()
    config["fixed_scope"]["planner_groups"].append("hybrid_rule_v0_minimal")
    config["fixed_scope"]["planner_algorithms"]["hybrid_rule_v0_minimal"] = "hybrid_rule_v0_minimal"
    config["fixed_scope"]["planner_opt_ins"] = {
        "hybrid_rule_v0_minimal": {"allow_testing_algorithms": True}
    }
    packet = _build(config)
    prereqs = "\n".join(packet["launch_prerequisites"])
    assert "planner_requires_explicit_opt_in:hybrid_rule_v0_minimal" not in prereqs
    record = next(
        item
        for item in packet["planner_resolution"]
        if item["planner_group"] == "hybrid_rule_v0_minimal"
    )
    assert record["catalog_requires_explicit_opt_in"] is True
    assert record["explicit_opt_in_satisfied"] is True
    assert record["requires_explicit_opt_in"] is False


def test_orca_rvo2_missing_stays_launch_prerequisite(monkeypatch: pytest.MonkeyPatch) -> None:
    """ORCA binds only when rvo2 is importable; otherwise launch remains fail-closed."""
    monkeypatch.setattr(
        "robot_sf.benchmark.fidelity_fixed_scope_preflight._rvo2_importable",
        lambda: False,
    )
    packet = _build(_base_config())
    prereqs = "\n".join(packet["launch_prerequisites"])
    assert "planner_requires_rvo2:orca" in prereqs


def test_bad_planner_algorithms_type_raises() -> None:
    """A non-mapping planner_algorithms surface is a hard config error."""
    config = _base_config()
    config["fixed_scope"]["planner_algorithms"] = ["orca"]
    with pytest.raises(ValueError, match="planner_algorithms"):
        _build(config)


def test_write_packet_round_trips(tmp_path: Path) -> None:
    """Writing then reading the packet preserves the decision."""
    packet = _build(_base_config())
    out = write_fixed_scope_preflight(packet, tmp_path)
    assert out.name == "fidelity_fixed_scope_preflight.json"
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["decision"] == packet["decision"]
    assert loaded["schema_version"] == SCHEMA_VERSION


def test_real_config_is_preflight_ready() -> None:
    """The shipped #3207 config resolves to a ready preflight with the full scope."""
    config = load_fidelity_sensitivity_config(REAL_CONFIG)
    packet = build_fixed_scope_preflight(
        config, config_path="configs/research/fidelity_sensitivity_v1.yaml", git_head="deadbeef"
    )
    assert packet["decision"] == DECISION_READY
    scope = packet["materialized_scope"]
    expected_variant_count = sum(len(axis["variants"]) for axis in config["axes"])
    expected_run_cells = (
        len(config["fixed_scope"]["planner_groups"])
        * expected_variant_count
        * len(config["fixed_scope"]["seeds"])
    )
    assert scope["planner_group_count"] == 3
    assert scope["variant_count"] == expected_variant_count
    assert scope["run_cells_per_scenario"] == expected_run_cells
    # Every resolved planner maps to a canonical catalog name.
    assert all(record["canonical_name"] for record in packet["planner_resolution"])


def test_real_config_unchanged_by_preflight() -> None:
    """Building the packet does not mutate the input config."""
    config = load_fidelity_sensitivity_config(REAL_CONFIG)
    before = copy.deepcopy(config)
    build_fixed_scope_preflight(config, config_path="x", git_head="x")
    assert config == before
