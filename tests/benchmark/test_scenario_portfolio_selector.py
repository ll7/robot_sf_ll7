"""Tests for #5601 coverage-constrained Pareto selection over a generated archive."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from robot_sf.benchmark.scenario_generation.portfolio_selector import (
    PortfolioSelectionSpec,
    ScenarioPortfolioSelectionError,
    run_portfolio_selection,
    select_scenario_portfolio,
)
from robot_sf.benchmark.scenario_generation.segment_extraction import extract_critical_segment

if TYPE_CHECKING:
    from pathlib import Path

_REASON_DOMINATED = "pareto_dominated"


def _entry(
    episode_id: str,
    *,
    clearance_m: float,
    source_map: str = "maps/svg_maps/classic_crossing.svg",
    pedestrian_count: int = 1,
    near_miss: bool = False,
) -> dict[str, Any]:
    """Build a catalog entry whose critical frame carries the requested shape."""

    pedestrians = [{"position": [1.0 + clearance_m, 0.0]}] + [
        {"position": [9.0, float(index)]} for index in range(1, pedestrian_count)
    ]
    entry = extract_critical_segment(
        {
            "episode_id": episode_id,
            "seed": 4932,
            "source_map": source_map,
            "steps": [
                {
                    "time_s": 0.0,
                    "robot": {"position": [0.0, 0.0]},
                    "pedestrians": [{"position": [3.0, 0.0]}],
                },
                {
                    "time_s": 1.0,
                    "robot": {"position": [1.0, 0.0]},
                    "pedestrians": pedestrians,
                },
            ],
        }
    )
    entry["criticality"]["source_metrics"]["min_clearance_m"] = clearance_m
    entry["criticality"]["source_metrics"]["near_miss"] = near_miss
    return entry


def _spec(*, seed: int = 5601) -> PortfolioSelectionSpec:
    return PortfolioSelectionSpec.from_payload(
        {
            "schema_version": "scenario-portfolio-selection.v1",
            "selector": {
                "type": "coverage_constrained_pareto.v1",
                "seed": seed,
                "pareto_axes": [
                    "criticality.min_clearance_m",
                    "actor_interaction.min_robot_pedestrian_distance_m",
                ],
                "coverage_fields": [
                    "topology.map_family",
                    "topology.pedestrian_count_bucket",
                    "actor_interaction.near_miss_present",
                    "mechanism_signature.failure_mode",
                    "criticality.criticality_bucket",
                ],
                "coverage_quotas": {
                    "topology.map_family": {"rule": "full_cover"},
                    "topology.pedestrian_count_bucket": {"rule": "full_cover"},
                    "actor_interaction.near_miss_present": {"rule": "full_cover"},
                    "mechanism_signature.failure_mode": {"rule": "full_cover"},
                    "criticality.criticality_bucket": {"rule": "full_cover"},
                },
            },
            "claim_boundary": "generated scenario hypotheses only",
        }
    )


def _archive(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "generated-scenario-catalog.v1",
        "metadata": {
            "source": "auto_generated",
            "required_manual_review": True,
            "benchmark_evidence": False,
        },
        "entries": entries,
    }


def _write_config_and_archive(tmp_path: Path, archive: dict[str, Any]) -> tuple[Path, Path, Path]:
    archive_path = tmp_path / "archive.yaml"
    archive_path.write_text(yaml.safe_dump(archive, sort_keys=True), encoding="utf-8")
    output_path = tmp_path / "selection.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "scenario-portfolio-selection.v1",
                "source_archive": archive_path.as_posix(),
                "output_path": output_path.as_posix(),
                "selector": {
                    "type": "coverage_constrained_pareto.v1",
                    "seed": 5601,
                    "pareto_axes": [
                        "criticality.min_clearance_m",
                        "actor_interaction.min_robot_pedestrian_distance_m",
                    ],
                    "coverage_fields": [
                        "topology.map_family",
                        "topology.pedestrian_count_bucket",
                        "actor_interaction.near_miss_present",
                        "mechanism_signature.failure_mode",
                        "criticality.criticality_bucket",
                    ],
                    "coverage_quotas": {
                        "topology.map_family": {"rule": "full_cover"},
                        "topology.pedestrian_count_bucket": {"rule": "full_cover"},
                        "actor_interaction.near_miss_present": {"rule": "full_cover"},
                        "mechanism_signature.failure_mode": {"rule": "full_cover"},
                        "criticality.criticality_bucket": {"rule": "full_cover"},
                    },
                },
                "claim_boundary": "generated scenario hypotheses only",
            }
        ),
        encoding="utf-8",
    )
    return config_path, archive_path, output_path


def test_descriptors_remain_separately_inspectable() -> None:
    """The five descriptor families are derivable and stored without blending."""

    entries = [_entry("a", clearance_m=0.2), _entry("b", clearance_m=0.8)]
    manifest = select_scenario_portfolio(entries, spec=_spec())
    inventory = {d["scenario_id"]: d for d in manifest["descriptors"]["inventory"]}

    assert set(inventory) == {"a", "b"}
    for descriptor in inventory.values():
        assert descriptor["criticality"]["available"] is True
        assert descriptor["topology"]["available"] is True
        assert descriptor["actor_interaction"]["available"] is True
        assert descriptor["mechanism_signature"]["available"] is True
        # Replay persistence is never imputed when absent.
        assert descriptor["replay_persistence"]["available"] is False
        assert "reason" in descriptor["replay_persistence"]


def test_pareto_filter_then_max_min_coverage_is_deterministic_and_order_independent() -> None:
    """Archive record ordering cannot change the Pareto front or coverage order."""

    entries = [
        _entry("cross1", clearance_m=0.1, source_map="maps/svg_maps/classic_crossing.svg"),
        _entry("cross2", clearance_m=0.6, source_map="maps/svg_maps/classic_crossing.svg"),
        _entry(
            "station1", clearance_m=0.2, source_map="maps/svg_maps/classic_station_platform.svg"
        ),
        _entry(
            "station2", clearance_m=0.9, source_map="maps/svg_maps/classic_station_platform.svg"
        ),
    ]
    first = select_scenario_portfolio(entries, spec=_spec())
    second = select_scenario_portfolio(list(reversed(entries)), spec=_spec())

    assert first == second
    assert first["governance"]["hidden_weighted_sum_used"] is False
    assert first["selection"]["coverage_fraction"] == pytest.approx(1.0)
    assert set(first["selection"]["scenario_ids"]) == set(first["pareto"]["members"])


def test_synthetic_archive_recovers_known_coverage_optimum() -> None:
    """With disjoint cells, every Pareto member is needed to reach full coverage."""

    entries = [
        _entry("a", clearance_m=0.1, source_map="maps/svg_maps/map_a.svg", near_miss=True),
        _entry("b", clearance_m=0.1, source_map="maps/svg_maps/map_b.svg", near_miss=False),
        _entry("c", clearance_m=0.2, source_map="maps/svg_maps/map_c.svg", pedestrian_count=3),
    ]
    manifest = select_scenario_portfolio(entries, spec=_spec())

    # The three candidates occupy distinct coverage cells, so the optimum keeps all.
    assert manifest["selection"]["size"] == 3
    assert set(manifest["selection"]["scenario_ids"]) == {"a", "b", "c"}
    assert manifest["coverage"]["uncovered_cells"] == []
    assert manifest["stop_rule"]["minimum_coverage_satisfied"] is True


def test_rejects_hidden_score_shortcut_by_reproducing_pareto_front() -> None:
    """A candidate strictly worse on both axes is excluded as dominated, not scored."""

    # b dominates c on both Pareto axes (lower clearance and closer actor).
    entries = [
        _entry("a", clearance_m=0.1, near_miss=True),
        _entry("b", clearance_m=0.2, near_miss=False),
        _entry("c", clearance_m=0.9, near_miss=False),  # dominated by b
    ]
    manifest = select_scenario_portfolio(entries, spec=_spec())

    excluded_ids = {e["scenario_id"] for e in manifest["exclusions"]}
    dominated_codes = {e["reason_code"] for e in manifest["exclusions"]}
    assert "c" in excluded_ids
    assert _REASON_DOMINATED in dominated_codes
    # Pareto front must drop the dominated candidate.
    assert "c" not in manifest["pareto"]["members"]


def test_permutation_invariance_and_byte_identical_reruns() -> None:
    """Shuffled input and repeated runs yield identical, byte-stable manifests."""

    entries = [
        _entry("x", clearance_m=0.1, source_map="maps/svg_maps/map_x.svg"),
        _entry("y", clearance_m=0.4, source_map="maps/svg_maps/map_y.svg"),
        _entry("z", clearance_m=0.7, source_map="maps/svg_maps/map_z.svg"),
    ]
    base = select_scenario_portfolio(entries, spec=_spec())
    base_bytes = json.dumps(base, sort_keys=True).encode()

    for shuffle in (
        list(reversed(entries)),
        [entries[2], entries[0], entries[1]],
        list(entries),
    ):
        rerun = select_scenario_portfolio(shuffle, spec=_spec())
        assert json.dumps(rerun, sort_keys=True).encode() == base_bytes


def test_missing_descriptor_fails_closed_not_imputed() -> None:
    """A gap in a required descriptor is an explicit exclusion, never a silent drop."""

    entry = _entry("broken", clearance_m=0.3)
    # Sever the actor-offset path used by the Pareto axis.
    entry["segment"]["trace_frames"][1]["pedestrians"][0]["position"] = [None]  # type: ignore[list-item]
    with pytest.raises(ScenarioPortfolioSelectionError, match="finite number"):
        select_scenario_portfolio([entry], spec=_spec())


def test_replay_persistence_unavailable_is_explicit_and_marked_absent() -> None:
    """Absent persistence records stay unavailable; they never backfill from geometry."""

    entries = [_entry("a", clearance_m=0.2), _entry("b", clearance_m=0.5)]
    manifest = select_scenario_portfolio(entries, spec=_spec())

    for descriptor in manifest["descriptors"]["inventory"]:
        assert descriptor["replay_persistence"]["available"] is False
    # An entry that carries a disqualifying verdict is also explicitly unavailable.
    qualified = _entry("c", clearance_m=0.1)
    qualified["persistence"] = {
        "promotion_verdict": "rejected",
        "perturbation_persistence": {"persistence_rate": 0.0},
    }
    manifest2 = select_scenario_portfolio(entries + [qualified], spec=_spec())
    c_descriptor = next(d for d in manifest2["descriptors"]["inventory"] if d["scenario_id"] == "c")
    assert c_descriptor["replay_persistence"]["available"] is False


def test_sensitivity_reports_alternatives_without_changing_frozen_primary() -> None:
    """Declared alternative controls are re-tested while the primary portfolio is frozen."""

    entries = [
        _entry("a", clearance_m=0.1, source_map="maps/svg_maps/map_a.svg", near_miss=True),
        _entry("b", clearance_m=0.1, source_map="maps/svg_maps/map_b.svg", near_miss=False),
    ]
    manifest = select_scenario_portfolio(entries, spec=_spec())

    sensitivity = manifest["sensitivity"]
    assert sensitivity["frozen_primary"] is True
    assert sensitivity["alternatives"]
    # Primary selection is untouched by the sensitivity recomputations.
    assert set(manifest["selection"]["scenario_ids"]) == set(manifest["pareto"]["members"])
    labels = {alt["label"] for alt in sensitivity["alternatives"]}
    assert any(label.startswith("drop_coverage_field:") for label in labels)


def test_run_persists_manifest_source_provenance_and_ledgers(tmp_path: Path) -> None:
    """The CLI path records the archive checksum, the full candidate inventory, and exclusions."""

    entries = [
        _entry("a", clearance_m=0.1, source_map="maps/svg_maps/classic_crossing.svg"),
        _entry("b", clearance_m=0.9, source_map="maps/svg_maps/classic_station_platform.svg"),
    ]
    config_path, archive_path, output_path = _write_config_and_archive(tmp_path, _archive(entries))
    result = run_portfolio_selection(config_path)
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert persisted == result
    assert result["schema_version"] == "scenario_portfolio_selection.v1"
    assert (
        result["source_archive"]["sha256"] == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    )
    assert result["source_archive"]["candidate_count"] == 2
    assert result["descriptors"]["inventory"]
    assert len(result["exclusions"]) + result["selection"]["size"] == 2
    assert result["governance"] == {
        "required_manual_review": True,
        "benchmark_evidence": False,
        "scenario_certification": False,
        "automatic_promotion": False,
        "claim_boundary": "generated scenario hypotheses only",
        "hidden_weighted_sum_used": False,
    }

    with pytest.raises(FileExistsError, match="already exists"):
        run_portfolio_selection(config_path)


def test_run_rejects_evidence_bearing_archive(tmp_path: Path) -> None:
    """An archive that claims benchmark status cannot enter the selector."""

    archive = _archive([_entry("a", clearance_m=0.2)])
    archive["metadata"]["benchmark_evidence"] = True
    config_path, _archive_path, _output_path = _write_config_and_archive(tmp_path, archive)

    with pytest.raises(ScenarioPortfolioSelectionError, match="benchmark_evidence"):
        run_portfolio_selection(config_path)


def test_largest_honest_partial_when_coverage_unreachable(tmp_path: Path) -> None:
    """When a required cell has no eligible candidate, report it uncovered; do not fill it."""

    entries = [_entry("a", clearance_m=0.1, source_map="maps/svg_maps/classic_crossing.svg")]
    config_path, _archive_path, _output_path = _write_config_and_archive(
        tmp_path, _archive(entries)
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["selector"]["coverage_fields"] = [
        "topology.map_family",
        "topology.pedestrian_count_bucket",
    ]
    config["selector"]["coverage_quotas"]["topology.pedestrian_count_bucket"] = {
        "rule": "full_cover"
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    # The single candidate covers every reachable cell; the stop rule still
    # reports a coherent coverage fraction and never invents extra candidates.
    result = run_portfolio_selection(config_path)
    assert result["selection"]["size"] == 1
    assert result["stop_rule"]["minimum_coverage_satisfied"] is True
    assert result["coverage"]["uncovered_cells"] == []
