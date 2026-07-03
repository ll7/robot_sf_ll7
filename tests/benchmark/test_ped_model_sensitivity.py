"""Tests for issue #3950 pedestrian-model sensitivity reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark.ped_model_sensitivity import (
    build_pedestrian_model_provenance,
    build_sensitivity_summary,
    load_jsonl_records,
    resolve_development_pedestrian_model,
    write_sensitivity_report,
)
from robot_sf.sim.pedestrian_model_variants import HSFM_TOTAL_FORCE_V1, SOCIAL_FORCE_DEFAULT


def test_episode_provenance_records_development_and_evaluation_models() -> None:
    """Episode provenance separates declared development and active evaluation model."""

    provenance = build_pedestrian_model_provenance(
        sim_config=SimpleNamespace(pedestrian_model=HSFM_TOTAL_FORCE_V1),
        policy_cfg={"development_pedestrian_model": SOCIAL_FORCE_DEFAULT},
        algorithm_metadata={},
    )

    assert provenance["development_model"] == SOCIAL_FORCE_DEFAULT
    assert provenance["evaluation_model"] == HSFM_TOTAL_FORCE_V1
    assert provenance["source"] == "simulation_config.pedestrian_model"
    assert provenance["claim_boundary"] == "diagnostic_cpu_sensitivity_no_training"
    assert provenance["fallback_degraded_status"] == "native"
    assert isinstance(provenance["selector_config_hash"], str)


def test_episode_provenance_uses_unknown_for_missing_development_model() -> None:
    """Existing policies must not be assigned guessed training pedestrian provenance."""

    provenance = build_pedestrian_model_provenance(
        sim_config=SimpleNamespace(pedestrian_model=SOCIAL_FORCE_DEFAULT),
        policy_cfg={},
        algorithm_metadata={},
    )

    assert provenance["development_model"] == "unknown"
    assert provenance["evaluation_model"] == SOCIAL_FORCE_DEFAULT


def test_development_model_resolver_uses_algorithm_metadata_fallback() -> None:
    """Human-model metadata can declare provenance when policy config omits it."""

    assert (
        resolve_development_pedestrian_model(
            None,
            algorithm_metadata={"human_model_variant": "upstream_hsfm"},
        )
        == "upstream_hsfm"
    )


def test_load_jsonl_records_reads_objects_and_rejects_non_objects(tmp_path: Path) -> None:
    """JSONL loader keeps blank lines harmless and fails closed on malformed rows."""

    records_path = tmp_path / "records.jsonl"
    records_path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    assert load_jsonl_records(records_path) == [{"a": 1}, {"b": 2}]

    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected JSON object"):
        load_jsonl_records(bad_path)


def test_sensitivity_summary_emits_four_cells_and_unavailable_missing_rows() -> None:
    """2x2 matrix keeps missing cells explicit instead of silently dropping them."""

    records = [
        _record(SOCIAL_FORCE_DEFAULT, SOCIAL_FORCE_DEFAULT, success=True, collision=False),
        _record(SOCIAL_FORCE_DEFAULT, HSFM_TOTAL_FORCE_V1, success=False, collision=True),
        _record(HSFM_TOTAL_FORCE_V1, SOCIAL_FORCE_DEFAULT, success=True, collision=False),
    ]

    summary = build_sensitivity_summary(
        records,
        development_models=[SOCIAL_FORCE_DEFAULT, HSFM_TOTAL_FORCE_V1],
        evaluation_models=[SOCIAL_FORCE_DEFAULT, HSFM_TOTAL_FORCE_V1],
        planner_key="goal",
        algo="goal",
    )

    assert summary["schema_version"] == "ped_model_sensitivity.v1"
    assert len(summary["cells"]) == 4
    by_key = {
        (cell["development_model"], cell["evaluation_model"]): cell for cell in summary["cells"]
    }
    assert by_key[(SOCIAL_FORCE_DEFAULT, SOCIAL_FORCE_DEFAULT)]["success_incidence"] == 1.0
    assert by_key[(SOCIAL_FORCE_DEFAULT, HSFM_TOTAL_FORCE_V1)]["collision_incidence"] == 1.0
    assert by_key[(HSFM_TOTAL_FORCE_V1, HSFM_TOTAL_FORCE_V1)]["status"] == "unavailable"
    assert by_key[(HSFM_TOTAL_FORCE_V1, HSFM_TOTAL_FORCE_V1)]["episodes"] == 0


def test_sensitivity_summary_handles_unknown_and_metric_only_rows() -> None:
    """Aggregation handles unknown development provenance and metric-only records."""

    summary = build_sensitivity_summary(
        [
            {
                "development_pedestrian_model": "unknown",
                "evaluation_pedestrian_model": SOCIAL_FORCE_DEFAULT,
                "metrics": {"success": 1.0, "collisions": 1.0},
                "algorithm_metadata": {"status": "fallback"},
            }
        ],
        development_models=["unknown"],
        evaluation_models=[SOCIAL_FORCE_DEFAULT],
        planner_key="goal",
        algo="goal",
    )

    cell = summary["cells"][0]
    assert cell["success_incidence"] == 1.0
    assert cell["collision_incidence"] == 1.0
    assert cell["fallback_degraded_rows"] == 1
    assert cell["status"] == "degraded"


def test_sensitivity_summary_rejects_unknown_evaluation_model() -> None:
    """Unsupported evaluation pedestrian models fail closed."""

    with pytest.raises(ValueError, match="Unsupported pedestrian_model"):
        build_sensitivity_summary(
            [],
            development_models=[SOCIAL_FORCE_DEFAULT],
            evaluation_models=["bogus_model"],
            planner_key="goal",
            algo="goal",
        )


def test_report_writers_are_deterministic(tmp_path: Path) -> None:
    """Report writer emits stable JSON, CSV, and Markdown surfaces."""

    summary = build_sensitivity_summary(
        [_record(SOCIAL_FORCE_DEFAULT, HSFM_TOTAL_FORCE_V1, success=True, collision=False)],
        development_models=[SOCIAL_FORCE_DEFAULT],
        evaluation_models=[HSFM_TOTAL_FORCE_V1],
        planner_key="goal",
        algo="goal",
    )

    paths = write_sensitivity_report(summary, tmp_path)

    loaded = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    assert loaded == summary
    with Path(paths["sensitivity_matrix_csv"]).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "development_model": SOCIAL_FORCE_DEFAULT,
            "evaluation_model": HSFM_TOTAL_FORCE_V1,
            "planner_key": "goal",
            "algo": "goal",
            "episodes": "1",
            "success_incidence": "1.0",
            "collision_incidence": "0.0",
            "fallback_degraded_rows": "0",
            "status": "ok",
        }
    ]
    readme = Path(paths["readme"]).read_text(encoding="utf-8")
    assert "CPU-only diagnostic smoke report" in readme
    assert "No new training" in readme


def test_report_writer_renders_unavailable_rates_as_na(tmp_path: Path) -> None:
    """Markdown report renders missing cells without numeric placeholders."""

    summary = build_sensitivity_summary(
        [],
        development_models=[SOCIAL_FORCE_DEFAULT],
        evaluation_models=[HSFM_TOTAL_FORCE_V1],
        planner_key="goal",
        algo="goal",
    )

    paths = write_sensitivity_report(summary, tmp_path)

    readme = Path(paths["readme"]).read_text(encoding="utf-8")
    assert "| social_force_default | hsfm_total_force_v1 | 0 | NA | NA | unavailable |" in readme


def _record(
    development_model: str,
    evaluation_model: str,
    *,
    success: bool,
    collision: bool,
) -> dict[str, object]:
    return {
        "development_pedestrian_model": development_model,
        "evaluation_pedestrian_model": evaluation_model,
        "outcome": {"success": success, "collision": collision},
        "algorithm_metadata": {},
    }
