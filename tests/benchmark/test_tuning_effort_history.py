"""Regression tests for historical tuning-effort backfill coverage (issue #5143)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.tuning_effort_history import DEFAULT_REGISTRY, main, validate_registry


def _explicit_tuning(source: str) -> dict[str, object]:
    return {
        "parameters_touched": ["speed"] if source == "backfilled" else [],
        "tuning_scenario_ids": [],
        "eval_set_disjoint": None,
        "budget_runs": None,
        "budget_hours": None,
        "tuned_by": None,
        "tuned_at_utc": None,
        "source": source,
    }


def _write_fixture(
    tmp_path: Path,
    *,
    records: list[dict[str, object]],
    planners: list[dict[str, object]] | None = None,
) -> Path:
    planners = planners or [{"key": "arm", "algo": "goal", "algo_config_path": None}]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"planners": planners}), encoding="utf-8")
    evidence = tmp_path / "evidence.yaml"
    evidence.write_text("evidence: retained\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "campaign_manifests": [manifest.name],
                "records": records,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return registry


def _record(*, source: str = "unknown") -> dict[str, object]:
    record: dict[str, object] = {
        "planner": {"key": "arm", "algo": "goal", "algo_config_path": None},
        "tuning": _explicit_tuning(source),
        "evidence_refs": ["evidence.yaml"],
    }
    if source == "unknown":
        record["unknown_reason"] = "Retained evidence does not identify tuning effort."
    return record


def test_repository_history_registry_covers_all_tracked_legacy_arms() -> None:
    """Every arm in every declared legacy manifest has one explicit backfilled/unknown record."""
    repo_root = Path(__file__).resolve().parents[2]
    summary = validate_registry(repo_root / DEFAULT_REGISTRY, repo_root=repo_root)
    assert summary == {
        "status": "ok",
        "campaign_manifest_count": 7,
        "arm_occurrence_count": 59,
        "unique_arm_record_count": 15,
        "arm_occurrences_by_source": {"backfilled": 16, "unknown": 43},
    }


def test_registry_fails_closed_when_historical_arm_is_missing(tmp_path: Path) -> None:
    """A legacy arm without a registry record is a blocker, not an implicit unknown."""
    registry = _write_fixture(
        tmp_path,
        records=[_record()],
        planners=[
            {"key": "arm", "algo": "goal", "algo_config_path": None},
            {"key": "missing", "algo": "orca", "algo_config_path": None},
        ],
    )
    with pytest.raises(ValueError, match="missing historical tuning record"):
        validate_registry(registry, repo_root=tmp_path)


def test_registry_fails_closed_on_duplicate_signature(tmp_path: Path) -> None:
    """Ambiguous records cannot silently choose one provenance account."""
    record = _record()
    registry = _write_fixture(tmp_path, records=[record, record])
    with pytest.raises(ValueError, match="duplicate tuning record"):
        validate_registry(registry, repo_root=tmp_path)


def test_unknown_record_cannot_populate_inferred_fields(tmp_path: Path) -> None:
    """Unknown means unknown; values cannot be smuggled in without backfill evidence."""
    record = _record()
    tuning = record["tuning"]
    assert isinstance(tuning, dict)
    tuning["budget_runs"] = 0
    registry = _write_fixture(tmp_path, records=[record])
    with pytest.raises(ValueError, match="unknown record fabricates populated fields"):
        validate_registry(registry, repo_root=tmp_path)


def test_backfilled_record_requires_parameter_and_evidence_paths(tmp_path: Path) -> None:
    """Backfilled status requires concrete retained evidence and at least one touched parameter."""
    record = _record(source="backfilled")
    tuning = record["tuning"]
    assert isinstance(tuning, dict)
    tuning["parameters_touched"] = []
    registry = _write_fixture(tmp_path, records=[record])
    with pytest.raises(ValueError, match="needs parameters_touched evidence"):
        validate_registry(registry, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("planner_not_mapping", "planner must be a mapping"),
        ("missing_key", "key must be a non-empty string"),
        ("missing_algo", "algo must be a non-empty string"),
        ("blank_config_path", "algo_config_path must be null or a non-empty string"),
        ("missing_tuning_field", "tuning is missing explicit fields"),
        ("extra_tuning_field", "tuning has unsupported fields"),
        ("invalid_source", "tuning.source must be one of"),
        ("invalid_parameter_list", "parameters_touched must be a list of strings"),
    ],
)
def test_registry_fails_closed_on_malformed_record(tmp_path: Path, case: str, message: str) -> None:
    """Malformed historical records fail at their exact contract boundary."""
    record = _record()
    planner = record["planner"]
    tuning = record["tuning"]
    assert isinstance(planner, dict)
    assert isinstance(tuning, dict)
    if case == "planner_not_mapping":
        record["planner"] = []
    elif case == "missing_key":
        planner["key"] = ""
    elif case == "missing_algo":
        planner["algo"] = ""
    elif case == "blank_config_path":
        planner["algo_config_path"] = ""
    elif case == "missing_tuning_field":
        tuning.pop("budget_runs")
    elif case == "extra_tuning_field":
        tuning["unsupported"] = None
    elif case == "invalid_source":
        tuning["source"] = "declared"
    elif case == "invalid_parameter_list":
        tuning["parameters_touched"] = "speed"
    registry = _write_fixture(tmp_path, records=[record])
    with pytest.raises(ValueError, match=message):
        validate_registry(registry, repo_root=tmp_path)


def test_cli_reports_success_and_failure(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The production CLI returns a machine-checkable status for both paths."""
    registry = _write_fixture(tmp_path, records=[_record()])
    with caplog.at_level("INFO"):
        assert main(["--registry", str(registry), "--repo-root", str(tmp_path)]) == 0
    assert '"status": "ok"' in caplog.text

    caplog.clear()
    assert main(["--registry", str(tmp_path / "missing.yaml")]) == 1
    assert '"status": "failed"' in caplog.text
