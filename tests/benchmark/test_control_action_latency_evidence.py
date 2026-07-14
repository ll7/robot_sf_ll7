"""Tests for the control-action-latency sweep evidence promotion (issue #5034)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.benchmark.control_action_latency_evidence import (
    CLAIM_BOUNDARY,
    FIXED_SCOPE_COVERAGE_CONTRACT,
    FIXED_SCOPE_PLAN_SCHEMA_VERSION,
    PROMOTION_SCHEMA_VERSION,
    REQUIRED_RESULT_METRICS,
    LatencyEvidenceError,
    aggregate_latency_metrics,
    build_latency_evidence,
    classify_latency_row,
    extract_latency_cells,
    load_fixed_scope_plan,
    load_latency_rows,
    promote_latency_evidence,
    validate_fixed_scope_latency_coverage,
    write_latency_evidence,
)
from robot_sf.benchmark.control_action_latency_preflight import AXIS_KEY

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = REPO_ROOT / "configs/research/fidelity_sensitivity_v1.yaml"
PROMOTER = REPO_ROOT / "scripts/benchmark/promote_control_action_latency_evidence.py"


def _latency_row(
    *,
    planner: str = "baseline_social_force",
    step: int,
    seed: int = 111,
    min_clearance: float = 0.4,
    baseline_variant: bool = False,
    identity: dict[str, Any] | None = None,
    execution_mode: str | None = None,
    availability_status: str | None = None,
) -> dict[str, Any]:
    """Build one representative control_action_latency episode row (runner shape).

    Rows default to a native, successful, collision-free outcome; classification
    markers (``execution_mode`` / ``availability_status``) are added only when
    passed so the exclusion path can be exercised.
    """
    identity = identity or {}
    ms = step * 100.0
    source_key = str(identity.get("variant_source_key") or f"step_{step}")
    return {
        "axis": AXIS_KEY,
        "variant": f"{AXIS_KEY}__{source_key}",
        "variant_source_key": source_key,
        "baseline_variant": baseline_variant,
        "runtime_binding": "sim_config.action_latency_steps",
        "action_latency": {
            "configured_steps": step,
            "configured_ms": None,
            "effective_steps": step,
            "effective_ms": ms,
        },
        "planner": planner,
        "scenario_id": str(identity.get("scenario_id") or f"scenario_{seed}"),
        "seed": seed,
        "success": True,
        "collision": False,
        "metrics": {
            "success_rate": 1.0,
            "collision_rate": 0.0,
            "min_clearance": min_clearance,
        },
        **({"planner_group": identity["planner_group"]} if identity.get("planner_group") else {}),
        **({"execution_mode": execution_mode} if execution_mode else {}),
        **({"availability_status": availability_status} if availability_status else {}),
    }


def _full_native_row_set() -> list[dict[str, Any]]:
    """Return a minimal native row set covering latency steps 0, 1, 3."""
    rows: list[dict[str, Any]] = []
    for step in (0, 1, 3):
        for seed in (111, 112):
            rows.append(
                _latency_row(
                    step=step,
                    seed=seed,
                    planner="baseline_social_force",
                    baseline_variant=(step == 0),
                    min_clearance=0.5 - 0.1 * step,
                )
            )
    return rows


def _load_real_config() -> dict[str, Any]:
    """Load the shipped fidelity-sensitivity config (post-#5026 carries the axis)."""
    return yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8")) or {}


def _fixed_scope_plan() -> dict[str, Any]:
    """Build a small ready plan with the real latency variant names."""
    cells = []
    for planner_group in ("orca", "default_social_force", "hybrid_rule_v0_minimal"):
        for variant in ("zero_step_nominal", "one_step_100ms", "three_step_300ms"):
            for seed in (111, 112):
                cells.append(
                    {
                        "axis": AXIS_KEY,
                        "planner_group": planner_group,
                        "variant": variant,
                        "seed": seed,
                    }
                )
    return {
        "schema_version": FIXED_SCOPE_PLAN_SCHEMA_VERSION,
        "preflight_ready": True,
        "scenario_resolution": {
            "status": "ready",
            "scenario_ids": ["scenario_a", "scenario_b"],
        },
        "run_cells": cells,
    }


def _fixed_scope_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one native row for every expected plan cell/scenario pair."""
    steps = {"zero_step_nominal": 0, "one_step_100ms": 1, "three_step_300ms": 3}
    planners = {
        "orca": "orca",
        "default_social_force": "baseline_social_force",
        "hybrid_rule_v0_minimal": "hybrid_rule_v0_minimal",
    }
    scenarios = plan["scenario_resolution"]["scenario_ids"]
    rows: list[dict[str, Any]] = []
    for cell in plan["run_cells"]:
        for scenario_id in scenarios:
            rows.append(
                _latency_row(
                    planner=planners[cell["planner_group"]],
                    step=steps[cell["variant"]],
                    seed=cell["seed"],
                    identity={
                        "planner_group": cell["planner_group"],
                        "scenario_id": scenario_id,
                        "variant_source_key": cell["variant"],
                    },
                )
            )
    return rows


# --- classify_latency_row -------------------------------------------------


def test_native_row_with_metadata_classifies_as_result() -> None:
    """A native row carrying action_latency metadata is a result cell."""
    cell = classify_latency_row(_latency_row(step=1))
    assert cell.classification == "result"
    assert cell.exclusion_reason is None
    assert cell.latency_step == 1
    assert cell.latency_ms == 100.0
    assert cell.success_rate == 1.0
    assert cell.collision_rate == 0.0
    assert cell.min_clearance == 0.4


def test_row_missing_action_latency_metadata_is_exclusion() -> None:
    """A latency row without action_latency metadata cannot confirm the contract."""
    row = _latency_row(step=1)
    row["action_latency"] = None
    cell = classify_latency_row(row)
    assert cell.classification == "exclusion"
    assert cell.exclusion_reason == "missing_action_latency_metadata"
    assert cell.latency_step is None


def test_fallback_execution_mode_is_exclusion() -> None:
    """A fallback execution-mode row is excluded per issue #691 policy."""
    row = _latency_row(step=1, execution_mode="fallback")
    cell = classify_latency_row(row)
    assert cell.classification == "exclusion"
    assert "non_native_execution_mode:fallback" in cell.exclusion_reason


def test_degraded_availability_is_exclusion() -> None:
    """A degraded/unavailable row is excluded per issue #691 policy."""
    row = _latency_row(step=1, availability_status="degraded")
    cell = classify_latency_row(row)
    assert cell.classification == "exclusion"
    assert "unavailable:degraded" in cell.exclusion_reason


def test_missing_required_result_metric_is_exclusion() -> None:
    """A row cannot fabricate an absent outcome metric as zero-valued evidence."""
    row = _latency_row(step=1)
    row["metrics"].pop("success_rate")
    cell = classify_latency_row(row)
    assert cell.classification == "exclusion"
    assert "missing_or_invalid_metric:success_rate" in cell.exclusion_reason
    assert cell.success_rate is None


# --- extract_latency_cells ------------------------------------------------


def test_extract_isolates_latency_axis_rows_only() -> None:
    """Rows on other fidelity axes never become latency cells."""
    rows = [
        *_full_native_row_set(),
        {
            "axis": "clearance_radius",
            "variant": "clearance_radius__radius_0_30",
            "planner": "baseline_social_force",
            "seed": 111,
            "scenario_id": "scenario_111",
            "metrics": {"success_rate": 1.0, "collision_rate": 0.0, "min_clearance": 0.3},
        },
    ]
    cells = extract_latency_cells(rows)
    assert len(cells) == 6
    assert all(cell.planner == "baseline_social_force" for cell in cells)


# --- aggregate_latency_metrics --------------------------------------------


def test_aggregate_metrics_group_by_planner_and_step() -> None:
    """Aggregates average success/collision/min_clearance per (planner, step)."""
    cells = extract_latency_cells(_full_native_row_set())
    aggregates = aggregate_latency_metrics(cells)
    by_key = {(row["planner"], row["action_latency_steps"]): row for row in aggregates}
    assert sorted(by_key) == [
        ("baseline_social_force", 0),
        ("baseline_social_force", 1),
        ("baseline_social_force", 3),
    ]
    for step in (0, 1, 3):
        row = by_key[("baseline_social_force", step)]
        assert row["cell_count"] == 2
        assert row["success_rate"] == 1.0
        assert row["collision_rate"] == 0.0
        assert row["min_clearance"] == pytest.approx(0.5 - 0.1 * step)
        assert set(REQUIRED_RESULT_METRICS).issubset(row)


def test_aggregate_excludes_exclusion_cells() -> None:
    """Exclusion cells (fallback) never enter the aggregate metrics."""
    rows = _full_native_row_set()
    rows.append(_latency_row(step=1, seed=999, execution_mode="fallback"))
    cells = extract_latency_cells(rows)
    aggregates = aggregate_latency_metrics(cells)
    step_one = next(row for row in aggregates if row["action_latency_steps"] == 1)
    # The fallback seed-999 row must NOT inflate the count.
    assert step_one["cell_count"] == 2


# --- build_latency_evidence (fail-closed gates) ---------------------------


def test_build_evidence_succeeds_on_complete_native_coverage() -> None:
    """A complete native 0/1/3 row set builds a promotable packet."""
    packet = build_latency_evidence(
        _full_native_row_set(),
        config=_load_real_config(),
        config_path="configs/research/fidelity_sensitivity_v1.yaml",
        git_head="abc123",
        date="2026-07-10",
        raw_rows_path="output/fidelity_latency_raw/episode_rows.jsonl",
    )
    assert packet["schema_version"] == PROMOTION_SCHEMA_VERSION
    assert packet["preflight_decision"] == "ready"
    assert packet["latency_coverage"]["coverage_complete"] is True
    assert packet["latency_coverage"]["missing_latency_steps"] == []
    assert packet["scope"]["result_row_count"] == 6
    assert packet["scope"]["excluded_row_count"] == 0
    assert CLAIM_BOUNDARY in packet["claim_boundary"]
    assert json.loads(json.dumps(packet)) == packet  # JSON-serializable


def test_build_evidence_fails_closed_when_required_step_missing() -> None:
    """Dropping all 3-step rows fails closed (partial run cannot be promoted)."""
    rows = [row for row in _full_native_row_set() if row["action_latency"]["effective_steps"] != 3]
    try:
        build_latency_evidence(
            rows,
            config=_load_real_config(),
            config_path="configs/research/fidelity_sensitivity_v1.yaml",
            git_head="abc123",
            date="2026-07-10",
            raw_rows_path="output/fidelity_latency_raw/episode_rows.jsonl",
        )
    except LatencyEvidenceError as exc:
        assert "missing" in str(exc)
        assert "[3]" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected LatencyEvidenceError for incomplete coverage")


def test_build_evidence_fails_closed_when_no_latency_rows() -> None:
    """Rows with no latency axis cannot be promoted as the latency sweep."""
    try:
        build_latency_evidence(
            [{"axis": "clearance_radius", "planner": "p", "metrics": {}}],
            config=_load_real_config(),
            config_path="configs/research/fidelity_sensitivity_v1.yaml",
            git_head="abc123",
            date="2026-07-10",
            raw_rows_path="output/x/episode_rows.jsonl",
        )
    except LatencyEvidenceError as exc:
        assert "no 'control_action_latency' axis rows" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected LatencyEvidenceError for missing latency rows")


def test_build_evidence_fails_closed_when_all_step3_rows_are_excluded() -> None:
    """If every 3-step row is fallback/degraded, coverage is incomplete → fail closed."""
    rows = _full_native_row_set()
    for row in rows:
        if row["action_latency"]["effective_steps"] == 3:
            row["execution_mode"] = "degraded"
    try:
        build_latency_evidence(
            rows,
            config=_load_real_config(),
            config_path="configs/research/fidelity_sensitivity_v1.yaml",
            git_head="abc123",
            date="2026-07-10",
            raw_rows_path="output/x/episode_rows.jsonl",
        )
    except LatencyEvidenceError as exc:
        assert "[3]" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected LatencyEvidenceError when all 3-step rows excluded")


def test_build_evidence_fails_closed_when_required_metrics_are_missing() -> None:
    """Rows without outcome metrics cannot provide a required latency cell."""
    rows = _full_native_row_set()
    for row in rows:
        if row["action_latency"]["effective_steps"] == 3:
            row["metrics"].pop("collision_rate")
    with pytest.raises(LatencyEvidenceError, match=r"missing=\[3\]"):
        build_latency_evidence(
            rows,
            config=_load_real_config(),
            config_path="configs/research/fidelity_sensitivity_v1.yaml",
            git_head="abc123",
            date="2026-07-10",
            raw_rows_path="output/x/episode_rows.jsonl",
        )


def test_build_evidence_records_exclusions_without_promoting_them() -> None:
    """Fallback rows appear in exclusions but never in aggregate results."""
    rows = _full_native_row_set()
    rows.append(
        _latency_row(step=1, seed=999, execution_mode="fallback", availability_status="fallback")
    )
    packet = build_latency_evidence(
        rows,
        config=_load_real_config(),
        config_path="configs/research/fidelity_sensitivity_v1.yaml",
        git_head="abc123",
        date="2026-07-10",
        raw_rows_path="output/x/episode_rows.jsonl",
    )
    assert packet["scope"]["excluded_row_count"] == 1
    assert packet["exclusions"]["excluded_row_count"] == 1
    assert packet["exclusions"]["reason_counts"]["non_native_execution_mode:fallback"] == 1
    # Aggregate step-1 count stays at 2 native rows.
    step_one = next(row for row in packet["aggregate_metrics"] if row["action_latency_steps"] == 1)
    assert step_one["cell_count"] == 2


def test_fixed_scope_coverage_requires_every_planner_variant_seed_scenario_cell() -> None:
    """A serialized plan upgrades step coverage to exact fixed-scope coverage."""
    plan = _fixed_scope_plan()
    packet = build_latency_evidence(
        _fixed_scope_rows(plan),
        config=_load_real_config(),
        config_path="configs/research/fidelity_sensitivity_v1.yaml",
        git_head="abc123",
        date="2026-07-14",
        raw_rows_path="output/fidelity_latency_raw/episode_rows.jsonl",
        fixed_scope_plan=plan,
    )
    coverage = packet["fixed_scope_coverage"]
    assert coverage["contract"] == FIXED_SCOPE_COVERAGE_CONTRACT
    assert coverage["status"] == "verified"
    assert coverage["expected_row_count"] == 36
    assert coverage["observed_result_row_count"] == 36
    assert coverage["expected_latency_steps"] == [0, 1, 3]


def test_fixed_scope_coverage_fails_closed_on_missing_cell() -> None:
    """Removing one expected cell cannot produce a promotable fixed-scope packet."""
    plan = _fixed_scope_plan()
    rows = _fixed_scope_rows(plan)[:-1]
    with pytest.raises(LatencyEvidenceError, match="missing_expected=1"):
        build_latency_evidence(
            rows,
            config=_load_real_config(),
            config_path="configs/research/fidelity_sensitivity_v1.yaml",
            git_head="abc123",
            date="2026-07-14",
            raw_rows_path="output/fidelity_latency_raw/episode_rows.jsonl",
            fixed_scope_plan=plan,
        )


def test_fixed_scope_coverage_fails_closed_on_excluded_expected_cell() -> None:
    """Fallback/degraded output for an expected cell is not fixed-scope evidence."""
    plan = _fixed_scope_plan()
    rows = _fixed_scope_rows(plan)
    rows[-1]["execution_mode"] = "fallback"
    with pytest.raises(LatencyEvidenceError, match="non_native_or_malformed_expected=1"):
        build_latency_evidence(
            rows,
            config=_load_real_config(),
            config_path="configs/research/fidelity_sensitivity_v1.yaml",
            git_head="abc123",
            date="2026-07-14",
            raw_rows_path="output/fidelity_latency_raw/episode_rows.jsonl",
            fixed_scope_plan=plan,
        )


def test_fixed_scope_coverage_fails_closed_on_duplicate_cell() -> None:
    """Duplicate result rows cannot silently inflate a fixed-scope denominator."""
    plan = _fixed_scope_plan()
    rows = _fixed_scope_rows(plan)
    rows.append(dict(rows[0]))
    with pytest.raises(LatencyEvidenceError, match="duplicate_expected=1"):
        validate_fixed_scope_latency_coverage(
            rows,
            extract_latency_cells(rows),
            plan=plan,
            config=_load_real_config(),
        )


# --- write_latency_evidence + promote_latency_evidence --------------------


def test_load_rows_fails_closed_when_raw_rows_path_is_missing(tmp_path: Path) -> None:
    """A missing raw-row file becomes a promotion error, never a filesystem traceback."""
    missing_path = tmp_path / "missing_episode_rows.jsonl"
    with pytest.raises(LatencyEvidenceError, match="cannot be read"):
        load_latency_rows(missing_path)


def test_load_rows_fails_closed_when_raw_rows_path_is_not_a_file(tmp_path: Path) -> None:
    """An unreadable raw-row path becomes the same fail-closed promotion error."""
    with pytest.raises(LatencyEvidenceError, match="cannot be read"):
        load_latency_rows(tmp_path)


def test_load_rows_fails_closed_when_raw_rows_contain_invalid_json(tmp_path: Path) -> None:
    """Malformed JSONL reports a promotion error with its source line."""
    raw_path = tmp_path / "malformed_episode_rows.jsonl"
    raw_path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(LatencyEvidenceError, match="invalid JSON on line 1"):
        load_latency_rows(raw_path)


def test_cli_missing_raw_rows_emits_blocked_json_without_traceback(tmp_path: Path) -> None:
    """The check-only CLI reports a missing input as a compact fail-closed status."""
    result = subprocess.run(
        [
            sys.executable,
            str(PROMOTER),
            "--raw-rows",
            str(tmp_path / "missing_episode_rows.jsonl"),
            "--check-only",
            "--require-ready",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 1
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr
    status = json.loads(result.stdout)
    assert status["schema_version"] == PROMOTION_SCHEMA_VERSION
    assert status["status"] == "blocked"
    assert status["issue"] == 5034
    assert "cannot be read" in status["reason"]


def test_cli_strict_fixed_scope_check_reports_verified_coverage(tmp_path: Path) -> None:
    """The real promoter CLI can enforce the serialized fixed-scope contract."""
    plan = _fixed_scope_plan()
    raw_path = tmp_path / "episode_rows.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _fixed_scope_rows(plan)),
        encoding="utf-8",
    )
    plan_path = tmp_path / "fidelity_fixed_scope_run_plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROMOTER),
            "--raw-rows",
            str(raw_path),
            "--fixed-scope-plan",
            str(plan_path),
            "--require-fixed-scope",
            "--check-only",
            "--require-ready",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    assert status["status"] == "promotable"
    assert status["fixed_scope_coverage"]["status"] == "verified"
    assert status["fixed_scope_coverage"]["expected_row_count"] == 36


def test_fixed_scope_plan_loader_rejects_missing_file(tmp_path: Path) -> None:
    """Strict promotion reports an absent serialized plan as a bounded input error."""
    with pytest.raises(LatencyEvidenceError, match="cannot be read"):
        load_fixed_scope_plan(tmp_path / "missing_plan.json")


def test_write_latency_evidence_writes_bundle(tmp_path: Path) -> None:
    """The evidence bundle writes summary, CSV, README, and a checksum manifest."""
    packet = build_latency_evidence(
        _full_native_row_set(),
        config=_load_real_config(),
        config_path="configs/research/fidelity_sensitivity_v1.yaml",
        git_head="abc123",
        date="2026-07-10",
        raw_rows_path="output/x/episode_rows.jsonl",
    )
    written = write_latency_evidence(packet, tmp_path / "ev")
    names = {path.name for path in written}
    assert names == {"summary.json", "per_cell_metrics.csv", "README.md", "manifest.sha256"}
    summary = json.loads((tmp_path / "ev" / "summary.json").read_text(encoding="utf-8"))
    assert summary == packet
    with (tmp_path / "ev" / "per_cell_metrics.csv").open(encoding="utf-8") as handle:
        records = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    assert len(records) == 6
    assert {record["classification"] for record in records} == {"result"}
    readme = (tmp_path / "ev" / "README.md").read_text(encoding="utf-8")
    assert "Plain-language summary" in readme


def test_promote_end_to_end_from_jsonl(tmp_path: Path) -> None:
    """promote_latency_evidence loads JSONL rows and writes the full bundle."""
    rows = _full_native_row_set()
    raw_path = tmp_path / "episode_rows.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = promote_latency_evidence(
        raw_path,
        tmp_path / "ev",
        config=_load_real_config(),
        config_path="configs/research/fidelity_sensitivity_v1.yaml",
        git_head="abc123",
        date="2026-07-10",
    )
    assert result["status"] == "promoted"
    assert result["result_row_count"] == 6
    assert (tmp_path / "ev" / "summary.json").exists()


def test_promote_fails_closed_on_incomplete_jsonl(tmp_path: Path) -> None:
    """An incomplete JSONL (missing step 3) fails closed at promotion time."""
    rows = [row for row in _full_native_row_set() if row["action_latency"]["effective_steps"] != 3]
    raw_path = tmp_path / "episode_rows.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    try:
        promote_latency_evidence(
            raw_path,
            tmp_path / "ev",
            config=_load_real_config(),
            config_path="configs/research/fidelity_sensitivity_v1.yaml",
            git_head="abc123",
            date="2026-07-10",
        )
    except LatencyEvidenceError as exc:
        assert "[3]" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected LatencyEvidenceError")
