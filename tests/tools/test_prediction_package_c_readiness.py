"""Tests for issue #3080 Package C prediction readiness preflight."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.tools.prediction_package_c_readiness import (
    ARMS,
    DEFAULT_CLOSED_LOOP_OUTPUT_ROOT,
    DEFAULT_OBSERVATION_REPLAY_OUTPUT_ROOT,
    REPO_ROOT,
    REQUIRED_CODE,
    REQUIRED_CONFIGS,
    assess_package_c_readiness,
    render_markdown,
    write_report_outputs,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_wired_repo(root: Path, *, with_coupling_store: bool = False) -> Path | None:
    """Materialize required Package C inputs under ``root``."""
    for rel in REQUIRED_CONFIGS + REQUIRED_CODE:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub\n", encoding="utf-8")

    (root / REQUIRED_CONFIGS[0]).write_text(
        "seeds:\n - 111\n - 2868\noutput:\n evidence_dir: docs/context/evidence/issue_2915\n",
        encoding="utf-8",
    )
    (root / REQUIRED_CONFIGS[1]).write_text(
        "fixture:\n seed: 111\n scenario_id: issue_2756_occluded_emergence\n",
        encoding="utf-8",
    )
    baseline_ids = [arm.baseline_id for arm in ARMS if arm.baseline_id is not None]
    (root / "robot_sf/benchmark/pedestrian_forecast.py").write_text(
        "\n".join(f"def {baseline_id}():\n    return None\n" for baseline_id in baseline_ids),
        encoding="utf-8",
    )

    if with_coupling_store:
        store = root / "result_store"
        store.mkdir(parents=True, exist_ok=True)
        (store / "summary.json").write_text("{}", encoding="utf-8")
        return store
    return None


def _write_coupling_report(root: Path, *, seed: int = 111, omit_metric: str | None = None) -> Path:
    """Write a minimal valid #2916 forecast-risk coupling report."""
    metrics = {
        "collision": False,
        "near_miss": False,
        "safety_events": 0,
        "progress_m": 1.0,
        "stop_yield_timing_steps": 0,
        "false_positive_stops": 0,
        "runtime_s": 0.01,
        "snqi": 0.5,
    }
    if omit_metric is not None:
        metrics.pop(omit_metric)
    rows = [
        ("no_forecast", "none"),
        ("cv_risk", "constant_velocity"),
        ("semantic_risk", "semantic_cv"),
        ("interaction_risk", "interaction_aware_cv"),
    ]
    report = {
        "issue": 2916,
        "claim_boundary": "diagnostic_only",
        "paper_grade": False,
        "config": {
            "fixture": {
                "seed": seed,
                "scenario_id": "issue_2756_occluded_emergence",
            }
        },
        "rows": [
            {
                "row": row,
                "risk_source": risk_source,
                "seed": seed,
                "scenario_id": "issue_2756_occluded_emergence",
                "classification": "ok",
                "metrics": dict(metrics),
            }
            for row, risk_source in rows
        ],
        "verdict": {"decision": "revise"},
    }
    path = root / "forecast_risk_coupling_gate_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_all_arms_blocked_when_inputs_present_but_no_coupling_artifact(tmp_path: Path) -> None:
    """Inputs alone fail closed until #2916 durable coupling artifacts are supplied."""
    _write_wired_repo(tmp_path)

    report = assess_package_c_readiness(tmp_path)

    assert report["overall_status"] == "blocked"
    assert [arm["status"] for arm in report["arms"]] == ["blocked"] * len(ARMS)
    assert report["seed_plan"] == [111, 2868]
    assert report["output_roots"] == [
        "docs/context/evidence/issue_2915",
        DEFAULT_OBSERVATION_REPLAY_OUTPUT_ROOT,
        DEFAULT_CLOSED_LOOP_OUTPUT_ROOT,
    ]
    assert all("#2916" in arm["blockers"][0] for arm in report["arms"])


def test_all_arms_ready_when_legacy_coupling_store_present(tmp_path: Path) -> None:
    """Legacy store marker remains compatible with the prior readiness contract."""
    store = _write_wired_repo(tmp_path, with_coupling_store=True)

    report = assess_package_c_readiness(tmp_path, coupling_result_store=store)

    assert report["overall_status"] == "ready"
    assert [arm["status"] for arm in report["arms"]] == ["ready"] * len(ARMS)
    assert report["coupling_result_store_available"] is True
    assert report["coupling_report_available"] is True
    assert all(arm["blockers"] == [] for arm in report["arms"])


def test_valid_coupling_report_clears_blocker(tmp_path: Path) -> None:
    """A validated #2916 report can clear the Package C assembly blocker."""
    _write_wired_repo(tmp_path)
    report_path = _write_coupling_report(tmp_path)

    report = assess_package_c_readiness(tmp_path, coupling_report=report_path)

    assert report["overall_status"] == "ready"
    assert report["coupling_report_available"] is True
    assert report["coupling_report_path"] == str(report_path)
    assert all(arm["blockers"] == [] for arm in report["arms"])


def test_coupling_report_missing_primary_metric_blocks(tmp_path: Path) -> None:
    """Malformed #2916 reports stay blocked and name the contract gap."""
    _write_wired_repo(tmp_path)
    report_path = _write_coupling_report(tmp_path, omit_metric="snqi")

    report = assess_package_c_readiness(tmp_path, coupling_report=report_path)

    assert report["overall_status"] == "blocked"
    assert report["coupling_report_available"] is False
    assert any("metrics.snqi missing" in blocker for blocker in report["coupling_report_blockers"])
    assert all(arm["status"] == "blocked" for arm in report["arms"])


def test_coupling_report_seed_outside_package_seed_plan_blocks(tmp_path: Path) -> None:
    """The supplied #2916 report must match the Package C same-seed plan."""
    _write_wired_repo(tmp_path)
    report_path = _write_coupling_report(tmp_path, seed=999)

    report = assess_package_c_readiness(tmp_path, coupling_report=report_path)

    assert report["overall_status"] == "blocked"
    assert any(
        "not in Package C seed plan" in blocker for blocker in report["coupling_report_blockers"]
    )


def test_missing_config_marks_arms_missing(tmp_path: Path) -> None:
    """A removed shared config fails closed as missing for every arm."""
    _write_wired_repo(tmp_path)
    (tmp_path / REQUIRED_CONFIGS[1]).unlink()

    report = assess_package_c_readiness(tmp_path)

    assert report["overall_status"] == "missing"
    assert all(arm["status"] == "missing" for arm in report["arms"])
    assert all(REQUIRED_CONFIGS[1] in arm["missing_inputs"] for arm in report["arms"])


def test_missing_seed_contract_marks_arms_missing(tmp_path: Path) -> None:
    """A seedless open-loop config cannot satisfy Package C same-seed planning."""
    _write_wired_repo(tmp_path, with_coupling_store=True)
    (tmp_path / REQUIRED_CONFIGS[0]).write_text(
        "output:\n evidence_dir: docs/context/evidence/issue_2915\n",
        encoding="utf-8",
    )

    report = assess_package_c_readiness(
        tmp_path,
        coupling_result_store=tmp_path / "result_store",
    )

    assert report["overall_status"] == "missing"
    assert all(arm["status"] == "missing" for arm in report["arms"])
    assert all(f"{REQUIRED_CONFIGS[0]}::seeds" in arm["missing_inputs"] for arm in report["arms"])


def test_mismatched_coupling_seed_marks_arms_missing(tmp_path: Path) -> None:
    """The closed-loop seed must be part of the open-loop same-seed plan."""
    _write_wired_repo(tmp_path, with_coupling_store=True)
    (tmp_path / REQUIRED_CONFIGS[1]).write_text(
        "fixture:\n seed: 999\n scenario_id: issue_2756_occluded_emergence\n",
        encoding="utf-8",
    )

    report = assess_package_c_readiness(
        tmp_path,
        coupling_result_store=tmp_path / "result_store",
    )

    assert report["overall_status"] == "missing"
    assert all(arm["status"] == "missing" for arm in report["arms"])
    assert all("fixture.seed=999" in arm["missing_inputs"][0] for arm in report["arms"])


def test_missing_baseline_only_marks_that_arm_missing(tmp_path: Path) -> None:
    """An absent deterministic baseline marks only its own arm missing."""
    _write_wired_repo(tmp_path)
    kept = [
        arm.baseline_id
        for arm in ARMS
        if arm.baseline_id is not None and arm.baseline_id != "semantic_cv_baseline"
    ]
    (tmp_path / "robot_sf/benchmark/pedestrian_forecast.py").write_text(
        "\n".join(f"def {baseline_id}():\n    return None\n" for baseline_id in kept),
        encoding="utf-8",
    )

    report = assess_package_c_readiness(tmp_path)
    by_arm = {arm["arm"]: arm for arm in report["arms"]}

    assert by_arm["semantic_cv"]["status"] == "missing"
    assert any("semantic_cv_baseline" in item for item in by_arm["semantic_cv"]["missing_inputs"])
    assert by_arm["no_forecast"]["status"] == "blocked"
    assert by_arm["cv"]["status"] == "blocked"
    assert by_arm["interaction_aware"]["status"] == "blocked"
    assert report["overall_status"] == "missing"


def test_markdown_renders_status_report_and_blockers(tmp_path: Path) -> None:
    """Markdown view surfaces overall status and #2916 blocker."""
    _write_wired_repo(tmp_path)
    report = assess_package_c_readiness(tmp_path)

    markdown = render_markdown(report)

    assert "Prediction Package C Readiness Preflight" in markdown
    assert "`blocked`" in markdown
    assert "Coupling report available" in markdown
    assert "#2916" in markdown


def test_write_report_outputs_persists_json_and_markdown(tmp_path: Path) -> None:
    """Durable rerun outputs can be written directly from a validated readiness report."""
    _write_wired_repo(tmp_path)
    report_path = _write_coupling_report(tmp_path)
    report = assess_package_c_readiness(tmp_path, coupling_report=report_path)
    output_dir = tmp_path / "docs/context/evidence/issue_3080_package_c_readiness"
    output_json = output_dir / "package_c_readiness_report.json"
    output_markdown = output_dir / "README.md"

    write_report_outputs(
        report,
        output_json=output_json,
        output_markdown=output_markdown,
    )

    assert json.loads(output_json.read_text(encoding="utf-8")) == report
    markdown = output_markdown.read_text(encoding="utf-8")
    assert markdown.endswith("\n")
    assert "Prediction Package C Readiness Preflight" in markdown
    assert "`ready`" in markdown
    assert str(report_path) in markdown


def test_real_repo_preflight_is_wired_and_fails_closed() -> None:
    """Real repo inputs resolve; default remains blocked without supplied artifact."""
    report = assess_package_c_readiness()

    assert report["overall_status"] == "blocked"
    assert all(arm["status"] == "blocked" for arm in report["arms"])
    assert all(arm["missing_inputs"] == [] for arm in report["arms"])
    assert report["seed_plan"] == [111, 2868]
    assert report["output_roots"] == [
        "docs/context/evidence/issue_2915_forecast_baselines_2026-06-20",
        DEFAULT_OBSERVATION_REPLAY_OUTPUT_ROOT,
        DEFAULT_CLOSED_LOOP_OUTPUT_ROOT,
    ]


def test_real_repo_coupling_report_clears_package_c_blocker() -> None:
    """The committed #2916 report should satisfy the current Package C contract."""
    report = assess_package_c_readiness(
        coupling_report=(
            REPO_ROOT
            / "docs/context/evidence/issue_2916_forecast_risk_coupling_2026-06-23"
            / "forecast_risk_coupling_gate_report.json"
        )
    )

    assert report["overall_status"] == "ready"
    assert report["coupling_report_available"] is True
    assert all(arm["status"] == "ready" for arm in report["arms"])
