"""Search-to-trace eligibility bridge fixtures (issue #9304).

A successful search must not end with an unanalysable result: every evaluation
gets an explicit analysis-eligibility verdict, excluded attempts carry reasons,
and the manifest preserves proposal -> effective-hash -> certificate ->
episode -> trace identity end to end.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.adversarial import search
from robot_sf.adversarial.attribution import FailureAttribution, attribution_from_episode_record
from robot_sf.adversarial.bundle import compute_effective_scenario_hash
from robot_sf.adversarial.certification import failed_status, passed_status
from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    SearchConfig,
)
from robot_sf.adversarial.eligibility import analysis_eligibility

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "adversarial"))
from test_adversarial_search import (
    _candidate,
    _config,
    _SequenceSampler,
)


def _evaluation(
    *,
    certified: bool = True,
    trace: bool = True,
    execution_mode: str | None = "native",
    scored: bool = True,
    hashed: bool = True,
    error: str | None = None,
) -> CandidateEvaluation:
    """Build one injected evaluation with controlled eligibility inputs."""
    details: dict[str, Any] = {}
    if execution_mode is not None:
        details["execution_mode"] = execution_mode
    return CandidateEvaluation(
        candidate=_candidate(7),
        certification_status=passed_status() if certified else failed_status("test block"),
        objective_value=0.5 if scored else None,
        failure_attribution=FailureAttribution(
            status="attributed",
            primary_failure=None,
            reasons=[],
            details=details,
        ),
        episode_record_path=Path("episode_records.jsonl") if trace else None,
        trajectory_csv_path=Path("trajectory.csv") if trace else None,
        scenario_yaml_path=Path("scenario.yaml"),
        bundle_path=Path("candidate_0000"),
        error=error,
        effective_scenario_hash="abc123" if hashed else None,
    )


def test_native_scored_evaluation_is_eligible() -> None:
    """A certified, traced, native, scored, hashed evaluation must pass the gate."""
    receipt = analysis_eligibility(_evaluation())
    assert receipt.eligible
    assert receipt.reason_codes == []
    assert receipt.trace_present and receipt.certificate_ok
    assert receipt.execution_mode == "native" and receipt.objective_scored


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"certified": False}, "certificate_not_allowed"),
        ({"trace": False}, "trace_missing"),
        ({"execution_mode": "degraded"}, "execution_mode_not_native"),
        ({"execution_mode": "fallback"}, "execution_mode_not_native"),
        ({"execution_mode": "failed"}, "execution_mode_not_native"),
        ({"execution_mode": "unavailable"}, "execution_mode_not_native"),
        ({"execution_mode": None}, "execution_mode_unknown"),
        ({"scored": False}, "objective_unscored"),
        ({"hashed": False}, "effective_hash_unbound"),
        ({"error": "boom"}, "evaluation_errored"),
    ],
)
def test_each_exclusion_names_its_reason(kwargs: dict[str, Any], reason: str) -> None:
    """Every non-analysable evaluation must carry its stable reason (issue #9304)."""
    receipt = analysis_eligibility(_evaluation(**kwargs))
    assert not receipt.eligible
    assert reason in receipt.reason_codes


def test_search_loop_records_eligibility_and_hash_in_manifest(tmp_path: Path) -> None:
    """Proposal -> hash -> certificate -> episode -> trace identity must round-trip."""
    config = _config(tmp_path)

    def evaluator(
        _config: SearchConfig,
        candidate: CandidateSpec,
        scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        record: dict[str, Any] = {
            "episode_id": f"episode-{candidate.scenario_seed}",
            "seed": candidate.scenario_seed,
            "status": "success",
            "steps": 3,
            "termination_reason": "success",
            "outcome": {"route_complete": True, "collision": False, "timeout": False},
            "metrics": {"snqi": 0.8, "success": 1.0},
        }
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        attribution = attribution_from_episode_record(record)
        attribution = replace(
            attribution, details={**attribution.details, "execution_mode": "native"}
        )
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status(),
            objective_value=None,
            failure_attribution=attribution,
            episode_record_path=episode_path,
            trajectory_csv_path=None,
            scenario_yaml_path=scenario_yaml_path,
            bundle_path=candidate_dir,
        )

    result = search.run_adversarial_search(
        config,
        evaluator=evaluator,
        certifier=lambda _candidate, _path, _required: passed_status("test certifier"),
        sampler=_SequenceSampler([_candidate(7), _candidate(7)]),
    )

    assert result.num_candidates == 2
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["num_analysis_eligible"] == 2
    for row in manifest["candidates"]:
        assert row["analysis_eligibility"]["eligible"]
        assert row["analysis_eligibility"]["reason_codes"] == []
        bundle_dir = Path(row["bundle_path"])
        scenario = yaml.safe_load((bundle_dir / "scenario.yaml").read_text(encoding="utf-8"))[
            "scenarios"
        ][0]
        route = yaml.safe_load((bundle_dir / "route_overrides.yaml").read_text(encoding="utf-8"))
        assert row["effective_scenario_hash"] == compute_effective_scenario_hash(scenario, route)


def test_search_loop_marks_excluded_attempts_with_reasons(tmp_path: Path) -> None:
    """Invalid and failed attempts must be retained with reasons, never vanish."""
    config = _config(tmp_path)

    def failing_evaluator(
        _config: SearchConfig,
        candidate: CandidateSpec,
        scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        raise RuntimeError("synthetic evaluator failure")

    result = search.run_adversarial_search(
        config,
        evaluator=failing_evaluator,
        certifier=lambda _candidate, _path, _required: passed_status("test certifier"),
        sampler=_SequenceSampler([_candidate(7), _candidate(999)]),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["num_analysis_eligible"] == 0
    assert manifest["summary"]["num_failed_evaluations"] == 1
    assert manifest["summary"]["num_invalid_candidates"] == 1
    by_kind = {row["failure_attribution"]["primary_failure"]: row for row in manifest["candidates"]}
    errored = [
        row for row in manifest["candidates"] if row["error"] and "synthetic" in row["error"]
    ]
    assert len(errored) == 1
    failed_row = errored[0]
    assert "evaluation_errored" in failed_row["analysis_eligibility"]["reason_codes"]
    assert "objective_unscored" in failed_row["analysis_eligibility"]["reason_codes"]
    invalid_row = by_kind["invalid_candidate"]
    assert "certificate_not_allowed" in invalid_row["analysis_eligibility"]["reason_codes"]
    assert "trace_missing" in invalid_row["analysis_eligibility"]["reason_codes"]


def test_single_candidate_path_binds_effective_hash(tmp_path: Path) -> None:
    """The production single-candidate pipeline must bind the hash too (issue #9304)."""
    config = _config(tmp_path)

    def evaluator(
        _config: SearchConfig,
        candidate: CandidateSpec,
        scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text('{"episode_id": "e1"}\n', encoding="utf-8")
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status(),
            objective_value=None,
            failure_attribution=FailureAttribution(
                status="attributed",
                primary_failure=None,
                reasons=[],
                details={"execution_mode": "native"},
            ),
            episode_record_path=episode_path,
            trajectory_csv_path=None,
            scenario_yaml_path=scenario_yaml_path,
            bundle_path=candidate_dir,
        )

    run_one = search.production_candidate_evaluator(
        evaluator=evaluator,
        certifier=lambda _candidate, _path, _required: passed_status("test certifier"),
    )
    evaluation = run_one(config, _candidate(7), 0)

    assert evaluation.objective_value is not None
    assert evaluation.effective_scenario_hash is not None
    receipt = analysis_eligibility(evaluation)
    assert receipt.eligible
