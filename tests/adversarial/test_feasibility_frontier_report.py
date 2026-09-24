"""Fixture tests for the issue #9654 frontier report contract."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import pytest

from robot_sf.adversarial.feasibility_frontier_report import (
    INPUT_SCHEMA_VERSION,
    FrontierReportError,
    build_frontier_report,
    render_frontier_markdown,
    write_frontier_report,
)

if TYPE_CHECKING:
    from pathlib import Path

_REVISION = "a" * 40
_CONFIG = "b" * 64


def _artifact(root: Path, name: str, *, role: str) -> dict[str, str]:
    path = root / "evidence" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"persisted fixture artifact: {name}\n".encode()
    path.write_bytes(content)
    return {
        "path": f"evidence/{name}",
        "sha256": hashlib.sha256(content).hexdigest(),
        "source_revision": _REVISION,
        "role": role,
        "schema_version": f"fixture-{role}.v1",
    }


def _episode(record_id: str, *, success: bool, collision: bool = False) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "evidence_status": "complete",
        "execution_mode": "normal",
        "eligible": True,
        "success": success,
        "collision": collision,
        "minimum_clearance": 0.2 if not collision else 0.0,
        "ped_force_q95": 0.7 if not success else 0.3,
    }


def _evaluation_set(
    root: Path, name: str, *, round_number: int, success_count: int
) -> dict[str, Any]:
    rows = [
        _episode(f"{name}-{round_number}-1", success=success_count >= 1),
        _episode(
            f"{name}-{round_number}-2", success=success_count >= 2, collision=round_number == 1
        ),
        {
            "record_id": f"{name}-{round_number}-fallback",
            "evidence_status": "complete",
            "execution_mode": "fallback",
            "eligible": False,
            "success": False,
            "collision": True,
            "minimum_clearance": 0.0,
            "ped_force_q95": 3.2,
        },
    ]
    return {
        "expected_episode_count": len(rows),
        "artifact": _artifact(
            root, f"round-{round_number}-{name}.jsonl", role=f"{name}-evaluation"
        ),
        "episodes": rows,
    }


def _candidate(
    *,
    candidate_id: str,
    evaluation_status: str,
    verdict: str,
    failure: bool | None,
    replay_status: str,
    disposition: str,
    case_id: str | None = None,
    replay_artifact: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "evaluation_status": evaluation_status,
        "admissibility_verdict": verdict,
        "target_failure_observed": failure,
        "replay_status": replay_status,
        "corpus_disposition": disposition,
        "case_id": case_id,
        "replay_artifact": replay_artifact,
    }


def _observation(
    root: Path,
    *,
    round_number: int,
    case_id: str,
    origin_round: int,
    origin_candidate_id: str | None,
    planner_status: str,
    verdict: str,
    replay_status: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "origin_round": origin_round,
        "origin_candidate_id": origin_candidate_id,
        "planner_status": planner_status,
        "admissibility_verdict": verdict,
        "replay_status": replay_status,
        "evidence_status": "complete" if planner_status != "unknown" else "unknown",
        "origin_search_artifact": _artifact(
            root,
            f"round-{origin_round or 'historical'}-search.json",
            role="falsification-search",
        ),
        "corpus_artifact": _artifact(root, f"round-{round_number}-corpus.json", role="corpus"),
        "replay_artifact": (
            _artifact(root, f"round-{round_number}-{case_id}-replay.json", role="replay")
            if replay_status == "verified"
            else None
        ),
    }


def _round(root: Path, round_number: int) -> dict[str, Any]:
    search_artifact = _artifact(
        root, f"round-{round_number}-search.json", role="falsification-search"
    )
    optimization = {
        "method": "random-search",
        "objective": (
            "priority: valid execution; avoid collisions; task completion; efficiency; comfort"
        ),
        "selection_rule": "lexicographic comparison in the declared priority order",
        "seeds": [1101, 2202],
        "budget": {
            "proposal_limit": 8,
            "proposals_completed": 3,
            "simulator_invocations": 12,
        },
        "artifact": _artifact(root, f"round-{round_number}-optimization.json", role="optimization"),
    }
    if round_number == 1:
        c1_replay = _artifact(root, "round-1-case-001-replay.json", role="replay")
        candidates = [
            _candidate(
                candidate_id="c1",
                evaluation_status="complete",
                verdict="empirically_feasible",
                failure=True,
                replay_status="verified",
                disposition="admitted",
                case_id="case-001",
                replay_artifact=c1_replay,
            ),
            _candidate(
                candidate_id="c-invalid",
                evaluation_status="invalid",
                verdict="structurally_invalid",
                failure=None,
                replay_status="not_attempted",
                disposition="rejected",
            ),
            _candidate(
                candidate_id="c-failed",
                evaluation_status="failed",
                verdict="admissible_feasibility_unknown",
                failure=None,
                replay_status="unknown",
                disposition="pending",
            ),
        ]
        observations = [
            _observation(
                root,
                round_number=1,
                case_id="case-001",
                origin_round=1,
                origin_candidate_id="c1",
                planner_status="unsolved",
                verdict="empirically_feasible",
                replay_status="verified",
            )
        ]
        success_count = 1
        stop_reason = "budget_exhausted"
    else:
        c2_replay = _artifact(root, "round-2-case-unknown-replay.json", role="replay")
        candidates = [
            _candidate(
                candidate_id="c2",
                evaluation_status="complete",
                verdict="admissible_feasibility_unknown",
                failure=True,
                replay_status="verified",
                disposition="admitted",
                case_id="case-unknown",
                replay_artifact=c2_replay,
            ),
            _candidate(
                candidate_id="c-impossible",
                evaluation_status="invalid",
                verdict="geometric_or_kinodynamic_impossibility",
                failure=None,
                replay_status="not_attempted",
                disposition="rejected",
            ),
            _candidate(
                candidate_id="c-degraded",
                evaluation_status="degraded",
                verdict="planner_specific_failure",
                failure=True,
                replay_status="verified",
                disposition="pending",
                replay_artifact=_artifact(root, "round-2-degraded-replay.json", role="replay"),
            ),
            _candidate(
                candidate_id="c-mismatch",
                evaluation_status="complete",
                verdict="empirically_feasible",
                failure=True,
                replay_status="mismatch",
                disposition="pending",
            ),
        ]
        observations = [
            _observation(
                root,
                round_number=2,
                case_id="case-001",
                origin_round=1,
                origin_candidate_id="c1",
                planner_status="solved",
                verdict="empirically_feasible",
                replay_status="verified",
            ),
            _observation(
                root,
                round_number=2,
                case_id="case-unknown",
                origin_round=2,
                origin_candidate_id="c2",
                planner_status="unknown",
                verdict="admissible_feasibility_unknown",
                replay_status="verified",
            ),
        ]
        success_count = 2
        stop_reason = "no_new_admissible_counterexample"
    return {
        "round_number": round_number,
        "source_revision": _REVISION,
        "planner": {
            "planner_id": "goal",
            "config_identity_sha256": _CONFIG if round_number == 1 else "c" * 64,
            "source_revision": _REVISION,
        },
        "optimization": optimization,
        "falsification": {
            "method": "tpe",
            "objective": "complete target-planner failure under the named failure predicate",
            "target_failure_predicate": "collision-or-timeout",
            "search_space_id": "crossing-gap-v1",
            "seeds": [1101, 2202],
            "budget": {
                "candidate_limit": 4,
                "candidates_completed": len(candidates),
                "simulator_invocations": 9,
            },
            "stop_reason": stop_reason,
            "artifact": search_artifact,
            "candidates": candidates,
        },
        "evaluation_sets": {
            name: _evaluation_set(
                root, name, round_number=round_number, success_count=success_count
            )
            for name in ("fixed", "regression", "held_out")
        },
        "case_observations": observations,
    }


def _evidence(root: Path) -> dict[str, Any]:
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "evidence_kind": "synthetic_fixture",
        "experiment_id": "fixture-two-round-loop",
        "source_revision": _REVISION,
        "simulator_identity": "robot-sf-fixture-simulator.v1",
        "scenario_space_id": "crossing-gap-v1",
        "rounds": [_round(root, 1), _round(root, 2)],
    }


def test_frontier_report_separates_valid_discoveries_unknowns_and_exclusions(
    tmp_path: Path,
) -> None:
    """Only complete, admitted, replayed failures with feasibility evidence count."""
    report = build_frontier_report(
        _evidence(tmp_path), evidence_root=tmp_path, input_sha256="d" * 64
    )

    first, second = report["rounds"]
    assert report["evidence_kind"] == "synthetic_fixture"
    assert first["evaluation_sets"]["held_out"]["success_rate"] == 0.5
    assert first["evaluation_sets"]["held_out"]["eligible_episode_count"] == 2
    assert first["evaluation_sets"]["held_out"]["execution_mode_counts"]["fallback"] == 1
    assert second["evaluation_sets"]["held_out"]["success_rate"] == 1.0
    assert second["evaluation_sets"]["held_out"]["minimum_clearance_min"] == 0.2
    assert first["falsification"]["verified_counterexample_case_ids"] == ["case-001"]
    assert second["falsification"]["verified_counterexample_case_ids"] == []
    assert second["falsification"]["admitted_unknown_feasibility_case_ids"] == ["case-unknown"]
    assert second["case_frontier"]["verified_counterexample_status"]["solved"] == 1
    assert second["falsification"]["candidate_status_counts"]["degraded"] == 1
    assert second["falsification"]["candidate_status_counts"]["invalid"] == 1
    assert (
        second["falsification"]["admissibility_verdict_counts"][
            "geometric_or_kinodynamic_impossibility"
        ]
        == 1
    )
    assert (
        next(
            item
            for item in second["falsification"]["candidate_records"]
            if item["candidate_id"] == "c-mismatch"
        )["replay_status"]
        == "mismatch"
    )
    assert second["case_frontier"]["admissibility_partition_counts"]["feasibility_unknown"] == 1
    assert second["falsification"]["no_verified_counterexample_statement"].endswith(
        "This does not establish that no counterexample exists."
    )
    markdown = render_frontier_markdown(report)
    assert "9 simulator invocations" in markdown
    assert "case-001" in markdown
    assert "case-unknown" in markdown
    assert "replay artifact" in markdown.lower()
    assert "Search candidate accounting" in markdown


def test_frontier_report_rejects_missing_provenance_and_artifact_digest_mismatch(
    tmp_path: Path,
) -> None:
    """Partial identities and changed evidence files cannot enter a report."""
    payload = _evidence(tmp_path)
    payload["rounds"][0]["planner"]["source_revision"] = "abc123"
    with pytest.raises(FrontierReportError, match="full hexadecimal"):
        build_frontier_report(payload, evidence_root=tmp_path)

    payload = _evidence(tmp_path)
    first_ref = payload["rounds"][0]["optimization"]["artifact"]
    (tmp_path / first_ref["path"]).write_text("changed bytes", encoding="utf-8")
    with pytest.raises(FrontierReportError, match="does not match"):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_rejects_missing_budget_and_incomplete_candidate_ledger(
    tmp_path: Path,
) -> None:
    """Exact finite budgets and one candidate row per completed evaluation are required."""
    payload = _evidence(tmp_path)
    payload["rounds"][0]["optimization"]["budget"].pop("simulator_invocations")
    with pytest.raises(FrontierReportError, match="simulator_invocations"):
        build_frontier_report(payload, evidence_root=tmp_path)

    payload = _evidence(tmp_path)
    payload["rounds"][0]["falsification"]["candidates"].pop()
    with pytest.raises(FrontierReportError, match="do not match completed evaluations"):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_writer_emits_deterministic_json_markdown_and_figure(
    tmp_path: Path,
) -> None:
    """Persisted fixture inputs produce the report and visible plot without a simulator run."""
    input_path = tmp_path / "round-evidence.json"
    input_path.write_text(json.dumps(_evidence(tmp_path)), encoding="utf-8")
    output_dir = tmp_path / "report"

    second_output_dir = tmp_path / "report-repeat"
    first = write_frontier_report(input_path, output_dir)
    first_json = (output_dir / "frontier_report.json").read_bytes()
    first_markdown = (output_dir / "frontier_report.md").read_bytes()
    first_sidecar = json.loads(
        (output_dir / "frontier.provenance.json").read_text(encoding="utf-8")
    )
    second = write_frontier_report(input_path, second_output_dir)

    assert first == second
    assert (second_output_dir / "frontier_report.json").read_bytes() == first_json
    assert (second_output_dir / "frontier_report.md").read_bytes() == first_markdown
    assert (output_dir / "frontier.png").is_file()
    assert (output_dir / "frontier.pdf").is_file()
    sidecar = first_sidecar
    second_sidecar = json.loads(
        (second_output_dir / "frontier.provenance.json").read_text(encoding="utf-8")
    )
    assert sidecar["source_artifacts"] == second_sidecar["source_artifacts"]
    assert sidecar["output_hashes"].keys() == {"frontier.png", "frontier.pdf"}
    for report_dir, provenance in (
        (output_dir, sidecar),
        (second_output_dir, second_sidecar),
    ):
        for name, expected_sha256 in provenance["output_hashes"].items():
            assert hashlib.sha256((report_dir / name).read_bytes()).hexdigest() == expected_sha256
    assert sidecar["repo_commit"] == _REVISION
    assert sidecar["claim_boundary"] == first["claim_boundary"]
    assert sidecar["source_artifacts"][-1]["path"] == input_path.name

    with pytest.raises(FrontierReportError, match="choose a new output directory"):
        write_frontier_report(input_path, output_dir)


def test_frontier_report_rejects_path_escape_and_noncanonical_admissibility(tmp_path: Path) -> None:
    """Case/artifact inputs cannot escape the bundle or invent verdict categories."""
    payload = _evidence(tmp_path)
    payload["rounds"][0]["optimization"]["artifact"]["path"] = "../outside.json"
    with pytest.raises(FrontierReportError, match="stay relative"):
        build_frontier_report(payload, evidence_root=tmp_path)

    payload = _evidence(tmp_path)
    payload["rounds"][0]["falsification"]["candidates"][0]["admissibility_verdict"] = (
        "feasible_for_all_time"
    )
    with pytest.raises(FrontierReportError, match="admissibility_verdict is unsupported"):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_rejects_case_observation_from_nonadmitted_candidate(
    tmp_path: Path,
) -> None:
    """Only an admitted discovery can become an origin for later corpus observations."""
    payload = _evidence(tmp_path)
    failed_candidate = payload["rounds"][0]["falsification"]["candidates"][2]
    failed_candidate["case_id"] = "case-not-admitted"
    payload["rounds"][1]["case_observations"].append(
        _observation(
            tmp_path,
            round_number=2,
            case_id="case-not-admitted",
            origin_round=1,
            origin_candidate_id=failed_candidate["candidate_id"],
            planner_status="unknown",
            verdict="admissible_feasibility_unknown",
            replay_status="verified",
        )
    )

    with pytest.raises(FrontierReportError, match="no admitted discovery"):
        build_frontier_report(payload, evidence_root=tmp_path)
