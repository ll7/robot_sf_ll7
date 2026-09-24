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


def _episode(
    record_id: str,
    *,
    success: bool | None,
    collision: bool | None = False,
    execution_mode: str = "native",
    readiness_status: str = "native",
    availability_status: str = "available",
    eligible: bool = True,
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "evidence_status": "complete",
        "execution_mode": execution_mode,
        "readiness_status": readiness_status,
        "availability_status": availability_status,
        "eligible": eligible,
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
            "execution_mode": "native",
            "readiness_status": "fallback",
            "availability_status": "not_available",
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
    assert first["evaluation_sets"]["held_out"]["readiness_status_counts"]["fallback"] == 1
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
    assert sidecar["evidence_kind"] == "synthetic_fixture"
    assert sidecar["figure_title"].startswith("Synthetic Fixture evidence")
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


def test_frontier_report_uses_canonical_execution_readiness_and_availability_axes(
    tmp_path: Path,
) -> None:
    """Native, adapter, and mixed modes stay distinct from fallback/degraded availability."""
    payload = _evidence(tmp_path)
    evaluation = payload["rounds"][0]["evaluation_sets"]["held_out"]
    rows = evaluation["episodes"]
    rows[1]["execution_mode"] = "adapter"
    rows[1]["readiness_status"] = "adapter"
    rows[2]["execution_mode"] = "mixed"
    rows[2]["eligible"] = True
    rows.append(
        _episode(
            "held_out-1-degraded",
            success=False,
            collision=True,
            execution_mode="mixed",
            readiness_status="degraded",
            availability_status="failed",
            eligible=True,
        )
    )
    rows.append(
        _episode(
            "held_out-1-partial",
            success=True,
            collision=False,
            execution_mode="adapter",
            readiness_status="adapter",
            availability_status="partial-failure",
            eligible=True,
        )
    )
    evaluation["expected_episode_count"] = len(rows)

    summary = build_frontier_report(payload, evidence_root=tmp_path)["rounds"][0][
        "evaluation_sets"
    ]["held_out"]

    assert summary["execution_mode_counts"] == {"adapter": 2, "mixed": 2, "native": 1}
    assert summary["eligible_episode_count"] == 2
    assert summary["readiness_status_counts"]["fallback"] == 1
    assert summary["readiness_status_counts"]["degraded"] == 1
    assert summary["availability_status_counts"]["failed"] == 1
    assert summary["availability_status_counts"]["partial-failure"] == 1
    assert summary["excluded_record_ids"] == [
        "held_out-1-fallback",
        "held_out-1-degraded",
        "held_out-1-partial",
    ]


def test_frontier_report_uses_separate_outcome_denominators(tmp_path: Path) -> None:
    """A missing collision result does not erase success evidence, or vice versa."""
    payload = _evidence(tmp_path)
    rows = payload["rounds"][0]["evaluation_sets"]["held_out"]["episodes"]
    rows[0]["success"] = True
    rows[0]["collision"] = None
    rows[1]["success"] = None
    rows[1]["collision"] = True

    summary = build_frontier_report(payload, evidence_root=tmp_path)["rounds"][0][
        "evaluation_sets"
    ]["held_out"]

    assert summary["eligible_episode_count"] == 2
    assert summary["success_denominator"] == 1
    assert summary["collision_denominator"] == 1
    assert summary["success_rate"] == 1.0
    assert summary["collision_rate"] == 1.0
    assert summary["missing_success_record_ids"] == ["held_out-1-2"]
    assert summary["missing_collision_record_ids"] == ["held_out-1-1"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("admissibility_verdict", "structurally_invalid", "same-round discovery admissibility"),
        ("replay_status", "mismatch", "same-round discovery replay status"),
        ("planner_status", "solved", "despite the same-round observed target failure"),
    ],
)
def test_frontier_report_rejects_same_round_discovery_observation_conflicts(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    """A candidate cannot be verified while its origin observation contradicts it."""
    payload = _evidence(tmp_path)
    payload["rounds"][0]["case_observations"][0][field] = value

    with pytest.raises(FrontierReportError, match=message):
        build_frontier_report(payload, evidence_root=tmp_path)


@pytest.mark.parametrize("include_evidence", [False, True])
def test_frontier_report_requires_evidence_to_strengthen_unknown_feasibility(
    tmp_path: Path, include_evidence: bool
) -> None:
    """A later solved planner state is allowed; a feasibility upgrade needs its own artifact."""
    payload = _evidence(tmp_path)
    first = payload["rounds"][0]
    origin_replay = _artifact(tmp_path, "round-1-unknown-origin-replay.json", role="replay")
    first["falsification"]["candidates"].append(
        _candidate(
            candidate_id="c-unknown-origin",
            evaluation_status="complete",
            verdict="admissible_feasibility_unknown",
            failure=True,
            replay_status="verified",
            disposition="admitted",
            case_id="case-origin-unknown",
            replay_artifact=origin_replay,
        )
    )
    first["falsification"]["budget"]["candidates_completed"] = 4
    first["case_observations"].append(
        _observation(
            tmp_path,
            round_number=1,
            case_id="case-origin-unknown",
            origin_round=1,
            origin_candidate_id="c-unknown-origin",
            planner_status="unsolved",
            verdict="admissible_feasibility_unknown",
            replay_status="verified",
        )
    )
    later_observation = _observation(
        tmp_path,
        round_number=2,
        case_id="case-origin-unknown",
        origin_round=1,
        origin_candidate_id="c-unknown-origin",
        planner_status="solved",
        verdict="empirically_feasible",
        replay_status="verified",
    )
    if include_evidence:
        later_observation["admissibility_evidence_artifact"] = _artifact(
            tmp_path,
            "round-2-case-origin-unknown-feasibility.json",
            role="admissibility-evidence",
        )
    payload["rounds"][1]["case_observations"].append(later_observation)

    if include_evidence:
        report = build_frontier_report(payload, evidence_root=tmp_path)
        second = report["rounds"][1]
        assert second["falsification"]["feasibility_upgrades_from_follow_up_case_ids"] == [
            "case-origin-unknown"
        ]
        assert second["case_frontier"]["verified_counterexamples_cumulative"] == 2
        assert second["case_frontier"]["verified_counterexample_status"]["solved"] == 2
    else:
        with pytest.raises(FrontierReportError, match="unsupported admissibility-verdict"):
            build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_renders_flat_all_round_no_discovery_as_budget_qualified_null(
    tmp_path: Path,
) -> None:
    """A flat, no-discovery campaign remains a useful result without implying absence."""
    payload = _evidence(tmp_path)
    for round_data in payload["rounds"]:
        for candidate in round_data["falsification"]["candidates"]:
            if candidate["corpus_disposition"] == "admitted":
                candidate["corpus_disposition"] = "pending"
                candidate["case_id"] = None
        round_data["case_observations"] = []
        for evaluation in round_data["evaluation_sets"].values():
            evaluation["episodes"][0]["success"] = True
            evaluation["episodes"][1]["success"] = False

    report = build_frontier_report(payload, evidence_root=tmp_path)
    assert [
        item["case_frontier"]["verified_counterexamples_cumulative"] for item in report["rounds"]
    ] == [0, 0]
    assert [item["evaluation_sets"]["held_out"]["success_rate"] for item in report["rounds"]] == [
        0.5,
        0.5,
    ]
    assert all(
        item["falsification"]["no_verified_counterexample_statement"] for item in report["rounds"]
    )

    input_path = tmp_path / "flat-no-discovery.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    output_dir = tmp_path / "flat-no-discovery-report"
    write_frontier_report(input_path, output_dir)
    markdown = (output_dir / "frontier_report.md").read_text(encoding="utf-8")
    assert "This does not establish that no counterexample exists." in markdown
    assert "finite search budget" in markdown
    assert (output_dir / "frontier.png").is_file()
    assert (output_dir / "frontier.pdf").is_file()
