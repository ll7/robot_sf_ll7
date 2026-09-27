"""Fixture tests for the issue #9654 frontier report contract."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

import robot_sf.adversarial.feasibility_frontier_report as frontier_module
from robot_sf.adversarial.feasibility_frontier_report import (
    INPUT_SCHEMA_VERSION,
    FrontierReportError,
    build_frontier_report,
    render_frontier_markdown,
    write_frontier_figure,
    write_frontier_report,
)
from robot_sf.benchmark.figures.provenance import _git_sha_short

_REVISION = "a" * 40
_CONFIG = "b" * 64
_EXPERIMENT_ID = "fixture-two-round-loop"


def test_docs_match_frontier_input_and_report_schema_versions() -> None:
    """The integration guide and context index name the code's current schemas."""
    root = Path(__file__).resolve().parents[2]
    contract = (root / "docs/context/adversarial_feasibility_frontier.md").read_text(
        encoding="utf-8"
    )
    index = (root / "docs/context/INDEX.md").read_text(encoding="utf-8")

    assert INPUT_SCHEMA_VERSION in contract
    assert frontier_module.REPORT_SCHEMA_VERSION in contract
    assert "Fixture-backed v3 input and v2 report output" in index


def test_report_import_and_clear_optional_renderer_error_without_matplotlib() -> None:
    script = r"""
import sys
from importlib.abc import MetaPathFinder
from pathlib import Path

class DenyMatplotlib(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "matplotlib" or fullname.startswith("matplotlib."):
            raise ModuleNotFoundError("matplotlib deliberately unavailable")
        return None

sys.meta_path.insert(0, DenyMatplotlib())
from robot_sf.adversarial.feasibility_frontier_report import (
    FrontierReportError,
    write_frontier_figure,
)
try:
    write_frontier_figure({}, Path("unused"))
except FrontierReportError as exc:
    assert "matplotlib is required" in str(exc)
else:
    raise AssertionError("missing matplotlib should produce FrontierReportError")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_report_writer_can_retry_after_missing_matplotlib(tmp_path: Path) -> None:
    """A failed renderer leaves no partial bundle and the same path can be retried."""
    input_path = tmp_path / "round-evidence.json"
    input_path.write_text(json.dumps(_evidence(tmp_path)), encoding="utf-8")
    output_dir = tmp_path / "report"
    script = r"""
import sys
from importlib.abc import MetaPathFinder
from pathlib import Path

class DenyMatplotlib(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "matplotlib" or fullname.startswith("matplotlib."):
            raise ModuleNotFoundError("matplotlib deliberately unavailable")
        return None

blocker = DenyMatplotlib()
sys.meta_path.insert(0, blocker)
from robot_sf.adversarial.feasibility_frontier_report import (
    FrontierReportError,
    write_frontier_report,
)
input_path, output_dir = map(Path, sys.argv[1:3])
expected = {
    "frontier_report.json",
    "frontier_report.md",
    "frontier.png",
    "frontier.pdf",
    "frontier.provenance.json",
}
try:
    write_frontier_report(input_path, output_dir)
except FrontierReportError as exc:
    assert "matplotlib is required" in str(exc)
else:
    raise AssertionError("missing matplotlib should fail report rendering")
assert not any((output_dir / name).exists() for name in expected)
sys.meta_path.remove(blocker)
write_frontier_report(input_path, output_dir)
assert all((output_dir / name).is_file() for name in expected)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(input_path), str(output_dir)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


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


def _scenario_id(case_id: str) -> str:
    return f"scenario-{case_id}"


def _scenario_artifact_bytes(case_id: str) -> bytes:
    return f"scenario_id: {_scenario_id(case_id)}\n".encode()


def _write_scenario_artifact(root: Path, case_id: str) -> Path:
    path = root / "fixture" / f"{_scenario_id(case_id)}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_scenario_artifact_bytes(case_id))
    return path


def _scenario_digest(case_id: str) -> str:
    return hashlib.sha256(_scenario_artifact_bytes(case_id)).hexdigest()


def _execution_record(
    *,
    case_id: str,
    scenario_id: str,
    planner_id: str,
    route_complete: bool,
    planner_config_sha256: str,
    role: str,
    evidence_ref: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "case_id": case_id,
        "scenario_id": scenario_id,
        "scenario_variant": "original",
        "planner_id": planner_id,
        "run_status": "ok",
        "fallback_or_degraded": False,
        "route_complete": route_complete,
        "seed": 17,
        "horizon_steps": 100,
        "scenario_sha256": _scenario_digest(case_id),
        "robot_model_sha256": "d" * 64,
        "simulator_config_sha256": "e" * 64,
        "planner_config_sha256": planner_config_sha256,
        "planner_checkpoint_sha256": "not_applicable",
        "environment_sha256": "f" * 64,
        "source_commit": _REVISION,
        "evidence_ref": evidence_ref or f"fixture/{role}/{case_id}",
    }
    if role in {"target", "replay"}:
        record.update(
            episode_id=f"episode-{case_id}-target",
            source_episodes_jsonl_sha256="a" * 64,
        )
    if role == "replay":
        record.update(determinism_check_status="pass", resimulated=True)
    return record


def _admissibility_artifact(
    root: Path,
    name: str,
    *,
    case_id: str,
    verdict: str,
    planner_config_sha256: str | None = None,
    source_revision: str = _REVISION,
    replay_artifact_path: str | None = None,
    target_route_complete: bool = False,
) -> dict[str, str]:
    """Write a case-bound #9651 verdict fixture with explicit feasibility support."""
    confirmed = verdict in {"empirically_feasible", "planner_specific_failure"}
    if planner_config_sha256 is None:
        round_match = re.search(r"round-(\d+)", name)
        planner_config_sha256 = (
            "c" * 64 if round_match and int(round_match.group(1)) > 1 else _CONFIG
        )
    scenario_id = _scenario_id(case_id)
    _write_scenario_artifact(root, case_id)
    evidence: dict[str, Any] = {
        "scenario_artifact_identity": {
            "status": "available",
            "path": f"fixture/{scenario_id}.yaml",
            "sha256": _scenario_digest(case_id),
            "effective_input_sha256": None,
            "requires_effective_input_binding": False,
            "effective_input_files": [],
        }
    }
    assumptions: dict[str, Any] = {}
    if verdict == "empirically_feasible":
        evidence.update(
            reference_execution=_execution_record(
                case_id=case_id,
                scenario_id=scenario_id,
                planner_id="orca",
                route_complete=True,
                planner_config_sha256="1" * 64,
                role="reference",
            ),
            target_execution=_execution_record(
                case_id=case_id,
                scenario_id=scenario_id,
                planner_id="goal",
                route_complete=target_route_complete,
                planner_config_sha256=planner_config_sha256,
                role="target",
            ),
        )
        if replay_artifact_path is not None:
            evidence["replay_execution"] = _execution_record(
                case_id=case_id,
                scenario_id=scenario_id,
                planner_id="goal",
                route_complete=target_route_complete,
                planner_config_sha256=planner_config_sha256,
                role="replay",
                evidence_ref=replay_artifact_path,
            )
        reason_codes = ["named_execution_completed_original_case"]
        target_outcome = "route_completed" if target_route_complete else "route_incomplete"
    elif verdict == "planner_specific_failure":
        evidence.update(
            reference_execution=_execution_record(
                case_id=case_id,
                scenario_id=scenario_id,
                planner_id="orca",
                route_complete=True,
                planner_config_sha256="1" * 64,
                role="reference",
            ),
            target_execution=_execution_record(
                case_id=case_id,
                scenario_id=scenario_id,
                planner_id="goal",
                route_complete=target_route_complete,
                planner_config_sha256=planner_config_sha256,
                role="target",
            ),
            replay_execution=_execution_record(
                case_id=case_id,
                scenario_id=scenario_id,
                planner_id="goal",
                route_complete=target_route_complete,
                planner_config_sha256=planner_config_sha256,
                role="replay",
                evidence_ref=replay_artifact_path,
            ),
        )
        reason_codes = ["matched_reference_target_failure_reproduced_by_replay"]
        target_outcome = "route_completed" if target_route_complete else "route_incomplete"
    elif replay_artifact_path is not None:
        evidence.update(
            target_execution=_execution_record(
                case_id=case_id,
                scenario_id=scenario_id,
                planner_id="goal",
                route_complete=False,
                planner_config_sha256=planner_config_sha256,
                role="target",
            ),
            replay_execution=_execution_record(
                case_id=case_id,
                scenario_id=scenario_id,
                planner_id="goal",
                route_complete=False,
                planner_config_sha256=planner_config_sha256,
                role="replay",
                evidence_ref=replay_artifact_path,
            ),
        )
        reason_codes = ["feasibility_not_demonstrated"]
        target_outcome = "route_incomplete"
    else:
        reason_codes = ["feasibility_not_demonstrated"]
        target_outcome = "unavailable"
    payload = {
        "schema_version": "scenario_admissibility.v1",
        "case_id": case_id,
        "scenario_id": scenario_id,
        "verdict": verdict,
        "target_planner_outcome": target_outcome,
        "search_disposition": "retain"
        if confirmed or verdict == "admissible_feasibility_unknown"
        else "reject",
        "reason_codes": reason_codes,
        "assumptions": assumptions,
        "evidence": evidence,
    }
    content = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path = root / "evidence" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": f"evidence/{name}",
        "sha256": hashlib.sha256(content).hexdigest(),
        "source_revision": source_revision,
        "role": "admissibility-evidence",
        "schema_version": "scenario_admissibility.v1",
    }


def _write_source_artifact(
    root: Path,
    reference: dict[str, Any],
    payload: dict[str, Any],
    *,
    schema_version: str,
) -> None:
    content = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path = root / reference["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    reference["sha256"] = hashlib.sha256(content).hexdigest()
    reference["source_revision"] = payload["source_revision"]
    reference["schema_version"] = schema_version


def _rewrite_admissibility_artifact(
    payload: dict[str, Any], root: Path, candidate_index: int, mutation: Any
) -> None:
    candidate = payload["rounds"][0]["falsification"]["candidates"][candidate_index]
    reference = candidate["admissibility_evidence_artifact"]
    path = root / reference["path"]
    record = json.loads(path.read_text(encoding="utf-8"))
    mutation(record)
    content = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(content)
    reference["sha256"] = hashlib.sha256(content).hexdigest()
    _refresh_source_artifacts(payload, root)


def _case_status_source_payload(
    observation: dict[str, Any],
    *,
    experiment_id: str,
    round_number: int,
    source_revision: str,
    planner: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": frontier_module._CORPUS_CASE_STATUS_SCHEMA,
        "experiment_id": experiment_id,
        "round_number": round_number,
        "source_revision": source_revision,
        "case_id": observation["case_id"],
        "origin_round": observation["origin_round"],
        "origin_candidate_id": observation["origin_candidate_id"],
        "scenario_id": observation["scenario_id"],
        "scenario_artifact_sha256": observation["scenario_artifact_sha256"],
        "admissibility_evidence_artifact": observation["admissibility_evidence_artifact"],
        "planner_id": planner["planner_id"],
        "config_identity_sha256": planner["config_identity_sha256"],
        "planner_status": observation["planner_status"],
        "admissibility_verdict": observation["admissibility_verdict"],
        "replay_status": observation["replay_status"],
        "evidence_status": observation["evidence_status"],
    }


def _refresh_source_artifacts(evidence: dict[str, Any], root: Path) -> None:
    """Re-emit fixture source bytes after intentional fixture edits."""
    rounds = evidence["rounds"]
    for round_data in rounds:
        round_number = round_data["round_number"]
        source_revision = round_data["source_revision"]
        planner = round_data["planner"]
        common = {
            "experiment_id": evidence["experiment_id"],
            "round_number": round_number,
            "source_revision": source_revision,
        }
        _write_source_artifact(
            root,
            round_data["optimization"]["artifact"],
            {
                **common,
                "schema_version": frontier_module._OPTIMIZER_SELECTION_SCHEMA,
                "selected_planner_id": planner["planner_id"],
                "selected_config_identity_sha256": planner["config_identity_sha256"],
            },
            schema_version=frontier_module._OPTIMIZER_SELECTION_SCHEMA,
        )
        candidates = round_data["falsification"]["candidates"]
        _write_source_artifact(
            root,
            round_data["falsification"]["artifact"],
            {
                **common,
                "schema_version": frontier_module._FALSIFICATION_SOURCE_SCHEMA,
                "target_planner_id": planner["planner_id"],
                "target_config_identity_sha256": planner["config_identity_sha256"],
                "candidate_records": [
                    {
                        field: candidate.get(field)
                        for field in frontier_module._SEARCH_CANDIDATE_SOURCE_FIELDS
                    }
                    for candidate in candidates
                ],
            },
            schema_version=frontier_module._FALSIFICATION_SOURCE_SCHEMA,
        )
        for set_name, evaluation in round_data["evaluation_sets"].items():
            expected_ids = evaluation.get("expected_episode_ids")
            if not isinstance(expected_ids, list):
                expected_ids = []
            else:
                expected_ids = list(expected_ids)
            expected_id_set = set(expected_ids)
            expected_identities = evaluation.get("expected_episode_identities", [])
            identity_by_id = {
                item["record_id"]: item
                for item in expected_identities
                if isinstance(item, dict) and isinstance(item.get("record_id"), str)
            }
            for row in evaluation["episodes"]:
                record_id = row.get("record_id")
                if isinstance(record_id, str) and record_id not in expected_id_set:
                    expected_ids.append(record_id)
                    expected_id_set.add(record_id)
                if isinstance(record_id, str) and record_id not in identity_by_id:
                    identity = {
                        "record_id": record_id,
                        "scenario_id": row.get("scenario_id"),
                        "scenario_seed": row.get("scenario_seed"),
                    }
                    expected_identities.append(identity)
                    identity_by_id[record_id] = identity
            evaluation["expected_episode_ids"] = expected_ids
            evaluation["expected_episode_identities"] = expected_identities
            _write_source_artifact(
                root,
                evaluation["artifact"],
                {
                    **common,
                    "schema_version": frontier_module._EVALUATION_SOURCE_SCHEMA,
                    "evaluation_set": set_name,
                    "planner_id": planner["planner_id"],
                    "config_identity_sha256": planner["config_identity_sha256"],
                    "expected_episode_ids": expected_ids,
                    "expected_episode_identities": expected_identities,
                    "episodes": [
                        {
                            field: row.get(field)
                            for field in frontier_module._EVALUATION_ROW_SOURCE_FIELDS
                        }
                        for row in evaluation["episodes"]
                    ],
                },
                schema_version=frontier_module._EVALUATION_SOURCE_SCHEMA,
            )

    _refresh_case_status_artifacts(evidence, root)


def _refresh_case_status_artifacts(evidence: dict[str, Any], root: Path) -> None:
    """Rebind each fixture case row to a unique checksummed corpus snapshot."""
    rounds = evidence["rounds"]
    for round_data in rounds:
        for observation in round_data["case_observations"]:
            origin_round = observation.get("origin_round")
            if isinstance(origin_round, int) and 1 <= origin_round <= len(rounds):
                origin_data = rounds[origin_round - 1]
                observation["origin_search_artifact"] = origin_data["falsification"]["artifact"]
                candidate = next(
                    (
                        item
                        for item in origin_data["falsification"]["candidates"]
                        if item.get("candidate_id") == observation.get("origin_candidate_id")
                        and item.get("case_id") == observation.get("case_id")
                        and item.get("corpus_disposition") == "admitted"
                    ),
                    None,
                )
                if candidate is not None:
                    observation["replay_artifact"] = candidate.get("replay_artifact")
                    if origin_round == round_data.get("round_number"):
                        observation["admissibility_evidence_artifact"] = candidate.get(
                            "admissibility_evidence_artifact"
                        )
            planner = round_data["planner"]
            corpus_digest = hashlib.sha256(observation["case_id"].encode("utf-8")).hexdigest()[:12]
            observation["corpus_artifact"]["path"] = (
                f"evidence/round-{round_data['round_number']}-case-status-{corpus_digest}.json"
            )
            _write_source_artifact(
                root,
                observation["corpus_artifact"],
                _case_status_source_payload(
                    observation,
                    experiment_id=evidence["experiment_id"],
                    round_number=round_data["round_number"],
                    source_revision=round_data["source_revision"],
                    planner=planner,
                ),
                schema_version=frontier_module._CORPUS_CASE_STATUS_SCHEMA,
            )


def _episode(
    record_id: str,
    *,
    scenario_identity: tuple[str, int],
    success: bool | None,
    collision: bool | None = False,
    execution_mode: str = "native",
    readiness_status: str = "native",
    availability_status: str = "available",
    eligible: bool = True,
) -> dict[str, Any]:
    scenario_id, scenario_seed = scenario_identity
    return {
        "record_id": record_id,
        "scenario_id": scenario_id,
        "scenario_seed": scenario_seed,
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
        _episode(
            f"{name}-{round_number}-1",
            scenario_identity=(f"{name}-scenario-1", 1101),
            success=success_count >= 1,
        ),
        _episode(
            f"{name}-{round_number}-2",
            scenario_identity=(f"{name}-scenario-2", 2202),
            success=success_count >= 2,
            collision=round_number == 1,
        ),
        {
            "record_id": f"{name}-{round_number}-fallback",
            "scenario_id": f"{name}-scenario-fallback",
            "scenario_seed": 3303,
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
    expected_ids = [row["record_id"] for row in rows]
    expected_identities = [
        {
            "record_id": row["record_id"],
            "scenario_id": row["scenario_id"],
            "scenario_seed": row["scenario_seed"],
        }
        for row in rows
    ]
    return {
        "expected_episode_count": len(rows),
        "expected_episode_ids": expected_ids,
        "expected_episode_identities": expected_identities,
        "artifact": _artifact(
            root, f"round-{round_number}-{name}.jsonl", role=f"{name}-evaluation"
        ),
        "episodes": rows,
    }


def _candidate(  # noqa: PLR0913 - fixture helper mirrors the persisted candidate fields.
    *,
    candidate_id: str,
    evaluation_status: str,
    verdict: str,
    failure: bool | None,
    replay_status: str,
    disposition: str,
    case_id: str | None = None,
    scenario_id: str | None = None,
    scenario_artifact_sha256: str | None = None,
    planner_config_sha256: str = _CONFIG,
    replay_artifact: dict[str, str] | None = None,
    admissibility_evidence_artifact: dict[str, str] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    if (
        admissibility_evidence_artifact is None
        and root is not None
        and disposition in {"admitted", "duplicate"}
        and isinstance(case_id, str)
    ):
        admissibility_evidence_artifact = _admissibility_artifact(
            root,
            f"candidate-{candidate_id}-admissibility.json",
            case_id=case_id,
            verdict=verdict,
            planner_config_sha256=planner_config_sha256,
            replay_artifact_path=(
                replay_artifact.get("path")
                if replay_status == "verified" and isinstance(replay_artifact, dict)
                else None
            ),
        )
    if case_id is not None:
        scenario_id = scenario_id or _scenario_id(case_id)
        scenario_artifact_sha256 = scenario_artifact_sha256 or _scenario_digest(case_id)
    return {
        "candidate_id": candidate_id,
        "evaluation_status": evaluation_status,
        "admissibility_verdict": verdict,
        "scenario_id": scenario_id,
        "scenario_artifact_sha256": scenario_artifact_sha256,
        "admissibility_evidence_artifact": admissibility_evidence_artifact,
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
    evidence_status = "complete" if planner_status != "unknown" else "unknown"
    replay_artifact = (
        _artifact(root, f"round-{round_number}-{case_id}-replay.json", role="replay")
        if replay_status == "verified"
        else None
    )
    corpus_ref = _artifact(
        root,
        f"round-{round_number}-case-status-{hashlib.sha256(case_id.encode()).hexdigest()[:12]}.json",
        role="corpus",
    )
    observation = {
        "case_id": case_id,
        "scenario_id": _scenario_id(case_id),
        "scenario_artifact_sha256": _scenario_digest(case_id),
        "origin_round": origin_round,
        "origin_candidate_id": origin_candidate_id,
        "planner_status": planner_status,
        "admissibility_verdict": verdict,
        "replay_status": replay_status,
        "evidence_status": evidence_status,
        "origin_search_artifact": _artifact(
            root,
            f"round-{origin_round or 'historical'}-search.json",
            role="falsification-search",
        ),
        "corpus_artifact": corpus_ref,
        "replay_artifact": replay_artifact,
        "admissibility_evidence_artifact": _admissibility_artifact(
            root,
            f"round-{round_number}-{hashlib.sha256(case_id.encode()).hexdigest()[:12]}-admissibility.json",
            case_id=case_id,
            verdict=verdict,
            planner_config_sha256=_CONFIG if round_number == 1 else "c" * 64,
            target_route_complete=planner_status == "solved",
            replay_artifact_path=(
                replay_artifact.get("path")
                if origin_round == 0
                and replay_status == "verified"
                and isinstance(replay_artifact, dict)
                else None
            ),
        ),
    }
    planner = {
        "planner_id": "goal",
        "config_identity_sha256": _CONFIG if round_number == 1 else "c" * 64,
    }
    _write_source_artifact(
        root,
        corpus_ref,
        _case_status_source_payload(
            observation,
            experiment_id=_EXPERIMENT_ID,
            round_number=round_number,
            source_revision=_REVISION,
            planner=planner,
        ),
        schema_version=frontier_module._CORPUS_CASE_STATUS_SCHEMA,
    )
    return observation


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
                replay_status="unavailable",
                disposition="pending",
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
    for candidate in candidates:
        if candidate.get("corpus_disposition") in {"admitted", "duplicate"} and isinstance(
            candidate.get("case_id"), str
        ):
            candidate["admissibility_evidence_artifact"] = _admissibility_artifact(
                root,
                f"round-{round_number}-{candidate['candidate_id']}-admissibility.json",
                case_id=candidate["case_id"],
                verdict=candidate["admissibility_verdict"],
                planner_config_sha256=_CONFIG if round_number == 1 else "c" * 64,
                replay_artifact_path=(
                    candidate["replay_artifact"].get("path")
                    if candidate.get("replay_status") == "verified"
                    and isinstance(candidate.get("replay_artifact"), dict)
                    else None
                ),
            )
    for observation in observations:
        if observation.get("origin_round") != round_number:
            continue
        candidate = next(
            (
                item
                for item in candidates
                if item.get("candidate_id") == observation.get("origin_candidate_id")
                and item.get("case_id") == observation.get("case_id")
                and item.get("corpus_disposition") == "admitted"
            ),
            None,
        )
        if candidate is not None:
            observation["admissibility_evidence_artifact"] = candidate.get(
                "admissibility_evidence_artifact"
            )
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
    evidence = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "evidence_kind": "synthetic_fixture",
        "experiment_id": _EXPERIMENT_ID,
        "source_revision": _REVISION,
        "simulator_identity": "robot-sf-fixture-simulator.v1",
        "scenario_space_id": "crossing-gap-v1",
        "rounds": [_round(root, 1), _round(root, 2)],
    }
    _refresh_source_artifacts(evidence, root)
    return evidence


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
    assert first["evaluation_sets"]["held_out"]["identity_accounting_status"] == "verified"
    assert len(first["evaluation_sets"]["held_out"]["expected_episode_ids_sha256"]) == 64
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
    assert second["case_frontier"]["current_unknown_feasibility_case_count"] == 1
    assert second["falsification"]["no_verified_counterexample_statement"].endswith(
        "This does not establish that no counterexample exists."
    )
    markdown = render_frontier_markdown(report)
    assert "9 simulator invocations" in markdown
    assert "case-001" in markdown
    assert "case-unknown" in markdown
    assert "replay artifact" in markdown.lower()
    assert "Search candidate accounting" in markdown


def test_frontier_report_does_not_count_repeated_case_as_new_discovery(tmp_path: Path) -> None:
    """A verified repeat is visible, but does not restart the unique discovery count."""
    payload = _evidence(tmp_path)
    repeat_replay = _artifact(tmp_path, "round-2-repeat-case-001-replay.json", role="replay")
    second_search = payload["rounds"][1]["falsification"]
    second_search["candidates"].append(
        _candidate(
            candidate_id="c-repeat-known",
            evaluation_status="complete",
            verdict="empirically_feasible",
            failure=True,
            replay_status="verified",
            disposition="duplicate",
            case_id="case-001",
            replay_artifact=repeat_replay,
            admissibility_evidence_artifact=_admissibility_artifact(
                tmp_path,
                "round-2-repeat-case-001-admissibility.json",
                case_id="case-001",
                verdict="empirically_feasible",
                replay_artifact_path=repeat_replay["path"],
            ),
        )
    )
    second_search["budget"]["candidate_limit"] = 5
    second_search["budget"]["candidates_completed"] = len(second_search["candidates"])

    _refresh_source_artifacts(payload, tmp_path)
    report = build_frontier_report(payload, evidence_root=tmp_path)
    second = report["rounds"][1]
    assert second["falsification"]["verified_counterexample_case_ids"] == []
    assert second["falsification"]["repeated_verified_counterexample_case_ids"] == ["case-001"]
    assert second["case_frontier"]["verified_counterexamples_cumulative"] == 1
    statement = second["falsification"]["no_verified_counterexample_statement"]
    assert "No new unique replay-verified" in statement
    assert "1 known corpus case(s) were replay-verified again." in statement


def test_frontier_report_rejects_duplicate_case_with_different_scenario_identity(
    tmp_path: Path,
) -> None:
    """A checksummed replay cannot turn changed scenario bytes into a known-case repeat."""
    payload = _evidence(tmp_path)
    candidate = payload["rounds"][1]["falsification"]["candidates"][0]
    candidate["corpus_disposition"] = "duplicate"
    candidate["case_id"] = "case-001"
    alternate_scenario_id = "unrelated-scenario"
    alternate_scenario_bytes = b"alternate scenario bytes for identity regression\n"
    alternate_scenario_sha256 = hashlib.sha256(alternate_scenario_bytes).hexdigest()
    scenario_path = tmp_path / "fixture" / f"{alternate_scenario_id}.yaml"
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.write_bytes(alternate_scenario_bytes)
    candidate["scenario_id"] = alternate_scenario_id
    candidate["scenario_artifact_sha256"] = alternate_scenario_sha256

    admissibility_ref = candidate["admissibility_evidence_artifact"]
    admissibility_path = tmp_path / admissibility_ref["path"]
    admissibility = json.loads(admissibility_path.read_text(encoding="utf-8"))
    admissibility["case_id"] = "case-001"
    admissibility["scenario_id"] = alternate_scenario_id
    identity = admissibility["evidence"]["scenario_artifact_identity"]
    identity["path"] = f"fixture/{alternate_scenario_id}.yaml"
    identity["sha256"] = alternate_scenario_sha256
    for role in ("target", "replay"):
        execution = admissibility["evidence"][f"{role}_execution"]
        execution["case_id"] = "case-001"
        execution["scenario_id"] = alternate_scenario_id
        execution["scenario_sha256"] = alternate_scenario_sha256
    admissibility_bytes = (
        json.dumps(admissibility, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    admissibility_path.write_bytes(admissibility_bytes)
    admissibility_ref["sha256"] = hashlib.sha256(admissibility_bytes).hexdigest()

    _refresh_source_artifacts(payload, tmp_path)
    with pytest.raises(
        FrontierReportError,
        match="duplicate candidate for case 'case-001' does not match its canonical scenario identity",
    ):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_accepts_case_identity_with_uppercase_digest_hex(
    tmp_path: Path,
) -> None:
    """Equivalent hexadecimal casing does not change a scenario's byte identity."""
    payload = _evidence(tmp_path)
    observation = payload["rounds"][0]["case_observations"][0]
    observation["scenario_artifact_sha256"] = observation["scenario_artifact_sha256"].upper()
    _refresh_source_artifacts(payload, tmp_path)

    report = build_frontier_report(payload, evidence_root=tmp_path)
    assert report["rounds"][0]["falsification"]["verified_counterexample_case_ids"] == ["case-001"]


def test_frontier_report_does_not_call_unknown_case_duplicate_a_verified_repeat(
    tmp_path: Path,
) -> None:
    """A duplicate is a repeat only after its case was confirmed before this round."""
    payload = _evidence(tmp_path)
    payload["rounds"][0]["case_observations"].append(
        _observation(
            tmp_path,
            round_number=1,
            case_id="historical-unknown-duplicate",
            origin_round=0,
            origin_candidate_id=None,
            planner_status="unknown",
            verdict="admissible_feasibility_unknown",
            replay_status="unavailable",
        )
    )
    repeat_replay = _artifact(tmp_path, "round-2-unknown-case-replay.json", role="replay")
    second_search = payload["rounds"][1]["falsification"]
    second_search["candidates"].append(
        _candidate(
            candidate_id="c-repeat-unknown",
            evaluation_status="complete",
            verdict="empirically_feasible",
            failure=True,
            replay_status="verified",
            disposition="duplicate",
            case_id="historical-unknown-duplicate",
            replay_artifact=repeat_replay,
            admissibility_evidence_artifact=_admissibility_artifact(
                tmp_path,
                "round-2-repeat-historical-unknown-admissibility.json",
                case_id="historical-unknown-duplicate",
                verdict="empirically_feasible",
                replay_artifact_path=repeat_replay["path"],
            ),
        )
    )
    second_search["budget"]["candidate_limit"] += 1
    second_search["budget"]["candidates_completed"] = len(second_search["candidates"])

    _refresh_source_artifacts(payload, tmp_path)
    report = build_frontier_report(payload, evidence_root=tmp_path)
    second = report["rounds"][1]
    assert second["falsification"]["repeated_verified_counterexample_case_ids"] == []
    assert second["falsification"]["verified_counterexample_case_ids"] == []
    assert second["case_frontier"]["current_unknown_feasibility_case_count"] == 2
    assert (
        "No new unique replay-verified"
        in second["falsification"]["no_verified_counterexample_statement"]
    )


def test_frontier_report_rejects_re_admission_of_known_case(tmp_path: Path) -> None:
    """Known stable IDs use duplicate disposition instead of a second admission."""
    payload = _evidence(tmp_path)
    payload["rounds"][1]["falsification"]["candidates"][0]["case_id"] = "case-001"
    with pytest.raises(FrontierReportError, match="re-admits known case"):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_seeds_confirmed_historical_cases_into_corpus_status(
    tmp_path: Path,
) -> None:
    """Pre-loop verified cases count in status and memory without becoming new discoveries."""
    payload = _evidence(tmp_path)
    historical_id = "historical-confirmed"
    for round_number, round_data in enumerate(payload["rounds"], start=1):
        round_data["case_observations"].append(
            _observation(
                tmp_path,
                round_number=round_number,
                case_id=historical_id,
                origin_round=0,
                origin_candidate_id=None,
                planner_status="unsolved",
                verdict="empirically_feasible",
                replay_status="verified",
            )
        )

    report = build_frontier_report(payload, evidence_root=tmp_path)
    first, second = report["rounds"]
    assert first["case_frontier"]["verified_counterexamples_cumulative"] == 1
    assert first["case_frontier"]["confirmed_counterexamples_in_corpus"] == 2
    assert first["case_frontier"]["verified_counterexample_status"]["unsolved"] == 2
    assert second["case_frontier"]["confirmed_counterexamples_in_corpus"] == 2
    assert second["case_frontier"]["verified_counterexample_status"]["solved"] == 1
    assert second["case_frontier"]["verified_counterexample_status"]["unsolved"] == 1


def test_frontier_report_can_upgrade_historical_unknown_feasibility_once(tmp_path: Path) -> None:
    """Origin-zero cases use their latest observation without indexing a search round."""
    payload = _evidence(tmp_path)
    first = payload["rounds"][0]
    first["case_observations"].append(
        _observation(
            tmp_path,
            round_number=1,
            case_id="historical-unknown",
            origin_round=0,
            origin_candidate_id=None,
            planner_status="unsolved",
            verdict="admissible_feasibility_unknown",
            replay_status="verified",
        )
    )
    upgraded = _observation(
        tmp_path,
        round_number=2,
        case_id="historical-unknown",
        origin_round=0,
        origin_candidate_id=None,
        planner_status="solved",
        verdict="empirically_feasible",
        replay_status="verified",
    )
    upgraded["admissibility_evidence_artifact"] = _admissibility_artifact(
        tmp_path,
        "historical-upgrade-proof.json",
        case_id="historical-unknown",
        verdict="empirically_feasible",
        planner_config_sha256="c" * 64,
        target_route_complete=True,
        replay_artifact_path=upgraded["replay_artifact"]["path"],
    )
    payload["rounds"][1]["case_observations"].append(upgraded)

    _refresh_source_artifacts(payload, tmp_path)
    report = build_frontier_report(payload, evidence_root=tmp_path)
    first_report, second_report = report["rounds"]
    assert first_report["case_frontier"]["current_unknown_feasibility_case_count"] == 1
    assert second_report["falsification"]["feasibility_upgrades_from_follow_up_case_ids"] == [
        "historical-unknown"
    ]
    assert second_report["case_frontier"]["verified_counterexamples_cumulative"] == 2
    assert second_report["case_frontier"]["confirmed_counterexamples_in_corpus"] == 2


@pytest.mark.parametrize("replay_status", ["unavailable", "mismatch"])
def test_frontier_report_does_not_count_historical_upgrade_without_verified_failure_replay(
    tmp_path: Path, replay_status: str
) -> None:
    """Feasibility evidence alone cannot turn a historical case into a verified counterexample."""
    payload = _evidence(tmp_path)
    payload["rounds"][0]["case_observations"].append(
        _observation(
            tmp_path,
            round_number=1,
            case_id="historical-unknown",
            origin_round=0,
            origin_candidate_id=None,
            planner_status="unsolved",
            verdict="admissible_feasibility_unknown",
            replay_status=replay_status,
        )
    )
    upgraded = _observation(
        tmp_path,
        round_number=2,
        case_id="historical-unknown",
        origin_round=0,
        origin_candidate_id=None,
        planner_status="solved",
        verdict="empirically_feasible",
        replay_status="unavailable",
    )
    upgraded["admissibility_evidence_artifact"] = _admissibility_artifact(
        tmp_path,
        "historical-unreplayed-upgrade-proof.json",
        case_id="historical-unknown",
        verdict="empirically_feasible",
        planner_config_sha256="c" * 64,
        target_route_complete=True,
    )
    payload["rounds"][1]["case_observations"].append(upgraded)

    _refresh_source_artifacts(payload, tmp_path)
    report = build_frontier_report(payload, evidence_root=tmp_path)
    second = report["rounds"][1]
    assert second["falsification"]["feasibility_upgrades_from_follow_up_case_ids"] == []
    assert second["falsification"][
        "feasibility_upgrades_without_verified_counterexample_case_ids"
    ] == ["historical-unknown"]
    assert second["falsification"]["verified_counterexample_case_ids"] == []
    assert second["case_frontier"]["verified_counterexamples_cumulative"] == 1
    assert second["case_frontier"]["confirmed_counterexamples_in_corpus"] == 1


def test_frontier_report_credits_historical_upgrade_when_replay_evidence_arrives(
    tmp_path: Path,
) -> None:
    """A later replay can validate an earlier upgrade only from its own round onward."""
    payload = _evidence(tmp_path)
    payload["rounds"][0]["case_observations"].append(
        _observation(
            tmp_path,
            round_number=1,
            case_id="historical-delayed-replay",
            origin_round=0,
            origin_candidate_id=None,
            planner_status="unsolved",
            verdict="admissible_feasibility_unknown",
            replay_status="unavailable",
        )
    )
    upgraded = _observation(
        tmp_path,
        round_number=2,
        case_id="historical-delayed-replay",
        origin_round=0,
        origin_candidate_id=None,
        planner_status="solved",
        verdict="empirically_feasible",
        replay_status="unavailable",
    )
    upgraded["admissibility_evidence_artifact"] = _admissibility_artifact(
        tmp_path,
        "historical-delayed-feasibility-proof.json",
        case_id="historical-delayed-replay",
        verdict="empirically_feasible",
        planner_config_sha256="c" * 64,
        target_route_complete=True,
    )
    payload["rounds"][1]["case_observations"].append(upgraded)
    third = _round(tmp_path, 3)
    third["falsification"]["candidates"] = []
    third["falsification"]["budget"]["candidates_completed"] = 0
    third["case_observations"] = [
        _observation(
            tmp_path,
            round_number=3,
            case_id="historical-delayed-replay",
            origin_round=0,
            origin_candidate_id=None,
            planner_status="unsolved",
            verdict="empirically_feasible",
            replay_status="verified",
        ),
        _observation(
            tmp_path,
            round_number=3,
            case_id="historical-late-confirmed",
            origin_round=0,
            origin_candidate_id=None,
            planner_status="unsolved",
            verdict="empirically_feasible",
            replay_status="verified",
        ),
    ]
    payload["rounds"].append(third)
    _refresh_source_artifacts(payload, tmp_path)

    report = build_frontier_report(payload, evidence_root=tmp_path)
    first_report, second, third_report = report["rounds"]
    assert first_report["case_frontier"]["confirmed_counterexamples_in_corpus"] == 1
    assert second["falsification"]["feasibility_upgrades_from_follow_up_case_ids"] == []
    assert second["falsification"][
        "feasibility_upgrades_without_verified_counterexample_case_ids"
    ] == ["historical-delayed-replay"]
    assert second["case_frontier"]["verified_counterexamples_cumulative"] == 1
    assert second["case_frontier"]["confirmed_counterexamples_in_corpus"] == 1
    assert third_report["falsification"]["feasibility_upgrades_from_follow_up_case_ids"] == [
        "historical-delayed-replay"
    ]
    assert third_report["falsification"]["verified_counterexample_case_ids"] == [
        "historical-delayed-replay"
    ]
    assert third_report["falsification"][
        "historical_counterexamples_confirmed_this_round_case_ids"
    ] == ["historical-delayed-replay", "historical-late-confirmed"]
    assert third_report["case_frontier"]["verified_counterexamples_cumulative"] == 2
    assert third_report["case_frontier"]["confirmed_counterexamples_in_corpus"] == 3


def test_frontier_report_rejects_case_origin_identity_changes(tmp_path: Path) -> None:
    """Stable case IDs retain the same origin round and candidate across observations."""
    payload = _evidence(tmp_path)
    payload["rounds"][1]["case_observations"].append(
        _observation(
            tmp_path,
            round_number=2,
            case_id="case-001",
            origin_round=0,
            origin_candidate_id=None,
            planner_status="solved",
            verdict="empirically_feasible",
            replay_status="verified",
        )
    )

    with pytest.raises(FrontierReportError, match="changes its origin round or candidate"):
        build_frontier_report(payload, evidence_root=tmp_path)


@pytest.mark.parametrize("target", ["candidate", "observation"])
def test_frontier_report_requires_replay_role_for_verified_artifact(
    tmp_path: Path, target: str
) -> None:
    """A valid digest cannot make a non-replay artifact prove replay verification."""
    payload = _evidence(tmp_path)
    first = payload["rounds"][0]
    if target == "candidate":
        first["falsification"]["candidates"][0]["replay_artifact"]["role"] = "optimization"
    else:
        first["case_observations"][0]["replay_artifact"]["role"] = "optimization"

    with pytest.raises(FrontierReportError, match="replay_artifact.role must be 'replay'"):
        build_frontier_report(payload, evidence_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "expected_role"),
    [
        ("origin_search_artifact", "falsification-search"),
        ("corpus_artifact", "corpus"),
    ],
)
def test_frontier_report_requires_typed_case_artifact_roles(
    tmp_path: Path, field: str, expected_role: str
) -> None:
    """A checksummed artifact cannot substitute for a search or corpus source by relabeling."""
    payload = _evidence(tmp_path)
    payload["rounds"][0]["case_observations"][0][field]["role"] = "optimization"

    with pytest.raises(
        FrontierReportError,
        match=rf"{field}\.role must be '{expected_role}'",
    ):
        build_frontier_report(payload, evidence_root=tmp_path)


@pytest.mark.parametrize(
    ("field_path", "expected_role"),
    [
        (("optimization", "artifact"), "optimization"),
        (("falsification", "artifact"), "falsification-search"),
        (("evaluation_sets", "fixed", "artifact"), "fixed-evaluation"),
        (("evaluation_sets", "regression", "artifact"), "regression-evaluation"),
        (("evaluation_sets", "held_out", "artifact"), "held_out-evaluation"),
    ],
)
def test_frontier_report_requires_round_artifact_roles(
    tmp_path: Path, field_path: tuple[str, ...], expected_role: str
) -> None:
    """Typed round artifacts must declare the role their field claims to reference."""
    payload = _evidence(tmp_path)
    target: dict[str, Any] = payload["rounds"][0]
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]]["role"] = "wrong-role"

    field_name = ".".join(("rounds[0]", *field_path))
    with pytest.raises(
        FrontierReportError,
        match=re.escape(f"{field_name}.role must be '{expected_role}'"),
    ):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_uses_later_historical_rows_to_guard_case_identity(
    tmp_path: Path,
) -> None:
    """Historical corpus IDs are known even if their first observation is in a later round."""
    payload = _evidence(tmp_path)
    first = payload["rounds"][0]
    historical_id = "historical-known-case"
    first_search = first["falsification"]
    first_search["candidates"].append(
        _candidate(
            candidate_id="c-reimports-historical",
            evaluation_status="complete",
            verdict="empirically_feasible",
            failure=True,
            replay_status="verified",
            disposition="admitted",
            case_id=historical_id,
            replay_artifact=_artifact(tmp_path, "historical-reimport-replay.json", role="replay"),
            root=tmp_path,
        )
    )
    first_search["budget"]["candidates_completed"] = len(first_search["candidates"])
    first["case_observations"].append(
        _observation(
            tmp_path,
            round_number=1,
            case_id=historical_id,
            origin_round=1,
            origin_candidate_id="c-reimports-historical",
            planner_status="unsolved",
            verdict="empirically_feasible",
            replay_status="verified",
        )
    )
    payload["rounds"][1]["case_observations"].append(
        _observation(
            tmp_path,
            round_number=2,
            case_id=historical_id,
            origin_round=0,
            origin_candidate_id=None,
            planner_status="unknown",
            verdict="admissible_feasibility_unknown",
            replay_status="unavailable",
        )
    )

    with pytest.raises(FrontierReportError, match="re-admits known case"):
        build_frontier_report(payload, evidence_root=tmp_path)


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


@pytest.mark.parametrize(
    ("artifact_kind", "payload_field", "message"),
    [
        ("optimization", "selected_planner_id", "selected planner does not match"),
        ("optimization", "selected_config_identity_sha256", "selected config does not match"),
        ("falsification", "target_planner_id", "target planner does not match"),
        ("falsification", "target_config_identity_sha256", "target config does not match"),
    ],
)
def test_frontier_report_binds_planner_identity_to_optimization_and_search_sources(
    tmp_path: Path, artifact_kind: str, payload_field: str, message: str
) -> None:
    """Checksummed optimizer/search identities must name the declared round planner."""
    payload = _evidence(tmp_path)
    artifact_ref = payload["rounds"][0][artifact_kind]["artifact"]
    source_path = tmp_path / artifact_ref["path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source[payload_field] = "d" * 64
    schema = (
        frontier_module._OPTIMIZER_SELECTION_SCHEMA
        if artifact_kind == "optimization"
        else frontier_module._FALSIFICATION_SOURCE_SCHEMA
    )
    _write_source_artifact(tmp_path, artifact_ref, source, schema_version=schema)

    with pytest.raises(FrontierReportError, match=message):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_binds_evaluation_source_to_round_planner_config(tmp_path: Path) -> None:
    """A checksummed evaluation from another planner configuration is not this round's score."""
    payload = _evidence(tmp_path)
    evaluation_ref = payload["rounds"][0]["evaluation_sets"]["held_out"]["artifact"]
    source_path = tmp_path / evaluation_ref["path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["config_identity_sha256"] = "d" * 64
    _write_source_artifact(
        tmp_path,
        evaluation_ref,
        source,
        schema_version=frontier_module._EVALUATION_SOURCE_SCHEMA,
    )

    with pytest.raises(FrontierReportError, match="artifact config does not match"):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_rejects_search_candidate_rows_not_bound_to_source(
    tmp_path: Path,
) -> None:
    """Candidate outcomes cannot be edited independently of their hashed search source."""
    payload = _evidence(tmp_path)
    payload["rounds"][0]["falsification"]["candidates"][0]["target_failure_observed"] = False

    with pytest.raises(FrontierReportError, match="do not match the checksummed search source"):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_binds_evaluation_metrics_and_ids_to_checksums(tmp_path: Path) -> None:
    """Same-size row substitutions and metric edits cannot pass count-only accounting."""
    payload = _evidence(tmp_path)
    evaluation = payload["rounds"][0]["evaluation_sets"]["held_out"]
    evaluation["episodes"][0]["success"] = False
    with pytest.raises(
        FrontierReportError, match="rows do not match the checksummed evaluation source"
    ):
        build_frontier_report(payload, evidence_root=tmp_path)

    payload = _evidence(tmp_path)
    evaluation = payload["rounds"][0]["evaluation_sets"]["held_out"]
    evaluation["episodes"][0]["record_id"] = "substituted-same-count-episode"
    with pytest.raises(FrontierReportError, match="identity accounting unknown"):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_rejects_held_out_identity_overlap_with_tuning_sets(
    tmp_path: Path,
) -> None:
    """Held-out scenarios cannot share a scenario/seed identity with fixed or regression."""
    payload = _evidence(tmp_path)
    round_data = payload["rounds"][0]
    fixed_identity = round_data["evaluation_sets"]["fixed"]["expected_episode_identities"][0]
    held_out = round_data["evaluation_sets"]["held_out"]
    held_out_row = held_out["episodes"][0]
    held_out_identity = next(
        item
        for item in held_out["expected_episode_identities"]
        if item["record_id"] == held_out_row["record_id"]
    )
    held_out_row["scenario_id"] = fixed_identity["scenario_id"]
    held_out_row["scenario_seed"] = fixed_identity["scenario_seed"]
    held_out_identity["scenario_id"] = fixed_identity["scenario_id"]
    held_out_identity["scenario_seed"] = fixed_identity["scenario_seed"]
    _refresh_source_artifacts(payload, tmp_path)

    with pytest.raises(FrontierReportError, match="held_out overlaps fixed evaluation identities"):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_marks_changed_held_out_cohort_non_comparable(
    tmp_path: Path,
) -> None:
    """A changed hidden cohort is reported but cannot create a connected trend claim."""
    payload = _evidence(tmp_path)
    held_out = payload["rounds"][1]["evaluation_sets"]["held_out"]
    row = held_out["episodes"][0]
    identity = next(
        item
        for item in held_out["expected_episode_identities"]
        if item["record_id"] == row["record_id"]
    )
    row["scenario_id"] = "new-held-out-case"
    row["scenario_seed"] = 9999
    identity["scenario_id"] = "new-held-out-case"
    identity["scenario_seed"] = 9999
    _refresh_source_artifacts(payload, tmp_path)

    report = build_frontier_report(payload, evidence_root=tmp_path)
    first, second = report["rounds"]
    assert first["evaluation_sets"]["held_out"]["cohort_comparison_status"] == "baseline"
    assert second["evaluation_sets"]["held_out"]["cohort_comparison_status"] == (
        "non_comparable_cohort"
    )
    assert second["evaluation_sets"]["held_out"]["cohort_comparable_to_previous_round"] is False
    assert second["evaluation_sets"]["held_out"]["success_rate_comparison_status"] == (
        "non_comparable_cohort"
    )
    assert second["evaluation_sets"]["held_out"]["collision_rate_comparison_status"] == (
        "non_comparable_cohort"
    )
    assert second["evaluation_sets"]["fixed"]["cohort_comparable_to_previous_round"] is True
    assert "cohort comparison `non_comparable_cohort`" in render_frontier_markdown(report)
    axis = Mock()
    frontier_module._plot_evaluation_performance(axis, report["rounds"], [1, 2])
    connecting_segments = [
        call for call in axis.plot.call_args_list if call.kwargs.get("label") == "_nolegend_"
    ]
    assert len(connecting_segments) == 2
    write_frontier_figure(report, tmp_path / "changed-cohort-frontier")


def test_frontier_report_omits_success_trend_when_eligible_sample_changes(
    tmp_path: Path,
) -> None:
    """Same planned cohort does not imply comparable rates when eligible denominators change."""
    payload = _evidence(tmp_path)
    payload["rounds"][1]["evaluation_sets"]["fixed"]["episodes"][0]["eligible"] = False
    _refresh_source_artifacts(payload, tmp_path)

    report = build_frontier_report(payload, evidence_root=tmp_path)
    fixed_summary = report["rounds"][1]["evaluation_sets"]["fixed"]
    assert fixed_summary["cohort_comparison_status"] == "comparable"
    assert fixed_summary["success_rate_comparison_status"] == "non_comparable_success_sample"
    assert fixed_summary["success_rate_comparable_to_previous_round"] is False
    assert fixed_summary["collision_rate_comparison_status"] == "non_comparable_collision_sample"
    assert fixed_summary["collision_rate_comparable_to_previous_round"] is False

    axis = Mock()
    frontier_module._plot_evaluation_performance(axis, report["rounds"], [1, 2])
    connecting_segments = [
        call for call in axis.plot.call_args_list if call.kwargs.get("label") == "_nolegend_"
    ]
    assert len(connecting_segments) == 2


def test_frontier_report_marks_collision_rates_non_comparable_when_outcome_sample_changes(
    tmp_path: Path,
) -> None:
    """The same cohort does not make collision rates comparable when observed rows differ."""
    payload = _evidence(tmp_path)
    first_rows = payload["rounds"][0]["evaluation_sets"]["held_out"]["episodes"]
    second_rows = payload["rounds"][1]["evaluation_sets"]["held_out"]["episodes"]
    first_rows[0]["collision"] = False
    first_rows[1]["collision"] = None
    second_rows[0]["collision"] = None
    second_rows[1]["collision"] = True
    _refresh_source_artifacts(payload, tmp_path)

    report = build_frontier_report(payload, evidence_root=tmp_path)
    first, second = (item["evaluation_sets"]["held_out"] for item in report["rounds"])
    assert first["collision_denominator"] == second["collision_denominator"] == 1
    assert first["collision_rate"] == 0.0
    assert second["collision_rate"] == 1.0
    assert first["collision_rate_sample_sha256"] != second["collision_rate_sample_sha256"]
    assert second["cohort_comparison_status"] == "comparable"
    assert second["success_rate_comparison_status"] == "comparable"
    assert second["collision_rate_comparable_to_previous_round"] is False
    assert second["collision_rate_comparison_status"] == "non_comparable_collision_sample"

    markdown = render_frontier_markdown(report)
    assert "collision-rate comparison `non_comparable_collision_sample`" in markdown
    assert "collision-rate comparison `baseline`" in markdown
    assert "0.000 (0/1) collision" in markdown
    assert "1.000 (1/1)* collision" in markdown


def test_frontier_report_does_not_compare_collision_rates_without_outcomes(
    tmp_path: Path,
) -> None:
    """An empty collision sample remains unknown, even when both rounds match."""
    payload = _evidence(tmp_path)
    for round_data in payload["rounds"]:
        for row in round_data["evaluation_sets"]["held_out"]["episodes"]:
            row["collision"] = None
    _refresh_source_artifacts(payload, tmp_path)

    report = build_frontier_report(payload, evidence_root=tmp_path)
    summary = report["rounds"][1]["evaluation_sets"]["held_out"]
    assert summary["collision_denominator"] == 0
    assert summary["collision_rate"] is None
    assert summary["collision_rate_comparable_to_previous_round"] is False
    assert summary["collision_rate_comparison_status"] == "no_collision_outcomes"
    assert "collision-rate comparison `no_collision_outcomes`" in render_frontier_markdown(report)


def test_frontier_report_rejects_fixed_cohort_manifest_drift(tmp_path: Path) -> None:
    """Fixed evaluation comparisons retain the same scenario/seed cohort across rounds."""
    payload = _evidence(tmp_path)
    fixed = payload["rounds"][1]["evaluation_sets"]["fixed"]
    row = fixed["episodes"][0]
    identity = next(
        item
        for item in fixed["expected_episode_identities"]
        if item["record_id"] == row["record_id"]
    )
    row["scenario_id"] = "replaced-fixed-case"
    identity["scenario_id"] = "replaced-fixed-case"
    _refresh_source_artifacts(payload, tmp_path)

    with pytest.raises(FrontierReportError, match="fixed evaluation scenario/seed cohort changed"):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_keeps_prior_held_out_cases_out_of_regression(
    tmp_path: Path,
) -> None:
    """A former held-out case cannot enter a later optimization-facing regression set."""
    payload = _evidence(tmp_path)
    first_held_out = payload["rounds"][0]["evaluation_sets"]["held_out"]
    held_out_identity = first_held_out["expected_episode_identities"][0]
    second_sets = payload["rounds"][1]["evaluation_sets"]
    changed_held_out = second_sets["held_out"]
    changed_row = changed_held_out["episodes"][0]
    changed_identity = changed_held_out["expected_episode_identities"][0]
    changed_row["scenario_id"] = changed_identity["scenario_id"] = "replacement-held-out"
    changed_row["scenario_seed"] = changed_identity["scenario_seed"] = 9900
    regression = second_sets["regression"]
    regression_row = regression["episodes"][0]
    regression_identity = regression["expected_episode_identities"][0]
    regression_row["scenario_id"] = regression_identity["scenario_id"] = held_out_identity[
        "scenario_id"
    ]
    regression_row["scenario_seed"] = regression_identity["scenario_seed"] = held_out_identity[
        "scenario_seed"
    ]
    _refresh_source_artifacts(payload, tmp_path)

    with pytest.raises(
        FrontierReportError, match="held_out overlaps regression evaluation identities"
    ):
        build_frontier_report(payload, evidence_root=tmp_path)


@pytest.mark.parametrize("malformed_status", [{}, []])
def test_frontier_report_wraps_unhashable_candidate_status_as_report_error(
    tmp_path: Path, malformed_status: Any
) -> None:
    """Malformed enum values fail through the public report error contract."""
    payload = _evidence(tmp_path)
    payload["rounds"][0]["falsification"]["candidates"][0]["evaluation_status"] = malformed_status

    with pytest.raises(FrontierReportError, match="evaluation_status is unsupported"):
        build_frontier_report(payload, evidence_root=tmp_path)


@pytest.mark.parametrize("malformed_verdict", [[], {}])
@pytest.mark.parametrize("record_kind", ["candidate", "observation"])
def test_frontier_report_wraps_unhashable_admissibility_verdict_as_report_error(
    tmp_path: Path, record_kind: str, malformed_verdict: Any
) -> None:
    """Malformed JSON verdicts fail the report contract without leaking TypeError."""
    payload = _evidence(tmp_path)
    records = (
        payload["rounds"][0]["falsification"]["candidates"]
        if record_kind == "candidate"
        else payload["rounds"][0]["case_observations"]
    )
    records[0]["admissibility_verdict"] = malformed_verdict
    _refresh_source_artifacts(payload, tmp_path)

    with pytest.raises(FrontierReportError, match="admissibility_verdict is unsupported"):
        build_frontier_report(payload, evidence_root=tmp_path)


@pytest.mark.parametrize("malformed_outcome", [[], {}])
def test_frontier_report_wraps_unhashable_target_planner_outcome_as_report_error(
    tmp_path: Path, malformed_outcome: Any
) -> None:
    """Malformed source categorical values are diagnosed through FrontierReportError."""
    payload = _evidence(tmp_path)
    _rewrite_admissibility_artifact(
        payload,
        tmp_path,
        0,
        lambda source: source.update(target_planner_outcome=malformed_outcome),
    )

    with pytest.raises(FrontierReportError, match="target_planner_outcome is unsupported"):
        build_frontier_report(payload, evidence_root=tmp_path)


def _malformed_enum_evidence(
    tmp_path: Path, category: str, malformed_value: Any
) -> tuple[dict[str, Any], str]:
    payload = _evidence(tmp_path)
    if category == "planner_status":
        payload["rounds"][0]["case_observations"][0]["planner_status"] = malformed_value
        message = "planner_status is unsupported"
        _refresh_source_artifacts(payload, tmp_path)
    elif category == "corpus_disposition":
        payload["rounds"][0]["falsification"]["candidates"][0]["corpus_disposition"] = (
            malformed_value
        )
        message = "corpus_disposition is unsupported"
        _refresh_source_artifacts(payload, tmp_path)
    elif category == "reference_planner_id":
        _rewrite_admissibility_artifact(
            payload,
            tmp_path,
            0,
            lambda source: source["evidence"]["reference_execution"].update(
                planner_id=malformed_value,
                planner_checkpoint_sha256="not_applicable",
            ),
        )
        message = "reference_execution.planner_id must be non-empty text"
    else:
        raise AssertionError(f"unsupported test category: {category}")
    return payload, message


@pytest.mark.parametrize("malformed_value", [[], {}])
@pytest.mark.parametrize(
    "category", ["planner_status", "corpus_disposition", "reference_planner_id"]
)
def test_frontier_report_wraps_unhashable_round_and_execution_categoricals(
    tmp_path: Path, category: str, malformed_value: Any
) -> None:
    """Malformed JSON enum and planner fields stay in the structured report error channel."""
    payload, message = _malformed_enum_evidence(tmp_path, category, malformed_value)

    with pytest.raises(FrontierReportError, match=message):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_cli_returns_structured_diagnostic_for_malformed_verdict(
    tmp_path: Path,
) -> None:
    """JSON-valid malformed categories stay in the CLI's structured error channel."""
    payload = _evidence(tmp_path)
    payload["rounds"][0]["falsification"]["candidates"][0]["admissibility_verdict"] = []
    _refresh_source_artifacts(payload, tmp_path)
    input_path = tmp_path / "malformed-round-evidence.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    output_dir = tmp_path / "malformed-report"
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts/tools/build_adversarial_feasibility_frontier_report.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input",
            str(input_path),
            "--out-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    diagnostic = json.loads(result.stderr)
    assert result.returncode == 2
    assert result.stdout == ""
    assert diagnostic["status"] == "error"
    assert "invalid frontier evidence" in diagnostic["error"]


@pytest.mark.parametrize(
    "category", ["planner_status", "corpus_disposition", "reference_planner_id"]
)
@pytest.mark.parametrize("malformed_value", [[], {}])
def test_frontier_report_cli_returns_exit_two_for_unhashable_categoricals(
    tmp_path: Path, category: str, malformed_value: Any
) -> None:
    """Malformed categories are reported as JSON diagnostics instead of tracebacks."""
    payload, message = _malformed_enum_evidence(tmp_path, category, malformed_value)
    input_path = tmp_path / f"malformed-{category}-{type(malformed_value).__name__}.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts/tools/build_adversarial_feasibility_frontier_report.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input",
            str(input_path),
            "--out-dir",
            str(tmp_path / f"malformed-{category}-report-{type(malformed_value).__name__}"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    diagnostic = json.loads(result.stderr)
    assert result.returncode == 2
    assert result.stdout == ""
    assert diagnostic["status"] == "error"
    assert message in diagnostic["error"]


def test_frontier_report_rejects_malformed_evaluation_record_id(tmp_path: Path) -> None:
    """Malformed row identities must produce a report error, not a Python TypeError."""
    payload = _evidence(tmp_path)
    payload["rounds"][0]["evaluation_sets"]["held_out"]["episodes"][0]["record_id"] = {}

    with pytest.raises(FrontierReportError, match=r"record_id must be non-empty text"):
        build_frontier_report(payload, evidence_root=tmp_path)

    payload = _evidence(tmp_path)
    evaluation_ref = payload["rounds"][0]["evaluation_sets"]["held_out"]["artifact"]
    source_path = tmp_path / evaluation_ref["path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["expected_episode_ids"][1] = "substituted-source-episode"
    _write_source_artifact(
        tmp_path,
        evaluation_ref,
        source,
        schema_version=frontier_module._EVALUATION_SOURCE_SCHEMA,
    )
    with pytest.raises(FrontierReportError, match="identity accounting unknown"):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_fails_closed_when_evaluation_source_has_no_episode_ids(
    tmp_path: Path,
) -> None:
    """Missing canonical episode identities remain explicitly unknown, not count-complete."""
    payload = _evidence(tmp_path)
    del payload["rounds"][0]["evaluation_sets"]["held_out"]["expected_episode_ids"]

    with pytest.raises(FrontierReportError, match="identity accounting unknown"):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_preserves_missing_expected_episode_identities(tmp_path: Path) -> None:
    """Known missing episodes stay identified and make row accounting incomplete."""
    payload = _evidence(tmp_path)
    evaluation = payload["rounds"][0]["evaluation_sets"]["held_out"]
    evaluation["episodes"].pop(1)
    _refresh_source_artifacts(payload, tmp_path)

    summary = build_frontier_report(payload, evidence_root=tmp_path)["rounds"][0][
        "evaluation_sets"
    ]["held_out"]
    assert summary["identity_accounting_status"] == "verified"
    assert summary["accounting_complete"] is False
    assert summary["missing_record_count"] == 1
    assert summary["missing_record_ids"] == ["held_out-1-2"]


@pytest.mark.parametrize("artifact_field", ["origin_search_artifact", "replay_artifact"])
def test_frontier_report_binds_case_observation_refs_to_origin_evidence(
    tmp_path: Path, artifact_field: str
) -> None:
    """Valid artifacts from another round/case cannot be substituted as origin evidence."""
    payload = _evidence(tmp_path)
    observation = payload["rounds"][1]["case_observations"][0]
    if artifact_field == "origin_search_artifact":
        observation[artifact_field] = payload["rounds"][1]["falsification"]["artifact"]
        message = "origin search artifact does not match"
    else:
        observation[artifact_field] = payload["rounds"][1]["falsification"]["candidates"][0][
            "replay_artifact"
        ]
        message = "replay artifact does not match its origin candidate"

    with pytest.raises(FrontierReportError, match=message):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_binds_corpus_status_artifact_to_case_identity(
    tmp_path: Path,
) -> None:
    """A valid digest for a corpus row naming another case cannot support this status."""
    payload = _evidence(tmp_path)
    observation = payload["rounds"][0]["case_observations"][0]
    reference = observation["corpus_artifact"]
    source = json.loads((tmp_path / reference["path"]).read_text(encoding="utf-8"))
    source["case_id"] = "different-case"
    _write_source_artifact(
        tmp_path,
        reference,
        source,
        schema_version=frontier_module._CORPUS_CASE_STATUS_SCHEMA,
    )

    with pytest.raises(FrontierReportError, match="corpus_artifact case/status identity"):
        build_frontier_report(payload, evidence_root=tmp_path)


@pytest.mark.parametrize(
    ("scenario_path", "message"),
    [
        ("fixture/missing-scenario.yaml", "does not resolve to an available file"),
        ("../outside-scenario.yaml", "must stay relative to the evidence bundle"),
    ],
)
def test_frontier_report_rejects_unavailable_or_escaping_scenario_artifact_paths(
    tmp_path: Path, scenario_path: str, message: str
) -> None:
    """Scenario identities must resolve to bytes inside the evidence bundle."""
    payload = _evidence(tmp_path)

    def replace_scenario_path(record: dict[str, Any]) -> None:
        record["evidence"]["scenario_artifact_identity"]["path"] = scenario_path

    _rewrite_admissibility_artifact(payload, tmp_path, 0, replace_scenario_path)

    with pytest.raises(FrontierReportError, match=message):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_binds_scenario_artifact_path_to_captured_bytes(tmp_path: Path) -> None:
    """Changing scenario bytes while retaining declared digests invalidates the case."""
    payload = _evidence(tmp_path)
    candidate = payload["rounds"][0]["falsification"]["candidates"][0]
    admissibility = json.loads(
        (tmp_path / candidate["admissibility_evidence_artifact"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    relative_path = admissibility["evidence"]["scenario_artifact_identity"]["path"]
    (tmp_path / relative_path).write_bytes(b"scenario_id: tampered-after-capture\n")

    with pytest.raises(FrontierReportError, match="path bytes do not match"):
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
    first_pdf = (output_dir / "frontier.pdf").read_bytes()
    first_sidecar_bytes = (output_dir / "frontier.provenance.json").read_bytes()
    first_sidecar = json.loads(first_sidecar_bytes.decode("utf-8"))
    second = write_frontier_report(input_path, second_output_dir)

    assert first == second
    assert (second_output_dir / "frontier_report.json").read_bytes() == first_json
    assert (second_output_dir / "frontier_report.md").read_bytes() == first_markdown
    assert (second_output_dir / "frontier.pdf").read_bytes() == first_pdf
    assert b"D:20000101000000" in first_pdf
    assert (second_output_dir / "frontier.provenance.json").read_bytes() == first_sidecar_bytes
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
    assert sidecar["repo_commit"] == _git_sha_short()
    assert sidecar["source_revision"] == _REVISION
    assert sidecar["evidence_kind"] == "synthetic_fixture"
    assert sidecar["figure_title"].startswith("Synthetic Fixture evidence")
    assert sidecar["claim_boundary"] == first["claim_boundary"]
    assert sidecar["source_artifacts"][-1]["path"] == input_path.name

    with pytest.raises(FrontierReportError, match="choose a new output directory"):
        write_frontier_report(input_path, output_dir)


@pytest.mark.parametrize(
    ("evidence_kind", "headline"),
    [
        (
            "synthetic_fixture",
            "Synthetic fixture (implementation-only) feasibility-frontier report",
        ),
        (
            "simulator_run",
            "Declared simulator-run evidence (unverified) feasibility-frontier report",
        ),
        ("historical_artifact", "Historical-artifact feasibility-frontier report"),
    ],
)
def test_frontier_markdown_headline_matches_evidence_kind(
    tmp_path: Path, evidence_kind: str, headline: str
) -> None:
    """Headlines preserve the declared kind without implying verified provenance."""
    evidence = _evidence(tmp_path)
    evidence["evidence_kind"] = evidence_kind

    report = build_frontier_report(evidence, evidence_root=tmp_path)

    assert render_frontier_markdown(report).splitlines()[0] == (f"# {headline}: {_EXPERIMENT_ID}")


def test_relabeling_synthetic_fixture_as_simulator_run_does_not_claim_empirical_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller-edited evidence kind cannot claim simulator provenance in any output."""
    from PIL import Image

    evidence = _evidence(tmp_path)
    assert evidence["evidence_kind"] == "synthetic_fixture"

    evidence["evidence_kind"] = "simulator_run"
    input_path = tmp_path / "relabelled-round-evidence.json"
    input_path.write_text(json.dumps(evidence), encoding="utf-8")
    output_dir = tmp_path / "relabelled-report"
    captured: dict[str, str] = {}
    save_figure = frontier_module.save_publication_figure

    def capture_figure_title(figure: Any, output_base: Path, **kwargs: Any) -> list[Path]:
        captured["figure_title"] = figure._suptitle.get_text()
        return save_figure(figure, output_base, **kwargs)

    monkeypatch.setattr(frontier_module, "save_publication_figure", capture_figure_title)
    report = write_frontier_report(input_path, output_dir)
    markdown = (output_dir / "frontier_report.md").read_text(encoding="utf-8")
    persisted_report = json.loads((output_dir / "frontier_report.json").read_text(encoding="utf-8"))
    sidecar = json.loads((output_dir / "frontier.provenance.json").read_text(encoding="utf-8"))
    with Image.open(output_dir / "frontier.png") as figure_image:
        embedded_provenance = json.loads(figure_image.info["Provenance"])

    assert "Declared simulator-run evidence (unverified)" in markdown.splitlines()[0]
    assert "Empirical feasibility frontier" not in markdown.splitlines()[0]
    assert "simulator_run is unverified" in report["claim_boundary"]
    assert persisted_report["claim_boundary"] == report["claim_boundary"]
    assert captured["figure_title"].startswith("Declared Simulator Run evidence (unverified)")
    assert sidecar["figure_title"] == captured["figure_title"]
    assert sidecar["claim_boundary"] == report["claim_boundary"]
    assert "simulator_run is unverified" in embedded_provenance["claim_boundary"]


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


def test_frontier_report_requires_admissibility_evidence_for_confirmed_observations(
    tmp_path: Path,
) -> None:
    """A confirmed corpus row cannot bypass its case-bound #9651 verdict artifact."""
    payload = _evidence(tmp_path)
    payload["rounds"][0]["case_observations"][0]["admissibility_evidence_artifact"] = None

    with pytest.raises(FrontierReportError, match="required for every case observation"):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_rejects_admissibility_artifact_for_another_case(
    tmp_path: Path,
) -> None:
    """A role and digest do not let one case reuse another case's admissibility verdict."""
    payload = _evidence(tmp_path)
    reference = payload["rounds"][0]["falsification"]["candidates"][0][
        "admissibility_evidence_artifact"
    ]
    artifact_path = tmp_path / reference["path"]
    record = json.loads(artifact_path.read_text(encoding="utf-8"))
    record["case_id"] = "different-case"
    content = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    artifact_path.write_bytes(content)
    reference["sha256"] = hashlib.sha256(content).hexdigest()
    _refresh_source_artifacts(payload, tmp_path)

    with pytest.raises(FrontierReportError, match="case_id does not match"):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_rejects_generic_admissibility_artifact_content(tmp_path: Path) -> None:
    """A checksummed role label without the structured verdict cannot confirm feasibility."""
    payload = _evidence(tmp_path)
    reference = payload["rounds"][0]["falsification"]["candidates"][0][
        "admissibility_evidence_artifact"
    ]
    artifact_path = tmp_path / reference["path"]
    content = b"persisted fixture artifact with no admissibility record\n"
    artifact_path.write_bytes(content)
    reference["sha256"] = hashlib.sha256(content).hexdigest()
    _refresh_source_artifacts(payload, tmp_path)

    with pytest.raises(FrontierReportError, match="could not parse artifact JSON"):
        build_frontier_report(payload, evidence_root=tmp_path)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("scenario_id",), "unrelated-scenario", "scenario_id does not match"),
        (
            ("evidence", "target_execution", "planner_id"),
            "unrelated-target",
            "target_execution planner/config does not match",
        ),
        (
            ("evidence", "target_execution", "planner_config_sha256"),
            "0" * 64,
            "target_execution planner/config does not match",
        ),
        (
            ("evidence", "target_execution", "source_commit"),
            "c" * 40,
            "target_execution.source_commit does not match",
        ),
        (
            ("evidence", "target_execution", "scenario_sha256"),
            "0" * 64,
            "target_execution.scenario_sha256 does not match",
        ),
    ],
)
def test_frontier_report_rejects_unbound_confirmed_execution_evidence(
    tmp_path: Path, path: tuple[str, ...], value: str, message: str
) -> None:
    """Confirmed feasibility cannot use a different scenario, revision, or planner."""
    payload = _evidence(tmp_path)

    def mutate(record: dict[str, Any]) -> None:
        target = record
        for field in path[:-1]:
            target = target[field]
        target[path[-1]] = value

    _rewrite_admissibility_artifact(payload, tmp_path, 0, mutate)
    with pytest.raises(FrontierReportError, match=message):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_rejects_route_boolean_without_producer_execution_provenance(
    tmp_path: Path,
) -> None:
    """A route-completion boolean cannot substitute for a named, bound execution record."""
    payload = _evidence(tmp_path)

    def remove_target_provenance(record: dict[str, Any]) -> None:
        record["evidence"]["target_execution"] = {
            "case_id": record["case_id"],
            "scenario_id": record["scenario_id"],
            "route_complete": False,
        }

    _rewrite_admissibility_artifact(payload, tmp_path, 0, remove_target_provenance)
    with pytest.raises(FrontierReportError, match="missing producer fields"):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_rejects_verified_candidate_without_replay_execution(
    tmp_path: Path,
) -> None:
    """A generic checksummed replay file cannot substitute for replay execution evidence."""
    payload = _evidence(tmp_path)

    def remove_replay_execution(record: dict[str, Any]) -> None:
        record["evidence"].pop("replay_execution")

    _rewrite_admissibility_artifact(payload, tmp_path, 0, remove_replay_execution)
    with pytest.raises(FrontierReportError, match="verified replay requires complete"):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_binds_replay_execution_to_outer_replay_artifact(
    tmp_path: Path,
) -> None:
    """The admissibility replay record must point at the linked checksummed replay artifact."""
    payload = _evidence(tmp_path)

    def mismatch_replay_reference(record: dict[str, Any]) -> None:
        record["evidence"]["replay_execution"]["evidence_ref"] = "fixture/unrelated-replay.json"

    _rewrite_admissibility_artifact(payload, tmp_path, 0, mismatch_replay_reference)
    with pytest.raises(
        FrontierReportError, match="evidence_ref does not match replay_artifact.path"
    ):
        build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_binds_planner_specific_failure_to_target_and_replay(
    tmp_path: Path,
) -> None:
    """The reference success and replayed target failure must share exact run identity."""
    payload = _evidence(tmp_path)
    candidate = payload["rounds"][0]["falsification"]["candidates"][0]
    observation = payload["rounds"][0]["case_observations"][0]
    candidate["admissibility_verdict"] = "planner_specific_failure"
    observation["admissibility_verdict"] = "planner_specific_failure"
    reference = _admissibility_artifact(
        tmp_path,
        "round-1-planner-specific-failure.json",
        case_id="case-001",
        verdict="planner_specific_failure",
        replay_artifact_path=candidate["replay_artifact"]["path"],
    )
    candidate["admissibility_evidence_artifact"] = reference
    observation["admissibility_evidence_artifact"] = reference
    followup = payload["rounds"][1]["case_observations"][0]
    followup["planner_status"] = "unsolved"
    followup["admissibility_verdict"] = "planner_specific_failure"
    followup["admissibility_evidence_artifact"] = _admissibility_artifact(
        tmp_path,
        "round-2-planner-specific-failure.json",
        case_id="case-001",
        verdict="planner_specific_failure",
        planner_config_sha256="c" * 64,
    )
    _refresh_source_artifacts(payload, tmp_path)
    report = build_frontier_report(payload, evidence_root=tmp_path)
    assert report["rounds"][0]["falsification"]["verified_counterexample_case_ids"] == ["case-001"]

    def mismatch_replay(record: dict[str, Any]) -> None:
        record["evidence"]["replay_execution"]["episode_id"] = "unrelated-episode"

    _rewrite_admissibility_artifact(payload, tmp_path, 0, mismatch_replay)
    with pytest.raises(FrontierReportError, match="matched planner-specific failure evidence"):
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
            scenario_identity=("held_out-scenario-degraded", 4404),
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
            scenario_identity=("held_out-scenario-partial", 5505),
            success=True,
            collision=False,
            execution_mode="adapter",
            readiness_status="adapter",
            availability_status="partial-failure",
            eligible=True,
        )
    )
    evaluation["expected_episode_count"] = len(rows)

    _refresh_source_artifacts(payload, tmp_path)
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

    _refresh_source_artifacts(payload, tmp_path)
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
            root=tmp_path,
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
        later_observation["admissibility_evidence_artifact"] = _admissibility_artifact(
            tmp_path,
            "round-2-case-origin-unknown-feasibility.json",
            case_id="case-origin-unknown",
            verdict="empirically_feasible",
            target_route_complete=True,
        )
    else:
        later_observation["admissibility_evidence_artifact"] = None
    payload["rounds"][1]["case_observations"].append(later_observation)

    if include_evidence:
        third = _round(tmp_path, 3)
        third["falsification"]["candidates"] = []
        third["falsification"]["budget"]["candidates_completed"] = 0
        third["case_observations"] = [
            _observation(
                tmp_path,
                round_number=3,
                case_id="case-origin-unknown",
                origin_round=1,
                origin_candidate_id="c-unknown-origin",
                planner_status="solved",
                verdict="empirically_feasible",
                replay_status="verified",
            )
        ]
        payload["rounds"].append(third)
        _refresh_source_artifacts(payload, tmp_path)
        report = build_frontier_report(payload, evidence_root=tmp_path)
        second = report["rounds"][1]
        third_report = report["rounds"][2]
        assert second["falsification"]["feasibility_upgrades_from_follow_up_case_ids"] == [
            "case-origin-unknown"
        ]
        assert second["case_frontier"]["verified_counterexamples_cumulative"] == 2
        assert second["case_frontier"]["verified_counterexample_status"]["solved"] == 2
        assert second["case_frontier"]["current_unknown_feasibility_case_count"] == 1
        assert second["case_frontier"]["admitted_unknown_feasibility_cases_cumulative"] == 2
        assert third_report["falsification"]["feasibility_upgrades_from_follow_up_case_ids"] == []
        assert third_report["case_frontier"]["verified_counterexamples_cumulative"] == 2
        assert third_report["falsification"]["no_verified_counterexample_statement"]
    else:
        with pytest.raises(FrontierReportError, match="unsupported admissibility-verdict"):
            build_frontier_report(payload, evidence_root=tmp_path)


def test_frontier_report_rejects_feasibility_downgrade_after_confirmation(tmp_path: Path) -> None:
    """After an unknown-to-feasible transition, later evidence cannot downgrade the verdict."""
    payload = _evidence(tmp_path)
    first = payload["rounds"][0]
    first["falsification"]["candidates"].append(
        _candidate(
            candidate_id="c-unknown-origin",
            evaluation_status="complete",
            verdict="admissible_feasibility_unknown",
            failure=True,
            replay_status="verified",
            disposition="admitted",
            case_id="case-origin-unknown",
            replay_artifact=_artifact(
                tmp_path, "round-1-unknown-downgrade-replay.json", role="replay"
            ),
            root=tmp_path,
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
    second = payload["rounds"][1]
    upgraded = _observation(
        tmp_path,
        round_number=2,
        case_id="case-origin-unknown",
        origin_round=1,
        origin_candidate_id="c-unknown-origin",
        planner_status="solved",
        verdict="empirically_feasible",
        replay_status="verified",
    )
    upgraded["admissibility_evidence_artifact"] = _admissibility_artifact(
        tmp_path,
        "round-2-unknown-downgrade-proof.json",
        case_id="case-origin-unknown",
        verdict="empirically_feasible",
        planner_config_sha256="c" * 64,
        target_route_complete=True,
        replay_artifact_path=upgraded["replay_artifact"]["path"],
    )
    second["case_observations"].append(upgraded)
    third = _round(tmp_path, 3)
    third["falsification"]["candidates"] = []
    third["falsification"]["budget"]["candidates_completed"] = 0
    third["case_observations"] = [
        _observation(
            tmp_path,
            round_number=3,
            case_id="case-origin-unknown",
            origin_round=1,
            origin_candidate_id="c-unknown-origin",
            planner_status="unknown",
            verdict="admissible_feasibility_unknown",
            replay_status="verified",
        )
    ]
    payload["rounds"].append(third)

    with pytest.raises(FrontierReportError, match="unsupported admissibility-verdict transition"):
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
                candidate["replay_status"] = "not_attempted"
                candidate["replay_artifact"] = None
                candidate["admissibility_evidence_artifact"] = None
        round_data["case_observations"] = []
        for evaluation in round_data["evaluation_sets"].values():
            evaluation["episodes"][0]["success"] = True
            evaluation["episodes"][1]["success"] = False

    _refresh_source_artifacts(payload, tmp_path)
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


def test_frontier_figure_labels_planner_and_scenario_dispositions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The figure distinguishes current planner outcomes and invalid/infeasible candidates."""
    report = build_frontier_report(_evidence(tmp_path), evidence_root=tmp_path)
    captured: dict[str, Any] = {}

    def capture_figure(figure: Any, _output_base: Path, **_kwargs: Any) -> list[Path]:
        captured["legend_labels"] = [
            item.get_text() for item in figure.axes[1].get_legend().get_texts()
        ]
        captured["title"] = figure._suptitle.get_text()
        return []

    monkeypatch.setattr(frontier_module, "save_publication_figure", capture_figure)
    write_frontier_figure(report, tmp_path / "frontier")

    labels = captured["legend_labels"]
    assert "Known cases solved by current planner" in labels
    assert "Known cases still failing for current planner" in labels
    assert "Known cases with mixed planner outcomes" in labels
    assert "Known cases with unknown planner outcome" in labels
    assert "Known cases not observed in round" in labels
    assert "Cases currently with unknown feasibility" in labels
    assert "Structurally invalid search candidates (round)" in labels
    assert "Geometric/kinodynamic impossibilities (round)" in labels
    assert "Synthetic Fixture evidence" in captured["title"]
