"""Fixture-backed tests for the versioned adversarial challenge corpus."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robot_sf.adversarial import counterexample_corpus
from robot_sf.adversarial.counterexample_corpus import (
    CorpusError,
    append_planner_evaluation,
    create_case_admission_replay_receipt,
    create_planner_replay_receipt,
    export_regression_slice,
    import_issue9645_packet,
    import_issue9656_candidates,
    new_corpus,
    promote_historical_candidate,
    recompute_planner_status,
    save_corpus,
    validate_corpus,
)
from robot_sf.benchmark.episode_input_identity import scenario_semantic_sha256
from scripts.tools.manage_adversarial_counterexample_corpus import main as corpus_cli_main

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PACKET = _REPO_ROOT / "tests/fixtures/adversarial_counterexample_corpus/issue_9645/payload"
_SOURCE_BUNDLE = _SOURCE_PACKET.parent
_ISSUE9656_SOURCE_REVISION = "f7ebdcae2375d085e925213197a75a386e26a79c"
_ISSUE9656_SOURCE_MATRIX = "configs/scenarios/classic_interactions_francis2023.yaml"
_ISSUE9656_SOURCE_MATRIX_SHA256 = "d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5"


def _issue9656_candidate_fixture(
    tmp_path: Path,
    statuses: tuple[str, ...] = (
        "not_attempted",
        "unavailable_model_artifact",
        "mismatch_different_revision",
    ),
    *,
    promotion_case: bool = False,
) -> tuple[Path, Path, Path, Path]:
    """Build a digest-bound miniature #9656 evidence and materialization bundle."""
    bundle_root = tmp_path / "issue_9656_bundle"
    payload = bundle_root / "payload"
    payload.mkdir(parents=True)
    materialized = tmp_path / "materialized"
    campaign_root = tmp_path / "campaign"
    manifest_cases: list[dict[str, object]] = []
    summary_cases: list[dict[str, object]] = []
    source_status_counts: dict[str, int] = {}
    anomaly_counts: dict[str, int] = {}
    map_path = _REPO_ROOT / "maps/svg_maps/classic_crossing.svg"
    for index, status in enumerate(statuses):
        summary_case = _build_issue9656_fixture_candidate(
            index, status, materialized, campaign_root, map_path, promotion_case
        )
        summary_cases.append(summary_case)
        manifest_cases.append(
            {"case_file": f"cases/{summary_case['case_id']}/case.json", **summary_case}
        )
        source_status_counts[status] = source_status_counts.get(status, 0) + 1
        for item in summary_case["criticality"]["anomalies"]:
            anomaly_counts[item] = anomaly_counts.get(item, 0) + 1

    attempts = [
        {
            "case_id": summary_cases[0]["case_id"],
            "return_code": 2,
            "replay_row_count": 0,
            "scheduled_jobs": 3,
        }
    ]
    summary = {
        "schema_version": "benchmark-hard-case-slice.v1",
        "status": "materialized",
        "evidence_tier": "diagnostic_only",
        "claim_boundary": "historical records and finite replays only",
        "source": {
            "source_revision": _ISSUE9656_SOURCE_REVISION,
            "source_campaign_id": "campaign-fixture",
            "bundle_sha256": "a" * 64,
            "matrix_path": _ISSUE9656_SOURCE_MATRIX,
            "matrix_sha256": _ISSUE9656_SOURCE_MATRIX_SHA256,
        },
        "selection": {
            "case_count": len(summary_cases),
            "case_ids": [case["case_id"] for case in summary_cases],
        },
        "cases": summary_cases,
        "replay": {"status_counts": source_status_counts},
        "criticality_anomaly_counts": anomaly_counts,
        "replay_budget_accounting": {"failed_setup_jobs": 12},
        "failed_replay_attempts": {"attempts": attempts},
    }
    summary_path = payload / "summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    report_path = payload / "report.md"
    report_path.write_text("fixture report\n", encoding="utf-8")
    manifest_entries = []
    checksum_lines = []
    for path in (summary_path, report_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_entries.append(
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": digest}
        )
        checksum_lines.append(f"{digest}  payload/{path.name}")
    bundle_manifest = {
        "schema_version": "evidence_bundle.v1",
        "files": manifest_entries,
        "totals": {
            "file_count": len(manifest_entries),
            "total_bytes": sum(item["size_bytes"] for item in manifest_entries),
        },
    }
    (bundle_root / "evidence_bundle_manifest.json").write_text(
        json.dumps(bundle_manifest, sort_keys=True), encoding="utf-8"
    )
    (bundle_root / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")
    materialized_manifest = {
        "schema_version": "benchmark-hard-case-slice.v1",
        "cases": manifest_cases,
        "source": summary["source"],
    }
    (materialized / "manifest.json").write_text(
        json.dumps(materialized_manifest, sort_keys=True), encoding="utf-8"
    )
    return summary_path, materialized, campaign_root, bundle_root


def _build_issue9656_fixture_candidate(
    index: int,
    status: str,
    materialized: Path,
    campaign_root: Path,
    map_path: Path,
    promotion_case: bool,
) -> dict[str, object]:
    case_id = f"case-{index + 1:016x}"
    matching_episode, matching_scenario = _issue9656_fixture_matching_source(index, promotion_case)
    source_row, planner_config_identity, source_record = _issue9656_fixture_episode(
        index, matching_episode, campaign_root
    )
    planner_key = source_row["algo"]
    canonical_algorithm = source_row["algorithm_metadata"]["canonical_algorithm"]
    case_dir = materialized / "cases" / case_id
    replay_input = _write_issue9656_fixture_replay_inputs(
        index, status, case_dir, map_path, source_row, matching_episode, matching_scenario
    )
    anomaly = ["collision_event_without_positive_collision_metric"] if index == 0 else []
    criticality = {
        "anomalies": anomaly,
        "evidence_tier": "diagnostic_only",
        "metrics": {"collisions_metric": 0.0, "minimum_clearance_m": 0.5 + index},
        "outcome": {"collision_event": False, "route_complete": False, "timeout_event": True},
    }
    replay = {"attempted": status == "mismatch_different_revision", "status": status}
    summary_case = {
        "benchmark_eligible": True,
        "case_id": case_id,
        "criticality": criticality,
        "planner_key": planner_key,
        "replay": replay,
        "replay_input": replay_input,
        "scenario_family": f"family_{index}",
        "scenario_id": source_row["scenario_id"],
        "seed": source_row["seed"],
        "selected_groups": ["timeout_event"],
        "source_record": source_record,
        "source_showcase_renderer": {"status": "unavailable", "artifacts": []},
    }
    planner = {
        "algorithm_metadata": {
            "canonical_algorithm": canonical_algorithm,
            "config_hash": planner_config_identity,
        },
        "key": planner_key,
    }
    materialized_case = _issue9656_fixture_materialized_case(
        index, case_id, source_row, source_record, criticality, planner, replay, replay_input
    )
    case_file = case_dir / "case.json"
    case_file.write_text(json.dumps(materialized_case, sort_keys=True), encoding="utf-8")
    return summary_case


def _issue9656_fixture_matching_source(
    index: int, promotion_case: bool
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if not promotion_case or index != 0:
        return None, None
    matching_episode = json.loads(
        (_SOURCE_PACKET / "historical_issue_1501_failure_0002/replay_1.jsonl").read_text(
            encoding="utf-8"
        )
    )
    matching_scenario = yaml.safe_load(
        (_SOURCE_PACKET / "historical_issue_1501_failure_0002/scenario.yaml").read_text(
            encoding="utf-8"
        )
    )["scenarios"][0]
    return matching_episode, matching_scenario


def _issue9656_fixture_episode(
    index: int,
    matching_episode: dict[str, object] | None,
    campaign_root: Path,
) -> tuple[dict[str, object], str, dict[str, object]]:
    config_identity = (
        matching_episode["algorithm_metadata"]["config_hash"]
        if matching_episode is not None
        else f"{index + 1:016x}"
    )
    source_row = (
        copy.deepcopy(matching_episode)
        if matching_episode is not None
        else {
            "algo": f"planner_{index}",
            "algorithm_metadata": {
                "canonical_algorithm": f"planner_{index}",
                "config_hash": config_identity,
            },
            "episode_id": f"episode-{index}",
            "git_hash": _ISSUE9656_SOURCE_REVISION,
            "scenario_params": {"algo_config_hash": f"scenario-config-{index}"},
            "scenario_id": f"scenario_{index}",
            "seed": 100 + index,
        }
    )
    if matching_episode is not None:
        source_row["git_hash"] = _ISSUE9656_SOURCE_REVISION
        source_row["scenario_params"]["algo_config_hash"] = config_identity
    episode_file = f"runs/planner_{index}/episodes.jsonl"
    raw_line = json.dumps(source_row, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    episode_path = campaign_root / episode_file
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    episode_path.write_bytes(raw_line)
    record_sha = hashlib.sha256(raw_line).hexdigest()
    source_record = {
        "episode_file": episode_file,
        "episode_file_sha256": record_sha,
        "line_number": 1,
        "record_sha256": record_sha,
    }
    return source_row, config_identity, source_record


def _write_issue9656_fixture_replay_inputs(
    index: int,
    status: str,
    case_dir: Path,
    map_path: Path,
    source_row: dict[str, object],
    matching_episode: dict[str, object] | None,
    matching_scenario: dict[str, object] | None,
) -> dict[str, object]:
    replay_dir = case_dir / "replay_input"
    replay_dir.mkdir(parents=True)
    planner_config = (
        yaml.safe_dump(matching_episode["algorithm_metadata"].get("config", {}), sort_keys=True)
        if matching_episode is not None
        else "planner_option: value\n"
    )
    (replay_dir / "planner_config.yaml").write_text(planner_config, encoding="utf-8")
    scenario_row = _issue9656_fixture_scenario_row(index, map_path, source_row, matching_scenario)
    matrix_text = yaml.safe_dump(
        {"map_search_paths": [str(map_path.parent)], "scenarios": [scenario_row]}, sort_keys=True
    )
    (replay_dir / "replay_matrix.yaml").write_text(matrix_text, encoding="utf-8")
    return {
        "planner_config_path": "replay_input/planner_config.yaml",
        "planner_config_sha256": hashlib.sha256(planner_config.encode()).hexdigest(),
        "replay_eligible": status != "unavailable_model_artifact",
        "replay_ineligibility": (
            "model artifact unavailable" if status == "unavailable_model_artifact" else None
        ),
        "scenario_matrix_path": "replay_input/replay_matrix.yaml",
        "scenario_matrix_sha256": hashlib.sha256(matrix_text.encode()).hexdigest(),
        "status": "materialized",
    }


def _issue9656_fixture_scenario_row(
    index: int,
    map_path: Path,
    source_row: dict[str, object],
    matching_scenario: dict[str, object] | None,
) -> dict[str, object]:
    if matching_scenario is None:
        return {
            "algo": f"planner_{index}",
            "id": f"scenario_{index}",
            "map_file": str(map_path),
            "seeds": [100 + index],
        }
    scenario_row = copy.deepcopy(matching_scenario)
    scenario_row["id"] = matching_scenario["name"]
    scenario_row["algo"] = source_row["algo"]
    scenario_row["map_file"] = str(map_path)
    return scenario_row


def _issue9656_fixture_materialized_case(
    index: int,
    case_id: str,
    source_row: dict[str, object],
    source_record: dict[str, object],
    criticality: dict[str, object],
    planner: dict[str, object],
    replay: dict[str, object],
    replay_input: dict[str, object],
) -> dict[str, object]:
    return {
        "case_file": f"cases/{case_id}/case.json",
        "case_id": case_id,
        "criticality": criticality,
        "planner": planner,
        "replay": replay,
        "replay_input": replay_input,
        "scenario": {
            "scenario_id": source_row["scenario_id"],
            "scenario_family": f"family_{index}",
            "seed": source_row["seed"],
        },
        "schema_version": "benchmark-hard-case.v1",
        "selection": {"selected_groups": ["timeout_event"]},
        "source": {
            **source_record,
            "bundle_sha256": "a" * 64,
            "campaign_id": "campaign-fixture",
            "campaign_source_revision": _ISSUE9656_SOURCE_REVISION,
            "planner_config_hash": planner["algorithm_metadata"]["config_hash"],
            "planner_key": planner["key"],
            "row_git_hash": _ISSUE9656_SOURCE_REVISION,
            "scenario_matrix": _ISSUE9656_SOURCE_MATRIX,
            "scenario_matrix_sha256": _ISSUE9656_SOURCE_MATRIX_SHA256,
        },
        "source_record": source_row,
    }


def _refresh_issue9656_bundle(bundle_root: Path) -> None:
    payload = bundle_root / "payload"
    manifest_path = bundle_root / "evidence_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        source = payload / item["path"]
        item["size_bytes"] = source.stat().st_size
        item["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest["totals"]["total_bytes"] = sum(item["size_bytes"] for item in manifest["files"])
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    lines = [f"{item['sha256']}  payload/{item['path']}" for item in manifest["files"]]
    (bundle_root / "checksums.sha256").write_text("\n".join(lines) + "\n")


@contextmanager
def _packet_copy() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=".test-counterexample-corpus-", dir=_REPO_ROOT) as raw:
        destination = Path(raw) / "issue_9645"
        destination.mkdir()
        shutil.copytree(_SOURCE_PACKET, destination / "payload")
        for filename in ("evidence_bundle_manifest.json", "checksums.sha256"):
            shutil.copyfile(_SOURCE_BUNDLE / filename, destination / filename)
        yield destination / "payload"


def _import(tmp_path: Path, payload: Path | None = None):
    corpus_root = tmp_path / "corpus"
    corpus, receipt = import_issue9645_packet(
        payload or _SOURCE_PACKET, new_corpus(), corpus_root=corpus_root
    )
    return corpus, receipt, corpus_root


def _stage_case_under_candidate(case: dict[str, object], corpus_root: Path, candidate_id: str):
    """Copy exact case inputs/replay bytes under one imported candidate's custody root."""
    case_record = copy.deepcopy(case)
    source_prefix = f"cases/{case['case_id']}/"
    destination_prefix = f"historical_candidates/{candidate_id}/admission_evidence/"
    source = corpus_root / source_prefix
    destination = corpus_root / destination_prefix
    shutil.copytree(source, destination)

    def relocate(relative: str) -> str:
        assert relative.startswith(source_prefix)
        return destination_prefix + relative[len(source_prefix) :]

    case_record["inputs"]["scenario_path"] = relocate(case_record["inputs"]["scenario_path"])
    case_record["inputs"]["route_overrides_path"] = relocate(
        case_record["inputs"]["route_overrides_path"]
    )
    for asset in case_record["inputs"]["map_assets"]:
        asset["path"] = relocate(asset["path"])
    receipt = case_record["replay_receipt"]
    for replay in receipt.get("replay_artifacts", []):
        for field in ("path", "provenance_path"):
            if replay.get(field):
                replay[field] = relocate(replay[field])
    for replay in receipt.get("artifact_receipts", []):
        replay["artifact_path"] = relocate(replay["artifact_path"])
    return case_record


def _single_replay_admission_receipt(
    case: dict[str, object], corpus_root: Path, *, target_revision: str | None = None
) -> dict[str, object]:
    """Create an exact-revision receipt from a stored episode fixture without running it."""
    old_receipt = case["replay_receipt"]
    source_path = corpus_root / old_receipt["artifact_receipts"][0]["artifact_path"]
    record = json.loads(source_path.read_text(encoding="utf-8"))
    run_id = f"fixture-promotion-{uuid.uuid4().hex}"
    binding = counterexample_corpus._case_runtime_input_binding(case, corpus_root)
    record.setdefault("provenance", {})["case_input_identity"] = {
        **binding,
        "run_id": run_id,
        "reason_codes": [],
    }
    artifact_path = f"cases/{case['case_id']}/replay_artifacts/{run_id}.jsonl"
    artifact = corpus_root / artifact_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    case["source_evidence"]["corpus_files"].append(
        {
            "path": artifact_path,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
    )
    projection = old_receipt["selected_projection"]
    observation = {
        "case_id": case["case_id"],
        "effective_scenario_sha256": case["effective_scenario_sha256"],
        "planner_id": projection["planner_id"],
        "planner_config_identity": projection["planner_config_identity"],
        "source_revision": projection["source_revision"],
        "episode_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "outcome": projection["outcome"],
        "termination_reason": projection["termination_reason"],
        "metrics": projection["metrics"],
        "execution_mode": "native",
        "readiness_status": "native",
        "availability_status": "available",
        "fallback_or_degraded": False,
        "evidence_status": "complete",
        "run_id": run_id,
    }
    # This fixture reuses historical bytes, so bind its synthetic target to the
    # fixture's recorded source revision only within the helper's test scope.
    with pytest.MonkeyPatch.context() as target_patch:
        target_patch.setattr(
            counterexample_corpus,
            "_current_target_revision",
            lambda: projection["source_revision"],
        )
        return create_case_admission_replay_receipt(
            observation,
            case,
            artifact_path=artifact_path,
            corpus_root=corpus_root,
            target_revision=target_revision,
        )


def _append_episode_evaluation(
    corpus: dict[str, object],
    corpus_root: Path,
    *,
    planner_id: str,
    config_identity: str,
    execution_mode: str,
    outcome: dict[str, bool] | None = None,
    termination_reason: str | None = None,
    degraded: bool = False,
) -> dict[str, object]:
    """Append a fixture observation whose receipt is derived from its stored JSONL row."""
    case = corpus["cases"][0]
    record_path = _SOURCE_PACKET / "historical_issue_1501_failure_0002/replay_1.jsonl"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    selected_outcome = outcome or {
        "collision_event": True,
        "route_complete": False,
        "timeout_event": False,
    }
    record["algo"] = planner_id
    record["algorithm_metadata"]["algorithm"] = planner_id
    record["algorithm_metadata"]["canonical_algorithm"] = planner_id
    record["algorithm_metadata"]["config_hash"] = config_identity
    record["algorithm_metadata"]["planner_kinematics"]["execution_mode"] = execution_mode
    record["outcome"] = selected_outcome
    record["event_ledger"]["planner"] = planner_id
    record["event_ledger"]["exact_events"] = {
        "collision": selected_outcome["collision_event"],
        "goal_reached": selected_outcome["route_complete"],
        "timeout": selected_outcome["timeout_event"],
        "invalid_run": False,
    }
    record["event_ledger"]["software_commit"] = record["git_hash"]
    if termination_reason is None:
        if selected_outcome["route_complete"]:
            termination_reason = "success"
        elif selected_outcome["collision_event"]:
            termination_reason = "collision"
        elif selected_outcome["timeout_event"]:
            termination_reason = "truncated"
    if termination_reason is not None:
        record["termination_reason"] = termination_reason
        record["status"] = counterexample_corpus.status_from_termination_reason(termination_reason)
    record["metrics"]["success"] = int(selected_outcome["route_complete"])
    collision_count = int(selected_outcome["collision_event"])
    record["metrics"]["collisions"] = collision_count
    record["metrics"]["total_collision_count"] = collision_count
    if collision_count == 0:
        record["event_ledger"]["collision_events"] = []
    record["event_ledger"]["reconciliation"]["collision_metric_value"] = collision_count
    if degraded:
        record["integrity"]["effective_view"]["degraded"] = True
    else:
        record["integrity"]["effective_view"]["degraded"] = False
    record.setdefault("provenance", {})["case_input_identity"] = {
        **counterexample_corpus._case_runtime_input_binding(case, corpus_root),
        "run_id": f"fixture-{uuid.uuid4().hex}",
        "reason_codes": [],
    }

    relative_artifact_path = f"replay_artifacts/{planner_id}-{execution_mode}.jsonl"
    artifact_path = corpus_root / relative_artifact_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(record, sort_keys=True) + "\n"
    artifact_path.write_text(rendered, encoding="utf-8")
    episode_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    metrics = {
        key: record["metrics"][key] for key in ("success", "collisions", "total_collision_count")
    }
    evaluation = {
        "case_id": case["case_id"],
        "effective_scenario_sha256": case["effective_scenario_sha256"],
        "planner_id": planner_id,
        "planner_config_identity": config_identity,
        "source_revision": record["git_hash"],
        "episode_sha256": episode_sha256,
        "execution_mode": execution_mode,
        "readiness_status": "native" if execution_mode == "native" else "adapter",
        "availability_status": "available",
        "fallback_or_degraded": degraded,
        "evidence_status": "complete",
        "error": None,
        "outcome": selected_outcome,
        "termination_reason": record["termination_reason"],
        "metrics": metrics,
    }
    evaluation["replay_receipt"] = create_planner_replay_receipt(
        evaluation,
        case,
        artifact_path=relative_artifact_path,
        corpus_root=corpus_root,
    )
    append_planner_evaluation(corpus, evaluation, corpus_root=corpus_root)
    return evaluation


def _refresh_evaluation_id(evaluation: dict[str, object]) -> None:
    identity = {
        key: evaluation.get(key)
        for key in (
            "case_id",
            "effective_scenario_sha256",
            "planner_id",
            "planner_config_identity",
            "source_revision",
            "episode_sha256",
            "outcome",
            "termination_reason",
            "metrics",
            "execution_mode",
            "readiness_status",
            "availability_status",
            "fallback_or_degraded",
            "evidence_status",
            "error",
            "replay_receipt",
        )
    }
    evaluation["evaluation_id"] = hashlib.sha256(
        counterexample_corpus._stable_json(identity).encode("utf-8")
    ).hexdigest()


def test_issue9645_packet_import_preserves_zero_discovery_and_unknown_feasibility(
    tmp_path: Path,
) -> None:
    corpus, receipt, corpus_root = _import(tmp_path)

    assert receipt["decision"] == "admitted"
    assert receipt["pilot_new_discoveries"] == 0
    assert len(corpus["search_runs"]) == 1
    pilot = corpus["search_runs"][0]
    assert pilot["attempted_candidates"] == 64
    assert pilot["new_counterexamples_discovered"] == 0
    assert pilot["new_counterexamples_admitted"] == 0
    assert pilot["evidence_tier"] == "diagnostic_only"

    assert len(corpus["cases"]) == 1
    case = corpus["cases"][0]
    assert case["admissibility"]["verdict"] == "admissible_feasibility_unknown"
    assert case["admissibility"]["dynamic_task_status"] == "unknown"
    assert case["discovery"]["origin_case_id"] == "issue_1501/failure_0002"
    assert len(corpus["planner_evaluations"]) == 2
    assert all(row["evidence_status"] == "complete" for row in corpus["planner_evaluations"])
    assert case["replay_receipt"]["input_binding_status"] == "unknown_historical"
    assert case["replay_receipt"]["input_binding_limitation"]
    assert len({row["run_id"] for row in case["replay_receipt"]["artifact_receipts"]}) == 2
    assert all(
        row["input_binding"]["status"] == "unknown_historical"
        for row in case["replay_receipt"]["artifact_receipts"]
    )
    historical_state = counterexample_corpus._evaluation_state(
        corpus["planner_evaluations"][0], case=case, corpus_root=corpus_root
    )
    assert historical_state == {
        "status": "unknown",
        "reason_codes": ["replay_input_binding_unknown_historical"],
    }

    replay_artifacts = case["replay_receipt"]["replay_artifacts"]
    assert (
        replay_artifacts[0]["source_artifact_sha256_before_path_normalization"]
        != (replay_artifacts[0]["normalized_bundle_sha256"])
    )
    assert (
        replay_artifacts[0]["provenance_sha256_before_path_normalization"]
        != (replay_artifacts[0]["provenance_sha256_normalized"])
    )
    assert replay_artifacts[0]["path_normalization"] == "scenario_params.route_overrides_file"
    assert replay_artifacts[0]["provenance_path_normalization"] == "run.invocation"
    for item in pilot["source_files"] + pilot["manifest_files"] + pilot["bundle_receipts"]:
        stored = corpus_root / item["path"]
        assert hashlib.sha256(stored.read_bytes()).hexdigest() == item["sha256"]
    validate_corpus(corpus)


def test_rehashed_route_input_cannot_reuse_a_stale_replay_jsonl(tmp_path: Path) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    evaluation = _append_episode_evaluation(
        corpus,
        corpus_root,
        planner_id="route-binding-check",
        config_identity="route-binding-config",
        execution_mode="native",
    )
    case = copy.deepcopy(corpus["cases"][0])
    row = copy.deepcopy(evaluation)
    legacy_claim = copy.deepcopy(evaluation)
    legacy_claim["replay_receipt"]["schema_version"] = "adversarial-planner-replay-receipt.v1"
    assert counterexample_corpus._evaluation_state(
        legacy_claim, case=case, corpus_root=corpus_root
    ) == {
        "status": "unknown",
        "reason_codes": ["replay_input_binding_unknown_historical"],
    }
    route_path = corpus_root / case["inputs"]["route_overrides_path"]
    route_payload = yaml.safe_load(route_path.read_text(encoding="utf-8"))
    route_payload["robot_routes"][0]["waypoints"][0][0] += 0.25
    route_path.write_text(yaml.safe_dump(route_payload, sort_keys=True), encoding="utf-8")

    case["inputs"]["route_overrides_sha256"] = hashlib.sha256(route_path.read_bytes()).hexdigest()
    scenario_path = corpus_root / case["inputs"]["scenario_path"]
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))["scenarios"][0]
    case["effective_scenario_sha256"] = counterexample_corpus.compute_case_effective_scenario_hash(
        scenario, route_payload, case["inputs"]["map_assets"]
    )
    case["case_id"] = f"case-{case['effective_scenario_sha256']}"
    row["case_id"] = case["case_id"]
    row["effective_scenario_sha256"] = case["effective_scenario_sha256"]
    replay_receipt = row["replay_receipt"]
    replay_receipt["case_id"] = case["case_id"]
    replay_receipt["effective_scenario_sha256"] = case["effective_scenario_sha256"]
    replay_receipt["route_overrides_sha256"] = case["inputs"]["route_overrides_sha256"]
    replay_receipt["input_binding"]["route_overrides_sha256"] = case["inputs"][
        "route_overrides_sha256"
    ]
    case["replay_receipt"]["effective_scenario_sha256"] = case["effective_scenario_sha256"]
    case["replay_receipt"]["route_overrides_sha256"] = case["inputs"]["route_overrides_sha256"]
    for inventory_row in case["source_evidence"]["corpus_files"]:
        if inventory_row["path"] == case["inputs"]["route_overrides_path"]:
            inventory_row["sha256"] = case["inputs"]["route_overrides_sha256"]

    errors = counterexample_corpus._validate_replay_artifact(row, case, replay_receipt, corpus_root)
    assert "replay_artifact_input_binding_route_overrides_sha256_mismatch" in errors
    assert "replay_receipt_input_binding_route_overrides_sha256_mismatch" in errors


def test_admission_replay_inventory_is_one_to_one_and_run_ids_are_distinct(
    tmp_path: Path,
) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    case = corpus["cases"][0]
    assert counterexample_corpus._validate_case_admission_replay(case, corpus_root) == []

    mismatched_inventory = copy.deepcopy(case)
    mismatched_inventory["replay_receipt"]["replay_artifacts"][0]["path"] = (
        "cases/fake/replay.jsonl"
    )
    mismatched_inventory["replay_receipt"]["replay_artifacts"][0]["sha256"] = "0" * 64
    assert "replay artifact inventory row does not match its artifact receipt" in (
        counterexample_corpus._validate_case_admission_replay(mismatched_inventory, corpus_root)
    )

    reused_run = copy.deepcopy(case)
    first_run_id = reused_run["replay_receipt"]["artifact_receipts"][0]["run_id"]
    reused_run["replay_receipt"]["artifact_receipts"][1]["run_id"] = first_run_id
    reused_run["replay_receipt"]["replay_artifacts"][1]["run_id"] = first_run_id
    assert "admission replay artifacts must have distinct run IDs" in (
        counterexample_corpus._validate_case_admission_replay(reused_run, corpus_root)
    )


def test_legacy_v1_replay_receipts_remain_readable_without_input_claim_upgrade(
    tmp_path: Path,
) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    case = copy.deepcopy(corpus["cases"][0])
    legacy_receipt = case["replay_receipt"]
    legacy_receipt["schema_version"] = "adversarial-case-admission-replay.v1"
    legacy_receipt.pop("input_binding_status")
    legacy_receipt.pop("input_binding_limitation")
    for artifact_receipt in legacy_receipt["artifact_receipts"]:
        artifact_receipt["schema_version"] = "adversarial-planner-replay-receipt.v1"
        artifact_receipt.pop("run_id")
        artifact_receipt.pop("input_binding")
    for inventory_row in legacy_receipt["replay_artifacts"]:
        inventory_row.pop("run_id")

    assert counterexample_corpus._validate_case_admission_replay(case, corpus_root) == []
    assert counterexample_corpus._case_replay_input_binding_status(case) == "unknown_legacy"
    validate_corpus({**corpus, "cases": [case]}, corpus_root=corpus_root)
    legacy_evaluation = copy.deepcopy(corpus["planner_evaluations"][0])
    legacy_evaluation["replay_receipt"]["schema_version"] = "adversarial-planner-replay-receipt.v1"
    legacy_evaluation["replay_receipt"].pop("run_id")
    legacy_evaluation["replay_receipt"].pop("input_binding")
    legacy_state = counterexample_corpus._evaluation_state(
        legacy_evaluation,
        case=case,
        corpus_root=corpus_root,
    )
    assert legacy_state["status"] == "unknown"


def test_unknown_feasibility_cannot_be_promoted_without_successful_replay_evidence(
    tmp_path: Path,
) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    case = corpus["cases"][0]
    assert case["admissibility"]["static_certificate"]["claim_boundary"].startswith(
        "static route certificate only"
    )
    unsupported_upgrade = copy.deepcopy(corpus)
    unsupported_upgrade["cases"][0]["admissibility"]["verdict"] = "empirically_feasible"

    with pytest.raises(
        CorpusError, match="positive feasibility verdict requires an evidence receipt"
    ):
        validate_corpus(unsupported_upgrade, corpus_root=corpus_root)

    evaluation = _append_episode_evaluation(
        corpus,
        corpus_root,
        planner_id="reference-planner",
        config_identity=case["target_planner"]["config_identity"],
        execution_mode="native",
        outcome={"collision_event": False, "route_complete": True, "timeout_event": False},
    )
    stored_evaluation = next(
        row
        for row in corpus["planner_evaluations"]
        if row["planner_id"] == evaluation["planner_id"]
    )
    case["admissibility"]["verdict"] = "empirically_feasible"
    case["admissibility"]["evidence_receipt"] = {
        "schema_version": "adversarial-case-admissibility-evidence.v1",
        "case_id": case["case_id"],
        "effective_scenario_sha256": case["effective_scenario_sha256"],
        "verdict": "empirically_feasible",
        "evaluation_id": stored_evaluation["evaluation_id"],
    }
    validate_corpus(corpus, corpus_root=corpus_root)

    stored_evaluation["outcome"] = {
        "collision_event": True,
        "route_complete": False,
        "timeout_event": False,
    }
    with pytest.raises(CorpusError, match="evaluation digest is invalid"):
        validate_corpus(corpus, corpus_root=corpus_root)


def test_selected_projection_must_match_every_verified_replay_artifact(tmp_path: Path) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    replay = corpus["cases"][0]["replay_receipt"]
    projection = replay["selected_projection"]
    projection["outcome"] = {
        "collision_event": False,
        "route_complete": True,
        "timeout_event": False,
    }
    projection["termination_reason"] = "success"
    projection["metrics"].update({"success": 1.0, "collisions": 0.0, "total_collision_count": 0.0})
    projection["selected_event_identity"]["exact_events"] = {
        "collision": False,
        "goal_reached": True,
        "timeout": False,
    }
    replay["selected_projection_sha256"] = hashlib.sha256(
        counterexample_corpus._stable_json(projection).encode("utf-8")
    ).hexdigest()

    with pytest.raises(
        CorpusError, match="selected projection differs from a verified replay artifact"
    ):
        validate_corpus(corpus, corpus_root=corpus_root)


def test_case_admission_rejects_unselected_contradictory_canonical_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw replay contradictions remain visible when a projection omits that metric."""
    corpus_root = tmp_path / "corpus"
    corpus, _pilot = import_issue9645_packet(_SOURCE_PACKET, new_corpus(), corpus_root=corpus_root)
    case = copy.deepcopy(corpus["cases"][0])
    receipt = _single_replay_admission_receipt(case, corpus_root)
    artifact_receipt = receipt["artifact_receipts"][0]
    artifact_path = corpus_root / artifact_receipt["artifact_path"]
    episode = json.loads(artifact_path.read_text(encoding="utf-8"))

    # Preserve the replay's collision projection while corrupting an unselected
    # canonical success metric. Update every byte receipt so only semantic
    # consistency, not a stale digest, can reject the case.
    episode["metrics"]["success"] = 1
    episode["metrics"]["success_rate"] = 1
    artifact_path.write_text(json.dumps(episode, sort_keys=True) + "\n", encoding="utf-8")
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    selected_metrics = {
        key: value
        for key, value in artifact_receipt["metrics"].items()
        if key in {"collisions", "total_collision_count"}
    }
    artifact_receipt["metrics"] = selected_metrics
    artifact_receipt["artifact_sha256"] = artifact_sha256
    artifact_receipt["episode_sha256"] = artifact_sha256
    receipt["replay_artifacts"][0]["sha256"] = artifact_sha256
    receipt["selected_projection"]["metrics"] = selected_metrics
    receipt["selected_projection_sha256"] = hashlib.sha256(
        counterexample_corpus._stable_json(receipt["selected_projection"]).encode("utf-8")
    ).hexdigest()
    for source_file in case["source_evidence"]["corpus_files"]:
        if source_file["path"] == artifact_receipt["artifact_path"]:
            source_file["sha256"] = artifact_sha256
            break
    else:
        raise AssertionError("admission replay artifact is absent from source evidence custody")
    case["replay_receipt"] = receipt
    monkeypatch.setattr(
        counterexample_corpus,
        "_current_target_revision",
        lambda: receipt["target_revision"],
    )

    corpus, admission = counterexample_corpus.admit_case_record(
        case,
        corpus,
        corpus_root=corpus_root,
        artifact_root=f"cases/{case['case_id']}",
        source_kind="test_unselected_metric_contradiction",
        source_id="contradictory-success-metric",
    )

    assert admission["decision"] == "rejected"
    assert any(
        "replay_artifact_outcome_metric_contradiction" in blocker
        and "success metrics > 0" in blocker
        for blocker in admission["blockers"]
    )
    assert len(corpus["cases"]) == 1
    validate_corpus(corpus, corpus_root=corpus_root)


def test_legacy_v1_no_row_admission_rejects_unselected_contradictory_metrics(
    tmp_path: Path,
) -> None:
    """Legacy no-row receipts cannot hide raw metrics omitted from their projection."""
    corpus_root = tmp_path / "corpus"
    corpus, _pilot = import_issue9645_packet(_SOURCE_PACKET, new_corpus(), corpus_root=corpus_root)
    case = copy.deepcopy(corpus["cases"][0])
    receipt = case["replay_receipt"]
    selected_metrics = {
        key: value
        for key, value in receipt["selected_projection"]["metrics"].items()
        if key in {"collisions", "total_collision_count"}
    }
    assert selected_metrics

    for replay in receipt["replay_artifacts"]:
        artifact_path = replay["path"]
        artifact = corpus_root / artifact_path
        episode = json.loads(artifact.read_text(encoding="utf-8"))
        episode["metrics"]["success"] = 1
        episode["metrics"]["success_rate"] = 1
        artifact.write_text(json.dumps(episode, sort_keys=True) + "\n", encoding="utf-8")
        artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        replay["sha256"] = artifact_sha256
        replay["normalized_bundle_sha256"] = artifact_sha256
        for source_file in case["source_evidence"]["corpus_files"]:
            if source_file["path"] == artifact_path:
                source_file["sha256"] = artifact_sha256
                break
        else:
            raise AssertionError("admission replay artifact is absent from source evidence custody")

    receipt["selected_projection"]["metrics"] = selected_metrics
    receipt["selected_projection_sha256"] = hashlib.sha256(
        counterexample_corpus._stable_json(receipt["selected_projection"]).encode("utf-8")
    ).hexdigest()
    receipt["schema_version"] = "adversarial-case-admission-replay.v1"
    receipt.pop("artifact_receipts")

    errors = counterexample_corpus._validate_case_admission_replay(case, corpus_root)
    assert any(
        "replay_artifact_outcome_metric_contradiction" in error and "success metrics > 0" in error
        for error in errors
    )
    with pytest.raises(CorpusError, match="replay_artifact_outcome_metric_contradiction"):
        validate_corpus({**corpus, "cases": [case]}, corpus_root=corpus_root)


def test_target_planner_configuration_snapshot_must_match_replay_metadata(
    tmp_path: Path,
) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    case = corpus["cases"][0]
    case["target_planner"]["configuration_snapshot"] = {"review_injected": True}

    with pytest.raises(CorpusError, match="replay_artifact_target_configuration_snapshot_mismatch"):
        validate_corpus(corpus, corpus_root=corpus_root)
    with pytest.raises(CorpusError, match="replay_artifact_target_configuration_snapshot_mismatch"):
        export_regression_slice(
            corpus,
            corpus_root=corpus_root,
            output_dir=tmp_path / "injected-config-slice",
        )


def test_issue9656_import_keeps_unverified_rows_in_separate_candidate_registry(
    tmp_path: Path,
) -> None:
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(tmp_path)
    corpus_root = tmp_path / "corpus"
    corpus, receipt = import_issue9656_candidates(
        summary_path,
        materialized,
        bundle_root,
        campaign_root,
        new_corpus(),
        corpus_root=corpus_root,
    )

    assert receipt["decision"] == "imported"
    assert receipt["candidate_count"] == 3
    assert receipt["source_rows_verified"] == 3
    assert receipt["source_replay_status_counts"] == {
        "mismatch_different_revision": 1,
        "not_attempted": 1,
        "unavailable_model_artifact": 1,
    }
    assert receipt["candidate_status_counts"] == {
        "blocked_replay_revision_mismatch": 1,
        "blocked_unavailable_model_artifact": 1,
        "pending_exact_replay": 1,
    }
    assert receipt["criticality_anomaly_counts"] == {
        "collision_event_without_positive_collision_metric": 1
    }
    assert receipt["failed_setup_jobs"] == 12
    assert receipt["failed_replay_attempts"]["attempts"][0]["replay_row_count"] == 0
    assert receipt["failed_replay_attempts"]["attempts"][0]["scheduled_jobs"] == 3
    assert corpus["cases"] == []
    assert corpus["admission_attempts"] == []
    assert any(item["path"] == "payload/report.md" for item in receipt["source_files"])
    assert (corpus_root / receipt["artifact_paths"]["payload_root"] / "report.md").is_file()
    assert {candidate["source_case_id"] for candidate in corpus["historical_candidates"]} == {
        "case-0000000000000001",
        "case-0000000000000002",
        "case-0000000000000003",
    }

    for candidate in corpus["historical_candidates"]:
        assert candidate["source_provenance"]["source_identity_binding_status"] == "verified"
        assert (
            candidate["target_planner"]["planner_id"]
            == candidate["source_provenance"]["raw_planner_alias"]
        )
        assert (
            candidate["target_planner"]["canonical_algorithm"]
            == candidate["source_provenance"]["episode_canonical_algorithm"]
        )
        assert candidate["source_provenance"]["episode_scenario_algo_config_hash"].startswith(
            "scenario-config-"
        )
        assert candidate["source_provenance"]["source_row_binding"] == (
            "verified_episode_file_and_line_sha256"
        )
        assert candidate["source_replay_status"] in {
            "mismatch_different_revision",
            "not_attempted",
            "unavailable_model_artifact",
        }
        assert candidate["benchmark_eligible"] is True
        assert len(candidate["source_record_sha256"]) == 64
        replay_matrix = corpus_root / candidate["artifact_paths"]["replay_matrix"]
        replay_config = corpus_root / candidate["artifact_paths"]["planner_config"]
        source_case = corpus_root / candidate["artifact_paths"]["source_case"]
        assert replay_matrix.is_file() and replay_config.is_file() and source_case.is_file()
        assert (
            hashlib.sha256(source_case.read_bytes()).hexdigest()
            == (candidate["source_provenance"]["source_case_file_sha256"])
        )
        normalized = yaml.safe_load(replay_matrix.read_text(encoding="utf-8"))
        normalized_map = (replay_matrix.parent / normalized["scenarios"][0]["map_file"]).resolve()
        assert normalized_map.is_file()
        assert len(candidate["replay_inputs"]["map_assets"]) == 1

    slice_manifest = export_regression_slice(
        corpus, corpus_root=corpus_root, output_dir=tmp_path / "candidate-slice"
    )
    assert slice_manifest["case_count"] == 0
    validate_corpus(corpus)

    corpus, duplicate = import_issue9656_candidates(
        summary_path, materialized, bundle_root, campaign_root, corpus, corpus_root=corpus_root
    )
    assert duplicate["decision"] == "duplicate"
    assert len(corpus["historical_candidates"]) == 3

    case_path = materialized / "cases/case-0000000000000001/case.json"
    case_document = json.loads(case_path.read_text(encoding="utf-8"))
    case_document["source"]["row_git_hash"] = "0" * 40
    case_path.write_text(json.dumps(case_document, sort_keys=True), encoding="utf-8")
    with pytest.raises(CorpusError, match="duplicate #9656 import materialized case differs"):
        import_issue9656_candidates(
            summary_path, materialized, bundle_root, campaign_root, corpus, corpus_root=corpus_root
        )
    assert len(corpus["historical_candidates"]) == 3


def test_pending_historical_candidate_promotes_after_exact_replay_and_input_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(
        tmp_path, statuses=("not_attempted",), promotion_case=True
    )
    corpus_root = tmp_path / "corpus"
    corpus, imported = import_issue9656_candidates(
        summary_path,
        materialized,
        bundle_root,
        campaign_root,
        new_corpus(),
        corpus_root=corpus_root,
    )
    assert imported["candidate_status_counts"] == {"pending_exact_replay": 1}
    corpus, pilot = import_issue9645_packet(_SOURCE_PACKET, corpus, corpus_root=corpus_root)
    assert pilot["decision"] == "admitted"

    candidate = corpus["historical_candidates"][0]
    assert candidate["source_provenance"]["raw_episode_artifact_custody"] == {
        "status": "digest_only_not_copied_from_campaign_output",
        "episode_file": candidate["source_provenance"]["episode_file"],
        "episode_file_sha256": candidate["source_provenance"]["episode_file_sha256"],
        "raw_episode_artifact_used_as_admission_evidence": False,
        "local_ignored_output_used_as_admission_evidence": False,
        "admission_requires": "exact_current_revision_replay",
    }
    case = copy.deepcopy(corpus["cases"][0])
    case["replay_receipt"] = _single_replay_admission_receipt(case, corpus_root)
    fixture_target_revision = case["replay_receipt"]["target_revision"]
    monkeypatch.setattr(
        counterexample_corpus,
        "_current_target_revision",
        lambda: fixture_target_revision,
    )
    validate_corpus({**corpus, "cases": [case]}, corpus_root=corpus_root)
    case_record = _stage_case_under_candidate(case, corpus_root, candidate["candidate_id"])
    case_record["discovery"]["historical_candidate_binding"] = {
        "candidate_id": candidate["candidate_id"],
        "source_issue": 9656,
        "source_case_id": candidate["source_case_id"],
        "source_record_sha256": candidate["source_record_sha256"],
        "source_replay_status": candidate["source_replay_status"],
    }

    case_path = tmp_path / "promotion-case.json"
    case_path.write_text(json.dumps(case_record, sort_keys=True), encoding="utf-8")
    corpus_path = corpus_root / "corpus.json"
    save_corpus(corpus_path, corpus)
    receipt_path = tmp_path / "promotion-receipt.json"
    cli_status = corpus_cli_main(
        [
            "promote-candidate",
            "--candidate-id",
            candidate["candidate_id"],
            "--case",
            str(case_path),
            "--corpus",
            str(corpus_path),
            "--corpus-root",
            str(corpus_root),
            "--output",
            str(receipt_path),
        ]
    )
    assert cli_status == 0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    candidate = corpus["historical_candidates"][0]

    assert receipt["decision"] == "duplicate"
    assert receipt["case_id"] == case["case_id"]
    assert candidate["source_candidate_status"] == "pending_exact_replay"
    assert candidate["candidate_status"] == "admitted"
    assert candidate["promoted_case_id"] == case["case_id"]
    assert candidate["promotion_attempt_id"] == receipt["attempt_id"]
    admitted_case = corpus["cases"][0]
    promotion_evidence = next(
        item
        for item in admitted_case["supporting_source_evidence"]
        if item.get("historical_candidate_promotion", {}).get("candidate_id")
        == candidate["candidate_id"]
    )["historical_candidate_promotion"]
    historical_replay = promotion_evidence
    assert historical_replay["source_replay_status"] == "not_attempted"
    assert historical_replay["raw_episode_artifact_used_as_admission_evidence"] is False
    assert historical_replay["local_ignored_output_used_as_admission_evidence"] is False
    assert historical_replay["admission_replay_matches_target_revision"] is True
    validate_corpus(corpus, corpus_root=corpus_root)


def test_admission_receipt_rejects_replay_revision_as_untrusted_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller cannot label a stale replay revision as the current target."""
    corpus_root = tmp_path / "corpus"
    corpus, _pilot = import_issue9645_packet(_SOURCE_PACKET, new_corpus(), corpus_root=corpus_root)
    case = copy.deepcopy(corpus["cases"][0])
    case["replay_receipt"] = _single_replay_admission_receipt(case, corpus_root)
    monkeypatch.setattr(counterexample_corpus, "_current_target_revision", lambda: "f" * 40)
    assert counterexample_corpus._validate_case_current_target_revision(case) == [
        "admission replay does not match the independently resolved current target revision"
    ]
    corpus, admission = counterexample_corpus.admit_case_record(
        case,
        corpus,
        corpus_root=corpus_root,
        artifact_root=f"cases/{case['case_id']}",
        source_kind="test_exact_current_revision",
        source_id="stale-replay-source-revision",
    )
    assert admission["decision"] == "rejected"
    assert any(
        "independently resolved current target revision" in blocker
        for blocker in admission["blockers"]
    )

    with pytest.raises(CorpusError, match="independently resolved current target"):
        _single_replay_admission_receipt(case, corpus_root, target_revision="0" * 40)


def test_case_admission_rejects_successful_target_replay_despite_criticality_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovery metadata cannot make a successful target replay a counterexample."""
    corpus_root = tmp_path / "corpus"
    corpus, _pilot = import_issue9645_packet(_SOURCE_PACKET, new_corpus(), corpus_root=corpus_root)
    case = copy.deepcopy(corpus["cases"][0])
    case["discovery"]["criticality"] = {
        "claimed_metric": "minimum_clearance_m",
        "claimed_threshold": -1000.0,
        "claimed_value": -1001.0,
    }
    target = case["target_planner"]
    observation = _append_episode_evaluation(
        corpus,
        corpus_root,
        planner_id=target["planner_id"],
        config_identity=target["config_identity"],
        execution_mode="native",
        outcome={"collision_event": False, "route_complete": True, "timeout_event": False},
    )

    source_artifact = corpus_root / observation["replay_receipt"]["artifact_path"]
    artifact_relative = f"cases/{case['case_id']}/replay_artifacts/successful_target.jsonl"
    artifact_path = corpus_root / artifact_relative
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_artifact, artifact_path)
    case["source_evidence"]["corpus_files"].append(
        {
            "path": artifact_relative,
            "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        }
    )
    monkeypatch.setattr(
        counterexample_corpus,
        "_current_target_revision",
        lambda: observation["source_revision"],
    )
    case["replay_receipt"] = create_case_admission_replay_receipt(
        observation,
        case,
        artifact_path=artifact_relative,
        corpus_root=corpus_root,
    )

    corpus, admission = counterexample_corpus.admit_case_record(
        case,
        corpus,
        corpus_root=corpus_root,
        artifact_root=f"cases/{case['case_id']}",
        source_kind="test_noncritical_target_replay",
        source_id="successful-target-replay",
    )

    assert admission["decision"] == "rejected"
    assert any(
        "does not verify canonical collision/timeout noncompletion" in blocker
        for blocker in admission["blockers"]
    )
    assert case["discovery"]["criticality"]["claimed_metric"] == "minimum_clearance_m"


def test_case_admission_rejects_missing_or_mismatched_current_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admission stays unknown when checkout HEAD cannot be independently resolved."""
    corpus_root = tmp_path / "corpus"
    corpus, _pilot = import_issue9645_packet(_SOURCE_PACKET, new_corpus(), corpus_root=corpus_root)
    case = copy.deepcopy(corpus["cases"][0])
    case["replay_receipt"] = _single_replay_admission_receipt(case, corpus_root)
    monkeypatch.setattr(counterexample_corpus, "_current_target_revision", lambda: None)
    corpus, admission = counterexample_corpus.admit_case_record(
        case,
        corpus,
        corpus_root=corpus_root,
        artifact_root=f"cases/{case['case_id']}",
        source_kind="test_exact_current_revision",
        source_id="missing-trusted-target-revision",
    )
    assert admission["decision"] == "rejected"
    assert any(
        "independent current target revision is unavailable" in blocker
        for blocker in admission["blockers"]
    )


@pytest.mark.parametrize(
    ("status_output", "expected_revision"),
    [("", "a" * 40), (" M robot_sf/planner.py\n", None), ("?? local_module.py\n", None)],
)
def test_current_target_revision_requires_clean_source_checkout(
    monkeypatch: pytest.MonkeyPatch,
    status_output: str,
    expected_revision: str | None,
) -> None:
    """Dirty tracked or untracked source cannot claim the checkout HEAD as exact."""
    timeout_expired = counterexample_corpus.subprocess.TimeoutExpired

    def fake_git_run(arguments: list[str], **_kwargs: object) -> SimpleNamespace:
        if "rev-parse" in arguments:
            return SimpleNamespace(returncode=0, stdout=f"{'a' * 40}\n")
        return SimpleNamespace(returncode=0, stdout=status_output)

    monkeypatch.setattr(
        counterexample_corpus,
        "subprocess",
        SimpleNamespace(run=fake_git_run, TimeoutExpired=timeout_expired),
    )
    assert counterexample_corpus._current_target_revision() == expected_revision


@pytest.mark.parametrize(
    ("tamper", "expected_blocker"),
    [
        ("replay_matrix", "candidate_replay_matrix_artifact_invalid"),
        ("planner_config", "candidate_planner_config_artifact_invalid"),
        ("map", "candidate_map_artifact_invalid"),
        ("scenario_metadata", "candidate_metadata_differs_from_checksum_pinned_summary_row"),
    ],
)
def test_pending_historical_candidate_rejects_source_metadata_or_input_tampering(
    tmp_path: Path, tamper: str, expected_blocker: str
) -> None:
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(
        tmp_path, statuses=("not_attempted",)
    )
    corpus_root = tmp_path / "corpus"
    corpus, _receipt = import_issue9656_candidates(
        summary_path,
        materialized,
        bundle_root,
        campaign_root,
        new_corpus(),
        corpus_root=corpus_root,
    )
    candidate = corpus["historical_candidates"][0]
    if tamper == "scenario_metadata":
        candidate["scenario_id"] = "forged-scenario"
    elif tamper == "map":
        map_file = corpus_root / candidate["replay_inputs"]["map_assets"][0]["stored_path"]
        map_file.write_bytes(map_file.read_bytes() + b"tampered")
    else:
        artifact = corpus_root / candidate["artifact_paths"][tamper]
        artifact.write_bytes(artifact.read_bytes() + b"\n")

    corpus, receipt = promote_historical_candidate(
        candidate["candidate_id"], {}, corpus, corpus_root=corpus_root
    )

    assert receipt["decision"] == "rejected"
    assert any(expected_blocker in blocker for blocker in receipt["blockers"])
    assert candidate["candidate_status"] == "pending_exact_replay"


def test_issue9656_import_rejects_duplicate_source_aliases_before_writing(
    tmp_path: Path,
) -> None:
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(
        tmp_path, statuses=("not_attempted",)
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["cases"].append(copy.deepcopy(summary["cases"][0]))
    summary["selection"]["case_ids"].append(summary["cases"][0]["case_id"])
    summary["selection"]["case_count"] = 2
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    _refresh_issue9656_bundle(bundle_root)

    corpus = new_corpus()
    corpus_root = tmp_path / "corpus"
    with pytest.raises(CorpusError, match="stable source case aliases"):
        import_issue9656_candidates(
            summary_path, materialized, bundle_root, campaign_root, corpus, corpus_root=corpus_root
        )
    assert corpus["historical_candidates"] == []
    assert not (corpus_root / "historical_candidates").exists()


def test_issue9656_import_rejects_tampered_replay_input_digest(tmp_path: Path) -> None:
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(
        tmp_path, statuses=("not_attempted",)
    )
    matrix = materialized / "cases/case-0000000000000001/replay_input/replay_matrix.yaml"
    matrix.write_text(matrix.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8")

    corpus = new_corpus()
    corpus_root = tmp_path / "corpus"
    with pytest.raises(CorpusError, match="replay input digest differs"):
        import_issue9656_candidates(
            summary_path, materialized, bundle_root, campaign_root, corpus, corpus_root=corpus_root
        )
    assert corpus["historical_candidates"] == []
    assert not (corpus_root / "historical_candidates").exists()


def test_issue9656_import_rejects_tampered_historical_episode_file(tmp_path: Path) -> None:
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(
        tmp_path, statuses=("not_attempted",)
    )
    episode_path = campaign_root / "runs/planner_0/episodes.jsonl"
    episode_path.write_bytes(episode_path.read_bytes() + b"{}\n")

    corpus = new_corpus()
    corpus_root = tmp_path / "corpus"
    with pytest.raises(CorpusError, match="source episode file digest differs"):
        import_issue9656_candidates(
            summary_path, materialized, bundle_root, campaign_root, corpus, corpus_root=corpus_root
        )
    assert corpus["historical_candidates"] == []
    assert not (corpus_root / "historical_candidates").exists()


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_issue"),
    [
        ("row_git_hash", "0" * 40, "row_git_hash"),
        ("campaign_source_revision", "0" * 40, "campaign_source_revision"),
        ("planner_config_hash", "tampered-config", "planner_config_hash"),
    ],
)
def test_issue9656_import_blocks_conflicting_materialized_source_identity(
    tmp_path: Path, field: str, bad_value: str, expected_issue: str
) -> None:
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(
        tmp_path, statuses=("not_attempted",)
    )
    case_path = materialized / "cases/case-0000000000000001/case.json"
    materialized_case = json.loads(case_path.read_text(encoding="utf-8"))
    materialized_case["source"][field] = bad_value
    case_path.write_text(json.dumps(materialized_case, sort_keys=True), encoding="utf-8")

    corpus_root = tmp_path / "corpus"
    corpus, receipt = import_issue9656_candidates(
        summary_path,
        materialized,
        bundle_root,
        campaign_root,
        new_corpus(),
        corpus_root=corpus_root,
    )

    candidate = corpus["historical_candidates"][0]
    assert receipt["source_replay_status_counts"] == {"not_attempted": 1}
    assert candidate["source_replay_status"] == "not_attempted"
    assert candidate["candidate_status"] == "blocked_source_provenance_mismatch"
    assert candidate["source_provenance"]["source_identity_binding_status"] == "blocked"
    assert candidate["source_provenance"]["source_identity_binding_issues"] == [expected_issue]
    assert candidate["target_planner"]["config_hash"] is None
    assert corpus["cases"] == []


def test_issue9656_import_blocks_canonical_algorithm_conflict(tmp_path: Path) -> None:
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(
        tmp_path, statuses=("not_attempted",)
    )
    case_path = materialized / "cases/case-0000000000000001/case.json"
    materialized_case = json.loads(case_path.read_text(encoding="utf-8"))
    materialized_case["planner"]["algorithm_metadata"]["canonical_algorithm"] = "wrong_planner"
    case_path.write_text(json.dumps(materialized_case, sort_keys=True), encoding="utf-8")

    corpus, receipt = import_issue9656_candidates(
        summary_path,
        materialized,
        bundle_root,
        campaign_root,
        new_corpus(),
        corpus_root=tmp_path / "corpus",
    )

    candidate = corpus["historical_candidates"][0]
    assert receipt["source_replay_status_counts"] == {"not_attempted": 1}
    assert candidate["candidate_status"] == "blocked_source_provenance_mismatch"
    assert candidate["source_provenance"]["source_identity_binding_issues"] == [
        "materialized_canonical_algorithm"
    ]
    assert candidate["target_planner"]["canonical_algorithm"] == "planner_0"
    assert candidate["target_planner"]["config_hash"] is None


@pytest.mark.parametrize("failure", ["mismatch", "unavailable"])
def test_issue9656_import_fails_closed_without_historical_map_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(
        tmp_path, statuses=("not_attempted",)
    )
    original = counterexample_corpus._issue9656_historical_git_blob

    def historical_blob(revision: str, relative_path: str) -> bytes:
        if relative_path.startswith("maps/svg_maps/"):
            if failure == "unavailable":
                raise CorpusError("#9656 historical Git blob is unavailable")
            return original(revision, relative_path) + b"historical mismatch"
        return original(revision, relative_path)

    monkeypatch.setattr(counterexample_corpus, "_issue9656_historical_git_blob", historical_blob)
    corpus = new_corpus()
    corpus_root = tmp_path / "corpus"
    expected = (
        "map differs from campaign source revision"
        if failure == "mismatch"
        else "historical Git blob is unavailable"
    )
    with pytest.raises(CorpusError, match=expected):
        import_issue9656_candidates(
            summary_path,
            materialized,
            bundle_root,
            campaign_root,
            corpus,
            corpus_root=corpus_root,
        )
    assert corpus["historical_candidates"] == []
    assert not (corpus_root / "historical_candidates").exists()


@pytest.mark.parametrize("failure", ["mismatch", "unavailable"])
def test_issue9656_import_fails_closed_without_historical_matrix_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(
        tmp_path, statuses=("not_attempted",)
    )
    original = counterexample_corpus._issue9656_historical_git_blob

    def historical_blob(revision: str, relative_path: str) -> bytes:
        if relative_path == _ISSUE9656_SOURCE_MATRIX:
            if failure == "unavailable":
                raise CorpusError("#9656 historical Git blob is unavailable")
            return original(revision, relative_path) + b"historical matrix mismatch"
        return original(revision, relative_path)

    monkeypatch.setattr(counterexample_corpus, "_issue9656_historical_git_blob", historical_blob)
    corpus = new_corpus()
    corpus_root = tmp_path / "corpus"
    expected = (
        "source matrix digest does not match its campaign revision"
        if failure == "mismatch"
        else "historical Git blob is unavailable"
    )
    with pytest.raises(CorpusError, match=expected):
        import_issue9656_candidates(
            summary_path,
            materialized,
            bundle_root,
            campaign_root,
            corpus,
            corpus_root=corpus_root,
        )
    assert corpus["historical_candidates"] == []
    assert not (corpus_root / "historical_candidates").exists()


def test_issue9656_candidate_import_cli_persists_receipt_and_candidates(tmp_path: Path) -> None:
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(
        tmp_path, statuses=("not_attempted",)
    )
    corpus_path = tmp_path / "corpus" / "corpus.json"
    receipt_path = tmp_path / "receipt.json"

    assert (
        corpus_cli_main(
            [
                "import-9656-candidates",
                "--summary",
                str(summary_path),
                "--materialized-root",
                str(materialized),
                "--evidence-root",
                str(bundle_root),
                "--campaign-root",
                str(campaign_root),
                "--corpus",
                str(corpus_path),
                "--corpus-root",
                str(corpus_path.parent),
                "--output",
                str(receipt_path),
            ]
        )
        == 0
    )
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert len(corpus["historical_candidates"]) == 1
    assert corpus["cases"] == []
    assert receipt["decision"] == "imported"


@pytest.mark.parametrize(
    ("tamper", "expected_fragment"),
    [
        ("missing_scenario", "historical_case_invalid"),
        ("missing_provenance", "historical_case_invalid"),
        ("normalization_hash", "historical_case_invalid"),
        ("replay_count", "historical_case_invalid"),
        ("feasibility_verdict", "historical_case_invalid"),
    ],
)
def test_historical_case_admission_fails_closed_and_retains_zero_pilot(
    tmp_path: Path, tamper: str, expected_fragment: str
) -> None:
    with _packet_copy() as payload:
        replay_dir = payload / "historical_issue_1501_failure_0002"
        if tamper == "missing_scenario":
            (replay_dir / "scenario.yaml").unlink()
        elif tamper == "missing_provenance":
            (replay_dir / "replay_1.provenance.json").unlink()
        elif tamper == "normalization_hash":
            receipt = json.loads((payload / "path_normalization.json").read_text())
            receipt["records"][0]["normalized_sha256"] = "0" * 64
            (payload / "path_normalization.json").write_text(json.dumps(receipt))
        else:
            receipt_path = payload / "replay_validation.json"
            receipt = json.loads(receipt_path.read_text())
            case = receipt["historical_issue_1501_case"]
            if tamper == "replay_count":
                case["replay_count"] = 1
            elif tamper == "feasibility_verdict":
                case["dynamic_task_feasibility"] = "feasible"
            receipt_path.write_text(json.dumps(receipt))

        corpus, receipt, _corpus_root = _import(tmp_path, payload)
        assert receipt["decision"] == "rejected"
        assert any(expected_fragment in blocker for blocker in receipt["blockers"])
        assert len(corpus["search_runs"]) == 1
        assert corpus["search_runs"][0]["new_counterexamples_discovered"] == 0
        assert corpus["cases"] == []
        assert corpus["admission_attempts"][-1]["decision"] == "rejected"
        validate_corpus(corpus)


def test_pilot_candidate_table_is_bound_to_the_outer_bundle_checksums(tmp_path: Path) -> None:
    with _packet_copy() as payload:
        table_path = payload / "candidate_evaluations.csv"
        table_path.write_bytes(table_path.read_bytes() + b"tampered\n")

        corpus, receipt, _corpus_root = _import(tmp_path, payload)

    assert receipt["decision"] == "rejected"
    assert any("pilot_evidence_invalid" in blocker for blocker in receipt["blockers"])
    assert corpus["search_runs"] == []
    assert corpus["cases"] == []


def test_bundle_checksum_sidecar_must_match_the_manifest(tmp_path: Path) -> None:
    with _packet_copy() as payload:
        checksum_path = payload.parent / "checksums.sha256"
        checksum_path.write_text(
            checksum_path.read_text().replace("payload/summary.json", "payload/tampered.json")
        )

        corpus, receipt, _corpus_root = _import(tmp_path, payload)

    assert receipt["decision"] == "rejected"
    assert any("pilot_evidence_invalid" in blocker for blocker in receipt["blockers"])
    assert corpus["search_runs"] == []


def _refresh_bundle_checksum_for_payload(payload: Path, relative: str) -> None:
    """Refresh test-only bundle receipts after a deliberate semantic mutation."""
    manifest_path = payload.parent / "evidence_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    target = payload / relative
    record = next(item for item in manifest["files"] if item["path"] == relative)
    record["size_bytes"] = target.stat().st_size
    record["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest["totals"]["total_bytes"] = sum(item["size_bytes"] for item in manifest["files"])
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksum_lines = [
        f"{item['sha256']}  payload/{item['path']}"
        for item in sorted(manifest["files"], key=lambda row: row["path"])
    ]
    (payload.parent / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")


def test_replay_records_with_different_selected_metric_identity_are_rejected(
    tmp_path: Path,
) -> None:
    with _packet_copy() as payload:
        replay_path = payload / "historical_issue_1501_failure_0002/replay_2.jsonl"
        episode = json.loads(replay_path.read_text())
        episode["metrics"]["min_clearance"] += 0.25
        replay_path.write_text(json.dumps(episode, separators=(",", ":")) + "\n")
        path_receipt = payload / "path_normalization.json"
        normalization = json.loads(path_receipt.read_text())
        replay_row = next(
            row
            for row in normalization["records"]
            if row["path"] == "historical_issue_1501_failure_0002/replay_2.jsonl"
        )
        replay_row["normalized_sha256"] = hashlib.sha256(replay_path.read_bytes()).hexdigest()
        path_receipt.write_text(json.dumps(normalization))
        _refresh_bundle_checksum_for_payload(
            payload, "historical_issue_1501_failure_0002/replay_2.jsonl"
        )
        _refresh_bundle_checksum_for_payload(payload, "path_normalization.json")

        corpus, receipt, _corpus_root = _import(tmp_path, payload)
        assert receipt["decision"] == "rejected"
        assert any(
            "current replay records disagree on selected event/metric identity" in blocker
            for blocker in receipt["blockers"]
        )
        assert corpus["cases"] == []


def test_replayed_historical_origin_is_not_mislabeled_as_raw_historical_match(
    tmp_path: Path,
) -> None:
    corpus, receipt, _corpus_root = _import(tmp_path)
    case = corpus["cases"][0]
    assert receipt["decision"] == "admitted"
    assert (
        case["discovery"]["search_source"]["historical_source_revision"]
        != (case["replay_receipt"]["replay_revision"])
    )
    assert case["replay_receipt"]["target_and_replay_revision_match"] is True
    assert case["replay_receipt"]["historical_origin_match"] == (
        "not_verifiable_original_raw_episode_absent"
    )
    assert case["source_evidence"]["historical_raw_episode_status"] == "not_archived"


def test_duplicate_import_is_deterministic_and_later_planner_solve_keeps_case(
    tmp_path: Path,
) -> None:
    payload = _SOURCE_PACKET
    corpus_root = tmp_path / "corpus"
    corpus, first = import_issue9645_packet(payload, new_corpus(), corpus_root=corpus_root)
    case_id = corpus["cases"][0]["case_id"]
    case_hash = corpus["cases"][0]["effective_scenario_sha256"]

    corpus, second = import_issue9645_packet(payload, corpus, corpus_root=corpus_root)
    assert first["decision"] == "admitted"
    assert second["decision"] == "duplicate"
    assert second["case_id"] == case_id
    assert second["candidate_identity"] == case_hash
    assert len(corpus["cases"]) == 1
    assert len(corpus["planner_evaluations"]) == 2
    assert len(corpus["admission_attempts"]) == 2

    old_case = corpus["cases"][0]
    config_identity = old_case["target_planner"]["config_identity"]
    old_status = recompute_planner_status(
        corpus,
        planner_id="goal",
        planner_config_identity=config_identity,
        corpus_root=corpus_root,
    )
    assert old_status["status_counts"]["unknown"] == 1
    assert "replay_input_binding_unknown_historical" in old_status["cases"][0]["reason_codes"]

    unsupported_claim = copy.deepcopy(corpus["planner_evaluations"][0])
    unsupported_claim.update(
        {
            "planner_id": "goal-optimized",
            "planner_config_identity": "goal-config-v2",
            "outcome": {
                "collision_event": False,
                "route_complete": True,
                "timeout_event": False,
            },
            "termination_reason": "success",
            "metrics": {
                "success": True,
                "collisions": 0,
                "total_collision_count": 0,
            },
            "error": None,
        }
    )
    unsupported_claim.pop("replay_receipt")
    with pytest.raises(CorpusError, match="replay_receipt_missing"):
        append_planner_evaluation(corpus, unsupported_claim, corpus_root=corpus_root)
    assert len(corpus["planner_evaluations"]) == 2

    forged_solve = copy.deepcopy(corpus["planner_evaluations"][0])
    forged_solve.update(
        {
            "planner_id": "goal-optimized",
            "planner_config_identity": "goal-config-v2",
            "outcome": {"collision_event": False, "route_complete": True, "timeout_event": False},
            "termination_reason": "success",
            "metrics": {"success": True, "collisions": 0, "total_collision_count": 0},
        }
    )
    forged_receipt = forged_solve["replay_receipt"]
    forged_receipt.update(
        {
            "planner_id": "goal-optimized",
            "planner_config_identity": "goal-config-v2",
            "outcome": forged_solve["outcome"],
            "termination_reason": "success",
            "metrics": forged_solve["metrics"],
        }
    )
    with pytest.raises(CorpusError, match="replay_artifact_planner_id_mismatch"):
        append_planner_evaluation(corpus, forged_solve, corpus_root=corpus_root)
    assert len(corpus["planner_evaluations"]) == 2

    solved = _append_episode_evaluation(
        corpus,
        corpus_root,
        planner_id="goal-optimized",
        config_identity="goal-config-v2",
        execution_mode="native",
        outcome={"collision_event": False, "route_complete": True, "timeout_event": False},
    )
    optimized_status = recompute_planner_status(
        corpus,
        planner_id="goal-optimized",
        planner_config_identity="goal-config-v2",
        corpus_root=corpus_root,
    )
    assert optimized_status["status_counts"]["solved"] == 1
    assert len(corpus["cases"]) == 1
    assert len(corpus["planner_evaluations"]) == 3
    assert sum(row["planner_id"] == "goal" for row in corpus["planner_evaluations"]) == 2

    artifact_path = corpus_root / solved["replay_receipt"]["artifact_path"]
    artifact_path.write_text(artifact_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    after_tamper = recompute_planner_status(
        corpus,
        planner_id="goal-optimized",
        planner_config_identity="goal-config-v2",
        corpus_root=corpus_root,
    )
    assert after_tamper["status_counts"]["unknown"] == 1
    assert "replay_artifact_checksum_mismatch" in after_tamper["cases"][0]["reason_codes"]
    assert len(corpus["cases"]) == 1
    assert len(corpus["planner_evaluations"]) == 3


def test_incomplete_evaluation_is_retained_as_unknown_not_discarded(tmp_path: Path) -> None:
    corpus, _receipt, _corpus_root = _import(tmp_path)
    case = corpus["cases"][0]
    incomplete = {
        "case_id": case["case_id"],
        "effective_scenario_sha256": case["effective_scenario_sha256"],
        "planner_id": "candidate-planner",
        "planner_config_identity": "config-under-test",
        "source_revision": "unknown",
        "episode_sha256": None,
        "execution_mode": "not_observed",
        "readiness_status": "unknown",
        "availability_status": "unavailable",
        "fallback_or_degraded": None,
        "evidence_status": "failed",
        "error": "planner process exited before emitting an episode",
        "outcome": None,
        "termination_reason": None,
        "metrics": {},
    }
    append_planner_evaluation(corpus, incomplete)

    status = recompute_planner_status(
        corpus,
        planner_id="candidate-planner",
        planner_config_identity="config-under-test",
        corpus_root=_corpus_root,
    )
    assert status["status_counts"]["unknown"] == 1
    assert status["cases"][0]["unknown_observation_count"] == 1
    assert status["cases"][0]["reason_codes"] == ["evaluation_evidence_failed"]
    assert len(corpus["planner_evaluations"]) == 3
    incomplete_row = next(
        row for row in corpus["planner_evaluations"] if row["planner_id"] == "candidate-planner"
    )
    assert incomplete_row["episode_sha256"] is None
    validate_corpus(corpus)


@pytest.mark.parametrize("episode_status", ["failure", "placeholder"])
def test_status_termination_mismatch_cannot_recompute_solved(
    tmp_path: Path, episode_status: str
) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    planner_id = f"status-mismatch-{episode_status}"
    config_identity = f"config-{episode_status}"
    _append_episode_evaluation(
        corpus,
        corpus_root,
        planner_id=planner_id,
        config_identity=config_identity,
        execution_mode="native",
        outcome={"collision_event": False, "route_complete": True, "timeout_event": False},
    )
    row = next(item for item in corpus["planner_evaluations"] if item["planner_id"] == planner_id)
    artifact = corpus_root / row["replay_receipt"]["artifact_path"]
    record = json.loads(artifact.read_text(encoding="utf-8"))
    record["status"] = episode_status
    artifact.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    row["episode_sha256"] = artifact_sha256
    row["replay_receipt"].update(
        {
            "episode_sha256": artifact_sha256,
            "artifact_sha256": artifact_sha256,
            "selected_event_identity": counterexample_corpus._selected_event_identity(record),
        }
    )
    _refresh_evaluation_id(row)

    status = recompute_planner_status(
        corpus,
        planner_id=planner_id,
        planner_config_identity=config_identity,
        corpus_root=corpus_root,
    )

    assert status["status_counts"]["solved"] == 0
    assert status["status_counts"]["unsolved"] == 0
    assert status["status_counts"]["unknown"] == 1
    assert "replay_artifact_status_termination_mismatch" in status["cases"][0]["reason_codes"]
    with pytest.raises(CorpusError, match="replay_artifact_status_termination_mismatch"):
        create_planner_replay_receipt(
            row,
            corpus["cases"][0],
            artifact_path=row["replay_receipt"]["artifact_path"],
            corpus_root=corpus_root,
        )


@pytest.mark.parametrize(
    ("termination_reason", "episode_status", "outcome", "expected_status"),
    [
        (
            "success",
            "success",
            {"collision_event": False, "route_complete": True, "timeout_event": False},
            "solved",
        ),
        (
            "collision",
            "collision",
            {"collision_event": True, "route_complete": False, "timeout_event": False},
            "unsolved",
        ),
        (
            "max_steps",
            "failure",
            {"collision_event": False, "route_complete": False, "timeout_event": False},
            "unsolved",
        ),
    ],
)
def test_canonical_episode_status_controls_remain_classifiable(
    tmp_path: Path,
    termination_reason: str,
    episode_status: str,
    outcome: dict[str, bool],
    expected_status: str,
) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    planner_id = f"canonical-status-{episode_status}"
    config_identity = f"config-{episode_status}"
    _append_episode_evaluation(
        corpus,
        corpus_root,
        planner_id=planner_id,
        config_identity=config_identity,
        execution_mode="native",
        outcome=outcome,
        termination_reason=termination_reason,
    )

    status = recompute_planner_status(
        corpus,
        planner_id=planner_id,
        planner_config_identity=config_identity,
        corpus_root=corpus_root,
    )

    assert status["status_counts"][expected_status] == 1
    assert status["status_counts"]["unknown"] == 0
    assert status["cases"][0]["reason_codes"] == []
    row = next(item for item in corpus["planner_evaluations"] if item["planner_id"] == planner_id)
    record = json.loads((corpus_root / row["replay_receipt"]["artifact_path"]).read_text())
    assert record["status"] == episode_status
    assert record["termination_reason"] == termination_reason


def test_complete_evaluation_with_unknown_revision_stays_unknown(tmp_path: Path) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    planner_id = "unknown-revision-planner"
    config_identity = "unknown-revision-config"
    _append_episode_evaluation(
        corpus,
        corpus_root,
        planner_id=planner_id,
        config_identity=config_identity,
        execution_mode="native",
        outcome={"collision_event": False, "route_complete": True, "timeout_event": False},
    )
    row = next(item for item in corpus["planner_evaluations"] if item["planner_id"] == planner_id)
    artifact = corpus_root / row["replay_receipt"]["artifact_path"]
    record = json.loads(artifact.read_text(encoding="utf-8"))
    record["git_hash"] = "unknown"
    record["event_ledger"]["software_commit"] = "unknown"
    artifact.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    row["source_revision"] = "unknown"
    row["episode_sha256"] = artifact_sha256
    row["replay_receipt"].update(
        {
            "source_revision": "unknown",
            "episode_sha256": artifact_sha256,
            "artifact_sha256": artifact_sha256,
            "selected_event_identity": counterexample_corpus._selected_event_identity(record),
        }
    )
    _refresh_evaluation_id(row)

    status = recompute_planner_status(
        corpus,
        planner_id=planner_id,
        planner_config_identity=config_identity,
        corpus_root=corpus_root,
    )

    assert status["status_counts"]["solved"] == 0
    assert status["status_counts"]["unknown"] == 1
    assert "source_revision_invalid" in status["cases"][0]["reason_codes"]


def test_invalid_run_replay_stays_unknown_not_unsolved(tmp_path: Path) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    planner_id = "invalid-run-planner"
    config_identity = "invalid-run-config"
    _append_episode_evaluation(
        corpus,
        corpus_root,
        planner_id=planner_id,
        config_identity=config_identity,
        execution_mode="native",
        outcome={"collision_event": True, "route_complete": False, "timeout_event": False},
    )
    row = next(item for item in corpus["planner_evaluations"] if item["planner_id"] == planner_id)
    artifact = corpus_root / row["replay_receipt"]["artifact_path"]
    record = json.loads(artifact.read_text(encoding="utf-8"))
    outcome = {"collision_event": False, "route_complete": False, "timeout_event": False}
    metrics = {"success": False, "collisions": 0, "total_collision_count": 0}
    record["status"] = "invalid"
    record["termination_reason"] = "error"
    record["outcome"] = outcome
    record["metrics"].update(metrics)
    record["event_ledger"]["exact_events"] = {
        "collision": False,
        "goal_reached": False,
        "timeout": False,
        "invalid_run": True,
    }
    record["event_ledger"]["collision_events"] = []
    record["event_ledger"]["reconciliation"]["collision_metric_value"] = 0
    artifact.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    row["outcome"] = outcome
    row["termination_reason"] = "error"
    row["metrics"] = metrics
    row["episode_sha256"] = artifact_sha256
    row["replay_receipt"].update(
        {
            "outcome": outcome,
            "termination_reason": "error",
            "metrics": metrics,
            "episode_sha256": artifact_sha256,
            "artifact_sha256": artifact_sha256,
            "selected_event_identity": counterexample_corpus._selected_event_identity(record),
        }
    )
    _refresh_evaluation_id(row)

    status = recompute_planner_status(
        corpus,
        planner_id=planner_id,
        planner_config_identity=config_identity,
        corpus_root=corpus_root,
    )

    assert status["status_counts"]["unsolved"] == 0
    assert status["status_counts"]["unknown"] == 1
    assert "replay_artifact_invalid_run" in status["cases"][0]["reason_codes"]


def test_replay_artifact_path_escape_stays_unknown(tmp_path: Path) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    planner_id = "path-escape-planner"
    config_identity = "path-escape-config"
    _append_episode_evaluation(
        corpus,
        corpus_root,
        planner_id=planner_id,
        config_identity=config_identity,
        execution_mode="native",
        outcome={"collision_event": False, "route_complete": True, "timeout_event": False},
    )
    row = next(item for item in corpus["planner_evaluations"] if item["planner_id"] == planner_id)
    row["replay_receipt"]["artifact_path"] = "../outside.jsonl"

    validate_corpus(corpus)
    status = recompute_planner_status(
        corpus,
        planner_id=planner_id,
        planner_config_identity=config_identity,
        corpus_root=corpus_root,
    )

    assert status["status_counts"]["unknown"] == 1
    assert any(
        reason.startswith("replay_artifact_path_invalid:")
        and "unsafe replay artifact path" in reason
        for reason in status["cases"][0]["reason_codes"]
    )
    assert len(corpus["cases"]) == 1
    assert len(corpus["planner_evaluations"]) == 3


def test_legacy_complete_evaluation_without_receipt_is_retained_as_unknown(
    tmp_path: Path,
) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    _append_episode_evaluation(
        corpus,
        corpus_root,
        planner_id="legacy-planner",
        config_identity="legacy-config",
        execution_mode="native",
        outcome={"collision_event": False, "route_complete": True, "timeout_event": False},
    )
    legacy = copy.deepcopy(corpus)
    legacy_row = next(
        row for row in legacy["planner_evaluations"] if row["planner_id"] == "legacy-planner"
    )
    legacy_row.pop("replay_receipt")

    validate_corpus(legacy)
    status = recompute_planner_status(
        legacy,
        planner_id="legacy-planner",
        planner_config_identity="legacy-config",
        corpus_root=corpus_root,
    )
    assert status["status_counts"]["unknown"] == 1
    assert status["cases"][0]["reason_codes"] == ["replay_receipt_missing"]
    assert len(legacy["cases"]) == len(corpus["cases"]) == 1
    assert len(legacy["planner_evaluations"]) == len(corpus["planner_evaluations"]) == 3


@pytest.mark.parametrize("execution_mode", ["adapter", "mixed"])
def test_benchmark_capable_adapter_modes_count_and_fallback_stays_unknown(
    tmp_path: Path, execution_mode: str
) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    planner_id = f"{execution_mode}-planner"
    config_identity = f"{execution_mode}-config"
    _append_episode_evaluation(
        corpus,
        corpus_root,
        planner_id=planner_id,
        config_identity=config_identity,
        execution_mode=execution_mode,
    )

    status = recompute_planner_status(
        corpus,
        planner_id=planner_id,
        planner_config_identity=config_identity,
        corpus_root=corpus_root,
    )
    assert status["status_counts"]["unsolved"] == 1
    assert status["cases"][0]["valid_observation_count"] == 1

    fallback_planner = f"{execution_mode}-fallback-planner"
    _append_episode_evaluation(
        corpus,
        corpus_root,
        planner_id=fallback_planner,
        config_identity=f"{execution_mode}-fallback-config",
        execution_mode=execution_mode,
        degraded=True,
    )
    fallback_status = recompute_planner_status(
        corpus,
        planner_id=fallback_planner,
        planner_config_identity=f"{execution_mode}-fallback-config",
        corpus_root=corpus_root,
    )
    assert fallback_status["status_counts"]["unknown"] == 1
    assert "fallback_or_degraded_execution" in fallback_status["cases"][0]["reason_codes"]


def test_regression_slice_export_is_stable_and_binds_scenario_route_and_config(
    tmp_path: Path,
) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    first = export_regression_slice(
        corpus, corpus_root=corpus_root, output_dir=tmp_path / "slice-a"
    )
    second = export_regression_slice(
        corpus, corpus_root=corpus_root, output_dir=tmp_path / "slice-b"
    )
    assert first == second
    assert (tmp_path / "slice-a/manifest.json").read_bytes() == (
        tmp_path / "slice-b/manifest.json"
    ).read_bytes()
    assert (tmp_path / "slice-a/replay_matrix.yaml").read_bytes() == (
        tmp_path / "slice-b/replay_matrix.yaml"
    ).read_bytes()
    case = first["cases"][0]
    slice_root = tmp_path / "slice-a"
    assert case["planner_config_path"] == (f"planner_configs/{case['case_id']}.yaml")
    assert case["case_id"] == f"case-{case['effective_scenario_sha256']}"
    identity_mapping = case["identity_mapping"]
    assert identity_mapping["schema_version"] == "adversarial-slice-identity-mapping.v1"
    assert identity_mapping["verification_status"] == "source_and_export_hashes_verified"
    assert identity_mapping["normalization_fields"] == ["route_overrides_file"]
    assert identity_mapping["source_effective_scenario_sha256"] == case["effective_scenario_sha256"]
    assert (
        identity_mapping["source_route_overrides_file"]
        != identity_mapping["exported_route_overrides_file"]
    )
    matrix = yaml.safe_load((slice_root / "replay_matrix.yaml").read_text(encoding="utf-8"))
    exported_scenario = matrix["scenarios"][0]
    exported_route = yaml.safe_load(
        (slice_root / identity_mapping["exported_route_overrides_file"]).read_text(encoding="utf-8")
    )
    assert (
        counterexample_corpus.compute_case_effective_scenario_hash(
            exported_scenario,
            exported_route,
            identity_mapping["map_assets"],
        )
        == identity_mapping["exported_effective_scenario_sha256"]
    )
    for asset in identity_mapping["map_assets"]:
        stored_asset = slice_root / asset["path"]
        assert hashlib.sha256(stored_asset.read_bytes()).hexdigest() == asset["sha256"]
    source_scenario_path = corpus_root / corpus["cases"][0]["inputs"]["scenario_path"]
    source_route_path = corpus_root / corpus["cases"][0]["inputs"]["route_overrides_path"]
    source_scenario = yaml.safe_load(source_scenario_path.read_text(encoding="utf-8"))["scenarios"][
        0
    ]
    source_route = yaml.safe_load(source_route_path.read_text(encoding="utf-8"))
    assert (
        counterexample_corpus.compute_case_effective_scenario_hash(
            source_scenario,
            source_route,
            corpus["cases"][0]["inputs"]["map_assets"],
        )
        == identity_mapping["source_effective_scenario_sha256"]
    )
    assert (
        hashlib.sha256((slice_root / case["planner_config_path"]).read_bytes()).hexdigest()
        == (case["planner_config_sha256"])
    )
    assert (slice_root / "results").is_dir()
    assert case["replay_command"].startswith("ROBOT_SF_MAP_REGISTRY=maps/registry.yaml ")
    assert "uv run robot_sf_bench run --matrix replay_matrix.yaml" in case["replay_command"]
    validate_corpus(corpus)


@pytest.mark.parametrize("asset_role", ["map", "map_registry"])
def test_map_asset_tampering_invalidates_case_identity_and_slice_export(
    tmp_path: Path, asset_role: str
) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    case = corpus["cases"][0]
    asset = next(item for item in case["inputs"]["map_assets"] if item["role"] == asset_role)
    asset_path = corpus_root / asset["path"]
    original_bytes = asset_path.read_bytes()
    changed_bytes = original_bytes + b"\n# adversarial test mutation\n"
    changed_sha256 = hashlib.sha256(changed_bytes).hexdigest()
    asset_path.write_bytes(changed_bytes)

    changed_identity_assets = copy.deepcopy(case["inputs"]["map_assets"])
    next(item for item in changed_identity_assets if item["role"] == asset_role)["sha256"] = (
        changed_sha256
    )
    scenario_document = yaml.safe_load(
        (corpus_root / case["inputs"]["scenario_path"]).read_text(encoding="utf-8")
    )
    route_payload = yaml.safe_load(
        (corpus_root / case["inputs"]["route_overrides_path"]).read_text(encoding="utf-8")
    )
    changed_hash = counterexample_corpus.compute_case_effective_scenario_hash(
        scenario_document["scenarios"][0], route_payload, changed_identity_assets
    )
    assert changed_hash != case["effective_scenario_sha256"]
    with pytest.raises(CorpusError, match="map asset|map registry|map file|map identity"):
        validate_corpus(corpus, corpus_root=corpus_root)
    with pytest.raises(CorpusError, match="map asset|map registry|map file|map identity"):
        export_regression_slice(
            corpus,
            corpus_root=corpus_root,
            output_dir=tmp_path / f"slice-{asset_role}",
        )


def test_append_evaluation_rejects_case_hash_mismatch(tmp_path: Path) -> None:
    corpus, _receipt, _corpus_root = _import(tmp_path)
    wrong = copy.deepcopy(corpus["planner_evaluations"][0])
    wrong["effective_scenario_sha256"] = "0" * 64
    with pytest.raises(CorpusError, match="replay_receipt_effective_scenario_sha256_mismatch"):
        append_planner_evaluation(corpus, wrong)


def test_validate_corpus_rejects_planner_evaluation_for_absent_case(
    tmp_path: Path,
) -> None:
    corpus, _receipt, _corpus_root = _import(tmp_path)
    corpus["planner_evaluations"][0]["case_id"] = f"case-{'0' * 64}"

    with pytest.raises(CorpusError, match="planner evaluation references absent case"):
        validate_corpus(corpus)


def test_validate_corpus_rejects_excluded_invalid_and_incomplete_case_records(
    tmp_path: Path,
) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    invalid = copy.deepcopy(corpus)
    invalid["cases"][0]["structural_validation"]["status"] = "invalid"
    invalid["cases"][0]["admissibility"]["verdict"] = "excluded"
    with pytest.raises(CorpusError):
        validate_corpus(invalid)
    with pytest.raises(CorpusError):
        recompute_planner_status(
            invalid,
            planner_id="goal",
            planner_config_identity=invalid["cases"][0]["target_planner"]["config_identity"],
            corpus_root=corpus_root,
        )
    with pytest.raises(CorpusError):
        export_regression_slice(
            invalid,
            corpus_root=corpus_root,
            output_dir=tmp_path / "invalid-slice",
        )

    incomplete = copy.deepcopy(corpus)
    del incomplete["cases"][0]["target_planner"]["configuration_snapshot"]
    with pytest.raises(CorpusError):
        validate_corpus(incomplete)


def test_corpus_cli_import_status_and_slice_work_without_simulator_run(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_path = corpus_root / "corpus.json"
    receipt_path = tmp_path / "admission.json"
    assert (
        corpus_cli_main(
            [
                "import-9645",
                "--payload",
                str(_SOURCE_PACKET),
                "--corpus",
                str(corpus_path),
                "--corpus-root",
                str(corpus_root),
                "--output",
                str(receipt_path),
            ]
        )
        == 0
    )
    receipt = json.loads(receipt_path.read_text())
    assert receipt["decision"] == "admitted"

    status_path = tmp_path / "status.json"
    case = json.loads(corpus_path.read_text())["cases"][0]
    assert (
        corpus_cli_main(
            [
                "status",
                "--corpus",
                str(corpus_path),
                "--planner-id",
                "goal",
                "--planner-config-identity",
                case["target_planner"]["config_identity"],
                "--output",
                str(status_path),
            ]
        )
        == 0
    )
    assert json.loads(status_path.read_text())["status_counts"]["unknown"] == 1

    slice_path = tmp_path / "slice"
    assert (
        corpus_cli_main(
            [
                "export-slice",
                "--corpus",
                str(corpus_path),
                "--corpus-root",
                str(corpus_root),
                "--output-dir",
                str(slice_path),
                "--output",
                str(tmp_path / "slice-manifest.json"),
            ]
        )
        == 0
    )
    manifest = json.loads((tmp_path / "slice-manifest.json").read_text())
    assert manifest["case_count"] == 1
    assert manifest["cases"][0]["replay_input_binding_status"] == "unknown_historical"
    source_scenario = yaml.safe_load(
        (corpus_root / case["inputs"]["scenario_path"]).read_text(encoding="utf-8")
    )["scenarios"][0]
    exported_scenario = yaml.safe_load((slice_path / "replay_matrix.yaml").read_text())[
        "scenarios"
    ][0]
    assert scenario_semantic_sha256(source_scenario, seed=case["scenario_seed"]) == (
        scenario_semantic_sha256(exported_scenario, seed=case["scenario_seed"])
    )
