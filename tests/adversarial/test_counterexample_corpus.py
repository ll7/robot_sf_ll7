"""Fixture-backed tests for the versioned adversarial challenge corpus."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from robot_sf.adversarial import counterexample_corpus
from robot_sf.adversarial.counterexample_corpus import (
    CorpusError,
    append_planner_evaluation,
    create_planner_replay_receipt,
    export_regression_slice,
    import_issue9645_packet,
    import_issue9656_candidates,
    new_corpus,
    recompute_planner_status,
    validate_corpus,
)
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
        case_id = f"case-{index + 1:016x}"
        planner_config_identity = f"{index + 1:016x}"
        case_dir = materialized / "cases" / case_id
        replay_dir = case_dir / "replay_input"
        replay_dir.mkdir(parents=True)
        episode_file = f"runs/planner_{index}/episodes.jsonl"
        source_row = {
            "algo": f"planner_{index}",
            "algorithm_metadata": {
                "canonical_algorithm": f"planner_{index}",
                "config_hash": planner_config_identity,
            },
            "episode_id": f"episode-{index}",
            "git_hash": _ISSUE9656_SOURCE_REVISION,
            "scenario_params": {"algo_config_hash": f"scenario-config-{index}"},
            "scenario_id": f"scenario_{index}",
            "seed": 100 + index,
        }
        raw_line = json.dumps(source_row, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        episode_path = campaign_root / episode_file
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        episode_path.write_bytes(raw_line)
        record_sha = hashlib.sha256(raw_line).hexdigest()
        source_record = {
            "episode_file": episode_file,
            "episode_file_sha256": hashlib.sha256(raw_line).hexdigest(),
            "line_number": 1,
            "record_sha256": record_sha,
        }
        anomaly = ["collision_event_without_positive_collision_metric"] if index == 0 else []
        criticality = {
            "anomalies": anomaly,
            "evidence_tier": "diagnostic_only",
            "metrics": {"collisions_metric": 0.0, "minimum_clearance_m": 0.5 + index},
            "outcome": {"collision_event": False, "route_complete": False, "timeout_event": True},
        }
        replay = {"attempted": status == "mismatch_different_revision", "status": status}
        planner_config = "planner_option: value\n"
        (replay_dir / "planner_config.yaml").write_text(planner_config, encoding="utf-8")
        planner_config_sha = hashlib.sha256(planner_config.encode()).hexdigest()
        scenario_matrix = {
            "map_search_paths": [str(map_path.parent)],
            "scenarios": [
                {
                    "algo": f"planner_{index}",
                    "id": f"scenario_{index}",
                    "map_file": str(map_path),
                    "seeds": [100 + index],
                }
            ],
        }
        matrix_text = yaml.safe_dump(scenario_matrix, sort_keys=True)
        (replay_dir / "replay_matrix.yaml").write_text(matrix_text, encoding="utf-8")
        matrix_sha = hashlib.sha256(matrix_text.encode()).hexdigest()
        replay_input = {
            "planner_config_path": "replay_input/planner_config.yaml",
            "planner_config_sha256": planner_config_sha,
            "replay_eligible": status != "unavailable_model_artifact",
            "replay_ineligibility": (
                "model artifact unavailable" if status == "unavailable_model_artifact" else None
            ),
            "scenario_matrix_path": "replay_input/replay_matrix.yaml",
            "scenario_matrix_sha256": matrix_sha,
            "status": "materialized",
        }
        summary_case = {
            "benchmark_eligible": True,
            "case_id": case_id,
            "criticality": criticality,
            "planner_key": f"planner_{index}",
            "replay": replay,
            "replay_input": replay_input,
            "scenario_family": f"family_{index}",
            "scenario_id": f"scenario_{index}",
            "seed": 100 + index,
            "selected_groups": ["timeout_event"],
            "source_record": source_record,
            "source_showcase_renderer": {"status": "unavailable", "artifacts": []},
        }
        materialized_case = {
            "case_file": f"cases/{case_id}/case.json",
            "case_id": case_id,
            "criticality": criticality,
            "planner": {
                "algorithm_metadata": {
                    "canonical_algorithm": f"planner_{index}",
                    "config_hash": planner_config_identity,
                },
                "key": f"planner_{index}",
            },
            "replay": replay,
            "replay_input": replay_input,
            "scenario": {
                "scenario_id": f"scenario_{index}",
                "scenario_family": f"family_{index}",
                "seed": 100 + index,
            },
            "schema_version": "benchmark-hard-case.v1",
            "selection": {"selected_groups": ["timeout_event"]},
            "source": {
                **source_record,
                "bundle_sha256": "a" * 64,
                "campaign_id": "campaign-fixture",
                "campaign_source_revision": _ISSUE9656_SOURCE_REVISION,
                "planner_config_hash": planner_config_identity,
                "planner_key": f"planner_{index}",
                "row_git_hash": _ISSUE9656_SOURCE_REVISION,
                "scenario_matrix": _ISSUE9656_SOURCE_MATRIX,
                "scenario_matrix_sha256": _ISSUE9656_SOURCE_MATRIX_SHA256,
            },
            "source_record": source_row,
        }
        case_file = case_dir / "case.json"
        case_file.write_text(json.dumps(materialized_case, sort_keys=True), encoding="utf-8")
        summary_cases.append(summary_case)
        manifest_cases.append({"case_file": f"cases/{case_id}/case.json", **summary_case})
        source_status_counts[status] = source_status_counts.get(status, 0) + 1
        for item in anomaly:
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


def _append_episode_evaluation(
    corpus: dict[str, object],
    corpus_root: Path,
    *,
    planner_id: str,
    config_identity: str,
    execution_mode: str,
    outcome: dict[str, bool] | None = None,
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
    if selected_outcome["route_complete"]:
        record["termination_reason"] = "success"
        record["status"] = "success"
        record["metrics"]["success"] = 1
        record["metrics"]["collisions"] = 0
        record["metrics"]["total_collision_count"] = 0
        record["event_ledger"]["reconciliation"]["collision_metric_value"] = 0
    if degraded:
        record["integrity"]["effective_view"]["degraded"] = True
    else:
        record["integrity"]["effective_view"]["degraded"] = False

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
    assert old_status["status_counts"]["unsolved"] == 1

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
        counterexample_corpus.compute_effective_scenario_hash(exported_scenario, exported_route)
        == identity_mapping["exported_effective_scenario_sha256"]
    )
    source_scenario_path = corpus_root / corpus["cases"][0]["inputs"]["scenario_path"]
    source_route_path = corpus_root / corpus["cases"][0]["inputs"]["route_overrides_path"]
    source_scenario = yaml.safe_load(source_scenario_path.read_text(encoding="utf-8"))["scenarios"][
        0
    ]
    source_route = yaml.safe_load(source_route_path.read_text(encoding="utf-8"))
    assert (
        counterexample_corpus.compute_effective_scenario_hash(source_scenario, source_route)
        == identity_mapping["source_effective_scenario_sha256"]
    )
    assert (
        hashlib.sha256((slice_root / case["planner_config_path"]).read_bytes()).hexdigest()
        == (case["planner_config_sha256"])
    )
    assert (slice_root / "results").is_dir()
    assert case["replay_command"].startswith(
        "uv run robot_sf_bench run --matrix replay_matrix.yaml"
    )
    validate_corpus(corpus)


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
    assert json.loads(status_path.read_text())["status_counts"]["unsolved"] == 1

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
    assert json.loads((tmp_path / "slice-manifest.json").read_text())["case_count"] == 1
