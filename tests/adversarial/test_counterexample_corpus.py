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

from robot_sf.adversarial.counterexample_corpus import (
    CorpusError,
    append_planner_evaluation,
    export_regression_slice,
    import_issue9645_packet,
    new_corpus,
    recompute_planner_status,
    validate_corpus,
)
from scripts.tools.manage_adversarial_counterexample_corpus import main as corpus_cli_main

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PACKET = _REPO_ROOT / "tests/fixtures/adversarial_counterexample_corpus/issue_9645/payload"
_SOURCE_BUNDLE = _SOURCE_PACKET.parent


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
        corpus, planner_id="goal", planner_config_identity=config_identity
    )
    assert old_status["status_counts"]["unsolved"] == 1

    solved = copy.deepcopy(corpus["planner_evaluations"][0])
    solved.update(
        {
            "planner_id": "goal-optimized",
            "planner_config_identity": "goal-config-v2",
            "source_revision": "b" * 40,
            "episode_sha256": "a" * 64,
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
    append_planner_evaluation(corpus, solved)
    optimized_status = recompute_planner_status(
        corpus, planner_id="goal-optimized", planner_config_identity="goal-config-v2"
    )
    assert optimized_status["status_counts"]["solved"] == 1
    assert len(corpus["cases"]) == 1
    assert len(corpus["planner_evaluations"]) == 3
    assert sum(row["planner_id"] == "goal" for row in corpus["planner_evaluations"]) == 2


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
    )
    assert status["status_counts"]["unknown"] == 1
    assert status["cases"][0]["unknown_observation_count"] == 1
    assert status["cases"][0]["reason_codes"] == ["evaluation_evidence_failed"]
    assert len(corpus["planner_evaluations"]) == 3
    assert corpus["planner_evaluations"][-1]["episode_sha256"] is None
    validate_corpus(corpus)


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
    with pytest.raises(CorpusError, match="input hash does not match"):
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
