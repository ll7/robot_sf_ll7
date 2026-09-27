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
    load_corpus,
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
_ISSUE9645_FIXTURE_RECEIPT = _SOURCE_BUNDLE / "fixture_source_receipt.json"
_ISSUE9656_PROMOTED_BUNDLE = (
    _REPO_ROOT / "tests/fixtures/adversarial_counterexample_corpus/issue_9656_current_head"
)
_ISSUE9656_PROMOTED_SUMMARY = _ISSUE9656_PROMOTED_BUNDLE / "payload/summary.json"
_ISSUE9656_PROMOTED_MANIFEST = (
    _ISSUE9656_PROMOTED_BUNDLE / "payload/current_head_no_replay_manifest.json"
)
_ISSUE9656_FIXTURE_RECEIPT = _ISSUE9656_PROMOTED_BUNDLE / "fixture_source_receipt.json"
_ISSUE9645_PROMOTED_SOURCE_HEAD = "f027f23b0fc4a99f30979a65a78bcb9feae12892"
_ISSUE9645_BUNDLE_MANIFEST_SHA256 = (
    "5e91c489147233552853fbfec1e0654875b0b375e3aef25a0669f2809b9f2002"
)
_ISSUE9645_CHECKSUMS_SHA256 = "f5f977be32eecdd60bcb3573c55f748bd0ff2820e2c62ae61d4ca154a1afe74b"
_ISSUE9645_SUMMARY_SHA256 = "9dd20b251a373eb995e0714bfec817d126f8be1537588d699f0b307bcf9af72d"
_ISSUE9645_REPORT_PROVENANCE_SHA256 = (
    "c5d330cb23566712c2617af2724b0d013ad227a56db52bdc414b6c69aef27183"
)
_ISSUE9656_PROMOTED_SOURCE_HEAD = "58b51c0419e6052e5c75eb3059e2b53205d8276d"
_ISSUE9656_PROMOTED_MATERIALIZER_REVISION = "dc9e8f6fdebb39da7c4450d2dcc1d5e9b991dfc0"
_ISSUE9656_BUNDLE_MANIFEST_SHA256 = (
    "40cfa4c052dea005f5b39984663ee134c09d69d282fdc36f49d35475802d2c2a"
)
_ISSUE9656_CHECKSUMS_SHA256 = "977f7e8d1176b8583391edba8269fc26ee7f7051c6b3801f1d5e8c4497deebf5"
_ISSUE9656_SUMMARY_SHA256 = "bc0ea6bb6ebbd9259611bffa9e3cf11a17861eb4e4a1ab93a837edf1d8f0be59"
_ISSUE9656_CURRENT_MANIFEST_SHA256 = (
    "c5f2d799d0b645ce13f2ab592263da85f55db922c1b54ff300a36e53b7f1418c"
)
_ISSUE9656_SOURCE_REVISION = "f7ebdcae2375d085e925213197a75a386e26a79c"
_ISSUE9656_REPLAY_REVISION = "5cccee50be333adceee4c978b54bf63d32454cc9"
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


def test_corpus_artifact_copy_preserves_exact_bytes_and_digest(tmp_path: Path) -> None:
    source_bytes = bytes(range(256)) + b"\x00\r\nsource evidence\n"
    source = tmp_path / "source.bin"
    source.write_bytes(source_bytes)
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()

    copied_from_path = corpus_root / "source" / "copy.bin"
    relative_path = counterexample_corpus._copy_artifact(source, copied_from_path, corpus_root)
    copied_from_bytes = corpus_root / "historical" / "copy.bin"
    counterexample_corpus._copy_artifact(source_bytes, copied_from_bytes, corpus_root)

    expected_sha256 = hashlib.sha256(source_bytes).hexdigest()
    assert relative_path == "source/copy.bin"
    assert copied_from_path.read_bytes() == source_bytes
    assert copied_from_bytes.read_bytes() == source_bytes
    assert hashlib.sha256(copied_from_path.read_bytes()).hexdigest() == expected_sha256
    assert hashlib.sha256(copied_from_bytes.read_bytes()).hexdigest() == expected_sha256


def _rekey_issue9656_import_as_legacy_v1(
    corpus: dict[str, object],
    corpus_root: Path,
    *,
    version_field: str,
    identity_source_issue: int = 9656,
) -> None:
    """Re-key a persisted-style #9656 import after declaring the legacy v1 identity."""
    import_record = corpus["historical_candidate_imports"][0]
    candidate = corpus["historical_candidates"][0]
    old_import_id = import_record["import_id"]
    old_candidate_id = candidate["candidate_id"]
    source_identity = import_record["source_identity"]
    source_identity.pop("source_materialization_bindings")
    source_identity.pop("replay_input_binding_schema")
    if version_field == "absent":
        source_identity.pop("candidate_schema_version")
    else:
        source_identity["candidate_schema_version"] = (
            counterexample_corpus.LEGACY_HISTORICAL_CANDIDATE_SCHEMA_VERSION
        )
    source_identity["source_issue"] = identity_source_issue
    candidate["schema_version"] = counterexample_corpus.LEGACY_HISTORICAL_CANDIDATE_SCHEMA_VERSION
    new_import_id = hashlib.sha256(
        counterexample_corpus._stable_json(source_identity).encode("utf-8")
    ).hexdigest()
    new_candidate_id = hashlib.sha256(
        counterexample_corpus._stable_json(
            {
                **source_identity,
                "source_case_id": candidate["source_case_id"],
                "source_record_sha256": candidate["source_record_sha256"],
            }
        ).encode("utf-8")
    ).hexdigest()
    import_record["import_id"] = new_import_id
    import_record["candidate_ids"] = [new_candidate_id]
    candidate["candidate_id"] = new_candidate_id
    candidate["source_provenance"]["import_id"] = new_import_id

    old_import_root = corpus_root / "historical_candidate_imports" / old_import_id
    new_import_root = corpus_root / "historical_candidate_imports" / new_import_id
    old_candidate_root = corpus_root / "historical_candidates" / old_candidate_id
    new_candidate_root = corpus_root / "historical_candidates" / new_candidate_id
    for receipt in import_record["source_files"]:
        receipt["stored_path"] = receipt["stored_path"].replace(
            f"historical_candidate_imports/{old_import_id}/",
            f"historical_candidate_imports/{new_import_id}/",
            1,
        )
    for paths in (import_record["artifact_paths"], candidate["artifact_paths"]):
        for name, relative in paths.items():
            relative = relative.replace(
                f"historical_candidate_imports/{old_import_id}/",
                f"historical_candidate_imports/{new_import_id}/",
                1,
            )
            relative = relative.replace(
                f"historical_candidates/{old_candidate_id}/",
                f"historical_candidates/{new_candidate_id}/",
                1,
            )
            paths[name] = relative
    for asset in candidate.get("replay_inputs", {}).get("map_assets", []):
        if isinstance(asset, dict) and isinstance(asset.get("stored_path"), str):
            asset["stored_path"] = asset["stored_path"].replace(
                f"historical_candidates/{old_candidate_id}/",
                f"historical_candidates/{new_candidate_id}/",
                1,
            )
    old_candidate_root.rename(new_candidate_root)
    old_import_root.rename(new_import_root)


def _legacy_admitted_issue9656_corpus_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], Path]:
    """Build a properly promoted candidate, then re-key it as a coherent legacy-v1 record."""
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(
        tmp_path, statuses=("not_attempted",), promotion_case=True
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
    _seed_bound_test_case(corpus, corpus_root)
    candidate = corpus["historical_candidates"][0]
    case = copy.deepcopy(corpus["cases"][0])
    case["replay_receipt"] = _single_replay_admission_receipt(case, corpus_root)
    target_revision = case["replay_receipt"]["target_revision"]
    monkeypatch.setattr(counterexample_corpus, "_current_target_revision", lambda: target_revision)
    case_record = _stage_case_under_candidate(case, corpus_root, candidate["candidate_id"])
    case_record["discovery"]["historical_candidate_binding"] = {
        "candidate_id": candidate["candidate_id"],
        "source_issue": 9656,
        "source_case_id": candidate["source_case_id"],
        "source_record_sha256": candidate["source_record_sha256"],
        "source_replay_status": candidate["source_replay_status"],
    }
    corpus, receipt = promote_historical_candidate(
        candidate["candidate_id"], case_record, corpus, corpus_root=corpus_root
    )
    assert receipt["decision"] == "duplicate"
    assert corpus["historical_candidates"][0]["candidate_status"] == "admitted"

    old_candidate_id = candidate["candidate_id"]
    old_attempt_id = candidate["promotion_attempt_id"]
    legacy = copy.deepcopy(corpus)
    _rekey_issue9656_import_as_legacy_v1(legacy, corpus_root, version_field="explicit_v1")
    candidate = legacy["historical_candidates"][0]
    new_candidate_id = candidate["candidate_id"]
    for case_record in legacy["cases"]:
        binding = case_record.get("discovery", {}).get("historical_candidate_binding")
        if isinstance(binding, dict) and binding.get("candidate_id") == old_candidate_id:
            binding["candidate_id"] = new_candidate_id
        evidence_records = [case_record.get("source_evidence")]
        evidence_records.extend(case_record.get("supporting_source_evidence", []))
        for evidence in evidence_records:
            if not isinstance(evidence, dict):
                continue
            promotion = evidence.get("historical_candidate_promotion")
            if isinstance(promotion, dict) and promotion.get("candidate_id") == old_candidate_id:
                promotion["candidate_id"] = new_candidate_id
                candidate_binding = promotion.get("historical_candidate_binding")
                if (
                    isinstance(candidate_binding, dict)
                    and candidate_binding.get("candidate_id") == old_candidate_id
                ):
                    candidate_binding["candidate_id"] = new_candidate_id

    attempt = next(
        row for row in legacy["admission_attempts"] if row["attempt_id"] == old_attempt_id
    )
    attempt["source_id"] = new_candidate_id
    attempt_payload = {
        key: attempt[key]
        for key in (
            "schema_version",
            "source_kind",
            "source_id",
            "decision",
            "blockers",
            "candidate_identity",
            "duplicate_case_id",
            "near_duplicate_report",
        )
    }
    attempt["attempt_id"] = hashlib.sha256(
        counterexample_corpus._stable_json(attempt_payload).encode("utf-8")
    ).hexdigest()
    candidate["promotion_attempt_id"] = attempt["attempt_id"]
    legacy["historical_candidates"].sort(key=lambda item: item["candidate_id"])
    legacy["historical_candidate_imports"].sort(key=lambda item: item["import_id"])
    return legacy, corpus_root


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
    replay = {
        "attempted": status == "mismatch_different_revision",
        "status": status,
        "source_revision": (
            _ISSUE9656_SOURCE_REVISION if status == "mismatch_different_revision" else None
        ),
        "replay_revision": (
            _ISSUE9656_REPLAY_REVISION if status == "mismatch_different_revision" else None
        ),
    }
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


@contextmanager
def _revision_separated_packet() -> Iterator[Path]:
    """Build a hash-consistent v3 packet with separate build and generator revisions."""
    with _packet_copy() as payload:
        bundle_root = payload.parent
        report_path = payload / "report_provenance.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest_path = bundle_root / "evidence_bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        build_revision = "dd703565459fe5431519d71b58f6a819f313db77"
        report["report_build_execution_checkout_head"] = build_revision
        manifest["commit"] = build_revision
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _refresh_bundle_checksum_for_payload(payload, "report_provenance.json")
        yield payload


@contextmanager
def _test_only_reconciled_packet() -> Iterator[Path]:
    """Build a hypothetical hash-consistent packet only for downstream mechanics tests.

    The tracked source packet is never modified. Its #1501 replay-provenance hash
    conflicts remain covered by the production importer rejection regression.
    """
    with _packet_copy() as payload:
        normalization_path = payload / "path_normalization.json"
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        for index in (1, 2):
            relative = f"historical_issue_1501_failure_0002/replay_{index}.provenance.json"
            artifact = payload / relative
            row = next(item for item in normalization["records"] if item["path"] == relative)
            row["normalized_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        normalization_path.write_text(
            json.dumps(normalization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _refresh_bundle_checksum_for_payload(payload, "path_normalization.json")
        yield payload


def _import(tmp_path: Path, payload: Path | None = None):
    corpus_root = tmp_path / "corpus"
    corpus, receipt = import_issue9645_packet(
        payload or _SOURCE_PACKET, new_corpus(), corpus_root=corpus_root
    )
    if payload is None and any(
        blocker.startswith(
            "historical_case_invalid:CorpusError:#1501 replay 1 normalized artifact binding differs:"
        )
        or "replay_input_binding_unknown_historical" in blocker
        for blocker in receipt["blockers"]
    ):
        _seed_bound_test_case(corpus, corpus_root)
    return corpus, receipt, corpus_root


def _seed_bound_test_case(corpus: dict[str, object], corpus_root: Path) -> None:
    """Seed corpus mechanics tests with a synthetic exact-input-bound replay fixture.

    This is test-only evidence used to exercise case storage and planner-status
    behavior. The production #9645 importer separately rejects the historical
    #1501 records because their recorded replay bytes lack direct input binding.
    No simulator is run.
    """
    with _test_only_reconciled_packet() as payload:
        case, source_files, _observations = counterexample_corpus._build_issue9645_historical_case(
            payload
        )
        counterexample_corpus._materialize_case_artifacts(case, source_files, corpus_root)
    from robot_sf.adversarial.scenario_admissibility import classify_scenario_admissibility

    classifier_result = classify_scenario_admissibility(
        case["case_id"],
        scenario_artifact_path=corpus_root / case["inputs"]["scenario_path"],
        scenario_id=case["scenario_id"],
    ).to_dict()
    case["admissibility"]["classifier_receipt"] = (
        counterexample_corpus.create_case_scenario_admissibility_receipt(
            case, classifier_result, corpus_root=corpus_root
        )
    )
    case_id = case["case_id"]
    case["replay_receipt"]["artifact_receipts"] = [
        {"artifact_path": f"cases/{case_id}/source_evidence/replay_{index}.jsonl"}
        for index in (1, 2)
    ]
    replay_receipts = [
        _single_replay_admission_receipt(case, corpus_root, source_index=index)
        for index in range(2)
    ]
    bound_receipt = copy.deepcopy(replay_receipts[0])
    bound_receipt["replay_count"] = len(replay_receipts)
    bound_receipt["replay_artifacts"] = [
        artifact for receipt in replay_receipts for artifact in receipt["replay_artifacts"]
    ]
    bound_receipt["artifact_receipts"] = [
        artifact for receipt in replay_receipts for artifact in receipt["artifact_receipts"]
    ]
    case["replay_receipt"] = bound_receipt
    corpus["cases"].append(case)
    corpus["cases"].sort(key=lambda item: item["case_id"])

    with pytest.MonkeyPatch.context() as target_patch:
        target_patch.setattr(
            counterexample_corpus,
            "_current_target_revision",
            lambda: bound_receipt["target_revision"],
        )
        for _ in range(2):
            _append_episode_evaluation(
                corpus,
                corpus_root,
                planner_id="goal",
                config_identity=case["target_planner"]["config_identity"],
                execution_mode="native",
                outcome={
                    "collision_event": True,
                    "route_complete": False,
                    "timeout_event": False,
                },
                termination_reason="collision",
            )
    validate_corpus(corpus, corpus_root=corpus_root)


def test_issue9645_fixture_binds_the_promoted_v3_packet() -> None:
    fixture_receipt = json.loads(_ISSUE9645_FIXTURE_RECEIPT.read_text(encoding="utf-8"))
    bundle_manifest = json.loads(
        (_SOURCE_BUNDLE / "evidence_bundle_manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads((_SOURCE_PACKET / "summary.json").read_text(encoding="utf-8"))
    report_provenance = json.loads(
        (_SOURCE_PACKET / "report_provenance.json").read_text(encoding="utf-8")
    )
    artifact_map = json.loads(
        (_SOURCE_PACKET / "pilot_report_artifact_path_map.v1.json").read_text(encoding="utf-8")
    )
    rebase_map = json.loads(
        (_SOURCE_PACKET / "reproduction_inputs/input_rebase_map.v1.json").read_text(
            encoding="utf-8"
        )
    )
    recovery = json.loads((_SOURCE_PACKET / "run_metadata.json").read_text(encoding="utf-8"))[
        "candidate_episode_record_recovery"
    ]

    assert fixture_receipt["source_head"] == _ISSUE9645_PROMOTED_SOURCE_HEAD
    assert fixture_receipt["bundle_manifest_sha256"] == _ISSUE9645_BUNDLE_MANIFEST_SHA256
    assert fixture_receipt["checksums_sha256"] == _ISSUE9645_CHECKSUMS_SHA256
    assert fixture_receipt["summary_sha256"] == _ISSUE9645_SUMMARY_SHA256
    assert fixture_receipt["report_provenance_sha256"] == _ISSUE9645_REPORT_PROVENANCE_SHA256
    assert fixture_receipt["packet_producer_revision"] == bundle_manifest["commit"]
    assert fixture_receipt["report_build_execution_checkout_head"] == bundle_manifest["commit"]
    assert (
        fixture_receipt["report_generator_commit"] == report_provenance["report_generator_commit"]
    )
    assert hashlib.sha256(
        (_SOURCE_BUNDLE / "evidence_bundle_manifest.json").read_bytes()
    ).hexdigest() == (_ISSUE9645_BUNDLE_MANIFEST_SHA256)
    assert hashlib.sha256((_SOURCE_BUNDLE / "checksums.sha256").read_bytes()).hexdigest() == (
        _ISSUE9645_CHECKSUMS_SHA256
    )
    assert hashlib.sha256((_SOURCE_PACKET / "summary.json").read_bytes()).hexdigest() == (
        _ISSUE9645_SUMMARY_SHA256
    )
    assert summary["schema_version"] == "issue_9645_bounded_pilot_summary.v2"
    assert summary["candidate_outcomes"]["new_counterexamples_discovered"] == 0
    assert summary["candidate_outcomes"]["pilot_safety_criticality_unknown_rows"] == 64
    assert summary["pilot_metrics"]["safety_criticality_status_counts"] == {
        "critical": 0,
        "not_critical": 0,
        "unknown": 64,
    }
    assert summary["feasibility"]["historical_replayed_case_dynamic_task_feasibility"] == (
        "unknown_without_reference_planner_success"
    )
    assert report_provenance["schema_version"] == "issue_9645_report_build_provenance.v3"
    assert report_provenance["episode_record_artifact_counts"] == {
        "tracked": 64,
        "digest_verified": 64,
        "missing": 0,
    }
    assert report_provenance["analysis_evidence_eligibility_counts"] == {
        "eligible": 64,
        "ineligible": 0,
    }
    assert report_provenance["trace_capture_flag_counts"] == {
        "scenario_params.record_simulation_step_trace": {"enabled": 0, "disabled": 64},
        "scenario_params.record_planner_decision_trace": {"enabled": 0, "disabled": 64},
    }
    assert report_provenance["embedded_digest_counts"] == {
        "provenance.scenario_digest": {"present": 0, "missing": 64},
        "provenance.map_digest": {"present": 0, "missing": 64},
    }
    assert report_provenance["search_or_simulation_rerun"] is False
    assert report_provenance["report_build_execution_checkout_head"] == bundle_manifest["commit"]
    assert report_provenance["report_generator_commit"] != bundle_manifest["commit"]
    artifact_candidates = {
        binding["producer_output_path"]: binding
        for binding in artifact_map["bindings"]
        if binding["artifact_kind"] == "candidate_episode_records"
    }
    rebase_candidates = {
        binding["producer_output_path"]: binding
        for manifest in rebase_map["manifest_bindings"]
        for binding in manifest["episode_record_bindings"]
    }
    assert len(artifact_candidates) == len(rebase_candidates) == 64
    assert set(artifact_candidates) == set(rebase_candidates)
    assert (
        sum(binding["source_size_bytes"] for binding in artifact_candidates.values())
        == (recovery["source_total_size_bytes"])
    )
    for producer_path, artifact_binding in artifact_candidates.items():
        rebase_binding = rebase_candidates[producer_path]
        assert (
            rebase_binding["source_sha256_before_path_normalization"]
            == (artifact_binding["source_sha256_before_path_normalization"])
        )
        assert rebase_binding["source_size_bytes"] == artifact_binding["source_size_bytes"]
        assert rebase_binding["normalized_sha256"] == artifact_binding["normalized_sha256"]
        assert rebase_binding["size_bytes"] == artifact_binding["size_bytes"]
    run_metadata_digest = hashlib.sha256(
        (_SOURCE_PACKET / "run_metadata.json").read_bytes()
    ).hexdigest()
    assert report_provenance["experiment_source_commit_sha256"] == run_metadata_digest
    assert fixture_receipt["source_experiment_revision"] == summary["source_revision"]


def test_issue9645_v3_bundle_and_generator_revisions_are_separate(tmp_path: Path) -> None:
    with _revision_separated_packet() as payload:
        manifest = json.loads(
            (payload.parent / "evidence_bundle_manifest.json").read_text(encoding="utf-8")
        )
        report = json.loads((payload / "report_provenance.json").read_text(encoding="utf-8"))
        corpus, _receipt = import_issue9645_packet(
            payload, new_corpus(), corpus_root=tmp_path / "corpus"
        )

    run = corpus["search_runs"][0]
    assert report["report_build_execution_checkout_head"] == manifest["commit"]
    assert report["report_generator_commit"] != manifest["commit"]
    assert run["report_build_execution_checkout_head"] == manifest["commit"]
    assert run["report_generator_revision"] == report["report_generator_commit"]
    assert run["new_counterexamples_discovered"] == 0
    assert run["new_counterexamples_admitted"] == 0
    assert run["historical_replay"]["dynamic_task_feasibility"] == (
        "unknown_without_reference_planner_success"
    )
    assert corpus["cases"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        "source_hash",
        "output_hash",
        "manifest_map",
        "rebase_map",
        "rerun",
        "analysis_eligibility",
        "detailed_trace_flags",
        "bundle_checkout_head",
        "generator_commit",
    ],
)
def test_issue9645_v3_report_provenance_mutations_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    with _revision_separated_packet() as payload:
        report_path = payload / "report_provenance.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if mutation == "source_hash":
            report["experiment_source_commit_sha256"] = "f" * 64
        elif mutation == "output_hash":
            report["outputs"][0]["sha256"] = "f" * 64
        elif mutation == "manifest_map":
            relative = "reproduction_inputs/pilot_report_manifest_path_map.v1.json"
            map_path = payload / relative
            manifest_map = json.loads(map_path.read_text(encoding="utf-8"))
            manifest_map["schema_version"] = "unsupported"
            map_path.write_text(json.dumps(manifest_map))
            report["manifest_path_map_sha256"] = hashlib.sha256(map_path.read_bytes()).hexdigest()
            _refresh_bundle_checksum_for_payload(payload, relative)
        elif mutation == "rebase_map":
            relative = "reproduction_inputs/input_rebase_map.v1.json"
            map_path = payload / relative
            rebase = json.loads(map_path.read_text(encoding="utf-8"))
            rebase["schema_version"] = "unsupported"
            map_path.write_text(json.dumps(rebase))
            report["reproduction_input_rebase_map_sha256"] = hashlib.sha256(
                map_path.read_bytes()
            ).hexdigest()
            _refresh_bundle_checksum_for_payload(payload, relative)
        elif mutation == "rerun":
            report["search_or_simulation_rerun"] = True
        elif mutation == "analysis_eligibility":
            report["analysis_evidence_eligibility_counts"] = {
                "eligible": 63,
                "ineligible": 1,
            }
        elif mutation == "detailed_trace_flags":
            report["trace_capture_flag_counts"]["scenario_params.record_planner_decision_trace"] = {
                "enabled": 1,
                "disabled": 63,
            }
        elif mutation == "bundle_checkout_head":
            report["report_build_execution_checkout_head"] = (
                "58e516aa4f69ff3098bf518199f483006589758c"
            )
        elif mutation == "generator_commit":
            report["report_generator_commit"] = "0" * 40
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        _refresh_bundle_checksum_for_payload(payload, "report_provenance.json")

        corpus, receipt = import_issue9645_packet(
            payload, new_corpus(), corpus_root=tmp_path / "corpus"
        )

    assert receipt["decision"] == "rejected"
    assert any("pilot_evidence_invalid" in blocker for blocker in receipt["blockers"])
    assert corpus["search_runs"] == []
    assert corpus["cases"] == []
    assert corpus["planner_evaluations"] == []


@pytest.mark.parametrize(
    "mutation",
    ["candidate_source_hash", "candidate_source_size", "candidate_normalization_source_hash"],
)
def test_issue9645_v3_candidate_normalization_bindings_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    with _revision_separated_packet() as payload:
        report_path = payload / "report_provenance.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _mutate_issue9645_candidate_normalization_binding(payload, report, mutation)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        _refresh_bundle_checksum_for_payload(payload, "report_provenance.json")
        corpus, receipt = import_issue9645_packet(
            payload, new_corpus(), corpus_root=tmp_path / "corpus"
        )

    assert receipt["decision"] == "rejected"
    assert any("pilot_evidence_invalid" in blocker for blocker in receipt["blockers"])
    assert corpus["search_runs"] == []
    assert corpus["cases"] == []
    assert corpus["planner_evaluations"] == []


def _mutate_issue9645_candidate_normalization_binding(
    payload: Path, report: dict[str, object], mutation: str
) -> None:
    if mutation in {"candidate_source_hash", "candidate_source_size"}:
        relative = "reproduction_inputs/input_rebase_map.v1.json"
        map_path = payload / relative
        rebase = json.loads(map_path.read_text(encoding="utf-8"))
        binding = rebase["manifest_bindings"][0]["episode_record_bindings"][0]
        if mutation == "candidate_source_hash":
            binding["source_sha256_before_path_normalization"] = "f" * 64
        else:
            binding["source_size_bytes"] += 1
        map_path.write_text(json.dumps(rebase, indent=2, sort_keys=True) + "\n")
        report["reproduction_input_rebase_map_sha256"] = hashlib.sha256(
            map_path.read_bytes()
        ).hexdigest()
    elif mutation == "candidate_normalization_source_hash":
        relative = "path_normalization.json"
        normalization_path = payload / relative
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        candidate = next(
            row
            for row in normalization["records"]
            if row["path"].startswith("source_episode_records/")
        )
        candidate["source_sha256_before_path_normalization"] = "f" * 64
        normalization_path.write_text(json.dumps(normalization, indent=2, sort_keys=True) + "\n")
        relative = "path_normalization.json"
    else:
        raise AssertionError(f"unexpected candidate normalization mutation: {mutation}")
    _refresh_bundle_checksum_for_payload(payload, relative)


def test_issue9645_import_rejects_replay_comparison_signature_mismatch(
    tmp_path: Path,
) -> None:
    with _packet_copy() as payload:
        path = payload / "replay_validation.json"
        validation = json.loads(path.read_text(encoding="utf-8"))
        validation["historical_issue_1501_case"]["comparison_signature"]["digest"] = "0" * 64
        path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
        _refresh_bundle_checksum_for_payload(payload, "replay_validation.json")

        corpus, receipt = import_issue9645_packet(
            payload, new_corpus(), corpus_root=tmp_path / "corpus"
        )

    assert receipt["decision"] == "rejected"
    assert any(
        "#1501 recorded replay comparison signature differs" in blocker
        for blocker in receipt["blockers"]
    )
    assert corpus["search_runs"][0]["new_counterexamples_discovered"] == 0
    assert corpus["cases"] == []
    assert corpus["planner_evaluations"] == []


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
    case: dict[str, object],
    corpus_root: Path,
    *,
    target_revision: str | None = None,
    source_index: int = 0,
) -> dict[str, object]:
    """Create an exact-revision receipt from a stored episode fixture without running it."""
    old_receipt = case["replay_receipt"]
    source_path = corpus_root / old_receipt["artifact_receipts"][source_index]["artifact_path"]
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

    relative_artifact_path = (
        f"replay_artifacts/{planner_id}-{execution_mode}-"
        f"{record['provenance']['case_input_identity']['run_id']}.jsonl"
    )
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
            "planner_configuration_snapshot",
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


def test_issue9645_packet_rejects_replay_provenance_hash_conflicts_without_admission(
    tmp_path: Path,
) -> None:
    corpus_root = tmp_path / "corpus"
    expected_conflicts = {}
    with _packet_copy() as payload:
        path = payload / "path_normalization.json"
        path_normalization = json.loads(path.read_text(encoding="utf-8"))
        for index in (1, 2):
            relative = f"historical_issue_1501_failure_0002/replay_{index}.provenance.json"
            row = next(item for item in path_normalization["records"] if item["path"] == relative)
            row["normalized_sha256"] = hashlib.sha256(
                f"deliberately-stale-replay-provenance-{index}".encode()
            ).hexdigest()
            expected_conflicts[relative] = (
                row["normalized_sha256"],
                hashlib.sha256((payload / relative).read_bytes()).hexdigest(),
            )
        path.write_text(json.dumps(path_normalization, indent=2, sort_keys=True) + "\n")
        _refresh_bundle_checksum_for_payload(payload, "path_normalization.json")
        corpus, receipt = import_issue9645_packet(payload, new_corpus(), corpus_root=corpus_root)

    assert receipt["decision"] == "rejected"
    assert len(receipt["blockers"]) == 1
    blocker = receipt["blockers"][0]
    assert blocker.startswith(
        "historical_case_invalid:CorpusError:#1501 replay 1 normalized artifact binding differs:"
    )
    for relative, (declared_hash, actual_hash) in expected_conflicts.items():
        assert f"provenance_path={relative}" in blocker
        assert f"declared_normalized_sha256={declared_hash}" in blocker
        assert f"actual_normalized_sha256={actual_hash}" in blocker
        assert declared_hash != actual_hash
    assert receipt["pilot_new_discoveries"] == 0
    assert receipt["candidate_identity"] is None
    assert len(corpus["search_runs"]) == 1
    pilot = corpus["search_runs"][0]
    assert pilot["attempted_candidates"] == 64
    assert pilot["new_counterexamples_discovered"] == 0
    assert pilot["new_counterexamples_admitted"] == 0
    assert pilot["evidence_tier"] == "diagnostic_only"
    assert pilot["criticality_status_counts"] == {
        "critical": 0,
        "not_critical": 0,
        "unknown": 64,
    }
    assert pilot["safety_criticality_unknown_candidates"] == 64
    assert pilot["analysis_evidence_eligibility_counts"] == {"eligible": 64, "ineligible": 0}
    assert pilot["episode_record_artifact_counts"] == {
        "tracked": 64,
        "digest_verified": 64,
        "missing": 0,
    }
    assert pilot["trace_capture_flag_counts"] == {
        "scenario_params.record_simulation_step_trace": {"enabled": 0, "disabled": 64},
        "scenario_params.record_planner_decision_trace": {"enabled": 0, "disabled": 64},
    }
    assert pilot["embedded_digest_counts"] == {
        "provenance.scenario_digest": {"present": 0, "missing": 64},
        "provenance.map_digest": {"present": 0, "missing": 64},
    }
    assert pilot["search_or_simulation_rerun"] is False
    assert pilot["report_generator_revision"] == "499ac172d50c2fdc979950d7138daf66d3606715"
    assert pilot["historical_replay"] == {
        "at_recorded_revision": 1,
        "at_current_code": 0,
        "recorded_revision": "58e516aa4f69ff3098bf518199f483006589758c",
        "dynamic_task_feasibility": "unknown_without_reference_planner_success",
    }

    historical = json.loads(
        (_SOURCE_PACKET / "historical_issue_1501_failure_0002.json").read_text(encoding="utf-8")
    )
    replay_validation = json.loads(
        (_SOURCE_PACKET / "replay_validation.json").read_text(encoding="utf-8")
    )
    for record in (
        historical,
        replay_validation["historical_issue_1501_case"],
    ):
        assert record["input_binding_status"] == "unknown_historical"
        assert record["admission_status"] == "not_admitted"
        assert record["regression_status"] == "pending_exact_historical_input_binding"

    assert corpus["cases"] == []
    assert corpus["planner_evaluations"] == []
    assert corpus["admission_attempts"][-1]["decision"] == "rejected"
    assert not (corpus_root / "cases").exists()
    for item in pilot["source_files"] + pilot["manifest_files"] + pilot["bundle_receipts"]:
        stored = corpus_root / item["path"]
        assert hashlib.sha256(stored.read_bytes()).hexdigest() == item["sha256"]
    validate_corpus(corpus, corpus_root=corpus_root)


def test_issue9645_v3_packet_rejects_relabeling_unknown_criticality() -> None:
    summary = json.loads((_SOURCE_PACKET / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((_SOURCE_PACKET / "run_metadata.json").read_text(encoding="utf-8"))
    summary["candidate_outcomes"]["pilot_safety_criticality_unknown_rows"] = 63

    with pytest.raises(CorpusError, match="criticality counts differ from 0/0/64"):
        counterexample_corpus._validate_pilot_summary(summary, metadata)


def test_generic_case_admission_rejects_historical_unknown_input_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_root = tmp_path / "corpus"
    corpus = new_corpus()
    with _test_only_reconciled_packet() as payload:
        case, source_files, _observations = counterexample_corpus._build_issue9645_historical_case(
            payload
        )
        counterexample_corpus._materialize_case_artifacts(case, source_files, corpus_root)
    monkeypatch.setattr(
        counterexample_corpus,
        "_current_target_revision",
        lambda: case["replay_receipt"]["target_revision"],
    )

    corpus, receipt = counterexample_corpus.admit_case_record(
        case,
        corpus,
        corpus_root=corpus_root,
        artifact_root=f"cases/{case['case_id']}",
        source_kind="test_historical_unknown_input_binding",
        source_id="issue_1501/failure_0002",
    )

    assert receipt["decision"] == "rejected"
    assert any(
        "replay_input_binding_unknown_historical" in blocker for blocker in receipt["blockers"]
    )
    assert corpus["cases"] == []
    assert corpus["planner_evaluations"] == []


def test_issue9645_import_rejects_stale_target_revision_after_input_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        counterexample_corpus,
        "_case_admission_input_binding_errors",
        lambda _case: [],
    )
    monkeypatch.setattr(
        counterexample_corpus,
        "_current_target_revision",
        lambda: "a" * 40,
    )

    with _test_only_reconciled_packet() as payload:
        corpus, receipt = import_issue9645_packet(
            payload, new_corpus(), corpus_root=tmp_path / "corpus"
        )

    assert receipt["decision"] == "rejected"
    assert any(
        "admission replay does not match the independently resolved current target revision"
        in blocker
        for blocker in receipt["blockers"]
    )
    assert len(corpus["search_runs"]) == 1
    assert corpus["cases"] == []
    assert corpus["planner_evaluations"] == []


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
    legacy_receipt.pop("input_binding_limitation", None)
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

    explicit_exclusion = copy.deepcopy(corpus)
    classifier_receipt = explicit_exclusion["cases"][0]["admissibility"]["classifier_receipt"]
    classifier_result = classifier_receipt["classifier_result"]
    classifier_result["verdict"] = "structurally_invalid"
    classifier_result["search_disposition"] = "reject"
    classifier_result["reason_codes"].append("structural_exclusion")
    classifier_receipt["classifier_result_sha256"] = hashlib.sha256(
        counterexample_corpus._stable_json(classifier_result).encode("utf-8")
    ).hexdigest()
    binding = {key: value for key, value in classifier_receipt.items() if key != "binding_sha256"}
    classifier_receipt["binding_sha256"] = hashlib.sha256(
        counterexample_corpus._stable_json(binding).encode("utf-8")
    ).hexdigest()
    with pytest.raises(CorpusError, match="explicitly excludes"):
        validate_corpus(explicit_exclusion, corpus_root=corpus_root)

    stored_evaluation["outcome"] = {
        "collision_event": True,
        "route_complete": False,
        "timeout_event": False,
    }
    with pytest.raises(CorpusError, match="evaluation digest is invalid"):
        validate_corpus(corpus, corpus_root=corpus_root)


def test_unknown_feasibility_requires_digest_bound_classifier_receipt(tmp_path: Path) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)

    missing_receipt = copy.deepcopy(corpus)
    missing_receipt["cases"][0]["admissibility"].pop("classifier_receipt")
    with pytest.raises(CorpusError, match="classifier_receipt"):
        validate_corpus(missing_receipt, corpus_root=corpus_root)

    tampered_result = copy.deepcopy(corpus)
    classifier_receipt = tampered_result["cases"][0]["admissibility"]["classifier_receipt"]
    classifier_receipt["classifier_result"]["reason_codes"].append("forged_reason")
    with pytest.raises(CorpusError, match="classifier result digest is invalid"):
        validate_corpus(tampered_result, corpus_root=corpus_root)

    tampered_map_binding = copy.deepcopy(corpus)
    classifier_receipt = tampered_map_binding["cases"][0]["admissibility"]["classifier_receipt"]
    classifier_receipt["input_binding"]["map_assets"][0]["sha256"] = "0" * 64
    binding = {key: value for key, value in classifier_receipt.items() if key != "binding_sha256"}
    classifier_receipt["binding_sha256"] = hashlib.sha256(
        counterexample_corpus._stable_json(binding).encode("utf-8")
    ).hexdigest()
    with pytest.raises(CorpusError, match="does not bind input_binding"):
        validate_corpus(tampered_map_binding, corpus_root=corpus_root)

    excluded = copy.deepcopy(corpus)
    classifier_receipt = excluded["cases"][0]["admissibility"]["classifier_receipt"]
    result = classifier_receipt["classifier_result"]
    result["verdict"] = "geometric_or_kinodynamic_impossibility"
    result["search_disposition"] = "reject"
    result["reason_codes"].append("geometric_exclusion")
    classifier_receipt["classifier_result_sha256"] = hashlib.sha256(
        counterexample_corpus._stable_json(result).encode("utf-8")
    ).hexdigest()
    binding = {key: value for key, value in classifier_receipt.items() if key != "binding_sha256"}
    classifier_receipt["binding_sha256"] = hashlib.sha256(
        counterexample_corpus._stable_json(binding).encode("utf-8")
    ).hexdigest()
    with pytest.raises(CorpusError, match="explicitly excludes"):
        validate_corpus(excluded, corpus_root=corpus_root)


def test_discovery_requires_round_and_persisted_search_run_or_explicit_history(
    tmp_path: Path,
) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    missing_round = copy.deepcopy(corpus)
    missing_round["cases"][0]["discovery"].pop("round_id")
    with pytest.raises(CorpusError, match="round_id"):
        validate_corpus(missing_round, corpus_root=corpus_root)

    missing_historical_manifest = copy.deepcopy(corpus)
    source = missing_historical_manifest["cases"][0]["discovery"]["search_source"]
    source["kind"] = "random_search"
    source.pop("historical_manifest")
    with pytest.raises(CorpusError, match="no persisted search run"):
        validate_corpus(missing_historical_manifest, corpus_root=corpus_root)

    run_linked = copy.deepcopy(corpus)
    run = run_linked["search_runs"][0]
    source = run_linked["cases"][0]["discovery"]["search_source"]
    source["kind"] = "bounded_search"
    source["run_id"] = run["run_id"]
    source["source_revision"] = run["source_revision"]
    source.pop("historical_source_revision")
    run_linked["cases"][0]["discovery"]["round_id"] = run["round_id"]
    validate_corpus(run_linked, corpus_root=corpus_root)

    orphaned = copy.deepcopy(run_linked)
    orphaned["cases"][0]["discovery"]["search_source"]["run_id"] = "missing-run"
    with pytest.raises(CorpusError, match="exactly one persisted run"):
        validate_corpus(orphaned, corpus_root=corpus_root)

    mismatched_round = copy.deepcopy(run_linked)
    mismatched_round["cases"][0]["discovery"]["round_id"] = "another-round"
    with pytest.raises(CorpusError, match="round ID differs"):
        validate_corpus(mismatched_round, corpus_root=corpus_root)


def test_duplicate_case_admission_preserves_second_planner_replay_and_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    existing = corpus["cases"][0]
    original_replay_receipt = copy.deepcopy(existing["replay_receipt"])
    prior_receipt = existing["replay_receipt"]["artifact_receipts"][0]
    source_artifact = corpus_root / prior_receipt["artifact_path"]
    episode = json.loads(source_artifact.read_text(encoding="utf-8"))

    planner_id = "goal_optimized"
    config_identity = "goal_optimized_config_v2"
    configuration_snapshot = {"max_speed_m_s": 1.25, "variant": "duplicate-test"}
    run_id = "duplicate-planner-b-replay"
    episode["algo"] = planner_id
    episode["algorithm_metadata"]["algorithm"] = planner_id
    episode["algorithm_metadata"]["canonical_algorithm"] = planner_id
    episode["algorithm_metadata"]["config_hash"] = config_identity
    episode["algorithm_metadata"]["config"] = configuration_snapshot
    episode["event_ledger"]["planner"] = planner_id
    episode["event_ledger"]["software_commit"] = episode["git_hash"]
    episode["provenance"]["case_input_identity"] = {
        **counterexample_corpus._case_runtime_input_binding(existing, corpus_root),
        "run_id": run_id,
        "reason_codes": [],
    }
    artifact_relative = f"cases/{existing['case_id']}/replay_artifacts/{run_id}.jsonl"
    artifact = corpus_root / artifact_relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(episode, sort_keys=True) + "\n", encoding="utf-8")

    duplicate = copy.deepcopy(existing)
    duplicate["target_planner"].update(
        {
            "planner_id": planner_id,
            "config_identity": config_identity,
            "configuration_snapshot": configuration_snapshot,
        }
    )
    observation = counterexample_corpus._admission_replay_observation(existing, prior_receipt)
    observation.update(
        {
            "planner_id": planner_id,
            "planner_config_identity": config_identity,
            "episode_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "outcome": episode["outcome"],
            "termination_reason": episode["termination_reason"],
            "metrics": prior_receipt["metrics"],
        }
    )
    target_revision = episode["git_hash"]
    monkeypatch.setattr(counterexample_corpus, "_current_target_revision", lambda: target_revision)
    duplicate["replay_receipt"] = create_case_admission_replay_receipt(
        observation,
        duplicate,
        artifact_path=artifact_relative,
        corpus_root=corpus_root,
        target_revision=target_revision,
    )
    duplicate["source_evidence"]["corpus_files"].append(
        {
            "path": artifact_relative,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
    )

    corpus, admission = counterexample_corpus.admit_case_record(
        duplicate,
        corpus,
        corpus_root=corpus_root,
        artifact_root=f"cases/{existing['case_id']}",
        source_kind="test_second_planner_discovery",
        source_id=run_id,
    )

    assert admission["decision"] == "duplicate", admission["blockers"]
    assert admission["case_id"] == existing["case_id"]
    assert len(corpus["cases"]) == 1
    stored = corpus["cases"][0]
    assert stored["replay_receipt"] == original_replay_receipt
    supporting_evaluation = next(
        evaluation
        for evaluation in corpus["planner_evaluations"]
        if evaluation["planner_id"] == planner_id
        and evaluation["planner_config_identity"] == config_identity
    )
    assert supporting_evaluation["planner_configuration_snapshot"] == configuration_snapshot
    replay = supporting_evaluation["replay_receipt"]
    assert replay["planner_configuration_snapshot"] == configuration_snapshot
    assert replay["artifact_path"].startswith(f"cases/{existing['case_id']}/supporting_replays/")
    stored_artifact = corpus_root / replay["artifact_path"]
    assert hashlib.sha256(stored_artifact.read_bytes()).hexdigest() == replay["artifact_sha256"]
    assert any(
        row["path"] == replay["artifact_path"] and row["sha256"] == replay["artifact_sha256"]
        for row in stored["source_evidence"]["corpus_files"]
    )
    status = recompute_planner_status(
        corpus,
        planner_id=planner_id,
        planner_config_identity=config_identity,
        corpus_root=corpus_root,
    )
    assert status["status_counts"]["unsolved"] == 1
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
    corpus, _receipt, corpus_root = _import(tmp_path)
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
    corpus, _receipt, corpus_root = _import(tmp_path)
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
    receipt["verification_status"] = "repeated_current_revision_match"
    receipt.pop("artifact_receipts")

    errors = counterexample_corpus._validate_case_admission_replay(case, corpus_root)
    assert any(
        "replay_artifact_outcome_metric_contradiction" in error and "success metrics > 0" in error
        for error in errors
    )
    with pytest.raises(CorpusError, match="replay_artifact_outcome_metric_contradiction"):
        validate_corpus({**corpus, "cases": [case]}, corpus_root=corpus_root)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("status_mismatch", "replay_artifact_status_termination_mismatch"),
        ("degraded_execution", "replay_artifact_fallback_status_mismatch"),
        ("metadata_degraded", "replay_artifact_fallback_status_mismatch"),
        ("foresight_prediction_fallback", "replay_artifact_fallback_status_mismatch"),
        ("fallback_reason", "replay_artifact_fallback_status_mismatch"),
        ("null_degraded_marker", "replay_artifact_fallback_status_mismatch"),
        ("null_top_level_fallback_marker", "replay_artifact_fallback_status_mismatch"),
    ],
)
def test_legacy_v1_no_row_admission_rejects_invalid_status_or_execution(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    """Legacy receipts must validate raw status and execution evidence after rehashing."""
    corpus, _receipt, corpus_root = _import(tmp_path)
    case = copy.deepcopy(corpus["cases"][0])
    receipt = case["replay_receipt"]

    for replay in receipt["replay_artifacts"]:
        artifact_path = replay["path"]
        artifact = corpus_root / artifact_path
        episode = json.loads(artifact.read_text(encoding="utf-8"))
        if mutation == "status_mismatch":
            episode["status"] = "success"
        elif mutation == "degraded_execution":
            episode["integrity"]["effective_view"]["degraded"] = True
            episode["algorithm_metadata"]["status"] = "degraded"
            episode["readiness_status"] = "fallback"
            episode["availability_status"] = "not_available"
        elif mutation == "metadata_degraded":
            episode["algorithm_metadata"]["degraded"] = True
            assert episode["algorithm_metadata"]["status"] == "ok"
            assert episode["integrity"]["effective_view"]["degraded"] is False
        elif mutation == "foresight_prediction_fallback":
            episode["algorithm_metadata"]["foresight_prediction"] = {"fallback_used": True}
            assert episode["algorithm_metadata"]["status"] == "ok"
            assert episode["integrity"]["effective_view"]["degraded"] is False
        elif mutation in {
            "fallback_reason",
            "null_degraded_marker",
            "null_top_level_fallback_marker",
        }:
            marker_paths = {
                "fallback_reason": (
                    ("algorithm_metadata", "fallback_reason"),
                    "unexpected runtime fallback",
                ),
                "null_degraded_marker": (("algorithm_metadata", "degraded"), None),
                "null_top_level_fallback_marker": (("fallback_or_degraded",), None),
            }
            path, value = marker_paths[mutation]
            target = episode
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            assert episode["algorithm_metadata"]["status"] == "ok"
            assert episode["integrity"]["effective_view"]["degraded"] is False
        else:
            raise AssertionError(f"unexpected replay mutation: {mutation}")
        artifact.write_text(json.dumps(episode, sort_keys=True) + "\n", encoding="utf-8")
        artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        replay["sha256"] = artifact_sha256
        replay["normalized_bundle_sha256"] = artifact_sha256
        replay["selected_event_identity"] = counterexample_corpus._selected_event_identity(episode)
        for source_file in case["source_evidence"]["corpus_files"]:
            if source_file["path"] == artifact_path:
                source_file["sha256"] = artifact_sha256
                break
        else:
            raise AssertionError("admission replay artifact is absent from source evidence custody")

    receipt["selected_projection"]["selected_event_identity"] = receipt["replay_artifacts"][0][
        "selected_event_identity"
    ]
    receipt["selected_projection_sha256"] = hashlib.sha256(
        counterexample_corpus._stable_json(receipt["selected_projection"]).encode("utf-8")
    ).hexdigest()
    receipt["schema_version"] = "adversarial-case-admission-replay.v1"
    receipt["verification_status"] = "repeated_current_revision_match"
    receipt.pop("artifact_receipts")

    errors = counterexample_corpus._validate_case_admission_replay(case, corpus_root)
    assert expected_error in errors
    with pytest.raises(CorpusError, match=expected_error):
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


def _assert_unknown_issue9656_candidate_status(candidate: dict[str, object]) -> None:
    assert candidate["schema_version"] == "adversarial-historical-candidate.v2"
    assert candidate["feasibility"] == {
        "verdict": "unknown",
        "reason_codes": ["historical_benchmark_evidence_does_not_establish_dynamic_feasibility"],
    }
    status = candidate["planner_status_at_import"]
    assert status["status"] == "unknown"
    assert status["valid_observation_count"] == 0
    assert status["planner_id"] == candidate["target_planner"]["planner_id"]
    assert status["config_hash"] == candidate["target_planner"]["config_hash"]
    assert "no_exact_current_revision_observation" in status["reason_codes"]


def test_issue9656_promoted_manifest_only_evidence_stays_unadmitted(
    tmp_path: Path,
) -> None:
    fixture_receipt = json.loads(_ISSUE9656_FIXTURE_RECEIPT.read_text(encoding="utf-8"))
    bundle_manifest = json.loads(
        (_ISSUE9656_PROMOTED_BUNDLE / "evidence_bundle_manifest.json").read_text(encoding="utf-8")
    )
    summary, summary_receipts = counterexample_corpus._verify_issue9656_source_summary(
        _ISSUE9656_PROMOTED_SUMMARY, _ISSUE9656_PROMOTED_BUNDLE
    )
    no_replay = json.loads(_ISSUE9656_PROMOTED_MANIFEST.read_text(encoding="utf-8"))
    current_status_counts = no_replay["replay"]["status_counts"]
    current_statuses = [case["replay"]["status"] for case in no_replay["cases"]]
    source_status_counts = summary["replay"]["status_counts"]

    assert fixture_receipt["source_head"] == _ISSUE9656_PROMOTED_SOURCE_HEAD
    assert fixture_receipt["materializer_revision"] == _ISSUE9656_PROMOTED_MATERIALIZER_REVISION
    assert fixture_receipt["bundle_manifest_sha256"] == _ISSUE9656_BUNDLE_MANIFEST_SHA256
    assert fixture_receipt["checksums_sha256"] == _ISSUE9656_CHECKSUMS_SHA256
    assert fixture_receipt["summary_sha256"] == _ISSUE9656_SUMMARY_SHA256
    assert fixture_receipt["current_head_manifest_sha256"] == _ISSUE9656_CURRENT_MANIFEST_SHA256
    assert summary_receipts["manifest_sha256"] == _ISSUE9656_BUNDLE_MANIFEST_SHA256
    assert summary_receipts["checksums_sha256"] == _ISSUE9656_CHECKSUMS_SHA256
    assert summary_receipts["summary_sha256"] == fixture_receipt["summary_sha256"]
    assert hashlib.sha256(_ISSUE9656_PROMOTED_MANIFEST.read_bytes()).hexdigest() == (
        _ISSUE9656_CURRENT_MANIFEST_SHA256
    )
    assert bundle_manifest["commit"] == _ISSUE9656_PROMOTED_MATERIALIZER_REVISION
    assert summary["source"]["source_revision"] == _ISSUE9656_SOURCE_REVISION
    assert summary["materializer_revision"] == "0a5f73283b98279900797b75d20adf6d4676086b"
    assert (
        summary["materializer_revision_status"] == "historical_id_unavailable_from_published_refs"
    )
    assert no_replay["source"]["summary_sha256"] == summary["source"]["summary_sha256"]
    assert no_replay["source"]["summary_sha256"] != summary_receipts["summary_sha256"]
    assert no_replay["replay"]["materializer_revision"] == (
        _ISSUE9656_PROMOTED_MATERIALIZER_REVISION
    )
    assert source_status_counts == {
        "mismatch_different_revision": 4,
        "not_attempted": 27,
        "unavailable_model_artifact": 5,
    }
    assert current_status_counts == {
        "unavailable_execution_evidence": 4,
        "not_attempted": 27,
        "unavailable_model_artifact": 5,
    }
    assert len(no_replay["cases"]) == 36
    assert len(summary["cases"]) == len(no_replay["cases"])
    assert set(summary["selection"]["case_ids"]) == {case["case_id"] for case in no_replay["cases"]}
    assert "exact_match" not in current_status_counts
    assert not any(status in {"exact_match", "exact_replay_match"} for status in current_statuses)
    assert not (_ISSUE9656_PROMOTED_BUNDLE / "payload/cases").exists()

    corpus = new_corpus()
    corpus_before = copy.deepcopy(corpus)
    corpus_root = tmp_path / "corpus"
    with pytest.raises(CorpusError):
        import_issue9656_candidates(
            _ISSUE9656_PROMOTED_SUMMARY,
            tmp_path / "no-materialized-cases",
            _ISSUE9656_PROMOTED_BUNDLE,
            tmp_path / "no-campaign-episodes",
            corpus,
            corpus_root=corpus_root,
        )
    assert corpus == corpus_before
    assert not corpus_root.exists()


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
    validate_corpus(corpus, corpus_root=corpus_root)
    with pytest.raises(CorpusError, match="corpus_root is required"):
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


def test_issue9656_candidate_claims_unknown_and_rejects_tampering(tmp_path: Path) -> None:
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
    assert receipt["feasibility_status_counts"] == {"unknown": 3}
    assert receipt["planner_status_counts_at_import"] == {"unknown": 3}
    for candidate in corpus["historical_candidates"]:
        _assert_unknown_issue9656_candidate_status(candidate)

    mismatch = next(
        row
        for row in corpus["historical_candidates"]
        if row["source_replay_status"] == "mismatch_different_revision"
    )
    assert mismatch["candidate_status"] == "blocked_replay_revision_mismatch"
    assert mismatch["source_provenance"]["source_row_binding"] == (
        "verified_episode_file_and_line_sha256"
    )
    assert mismatch["source_provenance"]["episode_file_sha256"]
    assert mismatch["source_provenance"]["line_number"] > 0
    assert (
        mismatch["source_replay"]["source_revision"] != mismatch["source_replay"]["replay_revision"]
    )
    assert mismatch["planner_status_at_import"]["reason_codes"] == [
        "no_exact_current_revision_observation",
        "source_replay_revision_mismatch",
    ]

    tampered_status = copy.deepcopy(corpus)
    tampered_candidate = next(
        row
        for row in tampered_status["historical_candidates"]
        if row["source_replay_status"] == "mismatch_different_revision"
    )
    tampered_candidate["planner_status_at_import"]["reason_codes"].remove(
        "source_replay_revision_mismatch"
    )
    with pytest.raises(CorpusError, match="historical candidate planner status differs"):
        validate_corpus(tampered_status, corpus_root=corpus_root)

    tampered_schema = copy.deepcopy(corpus)
    tampered_schema["historical_candidates"][0]["schema_version"] = (
        "adversarial-historical-candidate.v1"
    )
    with pytest.raises(CorpusError, match="historical candidate schema differs"):
        validate_corpus(tampered_schema, corpus_root=corpus_root)

    tampered_feasibility = copy.deepcopy(corpus)
    tampered_feasibility["historical_candidates"][0]["feasibility"]["verdict"] = "feasible"
    with pytest.raises(CorpusError, match="invalid corpus at historical_candidates"):
        validate_corpus(tampered_feasibility, corpus_root=corpus_root)


def test_issue9656_imported_replay_classification_cannot_be_rewritten(
    tmp_path: Path,
) -> None:
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(tmp_path)
    corpus_root = tmp_path / "corpus"
    corpus, _receipt = import_issue9656_candidates(
        summary_path,
        materialized,
        bundle_root,
        campaign_root,
        new_corpus(),
        corpus_root=corpus_root,
    )
    corpus_path = corpus_root / "corpus.json"
    save_corpus(corpus_path, corpus)

    tampered = copy.deepcopy(corpus)
    mismatch = next(
        row
        for row in tampered["historical_candidates"]
        if row["source_replay_status"] == "mismatch_different_revision"
    )
    mismatch["source_replay_status"] = "not_attempted"
    mismatch["source_replay"] = {
        "attempted": False,
        "status": "not_attempted",
        "source_revision": None,
        "replay_revision": None,
    }
    mismatch["candidate_status"] = "pending_exact_replay"
    mismatch["planner_status_at_import"] = (
        counterexample_corpus._issue9656_planner_status_at_import(mismatch)
    )
    imported = next(
        record
        for record in tampered["historical_candidate_imports"]
        if record["import_id"] == mismatch["source_provenance"]["import_id"]
    )
    imported["source_replay_status_counts"] = {
        "not_attempted": 2,
        "unavailable_model_artifact": 1,
    }
    imported["candidate_status_counts"] = {
        "blocked_unavailable_model_artifact": 1,
        "pending_exact_replay": 2,
    }

    with pytest.raises(CorpusError, match="source replay classification differs"):
        save_corpus(corpus_path, tampered)

    # Simulate direct JSON editing that bypasses save_corpus validation.
    corpus_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(CorpusError, match="source replay classification differs"):
        load_corpus(corpus_path)


def test_issue9656_source_identity_mismatch_cannot_be_reclassified_as_verified(
    tmp_path: Path,
) -> None:
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(
        tmp_path, statuses=("not_attempted",)
    )
    source_case_path = materialized / "cases/case-0000000000000001/case.json"
    source_case = json.loads(source_case_path.read_text(encoding="utf-8"))
    source_case["source"]["row_git_hash"] = "0" * 40
    source_case_path.write_text(json.dumps(source_case, sort_keys=True), encoding="utf-8")

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
    assert candidate["candidate_status"] == "blocked_source_provenance_mismatch"
    assert candidate["source_provenance"]["source_identity_binding_status"] == "blocked"
    assert candidate["source_provenance"]["source_identity_binding_issues"] == ["row_git_hash"]

    corpus_path = corpus_root / "corpus.json"
    save_corpus(corpus_path, corpus)
    tampered = copy.deepcopy(corpus)
    tampered_candidate = tampered["historical_candidates"][0]
    provenance = tampered_candidate["source_provenance"]
    provenance["source_identity_binding_status"] = "verified"
    provenance["source_identity_binding_issues"] = []
    tampered_candidate["candidate_status"] = "pending_exact_replay"
    tampered_candidate["target_planner"]["config_hash"] = provenance["episode_planner_config_hash"]
    tampered_candidate["planner_status_at_import"] = (
        counterexample_corpus._issue9656_planner_status_at_import(tampered_candidate)
    )

    with pytest.raises(CorpusError, match="source identity binding differs"):
        save_corpus(corpus_path, tampered)

    corpus_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(CorpusError, match="source identity binding differs"):
        load_corpus(corpus_path)


@pytest.mark.parametrize("version_field", ("absent", "explicit_v1"))
def test_issue9652_persisted_v2_to_legacy_downgrade_cannot_restore_pending_status(
    tmp_path: Path, version_field: str
) -> None:
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(
        tmp_path, statuses=("not_attempted",)
    )
    source_case_path = materialized / "cases/case-0000000000000001/case.json"
    source_case = json.loads(source_case_path.read_text(encoding="utf-8"))
    source_case["source"]["row_git_hash"] = "0" * 40
    source_case_path.write_text(json.dumps(source_case, sort_keys=True), encoding="utf-8")

    corpus_root = tmp_path / "corpus"
    corpus, _receipt = import_issue9656_candidates(
        summary_path,
        materialized,
        bundle_root,
        campaign_root,
        new_corpus(),
        corpus_root=corpus_root,
    )
    corpus_path = corpus_root / "corpus.json"
    save_corpus(corpus_path, corpus)
    assert corpus["historical_candidates"][0]["candidate_status"] == (
        "blocked_source_provenance_mismatch"
    )

    # Simulate a direct persisted-corpus edit: rewrite the blocked source identity to match
    # the checksum-pinned episode, remove the independent case-byte receipts, and recompute
    # content-derived import/candidate IDs and their references.
    tampered = copy.deepcopy(corpus)
    candidate = tampered["historical_candidates"][0]
    source_case_path = corpus_root / candidate["artifact_paths"]["source_case"]
    source_case = json.loads(source_case_path.read_text(encoding="utf-8"))
    source_case["source"]["row_git_hash"] = _ISSUE9656_SOURCE_REVISION
    source_case_path.write_text(json.dumps(source_case, sort_keys=True), encoding="utf-8")
    source_case_digest = hashlib.sha256(source_case_path.read_bytes()).hexdigest()

    summary = json.loads(
        (corpus_root / candidate["artifact_paths"]["import_summary"]).read_text(encoding="utf-8")
    )
    summary_case = summary["cases"][0]
    source_binding = counterexample_corpus._issue9656_source_identity_binding(
        summary_case, source_case, source_case["source_record"], summary["source"]
    )
    assert source_binding["status"] == "verified"
    provenance = candidate["source_provenance"]
    provenance.update(
        {
            "row_git_hash": _ISSUE9656_SOURCE_REVISION,
            "source_identity_binding_status": "verified",
            "source_identity_binding_issues": [],
            "source_case_file_sha256": source_case_digest,
        }
    )
    candidate["candidate_status"] = "pending_exact_replay"
    candidate["target_planner"]["config_hash"] = source_binding["episode_planner_config_hash"]
    candidate["planner_status_at_import"] = (
        counterexample_corpus._issue9656_planner_status_at_import(candidate)
    )
    _rekey_issue9656_import_as_legacy_v1(tampered, corpus_root, version_field=version_field)
    tampered["historical_candidates"].sort(key=lambda item: item["candidate_id"])
    tampered["historical_candidate_imports"].sort(key=lambda item: item["import_id"])

    with pytest.raises(CorpusError, match="legacy #9656 candidate must remain explicitly blocked"):
        validate_corpus(tampered, corpus_root=corpus_root)

    corpus_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    loaded = load_corpus(corpus_path)
    loaded_candidate = loaded["historical_candidates"][0]
    assert loaded_candidate["candidate_status"] == "blocked_source_provenance_mismatch"
    assert loaded_candidate["source_provenance"]["source_identity_binding_status"] == (
        "legacy_unpinned"
    )
    assert loaded_candidate["legacy_unpinned_source_evidence"] == {
        "schema_version": counterexample_corpus.LEGACY_UNPINNED_SOURCE_EVIDENCE_SCHEMA,
        "candidate_status_at_load": "pending_exact_replay",
        "source_identity_binding_status_at_load": "verified",
        "source_identity_binding_issues_at_load": [],
        "reason": "legacy_import_has_no_source_materialization_byte_receipt",
    }
    validate_corpus(loaded, corpus_root=corpus_root)


def test_issue9652_legacy_v1_blocked_import_remains_readable(tmp_path: Path) -> None:
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(
        tmp_path, statuses=("not_attempted",)
    )
    source_case_path = materialized / "cases/case-0000000000000001/case.json"
    source_case = json.loads(source_case_path.read_text(encoding="utf-8"))
    source_case["source"]["row_git_hash"] = "0" * 40
    source_case_path.write_text(json.dumps(source_case, sort_keys=True), encoding="utf-8")

    corpus_root = tmp_path / "corpus"
    corpus, _receipt = import_issue9656_candidates(
        summary_path,
        materialized,
        bundle_root,
        campaign_root,
        new_corpus(),
        corpus_root=corpus_root,
    )
    corpus_path = corpus_root / "corpus.json"
    save_corpus(corpus_path, corpus)
    tampered = copy.deepcopy(corpus)
    _rekey_issue9656_import_as_legacy_v1(tampered, corpus_root, version_field="explicit_v1")
    tampered["historical_candidates"].sort(key=lambda item: item["candidate_id"])
    tampered["historical_candidate_imports"].sort(key=lambda item: item["import_id"])
    corpus_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")

    loaded = load_corpus(corpus_path)
    candidate = loaded["historical_candidates"][0]
    assert candidate["candidate_status"] == "blocked_source_provenance_mismatch"
    assert candidate["source_provenance"]["source_identity_binding_status"] == ("legacy_unpinned")
    assert candidate["legacy_unpinned_source_evidence"]["candidate_status_at_load"] == (
        "blocked_source_provenance_mismatch"
    )
    assert (
        candidate["legacy_unpinned_source_evidence"]["source_identity_binding_status_at_load"]
        == "blocked"
    )


@pytest.mark.parametrize("source_binding_status", ("blocked", "verified"))
def test_issue9652_v1_downgrade_cannot_forge_admission_to_unrelated_case(
    tmp_path: Path, source_binding_status: str
) -> None:
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(
        tmp_path, statuses=("not_attempted",)
    )
    source_case_path = materialized / "cases/case-0000000000000001/case.json"
    if source_binding_status == "blocked":
        source_case = json.loads(source_case_path.read_text(encoding="utf-8"))
        source_case["source"]["row_git_hash"] = "0" * 40
        source_case_path.write_text(json.dumps(source_case, sort_keys=True), encoding="utf-8")

    corpus_root = tmp_path / "corpus"
    corpus, _import_receipt = import_issue9656_candidates(
        summary_path,
        materialized,
        bundle_root,
        campaign_root,
        new_corpus(),
        corpus_root=corpus_root,
    )
    _seed_bound_test_case(corpus, corpus_root)
    unrelated_case_id = corpus["cases"][0]["case_id"]
    assert corpus["historical_candidates"][0]["candidate_status"] == (
        "blocked_source_provenance_mismatch"
        if source_binding_status == "blocked"
        else "pending_exact_replay"
    )
    corpus_path = corpus_root / "corpus.json"
    save_corpus(corpus_path, corpus)

    tampered = copy.deepcopy(corpus)
    _rekey_issue9656_import_as_legacy_v1(tampered, corpus_root, version_field="explicit_v1")
    candidate = tampered["historical_candidates"][0]
    candidate["source_candidate_status"] = "pending_exact_replay"
    candidate["candidate_status"] = "admitted"
    candidate["promoted_case_id"] = unrelated_case_id
    if source_binding_status == "blocked":
        # Forge the status flag as well as the admission record. Load must
        # recompute source identity from the retained summary/materialization.
        candidate["source_provenance"]["source_identity_binding_status"] = "verified"
        candidate["source_provenance"]["source_identity_binding_issues"] = []
    tampered, attempt = counterexample_corpus._record_attempt(
        tampered,
        source_kind="issue_9656_historical_candidate",
        source_id=candidate["candidate_id"],
        decision="duplicate",
        blockers=[],
        candidate_identity=unrelated_case_id.removeprefix("case-"),
        duplicate_case_id=unrelated_case_id,
        near_duplicate_report=counterexample_corpus._unassessed_near_duplicates(),
    )
    candidate["promotion_attempt_id"] = attempt["attempt_id"]
    tampered["historical_candidates"].sort(key=lambda item: item["candidate_id"])
    tampered["historical_candidate_imports"].sort(key=lambda item: item["import_id"])
    corpus_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")

    expected_error = (
        "candidate source identity binding differs from pinned materialized evidence"
        if source_binding_status == "blocked"
        else "promoted case has no historical-candidate binding"
    )
    with pytest.raises(CorpusError, match=expected_error):
        load_corpus(corpus_path)


def test_issue9652_v1_candidate_cannot_use_fabricated_binding_to_unrelated_case(
    tmp_path: Path,
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
    _seed_bound_test_case(corpus, corpus_root)
    unrelated_case_id = corpus["cases"][0]["case_id"]
    tampered = copy.deepcopy(corpus)
    _rekey_issue9656_import_as_legacy_v1(tampered, corpus_root, version_field="explicit_v1")
    candidate = tampered["historical_candidates"][0]
    candidate["source_candidate_status"] = "pending_exact_replay"
    candidate["candidate_status"] = "admitted"
    candidate["promoted_case_id"] = unrelated_case_id
    case = next(item for item in tampered["cases"] if item["case_id"] == unrelated_case_id)
    binding = {
        "candidate_id": candidate["candidate_id"],
        "source_issue": 9656,
        "source_case_id": candidate["source_case_id"],
        "source_record_sha256": candidate["source_record_sha256"],
        "source_replay_status": candidate["source_replay_status"],
    }
    case["discovery"]["historical_candidate_binding"] = binding
    replay_revision = case["replay_receipt"]["replay_revision"]
    case.setdefault("supporting_source_evidence", []).append(
        {
            "historical_candidate_promotion": {
                "candidate_id": candidate["candidate_id"],
                "historical_candidate_binding": binding,
                "admission_decision": "duplicate",
                "source_replay_status": candidate["source_replay_status"],
                "raw_episode_artifact_custody": "digest_only_not_copied_from_campaign_output",
                "raw_episode_artifact_used_as_admission_evidence": False,
                "local_ignored_output_used_as_admission_evidence": False,
                "admission_replay_matches_target_revision": True,
                "admission_replay_revision": replay_revision,
            }
        }
    )
    tampered, attempt = counterexample_corpus._record_attempt(
        tampered,
        source_kind="issue_9656_historical_candidate",
        source_id=candidate["candidate_id"],
        decision="duplicate",
        blockers=[],
        candidate_identity=case["effective_scenario_sha256"],
        duplicate_case_id=case["case_id"],
        near_duplicate_report=counterexample_corpus._unassessed_near_duplicates(),
    )
    candidate["promotion_attempt_id"] = attempt["attempt_id"]

    corpus_path = corpus_root / "corpus.json"
    corpus_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(CorpusError, match="case_record_scenario_or_planner_differs_from_candidate"):
        load_corpus(corpus_path)


def test_issue9652_properly_promoted_legacy_v1_candidate_remains_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy, corpus_root = _legacy_admitted_issue9656_corpus_fixture(tmp_path, monkeypatch)
    corpus_path = corpus_root / "corpus.json"
    corpus_path.write_text(json.dumps(legacy, sort_keys=True), encoding="utf-8")

    loaded = load_corpus(corpus_path)

    candidate = loaded["historical_candidates"][0]
    assert (
        candidate["schema_version"]
        == counterexample_corpus.LEGACY_HISTORICAL_CANDIDATE_SCHEMA_VERSION
    )
    assert candidate["candidate_status"] == "admitted"
    assert candidate["source_provenance"]["source_identity_binding_status"] == "verified"
    case = next(
        item for item in loaded["cases"] if item["case_id"] == candidate["promoted_case_id"]
    )
    assert any(
        evidence.get("historical_candidate_promotion", {}).get("candidate_id")
        == candidate["candidate_id"]
        for evidence in case.get("supporting_source_evidence", [])
    )
    promotion = next(
        evidence["historical_candidate_promotion"]
        for evidence in case.get("supporting_source_evidence", [])
        if evidence.get("historical_candidate_promotion", {}).get("candidate_id")
        == candidate["candidate_id"]
    )
    assert promotion["historical_candidate_binding"] == {
        "candidate_id": candidate["candidate_id"],
        "source_issue": 9656,
        "source_case_id": candidate["source_case_id"],
        "source_record_sha256": candidate["source_record_sha256"],
        "source_replay_status": candidate["source_replay_status"],
    }


def test_duplicate_promotions_preserve_primary_and_supporting_candidate_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy, _corpus_root = _legacy_admitted_issue9656_corpus_fixture(tmp_path, monkeypatch)
    candidate_a = legacy["historical_candidates"][0]
    case = next(
        item for item in legacy["cases"] if item["case_id"] == candidate_a["promoted_case_id"]
    )
    attempt_a = next(
        item
        for item in legacy["admission_attempts"]
        if item["attempt_id"] == candidate_a["promotion_attempt_id"]
    )
    import_record = next(
        item
        for item in legacy["historical_candidate_imports"]
        if item["import_id"] == candidate_a["source_provenance"]["import_id"]
    )
    binding_a = {
        "candidate_id": candidate_a["candidate_id"],
        "source_issue": 9656,
        "source_case_id": candidate_a["source_case_id"],
        "source_record_sha256": candidate_a["source_record_sha256"],
        "source_replay_status": candidate_a["source_replay_status"],
    }
    case["discovery"]["historical_candidate_binding"] = binding_a

    candidate_b = copy.deepcopy(candidate_a)
    candidate_b["source_case_id"] = "case-candidate-b"
    candidate_b["source_record_sha256"] = hashlib.sha256(b"candidate B source row").hexdigest()
    candidate_b["candidate_id"] = hashlib.sha256(
        counterexample_corpus._stable_json(
            {
                **import_record["source_identity"],
                "source_case_id": candidate_b["source_case_id"],
                "source_record_sha256": candidate_b["source_record_sha256"],
            }
        ).encode("utf-8")
    ).hexdigest()
    candidate_b["promoted_case_id"] = case["case_id"]
    candidate_b["source_candidate_status"] = "pending_exact_replay"
    candidate_b["candidate_status"] = "admitted"

    attempt_b = {
        "schema_version": counterexample_corpus.ATTEMPT_SCHEMA_VERSION,
        "source_kind": "issue_9656_historical_candidate",
        "source_id": candidate_b["candidate_id"],
        "decision": "duplicate",
        "blockers": [],
        "candidate_identity": case["effective_scenario_sha256"],
        "duplicate_case_id": case["case_id"],
        "near_duplicate_report": copy.deepcopy(attempt_a["near_duplicate_report"]),
    }
    attempt_b["attempt_id"] = hashlib.sha256(
        counterexample_corpus._stable_json(attempt_b).encode("utf-8")
    ).hexdigest()
    candidate_b["promotion_attempt_id"] = attempt_b["attempt_id"]

    binding_b = {
        "candidate_id": candidate_b["candidate_id"],
        "source_issue": 9656,
        "source_case_id": candidate_b["source_case_id"],
        "source_record_sha256": candidate_b["source_record_sha256"],
        "source_replay_status": candidate_b["source_replay_status"],
    }
    replay_revision = case["replay_receipt"]["replay_revision"]
    case.setdefault("supporting_source_evidence", []).append(
        {
            "historical_candidate_promotion": {
                "candidate_id": candidate_b["candidate_id"],
                "historical_candidate_binding": binding_b,
                "admission_decision": "duplicate",
                "source_replay_status": candidate_b["source_replay_status"],
                "raw_episode_artifact_custody": "digest_only_not_copied_from_campaign_output",
                "raw_episode_artifact_used_as_admission_evidence": False,
                "local_ignored_output_used_as_admission_evidence": False,
                "admission_replay_matches_target_revision": True,
                "admission_replay_revision": replay_revision,
            }
        }
    )
    attempts_by_id = {
        attempt_a["attempt_id"]: [attempt_a],
        attempt_b["attempt_id"]: [attempt_b],
    }
    cases_by_id = {case["case_id"]: case}

    counterexample_corpus._validate_promoted_historical_candidate(
        candidate_a, cases_by_id, attempts_by_id
    )
    counterexample_corpus._validate_promoted_historical_candidate(
        candidate_b, cases_by_id, attempts_by_id
    )
    assert (
        case["discovery"]["historical_candidate_binding"]["candidate_id"]
        == (candidate_a["candidate_id"])
    )
    assert any(
        evidence.get("historical_candidate_promotion", {}).get("candidate_id")
        == candidate_b["candidate_id"]
        and evidence["historical_candidate_promotion"]["historical_candidate_binding"] == binding_b
        for evidence in case["supporting_source_evidence"]
    )

    forged_case = copy.deepcopy(case)
    forged_promotion = next(
        evidence["historical_candidate_promotion"]
        for evidence in forged_case["supporting_source_evidence"]
        if evidence.get("historical_candidate_promotion", {}).get("candidate_id")
        == candidate_b["candidate_id"]
    )
    forged_promotion["admission_replay_matches_target_revision"] = False
    with pytest.raises(
        CorpusError, match="promoted case evidence does not bind this historical candidate"
    ):
        counterexample_corpus._validate_promoted_historical_candidate(
            candidate_b, {forged_case["case_id"]: forged_case}, attempts_by_id
        )


def test_issue9652_legacy_v1_promotion_attempt_receipt_is_recomputed_on_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy, corpus_root = _legacy_admitted_issue9656_corpus_fixture(tmp_path, monkeypatch)
    candidate = legacy["historical_candidates"][0]
    attempt = next(
        item
        for item in legacy["admission_attempts"]
        if item["attempt_id"] == candidate["promotion_attempt_id"]
    )
    attempt["source_id"] = "0" * 64

    corpus_path = corpus_root / "corpus.json"
    corpus_path.write_text(json.dumps(legacy, sort_keys=True), encoding="utf-8")
    with pytest.raises(CorpusError, match="promotion attempt ID does not bind its stored receipt"):
        load_corpus(corpus_path)


def test_issue9652_legacy_v1_promotion_attempt_decision_must_match_case_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy, corpus_root = _legacy_admitted_issue9656_corpus_fixture(tmp_path, monkeypatch)
    candidate = legacy["historical_candidates"][0]
    attempt = next(
        item
        for item in legacy["admission_attempts"]
        if item["attempt_id"] == candidate["promotion_attempt_id"]
    )
    assert attempt["decision"] == "duplicate"
    attempt["decision"] = "admitted"
    attempt["duplicate_case_id"] = None
    attempt_payload = {
        key: attempt[key]
        for key in (
            "schema_version",
            "source_kind",
            "source_id",
            "decision",
            "blockers",
            "candidate_identity",
            "duplicate_case_id",
            "near_duplicate_report",
        )
    }
    attempt["attempt_id"] = hashlib.sha256(
        counterexample_corpus._stable_json(attempt_payload).encode("utf-8")
    ).hexdigest()
    candidate["promotion_attempt_id"] = attempt["attempt_id"]

    corpus_path = corpus_root / "corpus.json"
    corpus_path.write_text(json.dumps(legacy, sort_keys=True), encoding="utf-8")
    with pytest.raises(
        CorpusError, match="promoted case evidence does not bind this historical candidate"
    ):
        load_corpus(corpus_path)


def test_issue9652_legacy_rekey_cannot_change_identity_source_issue(tmp_path: Path) -> None:
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
    corpus_path = corpus_root / "corpus.json"
    save_corpus(corpus_path, corpus)
    tampered = copy.deepcopy(corpus)
    _rekey_issue9656_import_as_legacy_v1(
        tampered,
        corpus_root,
        version_field="absent",
        identity_source_issue=9657,
    )
    corpus_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")

    with pytest.raises(CorpusError, match="source issue differs from its import receipt"):
        load_corpus(corpus_path)


def test_issue9656_import_receipt_fields_cannot_be_rewritten(
    tmp_path: Path,
) -> None:
    summary_path, materialized, campaign_root, bundle_root = _issue9656_candidate_fixture(tmp_path)
    corpus_root = tmp_path / "corpus"
    corpus, _receipt = import_issue9656_candidates(
        summary_path,
        materialized,
        bundle_root,
        campaign_root,
        new_corpus(),
        corpus_root=corpus_root,
    )
    corpus_path = corpus_root / "corpus.json"
    save_corpus(corpus_path, corpus)

    changed_values = {
        "failed_setup_jobs": 0,
        "failed_replay_attempts": {"attempts": []},
        "criticality_anomaly_counts": {},
        "feasibility_status_counts": {},
        "planner_status_counts_at_import": {},
        "evidence_tier": "paper_facing",
        "claim_boundary": "claims are proven",
    }
    import_id = corpus["historical_candidate_imports"][0]["import_id"]
    for field, value in changed_values.items():
        tampered = copy.deepcopy(corpus)
        import_record = next(
            record
            for record in tampered["historical_candidate_imports"]
            if record["import_id"] == import_id
        )
        import_record[field] = value
        with pytest.raises(CorpusError, match="#9652 import receipt"):
            save_corpus(corpus_path, tampered)

        corpus_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
        with pytest.raises(CorpusError, match="#9652 import receipt"):
            load_corpus(corpus_path)
        corpus_path.write_text(json.dumps(corpus, sort_keys=True), encoding="utf-8")


@pytest.mark.parametrize(
    "field",
    ("criticality", "scenario_id", "scenario_family", "scenario_seed", "planner_id"),
)
def test_issue9656_candidate_identity_projection_cannot_be_rewritten(
    tmp_path: Path, field: str
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
    candidate = copy.deepcopy(corpus["historical_candidates"][0])
    if field == "criticality":
        candidate["criticality"]["metrics"]["minimum_clearance_m"] = 999.0
    elif field == "planner_id":
        candidate["target_planner"]["planner_id"] = "rewritten-planner-alias"
        candidate["planner_status_at_import"] = (
            counterexample_corpus._issue9656_planner_status_at_import(candidate)
        )
    elif field == "scenario_seed":
        candidate["scenario_seed"] = 999
    else:
        candidate[field] = f"rewritten-{field}"
    tampered = copy.deepcopy(corpus)
    tampered["historical_candidates"][0] = candidate

    with pytest.raises(CorpusError, match="pinned"):
        validate_corpus(tampered, corpus_root=corpus_root)


@pytest.mark.parametrize(
    "missing_paths",
    (
        ("evidence_bundle_manifest.json",),
        ("checksums.sha256",),
        ("evidence_bundle_manifest.json", "checksums.sha256"),
    ),
)
def test_issue9656_retained_bundle_receipt_inventory_must_be_complete(
    tmp_path: Path, missing_paths: tuple[str, ...]
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
    corpus_path = corpus_root / "corpus.json"
    save_corpus(corpus_path, corpus)
    tampered = copy.deepcopy(corpus)
    import_record = tampered["historical_candidate_imports"][0]
    import_record["source_files"] = [
        item for item in import_record["source_files"] if item["path"] not in missing_paths
    ]

    with pytest.raises(CorpusError, match="retained source inventory"):
        save_corpus(corpus_path, tampered)

    corpus_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(CorpusError, match="retained source inventory"):
        load_corpus(corpus_path)


@pytest.mark.parametrize("receipt_field", ("path", "stored_path", "sha256"))
def test_issue9656_retained_bundle_receipt_must_match_pinned_identity(
    tmp_path: Path, receipt_field: str
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
    tampered = copy.deepcopy(corpus)
    bundle_receipt = next(
        item
        for item in tampered["historical_candidate_imports"][0]["source_files"]
        if item["path"] == "evidence_bundle_manifest.json"
    )
    if receipt_field == "sha256":
        bundle_receipt[receipt_field] = "0" * 64
    else:
        bundle_receipt[receipt_field] = f"rewritten/{bundle_receipt[receipt_field]}"

    with pytest.raises(CorpusError, match="retained source inventory"):
        validate_corpus(tampered, corpus_root=corpus_root)


@pytest.mark.parametrize(
    "tamper",
    ("source_matrix", "source_planner_config", "normalized_matrix"),
)
def test_issue9656_replay_input_artifact_rejects_coordinated_byte_digest_edits(
    tmp_path: Path, tamper: str
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
    tampered = copy.deepcopy(corpus)
    candidate = tampered["historical_candidates"][0]
    paths = candidate["artifact_paths"]
    replay_inputs = candidate["replay_inputs"]
    provenance = candidate["source_provenance"]

    if tamper == "source_matrix":
        artifact = corpus_root / paths["source_matrix"]
        updated = artifact.read_bytes() + b"\n# coordinated source and digest edit\n"
        artifact.write_bytes(updated)
        digest = hashlib.sha256(updated).hexdigest()
        replay_inputs["source_matrix_sha256"] = digest
        provenance["source_replay_matrix_sha256"] = digest
    elif tamper == "source_planner_config":
        updated = (corpus_root / paths["source_planner_config"]).read_bytes() + b"\n# edited\n"
        for artifact_key in ("source_planner_config", "planner_config"):
            (corpus_root / paths[artifact_key]).write_bytes(updated)
        digest = hashlib.sha256(updated).hexdigest()
        replay_inputs["source_planner_config_sha256"] = digest
        replay_inputs["normalized_planner_config_sha256"] = digest
        provenance["source_planner_config_sha256"] = digest
    else:
        artifact = corpus_root / paths["replay_matrix"]
        matrix = yaml.safe_load(artifact.read_text(encoding="utf-8"))
        matrix["scenarios"][0]["id"] = "coordinated_normalized_matrix_edit"
        updated = yaml.safe_dump(matrix, sort_keys=True, allow_unicode=True).encode("utf-8")
        artifact.write_bytes(updated)
        replay_inputs["normalized_matrix_sha256"] = hashlib.sha256(updated).hexdigest()

    with pytest.raises(CorpusError):
        validate_corpus(tampered, corpus_root=corpus_root)


@pytest.mark.parametrize(
    ("path_collection", "path_field"),
    (
        ("replay_inputs", "source_matrix_path"),
        ("replay_inputs", "source_planner_config_path"),
        ("source_provenance", "source_replay_matrix_path"),
        ("source_provenance", "source_planner_config_path"),
    ),
)
def test_issue9656_candidate_source_input_path_mirrors_are_pinned(
    tmp_path: Path, path_collection: str, path_field: str
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
    tampered = copy.deepcopy(corpus)
    candidate = tampered["historical_candidates"][0]
    candidate[path_collection][path_field] = "replay_input/rewritten.yaml"

    with pytest.raises(CorpusError):
        validate_corpus(tampered, corpus_root=corpus_root)


@pytest.mark.parametrize("artifact_key", ("source_matrix", "source_planner_config"))
def test_issue9656_candidate_source_artifact_paths_are_canonical(
    tmp_path: Path, artifact_key: str
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
    tampered = copy.deepcopy(corpus)
    tampered["historical_candidates"][0]["artifact_paths"][artifact_key] = (
        "historical_candidates/elsewhere/input.yaml"
    )

    with pytest.raises(CorpusError, match="artifact paths"):
        validate_corpus(tampered, corpus_root=corpus_root)


def test_issue9656_revision_mismatch_claim_requires_distinct_full_revisions(
    tmp_path: Path,
) -> None:
    summary_path, materialized, _campaign_root, _bundle_root = _issue9656_candidate_fixture(
        tmp_path
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads((materialized / "manifest.json").read_text(encoding="utf-8"))
    summary_case = next(
        row for row in summary["cases"] if row["replay"]["status"] == "mismatch_different_revision"
    )
    manifest_case = next(
        row for row in manifest["cases"] if row["case_id"] == summary_case["case_id"]
    )
    source_revision = summary_case["replay"]["source_revision"]
    summary_case["replay"]["replay_revision"] = source_revision
    manifest_case["replay"]["replay_revision"] = source_revision

    with pytest.raises(CorpusError, match="distinct full revision bindings"):
        counterexample_corpus._validate_issue9656_summary_manifest_row(summary_case, manifest_case)


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
    _seed_bound_test_case(corpus, corpus_root)

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
    assert corpus["historical_candidate_imports"][0]["candidate_status_counts"] == {
        "pending_exact_replay": 1
    }
    assert corpus["historical_candidate_imports"][0]["source_replay_status_counts"] == {
        "not_attempted": 1
    }
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
    corpus, _receipt, corpus_root = _import(tmp_path)
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
    corpus, _receipt, corpus_root = _import(tmp_path)
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
    corpus, _receipt, corpus_root = _import(tmp_path)
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

    blockers = counterexample_corpus._historical_candidate_source_blockers(
        candidate, corpus, corpus_root
    )
    assert any(expected_blocker in blocker for blocker in blockers)
    with pytest.raises(CorpusError):
        promote_historical_candidate(candidate["candidate_id"], {}, corpus, corpus_root=corpus_root)
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
    ("tamper", "expected_fragment", "pilot_accounting_persists"),
    [
        ("missing_scenario", "pilot_evidence_invalid", False),
        ("missing_provenance", "pilot_evidence_invalid", False),
        ("normalization_hash", "historical_case_invalid", True),
        ("replay_count", "historical_case_invalid", True),
        ("feasibility_verdict", "historical_case_invalid", True),
    ],
)
def test_historical_case_admission_fails_closed_and_retains_zero_pilot(
    tmp_path: Path, tamper: str, expected_fragment: str, pilot_accounting_persists: bool
) -> None:
    with _test_only_reconciled_packet() as payload:
        replay_dir = payload / "historical_issue_1501_failure_0002"
        if tamper == "missing_scenario":
            (replay_dir / "scenario.yaml").unlink()
        elif tamper == "missing_provenance":
            (replay_dir / "replay_1.provenance.json").unlink()
        elif tamper == "normalization_hash":
            path = payload / "path_normalization.json"
            normalization = json.loads(path.read_text())
            replay_provenance = "historical_issue_1501_failure_0002/replay_1.provenance.json"
            row = next(
                item for item in normalization["records"] if item["path"] == replay_provenance
            )
            row["normalized_sha256"] = "0" * 64
            path.write_text(json.dumps(normalization))
            _refresh_bundle_checksum_for_payload(payload, "path_normalization.json")
        else:
            receipt_path = payload / "replay_validation.json"
            receipt = json.loads(receipt_path.read_text())
            case = receipt["historical_issue_1501_case"]
            if tamper == "replay_count":
                case["replay_count"] = 1
            elif tamper == "feasibility_verdict":
                case["dynamic_task_feasibility"] = "feasible"
            receipt_path.write_text(json.dumps(receipt))
            _refresh_bundle_checksum_for_payload(payload, "replay_validation.json")

        corpus, receipt, corpus_root = _import(tmp_path, payload)
        assert receipt["decision"] == "rejected"
        assert any(expected_fragment in blocker for blocker in receipt["blockers"])
        assert bool(corpus["search_runs"]) is pilot_accounting_persists
        if pilot_accounting_persists:
            assert len(corpus["search_runs"]) == 1
            assert corpus["search_runs"][0]["new_counterexamples_discovered"] == 0
        assert corpus["cases"] == []
        assert corpus["admission_attempts"][-1]["decision"] == "rejected"
        validate_corpus(corpus, corpus_root=corpus_root)


def test_historical_normalized_replay_must_match_hash_bound_recorded_payload(
    tmp_path: Path,
) -> None:
    with _test_only_reconciled_packet() as payload:
        replay_path = payload / "historical_issue_1501_failure_0002/replay_1.jsonl"
        episode = json.loads(replay_path.read_text(encoding="utf-8"))
        episode["algorithm_metadata"]["config_hash"] = "review-mutated-config-hash"
        replay_path.write_text(
            json.dumps(episode, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

        normalization_path = payload / "path_normalization.json"
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        replay_row = next(
            row
            for row in normalization["records"]
            if row["path"] == "historical_issue_1501_failure_0002/replay_1.jsonl"
        )
        replay_row["normalized_sha256"] = hashlib.sha256(replay_path.read_bytes()).hexdigest()
        normalization_path.write_text(
            json.dumps(normalization, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _refresh_bundle_checksum_for_payload(
            payload, "historical_issue_1501_failure_0002/replay_1.jsonl"
        )
        _refresh_bundle_checksum_for_payload(payload, "path_normalization.json")

        corpus, receipt, corpus_root = _import(tmp_path, payload)

    assert receipt["decision"] == "rejected"
    assert any("normalized artifact binding differs" in blocker for blocker in receipt["blockers"])
    assert len(corpus["search_runs"]) == 1
    assert corpus["search_runs"][0]["new_counterexamples_discovered"] == 0
    assert corpus["cases"] == []
    validate_corpus(corpus, corpus_root=corpus_root)


def test_pilot_candidate_table_is_bound_to_the_outer_bundle_checksums(tmp_path: Path) -> None:
    with _packet_copy() as payload:
        table_path = payload / "candidate_evaluations.csv"
        table_path.write_bytes(table_path.read_bytes() + b"tampered\n")

        corpus, receipt, _corpus_root = _import(tmp_path, payload)

    assert receipt["decision"] == "rejected"
    assert any("pilot_evidence_invalid" in blocker for blocker in receipt["blockers"])
    assert corpus["search_runs"] == []
    assert corpus["cases"] == []


def test_imported_search_receipts_reject_copied_candidate_table_tampering(tmp_path: Path) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    corpus_path = corpus_root / "corpus.json"
    save_corpus(corpus_path, corpus)

    pilot = corpus["search_runs"][0]
    candidate_receipt = next(
        item for item in pilot["source_files"] if item["path"].endswith("candidate_evaluations.csv")
    )
    artifact_path = corpus_root / candidate_receipt["path"]
    source_digest = candidate_receipt["sha256"]
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered after import\n")

    with pytest.raises(CorpusError, match="search-run artifact digest mismatch"):
        validate_corpus(corpus, corpus_root=corpus_root)
    with pytest.raises(CorpusError, match="search-run artifact digest mismatch"):
        load_corpus(corpus_path)
    assert candidate_receipt["sha256"] == source_digest
    assert corpus["cases"][0]["admissibility"]["verdict"] == "admissible_feasibility_unknown"


def test_issue9645_search_run_fields_are_recomputed_from_pinned_packet(tmp_path: Path) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)

    renamed_pilot = copy.deepcopy(corpus)
    renamed_run = renamed_pilot["search_runs"][0]
    renamed_run["run_id"] = "renamed_run"
    renamed_run["source_issue"] = 999
    renamed_run["new_counterexamples_discovered"] = 1
    with pytest.raises(CorpusError, match="#9645 search-run identity or evidence root"):
        validate_corpus(renamed_pilot, corpus_root=corpus_root)
    orphaned_packet = copy.deepcopy(corpus)
    orphaned_packet["search_runs"] = []
    with pytest.raises(CorpusError, match="#9645 packet has no bound search-run record"):
        validate_corpus(orphaned_packet, corpus_root=corpus_root)

    mutations = {
        "attempted_candidates": 63,
        "completed_candidates": 63,
        "failed_candidates": 1,
        "invalid_candidates": 1,
        "new_counterexamples_discovered": 1,
        "new_counterexamples_admitted": 1,
        "source_revision": "0" * 40,
        "objective": {"name": "changed_objective"},
        "search_space": {"path": "changed.yaml", "sha256": "0" * 64},
        "sampler_seeds": [9999],
        "all_objective_values": [1.0],
        "terminal_decision": "GO",
        "evidence_tier": "paper_facing",
    }
    for field, value in mutations.items():
        tampered = copy.deepcopy(corpus)
        tampered["search_runs"][0][field] = value
        with pytest.raises(CorpusError, match="differs from pinned source evidence"):
            validate_corpus(tampered, corpus_root=corpus_root)

    # Loading a persisted record must enforce the same source reconciliation as direct
    # validation; a changed zero count cannot survive after the initial import.
    corpus_path = corpus_root / "corpus.json"
    save_corpus(corpus_path, corpus)
    persisted = json.loads(corpus_path.read_text(encoding="utf-8"))
    persisted["search_runs"][0]["new_counterexamples_discovered"] = 1
    corpus_path.write_text(json.dumps(persisted, sort_keys=True), encoding="utf-8")
    with pytest.raises(CorpusError, match="new_counterexamples_discovered"):
        load_corpus(corpus_path)


def test_case_admission_recomputes_structural_scenario_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller-supplied valid receipt cannot admit a schema-invalid scenario row."""
    corpus_root = tmp_path / "corpus"
    corpus, _receipt, corpus_root = _import(tmp_path)
    case = _stage_case_under_candidate(
        copy.deepcopy(corpus["cases"][0]), corpus_root, "invalid-structure-probe"
    )
    scenario_path = corpus_root / case["inputs"]["scenario_path"]
    scenario_document = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    scenario_document["scenarios"][0]["typo_field"] = "not-a-canonical-scenario-field"
    scenario_path.write_text(yaml.safe_dump(scenario_document, sort_keys=False), encoding="utf-8")
    scenario_digest = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
    case["inputs"]["scenario_sha256"] = scenario_digest
    case["structural_validation"]["scenario_sha256"] = scenario_digest

    route_payload = yaml.safe_load(
        (corpus_root / case["inputs"]["route_overrides_path"]).read_text(encoding="utf-8")
    )
    scenario_row = scenario_document["scenarios"][0]
    effective_hash = counterexample_corpus.compute_case_effective_scenario_hash(
        scenario_row,
        route_payload,
        case["inputs"]["map_assets"],
    )
    case["effective_scenario_sha256"] = effective_hash
    case["case_id"] = f"case-{effective_hash}"

    # Build a synthetic exact-current replay for the mutated input, then move its
    # artifact into the candidate bundle so admission sees only the staged inputs.
    replay_receipt = _single_replay_admission_receipt(case, corpus_root)
    original_replay_path = replay_receipt["artifact_receipts"][0]["artifact_path"]
    candidate_root = (
        corpus_root / "historical_candidates/invalid-structure-probe/admission_evidence"
    )
    replay_destination = candidate_root / "replay_artifacts" / Path(original_replay_path).name
    replay_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(corpus_root / original_replay_path, replay_destination)
    shutil.rmtree(corpus_root / "cases" / case["case_id"])
    replay_relative = replay_destination.relative_to(corpus_root).as_posix()
    for replay in replay_receipt["replay_artifacts"]:
        replay["path"] = replay_relative
    for artifact in replay_receipt["artifact_receipts"]:
        artifact["artifact_path"] = replay_relative
    for source_file in case["source_evidence"]["corpus_files"]:
        if source_file["path"] == original_replay_path:
            source_file["path"] = replay_relative
            break
    case["replay_receipt"] = replay_receipt
    monkeypatch.setattr(
        counterexample_corpus,
        "_current_target_revision",
        lambda: replay_receipt["target_revision"],
    )

    corpus, admission = counterexample_corpus.admit_case_record(
        case,
        corpus,
        corpus_root=corpus_root,
        artifact_root="historical_candidates/invalid-structure-probe/admission_evidence",
        source_kind="test_structural_admission",
        source_id="schema-invalid-scenario-row",
    )

    assert admission["decision"] == "rejected"
    assert any(
        "Unknown scenario field 'typo_field'." in blocker for blocker in admission["blockers"]
    )
    assert len(corpus["cases"]) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("include", "shared.yaml"),
        ("includes", ["shared.yaml"]),
        ("scenario_files", ["shared.yaml"]),
        ("map_search_paths", ["maps"]),
        ("select_scenarios", ["case"]),
        ("scenario_overrides", {"simulation_config": {"max_episode_steps": 999}}),
        ("scenario_overrides_by_name", {"case": {"simulation_config": {}}}),
    ],
)
def test_case_scenario_validation_rejects_manifest_loader_transforms(
    field: str, value: object
) -> None:
    """Case digests cannot omit loader-applied overrides or external scenario includes."""
    manifest = {"scenarios": [{"name": "case"}], field: value}
    row = counterexample_corpus._case_scenario_structure_row(
        manifest,
        map_asset_path=None,
    )

    assert isinstance(row, str)
    assert field in row


def test_search_run_evidence_rejects_missing_paths_and_receipt_conflicts(tmp_path: Path) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    run = corpus["search_runs"][0]

    missing = copy.deepcopy(corpus)
    missing["search_runs"][0]["source_files"][0]["path"] = (
        "evidence/issue_9645_pilot/not-present.csv"
    )
    with pytest.raises(CorpusError, match="search-run artifact is missing"):
        validate_corpus(missing, corpus_root=corpus_root)

    escaped = copy.deepcopy(corpus)
    escaped["search_runs"][0]["source_files"][0]["path"] = "../outside.csv"
    with pytest.raises(CorpusError, match="unsafe path"):
        validate_corpus(escaped, corpus_root=corpus_root)

    duplicate_run = copy.deepcopy(corpus)
    duplicate_run["search_runs"].append(copy.deepcopy(run))
    with pytest.raises(CorpusError, match="run_id values must be unique"):
        validate_corpus(duplicate_run, corpus_root=corpus_root)

    duplicate_source = copy.deepcopy(corpus)
    duplicate_source["search_runs"][0]["source_files"].append(
        copy.deepcopy(duplicate_source["search_runs"][0]["source_files"][0])
    )
    with pytest.raises(CorpusError, match="duplicate path"):
        validate_corpus(duplicate_source, corpus_root=corpus_root)

    conflict = copy.deepcopy(corpus)
    conflict["search_runs"][0]["manifest_files"][0]["sha256"] = "0" * 64
    with pytest.raises(CorpusError, match="conflicting artifact receipt identity"):
        validate_corpus(conflict, corpus_root=corpus_root)

    duplicate_bundle = copy.deepcopy(corpus)
    bundle_receipt = duplicate_bundle["search_runs"][0]["bundle_receipts"][0]
    duplicate_bundle["search_runs"][0]["bundle_receipts"].append(copy.deepcopy(bundle_receipt))
    with pytest.raises(CorpusError, match="duplicate source identity"):
        validate_corpus(duplicate_bundle, corpus_root=corpus_root)


def test_search_run_evidence_rejects_symlink_escape_and_requires_corpus_root(
    tmp_path: Path,
) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_text("outside evidence\n", encoding="utf-8")
    link = corpus_root / "evidence" / "issue_9645_pilot" / "symlink.csv"
    link.symlink_to(outside)

    escaped_symlink = copy.deepcopy(corpus)
    receipt = escaped_symlink["search_runs"][0]["source_files"][0]
    receipt["path"] = link.relative_to(corpus_root).as_posix()
    receipt["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    with pytest.raises(CorpusError, match="escapes corpus root"):
        validate_corpus(escaped_symlink, corpus_root=corpus_root)
    with pytest.raises(CorpusError, match="corpus_root is required"):
        validate_corpus(corpus)


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


def test_normalized_replay_metric_tampering_is_rejected_before_projection_comparison(
    tmp_path: Path,
) -> None:
    with _test_only_reconciled_packet() as payload:
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
            "normalized artifact binding differs" in blocker for blocker in receipt["blockers"]
        )
        assert corpus["cases"] == []


def test_replayed_historical_origin_is_not_mislabeled_as_raw_historical_match(
    tmp_path: Path,
) -> None:
    corpus_root = tmp_path / "corpus"
    with _test_only_reconciled_packet() as payload:
        corpus, receipt = import_issue9645_packet(payload, new_corpus(), corpus_root=corpus_root)
        case, _source_files, _observations = counterexample_corpus._build_issue9645_historical_case(
            payload
        )
    assert receipt["decision"] == "rejected"
    assert receipt["blockers"] == ["replay_input_binding_unknown_historical"]
    assert corpus["cases"] == []
    assert corpus["planner_evaluations"] == []
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
    corpus_root = tmp_path / "corpus"
    with _test_only_reconciled_packet() as payload:
        corpus, first = import_issue9645_packet(payload, new_corpus(), corpus_root=corpus_root)
        case, _source_files, _observations = counterexample_corpus._build_issue9645_historical_case(
            payload
        )
        case_hash = case["effective_scenario_sha256"]
        corpus, second = import_issue9645_packet(payload, corpus, corpus_root=corpus_root)
    assert first["decision"] == "rejected"
    assert second["decision"] == "rejected"
    assert first["blockers"] == ["replay_input_binding_unknown_historical"]
    assert second["blockers"] == ["replay_input_binding_unknown_historical"]
    assert second["candidate_identity"] == case_hash
    assert len(corpus["cases"]) == 0
    assert len(corpus["planner_evaluations"]) == 0
    assert len(corpus["search_runs"]) == 1
    assert len(corpus["admission_attempts"]) == 1
    assert first["attempt_id"] == second["attempt_id"]

    corpus, _receipt, corpus_root = _import(tmp_path / "synthetic-corpus")
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
    corpus, _receipt, corpus_root = _import(tmp_path)
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
    append_planner_evaluation(corpus, incomplete, corpus_root=corpus_root)

    status = recompute_planner_status(
        corpus,
        planner_id="candidate-planner",
        planner_config_identity="config-under-test",
        corpus_root=corpus_root,
    )
    assert status["status_counts"]["unknown"] == 1
    assert status["cases"][0]["unknown_observation_count"] == 1
    assert status["cases"][0]["reason_codes"] == ["evaluation_evidence_failed"]
    assert len(corpus["planner_evaluations"]) == 3
    incomplete_row = next(
        row for row in corpus["planner_evaluations"] if row["planner_id"] == "candidate-planner"
    )
    assert incomplete_row["episode_sha256"] is None
    validate_corpus(corpus, corpus_root=corpus_root)


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

    validate_corpus(corpus, corpus_root=corpus_root)
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

    validate_corpus(legacy, corpus_root=corpus_root)
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
    assert first["review_marker"] == "AI-GENERATED NEEDS-REVIEW"
    assert json.loads((tmp_path / "slice-a/manifest.json").read_text(encoding="utf-8")) == first
    replay_matrix_text = (tmp_path / "slice-a/replay_matrix.yaml").read_text(encoding="utf-8")
    assert replay_matrix_text.startswith("# AI-GENERATED NEEDS-REVIEW\n")
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
    validate_corpus(corpus, corpus_root=corpus_root)


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
    corpus, _receipt, corpus_root = _import(tmp_path)
    wrong = copy.deepcopy(corpus["planner_evaluations"][0])
    wrong["effective_scenario_sha256"] = "0" * 64
    with pytest.raises(CorpusError, match="replay_receipt_effective_scenario_sha256_mismatch"):
        append_planner_evaluation(corpus, wrong, corpus_root=corpus_root)


def test_validate_corpus_rejects_planner_evaluation_for_absent_case(
    tmp_path: Path,
) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    corpus["planner_evaluations"][0]["case_id"] = f"case-{'0' * 64}"

    with pytest.raises(CorpusError, match="planner evaluation references absent case"):
        validate_corpus(corpus, corpus_root=corpus_root)


def test_validate_corpus_rejects_excluded_invalid_and_incomplete_case_records(
    tmp_path: Path,
) -> None:
    corpus, _receipt, corpus_root = _import(tmp_path)
    invalid = copy.deepcopy(corpus)
    invalid["cases"][0]["structural_validation"]["status"] = "invalid"
    invalid["cases"][0]["admissibility"]["verdict"] = "excluded"
    with pytest.raises(CorpusError):
        validate_corpus(invalid, corpus_root=corpus_root)
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
        validate_corpus(incomplete, corpus_root=corpus_root)


def test_corpus_cli_import_status_and_slice_work_without_simulator_run(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_path = corpus_root / "corpus.json"
    receipt_path = tmp_path / "admission.json"
    corpus_root.mkdir()
    with _test_only_reconciled_packet() as payload:
        assert (
            corpus_cli_main(
                [
                    "import-9645",
                    "--payload",
                    str(payload),
                    "--corpus",
                    str(corpus_path),
                    "--corpus-root",
                    str(corpus_root),
                    "--output",
                    str(receipt_path),
                ]
            )
            == 2
        )
    receipt = json.loads(receipt_path.read_text())
    assert receipt["decision"] == "rejected"
    assert receipt["blockers"] == ["replay_input_binding_unknown_historical"]
    corpus = load_corpus(corpus_path)
    assert corpus["cases"] == []
    assert corpus["planner_evaluations"] == []
    assert corpus["search_runs"][0]["new_counterexamples_discovered"] == 0

    _seed_bound_test_case(corpus, corpus_root)
    save_corpus(corpus_path, corpus)

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
    manifest = json.loads((tmp_path / "slice-manifest.json").read_text())
    assert manifest["case_count"] == 1
    assert manifest["cases"][0]["replay_input_binding_status"] == "bound"
    source_scenario = yaml.safe_load(
        (corpus_root / case["inputs"]["scenario_path"]).read_text(encoding="utf-8")
    )["scenarios"][0]
    exported_scenario = yaml.safe_load((slice_path / "replay_matrix.yaml").read_text())[
        "scenarios"
    ][0]
    assert scenario_semantic_sha256(source_scenario, seed=case["scenario_seed"]) == (
        scenario_semantic_sha256(exported_scenario, seed=case["scenario_seed"])
    )
