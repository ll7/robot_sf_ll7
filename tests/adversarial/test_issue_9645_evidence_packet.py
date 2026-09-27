"""Integrity checks for the committed issue #9645 evidence packet."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

PAYLOAD = (
    Path(__file__).resolve().parents[2]
    / "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload"
)
REPO_ROOT = Path(__file__).resolve().parents[2]

_PROVENANCE_ADDITIONS = {
    "inputs.scenario_matrix.producer_output_path",
    "inputs.schema_path.producer_path_at_source_revision",
    "raw_artifacts[0].bundle_retention_status",
    "raw_artifacts[0].producer_output_path",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_file(provenance_path: str) -> Path:
    path = REPO_ROOT / provenance_path
    assert path.is_file(), provenance_path
    return path


def _historical_replay_projection(path: Path) -> tuple[dict[str, Any], bytes]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    record = deepcopy(rows[0])
    for dotted_path in (
        "timestamps.start",
        "timestamps.end",
        "timing.steps_per_second",
        "wall_time_sec",
    ):
        keys = dotted_path.split(".")
        value = record
        for key in keys[:-1]:
            value = value[key]
        value.pop(keys[-1])
    canonical = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return record, canonical


def test_final_provenance_hashes_and_bundle_only_fields_are_recorded() -> None:
    normalization = _read_json(PAYLOAD / "path_normalization.json")
    provenance_records = [
        row for row in normalization["records"] if row["path"].endswith(".provenance.json")
    ]

    assert len(provenance_records) == 5
    for row in provenance_records:
        final_path = PAYLOAD / row["path"]
        assert _sha256(final_path) == row["normalized_sha256"]
        assert len(row["source_sha256_before_path_normalization"]) == 64
        assert _PROVENANCE_ADDITIONS == set(row["field_additions"])
        assert {
            "inputs.scenario_matrix.path",
            "inputs.schema_path.path",
            "raw_artifacts[0].path",
            "raw_artifacts[0].sha256",
        } <= set(row["field_rewrites"])


def test_route_path_normalization_is_hash_bound_for_candidates_and_recorded_replays() -> None:
    normalization = _read_json(PAYLOAD / "path_normalization.json")
    ledger = {row["path"]: row for row in normalization["records"]}
    jsonl_rows = [row for row in normalization["records"] if row["path"].endswith(".jsonl")]
    assert len(jsonl_rows) == 74

    artifact_map = _read_json(PAYLOAD / "pilot_report_artifact_path_map.v1.json")
    candidates = [
        row
        for row in artifact_map["bindings"]
        if row.get("artifact_kind") == "candidate_episode_records"
    ]
    assert len(candidates) == 64
    assert artifact_map["totals"]["artifact_count"] == len(artifact_map["bindings"])
    assert artifact_map["totals"]["exact_artifact_count"] == sum(
        row["retention_status"] == "retained_exact_copy" for row in artifact_map["bindings"]
    )
    assert artifact_map["totals"]["path_normalized_candidate_episode_record_count"] == 64
    assert artifact_map["totals"]["exact_artifact_count"] == 5
    assert artifact_map["claim_boundary"] == (
        "Mixed retention: bindings labeled retained_exact_copy preserve producer bytes; "
        "bindings labeled retained_path_normalized_copy are path-rewritten copies with "
        "source_sha256_before_path_normalization and normalized_sha256 bindings. Totals report "
        "5 exact artifacts and 64 path-normalized candidate episode records. No feasibility "
        "or planner safety claim."
    )
    rebase = _read_json(PAYLOAD / "reproduction_inputs/input_rebase_map.v1.json")
    artifact_map_sha256 = _sha256(PAYLOAD / "pilot_report_artifact_path_map.v1.json")
    assert rebase["source_episode_artifact_map_sha256"] == artifact_map_sha256
    recovery = _read_json(PAYLOAD / "run_metadata.json")["candidate_episode_record_recovery"]
    assert recovery["count"] == 64
    assert recovery["artifact_path_map_sha256"] == artifact_map_sha256
    assert recovery["total_size_bytes"] == artifact_map["totals"]["candidate_episode_record_bytes"]
    assert recovery["absolute_path_fields"]["scenario_params.route_overrides_file"] == 0
    rebased_candidates = {
        row["bundle_path"]: row
        for manifest in rebase["manifest_bindings"]
        for row in manifest["episode_record_bindings"]
    }

    payload_relative = PAYLOAD.relative_to(REPO_ROOT)
    for binding in candidates:
        relative = Path(binding["bundle_path"]).relative_to(payload_relative)
        path = PAYLOAD / relative
        digest = _sha256(path)
        normalization_row = ledger[str(relative)]
        expected_route = (
            binding["producer_output_path"].removesuffix("episode_records.jsonl")
            + "route_overrides.yaml"
        )
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        assert len(rows) == 1
        route_path = rows[0]["scenario_params"]["route_overrides_file"]
        assert route_path == expected_route
        assert not Path(route_path).is_absolute()
        assert binding["retention_status"] == "retained_path_normalized_copy"
        assert binding["sha256"] == binding["normalized_sha256"] == digest
        assert (
            binding["source_sha256_before_path_normalization"]
            == normalization_row["source_sha256_before_path_normalization"]
        )
        assert normalization_row["normalized_sha256"] == digest
        assert binding["size_bytes"] == path.stat().st_size
        assert binding["source_size_bytes"] >= binding["size_bytes"]
        rebased = rebased_candidates[binding["bundle_path"]]
        assert rebased["sha256"] == digest
        assert (
            rebased["source_sha256_before_path_normalization"]
            == binding["source_sha256_before_path_normalization"]
        )
        assert rebased["normalized_sha256"] == digest

    recorded_to_normalized = {
        "historical_issue_1501_failure_0002/replay_1_recorded.jsonl": "historical_issue_1501_failure_0002/replay_1.jsonl",
        "historical_issue_1501_failure_0002/replay_2_recorded.jsonl": "historical_issue_1501_failure_0002/replay_2.jsonl",
        "representative_success/episode_records_original_recorded.jsonl": "representative_success/episode_records_original.jsonl",
        "representative_success/episode_records_replay_recorded.jsonl": "representative_success/episode_records_replay.jsonl",
        "smoke/corrected_success_smoke/episode_records_recorded.jsonl": "smoke/corrected_success_smoke/episode_records.jsonl",
    }
    for recorded, normalized in recorded_to_normalized.items():
        path = PAYLOAD / recorded
        row = ledger[recorded]
        actual = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        counterpart = json.loads((PAYLOAD / normalized).read_text(encoding="utf-8").splitlines()[0])
        route_path = actual["scenario_params"]["route_overrides_file"]
        assert route_path == counterpart["scenario_params"]["route_overrides_file"]
        assert not Path(route_path).is_absolute()
        assert row["normalized_sha256"] == _sha256(path)
        assert len(row["source_sha256_before_path_normalization"]) == 64


def test_rebuilt_report_provenance_binds_inputs_and_generated_outputs() -> None:
    provenance = _read_json(PAYLOAD / "report_provenance.json")
    source_metadata_path = REPO_ROOT / provenance["experiment_source_commit_evidence_path"]
    source_metadata = _read_json(source_metadata_path)
    assert source_metadata_path == PAYLOAD / "run_metadata.json"
    assert provenance["experiment_source_commit"] == source_metadata["experiment_source_commit"]
    assert provenance["experiment_source_commit_sha256"] == _sha256(source_metadata_path)
    assert provenance["comparison_sha256"] == _sha256(PAYLOAD / "pilot_comparison.json")
    assert provenance["manifest_path_map_sha256"] == _sha256(
        PAYLOAD / "reproduction_inputs/pilot_report_manifest_path_map.v1.json"
    )
    assert provenance["reproduction_input_rebase_map_sha256"] == _sha256(
        PAYLOAD / "reproduction_inputs/input_rebase_map.v1.json"
    )
    assert provenance["expected_episode_record_count"] == 64
    assert (
        provenance["episode_record_path_field_disposition"]["absolute_producer_paths_remaining"]
        == 0
    )
    assert (
        provenance["episode_record_path_field_disposition"][
            "portable_replay_inputs_retained_for_all_candidates"
        ]
        is False
    )
    assert len(provenance["outputs"]) == 3
    for output in provenance["outputs"]:
        path = _repo_file(output["path"])
        assert _sha256(path) == output["sha256"]
        assert path.stat().st_size == output["size_bytes"]


def test_historical_replay_provenance_binds_recorded_inputs_and_outputs() -> None:
    validation = _read_json(PAYLOAD / "replay_validation.json")["historical_issue_1501_case"]
    assert validation["input_binding_status"] == "unknown_historical"
    assert validation["admission_status"] == "not_admitted"
    assert validation["regression_status"] == "pending_exact_historical_input_binding"
    assert validation["dynamic_task_feasibility"].startswith("unknown_")

    replay_paths = {
        "replay_1": "historical_issue_1501_failure_0002/replay_1_recorded.jsonl",
        "replay_2": "historical_issue_1501_failure_0002/replay_2_recorded.jsonl",
    }
    provenance_paths = {
        "replay_1": "historical_issue_1501_failure_0002/replay_1.provenance.json",
        "replay_2": "historical_issue_1501_failure_0002/replay_2.provenance.json",
    }
    normalized_hashes = {
        row["path"]: row["normalized_sha256"]
        for row in _read_json(PAYLOAD / "path_normalization.json")["records"]
    }
    validation_bindings = {item["role"]: item for item in validation["replay_artifact_bindings"]}

    for role, recorded_relpath in replay_paths.items():
        provenance_relpath = provenance_paths[role]
        provenance = _read_json(PAYLOAD / provenance_relpath)
        recorded_episode = PAYLOAD / recorded_relpath
        raw_artifact = provenance["raw_artifacts"][0]

        assert raw_artifact["sha256"] == _sha256(recorded_episode)
        assert raw_artifact["path"].endswith(recorded_relpath)
        assert normalized_hashes[provenance_relpath] == _sha256(PAYLOAD / provenance_relpath)
        assert provenance["run"]["repo_commit"] == validation["regeneration_commit"]
        for input_id in ("scenario_matrix", "schema_path"):
            source_input = provenance["inputs"][input_id]
            assert source_input["sha256"] == _sha256(_repo_file(source_input["path"]))

        validation_binding = validation_bindings[role]
        assert validation_binding["path"].endswith(recorded_relpath)
        assert validation_binding["sha256"] == _sha256(recorded_episode)


def test_historical_replay_comparison_signature_is_reproducible() -> None:
    validation = _read_json(PAYLOAD / "replay_validation.json")["historical_issue_1501_case"]
    replay_1 = PAYLOAD / "historical_issue_1501_failure_0002/replay_1_recorded.jsonl"
    replay_2 = PAYLOAD / "historical_issue_1501_failure_0002/replay_2_recorded.jsonl"
    metadata = validation["comparison_signature"]

    projection_1, canonical_1 = _historical_replay_projection(replay_1)
    projection_2, canonical_2 = _historical_replay_projection(replay_2)

    assert replay_1.read_bytes() != replay_2.read_bytes()
    assert validation["replay_file_bytes_identical"] is False
    assert validation["comparison_signatures_match"] is True
    assert "replays_identical" not in validation
    assert projection_1 == projection_2
    assert canonical_1 == canonical_2
    assert len(canonical_1) == metadata["projected_bytes"] == 23877
    assert hashlib.sha256(canonical_1).hexdigest() == metadata["digest"]
    assert metadata["digest"] == (
        "8fe60a2d9bb0c80ab969151fa098e18beb697485c41a964c8f58eb7044c05899"
    )
    assert metadata["legacy_recorded_signature_status"] == "not_independently_reconstructed"
    assert metadata["legacy_recorded_signature_sha256"] == (
        "6e34dc5d810fecaea9921934fd6b386e4c21319a3e59d225538fb8b0cb818c00"
    )
