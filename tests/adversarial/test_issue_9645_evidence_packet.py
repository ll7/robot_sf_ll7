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
        } <= set(row["field_rewrites"])


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
    assert len(canonical_1) == metadata["projected_bytes"] == 23907
    assert hashlib.sha256(canonical_1).hexdigest() == metadata["digest"]
    assert metadata["digest"] == (
        "5b5047919ac185317eeef78268564a3ed1a9667ddab915ecf1e5d4ddf1aeb1f6"
    )
    assert metadata["legacy_recorded_signature_status"] == "not_independently_reconstructed"
    assert metadata["legacy_recorded_signature_sha256"] == (
        "6e34dc5d810fecaea9921934fd6b386e4c21319a3e59d225538fb8b0cb818c00"
    )
