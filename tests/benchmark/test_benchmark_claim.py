"""Tests for benchmark claim artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from robot_sf.benchmark.benchmark_claim import (
    BENCHMARK_CLAIM_SCHEMA_VERSION,
    BenchmarkClaimError,
    build_benchmark_claim,
    load_benchmark_claim_schema,
)
from robot_sf.benchmark.cli import cli_main
from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256

FIXTURE_ROOT = Path("tests/fixtures/benchmark_claim/v1")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON fixture file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _claim_inputs() -> dict[str, Path | str]:
    """Return the checked-in tiny fixture set for claim tests."""
    matrix = FIXTURE_ROOT / "scenario_matrix.yaml"
    training = FIXTURE_ROOT / "episodes" / "training.jsonl"
    validation = FIXTURE_ROOT / "episodes" / "validation.jsonl"
    final = FIXTURE_ROOT / "episodes" / "final.jsonl"
    policy_metadata = FIXTURE_ROOT / "policy_metadata.json"
    aggregate = FIXTURE_ROOT / "aggregate.json"
    return {
        "matrix": matrix,
        "matrix_sha256": _sha256(matrix),
        "training": training,
        "validation": validation,
        "final": final,
        "policy_metadata": policy_metadata,
        "aggregate": aggregate,
    }


def test_build_benchmark_claim_validates_against_schema() -> None:
    """Builder output should be a compact schema-valid claim artifact."""
    inputs = _claim_inputs()

    claim = build_benchmark_claim(
        claim_id="paper-claim-smoke",
        statement="Final benchmark smoke claim",
        scenario_matrix_path=Path(str(inputs["matrix"])),
        scenario_matrix_sha256=str(inputs["matrix_sha256"]),
        policy_metadata_path=Path(str(inputs["policy_metadata"])),
        final_benchmark_episodes=[Path(str(inputs["final"]))],
        training_episodes=[Path(str(inputs["training"]))],
        validation_episodes=[Path(str(inputs["validation"]))],
        aggregate_reports=[Path(str(inputs["aggregate"]))],
        dependency_group="dev",
    )

    assert claim["schema_version"] == BENCHMARK_CLAIM_SCHEMA_VERSION
    assert claim["claim_id"] == "paper-claim-smoke"
    assert claim["evidence"]["scenario_matrix"]["sha256"] == inputs["matrix_sha256"]
    assert claim["evidence"]["episode_groups"]["training"][0]["episode_count"] == 1
    assert claim["evidence"]["episode_groups"]["validation"][0]["episode_count"] == 1
    assert claim["evidence"]["episode_groups"]["final_benchmark"][0]["episode_count"] == 1
    assert claim["evidence"]["policy_metadata"]["schema_version"] == "policy-metadata.v1"
    assert claim["evidence"]["policy_metadata"]["policies"][0]["sha256"] == "a" * 64
    assert claim["environment"]["uv_lock_sha256"]

    jsonschema.validate(claim, load_benchmark_claim_schema())


def test_benchmark_claim_cli_writes_json_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public benchmark CLI should emit a claim JSON artifact from tiny fixtures."""
    inputs = _claim_inputs()
    output = tmp_path / "claim.json"

    exit_code = cli_main(
        [
            "claim",
            "--claim-id",
            "paper-claim-cli",
            "--statement",
            "CLI smoke claim",
            "--scenario-matrix",
            str(inputs["matrix"]),
            "--scenario-matrix-sha256",
            str(inputs["matrix_sha256"]),
            "--policy-metadata",
            str(inputs["policy_metadata"]),
            "--training-episodes",
            str(inputs["training"]),
            "--validation-episodes",
            str(inputs["validation"]),
            "--final-benchmark-episodes",
            str(inputs["final"]),
            "--aggregate-report",
            str(inputs["aggregate"]),
            "--dependency-group",
            "dev",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == BENCHMARK_CLAIM_SCHEMA_VERSION
    assert payload["evidence"]["episode_groups"]["final_benchmark"][0]["path"]
    stdout_payload = json.loads(capsys.readouterr().out)
    assert stdout_payload["claim_path"] == str(output)
    assert stdout_payload["schema_version"] == BENCHMARK_CLAIM_SCHEMA_VERSION


def test_benchmark_claim_fails_closed_for_missing_hashes_and_versions(tmp_path: Path) -> None:
    """Missing policy hashes, matrix hashes, and schema versions should be actionable errors."""
    inputs = _claim_inputs()
    bad_policy_metadata = tmp_path / "bad_policy_metadata.json"
    _write_json(
        bad_policy_metadata,
        {"schema_version": "policy-metadata.v1", "policies": [{"policy_id": "ppo-final"}]},
    )

    with pytest.raises(BenchmarkClaimError, match="scenario_matrix_sha256"):
        build_benchmark_claim(
            claim_id="bad-matrix",
            statement="Bad matrix hash",
            scenario_matrix_path=Path(str(inputs["matrix"])),
            scenario_matrix_sha256="",
            policy_metadata_path=Path(str(inputs["policy_metadata"])),
            final_benchmark_episodes=[Path(str(inputs["final"]))],
        )

    with pytest.raises(BenchmarkClaimError, match="policies\\[0\\]\\.sha256"):
        build_benchmark_claim(
            claim_id="bad-policy",
            statement="Bad policy hash",
            scenario_matrix_path=Path(str(inputs["matrix"])),
            scenario_matrix_sha256=str(inputs["matrix_sha256"]),
            policy_metadata_path=bad_policy_metadata,
            final_benchmark_episodes=[Path(str(inputs["final"]))],
        )

    bad_final = tmp_path / "bad_final.jsonl"
    bad_final.write_text(
        json.dumps({"episode_id": "missing-schema-version", "scenario_id": "s1"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkClaimError, match="schema version"):
        build_benchmark_claim(
            claim_id="bad-episode-schema",
            statement="Bad episode schema",
            scenario_matrix_path=Path(str(inputs["matrix"])),
            scenario_matrix_sha256=str(inputs["matrix_sha256"]),
            policy_metadata_path=Path(str(inputs["policy_metadata"])),
            final_benchmark_episodes=[bad_final],
        )

    bad_json = tmp_path / "bad_json.jsonl"
    bad_json.write_text('{"version": "v1", "seed": 1}\n{bad json}\n', encoding="utf-8")
    with pytest.raises(
        BenchmarkClaimError,
        match="Line 2 in episode_groups\\.final_benchmark\\[0\\]",
    ):
        build_benchmark_claim(
            claim_id="bad-episode-json",
            statement="Bad episode JSON",
            scenario_matrix_path=Path(str(inputs["matrix"])),
            scenario_matrix_sha256=str(inputs["matrix_sha256"]),
            policy_metadata_path=Path(str(inputs["policy_metadata"])),
            final_benchmark_episodes=[bad_json],
        )
