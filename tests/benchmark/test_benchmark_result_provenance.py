"""Tests for the benchmark result provenance manifest."""

from __future__ import annotations

import json
import platform
from hashlib import sha256
from io import StringIO
from pathlib import Path

import numba
import numpy as np
import pytest

from robot_sf._execution_context import EXECUTION_CONTEXT_FIELDS, execution_context_digest
from robot_sf._numerical_thread_env import pin_thread_env_for_determinism
from robot_sf.benchmark.result_provenance import (
    INPUT_BINDING_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ProvenanceArtifactError,
    ProvenanceRequiredFieldError,
    ProvenanceValidationError,
    _cpu_model,
    build_execution_context_provenance,
    build_result_provenance_manifest,
    build_row_result_provenance,
    manifest_path_for_result_jsonl,
    validate_result_provenance_manifest,
    write_result_provenance_manifest,
)
from scripts.validation import check_benchmark_result_provenance

# Canonical context fields the generic benchmark provenance path can actually
# observe. ``cpu_only``/``workers`` belong to callers that enforce or measure the
# execution mode (exact-repeat), not to every benchmark run.
_OBSERVED_CONTEXT_FIELDS = tuple(
    field for field in EXECUTION_CONTEXT_FIELDS if field not in {"cpu_only", "workers"}
)


def _write_input_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create deterministic benchmark input files for complete manifests."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    schema_path = tmp_path / "schema.json"
    scenario_path = tmp_path / "scenarios.yaml"
    algo_config_path = tmp_path / "algo.yaml"
    schema_path.write_text('{"type":"object"}\n', encoding="utf-8")
    scenario_path.write_text("- id: a\n", encoding="utf-8")
    algo_config_path.write_text("algo: goal\n", encoding="utf-8")
    return schema_path, scenario_path, algo_config_path


def _complete_manifest(
    tmp_path: Path,
    *,
    algo_config: bool = True,
    scenario_file: bool = True,
) -> dict[str, object]:
    """Build a deterministic complete manifest with byte-resolvable inputs."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = tmp_path / "episodes.jsonl"
    jsonl_path.write_text(
        '{"episode_id":"test--0","scenario_id":"test","seed":0}\n',
        encoding="utf-8",
    )
    schema_path, scenario_path, algo_config_path = _write_input_files(tmp_path)
    if not scenario_file:
        scenario_path.unlink()
    return build_result_provenance_manifest(
        out_path=jsonl_path,
        episode_records=[
            {
                "episode_id": "test--0",
                "scenario_id": "test",
                "seed": 0,
                "config_hash": "abc",
                "git_hash": "def",
            },
        ],
        schema_path=schema_path,
        scenario_path=scenario_path,
        scenarios=[{"name": "test"}],
        algo="goal",
        algo_config_path=algo_config_path if algo_config else None,
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=1,
        written=1,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )


def test_manifest_path_convention() -> None:
    """The provenance manifest path appends .provenance.json to the JSONL path."""
    jsonl_path = Path("output/some_run/episodes.jsonl")
    manifest_path = manifest_path_for_result_jsonl(jsonl_path)
    assert manifest_path == Path("output/some_run/episodes.jsonl.provenance.json")


def test_build_manifest_has_correct_schema_version() -> None:
    """Every emitted manifest must carry the expected schema version."""
    manifest = build_result_provenance_manifest(
        out_path=Path("episodes.jsonl"),
        episode_records=[],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=0,
        written=0,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["input_binding_schema_version"] == INPUT_BINDING_SCHEMA_VERSION


def test_build_manifest_records_execution_context() -> None:
    """The manifest records the complete canonical execution context and digest."""
    manifest = build_result_provenance_manifest(
        out_path=Path("episodes.jsonl"),
        episode_records=[],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=0,
        written=0,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )
    ctx = manifest["run"]["execution_context"]
    assert set(ctx) == {"hostname", *_OBSERVED_CONTEXT_FIELDS, "execution_context_sha256"}
    assert isinstance(ctx["hostname"], str) and ctx["hostname"]
    assert isinstance(ctx["thread_env"], dict)
    assert set(ctx["thread_env"]) >= {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"}
    canonical_context = {field: ctx[field] for field in _OBSERVED_CONTEXT_FIELDS}
    assert canonical_context["numpy_version"] == np.__version__
    assert canonical_context["numba_version"] == str(numba.__version__)
    assert ctx["execution_context_sha256"] == execution_context_digest(canonical_context)


def test_manifest_execution_context_does_not_assert_unobserved_execution_mode() -> None:
    """The generic run context must not claim CPU-only single-worker execution.

    ``build_execution_context_provenance`` runs for every benchmark campaign,
    including multi-worker camera-ready runs whose real worker count is recorded
    separately in run metadata. Restating an unobserved ``cpu_only``/``workers``
    value here would write a false provenance record (issue #7128 review).
    """
    provenance = build_execution_context_provenance()

    assert "cpu_only" not in provenance
    assert "workers" not in provenance


@pytest.mark.parametrize("field", ["numpy_version", "numba_version"])
def test_execution_context_digest_binds_runtime_versions(field: str) -> None:
    """Changing either runtime version changes the canonical context digest."""
    provenance = build_execution_context_provenance()
    canonical_context = {name: provenance[name] for name in _OBSERVED_CONTEXT_FIELDS}
    drifted_context = dict(canonical_context)
    drifted_context[field] = f"{drifted_context[field]}-drifted"

    assert execution_context_digest(drifted_context) != provenance["execution_context_sha256"]


def test_execution_context_captures_thread_env(monkeypatch) -> None:
    """Thread-env snapshot reflects the active environment so pinned runs are recordable."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    ctx = build_execution_context_provenance()
    assert ctx["thread_env"]["OMP_NUM_THREADS"] == "1"
    assert ctx["thread_env"]["MKL_NUM_THREADS"] == "4"
    assert ctx["thread_env"]["OPENBLAS_NUM_THREADS"] is None


def test_pin_thread_env_forces_one_over_inherited_values(monkeypatch) -> None:
    """The camera-ready determinism contract overrides inherited thread counts."""
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "32")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    assert pin_thread_env_for_determinism() == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    assert build_execution_context_provenance()["thread_env"] == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def test_cpu_model_malformed_proc_line_uses_platform_fallback(monkeypatch) -> None:
    """A colon-free proc model line cannot crash provenance capture."""
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: StringIO("model name malformed\n"))
    monkeypatch.setattr(platform, "processor", lambda: "Fallback CPU")
    assert _cpu_model() == "Fallback CPU"


def test_build_manifest_records_optional_algo_config_absence() -> None:
    """When algo_config_path is None the input entry should be explicit."""
    manifest = build_result_provenance_manifest(
        out_path=Path("episodes.jsonl"),
        episode_records=[],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=0,
        written=0,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )
    algo_input = manifest["inputs"]["algo_config"]
    assert algo_input["artifact_status"] == "not_provided"
    assert algo_input["path"] is None
    assert algo_input["sha256"] is None


def test_build_manifest_treats_directory_algo_config_as_missing(tmp_path: Path) -> None:
    """Directory algorithm config paths are not treated as readable files."""
    algo_config_dir = tmp_path / "algo-config-dir"
    algo_config_dir.mkdir()

    manifest = build_result_provenance_manifest(
        out_path=Path("episodes.jsonl"),
        episode_records=[],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[],
        algo="goal",
        algo_config_path=algo_config_dir,
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=0,
        written=0,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )

    assert manifest["inputs"]["algo_config"] == {
        "path": str(algo_config_dir),
        "sha256": None,
        "artifact_status": "missing",
    }


def test_build_manifest_treats_directory_scenario_matrix_as_missing(
    tmp_path: Path,
) -> None:
    """Directory scenario matrix paths are not treated as readable files."""
    scenario_dir = tmp_path / "scenario-dir"
    scenario_dir.mkdir()

    manifest = build_result_provenance_manifest(
        out_path=Path("episodes.jsonl"),
        episode_records=[],
        schema_path="schema.json",
        scenario_path=scenario_dir,
        scenarios=[],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=0,
        written=0,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )

    assert manifest["inputs"]["scenario_matrix"] == {
        "path": str(scenario_dir),
        "sha256": None,
        "artifact_status": "missing",
    }


def test_build_manifest_with_rows(tmp_path: Path) -> None:
    """A manifest built from episode records should link each row correctly."""
    jsonl_path = tmp_path / "episodes.jsonl"
    jsonl_path.write_text(
        '{"episode_id":"scenario-a--7","scenario_id":"scenario-a","seed":7}\n'
        '{"episode_id":"scenario-b--42","scenario_id":"scenario-b","seed":42}\n',
        encoding="utf-8",
    )

    records = [
        {
            "episode_id": "scenario-a--7",
            "scenario_id": "scenario-a",
            "seed": 7,
            "config_hash": "abc123",
            "git_hash": "deadbeef",
        },
        {
            "episode_id": "scenario-b--42",
            "scenario_id": "scenario-b",
            "seed": 42,
            "config_hash": "def456",
            "git_hash": "cafebabe",
        },
    ]

    manifest = build_result_provenance_manifest(
        out_path=jsonl_path,
        episode_records=records,
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=2,
        written=2,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )

    assert len(manifest["rows"]) == 2
    row0 = manifest["rows"][0]
    assert row0["episode_id"] == "scenario-a--7"
    assert row0["scenario_id"] == "scenario-a"
    assert row0["seed"] == 7
    assert row0["config_hash"] == "abc123"
    assert row0["repo_commit"] == "deadbeef"
    assert row0["jsonl_line"] == 0
    assert row0["raw_artifact"] == str(jsonl_path)
    assert row0["simulator_settings"]["horizon"] == 100
    assert row0["simulator_settings"]["dt"] == 0.1
    assert row0["simulator_settings"]["record_forces"] is True
    assert row0["postprocessing"] == [
        {"step": "compute_all_metrics", "status": "completed"},
        {"step": "post_process_metrics", "status": "completed"},
    ]

    row1 = manifest["rows"][1]
    assert row1["episode_id"] == "scenario-b--42"
    assert row1["seed"] == 42
    assert row1["jsonl_line"] == 1


def test_build_row_result_provenance_uses_supplied_postprocessing_steps() -> None:
    """Custom post-processing steps are preserved when supplied."""
    row = build_row_result_provenance(
        episode_id="scenario-a--7",
        scenario_id="scenario-a",
        seed=7,
        config_hash="abc123",
        repo_commit="deadbeef",
        raw_artifact_path="episodes.jsonl",
        jsonl_line=0,
        dt=0.1,
        horizon=100,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
        noise_hash="noise-hash",
        tracking_precision_hash="tracking-hash",
        postprocessing_steps=[{"step": "custom_step", "status": "completed"}],
    )

    assert row["postprocessing"] == [{"step": "custom_step", "status": "completed"}]
    assert row["simulator_settings"]["observation_noise_hash"] == "noise-hash"
    assert row["simulator_settings"]["tracking_precision_hash"] == "tracking-hash"


def test_validator_passes_complete_manifest(tmp_path: Path) -> None:
    """A well-formed manifest should pass validation without raising."""
    manifest = _complete_manifest(tmp_path)
    validate_result_provenance_manifest(manifest)


def test_validator_fails_on_available_input_missing_sha256(tmp_path: Path) -> None:
    """Available benchmark inputs must carry a full SHA-256 digest."""
    manifest = _complete_manifest(tmp_path)
    manifest["inputs"]["schema_path"]["sha256"] = None

    with pytest.raises(ProvenanceRequiredFieldError, match="inputs.schema_path.sha256"):
        validate_result_provenance_manifest(manifest)


def test_validator_fails_on_available_input_malformed_sha256(tmp_path: Path) -> None:
    """Short or non-hex input hashes are not complete evidence."""
    manifest = _complete_manifest(tmp_path)
    manifest["inputs"]["schema_path"]["sha256"] = "abc123"

    with pytest.raises(ProvenanceRequiredFieldError, match="inputs.schema_path.sha256"):
        validate_result_provenance_manifest(manifest)


def test_validator_fails_on_required_input_marked_missing(tmp_path: Path) -> None:
    """Required inputs cannot be marked missing for complete evidence."""
    manifest = _complete_manifest(tmp_path)
    manifest["inputs"]["scenario_matrix"] = {
        "path": str(tmp_path / "missing-scenarios.yaml"),
        "sha256": None,
        "artifact_status": "missing",
    }

    with pytest.raises(ProvenanceArtifactError, match="inputs.scenario_matrix is missing"):
        validate_result_provenance_manifest(manifest)


def test_validator_fails_on_contradictory_input_status_fields(tmp_path: Path) -> None:
    """Status, path, and digest must agree internally."""
    manifest = _complete_manifest(tmp_path)
    manifest["inputs"]["algo_config"] = {
        "path": str(tmp_path / "algo.yaml"),
        "sha256": sha256(b"algo: goal\n").hexdigest(),
        "artifact_status": "not_provided",
    }

    with pytest.raises(ProvenanceRequiredFieldError, match="inputs.algo_config.path"):
        validate_result_provenance_manifest(manifest)


def test_validator_fails_when_referenced_input_file_bytes_change(tmp_path: Path) -> None:
    """Local validation recomputes referenced input hashes and catches drift."""
    manifest = _complete_manifest(tmp_path)
    schema_path = Path(manifest["inputs"]["schema_path"]["path"])
    schema_path.write_text('{"type":"array"}\n', encoding="utf-8")

    with pytest.raises(ProvenanceRequiredFieldError, match="schema_path.sha256"):
        validate_result_provenance_manifest(manifest)


def test_validator_accepts_optional_algo_config_not_provided(tmp_path: Path) -> None:
    """Algorithm config absence is valid only through the explicit optional shape."""
    manifest = _complete_manifest(tmp_path, algo_config=False)

    assert manifest["inputs"]["algo_config"] == {
        "path": None,
        "sha256": None,
        "artifact_status": "not_provided",
    }
    validate_result_provenance_manifest(manifest)


def test_validator_accepts_inline_generated_scenario_input(tmp_path: Path) -> None:
    """Inline/generated scenario inputs use the documented not_applicable shape."""
    manifest = _complete_manifest(tmp_path, scenario_file=False)

    assert manifest["inputs"]["scenario_matrix"] == {
        "path": None,
        "sha256": None,
        "artifact_status": "not_applicable",
        "reason": "inline_or_generated_scenarios",
    }
    validate_result_provenance_manifest(manifest)


def test_input_bundle_sha256_ignores_path_aliases_to_identical_bytes(tmp_path: Path) -> None:
    """The input bundle identity is byte-bound, not path-string-bound."""
    schema_path, scenario_path, algo_config_path = _write_input_files(tmp_path)
    alias_schema = tmp_path / "schema-alias.json"
    alias_scenario = tmp_path / "scenarios-alias.yaml"
    alias_algo = tmp_path / "algo-alias.yaml"
    alias_schema.write_bytes(schema_path.read_bytes())
    alias_scenario.write_bytes(scenario_path.read_bytes())
    alias_algo.write_bytes(algo_config_path.read_bytes())

    jsonl_path = tmp_path / "episodes.jsonl"
    jsonl_path.write_text(
        '{"episode_id":"test--0","scenario_id":"test","seed":0}\n',
        encoding="utf-8",
    )
    kwargs = {
        "out_path": jsonl_path,
        "episode_records": [
            {
                "episode_id": "test--0",
                "scenario_id": "test",
                "seed": 0,
                "config_hash": "abc",
                "git_hash": "def",
            }
        ],
        "scenarios": [{"name": "test"}],
        "algo": "goal",
        "benchmark_profile": "baseline-safe",
        "suite_key": "test_suite",
        "total_jobs": 1,
        "written": 1,
        "horizon": 100,
        "dt": 0.1,
        "record_forces": True,
        "active_observation_mode": "lidar",
        "active_observation_level": "full",
    }
    original = build_result_provenance_manifest(
        schema_path=schema_path,
        scenario_path=scenario_path,
        algo_config_path=algo_config_path,
        **kwargs,
    )
    alias = build_result_provenance_manifest(
        schema_path=alias_schema,
        scenario_path=alias_scenario,
        algo_config_path=alias_algo,
        **kwargs,
    )

    assert (
        original["campaign_identity"]["input_bundle_sha256"]
        == alias["campaign_identity"]["input_bundle_sha256"]
    )
    assert original["campaign_identity"]["config_hash"] != alias["campaign_identity"]["config_hash"]


def test_input_bundle_sha256_changes_when_input_bytes_change(tmp_path: Path) -> None:
    """Different input bytes cannot retain the same full input-bundle digest."""
    first = _complete_manifest(tmp_path / "first")
    second = _complete_manifest(tmp_path / "second")
    second_schema = Path(second["inputs"]["schema_path"]["path"])
    second_schema.write_text('{"type":"array"}\n', encoding="utf-8")
    second["inputs"]["schema_path"]["sha256"] = sha256(second_schema.read_bytes()).hexdigest()
    second["campaign_identity"]["input_bundle_sha256"] = build_result_provenance_manifest(
        out_path=Path(second["raw_artifacts"][0]["path"]),
        episode_records=[
            {
                "episode_id": "test--0",
                "scenario_id": "test",
                "seed": 0,
                "config_hash": "abc",
                "git_hash": "def",
            }
        ],
        schema_path=second_schema,
        scenario_path=Path(second["inputs"]["scenario_matrix"]["path"]),
        scenarios=[{"name": "test"}],
        algo="goal",
        algo_config_path=Path(second["inputs"]["algo_config"]["path"]),
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=1,
        written=1,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )["campaign_identity"]["input_bundle_sha256"]

    assert (
        first["campaign_identity"]["input_bundle_sha256"]
        != second["campaign_identity"]["input_bundle_sha256"]
    )


def test_validator_rejects_unmarked_strengthened_manifest(tmp_path: Path) -> None:
    """Strengthened fields require the explicit additive migration marker."""
    manifest = _complete_manifest(tmp_path)
    manifest.pop("input_binding_schema_version")

    with pytest.raises(ProvenanceValidationError, match="cannot claim strengthened"):
        validate_result_provenance_manifest(manifest)


def test_validator_accepts_unmarked_historical_v1_manifest(tmp_path: Path) -> None:
    """Historical v1 manifests remain legacy-only, not strengthened evidence."""
    manifest = _complete_manifest(tmp_path)
    manifest.pop("input_binding_schema_version")
    manifest["campaign_identity"].pop("input_bundle_sha256")
    manifest["campaign_identity"].pop("algorithm")

    validate_result_provenance_manifest(manifest)


def test_validator_rejects_unknown_input_binding_version(tmp_path: Path) -> None:
    """Unknown additive contract versions fail closed."""
    manifest = _complete_manifest(tmp_path)
    manifest["input_binding_schema_version"] = "benchmark_result_provenance.input_binding.v3"

    with pytest.raises(ProvenanceRequiredFieldError, match="input_binding_schema_version"):
        validate_result_provenance_manifest(manifest)


def test_validator_fails_on_incomplete_manifest(tmp_path: Path) -> None:
    """A manifest with missing required fields should raise."""
    manifest = _complete_manifest(tmp_path)

    # Corrupt the manifest by deleting a required row field.
    manifest["run"]["repo_commit"] = ""

    with pytest.raises(ProvenanceRequiredFieldError, match="run.repo_commit"):
        validate_result_provenance_manifest(manifest)


def test_validator_fails_closed_on_non_dict_row(tmp_path: Path) -> None:
    """Malformed row entries fail with a clean provenance validation error."""
    manifest = _complete_manifest(tmp_path)
    manifest["rows"] = ["not-a-dict"]

    with pytest.raises(ProvenanceRequiredFieldError, match=r"rows\[0\] must be a dict"):
        validate_result_provenance_manifest(manifest)


def test_validator_fails_on_missing_scenario_matrix_hash(tmp_path: Path) -> None:
    """A manifest without campaign_identity.scenario_matrix_hash should fail."""
    manifest = _complete_manifest(tmp_path)

    del manifest["campaign_identity"]["scenario_matrix_hash"]

    with pytest.raises(
        ProvenanceRequiredFieldError, match="campaign_identity.scenario_matrix_hash"
    ):
        validate_result_provenance_manifest(manifest)


def test_validator_fails_on_missing_episodes_jsonl_artifact(tmp_path: Path) -> None:
    """A manifest without an episodes_jsonl artifact entry should fail validation."""
    manifest = _complete_manifest(tmp_path)

    manifest["raw_artifacts"] = []

    with pytest.raises(ProvenanceRequiredFieldError, match="episodes_jsonl"):
        validate_result_provenance_manifest(manifest)


def test_write_and_reload_roundtrip(tmp_path: Path) -> None:
    """Writing a manifest and reloading it should preserve all fields."""
    jsonl_path = tmp_path / "episodes.jsonl"
    jsonl_path.write_text(
        '{"episode_id":"a--1","scenario_id":"a","seed":1}\n',
        encoding="utf-8",
    )
    manifest = build_result_provenance_manifest(
        out_path=jsonl_path,
        episode_records=[
            {
                "episode_id": "a--1",
                "scenario_id": "a",
                "seed": 1,
                "config_hash": "abc",
                "git_hash": "def",
            },
        ],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[{"name": "a"}],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test",
        total_jobs=1,
        written=1,
        horizon=100,
        dt=0.1,
        record_forces=False,
        active_observation_mode="lidar",
        active_observation_level="full",
    )

    manifest_path = manifest_path_for_result_jsonl(jsonl_path)
    write_result_provenance_manifest(manifest_path, manifest)
    assert manifest_path.exists()

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION
    assert len(raw["rows"]) == 1
    assert raw["rows"][0]["episode_id"] == "a--1"
    assert raw["rows"][0]["simulator_settings"]["record_forces"] is False


def test_build_manifest_uses_git_hash_from_records(tmp_path: Path) -> None:
    """Row repo_commit should use the record's git_hash when present."""
    jsonl_path = tmp_path / "episodes.jsonl"
    jsonl_path.write_text(
        '{"episode_id":"x--0","scenario_id":"x","seed":0}\n',
        encoding="utf-8",
    )
    manifest = build_result_provenance_manifest(
        out_path=jsonl_path,
        episode_records=[
            {
                "episode_id": "x--0",
                "scenario_id": "x",
                "seed": 0,
                "config_hash": "abc",
                "git_hash": "from_record",
            },
        ],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[{"name": "x"}],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test",
        total_jobs=1,
        written=1,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )
    assert manifest["rows"][0]["repo_commit"] == "from_record"


def test_skipped_manifest_has_not_applicable_completeness() -> None:
    """A manifest for a skipped (preflight) run should have appropriate completeness."""
    manifest = build_result_provenance_manifest(
        out_path=Path("episodes.jsonl"),
        episode_records=[],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=0,
        written=0,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )
    assert manifest["completeness"]["status"] == "not_applicable"
    assert manifest["completeness"]["reason"] == "preflight_skipped"


def test_build_manifest_marks_partial_batch_incomplete(tmp_path: Path) -> None:
    """A partially written batch must not be reported as complete provenance."""
    jsonl_path = tmp_path / "episodes.jsonl"
    jsonl_path.write_text(
        '{"episode_id":"test--0","scenario_id":"test","seed":0}\n',
        encoding="utf-8",
    )

    manifest = build_result_provenance_manifest(
        out_path=jsonl_path,
        episode_records=[
            {
                "episode_id": "test--0",
                "scenario_id": "test",
                "seed": 0,
                "config_hash": "abc",
                "git_hash": "def",
            },
        ],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[{"name": "test"}],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=2,
        written=1,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )

    assert manifest["completeness"]["status"] == "partial"
    assert manifest["completeness"]["reason"] == "partial_batch_failure"


def test_cli_checker_accepts_valid_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The validation CLI exits 0 for a known-good manifest."""
    manifest = _complete_manifest(tmp_path)
    manifest_path = tmp_path / "episodes.jsonl.provenance.json"
    write_result_provenance_manifest(manifest_path, manifest)

    monkeypatch.setattr(
        "sys.argv",
        ["check_benchmark_result_provenance.py", "--manifest", str(manifest_path)],
    )

    with pytest.raises(SystemExit) as exc_info:
        check_benchmark_result_provenance.main()

    assert exc_info.value.code == 0
    assert "OK:" in capsys.readouterr().err


def test_cli_checker_fails_closed_on_invalid_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The validation CLI exits 2 when required provenance fields are missing."""
    manifest = _complete_manifest(tmp_path)
    manifest["run"]["repo_commit"] = ""
    manifest_path = tmp_path / "invalid.provenance.json"
    write_result_provenance_manifest(manifest_path, manifest)

    monkeypatch.setattr(
        "sys.argv",
        ["check_benchmark_result_provenance.py", "--manifest", str(manifest_path)],
    )

    with pytest.raises(SystemExit) as exc_info:
        check_benchmark_result_provenance.main()

    assert exc_info.value.code == 2
    assert "FAIL:" in capsys.readouterr().err
