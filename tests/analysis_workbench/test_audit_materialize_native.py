"""A real canonical episode can supply retained state for a missing video."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from typing import TYPE_CHECKING

import pytest

from robot_sf.analysis_workbench import audit_native_diagnostic as native
from robot_sf.analysis_workbench.audit_materialize import (
    DERIVED_RENDER,
    FIDELITY_UNVERIFIABLE,
    materialize_episode,
)
from robot_sf.benchmark.analysis_trace import trace_artifact_sha256
from robot_sf.benchmark.runner import run_episode

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def admitted_native_case(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Reuse the real admitted source fixture for exact-input proof."""

    from tests.analysis_workbench import test_audit_native_diagnostic as native_tests

    return native_tests.native_case.__wrapped__(tmp_path_factory)


@pytest.fixture(scope="module")
def native_episode() -> dict:
    """Retain one true canonical runner record for positive and negative checks."""

    original = run_episode(
        {"id": "ba05_native_probe", "density": "low", "flow": "uni", "obstacle": "open"},
        42,
        horizon=8,
        dt=0.1,
        robot_start=(-4.0, 0.0),
        robot_goal=(4.0, 0.0),
        record_forces=False,
        algo="simple_policy",
        telemetry={"analysis_trace": "all"},
        provenance={"test": "ba05-native-materialization"},
    )
    assert original["algorithm_metadata"]["planner_kinematics"]["execution_mode"] == "native"
    return original


def _selected_episode(original: dict) -> dict:
    return {
        "episode_id": original["episode_id"],
        "seed": original["seed"],
        "scenario_id": "ba05_native_probe",
        "algo": "simple_policy",
        "source_commit": original["git_hash"],
        "config_identity": original["algorithm_metadata"]["analysis_trace"]["config_digest"],
        "config_digest": original["algorithm_metadata"]["analysis_trace"]["config_digest"],
        "retained_trace": original["algorithm_metadata"]["analysis_trace"],
    }


def test_native_missing_video_renders_retained_trace_without_simulation(
    tmp_path: Path, native_episode: dict
) -> None:
    """Keep a newly rendered view distinct from the native source execution."""

    result = materialize_episode(
        _selected_episode(native_episode),
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
        output_directory="native-retained-render",
    )

    assert result.status == "complete", result.diagnostics
    assert result.materialization_kind == DERIVED_RENDER
    assert result.fidelity == FIDELITY_UNVERIFIABLE
    assert result.simulation_executed is False
    assert result.provenance["simulation_advanced"] is False
    assert "native_retained_trace_projected" in result.diagnostics
    trace_digest = native_episode["algorithm_metadata"]["analysis_trace"]["artifact_sha256"]
    assert result.source_digest == trace_digest
    assert result.provenance["native_trace_digest"] == trace_digest
    assert result.provenance["retained_state_digest"] != trace_digest
    manifest = json.loads(
        (
            tmp_path / "derived" / "native-retained-render" / "audit-materialization.v1.json"
        ).read_text()
    )
    assert manifest["source_digest"] == trace_digest
    assert (tmp_path / "derived" / "native-retained-render" / "trajectory.png").is_file()


def test_retained_trace_accepts_distinct_source_config_identity_and_trace_digest(
    tmp_path: Path, native_episode: dict
) -> None:
    """The campaign's config identity and resolved trace digest are separate claims."""

    selected = _selected_episode(native_episode)
    selected["config_identity"] = "native-config-sha256:distinct-source-identity"

    result = materialize_episode(
        selected,
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
    )

    assert result.status == "complete", result.to_dict()
    assert selected["config_digest"] == selected["retained_trace"]["config_digest"]


def test_exact_input_regeneration_renders_trace_removed_from_selected_row(
    tmp_path: Path, admitted_native_case: dict
) -> None:
    """A real admitted simple-policy source can repair a state-free row."""

    from tests.analysis_workbench import test_audit_native_diagnostic as native_tests

    case = admitted_native_case
    selected = {
        "episode_id": case["original"]["episode_id"],
        "scenario_id": case["original"]["scenario_id"],
        "seed": case["original"]["seed"],
        "planner_id": "simple_policy",
        "source_commit": case["original"]["git_hash"],
        "config_identity": case["source_document"]["identity"]["config_identity"],
    }
    result = native.materialize_exact_input(
        native.NativeDiagnosticRequest.from_mapping(native_tests._request(case)),
        selected,
        output_root=tmp_path / "derived",
        output_directory="exact-input",
        execution_id="ba05-exact-materialization-1",
    )

    assert result.status == "complete", result.to_dict()
    assert result.materialization_kind == DERIVED_RENDER
    assert result.fidelity == "verified"
    assert result.simulation_executed is True
    assert result.provenance["execution_id"] == "ba05-exact-materialization-1"
    assert result.provenance["historical_episode_id"] == selected["episode_id"]
    assert result.provenance["original_telemetry_fabricated"] is False
    assert result.provenance["simulation_advanced"] is True
    assert (
        result.provenance["source_commit"] == case["source_document"]["identity"]["source_commit"]
    )
    assert (
        result.provenance["config_identity"]
        == case["source_document"]["identity"]["config_identity"]
    )
    assert (
        result.provenance["initial_state_sha256"]
        == case["source_document"]["identity"]["initial_state_sha256"]
    )
    assert (
        result.provenance["environment_identity"]
        == case["source_document"]["identity"]["environment_identity"]
    )
    assert result.provenance["source_admission"]["source_sha256"]
    assert result.provenance["input_digests"]["runner_input_sha256"]
    assert (tmp_path / "derived" / "exact-input" / "trajectory.png").is_file()
    manifest = json.loads(
        (tmp_path / "derived" / "exact-input" / "audit-materialization.v1.json").read_text()
    )
    assert manifest["fidelity"] == "verified"
    assert manifest["provenance"]["execution_id"] == "ba05-exact-materialization-1"


def _exact_selected(case: dict) -> dict:
    """Return a state-free selected row bound only to admitted identities."""

    original = case["original"]
    return {
        "episode_id": original["episode_id"],
        "scenario_id": original["scenario_id"],
        "seed": original["seed"],
        "planner_id": "simple_policy",
        "source_commit": original["git_hash"],
        "config_identity": case["source_document"]["identity"]["config_identity"],
    }


def _exact_request(case: dict) -> native.NativeDiagnosticRequest:
    """Parse the launcher-owned native request fixture."""

    from tests.analysis_workbench import test_audit_native_diagnostic as native_tests

    return native.NativeDiagnosticRequest.from_mapping(native_tests._request(case))


def test_exact_input_source_mutation_is_unavailable(
    tmp_path: Path, admitted_native_case: dict
) -> None:
    """Changed admitted bytes cannot reach the exact-input child."""

    source_path = admitted_native_case["root"] / "original.json"
    original_bytes = source_path.read_bytes()
    source_path.write_bytes(original_bytes + b" ")
    try:
        result = native.materialize_exact_input(
            _exact_request(admitted_native_case),
            _exact_selected(admitted_native_case),
            output_root=tmp_path / "derived",
            execution_id="ba05-exact-source-mutation",
        )
    finally:
        source_path.write_bytes(original_bytes)

    assert result.status == "unavailable"
    assert "source_mutated" in result.reason
    assert result.simulation_executed is False


def test_exact_input_stateful_source_is_rejected_before_runner(
    tmp_path: Path, admitted_native_case: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checkpoint-bearing source inputs never enter exact regeneration."""

    from tests.analysis_workbench import test_audit_native_diagnostic as native_tests

    case = native_tests._make_case(
        admitted_native_case["root"].parent / "exact-stateful",
        admitted_native_case["original"],
        runner_input_overrides={"algo_config_path": "checkpoint.pt"},
    )
    started = False

    def child_must_not_start(*_args: object, **_kwargs: object) -> None:
        nonlocal started
        started = True
        raise AssertionError("stateful exact-input child started")

    monkeypatch.setattr(native, "_run_bounded", child_must_not_start)
    result = native.materialize_exact_input(
        _exact_request(case),
        _exact_selected(case),
        output_root=tmp_path / "derived",
        execution_id="ba05-exact-stateful",
    )

    assert result.status == "failed"
    assert "stateful/checkpoint" in result.reason
    assert result.simulation_executed is False
    assert started is False


def test_exact_input_identity_mismatch_stops_before_runner(
    tmp_path: Path, admitted_native_case: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit selected identity mismatch cannot authorize execution."""

    started = False

    def child_must_not_start(*_args: object, **_kwargs: object) -> None:
        nonlocal started
        started = True
        raise AssertionError("exact-input child started after identity mismatch")

    monkeypatch.setattr(native, "_run_bounded", child_must_not_start)
    selected = _exact_selected(admitted_native_case)
    selected["initial_state_digest"] = "0" * 64
    result = native.materialize_exact_input(
        _exact_request(admitted_native_case),
        selected,
        output_root=tmp_path / "derived",
        execution_id="ba05-exact-identity-mismatch",
    )

    assert result.status == "unavailable"
    assert "identity_mismatch" in result.reason
    assert started is False
    assert result.simulation_executed is False


@pytest.mark.parametrize(
    ("claim", "value"),
    [
        ("scenario_params", {"id": "different-scenario"}),
        ("planner_config", {"checkpoint_path": "checkpoint.pt"}),
        ("initial_state", {"robot_start": [0.0, 0.0], "robot_goal": [1.0, 1.0]}),
        ("checkpoint_digest", "f" * 64),
        ("execution_id", "different-execution"),
        ("stateful", True),
    ],
)
def test_exact_input_unsupported_selected_claims_stop_before_runner(
    tmp_path: Path,
    admitted_native_case: dict,
    monkeypatch: pytest.MonkeyPatch,
    claim: str,
    value: object,
) -> None:
    """State/planner claims on a row cannot be silently ignored."""

    started = False

    def child_must_not_start(*_args: object, **_kwargs: object) -> None:
        nonlocal started
        started = True
        raise AssertionError(f"exact-input child started for {claim}")

    monkeypatch.setattr(native, "_run_bounded", child_must_not_start)
    selected = _exact_selected(admitted_native_case)
    selected[claim] = value
    result = native.materialize_exact_input(
        _exact_request(admitted_native_case),
        selected,
        output_root=tmp_path / "derived",
        execution_id=f"ba05-exact-claim-{claim}",
    )

    assert result.status == "unavailable", result.to_dict()
    assert claim in " ".join(result.diagnostics)
    assert result.simulation_executed is False
    assert started is False


@pytest.mark.parametrize(
    "identity_key",
    ["source_commit", "config_identity", "initial_state_sha256", "environment_identity"],
)
def test_exact_input_recipe_identity_mismatch_stops_before_runner(
    tmp_path: Path,
    admitted_native_case: dict,
    monkeypatch: pytest.MonkeyPatch,
    identity_key: str,
) -> None:
    """Every recipe/source identity field remains bound before execution."""

    from tests.analysis_workbench import test_audit_native_diagnostic as native_tests

    request = native_tests._request(admitted_native_case)
    if identity_key == "source_commit":
        mutated: object = "0" * 40
    elif identity_key == "config_identity":
        mutated = "simple_policy-config.v1:" + ("0" * 64)
    elif identity_key == "initial_state_sha256":
        mutated = "f" * 64
    else:
        mutated = {"scenario_id": "different-environment", "scenario_digest": "0" * 64}
    request["recipe"]["source_identity"][identity_key] = mutated
    receipt_path = admitted_native_case["root"] / "receipt.json"
    original_receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    recipe = native_tests.experiment_recipe_from_dict(request["recipe"], source="test mutation")
    receipt["recipe_sha256"] = native_tests.experiment_recipe_canonical_digest(recipe)
    receipt_bytes = native_tests._write_json(receipt_path, receipt)
    request["admission"]["receipt_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()

    started = False

    def child_must_not_start(*_args: object, **_kwargs: object) -> None:
        nonlocal started
        started = True
        raise AssertionError(f"recipe identity child started for {identity_key}")

    monkeypatch.setattr(native, "_run_bounded", child_must_not_start)
    try:
        result = native.materialize_exact_input(
            request,
            _exact_selected(admitted_native_case),
            output_root=tmp_path / "derived",
            execution_id=f"ba05-exact-recipe-{identity_key}",
        )
    finally:
        receipt_path.write_bytes(original_receipt_bytes)

    assert result.status in {"failed", "unavailable"}
    assert started is False


def test_exact_input_cancellation_before_admission_does_not_run(
    tmp_path: Path, admitted_native_case: dict
) -> None:
    """Cancellation before admission leaves the source and runner untouched."""

    cancel_event = threading.Event()
    cancel_event.set()
    result = native.materialize_exact_input(
        _exact_request(admitted_native_case),
        _exact_selected(admitted_native_case),
        output_root=tmp_path / "derived",
        execution_id="ba05-exact-cancelled",
        cancel_event=cancel_event,
    )

    assert result.status == "cancelled"
    assert result.simulation_executed is False
    assert "before exact-input admission" in result.reason


def test_exact_input_cancellation_after_child_is_distinct(
    tmp_path: Path, admitted_native_case: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child cancellation is retained as a diagnostic execution outcome."""

    def cancelled(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "status": native.STATUS_CANCELLED,
            "reason": "cancelled_by_user: owned child terminated",
            "elapsed_s": 0.0,
        }

    monkeypatch.setattr(native, "_run_bounded", cancelled)
    result = native.materialize_exact_input(
        _exact_request(admitted_native_case),
        _exact_selected(admitted_native_case),
        output_root=tmp_path / "derived",
        execution_id="ba05-exact-cancelled-child",
    )

    assert result.status == "cancelled"
    assert result.simulation_executed is True
    assert "owned child terminated" in result.reason


def test_exact_input_divergence_remains_a_derived_diagnostic(
    tmp_path: Path, admitted_native_case: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A generated outcome mismatch is rendered but never relabeled verified."""

    from tests.analysis_workbench import test_audit_native_diagnostic as native_tests

    generated = native_tests._diagnostic_record(
        admitted_native_case, native.ROLE_EXACT_INPUT_REGENERATION
    )
    generated["outcome"] = {"route_complete": True, "collision": True}
    monkeypatch.setattr(
        native,
        "_run_bounded",
        lambda *_args, **_kwargs: {"status": "ok", "record": generated, "elapsed_s": 0.0},
    )

    result = native.materialize_exact_input(
        _exact_request(admitted_native_case),
        _exact_selected(admitted_native_case),
        output_root=tmp_path / "derived",
        output_directory="diverged",
        execution_id="ba05-exact-diverged",
    )

    assert result.status == "complete", result.to_dict()
    assert result.fidelity == "diverged"
    assert result.materialization_kind == DERIVED_RENDER
    assert result.provenance["scientific_claim_allowed"] is False


def test_exact_input_missing_historical_telemetry_is_unverifiable(
    tmp_path: Path, admitted_native_case: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing original trace does not get copied into the new artifact."""

    from tests.analysis_workbench import test_audit_native_diagnostic as native_tests

    historical = copy.deepcopy(admitted_native_case["original"])
    historical["algorithm_metadata"].pop("analysis_trace")
    case = native_tests._make_case(
        admitted_native_case["root"].parent / "missing-exact-telemetry", historical
    )
    generated = native_tests._diagnostic_record(
        admitted_native_case, native.ROLE_EXACT_INPUT_REGENERATION
    )
    generated["provenance"]["native_diagnostic"].update(
        {
            "source_id": case["source_document"]["source_id"],
            "source_commit": case["source_document"]["identity"]["source_commit"],
            "config_identity": case["source_document"]["identity"]["config_identity"],
            "request_id": case["request_document"]["request_id"],
        }
    )
    monkeypatch.setattr(
        native,
        "_run_bounded",
        lambda *_args, **_kwargs: {"status": "ok", "record": generated, "elapsed_s": 0.0},
    )

    result = native.materialize_exact_input(
        _exact_request(case),
        _exact_selected(case),
        output_root=tmp_path / "derived",
        output_directory="unverifiable",
        execution_id="ba05-exact-unverifiable",
    )

    assert result.status == "complete", result.to_dict()
    assert result.fidelity == "unverifiable"
    assert "historical_telemetry_unavailable" in result.diagnostics
    assert result.provenance["original_telemetry_fabricated"] is False


@pytest.mark.parametrize(
    "mutation",
    ["digest", "scenario", "config", "planner", "commit", "episode", "seed", "execution"],
)
def test_native_trace_identity_or_digest_change_fails_closed(
    tmp_path: Path, native_episode: dict, mutation: str
) -> None:
    """A stale selected identity cannot produce a derived artifact."""

    selected = copy.deepcopy(_selected_episode(native_episode))
    if mutation == "digest":
        selected["retained_trace"]["steps"][0]["robot"]["position"][0] += 1.0
    elif mutation == "scenario":
        selected["scenario_id"] = "different-scenario"
    elif mutation == "config":
        selected["config_digest"] = "0" * 64
    elif mutation == "planner":
        selected["algo"] = "different_planner"
    elif mutation == "episode":
        selected["episode_id"] = "ba05_native_probe--43"
    elif mutation == "seed":
        selected["seed"] = 43
    elif mutation == "execution":
        selected["execution_id"] = "different-execution"
    else:
        selected["source_commit"] = "0" * 40

    result = materialize_episode(
        selected, source_root=tmp_path / "source", output_root=tmp_path / "derived"
    )

    assert result.status == "unavailable"
    assert result.reason == "native_retained_trace_unavailable"
    assert not (tmp_path / "derived").exists()


@pytest.mark.parametrize("missing", ["episode_id", "seed"])
def test_older_native_trace_without_episode_binding_fails_closed(
    tmp_path: Path, native_episode: dict, missing: str
) -> None:
    """Digest-valid older envelopes cannot claim an unrelated selected run."""

    selected = copy.deepcopy(_selected_episode(native_episode))
    trace = selected["retained_trace"]
    trace.pop(missing)
    trace["artifact_sha256"] = trace_artifact_sha256(trace)

    result = materialize_episode(
        selected, source_root=tmp_path / "source", output_root=tmp_path / "derived"
    )

    assert result.status == "unavailable"
    assert result.reason == "native_retained_trace_unavailable"
    assert not (tmp_path / "derived").exists()


@pytest.mark.parametrize(
    "claimed_identity",
    [
        "campaign_id",
        "source_digest",
        "checkpoint_digest",
        "environment_digest",
        "initial_state_digest",
    ],
)
def test_native_trace_does_not_ignore_claimed_selected_identity(
    tmp_path: Path, native_episode: dict, claimed_identity: str
) -> None:
    """Missing optional trace identities cannot be silently treated as matches."""

    selected = _selected_episode(native_episode)
    selected[claimed_identity] = "different-identity"

    result = materialize_episode(
        selected, source_root=tmp_path / "source", output_root=tmp_path / "derived"
    )

    assert result.status == "unavailable"
    assert result.reason == "native_retained_trace_unavailable"
    assert not (tmp_path / "derived").exists()


def test_path_retained_native_trace_has_file_and_trace_digests(
    tmp_path: Path, native_episode: dict
) -> None:
    """A path-backed native trace remains a derived view, not original media."""

    selected = _selected_episode(native_episode)
    trace = selected.pop("retained_trace")
    source_root = tmp_path / "source"
    source_root.mkdir()
    content = json.dumps(trace, sort_keys=True).encode()
    (source_root / "trace.json").write_bytes(content)
    file_digest = hashlib.sha256(content).hexdigest()
    selected["retained_trace_source"] = {"uri": "trace.json", "sha256": file_digest}

    result = materialize_episode(
        selected, source_root=source_root, output_root=tmp_path / "derived"
    )

    assert result.status == "complete", result.diagnostics
    assert result.materialization_kind == DERIVED_RENDER
    assert result.source_digest == trace["artifact_sha256"]
    assert result.provenance["native_trace_digest"] == trace["artifact_sha256"]
    assert result.provenance["retained_source_file_digest"] == file_digest


def test_native_trace_matching_execution_id_is_bound(tmp_path: Path, native_episode: dict) -> None:
    """A claimed execution ID must be present in the digest-bound trace."""

    selected = copy.deepcopy(_selected_episode(native_episode))
    selected["execution_id"] = "probe-execution"
    trace = selected["retained_trace"]
    trace["execution_id"] = "probe-execution"
    trace["artifact_sha256"] = trace_artifact_sha256(trace)

    result = materialize_episode(
        selected, source_root=tmp_path / "source", output_root=tmp_path / "derived"
    )

    assert result.status == "complete", result.diagnostics
    assert result.materialization_kind == DERIVED_RENDER


def test_path_retained_native_trace_wrong_selected_seed_fails_closed(
    tmp_path: Path, native_episode: dict
) -> None:
    """A verified file digest is not sufficient to change the selected episode."""

    selected = _selected_episode(native_episode)
    trace = selected.pop("retained_trace")
    selected["episode_id"] = "ba05_native_probe--43"
    selected["seed"] = 43
    source_root = tmp_path / "source"
    source_root.mkdir()
    content = json.dumps(trace).encode()
    (source_root / "trace.json").write_bytes(content)
    selected["retained_trace_source"] = {
        "uri": "trace.json",
        "sha256": hashlib.sha256(content).hexdigest(),
    }

    result = materialize_episode(
        selected, source_root=source_root, output_root=tmp_path / "derived"
    )

    assert result.status == "unavailable"
    assert result.materialization_kind != DERIVED_RENDER
    assert not (tmp_path / "derived").exists()


def test_misdeclared_native_json_cannot_be_historical_original(
    tmp_path: Path, native_episode: dict
) -> None:
    """A digest-verified analysis trace is never an original recording."""

    selected = _selected_episode(native_episode)
    selected.pop("retained_trace")
    source_root = tmp_path / "source"
    source_root.mkdir()
    content = json.dumps(native_episode["algorithm_metadata"]["analysis_trace"]).encode()
    (source_root / "trace.json").write_bytes(content)
    selected["recording"] = {
        "uri": "trace.json",
        "sha256": hashlib.sha256(content).hexdigest(),
        "scenario_id": selected["scenario_id"],
        "source_commit": selected["source_commit"],
        "config_identity": selected["config_identity"],
    }

    result = materialize_episode(
        selected, source_root=source_root, output_root=tmp_path / "derived"
    )

    assert result.status == "unavailable"
    assert result.materialization_kind != "historical_original"
    assert "original_recording_format_unsupported" in result.diagnostics


@pytest.mark.parametrize(
    ("algo", "planner_id"),
    [("simple_policy", "foreign"), ("foreign", "simple_policy"), (None, "foreign")],
)
def test_conflicting_native_planner_aliases_fail_closed(
    tmp_path: Path, native_episode: dict, algo: str | None, planner_id: str
) -> None:
    """Every selected planner claim must match the retained native trace."""

    selected = _selected_episode(native_episode)
    selected["algo"] = algo
    selected["planner_id"] = planner_id

    result = materialize_episode(
        selected, source_root=tmp_path / "source", output_root=tmp_path / "derived"
    )

    assert result.status == "unavailable"
    assert "native_retained_trace_unavailable" == result.reason
    assert not (tmp_path / "derived").exists()


def test_misdeclared_runner_record_json_cannot_be_historical_original(
    tmp_path: Path, native_episode: dict
) -> None:
    """A runner JSON wrapper around a native trace is not historical video."""

    selected = _selected_episode(native_episode)
    selected.pop("retained_trace")
    source_root = tmp_path / "source"
    source_root.mkdir()
    content = json.dumps(native_episode).encode()
    (source_root / "record.json").write_bytes(content)
    selected["recording"] = {
        "uri": "record.json",
        "format": "video/mp4",
        "sha256": hashlib.sha256(content).hexdigest(),
        "scenario_id": selected["scenario_id"],
        "source_commit": selected["source_commit"],
        "config_identity": selected["config_identity"],
    }

    result = materialize_episode(
        selected, source_root=source_root, output_root=tmp_path / "derived"
    )

    assert result.status == "unavailable"
    assert result.materialization_kind != "historical_original"
    assert "original_recording_media_type_mismatch" in result.diagnostics


@pytest.mark.parametrize(
    "shape",
    [
        "duplicate_direct",
        "duplicate_wrapper",
        "deep_wrapper",
        "bom_direct",
        "bom_wrapper",
        "truncated_utf16le",
        "truncated_utf16be",
        "truncated_utf32le",
        "truncated_utf32be",
        "truncated_utf16le_nobom",
        "truncated_utf16be_nobom",
        "truncated_utf32le_nobom",
        "truncated_utf32be_nobom",
    ],
)
def test_uninspectable_native_json_cannot_be_historical_original(
    tmp_path: Path, native_episode: dict, shape: str
) -> None:
    """Duplicate or deep JSON cannot hide a native trace behind a media label."""

    selected = _selected_episode(native_episode)
    selected.pop("retained_trace")
    source_root = tmp_path / "source"
    source_root.mkdir()
    if shape == "duplicate_direct":
        content = json.dumps(native_episode["algorithm_metadata"]["analysis_trace"]).encode()
        content = content[:-1] + b',"schema_version":"opaque.v1"}'
    elif shape == "duplicate_wrapper":
        content = json.dumps(native_episode).encode()
        content = content[:-1] + b',"algorithm_metadata":{}}'
    elif shape == "bom_direct":
        content = (
            b"\xef\xbb\xbf"
            + json.dumps(native_episode["algorithm_metadata"]["analysis_trace"]).encode()
        )
    elif shape == "bom_wrapper":
        content = b"\xef\xbb\xbf" + json.dumps(native_episode).encode()
    elif shape.startswith("truncated_"):
        encoding = shape.removeprefix("truncated_")
        with_bom = not encoding.endswith("_nobom")
        encoding = encoding.removesuffix("_nobom")
        marker = {
            "utf16le": b"\xff\xfe",
            "utf16be": b"\xfe\xff",
            "utf32le": b"\xff\xfe\x00\x00",
            "utf32be": b"\x00\x00\xfe\xff",
        }[encoding]
        codec = {
            "utf16le": "utf-16-le",
            "utf16be": "utf-16-be",
            "utf32le": "utf-32-le",
            "utf32be": "utf-32-be",
        }[encoding]
        content = (marker if with_bom else b"") + json.dumps(
            native_episode["algorithm_metadata"]["analysis_trace"]
        )[:-1].encode(codec)
    else:
        nested = b"[" * 10_000 + b"0" + b"]" * 10_000
        metadata = json.dumps(native_episode["algorithm_metadata"]).encode()
        content = b'{"padding":' + nested + b',"algorithm_metadata":' + metadata + b"}"
    (source_root / "record.json").write_bytes(content)
    selected["recording"] = {
        "uri": "record.json",
        "format": "video/mp4",
        "sha256": hashlib.sha256(content).hexdigest(),
        "scenario_id": selected["scenario_id"],
        "source_commit": selected["source_commit"],
        "config_identity": selected["config_identity"],
    }

    result = materialize_episode(
        selected, source_root=source_root, output_root=tmp_path / "derived"
    )

    assert result.status == "unavailable"
    assert result.materialization_kind != "historical_original"
    assert "original_recording_media_type_mismatch" in result.diagnostics
