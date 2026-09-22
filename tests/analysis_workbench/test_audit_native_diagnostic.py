"""Focused proof for the bounded BA-05 native diagnostic adapter."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from typing import TYPE_CHECKING, Any

import pytest

from robot_sf.analysis_workbench import audit_native_diagnostic as native
from robot_sf.analysis_workbench.audit_materialize import materialize_episode
from robot_sf.analysis_workbench.audit_scan import scan_campaign
from robot_sf.analysis_workbench.audit_service import (
    AuditSelectionContext,
    AuditService,
    SessionPolicy,
)
from robot_sf.analysis_workbench.review_contracts import (
    component_request_canonical_digest,
    component_request_from_dict,
    experiment_recipe_canonical_digest,
    experiment_recipe_from_dict,
)
from robot_sf.benchmark.analysis_trace import trace_artifact_sha256
from robot_sf.benchmark.runner import run_episode

if TYPE_CHECKING:
    from pathlib import Path


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_json(path: Path, value: Any) -> bytes:
    content = _json_bytes(value)
    path.write_bytes(content)
    return content


def _base_runner_input() -> dict[str, Any]:
    runner_input = {
        "scenario_params": {
            "id": "ba05_native_probe",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
        },
        "seed": 42,
        "horizon": 8,
        "dt": 0.1,
        "robot_start": [-4.0, 0.0],
        "robot_goal": [4.0, 0.0],
        "algo": "simple_policy",
        "algo_config_path": None,
        "record_forces": False,
        "telemetry": {"analysis_trace": "all"},
    }
    runner_input["environment_identity"] = native._environment_identity_for_runner(runner_input)
    return runner_input


class _ReadyOnlyConnection:
    def __init__(self, poll_budgets: list[float]) -> None:
        self.poll_budgets = poll_budgets

    def poll(self, timeout: float) -> bool:
        self.poll_budgets.append(timeout)
        return len(self.poll_budgets) == 1

    def recv(self) -> dict[str, str]:
        return {"status": "ready"}

    def close(self) -> None:
        pass


class _NeverReadyConnection:
    def __init__(self, poll_budgets: list[float], cancel_event: threading.Event) -> None:
        self.poll_budgets = poll_budgets
        self.cancel_event = cancel_event

    def poll(self, timeout: float) -> bool:
        self.poll_budgets.append(timeout)
        self.cancel_event.set()
        return False

    def close(self) -> None:
        pass


class _BoundedFakeProcess:
    pid = 1

    def __init__(self) -> None:
        self.alive = True
        self.terminated = False

    def start(self) -> None:
        pass

    def join(self, _timeout: float) -> None:
        pass

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False


class _BoundedFakeContext:
    def __init__(self, process: _BoundedFakeProcess, poll_budgets: list[float]) -> None:
        self.process = process
        self.poll_budgets = poll_budgets

    def Pipe(self, *, duplex: bool) -> tuple[_ReadyOnlyConnection, _ReadyOnlyConnection]:
        assert duplex is False
        return _ReadyOnlyConnection(self.poll_budgets), _ReadyOnlyConnection([])

    def Process(self, **_kwargs: Any) -> _BoundedFakeProcess:
        return self.process


class _StartFailingFakeProcess:
    pid = None

    def __init__(self) -> None:
        self.is_alive_calls = 0
        self.terminate_calls = 0

    def start(self) -> None:
        raise RuntimeError("spawn failed before pid")

    def is_alive(self) -> bool:
        self.is_alive_calls += 1
        raise AssertionError("can only test child process after it has been started")

    def terminate(self) -> None:
        self.terminate_calls += 1


class _StartFailingFakeContext:
    def __init__(self, process: _StartFailingFakeProcess) -> None:
        self.process = process

    def Pipe(self, *, duplex: bool) -> tuple[_ReadyOnlyConnection, _ReadyOnlyConnection]:
        assert duplex is False
        return _ReadyOnlyConnection([]), _ReadyOnlyConnection([])

    def Process(self, **_kwargs: Any) -> _StartFailingFakeProcess:
        return self.process


class _StartupBudgetFakeContext:
    def __init__(
        self,
        process: _BoundedFakeProcess,
        poll_budgets: list[float],
        cancel_event: threading.Event,
    ) -> None:
        self.process = process
        self.poll_budgets = poll_budgets
        self.cancel_event = cancel_event

    def Pipe(self, *, duplex: bool) -> tuple[_NeverReadyConnection, _ReadyOnlyConnection]:
        assert duplex is False
        return (
            _NeverReadyConnection(self.poll_budgets, self.cancel_event),
            _ReadyOnlyConnection([]),
        )

    def Process(self, **_kwargs: Any) -> _BoundedFakeProcess:
        return self.process


def _make_case(
    tmp_path: Path,
    original: dict[str, Any],
    *,
    runner_input_overrides: dict[str, Any] | None = None,
    source_identity_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = tmp_path / "admitted"
    root.mkdir(parents=True)
    runner_input = _base_runner_input()
    runner_input.update(runner_input_overrides or {})
    runner_input["environment_identity"] = native._environment_identity_for_runner(runner_input)
    config_identity = native._config_identity_for_runner(runner_input)
    initial_state_sha256 = native._initial_state_digest(runner_input)
    source_identity = {
        "source_commit": original["git_hash"],
        "config_identity": config_identity,
        "initial_state_sha256": initial_state_sha256,
        "environment_identity": runner_input["environment_identity"],
    }
    source_identity.update(source_identity_overrides or {})
    source_document = {
        "schema_version": native.NATIVE_SOURCE_SCHEMA_VERSION,
        "source_id": "ba05-native-original-42",
        "original_record": original,
        "runner_input": runner_input,
        "identity": source_identity,
    }
    source_bytes = _write_json(root / "original.json", source_document)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source_ref = {
        "artifact_id": "ba05-native-original",
        "uri": "original.json",
        "format": native.NATIVE_SOURCE_FORMAT,
        "schema": native.NATIVE_SOURCE_SCHEMA_VERSION,
        "sha256": source_sha256,
        "source_commit": original["git_hash"],
        "config_identity": config_identity,
        "units": "metres",
        "coordinate_frame": "world",
    }
    request_id = "ba05-native-request"
    request_document = {
        "schema_version": "component-request.v1",
        "request_id": request_id,
        "component_id": native.COMPONENT_ID,
        "sources": [source_ref],
        "output_directory": "native-diagnostic",
        "config": {},
        "required_capabilities": [],
    }
    recipe_id = "ba05-native-recipe"
    recipe_document = {
        "schema_version": "experiment-recipe.v1",
        "recipe_id": recipe_id,
        "hypothesis": "A bounded native diagnostic can measure goal intervention activation.",
        "source_identity": {
            "kind": "diagnostic",
            "source_uri": "original.json",
            "source_format": native.NATIVE_SOURCE_FORMAT,
            "source_schema": native.NATIVE_SOURCE_SCHEMA_VERSION,
            "source_commit": source_identity["source_commit"],
            "config_identity": source_identity["config_identity"],
            "initial_state_sha256": source_identity["initial_state_sha256"],
            "environment_identity": source_identity["environment_identity"],
            "units": "metres",
            "coordinate_frame": "world",
            "evidence_boundary": native.DIAGNOSTIC_EVIDENCE_BOUNDARY,
            "scientific_claim_allowed": False,
            "dependent_family_status": native.DEPENDENT_FAMILY_STATUS,
        },
        "interventions": [{"intervention_id": "goal-y", "factor": "robot_goal", "priority": 1}],
        "control_conditions": {"same_seed": True, "same_config": True},
        "measurements": [
            {"name": "trajectory_position_delta", "units": "metres"},
            {"name": "control_trace_fidelity", "units": "boolean"},
        ],
        "budget": {"max_runs": 2, "timeout_s": 30},
        "stop_rules": ["stop on source mutation", "stop on control divergence"],
        "preservation_destination": "diagnostic-result-envelope",
        "admission_reference": "ba05-native-receipt",
    }
    parsed_request = component_request_from_dict(request_document, source="test request")
    parsed_recipe = experiment_recipe_from_dict(recipe_document, source="test recipe")
    admission_for_digest = native.NativeDiagnosticAdmission(
        source_root=root,
        receipt_reference="receipt.json",
        receipt_sha256="0" * 64,
        expected_source_commit=original["git_hash"],
        expected_config_identity=config_identity,
    )
    receipt_document = {
        "schema_version": "admitted-source-receipt.v1",
        "receipt_id": "ba05-native-receipt",
        "source": {
            key: source_ref[key]
            for key in (
                "uri",
                "format",
                "schema",
                "sha256",
                "source_commit",
                "config_identity",
                "units",
                "coordinate_frame",
            )
        },
        "request_sha256": component_request_canonical_digest(
            native._source_request_projection(parsed_request, admission_for_digest)
        ),
        "recipe_sha256": experiment_recipe_canonical_digest(parsed_recipe),
        "status": "admitted",
        "source_kind": "diagnostic",
        "evidence_boundary": native.DIAGNOSTIC_EVIDENCE_BOUNDARY,
        "scientific_claim_allowed": False,
    }
    receipt_bytes = _write_json(root / "receipt.json", receipt_document)
    admission_document = {
        "source_root": str(root),
        "receipt_reference": "receipt.json",
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "expected_source_commit": original["git_hash"],
        "expected_config_identity": config_identity,
    }
    return {
        "root": root,
        "source_document": source_document,
        "request_document": request_document,
        "admission_document": admission_document,
        "recipe_document": recipe_document,
        "original": original,
    }


@pytest.fixture(scope="module")
def native_case(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Build one source-admitted case from a real canonical runner episode."""
    runner_input = _base_runner_input()
    record_identity = {
        "config_identity": native._config_identity_for_runner(runner_input),
        "initial_state_sha256": native._initial_state_digest(runner_input),
        "environment_identity": runner_input["environment_identity"],
    }
    original = run_episode(
        runner_input["scenario_params"],
        runner_input["seed"],
        horizon=runner_input["horizon"],
        dt=runner_input["dt"],
        robot_start=runner_input["robot_start"],
        robot_goal=runner_input["robot_goal"],
        record_forces=runner_input["record_forces"],
        algo=runner_input["algo"],
        algo_config_path=runner_input["algo_config_path"],
        telemetry=runner_input["telemetry"],
        provenance={
            "native_diagnostic": {
                "identity": "historical_original",
                "source_id": "ba05-native-original-42",
                "config_identity": record_identity["config_identity"],
                "source_identity": record_identity,
                "runner_input_identity": native._runner_input_identity(runner_input),
            }
        },
    )
    original["provenance"]["native_diagnostic"]["source_commit"] = original["git_hash"]
    assert isinstance(original.get("git_hash"), str)
    assert len(original["git_hash"]) == 40
    return _make_case(tmp_path_factory.mktemp("native-diagnostic"), original)


def _retained_campaign_row(original: dict[str, Any]) -> dict[str, Any]:
    """Project a real runner result into one retained campaign row.

    Returns:
        Current-schema row with the unmodified canonical analysis trace.
    """

    trace = original["algorithm_metadata"]["analysis_trace"]
    return {
        "episode_id": original["episode_id"],
        "scenario_id": original["scenario_id"],
        "seed": original["seed"],
        "planner_id": original["algo"],
        "source_commit": original["git_hash"],
        "config_identity": trace["config_digest"],
        "config_digest": trace["config_digest"],
        "metrics": copy.deepcopy(original["metrics"]),
        "outcome": copy.deepcopy(original["outcome"]),
        "retained_trace": copy.deepcopy(trace),
    }


def test_service_materializes_real_retained_native_trace_without_replay(
    native_case: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selected real runner trace yields derived media, never a historical recording."""
    original = native_case["original"]
    trace = original["algorithm_metadata"]["analysis_trace"]
    row = _retained_campaign_row(original)
    # Source-controlled metadata cannot grant a scanner-default exception.
    row["_audit_scan_identity_defaults"] = {"execution_id": "forged"}
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    _write_json(
        source,
        {
            "schema_version": "campaign-result.v1",
            "campaign_id": "ba05-native-retained-campaign",
            "source_commit": original["git_hash"],
            "config_identity": trace["config_digest"],
            "execution_status": "native",
            "expected_episode_ids": [original["episode_id"]],
            "episodes": [row],
        },
    )
    service = AuditService(tmp_path / "store", campaign_source=source, source_root=root)
    try:
        session = service.open_session(
            AuditSelectionContext(campaign_id="ba05-native-retained-campaign"),
            policy=SessionPolicy(allowed_roots=(str(root),), compute_budget=1.0),
        )
        selected = service.update_context(
            session,
            session.context.next_revision(episode_id=original["episode_id"]),
            expected_context_revision=0,
        )
        assert selected.status == "committed", selected.reason
        episode = service.read_episode(session, original["episode_id"])
        assert episode.status == "complete"
        assert episode.value is not None and episode.value.status == "readable"
        defaults = episode.value.row["_audit_scan_identity_defaults"]
        assert set(defaults) == {"campaign_id", "source_digest", "execution_id"}
        assert episode.value.row["source_digest"] != trace["artifact_sha256"]

        def replay_forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("materialization must not start a simulator")

        monkeypatch.setattr("robot_sf.benchmark.runner.run_episode", replay_forbidden)
        output = root / "derived"
        materialized = service.materialize_selected(
            session,
            output_root=output,
            output_directory="real-native-trace",
            operation_id="real-native-retained-materialization",
        )
        assert materialized.status == "complete", (
            materialized.value.diagnostics if materialized.value else materialized.reason,
        )
        assert materialized.value is not None
        assert materialized.value.materialization_kind == "derived_render"
        assert materialized.value.fidelity == "unverifiable"
        assert materialized.value.source_digest == trace["artifact_sha256"]
        assert materialized.value.simulation_executed is False
        assert "native_retained_trace_projected" in materialized.value.diagnostics
        assert service.get_session(session).usage.compute == 1.0
        manifest = json.loads(
            (output / "real-native-trace" / "audit-materialization.v1.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["materialization_kind"] == "derived_render"
        assert manifest["source_digest"] == trace["artifact_sha256"]
        assert manifest["provenance"]["simulation_advanced"] is False
        assert manifest["provenance"]["native_trace_digest"] == trace["artifact_sha256"]
        assert (output / "real-native-trace" / "trajectory.png").is_file()
    finally:
        service.close()


def test_real_retained_native_trace_explicit_claims_and_tampering_fail_closed(
    native_case: dict[str, Any], tmp_path: Path
) -> None:
    """Scanner defaults never relax standalone explicit row identities."""
    original = native_case["original"]
    trace = original["algorithm_metadata"]["analysis_trace"]
    row = _retained_campaign_row(original)
    root = tmp_path / "source"
    root.mkdir()
    conflicting_execution = materialize_episode(
        {**row, "execution_id": "explicit-other-execution"},
        source_root=root,
        output_root=root / "rejected-execution",
    )
    assert conflicting_execution.reason == "native_retained_trace_unavailable"
    assert any("execution_id" in item for item in conflicting_execution.diagnostics)
    assert conflicting_execution.simulation_executed is False

    explicit_same_execution = materialize_episode(
        {**row, "execution_id": original["episode_id"]},
        source_root=root,
        output_root=root / "rejected-explicit-execution",
    )
    assert explicit_same_execution.reason == "native_retained_trace_unavailable"
    assert any("execution_id" in item for item in explicit_same_execution.diagnostics)

    explicit_campaign = materialize_episode(
        {**row, "campaign_id": "ba05-native-retained-campaign"},
        source_root=root,
        output_root=root / "rejected-explicit-campaign",
    )
    assert explicit_campaign.reason == "native_retained_trace_unavailable"
    assert any("campaign_id" in item for item in explicit_campaign.diagnostics)

    declared_source_trace = copy.deepcopy(trace)
    declared_source_trace["source_digest"] = "1" * 64
    declared_source_trace["artifact_sha256"] = trace_artifact_sha256(declared_source_trace)
    explicit_source = materialize_episode(
        {**row, "source_digest": "2" * 64, "retained_trace": declared_source_trace},
        source_root=root,
        output_root=root / "rejected-explicit-source",
    )
    assert explicit_source.reason == "native_retained_trace_unavailable"
    assert any("source_digest" in item for item in explicit_source.diagnostics)

    tampered_trace = copy.deepcopy(trace)
    tampered_trace["steps"][0]["robot"]["position"][0] += 1.0
    divergent = materialize_episode(
        {**row, "retained_trace": tampered_trace},
        source_root=root,
        output_root=root / "rejected-trace",
    )
    assert divergent.reason == "native_retained_trace_unavailable"
    assert any("digest mismatch" in item for item in divergent.diagnostics)
    assert divergent.simulation_executed is False


def test_direct_materialization_rejects_forged_scanner_identity_defaults(
    native_case: dict[str, Any], tmp_path: Path
) -> None:
    """A raw row cannot turn its explicit campaign into a scanner default."""
    row = _retained_campaign_row(native_case["original"])
    row["campaign_id"] = "selected-campaign"
    row["_audit_scan_identity_defaults"] = {"campaign_id": "selected-campaign"}
    root = tmp_path / "allowed"
    root.mkdir()

    result = materialize_episode(row, source_root=root, output_root=root / "derived")

    assert result.status == "unavailable"
    assert result.reason == "native_retained_trace_unavailable"
    assert any("campaign_id" in item for item in result.diagnostics)
    assert not (root / "derived").exists()
    malformed = materialize_episode(
        row,
        source_root=root,
        output_root=root / "malformed",
        _trusted_scan_identity_defaults={"campaign_id": 42},
    )
    assert malformed.reason == "invalid_episode"


@pytest.mark.parametrize("campaign_alias", ["campaign", "config.campaign"])
@pytest.mark.parametrize("source_form", ["mapping", "json_path", "store_directory"])
def test_campaign_alias_binds_foreign_native_trace_across_source_forms(
    native_case: dict[str, Any], tmp_path: Path, campaign_alias: str, source_form: str
) -> None:
    """Accepted source aliases cannot lose campaign identity before projection."""
    original = native_case["original"]
    row = _retained_campaign_row(original)
    foreign_trace = row["retained_trace"]
    if source_form == "mapping":
        # Mapping-backed sessions have a deliberately small authority envelope.
        row.pop("metrics")
        row.pop("outcome")
        first_step = foreign_trace["steps"][0]
        foreign_trace = {
            key: foreign_trace[key]
            for key in (
                "schema_version",
                "episode_id",
                "seed",
                "scenario_id",
                "planner",
                "git_hash",
                "config_digest",
            )
        } | {
            "steps": [
                {
                    "time_s": first_step["time_s"],
                    "robot": first_step["robot"],
                    "pedestrians": [],
                }
            ]
        }
        row["retained_trace"] = foreign_trace
    foreign_trace["campaign_id"] = "foreign-campaign"
    foreign_trace["artifact_sha256"] = trace_artifact_sha256(foreign_trace)
    declaration = (
        {"campaign": "selected-campaign"}
        if campaign_alias == "campaign"
        else {"config": {"campaign": "selected-campaign"}}
    )
    payload = {
        **declaration,
        "schema_version": "campaign-result.v1",
        "expected_episode_ids": [original["episode_id"]],
        "execution_status": "native",
        "episodes": [row],
    }
    root = tmp_path / "allowed"
    root.mkdir()
    if source_form == "mapping":
        source = payload
    elif source_form == "json_path":
        source = root / "campaign.json"
        _write_json(source, payload)
    else:
        source = root / "campaign-store"
        source.mkdir()
        manifest = {key: value for key, value in payload.items() if key != "episodes"}
        manifest["schema_version"] = "campaign-result-store.v2"
        _write_json(source / "manifest.json", manifest)
        (source / "episodes.jsonl").write_bytes(_json_bytes(row) + b"\n")

    report = scan_campaign(source, root=root)
    episode = next(item for item in report.inventory if item.episode_id == original["episode_id"])
    assert episode.status == "readable", report.counts
    assert episode.row is not None
    assert episode.row["campaign_id"] == "selected-campaign"
    assert episode.row["_audit_scan_identity_defaults"]["campaign_id"] == "selected-campaign"
    if source_form == "store_directory":
        return

    service = AuditService(tmp_path / "store", campaign_source=source, source_root=root)
    try:
        session = service.open_session(
            AuditSelectionContext(campaign_id="selected-campaign"),
            policy=SessionPolicy(allowed_roots=(str(root),), compute_budget=1.0),
        )
        selected = service.update_context(
            session,
            session.context.next_revision(episode_id=original["episode_id"]),
            expected_context_revision=0,
        )
        assert selected.status == "committed", selected.reason
        materialized = service.materialize_selected(
            session,
            output_root=root / "derived",
            operation_id=f"foreign-{campaign_alias}-{source_form}",
        )
        assert materialized.status == "unavailable"
        assert materialized.value is not None
        assert materialized.value.reason == "native_retained_trace_unavailable"
        assert any("campaign_id" in item for item in materialized.value.diagnostics)
        assert materialized.value.simulation_executed is False
        assert not (root / "derived").exists()
    finally:
        service.close()


def test_explicit_row_config_campaign_requires_native_trace_campaign(
    native_case: dict[str, Any], tmp_path: Path
) -> None:
    """A row claim is not a scanner default when the retained trace omits it."""
    original = native_case["original"]
    row = _retained_campaign_row(original)
    assert "campaign_id" not in row["retained_trace"]
    row["config"] = {"campaign": "selected-campaign"}
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    _write_json(
        source,
        {
            "schema_version": "campaign-result.v1",
            "campaign_id": "selected-campaign",
            "expected_episode_ids": [original["episode_id"]],
            "execution_status": "native",
            "episodes": [row],
        },
    )
    report = scan_campaign(source, root=root)
    admitted = report.inventory[0]
    assert admitted.status == "readable"
    assert admitted.row is not None
    assert "campaign_id" not in admitted.row.get("_audit_scan_identity_defaults", {})
    service = AuditService(tmp_path / "store", campaign_source=source, source_root=root)
    try:
        session = service.open_session(
            AuditSelectionContext(campaign_id="selected-campaign"),
            policy=SessionPolicy(allowed_roots=(str(root),), compute_budget=1.0),
        )
        selected = service.update_context(
            session,
            session.context.next_revision(episode_id=original["episode_id"]),
            expected_context_revision=0,
        )
        assert selected.status == "committed", selected.reason
        materialized = service.materialize_selected(
            session, output_root=root / "derived", operation_id="explicit-row-campaign"
        )
        assert materialized.status == "unavailable"
        assert materialized.value is not None
        assert materialized.value.reason == "native_retained_trace_unavailable"
        assert any("campaign_id" in item for item in materialized.value.diagnostics)
    finally:
        service.close()


def _request(case: dict[str, Any], *, goal: tuple[float, float] = (4.0, 4.0)) -> dict[str, Any]:
    return {
        "schema_version": native.NATIVE_DIAGNOSTIC_SCHEMA_VERSION,
        "request_id": case["request_document"]["request_id"],
        "admission": copy.deepcopy(case["admission_document"]),
        "request": copy.deepcopy(case["request_document"]),
        "recipe": copy.deepcopy(case["recipe_document"]),
        "intervention": {
            "intervention_id": "goal-y",
            "factor": "robot_goal",
            "robot_goal": list(goal),
            "activation_epsilon_m": 1e-9,
        },
        "timeout_s": 30.0,
    }


def _diagnostic_record(
    case: dict[str, Any],
    role: str,
    *,
    goal: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Return a fixture record with explicit diagnostic role binding."""
    record = copy.deepcopy(case["original"])
    native_provenance = record["provenance"]["native_diagnostic"]
    native_provenance.update(
        {
            "identity": role,
            "source_id": case["source_document"]["source_id"],
            "source_commit": case["original"]["git_hash"],
            "config_identity": case["source_document"]["identity"]["config_identity"],
        }
    )
    if role != native.ROLE_HISTORICAL_ORIGINAL:
        native_provenance["request_id"] = case["request_document"]["request_id"]
    if goal is not None:
        treatment_input = {
            **case["source_document"]["runner_input"],
            "robot_goal": list(goal),
        }
        native_provenance["runner_input_identity"] = native._runner_input_identity(treatment_input)
    return record


def test_native_smoke_measures_control_and_intervention(native_case: dict[str, Any]) -> None:
    """Exercise the true spawned native route with a real canonical episode."""
    result = native.run_native_diagnostic(_request(native_case))

    assert result.status == native.STATUS_COMPLETE, result.to_dict()
    assert result.fidelity["status"] == "verified"
    assert result.fidelity["checks"]["metrics"] is True
    assert result.activation["status"] == "verified"
    assert result.activation["input_changed"] is True
    assert result.activation["trace_changed"] is True
    assert result.activation["max_position_delta_m"] > 0.0
    assert result.original_identity["kind"] == "historical_original"
    assert result.control_identity["kind"] == "new_diagnostic_control"
    assert result.intervention_identity["kind"] == "new_diagnostic_intervention"
    assert result.provenance["route"] == "native"
    assert result.provenance["canonical_runner"] == "robot_sf.benchmark.runner.run_episode"
    assert result.provenance["input_digests"]["source_bytes_sha256"]
    assert result.activation["control_trace_sha256"]
    assert result.activation["intervention_trace_sha256"]
    assert result.scientific_claim_allowed is False


def test_native_source_mutation_is_unavailable(native_case: dict[str, Any]) -> None:
    """A byte change after admission cannot enter the native runner."""
    source_path = native_case["root"] / "original.json"
    source_bytes = source_path.read_bytes()
    source_path.write_bytes(source_bytes + b" ")
    try:
        result = native.run_native_diagnostic(_request(native_case))
    finally:
        source_path.write_bytes(source_bytes)

    assert result.status == native.STATUS_UNAVAILABLE
    assert "source_mutated" in result.reason


def test_native_missing_original_trace_is_unavailable(native_case: dict[str, Any]) -> None:
    """Missing original telemetry is not converted into a new-run claim."""
    original = copy.deepcopy(native_case["original"])
    original["algorithm_metadata"].pop("analysis_trace")
    case = _make_case(native_case["root"].parent / "missing-trace", original)

    result = native.run_native_diagnostic(_request(case))

    assert result.status == native.STATUS_UNAVAILABLE
    assert "original_telemetry_missing" in result.reason


def test_native_stateful_configuration_fails_closed(native_case: dict[str, Any]) -> None:
    """Checkpoint/stateful planner inputs are rejected before child execution."""
    case = _make_case(
        native_case["root"].parent / "stateful",
        native_case["original"],
        runner_input_overrides={"algo_config_path": "checkpoint.pt"},
    )

    result = native.run_native_diagnostic(_request(case))

    assert result.status == native.STATUS_FAILED
    assert "stateful/checkpoint" in result.reason


def test_native_source_initial_identity_mismatch_fails_closed(
    native_case: dict[str, Any],
) -> None:
    case = _make_case(
        native_case["root"].parent / "initial-identity",
        native_case["original"],
        source_identity_overrides={"initial_state_sha256": "0" * 64},
    )

    result = native.run_native_diagnostic(_request(case))

    assert result.status == native.STATUS_FAILED
    assert "initial_state_sha256" in result.reason


def test_native_source_environment_identity_is_required(native_case: dict[str, Any]) -> None:
    case = _make_case(
        native_case["root"].parent / "environment-identity",
        native_case["original"],
        source_identity_overrides={"environment_identity": None},
    )

    result = native.run_native_diagnostic(_request(case))

    assert result.status == native.STATUS_FAILED
    assert "environment_identity" in result.reason


def test_native_historical_config_metadata_mismatch_fails_closed(
    native_case: dict[str, Any],
) -> None:
    original = copy.deepcopy(native_case["original"])
    original["algorithm_metadata"]["config"] = {"unexpected": True}
    case = _make_case(native_case["root"].parent / "config-metadata", original)

    result = native.run_native_diagnostic(_request(case))

    assert result.status == native.STATUS_FAILED
    assert "planner config metadata" in result.reason


def test_native_historical_initial_trace_mismatch_fails_closed(
    native_case: dict[str, Any],
) -> None:
    original = copy.deepcopy(native_case["original"])
    original["algorithm_metadata"]["analysis_trace"]["steps"][0]["robot"]["position"] = [
        99.0,
        99.0,
    ]
    case = _make_case(native_case["root"].parent / "initial-trace", original)

    result = native.run_native_diagnostic(_request(case))

    assert result.status in {native.STATUS_FAILED, native.STATUS_UNAVAILABLE}
    assert "initial state" in result.reason


def test_native_historical_runner_identity_mismatch_fails_closed(
    native_case: dict[str, Any],
) -> None:
    original = copy.deepcopy(native_case["original"])
    original["provenance"]["native_diagnostic"]["runner_input_identity"]["initial_state_sha256"] = (
        "0" * 64
    )
    case = _make_case(native_case["root"].parent / "runner-identity", original)

    result = native.run_native_diagnostic(_request(case))

    assert result.status == native.STATUS_FAILED
    assert "runner input identity provenance" in result.reason


@pytest.mark.parametrize(
    "role",
    [native.ROLE_DIAGNOSTIC_CONTROL, native.ROLE_DIAGNOSTIC_INTERVENTION],
)
def test_native_historical_role_mismatch_fails_closed(
    native_case: dict[str, Any], role: str
) -> None:
    original = copy.deepcopy(native_case["original"])
    original["provenance"]["native_diagnostic"]["identity"] = role
    case = _make_case(native_case["root"].parent / f"historical-role-{role}", original)

    result = native.run_native_diagnostic(_request(case))

    assert result.status == native.STATUS_FAILED
    assert "record role differs" in result.reason


def test_native_new_control_requires_control_role(
    native_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def historical_pair(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {"status": "ok", "record": copy.deepcopy(native_case["original"]), "elapsed_s": 0.0}

    monkeypatch.setattr(native, "_run_bounded", historical_pair)
    result = native.run_native_diagnostic(_request(native_case))

    assert result.status == native.STATUS_FAILED
    assert "record role differs: expected 'new_diagnostic_control'" in result.reason


def test_native_role_source_and_request_binding_fails_closed(native_case: dict[str, Any]) -> None:
    record = _diagnostic_record(native_case, native.ROLE_DIAGNOSTIC_CONTROL)
    runner_input = native_case["source_document"]["runner_input"]
    source_identity = native_case["source_document"]["identity"]
    provenance = record["provenance"]["native_diagnostic"]

    provenance["source_id"] = "wrong-source"
    errors = native._record_execution_errors(
        record,
        runner_input,
        source_commit=source_identity["source_commit"],
        source_identity=source_identity,
        expected_role=native.ROLE_DIAGNOSTIC_CONTROL,
        request_id=native_case["request_document"]["request_id"],
        source_id=native_case["source_document"]["source_id"],
    )
    assert "record source_id provenance differs" in errors

    provenance["source_id"] = native_case["source_document"]["source_id"]
    provenance.pop("request_id")
    errors = native._record_execution_errors(
        record,
        runner_input,
        source_commit=source_identity["source_commit"],
        source_identity=source_identity,
        expected_role=native.ROLE_DIAGNOSTIC_CONTROL,
        request_id=native_case["request_document"]["request_id"],
        source_id=native_case["source_document"]["source_id"],
    )
    assert "record request_id provenance differs" in errors


@pytest.mark.parametrize(
    "marker", ["fallback_reason", "evidence_eligible", "adapter_active", "nested", "count"]
)
def test_native_fallback_or_ineligible_metadata_fails_closed(
    native_case: dict[str, Any], marker: str
) -> None:
    original = copy.deepcopy(native_case["original"])
    if marker == "fallback_reason":
        original["algorithm_metadata"]["fallback_reason"] = "spoofed"
    elif marker == "evidence_eligible":
        original["algorithm_metadata"]["evidence_eligible"] = False
    elif marker == "adapter_active":
        original["algorithm_metadata"]["planner_kinematics"]["adapter_active"] = True
    elif marker == "nested":
        original["algorithm_metadata"]["planner_diagnostics"] = {"fallback": True}
    else:
        original["algorithm_metadata"]["planner_diagnostics"] = {"fallback_count": 1}
    case = _make_case(native_case["root"].parent / f"metadata-{marker}", original)

    result = native.run_native_diagnostic(_request(case))

    assert result.status == native.STATUS_FAILED
    assert "planner metadata is fallback/ineligible" in result.reason


@pytest.mark.parametrize(
    ("marker_path", "marker_value"),
    [
        (("algorithm_metadata", "execution_mode"), "adapter"),
        (("execution_mode",), "adapter"),
        (("algorithm_metadata", "planner_runtime", "status"), "fallback"),
        (("algorithm_metadata", "planner_diagnostics", "execution_mode"), "adapter"),
        (("algorithm_metadata", "evidence_eligible"), 0),
        (("evidence_eligible",), False),
    ],
    ids=(
        "metadata-execution-mode",
        "record-execution-mode",
        "nested-runtime-status",
        "nested-diagnostic-mode",
        "metadata-malformed-eligibility",
        "record-ineligible",
    ),
)
def test_native_actual_record_canonical_markers_fail_closed(
    native_case: dict[str, Any],
    marker_path: tuple[str, ...],
    marker_value: Any,
) -> None:
    """Canonical marker spoofing is rejected on the real runner record shape."""
    record = copy.deepcopy(native_case["original"])
    target: dict[str, Any] = record
    for key in marker_path[:-1]:
        target = target.setdefault(key, {})
    target[marker_path[-1]] = marker_value
    runner_input = native_case["source_document"]["runner_input"]
    source_identity = native_case["source_document"]["identity"]

    errors = native._record_execution_errors(
        record,
        runner_input,
        source_commit=source_identity["source_commit"],
        source_identity=source_identity,
        expected_role=native.ROLE_HISTORICAL_ORIGINAL,
        source_id=native_case["source_document"]["source_id"],
    )

    assert any("fallback/ineligible" in error for error in errors)


def test_native_actual_record_positive_metadata_has_no_ineligible_markers(
    native_case: dict[str, Any],
) -> None:
    """The real canonical record's neutral fallback fields remain admissible."""
    metadata = native_case["original"]["algorithm_metadata"]

    assert native._metadata_ineligibility_markers(metadata) == ()


def test_native_recipe_identity_change_is_stale(native_case: dict[str, Any]) -> None:
    request = _request(native_case)
    request["recipe"]["source_identity"]["initial_state_sha256"] = "f" * 64

    result = native.run_native_diagnostic(request)

    assert result.status == native.STATUS_UNAVAILABLE
    assert "receipt_stale" in result.reason


def test_native_equal_goal_is_not_activation(native_case: dict[str, Any]) -> None:
    result = native.run_native_diagnostic(_request(native_case, goal=(4.0, 0.0)))

    assert result.status == native.STATUS_FAILED
    assert "intervention_not_effective" in result.reason


def test_native_fidelity_compares_full_metric_mapping(native_case: dict[str, Any]) -> None:
    original = copy.deepcopy(native_case["original"])
    control = copy.deepcopy(original)
    control["metrics"]["distance_travelled"] = (
        float(control["metrics"].get("distance_travelled", 0.0)) + 1.0
    )

    fidelity = native._compare_control_fidelity(original, control)

    assert fidelity["status"] == "diverged"
    assert fidelity["checks"]["metrics"] is False
    assert "metrics" in fidelity["differences"]


def test_native_record_validation_rejects_identity_and_telemetry_gaps(
    native_case: dict[str, Any],
) -> None:
    runner_input = native_case["source_document"]["runner_input"]
    source_identity = native_case["source_document"]["identity"]
    source_commit = source_identity["source_commit"]

    def errors(record: Any) -> tuple[str, ...]:
        return native._record_execution_errors(
            record,
            runner_input,
            source_commit=source_commit,
            source_identity=source_identity,
            expected_role=native.ROLE_HISTORICAL_ORIGINAL,
            source_id=native_case["source_document"]["source_id"],
        )

    assert "record is not a mapping" in errors(None)
    cases = (
        ({"scenario_id": "other"}, "scenario identity"),
        ({"seed": 7}, "seed differs"),
        ({"horizon": 7}, "horizon differs"),
        ({"dt_s": "bad"}, "record dt_s is missing"),
        ({"algo": "other"}, "planner differs"),
        ({"git_hash": "0" * 40}, "source commit differs"),
        ({"scenario_params": {"id": "other"}}, "environment scenario"),
        ({"algorithm_metadata": None}, "algorithm metadata is missing"),
        ({"algorithm_metadata": {"algorithm": "other"}}, "not the intended planner"),
        ({"algorithm_metadata": {"status": "failed"}}, "execution status"),
        ({"algorithm_metadata": {"config": {"x": 1}}}, "planner config metadata"),
        ({"algorithm_metadata": {"planner_kinematics": {}}}, "not native"),
        ({"algorithm_metadata": {"force_diagnostics": {"fallback": True}}}, "fallback"),
        ({"algorithm_metadata": {"analysis_trace": {"schema_version": "other"}}}, "trace schema"),
        ({"algorithm_metadata": {"analysis_trace": {"planner": "other"}}}, "trace planner"),
        ({"algorithm_metadata": {"analysis_trace": {"scenario_id": "other"}}}, "trace scenario"),
        (
            {"algorithm_metadata": {"analysis_trace": {"scenario_digest": "other"}}},
            "environment identity",
        ),
        ({"algorithm_metadata": {"analysis_trace": {"git_hash": "other"}}}, "trace source commit"),
        (
            {"algorithm_metadata": {"analysis_trace": {"artifact_sha256": "0" * 64}}},
            "artifact digest",
        ),
        ({"algorithm_metadata": {"analysis_trace": {"steps": []}}}, "at least two"),
        ({"algorithm_metadata": {"analysis_trace": {"steps": [None, {}]}}}, "step 0 is not"),
        ({"algorithm_metadata": {"analysis_trace": {"steps": [{}, {}]}}}, "no robot state"),
        (
            {"algorithm_metadata": {"analysis_trace": {"steps": [{"robot": {}}, {"robot": {}}]}}},
            "no finite robot position",
        ),
        ({"outcome": None}, "episode outcome"),
        ({"metrics": object()}, "episode metrics"),
        ({"provenance": None}, "source identity provenance"),
        (
            {"provenance": {"native_diagnostic": {"source_identity": {}}}},
            "source identity provenance differs",
        ),
        (
            {
                "provenance": {
                    "native_diagnostic": {
                        "source_identity": native._record_identity_proof(source_identity),
                    }
                }
            },
            "runner input identity provenance is missing",
        ),
        ({"status": "pending"}, "completed runner outcome"),
    )
    for changes, expected in cases:
        record = copy.deepcopy(native_case["original"])
        for key, value in changes.items():
            if key == "algorithm_metadata" and isinstance(value, dict):
                record["algorithm_metadata"].update(value)
            elif key == "provenance" and isinstance(value, dict):
                record["provenance"] = value
            else:
                record[key] = value
        assert any(expected in error for error in errors(record)), (changes, errors(record))


def test_native_low_level_validation_rejects_unsafe_values() -> None:
    with pytest.raises(native.NativeDiagnosticError):
        native._finite(float("nan"), name="value")
    with pytest.raises(native.NativeDiagnosticError):
        native._finite("not-a-number", name="value")
    with pytest.raises(native.NativeDiagnosticError):
        native._text("", name="value")
    with pytest.raises(native.NativeDiagnosticError):
        native._text("x" * (native.MAX_TEXT_CHARS + 1), name="value")
    with pytest.raises(native.NativeDiagnosticError):
        native._text("bad\x00value", name="value")
    with pytest.raises(native.NativeDiagnosticError):
        native._sha256("bad", name="digest")
    with pytest.raises(native.NativeDiagnosticError):
        native._sha40("bad", name="commit")
    with pytest.raises(native.NativeDiagnosticError):
        native._strict_json(float("nan"))
    with pytest.raises(native.NativeDiagnosticError):
        native._strict_json({1: "non-string key"})
    with pytest.raises(native.NativeDiagnosticError):
        native._strict_json([[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[0]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]])
    with pytest.raises(native.NativeDiagnosticError):
        native._canonical_json(object())
    with pytest.raises(native.NativeDiagnosticError):
        native._vector2([1.0], name="vector")


def test_native_control_divergence_is_failed(
    native_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def diverging_control(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        record = _diagnostic_record(native_case, native.ROLE_DIAGNOSTIC_CONTROL)
        record["outcome"] = {"route_complete": True, "collision": False, "timeout": False}
        return {"status": "ok", "record": record, "elapsed_s": 0.0}

    monkeypatch.setattr(native, "_run_bounded", diverging_control)
    result = native.run_native_diagnostic(_request(native_case))

    assert result.status == native.STATUS_FAILED
    assert "control_fidelity_diverged" in result.reason


def test_native_cancel_after_control_is_cancelled(
    native_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    event = threading.Event()

    def cancel_after_control(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        cancel_event = kwargs["cancel_event"]
        cancel_event.set()
        return {
            "status": "ok",
            "record": _diagnostic_record(native_case, native.ROLE_DIAGNOSTIC_CONTROL),
            "elapsed_s": 0.0,
        }

    monkeypatch.setattr(native, "_run_bounded", cancel_after_control)
    result = native.run_native_diagnostic(_request(native_case), cancel_event=event)

    assert result.status == native.STATUS_CANCELLED
    assert "before intervention" in result.reason


def test_native_intervention_crash_is_failed(
    native_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def crash_on_intervention(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        del args, kwargs
        calls += 1
        if calls == 1:
            return {
                "status": "ok",
                "record": _diagnostic_record(native_case, native.ROLE_DIAGNOSTIC_CONTROL),
                "elapsed_s": 0.0,
            }
        return {"status": native.STATUS_FAILED, "reason": "child_crashed_without_result"}

    monkeypatch.setattr(native, "_run_bounded", crash_on_intervention)
    result = native.run_native_diagnostic(_request(native_case))

    assert result.status == native.STATUS_FAILED
    assert "intervention_failed" in result.reason
    assert "child_crashed_without_result" in result.reason


def test_native_post_admission_source_mutation_is_unavailable(
    native_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = native_case["root"] / "original.json"
    source_bytes = source_path.read_bytes()
    calls = 0

    def mutate_on_intervention(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        del args, kwargs
        calls += 1
        if calls == 2:
            source_path.write_bytes(source_bytes + b" ")
            treatment_record = _diagnostic_record(
                native_case,
                native.ROLE_DIAGNOSTIC_INTERVENTION,
                goal=(4.0, 4.0),
            )
            return {"status": "ok", "record": treatment_record, "elapsed_s": 0.0}
        return {
            "status": "ok",
            "record": _diagnostic_record(native_case, native.ROLE_DIAGNOSTIC_CONTROL),
            "elapsed_s": 0.0,
        }

    monkeypatch.setattr(native, "_run_bounded", mutate_on_intervention)
    try:
        result = native.run_native_diagnostic(_request(native_case))
    finally:
        source_path.write_bytes(source_bytes)

    assert result.status == native.STATUS_UNAVAILABLE
    assert "source_mutated" in result.reason


def test_native_cancel_before_admission(native_case: dict[str, Any]) -> None:
    event = threading.Event()
    event.set()

    result = native.run_native_diagnostic(_request(native_case), cancel_event=event)

    assert result.status == native.STATUS_CANCELLED
    assert "before admission" in result.reason


def test_native_child_deadline_terminates_owned_process(native_case: dict[str, Any]) -> None:
    runner_input = native_case["source_document"]["runner_input"]

    result = native._run_bounded(
        runner_input,
        robot_goal=(4.0, 0.0),
        provenance={"test": "deadline"},
        timeout_s=0.001,
    )

    assert result["status"] in {native.STATUS_UNAVAILABLE, native.STATUS_FAILED}
    assert any(
        marker in result["reason"] for marker in ("per_execution_timeout", "child_startup_timeout")
    )
    assert result["settled"] is True


def test_native_child_execution_deadline_starts_after_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the caller deadline starts after child import readiness."""

    poll_budgets: list[float] = []
    process = _BoundedFakeProcess()
    context = _BoundedFakeContext(process, poll_budgets)
    monotonic_values = iter((100.0, 100.0, 101.0, 101.0005, 101.002, 101.003))
    monkeypatch.setattr(native.multiprocessing, "get_context", lambda _method: context)
    monkeypatch.setattr(native.time, "monotonic", lambda: next(monotonic_values))

    result = native._run_bounded(
        _base_runner_input(),
        robot_goal=(4.0, 0.0),
        provenance={"test": "separate-startup-budget"},
        timeout_s=0.001,
    )

    assert poll_budgets[0] == pytest.approx(0.05)
    assert poll_budgets[1] == pytest.approx(0.0005)
    assert result["status"] == native.STATUS_UNAVAILABLE
    assert result["reason"] == "per_execution_timeout: owned child terminated"
    assert result["settled"] is True
    assert process.terminated is True


def test_native_child_start_failure_before_pid_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-pid spawn failure returns a settled result without cleanup escape."""

    process = _StartFailingFakeProcess()
    context = _StartFailingFakeContext(process)
    monotonic_values = iter((100.0, 100.25))
    monkeypatch.setattr(native.multiprocessing, "get_context", lambda _method: context)
    monkeypatch.setattr(native.time, "monotonic", lambda: next(monotonic_values))

    result = native._run_bounded(
        _base_runner_input(),
        robot_goal=(4.0, 0.0),
        provenance={"test": "child-start-failed-before-pid"},
        timeout_s=30.0,
    )

    assert result == {
        "status": native.STATUS_FAILED,
        "reason": "child_start_failed: RuntimeError: spawn failed before pid",
        "settled": True,
        "elapsed_s": pytest.approx(0.25),
    }
    assert process.is_alive_calls == 0
    assert process.terminate_calls == 0


def test_native_child_startup_budget_is_not_shrunk_to_execution_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect the macOS startup allowance from the historical 30-second shrink."""

    poll_budgets: list[float] = []
    cancel_event = threading.Event()
    process = _BoundedFakeProcess()
    context = _StartupBudgetFakeContext(process, poll_budgets, cancel_event)
    # At 31 seconds, the historical formula had already exhausted the caller's
    # 30-second deadline.  The adapter-owned 60-second startup ceiling must
    # still permit polling until cancellation is observed.
    monotonic_values = iter((100.0, 131.0, 131.5))
    monkeypatch.setattr(native.multiprocessing, "get_context", lambda _method: context)
    monkeypatch.setattr(native.time, "monotonic", lambda: next(monotonic_values))

    result = native._run_bounded(
        _base_runner_input(),
        robot_goal=(4.0, 0.0),
        provenance={"test": "startup-budget-not-shrunk"},
        timeout_s=30.0,
        cancel_event=cancel_event,
    )

    assert poll_budgets == [pytest.approx(0.05)]
    assert result["status"] == native.STATUS_CANCELLED
    assert result["reason"] == "cancelled_by_user: owned child terminated"
    assert result["settled"] is True
    assert process.terminated is True
