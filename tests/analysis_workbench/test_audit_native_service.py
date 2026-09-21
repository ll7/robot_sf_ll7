"""Focused proof for the BA-05 native diagnostic service boundary."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import pytest

from robot_sf.analysis_workbench import audit_native_diagnostic as native
from robot_sf.analysis_workbench.audit_mcp import AuditMCPDispatcher
from robot_sf.analysis_workbench.audit_service import (
    AuditSelectionContext,
    AuditService,
    AuditValidationError,
    NativeDiagnosticBinding,
    NativeDiagnosticServiceConfig,
    SessionPolicy,
)
from robot_sf.analysis_workbench.review_contracts import (
    component_request_from_dict,
    experiment_recipe_from_dict,
)
from tests.analysis_workbench import test_audit_native_diagnostic as native_tests

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def admitted_native_case(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Reuse the accepted real-runner source fixture for service-level proof."""

    return native_tests.native_case.__wrapped__(tmp_path_factory)


def _binding(case: dict[str, Any]) -> NativeDiagnosticBinding:
    return NativeDiagnosticBinding(
        admission=native.NativeDiagnosticAdmission.from_mapping(case["admission_document"]),
        request=component_request_from_dict(case["request_document"], source="service test"),
        recipe=experiment_recipe_from_dict(case["recipe_document"], source="service test"),
    )


def _service(
    tmp_path: Path,
    case: dict[str, Any],
    *,
    allowed_roots: tuple[str, ...] | None = None,
    allowed_recipes: tuple[str, ...] | None = None,
    compute_budget: float = 4.0,
) -> tuple[AuditService, Any]:
    binding = _binding(case)
    source = binding.request.sources[0]
    context = AuditSelectionContext(
        campaign_id="native-service-campaign",
        episode_id=case["original"]["episode_id"],
        source_identity=source.sha256,
        source_revision=7,
    )
    service = AuditService(
        tmp_path / "store",
        # Native admission is the source authority for this focused service;
        # the binding revalidates the bytes through its pinned receipt/root.
        campaign_source=None,
        source_ref=source,
        native_diagnostic_config=NativeDiagnosticServiceConfig(bindings=(binding,)),
    )
    policy = SessionPolicy(
        allowed_roots=allowed_roots or (str(case["root"]),),
        allowed_recipes=allowed_recipes or (binding.recipe.recipe_id,),
        compute_budget=compute_budget,
    )
    session = service.open_session(context, policy=policy)
    return service, session


def _request_payload(session: Any, *, operation_id: str = "native-service-op") -> dict[str, Any]:
    return {
        "schema_version": "audit-mcp-request.v1",
        "request_id": operation_id,
        "session_id": session.session_id,
        "session_token": session.session_token,
        "origin": "http://127.0.0.1",
        "operation": "run_native_diagnostic",
        "payload": {
            "operation_id": operation_id,
            "intervention_id": "goal-y",
            "robot_goal": [4.0, 4.0],
        },
    }


def test_native_launcher_config_round_trips_closed_contracts(
    admitted_native_case: dict[str, Any], tmp_path: Path
) -> None:
    """Launcher config accepts only typed, source-bound contract documents."""

    binding = _binding(admitted_native_case)
    typed = NativeDiagnosticBinding.from_mapping(
        {
            "admission": binding.admission,
            "request": binding.request,
            "recipe": binding.recipe,
        }
    )
    assert typed == binding

    document = binding.to_dict()
    parsed = NativeDiagnosticBinding.from_mapping(document)
    assert parsed.request.sources == binding.request.sources
    assert parsed.recipe.recipe_id == binding.recipe.recipe_id

    config = NativeDiagnosticServiceConfig.from_mapping(
        {"bindings": [document], "max_timeout_s": 12.0, "compute_cost": 3.0}
    )
    assert config.max_timeout_s == 12.0
    assert config.compute_cost == 3.0
    assert config.to_dict()["bindings"][0]["admission"] == document["admission"]

    # Exercise the single-binding and empty launcher document forms, then the
    # service's mapping/sequence/override normalization paths.
    single = NativeDiagnosticServiceConfig.from_mapping({**document, "max_timeout_s": 11.0})
    assert single.bindings == (binding,)
    assert NativeDiagnosticServiceConfig.from_mapping({}).bindings == ()
    assert AuditService._coerce_native_diagnostic_config(
        document,
        admission=None,
        request=None,
        recipe=None,
        max_timeout_s=None,
        compute_cost=None,
    ).bindings == (binding,)
    assert AuditService._coerce_native_diagnostic_config(
        [binding],
        admission=None,
        request=None,
        recipe=None,
        max_timeout_s=None,
        compute_cost=None,
    ).bindings == (binding,)
    overridden = AuditService._coerce_native_diagnostic_config(
        NativeDiagnosticServiceConfig(bindings=(binding,)),
        admission=None,
        request=None,
        recipe=None,
        max_timeout_s=9.0,
        compute_cost=1.5,
    )
    assert overridden.max_timeout_s == 9.0
    assert overridden.compute_cost == 1.5

    # ``config=None`` is the disabled native-service launcher mode.
    disabled = AuditService(
        tmp_path / "disabled-store",
        campaign_source=None,
        source_ref=binding.request.sources[0],
    )
    try:
        assert disabled.native_diagnostic_config.bindings == ()
    finally:
        disabled.close()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "must be a mapping"),
        ({"extra": True}, "unknown fields"),
        ({"admission": {}, "request": {}, "recipe": {}}, "invalid native diagnostic binding"),
    ],
)
def test_native_launcher_config_rejects_untrusted_shapes(payload: Any, message: str) -> None:
    """Malformed launcher input fails before a service can admit native work."""

    with pytest.raises(AuditValidationError, match=message):
        NativeDiagnosticBinding.from_mapping(payload)


def test_native_launcher_config_rejects_duplicate_and_unbounded_values(
    admitted_native_case: dict[str, Any],
) -> None:
    """Recipe identity and resource ceilings remain unique and finite."""

    binding = _binding(admitted_native_case)
    with pytest.raises(AuditValidationError, match="bindings are malformed"):
        NativeDiagnosticServiceConfig(bindings=(object(),))
    with pytest.raises(AuditValidationError, match="recipe IDs must be unique"):
        NativeDiagnosticServiceConfig(bindings=(binding, binding))
    with pytest.raises(AuditValidationError, match="max timeout"):
        NativeDiagnosticServiceConfig(bindings=(binding,), max_timeout_s=0.0)
    with pytest.raises(AuditValidationError, match="compute cost"):
        NativeDiagnosticServiceConfig(bindings=(binding,), compute_cost=0.0)
    with pytest.raises(AuditValidationError, match="config must be a mapping"):
        NativeDiagnosticServiceConfig.from_mapping(None)
    with pytest.raises(AuditValidationError, match="unknown fields"):
        NativeDiagnosticServiceConfig.from_mapping({"unexpected": True})
    with pytest.raises(AuditValidationError, match="bindings must be an array"):
        NativeDiagnosticServiceConfig.from_mapping({"bindings": "bad"})
    with pytest.raises(AuditValidationError, match="binding must be a mapping"):
        NativeDiagnosticServiceConfig.from_mapping({"bindings": [object()]})

    with pytest.raises(AuditValidationError, match="cannot be mixed"):
        AuditService._coerce_native_diagnostic_config(
            NativeDiagnosticServiceConfig(bindings=(binding,)),
            admission=binding.admission,
            request=None,
            recipe=None,
            max_timeout_s=None,
            compute_cost=None,
        )
    with pytest.raises(AuditValidationError, match="config is malformed"):
        AuditService._coerce_native_diagnostic_config(
            "bad",
            admission=None,
            request=None,
            recipe=None,
            max_timeout_s=None,
            compute_cost=None,
        )


@pytest.mark.parametrize("failure", ["wrong-root", "wrong-recipe", "wrong-context", "budget"])
def test_native_service_admission_rejections_start_no_child(
    admitted_native_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Root, recipe, context, and compute policy failures are receipted pre-child."""

    case = admitted_native_case
    if failure == "wrong-root":
        service, session = _service(tmp_path, case, allowed_roots=(str(tmp_path),))
    elif failure == "wrong-recipe":
        service, session = _service(tmp_path, case, allowed_recipes=("other-recipe",))
    elif failure == "budget":
        service, session = _service(tmp_path, case, compute_budget=1.0)
    else:
        service, session = _service(tmp_path, case)

    called = False

    def child_must_not_start(*_args: Any, **_kwargs: Any) -> None:
        nonlocal called
        called = True
        raise AssertionError("native child was started after service admission failure")

    monkeypatch.setattr(native, "run_native_diagnostic", child_must_not_start)
    context = (
        session.context.next_revision(episode_id="different")
        if failure == "wrong-context"
        else None
    )
    try:
        result = service.run_native_diagnostic(
            session,
            intervention={"intervention_id": "goal-y", "robot_goal": [4.0, 4.0]},
            context=context,
            operation_id=f"native-{failure}",
        )
        assert result.status in {"conflict", "denied"}
        assert result.operation is not None
        assert result.operation.operation_type == "diagnostic.native"
        assert not called
        assert session.usage.compute == 0.0
    finally:
        service.close()


def test_native_service_requires_selected_episode(
    admitted_native_case: dict[str, Any], tmp_path: Path
) -> None:
    service, session = _service(tmp_path, admitted_native_case)
    try:
        unselected = AuditSelectionContext(
            campaign_id=session.context.campaign_id,
            source_identity=session.context.source_identity,
            source_revision=session.context.source_revision,
        )
        result = service.run_native_diagnostic(
            session,
            intervention={"intervention_id": "goal-y", "robot_goal": [4.0, 4.0]},
            context=unselected,
            operation_id="native-unselected",
        )
        assert result.status == "conflict"
        assert "stale" in result.reason or "selected episode" in result.reason
    finally:
        service.close()


def test_native_service_equal_goal_is_failed_not_evidence(
    admitted_native_case: dict[str, Any], tmp_path: Path
) -> None:
    service, session = _service(tmp_path, admitted_native_case)
    try:
        result = service.run_native_diagnostic(
            session,
            intervention={"intervention_id": "goal-y", "robot_goal": [4.0, 0.0]},
            operation_id="native-equal-goal",
        )
        assert result.status == "failed"
        assert "intervention_not_effective" in result.reason
        assert session.usage.compute == 0.0
    finally:
        service.close()


def test_native_service_mcp_rejects_source_and_recipe_payloads(
    admitted_native_case: dict[str, Any], tmp_path: Path
) -> None:
    service, session = _service(tmp_path, admitted_native_case)
    try:
        response = AuditMCPDispatcher(service).dispatch(
            {
                **_request_payload(session, operation_id="native-mcp-closed"),
                "payload": {
                    "operation_id": "native-mcp-closed",
                    "intervention_id": "goal-y",
                    "robot_goal": [4.0, 4.0],
                    "source_root": str(admitted_native_case["root"]),
                    "recipe_id": "ba05-native-recipe",
                },
            }
        )
        assert response.status == "failed"
        assert "forbidden fields" in response.reason
    finally:
        service.close()


def test_native_service_mcp_routes_through_shared_dispatcher(
    admitted_native_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, session = _service(tmp_path, admitted_native_case)

    def complete(request: Any, *, cancel_event: Any = None) -> native.NativeDiagnosticResult:
        del cancel_event
        return native.NativeDiagnosticResult(
            request_id=request.request_id,
            status=native.STATUS_COMPLETE,
            reason="fake complete",
            evidence_boundary="diagnostic_only",
            scientific_claim_allowed=False,
        )

    monkeypatch.setattr(native, "run_native_diagnostic", complete)
    try:
        response = AuditMCPDispatcher(service).dispatch(_request_payload(session))
        assert response.status == "complete"
        assert response.result["value"]["evidence_boundary"] == "diagnostic_only"
    finally:
        service.close()


def test_native_service_real_smoke_and_duplicate_receipt(
    admitted_native_case: dict[str, Any], tmp_path: Path
) -> None:
    """The service runs the real native pair and replays its durable receipt."""

    service, session = _service(tmp_path, admitted_native_case)
    try:
        first = service.run_native_diagnostic(
            session,
            intervention={"intervention_id": "goal-y", "robot_goal": [4.0, 4.0]},
            operation_id="native-real-smoke",
        )
        assert first.status == "complete", first.to_dict()
        assert first.value is not None
        assert first.value.evidence_boundary == "diagnostic_only"
        assert first.value.scientific_claim_allowed is False
        assert first.operation is not None
        assert (
            first.operation.source_digest
            == admitted_native_case["request_document"]["sources"][0]["sha256"]
        )
        assert session.usage.compute == 2.0

        replay = service.run_native_diagnostic(
            session,
            intervention={"intervention_id": "goal-y", "robot_goal": [4.0, 4.0]},
            operation_id="native-real-smoke",
        )
        assert replay.status == "complete"
        assert replay.operation is not None and replay.operation.replayed
        assert replay.value is None
        assert session.usage.compute == 2.0
    finally:
        service.close()


def test_native_service_replay_rechecks_source_integrity(
    admitted_native_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed admitted source cannot turn a replay into success evidence."""

    service, session = _service(tmp_path, admitted_native_case)
    source_path = admitted_native_case["root"] / "original.json"
    source_bytes = source_path.read_bytes()

    def complete(request: Any, *, cancel_event: Any = None) -> native.NativeDiagnosticResult:
        del cancel_event
        return native.NativeDiagnosticResult(
            request_id=request.request_id,
            status=native.STATUS_COMPLETE,
            reason="fake complete",
            evidence_boundary="diagnostic_only",
            scientific_claim_allowed=False,
        )

    monkeypatch.setattr(native, "run_native_diagnostic", complete)
    try:
        first = service.run_native_diagnostic(
            session,
            intervention={"intervention_id": "goal-y", "robot_goal": [4.0, 4.0]},
            operation_id="native-replay-source",
        )
        assert first.status == "complete", first.to_dict()
        source_path.write_bytes(source_bytes + b" ")
        replay = service.run_native_diagnostic(
            session,
            intervention={"intervention_id": "goal-y", "robot_goal": [4.0, 4.0]},
            operation_id="native-replay-source",
        )
        assert replay.status == "unavailable"
        assert "source" in replay.reason or "receipt" in replay.reason
    finally:
        source_path.write_bytes(source_bytes)
        service.close()


def test_native_service_reconnect_revalidates_admitted_source(
    admitted_native_case: dict[str, Any], tmp_path: Path
) -> None:
    """A durable session reconnect checks native source bytes before reuse."""

    service, session = _service(tmp_path, admitted_native_case)
    binding = _binding(admitted_native_case)
    policy = SessionPolicy(
        allowed_roots=(str(admitted_native_case["root"]),),
        allowed_recipes=(binding.recipe.recipe_id,),
        compute_budget=4.0,
    )
    session_id = session.session_id
    token = session.session_token
    service.close()
    recovered_service = AuditService(
        tmp_path / "store",
        campaign_source=None,
        source_ref=binding.request.sources[0],
        native_diagnostic_config=NativeDiagnosticServiceConfig(bindings=(binding,)),
        trusted_policy=policy,
    )
    try:
        recovered = recovered_service.reconnect_session(session_id, token, policy=policy)
        assert recovered.source_digest == binding.request.sources[0].sha256
        assert recovered.source_ref == binding.request.sources[0]
    finally:
        recovered_service.close()


def test_native_service_crash_leaves_operation_inflight(
    admitted_native_case: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process-level crash does not become a successful retry."""

    service, session = _service(tmp_path, admitted_native_case)

    def crash(*_args: Any, **_kwargs: Any) -> None:
        raise KeyboardInterrupt("simulated child crash")

    monkeypatch.setattr(native, "run_native_diagnostic", crash)
    try:
        with pytest.raises(KeyboardInterrupt):
            service.run_native_diagnostic(
                session,
                intervention={"intervention_id": "goal-y", "robot_goal": [4.0, 4.0]},
                operation_id="native-crash",
            )
        retry = service.run_native_diagnostic(
            session,
            intervention={"intervention_id": "goal-y", "robot_goal": [4.0, 4.0]},
            operation_id="native-crash",
        )
        assert retry.status == "conflict"
        assert "progress" in retry.reason or "inflight" in retry.reason
    finally:
        service.close()


def test_native_service_kill_switch_sets_child_cancel_event(
    admitted_native_case: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared kill switch reaches the native adapter's cancellation event."""

    service, session = _service(tmp_path, admitted_native_case)
    entered = threading.Event()
    observed_cancel = threading.Event()
    result_holder: list[Any] = []

    def blocking(request: Any, *, cancel_event: Any = None) -> native.NativeDiagnosticResult:
        entered.set()
        assert cancel_event is not None
        while not cancel_event.is_set():
            cancel_event.wait(0.01)
        observed_cancel.set()
        return native.NativeDiagnosticResult(
            request_id=request.request_id,
            status=native.STATUS_CANCELLED,
            reason="cancelled_by_user",
            provenance={"evidence_boundary": "diagnostic_only"},
        )

    monkeypatch.setattr(native, "run_native_diagnostic", blocking)
    worker = threading.Thread(
        target=lambda: result_holder.append(
            service.run_native_diagnostic(
                session,
                intervention={"intervention_id": "goal-y", "robot_goal": [4.0, 4.0]},
                operation_id="native-kill",
            )
        )
    )
    try:
        worker.start()
        assert entered.wait(timeout=10)
        cancelled = service.kill_switch(session, operation_id="native-kill-switch")
        assert cancelled.status == "cancelled"
        worker.join(timeout=10)
        assert not worker.is_alive()
        assert observed_cancel.is_set()
        assert result_holder and result_holder[0].status == "cancelled"
    finally:
        if worker.is_alive():
            service.kill_switch(session, operation_id="native-kill-cleanup")
            worker.join(timeout=10)
        service.close()
