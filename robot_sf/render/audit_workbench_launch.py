"""One local, source-bound launch path for the live Benchmark Auditor workbench.

This opens one BA-05 service/store, scans the supplied campaign, builds its
BA-02 queue, and mounts the BA-06 extension in the existing SREV-15 shell.
The audit token never enters the browser document or command-line arguments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.audit_codex import (
    AuditCodexClient,
    CodexCapabilityInspection,
    CodexRouteReceipt,
)
from robot_sf.analysis_workbench.audit_codex_app_server import (
    APP_SERVER_PROTOCOL,
    TESTED_CODEX_CLI_VERSION,
    AppServerMCPConfig,
    CodexAppServerConfig,
    CodexAppServerProvider,
    inspect_live_capabilities,
)
from robot_sf.analysis_workbench.audit_contracts import ReviewRecord
from robot_sf.analysis_workbench.audit_coverage import AuditHealthReport, evaluate_coverage
from robot_sf.analysis_workbench.audit_mcp import LOOPBACK_ORIGINS, AuditMCPDispatcher
from robot_sf.analysis_workbench.audit_mcp_stdio import AuditMCPBridge
from robot_sf.analysis_workbench.audit_queue import AuditQueue, QueueDataset, ScanSummary
from robot_sf.analysis_workbench.audit_scan import scan_campaign
from robot_sf.analysis_workbench.audit_service import (
    AuditContextConflict,
    AuditSelectionContext,
    AuditService,
    AuditServiceError,
    CapabilityUnavailable,
    NativeDiagnosticServiceConfig,
    SessionPolicy,
)
from robot_sf.analysis_workbench.audit_service_adapters import (
    AuditSourceBinding,
    CoverageReadAdapter,
    QueueNextAdapter,
    QueueReadAdapter,
    coverage_reviews_for_scan,
)
from robot_sf.render.audit_workbench import ServiceAuditWorkbenchFacade
from robot_sf.render.audit_workbench_server import make_audit_workbench_server
from robot_sf.render.review_workbench import (
    AUDIT_WORKBENCH_EXTENSION_KEY,
    COMPONENT_ID,
    COMPONENT_VERSION,
    WORKBENCH_SCHEMA_VERSION,
    live_audit_document,
)


@dataclass(slots=True)
class LiveAuditWorkbench:
    """Own one exact-source service and loopback server for a launch lifetime."""

    service: AuditService
    server: Any
    document: dict[str, Any]
    codex_provider: CodexAppServerProvider | None = None
    codex_capability: CodexCapabilityInspection | None = None

    @property
    def url(self) -> str:
        """Return the exact browser origin and SREV-15 route."""

        return f"http://127.0.0.1:{self.server.server_port}/review-workbench.v1.html"

    def close(self) -> None:
        """Release the server socket and audit store."""

        self.server.server_close()
        try:
            if self.codex_provider is not None:
                self.codex_provider.close()
        finally:
            self.service.close()


@dataclass(frozen=True, slots=True)
class CodexAppServerLaunchConfig:
    """Trusted, launcher-only opt-in for one local Codex App Server route.

    The launcher owns the App Server process configuration and the private MCP
    bridge.  Browser requests can supply only bounded prompt data and
    operation/budget values accepted by the existing server contract; prompt
    data is never authority and the browser cannot supply this configuration,
    credentials, route, or process paths.
    """

    app_server: CodexAppServerConfig = field(default_factory=CodexAppServerConfig)
    route_id: str | None = None

    def __post_init__(self) -> None:
        """Reject unpinned or browser-rebindable capability settings."""

        if not isinstance(self.app_server, CodexAppServerConfig):
            raise TypeError("app_server must be a CodexAppServerConfig")
        if self.app_server.expected_cli_version != TESTED_CODEX_CLI_VERSION:
            raise ValueError(
                f"live Codex App Server launch is pinned to codex-cli {TESTED_CODEX_CLI_VERSION}"
            )
        if self.app_server.mcp is not None:
            raise ValueError("the live launcher owns the private MCP bridge")
        if self.route_id is not None and (
            not isinstance(self.route_id, str) or not self.route_id.strip()
        ):
            raise ValueError("route_id must be a non-empty trusted string")


# Short aliases keep the capability discoverable without introducing a second
# configuration shape for callers that use the live-workbench terminology.
LiveCodexConfig = CodexAppServerLaunchConfig
CodexLaunchConfig = CodexAppServerLaunchConfig


@dataclass(frozen=True, slots=True)
class _LiveCodexBinding:
    """Private launcher products retained for the workbench lifetime."""

    client: AuditCodexClient
    provider: CodexAppServerProvider
    capability: CodexCapabilityInspection
    route: CodexRouteReceipt


def _bind_live_codex(
    config: CodexAppServerLaunchConfig,
    *,
    service: AuditService,
    session: Any,
    source_root: Path,
) -> _LiveCodexBinding:
    """Admit one actual App Server route and bind it to the service session.

    Capability discovery runs before a facade or HTTP server is returned.  A
    missing executable, unsupported CLI version, ambiguous route, malformed
    capability response, or private-bridge failure therefore leaves no
    browser-visible partial launch.

    Returns:
        The server-held Codex client, provider, capability inspection, and
        exact discovered route.
    """

    app_server = config.app_server
    if app_server.cwd is None:
        app_server = replace(app_server, cwd=source_root)
    bridge = AuditMCPBridge(
        AuditMCPDispatcher(service),
        session_id=session.session_id,
        session_token=session.session_token,
        origin="http://127.0.0.1",
    )
    if bridge.origin not in LOOPBACK_ORIGINS:
        bridge.close()
        raise ValueError("live Codex MCP bridge must use a loopback origin")
    try:
        bridge.start()
        app_server = replace(app_server, mcp=AppServerMCPConfig(bridge))
        capability = inspect_live_capabilities(app_server)
        route = capability.choose(config.route_id)
        if capability.status != "available" or route is None:
            reason = capability.reason or "no uniquely discovered App Server route"
            raise CapabilityUnavailable(f"live Codex App Server is unavailable: {reason}")
        if route.protocol != APP_SERVER_PROTOCOL:
            raise CapabilityUnavailable("live Codex route uses an unsupported App Server protocol")
        if route.client_version != TESTED_CODEX_CLI_VERSION:
            raise CapabilityUnavailable("live Codex route CLI version does not match the adapter")
        # Keep the inspector stable for this launch.  Re-running discovery for
        # every browser request could silently rebind a durable service session
        # to a different model/provider route.
        bound_capability = CodexCapabilityInspection(
            "available",
            routes=(route,),
            inspector=capability.inspector,
            installed=capability.installed,
        )
        provider = CodexAppServerProvider(config=app_server)
        client = AuditCodexClient(
            service,
            inspector=lambda: bound_capability,
            provider=provider,
        )
        return _LiveCodexBinding(client, provider, bound_capability, route)
    except Exception:
        bridge.close()
        raise


def _coerce_native_diagnostic_config(
    config: NativeDiagnosticServiceConfig | Mapping[str, Any] | None,
) -> NativeDiagnosticServiceConfig | None:
    """Normalize one launcher-owned native diagnostic configuration.

    Returns:
        The validated native configuration, or ``None`` when disabled.
    """

    if config is None:
        return None
    try:
        selected = (
            config
            if isinstance(config, NativeDiagnosticServiceConfig)
            else NativeDiagnosticServiceConfig.from_mapping(config)
        )
    except (AuditServiceError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"native diagnostic config is invalid: {exc}") from exc
    if len(selected.bindings) != 1:
        raise ValueError("native diagnostic config requires exactly one binding")
    if selected.bindings[0].campaign_row is None:
        raise ValueError("native diagnostic config requires a campaign_row binding")
    if not math.isfinite(selected.compute_cost) or selected.compute_cost <= 0.0:
        raise ValueError("native diagnostic compute_cost must be positive and finite")
    return selected


def _validate_native_diagnostic_policy(
    config: NativeDiagnosticServiceConfig,
    *,
    policy: SessionPolicy | None,
) -> None:
    """Require explicit policy authority for one native launch."""

    if policy is None:
        raise ValueError("native diagnostic launch requires an explicit session policy")
    if policy.compute_budget < config.compute_cost:
        raise ValueError(
            "session policy compute_budget is insufficient for native diagnostic compute_cost"
        )
    binding = config.bindings[0]
    root_value = binding.admission.source_root
    try:
        native_root = Path(root_value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("native diagnostic source root is unavailable") from exc
    if not native_root.is_dir():
        raise ValueError("native diagnostic source root is not a directory")
    if not policy.allows_path(native_root):
        raise ValueError("session policy must allow the native diagnostic source root")
    if not policy.allows_recipe(binding.recipe.recipe_id):
        raise ValueError("session policy must allow the native diagnostic recipe")


def _native_episode_ref_matches_row(ref: Any, row: Mapping[str, Any]) -> bool:
    """Match BA-01 identity fields without trusting a literal episode ID.

    Returns:
        Whether the generated reference and retained row share the identity.
    """

    execution_id = row.get("execution_id") or row.get("run_id") or row.get("episode_id")
    planner_id = row.get("planner_id") or row.get("planner") or row.get("algo") or ""
    scenario_id = row.get("scenario_id") or row.get("scenario") or ""
    checkpoint_digest = row.get("checkpoint_digest") or row.get("checkpoint_hash") or ""
    environment_digest = row.get("environment_digest") or row.get("environment_hash") or ""
    config_digest = row.get("config_digest") or row.get("config_hash") or ""
    if not isinstance(config_digest, str):
        return False
    return (
        isinstance(execution_id, str)
        and ref.execution_id == execution_id.strip()
        and ref.planner_id == planner_id
        and ref.scenario_id == scenario_id
        and ref.seed == row.get("seed")
        and ref.attempt == row.get("attempt", 0)
        and ref.config_digest == config_digest.lower()
        and ref.checkpoint_digest == checkpoint_digest
        and ref.environment_digest == environment_digest
    )


def _native_episode_ref_for_binding(
    service: AuditService,
    report: Any,
    descriptor: Any,
) -> Any:
    """Resolve one source-bound BA-01 ref for a campaign-row descriptor.

    Returns:
        The unique generated episode reference.
    """

    matches: list[Any] = []
    for item in report.inventory:
        row = item.row
        if item.status != "readable" or not isinstance(row, Mapping):
            continue
        if service._native_campaign_row_claim_errors(row, descriptor):
            continue
        refs = tuple(
            ref
            for ref in report.episode_refs
            if ref.campaign_digest == report.audit.campaign_digest
            and ref.source_digest == report.audit.source_digest
            and ref.source is not None
            and ref.source.uri == report.audit.source.uri
            and _native_episode_ref_matches_row(ref, row)
        )
        if len(refs) > 1:
            raise AuditContextConflict(
                "native diagnostic campaign row resolves to ambiguous EpisodeRefs"
            )
        if len(refs) == 1:
            matches.append(refs[0])
    if len(matches) != 1:
        raise AuditContextConflict(
            "native diagnostic campaign row does not resolve to one EpisodeRef"
        )
    return matches[0]


def _native_launch_preflight(
    service: AuditService,
    session: Any,
    report: Any,
    config: NativeDiagnosticServiceConfig,
) -> None:
    """Revalidate campaign/native identities before exposing a browser route."""

    try:
        from robot_sf.analysis_workbench import audit_native_diagnostic as native  # noqa: PLC0415
    except (ImportError, ModuleNotFoundError) as exc:
        raise CapabilityUnavailable("native diagnostic adapter is unavailable") from exc
    binding = service._native_campaign_binding_for_session(session)
    descriptor = binding.campaign_row
    if descriptor is None:
        raise ValueError("native diagnostic campaign_row binding is missing")
    source = report.audit.source
    if source is None:
        raise ValueError("native diagnostic campaign source identity is unavailable")
    if (
        descriptor.campaign_uri != source.uri
        or descriptor.campaign_sha256 != report.audit.source_digest
    ):
        raise ValueError("native diagnostic campaign binding is stale")
    episode_ref = _native_episode_ref_for_binding(service, report, descriptor)
    selected = session.context.next_revision(episode_id=episode_ref.episode_id)
    episode = service._native_campaign_episode_for_session(session, selected, binding)

    # Read the admitted native source once to choose a guaranteed-different
    # bounded goal for the launcher preflight. The service preflight below
    # repeats the full source/receipt/row validation immediately afterwards.
    probe = service._native_request(
        binding,
        {
            "intervention_id": "native-launch-preflight",
            "factor": "robot_goal",
            "robot_goal": (0.0, 0.0),
            "activation_epsilon_m": 0.0,
        },
        deadline_s=config.max_timeout_s,
    )
    root_fd: int | None = None
    try:
        native._validate_request(probe)
        _root, root_fd, _receipt, _receipt_digest, resolution = native._open_and_resolve_source(
            probe
        )
        if resolution.receipt is None:
            raise ValueError("native diagnostic source receipt is unavailable")
        source_document = native._source_document(
            resolution.source_bytes or b"", resolution.receipt.source
        )
        original_goal = native._vector2(
            source_document.runner_input["robot_goal"], name="runner_input.robot_goal"
        )
    except (
        native.NativeDiagnosticError,
        KeyError,
        IndexError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        detail = " ".join(str(exc).split())[:256] or "invalid admitted source"
        raise CapabilityUnavailable(
            f"native diagnostic source preflight is unavailable: {detail}"
        ) from exc
    finally:
        if root_fd is not None:
            try:
                os.close(root_fd)
            except OSError:
                pass
    intervention_goal = (
        (1.0, original_goal[1]) if original_goal[0] == 0.0 else (0.0, original_goal[1])
    )
    request = service._native_request(
        binding,
        {
            "intervention_id": "native-launch-preflight",
            "factor": "robot_goal",
            "robot_goal": intervention_goal,
            "activation_epsilon_m": 0.0,
        },
        deadline_s=config.max_timeout_s,
    )
    service._preflight_native_request(
        request,
        selected,
        campaign_row=episode.row,
        campaign_binding=descriptor,
    )


def _launch_policy(
    source_root: Path,
    *,
    policy: SessionPolicy | None,
    materialization_output_root: Path | None,
    materialization_compute_budget: float,
    codex_config: CodexAppServerLaunchConfig | None = None,
    native_diagnostic_config: NativeDiagnosticServiceConfig | None = None,
) -> tuple[SessionPolicy, Path | None]:
    """Resolve one trusted output root and finite launch policy.

    Returns:
        The admitted policy and optional server-held output root.
    """

    output_root = (
        materialization_output_root.resolve(strict=False)
        if materialization_output_root is not None
        else None
    )
    if output_root is not None and (
        output_root == source_root
        or output_root.is_relative_to(source_root)
        or source_root.is_relative_to(output_root)
    ):
        raise ValueError("materialization output root must be separate from source root")
    if output_root is None and materialization_compute_budget != 0.0:
        raise ValueError("materialization budget requires a trusted output root")
    output_policy_root = (
        output_root.parent.resolve(strict=True) if output_root is not None else None
    )
    selected_policy = policy or SessionPolicy(
        allowed_roots=tuple(
            dict.fromkeys(
                (str(source_root), str(output_policy_root))
                if output_policy_root is not None
                else (str(source_root),)
            )
        ),
        compute_budget=materialization_compute_budget,
    )
    if str(source_root) not in selected_policy.allowed_roots:
        raise ValueError("session policy must allow the admitted source root")
    if output_root is not None and not selected_policy.allows_path(output_root):
        raise ValueError("session policy must allow the trusted materialization output root")
    if codex_config is not None:
        if selected_policy.token_budget < 1:
            raise ValueError("live Codex launch requires a positive trusted token budget")
        if selected_policy.compute_budget < 1.0:
            raise ValueError(
                "live Codex launch requires trusted compute_budget >= 1.0; "
                "the App Server charge is conservative and unmeasured"
            )
    if native_diagnostic_config is not None:
        _validate_native_diagnostic_policy(native_diagnostic_config, policy=selected_policy)
    return selected_policy, output_root


def open_live_audit_workbench(  # noqa: C901, PLR0913
    campaign: Path,
    *,
    source_root: Path,
    store_root: Path,
    actor_id: str = "local-reviewer",
    policy: SessionPolicy | None = None,
    materialization_output_root: Path | None = None,
    materialization_compute_budget: float = 0.0,
    codex_config: CodexAppServerLaunchConfig | None = None,
    native_diagnostic_config: NativeDiagnosticServiceConfig | Mapping[str, Any] | None = None,
) -> LiveAuditWorkbench:
    """Construct one truthful service-backed launch from an admitted campaign.

    No browser route is exposed until the scan, source identity, queue binding,
    service session, coverage adapter, and SREV-15 mount have all succeeded.

    Returns:
        An owned loopback server and service, requiring ``close`` by the caller.
    """

    allowed_root = source_root.resolve(strict=True)
    source = campaign.resolve(strict=True)
    if not source.is_file() or not source.is_relative_to(allowed_root):
        raise ValueError("campaign must be a regular file inside source_root")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("campaign_id"), str):
        raise ValueError("campaign source requires a campaign_id")
    native_config = _coerce_native_diagnostic_config(native_diagnostic_config)
    selected_policy, output_root = _launch_policy(
        allowed_root,
        policy=policy,
        materialization_output_root=materialization_output_root,
        materialization_compute_budget=materialization_compute_budget,
        codex_config=codex_config,
        native_diagnostic_config=native_config,
    )
    service = AuditService(
        store_root,
        campaign_source=source,
        source_root=allowed_root,
        trusted_policy=selected_policy,
        native_diagnostic_config=native_config,
    )
    live_codex: _LiveCodexBinding | None = None
    try:
        report = scan_campaign(source, root=allowed_root, store=service.store)
        session = service.open_session(
            AuditSelectionContext(campaign_id=payload["campaign_id"]),
            actor="human",
            actor_id=actor_id,
            policy=selected_policy,
        )
        if report.audit.source_digest != session.source_digest:
            raise ValueError("scan and service disagree on source digest")
        bound = service.update_context(
            session,
            session.context.next_revision(source_identity=session.source_digest),
            expected_context_revision=0,
        )
        if bound.status != "committed":
            raise ValueError("could not bind the audit session to the admitted source")
        dataset = QueueDataset(
            candidates=report.episode_refs,
            signals=report.signals,
            campaign_digest=report.audit.campaign_digest,
            source_digest=session.source_digest,
            protocol_version=report.audit.protocol_version,
            protocol_digest=report.audit.protocol_digest,
            scan_summary=ScanSummary.from_scan_report(report),
        )
        binding = AuditSourceBinding(
            payload["campaign_id"],
            report.audit.campaign_digest,
            session.source_digest,
            session.source_revision,
        )
        queue_path = store_root / "audit-workbench-queue.json"

        def read_queue() -> AuditQueue:
            # BA-02 appends explicit reviews to its input identity. Rebuild
            # only those receipts named by durable queue state on reconnect;
            # arbitrary BA-03 reviews must not silently mutate queue input.
            probe = AuditQueue(dataset, state_path=queue_path, store=service.store)
            if not probe.state.operation_record_ids:
                return probe
            committed = []
            for review_id in probe.state.operation_record_ids.values():
                stored = service.store.get(review_id)
                if (
                    stored is None
                    or stored.record_type != "review_record"
                    or not isinstance(stored.record, ReviewRecord)
                ):
                    raise CapabilityUnavailable("BA-02 durable review receipt is unavailable")
                committed.append(stored)
            committed.sort(key=lambda item: item.global_revision)
            resumed_dataset = replace(
                dataset, review_records=tuple(item.record for item in committed)
            )
            return AuditQueue(resumed_dataset, state_path=queue_path, store=service.store)

        service.queue_adapter = QueueReadAdapter(binding, read_queue)
        queue_adapter = QueueNextAdapter(binding, read_queue)
        service.next_adapter = queue_adapter
        service.queue_next_adapter = queue_adapter

        def coverage_report() -> AuditHealthReport:
            """Recompute BA-04 from the admitted scan and typed BA-03 reviews.

            Returns:
                A freshly evaluated BA-04 health report.
            """

            if (
                report.audit.campaign_digest != binding.campaign_digest
                or report.audit.source_digest != binding.source_digest
            ):
                raise AuditContextConflict("BA-01 scan identity changed")
            projected = service.store.list_records(record_type="review_record")
            reviews: list[ReviewRecord] = []
            for stored in projected:
                if stored.record_type != "review_record" or not isinstance(
                    stored.record, ReviewRecord
                ):
                    raise CapabilityUnavailable("BA-03 review record projection is malformed")
                reviews.append(stored.record)
            # BA-04 keeps BA-01's source commit and scan cache key as review
            # provenance.  BA-05's numeric source revision separately protects
            # the service session and is never substituted for that commit.
            return evaluate_coverage(
                scan=report,
                review_records=coverage_reviews_for_scan(report, tuple(reviews)),
            )

        service.coverage_adapter = CoverageReadAdapter(
            binding,
            coverage_report,
            report_source_revision=(
                report.audit.source.source_commit if report.audit.source is not None else ""
            ),
        )
        if native_config is not None:
            _native_launch_preflight(service, session, report, native_config)
        live_codex = (
            _bind_live_codex(
                codex_config,
                service=service,
                session=session,
                source_root=allowed_root,
            )
            if codex_config is not None
            else None
        )
        facade = ServiceAuditWorkbenchFacade(
            service,
            session,
            token=session.session_token,
            codex_client=live_codex.client if live_codex is not None else None,
        )
        base_document: dict[str, Any] = {
            "schema_version": WORKBENCH_SCHEMA_VERSION,
            "component_id": COMPONENT_ID,
            "component_version": COMPONENT_VERSION,
            "request_id": "audit-workbench-local",
            "output_directory": "",
            "episodes": [],
            "artifacts": [],
            "verified_artifacts": [],
            "time_base": {
                "authority": "simulation_telemetry",
                "video_alignment": "unavailable",
            },
            "presentation": {"status": "unavailable", "reason": "no case selected"},
            "extension_slots": ["panels", "annotations", "storyboard", "comparison"],
            "extensions": {AUDIT_WORKBENCH_EXTENSION_KEY: {"slot": "panels"}},
            "diagnostics": [],
            "provenance": {"component_id": COMPONENT_ID, "admission": "not_evaluated"},
        }
        document = live_audit_document(base_document, facade)
        server = make_audit_workbench_server(
            document, facade, materialization_output_root=output_root
        )
        return LiveAuditWorkbench(
            service,
            server,
            document,
            codex_provider=live_codex.provider if live_codex is not None else None,
            codex_capability=live_codex.capability if live_codex is not None else None,
        )
    except Exception:
        if live_codex is not None:
            live_codex.provider.close()
        service.close()
        raise


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Read one trusted launcher object without accepting partial authority.

    Returns:
        One JSON object loaded from the regular file.
    """

    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} file is unavailable: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} path is not a regular file: {path}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} file is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _load_session_policy(path: Path) -> SessionPolicy:
    """Load a versioned, finite session policy for the trusted CLI launch.

    Returns:
        The validated finite session policy.
    """

    payload = _read_json_object(path, label="session policy")
    try:
        return SessionPolicy.from_mapping(payload)
    except (AuditServiceError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"session policy is invalid: {exc}") from exc


def _load_native_diagnostic_config(path: Path) -> NativeDiagnosticServiceConfig:
    """Load one guarded, campaign-row-bound native diagnostic config.

    Returns:
        The validated campaign-row-bound configuration.
    """

    payload = _read_json_object(path, label="native diagnostic config")
    try:
        config = NativeDiagnosticServiceConfig.from_mapping(payload)
    except (AuditServiceError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"native diagnostic config is invalid: {exc}") from exc
    selected = _coerce_native_diagnostic_config(config)
    assert selected is not None
    return selected


def _load_codex_launch_config(
    path: Path, *, route_id: str | None = None
) -> CodexAppServerLaunchConfig:
    """Load the launcher-owned App Server settings and optional route pin.

    Returns:
        The validated launcher-only App Server capability configuration.
    """

    payload = _read_json_object(path, label="Codex App Server config")
    unknown = set(payload) - {"app_server", "route_id"}
    if unknown:
        raise ValueError(
            "Codex App Server config contains unknown fields: " + ", ".join(sorted(unknown))
        )
    app_server_payload = payload.get("app_server")
    if not isinstance(app_server_payload, Mapping):
        raise ValueError("Codex App Server config requires an app_server object")
    if "mcp" in app_server_payload:
        raise ValueError("Codex App Server config cannot supply MCP; the launcher owns it")
    env = app_server_payload.get("env")
    if env is not None:
        if not isinstance(env, Mapping):
            raise ValueError("Codex App Server config env must be a JSON object")
        if any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in env.items()
        ):
            raise ValueError("Codex App Server config env keys and values must be strings")
    config_route_id = payload.get("route_id")
    if route_id is not None and config_route_id is not None and route_id != config_route_id:
        raise ValueError("CLI route ID conflicts with the configured trusted route ID")
    selected_route_id = route_id if route_id is not None else config_route_id
    try:
        app_server = CodexAppServerConfig(**dict(app_server_payload))
        return CodexAppServerLaunchConfig(
            app_server=app_server,
            route_id=selected_route_id,
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"Codex App Server config is invalid: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    """Launch an exact-origin local workbench until interrupted.

    Returns:
        Exit code zero after a normal keyboard interruption.
    """

    parser = argparse.ArgumentParser(description="Open a local Benchmark Auditor workbench")
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--actor-id", default="local-reviewer")
    parser.add_argument("--materialization-output-root", type=Path)
    parser.add_argument("--materialization-compute-budget", type=float, default=0.0)
    parser.add_argument(
        "--codex-app-server",
        "--codex-config",
        dest="codex_app_server",
        type=Path,
        help="trusted JSON config for the opt-in Codex App Server route",
    )
    parser.add_argument(
        "--session-policy",
        type=Path,
        help="trusted versioned JSON SessionPolicy required by Codex/native diagnostics",
    )
    parser.add_argument(
        "--native-diagnostic-config",
        type=Path,
        help="trusted JSON config for one campaign-bound native diagnostic",
    )
    parser.add_argument(
        "--codex-route-id",
        "--codex-route",
        dest="codex_route_id",
        help="optional trusted route ID from App Server capability discovery",
    )
    args = parser.parse_args(argv)
    if args.codex_route_id is not None and args.codex_app_server is None:
        parser.error("--codex-route-id requires --codex-app-server")
    if args.codex_app_server is not None and args.session_policy is None:
        parser.error("--session-policy is required with --codex-app-server")
    if args.native_diagnostic_config is not None and args.session_policy is None:
        parser.error("--session-policy is required with --native-diagnostic-config")
    try:
        policy = _load_session_policy(args.session_policy) if args.session_policy else None
        native_diagnostic_config = (
            _load_native_diagnostic_config(args.native_diagnostic_config)
            if args.native_diagnostic_config is not None
            else None
        )
        codex_config = (
            _load_codex_launch_config(args.codex_app_server, route_id=args.codex_route_id)
            if args.codex_app_server is not None
            else None
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    try:
        opened = open_live_audit_workbench(
            args.campaign,
            source_root=args.source_root,
            store_root=args.store_root,
            actor_id=args.actor_id,
            policy=policy,
            materialization_output_root=args.materialization_output_root,
            materialization_compute_budget=args.materialization_compute_budget,
            codex_config=codex_config,
            native_diagnostic_config=native_diagnostic_config,
        )
    except (AuditServiceError, OSError, TypeError, ValueError) as exc:
        parser.error(f"launch denied: {exc}")
    try:
        print(opened.url)  # noqa: T201 - local CLI exposes only its loopback URL
        opened.server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        opened.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
