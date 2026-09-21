"""One-command live workbench launch contracts."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import stat
import threading
from dataclasses import replace as dataclass_replace
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pytest

from robot_sf.analysis_workbench import audit_native_diagnostic as native
from robot_sf.analysis_workbench.audit_codex_app_server import CodexAppServerConfig
from robot_sf.analysis_workbench.audit_service import (
    AuditContextConflict,
    CapabilityUnavailable,
    NativeDiagnosticBinding,
    NativeDiagnosticCampaignRowBinding,
    NativeDiagnosticServiceConfig,
    SessionPolicy,
)
from robot_sf.analysis_workbench.review_contracts import (
    component_request_from_dict,
    experiment_recipe_from_dict,
)
from robot_sf.render import audit_workbench_launch as launch_module
from robot_sf.render.audit_workbench_launch import (
    CodexAppServerLaunchConfig,
    open_live_audit_workbench,
)
from tests.analysis_workbench import test_audit_native_diagnostic as native_diagnostic_tests

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "audit_campaign_v1"
    / "campaign.json"
)
TRACE_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
    / "minimal_trace.json"
)

FAKE_APP_SERVER_BODY = r"""
import json
import sys

if "--version" in sys.argv:
    print("codex-cli 0.154.0", flush=True)
    raise SystemExit(0)

turn_number = 0
for raw in sys.stdin:
    request = json.loads(raw)
    method = request.get("method")
    request_id = request.get("id")

    def reply(result):
        print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)

    if method == "initialize":
        reply({"userAgent": "fake-codex/0.154.0"})
    elif method == "initialized":
        pass
    elif method == "model/list":
        reply({"data": [{"id": "model-fixture", "isDefault": True}]})
    elif method == "thread/start":
        reply(
            {
                "thread": {
                    "id": "thread-fixture",
                    "ephemeral": bool(request.get("params", {}).get("ephemeral", False)),
                },
                "model": "model-fixture",
                "modelProvider": "provider-fixture",
            }
        )
    elif method == "turn/start":
        turn_number += 1
        params = request.get("params", {})
        thread_id = params.get("threadId", "thread-fixture")
        turn_id = "turn-" + str(turn_number)
        reply({"turn": {"id": turn_id}})
        usage = {
            "last": {"totalTokens": 7},
            "total": {"totalTokens": 7 * turn_number},
        }
        print(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "thread/tokenUsage/updated",
                    "params": {"threadId": thread_id, "turnId": turn_id, "tokenUsage": usage},
                }
            ),
            flush=True,
        )
        print(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "turn/completed",
                    "params": {
                        "threadId": thread_id,
                        "turn": {"id": turn_id, "status": "completed"},
                    },
                }
            ),
            flush=True,
        )
    elif method in {"thread/resume", "turn/interrupt"}:
        if method == "thread/resume":
            reply(
                {
                    "thread": {"id": "thread-fixture", "ephemeral": False},
                    "model": "model-fixture",
                    "modelProvider": "provider-fixture",
                }
            )
        else:
            reply({})
    else:
        reply({})
"""


def _fake_app_server(tmp_path: Path) -> Path:
    executable = tmp_path / "fake-codex"
    executable.write_text(
        "#!" + os.sys.executable + "\n" + FAKE_APP_SERVER_BODY,
        encoding="utf-8",
    )
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return executable


@pytest.fixture(scope="module")
def admitted_native_case(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Reuse the canonical real-runner native admission fixture."""

    return native_diagnostic_tests.native_case.__wrapped__(tmp_path_factory)


def _native_campaign_config(
    case: dict[str, object], *, descriptor_overrides: dict[str, object] | None = None
) -> tuple[Path, NativeDiagnosticServiceConfig]:
    """Create one campaign row and its trusted native binding document."""

    root = case["root"]
    original = case["original"]
    source_document = case["source_document"]
    assert isinstance(root, Path)
    assert isinstance(original, dict)
    assert isinstance(source_document, dict)
    config_identity = source_document["identity"]["config_identity"]
    campaign_payload = {
        "schema_version": "campaign-result.v1",
        "campaign_id": "native-launch-campaign",
        "source_commit": original["git_hash"],
        "config_identity": config_identity,
        "execution_status": "native",
        "expected_episode_ids": [original["episode_id"]],
        "episodes": [
            {
                "episode_id": original["episode_id"],
                "scenario_id": original["scenario_id"],
                "seed": original["seed"],
                "planner_id": original["algo"],
                "source_commit": original["git_hash"],
                "config_identity": config_identity,
            }
        ],
    }
    campaign = root / "native-launch-campaign.json"
    campaign_bytes = json.dumps(campaign_payload, sort_keys=True).encode("utf-8")
    campaign.write_bytes(campaign_bytes)

    admission = native.NativeDiagnosticAdmission.from_mapping(case["admission_document"])
    request = component_request_from_dict(case["request_document"], source="launch test")
    recipe = experiment_recipe_from_dict(case["recipe_document"], source="launch test")
    source = request.sources[0]
    descriptor = NativeDiagnosticCampaignRowBinding(
        campaign_uri=campaign.name,
        campaign_sha256=hashlib.sha256(campaign_bytes).hexdigest(),
        native_uri=source.uri,
        native_sha256=source.sha256,
        episode_id=original["episode_id"],
        scenario_id=original["scenario_id"],
        seed=original["seed"],
        planner_id=original["algo"],
        source_commit=original["git_hash"],
        config_identity=config_identity,
    )
    if descriptor_overrides:
        descriptor_values = descriptor.to_dict()
        descriptor_values.update(descriptor_overrides)
        descriptor = NativeDiagnosticCampaignRowBinding.from_mapping(descriptor_values)
    binding = NativeDiagnosticBinding(
        admission=admission,
        request=request,
        recipe=recipe,
        campaign_row=descriptor,
    )
    return campaign, NativeDiagnosticServiceConfig(bindings=(binding,), compute_cost=2.0)


def test_native_launch_runs_selected_campaign_row_over_http_and_rejects_stale_selection(
    admitted_native_case: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A generated BA-01 selection reaches the real diagnostic child safely."""

    campaign, native_config = _native_campaign_config(admitted_native_case)
    root = admitted_native_case["root"]
    assert isinstance(root, Path)
    binding = native_config.bindings[0]
    assert binding.campaign_row is not None
    policy = SessionPolicy(
        allowed_roots=(str(root),),
        allowed_recipes=(binding.recipe.recipe_id,),
        compute_budget=2.0,
    )
    opened = open_live_audit_workbench(
        campaign,
        source_root=root,
        store_root=tmp_path / "store",
        policy=policy,
        native_diagnostic_config=native_config,
    )
    server_thread = threading.Thread(target=opened.server.serve_forever, daemon=True)
    server_thread.start()
    try:
        with urlopen(opened.url, timeout=5) as response:
            browser_cookie = response.headers["Set-Cookie"].split(";", 1)[0]
        parsed = urlsplit(opened.url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        def post(operation: str, arguments: dict[str, object]) -> dict[str, object]:
            request = Request(
                origin + "/api/audit",
                data=json.dumps({"operation": operation, "arguments": arguments}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Origin": origin,
                    "Cookie": browser_cookie,
                },
                method="POST",
            )
            with urlopen(request, timeout=60) as response:
                return json.load(response)

        selected = post(
            "next",
            {"expected_selection_revision": 0, "operation_id": "native-launch-next"},
        )
        assert selected["status"] == "complete", selected
        selected_episode_id = selected["packet"]["primary"]["episode_id"]
        assert selected_episode_id != binding.campaign_row.episode_id

        result = post(
            "run_native_diagnostic",
            {
                "operation_id": "native-launch-positive",
                "expected_selection_revision": selected["selection_revision"],
                "expected_context_revision": selected["context"]["context_revision"],
                "intervention_id": "goal-y",
                "robot_goal": [4.0, 4.0],
                "activation_epsilon_m": 1e-9,
                "deadline_s": 30.0,
            },
        )
        assert result["status"] == "complete", result
        assert result["diagnostic_only"] is True
        assert result["scientific_claim_allowed"] is False
        assert result["control_fidelity"] == "verified"
        assert result["activation"] == "verified"

        def child_must_not_start(*_args: object, **_kwargs: object) -> None:
            pytest.fail("stale native selection reached the child runner")

        monkeypatch.setattr(native, "run_native_diagnostic", child_must_not_start)
        with pytest.raises(HTTPError, match="409"):
            post(
                "run_native_diagnostic",
                {
                    "operation_id": "native-launch-stale-selection",
                    "expected_selection_revision": selected["selection_revision"] + 1,
                    "expected_context_revision": selected["context"]["context_revision"],
                    "intervention_id": "goal-y",
                    "robot_goal": [4.0, 4.0],
                    "activation_epsilon_m": 1e-9,
                    "deadline_s": 30.0,
                },
            )
    finally:
        opened.server.shutdown()
        server_thread.join(timeout=5)
        opened.close()


def test_native_launch_rejects_stale_campaign_binding_before_server(
    admitted_native_case: dict[str, object], tmp_path: Path
) -> None:
    """A stale campaign digest fails before a browser server is returned."""

    campaign, native_config = _native_campaign_config(
        admitted_native_case, descriptor_overrides={"campaign_sha256": "0" * 64}
    )
    root = admitted_native_case["root"]
    assert isinstance(root, Path)
    binding = native_config.bindings[0]
    policy = SessionPolicy(
        allowed_roots=(str(root),),
        allowed_recipes=(binding.recipe.recipe_id,),
        compute_budget=2.0,
    )
    with pytest.raises(CapabilityUnavailable, match="campaign row binding"):
        open_live_audit_workbench(
            campaign,
            source_root=root,
            store_root=tmp_path / "store",
            policy=policy,
            native_diagnostic_config=native_config,
        )


def test_native_launch_rejects_ambiguous_campaign_rows_before_server(
    admitted_native_case: dict[str, object], tmp_path: Path
) -> None:
    """Duplicate campaign rows cannot choose an arbitrary generated ref."""

    campaign, native_config = _native_campaign_config(admitted_native_case)
    payload = json.loads(campaign.read_text(encoding="utf-8"))
    payload["episodes"].append(dict(payload["episodes"][0]))
    campaign_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    campaign.write_bytes(campaign_bytes)
    binding = native_config.bindings[0]
    assert binding.campaign_row is not None
    descriptor = dataclass_replace(
        binding.campaign_row,
        campaign_sha256=hashlib.sha256(campaign_bytes).hexdigest(),
    )
    native_config = dataclass_replace(
        native_config,
        bindings=(dataclass_replace(binding, campaign_row=descriptor),),
    )
    root = admitted_native_case["root"]
    assert isinstance(root, Path)
    policy = SessionPolicy(
        allowed_roots=(str(root),),
        allowed_recipes=(binding.recipe.recipe_id,),
        compute_budget=2.0,
    )
    with pytest.raises(AuditContextConflict, match="one EpisodeRef"):
        open_live_audit_workbench(
            campaign,
            source_root=root,
            store_root=tmp_path / "store",
            policy=policy,
            native_diagnostic_config=native_config,
        )


def test_native_launch_rejects_insufficient_compute_before_service(
    admitted_native_case: dict[str, object], tmp_path: Path
) -> None:
    """Native compute cost must fit the explicit session budget before launch."""

    campaign, native_config = _native_campaign_config(admitted_native_case)
    root = admitted_native_case["root"]
    assert isinstance(root, Path)
    binding = native_config.bindings[0]
    policy = SessionPolicy(
        allowed_roots=(str(root),),
        allowed_recipes=(binding.recipe.recipe_id,),
        compute_budget=1.0,
    )
    with pytest.raises(ValueError, match="insufficient"):
        open_live_audit_workbench(
            campaign,
            source_root=root,
            store_root=tmp_path / "store",
            policy=policy,
            native_diagnostic_config=native_config,
        )


def test_native_launch_bounds_source_preflight_errors(
    admitted_native_case: dict[str, object], tmp_path: Path
) -> None:
    """Malformed admitted source bytes become a bounded capability error."""

    campaign, native_config = _native_campaign_config(admitted_native_case)
    root = admitted_native_case["root"]
    assert isinstance(root, Path)
    original_bytes = (root / "original.json").read_bytes()
    (root / "original.json").write_text("{not-json", encoding="utf-8")
    binding = native_config.bindings[0]
    policy = SessionPolicy(
        allowed_roots=(str(root),),
        allowed_recipes=(binding.recipe.recipe_id,),
        compute_budget=2.0,
    )
    try:
        with pytest.raises(CapabilityUnavailable, match="source preflight is unavailable"):
            open_live_audit_workbench(
                campaign,
                source_root=root,
                store_root=tmp_path / "store",
                policy=policy,
                native_diagnostic_config=native_config,
            )
    finally:
        (root / "original.json").write_bytes(original_bytes)


def test_launch_scans_and_serves_one_source_bound_workbench(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    campaign = root / "campaign.json"
    campaign_payload = json.loads(FIXTURE.read_text())
    trace_payload = json.loads(TRACE_FIXTURE.read_text())
    source = trace_payload["source"]
    row = campaign_payload["episodes"][0]
    source.update(
        episode_id=row["episode_id"],
        scenario_id=row["scenario_id"],
        seed=row["seed"],
        planner_id=row["planner_id"],
    )
    trace_bytes = json.dumps(trace_payload).encode("utf-8")
    (root / "trace.json").write_bytes(trace_bytes)
    row["retained_trace_source"] = {
        "uri": "trace.json",
        "sha256": hashlib.sha256(trace_bytes).hexdigest(),
    }
    campaign.write_text(json.dumps(campaign_payload), encoding="utf-8")
    output_root = tmp_path / "trusted-output"
    opened = open_live_audit_workbench(
        campaign,
        source_root=root,
        store_root=tmp_path / "store",
        materialization_output_root=output_root,
        materialization_compute_budget=1.0,
    )
    server_thread = threading.Thread(target=opened.server.serve_forever, daemon=True)
    server_thread.start()
    try:
        assert opened.url.startswith("http://127.0.0.1:")
        extension = opened.document["extensions"]["audit_workbench"]
        assert extension["mode"] == "service_live"
        assert extension["model"]["service"]["token_transport"] == "server_only"
        assert "session_token" not in str(opened.document)
        with urlopen(opened.url, timeout=5) as response:
            html = response.read().decode("utf-8")
            assert response.status == 200
            assert "Benchmark audit workbench" in html
            assert "session_token" not in html
            assert str(root) not in html
            assert str(tmp_path / "store") not in html
            browser_cookie = response.headers["Set-Cookie"].split(";", 1)[0]
        parsed = urlsplit(opened.url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        request = Request(
            origin + "/api/audit",
            data=json.dumps(
                {
                    "operation": "next",
                    "arguments": {
                        "expected_selection_revision": 0,
                        "operation_id": "launch-next",
                    },
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": origin,
                "Cookie": browser_cookie,
            },
            method="POST",
        )
        # Scanning a retained source can exceed the browser smoke budget under
        # the full xdist lane; the server remains live and the store must not
        # be closed while this request is still draining.
        with urlopen(request, timeout=30) as response:
            selected = json.load(response)
        assert selected["status"] == "complete", selected.get("reason")
        assert selected["inspection_status"] == "complete"
        assert selected["packet"]["primary"]["episode_id"]
        assert str(root) not in json.dumps(selected)
        assert str(tmp_path / "store") not in json.dumps(selected)
        assert str(output_root) not in json.dumps(selected)
        action = Request(
            origin + "/api/audit",
            data=json.dumps(
                {
                    "operation": "materialize_selected",
                    "arguments": {
                        "operation_id": "launch-materialize",
                        "expected_selection_revision": selected["selection_revision"],
                        "expected_context_revision": selected["context"]["context_revision"],
                    },
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": origin,
                "Cookie": browser_cookie,
            },
            method="POST",
        )
        with urlopen(action, timeout=90) as response:
            materialized = json.load(response)
        assert materialized["status"] == "complete", materialized
        assert materialized["classification"] == "derived_render"
        assert materialized["fidelity"] == "unverifiable"
        assert materialized["diagnostic_only"] is True
        assert materialized["scientific_claim_allowed"] is False
        assert str(root) not in json.dumps(materialized)
        assert str(output_root) not in json.dumps(materialized)
    finally:
        opened.server.shutdown()
        server_thread.join(timeout=5)
        opened.close()


def test_launch_rejects_campaign_outside_explicit_root(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    campaign = tmp_path / "outside.json"
    shutil.copyfile(FIXTURE, campaign)
    with pytest.raises(ValueError, match="inside source_root"):
        open_live_audit_workbench(campaign, source_root=root, store_root=tmp_path / "store")


def test_launch_rejects_output_without_budget_or_policy_authority(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    campaign = root / "campaign.json"
    shutil.copyfile(FIXTURE, campaign)
    with pytest.raises(ValueError, match="trusted output root"):
        open_live_audit_workbench(
            campaign,
            source_root=root,
            store_root=tmp_path / "store",
            materialization_compute_budget=1.0,
        )
    with pytest.raises(ValueError, match="separate from source root"):
        open_live_audit_workbench(
            campaign,
            source_root=root,
            store_root=tmp_path / "store",
            materialization_output_root=root / "generated",
            materialization_compute_budget=1.0,
        )


def test_launch_opt_in_binds_fake_app_server_and_private_mcp(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    campaign = root / "campaign.json"
    shutil.copyfile(FIXTURE, campaign)
    policy = SessionPolicy(
        allowed_roots=(str(root),),
        token_budget=7,
        compute_budget=1.0,
    )
    config = CodexAppServerLaunchConfig(
        app_server=CodexAppServerConfig(
            executable=str(_fake_app_server(tmp_path)),
            cwd=root,
            request_timeout_seconds=2.0,
            startup_timeout_seconds=2.0,
            lifetime_seconds=10.0,
            close_timeout_seconds=0.5,
        )
    )
    opened = open_live_audit_workbench(
        campaign,
        source_root=root,
        store_root=tmp_path / "store",
        policy=policy,
        codex_config=config,
    )
    server_thread = threading.Thread(target=opened.server.serve_forever, daemon=True)
    server_thread.start()
    try:
        assert opened.codex_provider is not None
        assert opened.codex_capability is not None
        assert opened.codex_capability.status == "available"
        route = opened.codex_capability.choose()
        assert route is not None
        assert route.route_id == "provider-fixture:model-fixture"
        assert opened.codex_provider.config.mcp is not None
        bridge = opened.codex_provider.config.mcp.bridge
        service_session = next(iter(opened.service._sessions.values()))
        assert bridge.session_id == service_session.session_id
        assert bridge.session_token == service_session.session_token

        with urlopen(opened.url, timeout=5) as response:
            assert response.status == 200
            browser_cookie = response.headers["Set-Cookie"].split(";", 1)[0]
        parsed = urlsplit(opened.url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        def post(operation: str, arguments: dict[str, object]) -> dict[str, object]:
            request = Request(
                origin + "/api/audit",
                data=json.dumps({"operation": operation, "arguments": arguments}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Origin": origin,
                    "Cookie": browser_cookie,
                },
                method="POST",
            )
            with urlopen(request, timeout=10) as response:
                return json.load(response)

        selected = post(
            "next",
            {"expected_selection_revision": 0, "operation_id": "live-next"},
        )
        assert selected["status"] == "complete", selected
        started = post(
            "codex_start",
            {
                "prompt": "Inspect only the selected evidence.",
                "operation_id": "live-codex-start",
                "token_budget": 7,
                "compute_budget": 1.0,
            },
        )
        assert started["status"] == "complete", started
        assert started["route_id"] == route.route_id
        assert started["usage"]["total_tokens"] == 7
        assert started["usage"]["measured_compute"] == 1.0
        assert service_session.session_token not in json.dumps(started)

        with pytest.raises(HTTPError, match="403"):
            post(
                "codex_start",
                {
                    "prompt": "Inspect only the selected evidence.",
                    "operation_id": "live-codex-forged-route",
                    "token_budget": 7,
                    "compute_budget": 1.0,
                    "route_id": route.route_id,
                },
            )
        denied_prompt = post(
            "codex_start",
            {
                "prompt": service_session.session_token,
                "operation_id": "live-codex-secret-prompt",
                "token_budget": 7,
                "compute_budget": 1.0,
            },
        )
        assert denied_prompt["status"] == "denied"
    finally:
        opened.server.shutdown()
        server_thread.join(timeout=5)
        opened.close()


def test_launch_fails_closed_when_live_app_server_is_unavailable(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    campaign = root / "campaign.json"
    shutil.copyfile(FIXTURE, campaign)
    policy = SessionPolicy(
        allowed_roots=(str(root),),
        token_budget=7,
        compute_budget=1.0,
    )
    config = CodexAppServerLaunchConfig(
        app_server=CodexAppServerConfig(
            executable=str(tmp_path / "missing-codex"),
            cwd=root,
            request_timeout_seconds=1.0,
            startup_timeout_seconds=1.0,
        )
    )
    with pytest.raises(CapabilityUnavailable, match="live Codex App Server is unavailable"):
        open_live_audit_workbench(
            campaign,
            source_root=root,
            store_root=tmp_path / "store",
            policy=policy,
            codex_config=config,
        )


def test_launch_config_rejects_unpinned_codex_version() -> None:
    with pytest.raises(ValueError, match="pinned to codex-cli 0.154.0"):
        CodexAppServerLaunchConfig(app_server=CodexAppServerConfig(expected_cli_version="0.155.0"))


def _write_cli_policy(
    path: Path, root: Path, *, token_budget: int = 7, compute_budget: float = 1.0
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "audit-session-policy.v1",
                "allowed_roots": [str(root)],
                "token_budget": token_budget,
                "compute_budget": compute_budget,
            }
        ),
        encoding="utf-8",
    )


def _write_cli_codex_config(
    path: Path, *, executable: Path, cwd: Path, route_id: str | None = None
) -> None:
    payload: dict[str, object] = {
        "app_server": {
            "executable": str(executable),
            "cwd": str(cwd),
            "request_timeout_seconds": 2.0,
            "startup_timeout_seconds": 2.0,
            "lifetime_seconds": 10.0,
            "close_timeout_seconds": 0.5,
        }
    }
    if route_id is not None:
        payload["route_id"] = route_id
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_cli_native_config(path: Path, config: NativeDiagnosticServiceConfig) -> None:
    """Write one typed native configuration through its canonical projection."""

    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")


def _cli_arguments(root: Path, campaign: Path, store: Path) -> list[str]:
    return [
        "--campaign",
        str(campaign),
        "--source-root",
        str(root),
        "--store-root",
        str(store),
    ]


def test_launcher_imports_native_adapter_only_for_opt_in_preflight() -> None:
    """Ordinary launcher import must not pay the native adapter cold-start cost."""

    source = Path(launch_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    eager_native_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "robot_sf.analysis_workbench"
        and any(alias.name == "audit_native_diagnostic" for alias in node.names)
    ]
    assert eager_native_imports == []
    assert "audit_native_diagnostic as native" in source


def test_cli_native_files_reach_open_live_with_typed_config(
    admitted_native_case: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = admitted_native_case["root"]
    assert isinstance(root, Path)
    campaign, native_config = _native_campaign_config(admitted_native_case)
    config_path = tmp_path / "native.json"
    policy_path = tmp_path / "policy.json"
    _write_cli_native_config(config_path, native_config)
    _write_cli_policy(policy_path, root, compute_budget=2.0)
    captured: dict[str, object] = {}

    class _StubServer:
        server_port = 43124

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

    class _StubWorkbench:
        url = "http://127.0.0.1:43124/review-workbench.v1.html"
        server = _StubServer()

        def close(self) -> None:
            captured["closed"] = True

    def fake_open(campaign_path: Path, **kwargs: object) -> _StubWorkbench:
        captured["campaign"] = campaign_path
        captured.update(kwargs)
        return _StubWorkbench()

    monkeypatch.setattr(launch_module, "open_live_audit_workbench", fake_open)
    assert (
        launch_module.main(
            _cli_arguments(root, campaign, tmp_path / "store")
            + ["--native-diagnostic-config", str(config_path), "--session-policy", str(policy_path)]
        )
        == 0
    )
    assert captured["campaign"] == campaign
    assert isinstance(captured["policy"], SessionPolicy)
    assert isinstance(captured["native_diagnostic_config"], NativeDiagnosticServiceConfig)
    assert captured["closed"] is True
    assert "http://127.0.0.1:43124" in capsys.readouterr().out


def test_cli_requires_policy_for_native_before_open(
    admitted_native_case: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = admitted_native_case["root"]
    assert isinstance(root, Path)
    campaign, native_config = _native_campaign_config(admitted_native_case)
    config_path = tmp_path / "native.json"
    _write_cli_native_config(config_path, native_config)
    monkeypatch.setattr(
        launch_module,
        "open_live_audit_workbench",
        lambda *_args, **_kwargs: pytest.fail("native CLI without policy reached server launch"),
    )
    with pytest.raises(SystemExit) as exc_info:
        launch_module.main(
            _cli_arguments(root, campaign, tmp_path / "store")
            + ["--native-diagnostic-config", str(config_path)]
        )
    assert exc_info.value.code == 2


def test_cli_rejects_malformed_native_config_before_open(
    admitted_native_case: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = admitted_native_case["root"]
    assert isinstance(root, Path)
    campaign, _native_config = _native_campaign_config(admitted_native_case)
    config_path = tmp_path / "native.json"
    policy_path = tmp_path / "policy.json"
    config_path.write_text("{}", encoding="utf-8")
    _write_cli_policy(policy_path, root, compute_budget=2.0)
    monkeypatch.setattr(
        launch_module,
        "open_live_audit_workbench",
        lambda *_args, **_kwargs: pytest.fail("malformed native config reached server launch"),
    )
    with pytest.raises(SystemExit) as exc_info:
        launch_module.main(
            _cli_arguments(root, campaign, tmp_path / "store")
            + [
                "--native-diagnostic-config",
                str(config_path),
                "--session-policy",
                str(policy_path),
            ]
        )
    assert exc_info.value.code == 2


def test_cli_codex_files_reach_open_live_with_trusted_route_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    campaign = root / "campaign.json"
    shutil.copyfile(FIXTURE, campaign)
    config_path = tmp_path / "codex.json"
    policy_path = tmp_path / "policy.json"
    _write_cli_codex_config(config_path, executable=_fake_app_server(tmp_path), cwd=root)
    _write_cli_policy(policy_path, root)
    captured: dict[str, object] = {}

    class _StubServer:
        server_port = 43123

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

    class _StubWorkbench:
        url = "http://127.0.0.1:43123/review-workbench.v1.html"
        server = _StubServer()

        def close(self) -> None:
            captured["closed"] = True

    def fake_open(campaign_path: Path, **kwargs: object) -> _StubWorkbench:
        captured["campaign"] = campaign_path
        captured.update(kwargs)
        return _StubWorkbench()

    monkeypatch.setattr(launch_module, "open_live_audit_workbench", fake_open)
    assert (
        launch_module.main(
            _cli_arguments(root, campaign, tmp_path / "store")
            + [
                "--codex-app-server",
                str(config_path),
                "--session-policy",
                str(policy_path),
                "--codex-route-id",
                "provider-fixture:model-fixture",
            ]
        )
        == 0
    )
    assert captured["campaign"] == campaign
    assert isinstance(captured["policy"], SessionPolicy)
    codex_config = captured["codex_config"]
    assert isinstance(codex_config, CodexAppServerLaunchConfig)
    assert codex_config.route_id == "provider-fixture:model-fixture"
    assert codex_config.app_server.executable == str(tmp_path / "fake-codex")
    assert captured["closed"] is True
    assert "http://127.0.0.1:43123" in capsys.readouterr().out


def test_cli_requires_policy_for_codex_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    campaign = root / "campaign.json"
    shutil.copyfile(FIXTURE, campaign)
    config_path = tmp_path / "codex.json"
    _write_cli_codex_config(config_path, executable=_fake_app_server(tmp_path), cwd=root)
    monkeypatch.setattr(
        launch_module,
        "open_live_audit_workbench",
        lambda *_args, **_kwargs: pytest.fail("invalid CLI authority reached server launch"),
    )
    with pytest.raises(SystemExit) as exc_info:
        launch_module.main(
            _cli_arguments(root, campaign, tmp_path / "store")
            + ["--codex-app-server", str(config_path)]
        )
    assert exc_info.value.code == 2


@pytest.mark.parametrize("invalid_kind", ("config", "policy"))
def test_cli_rejects_malformed_authority_file_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_kind: str
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    campaign = root / "campaign.json"
    shutil.copyfile(FIXTURE, campaign)
    config_path = tmp_path / "codex.json"
    policy_path = tmp_path / "policy.json"
    _write_cli_codex_config(config_path, executable=_fake_app_server(tmp_path), cwd=root)
    _write_cli_policy(policy_path, root)
    invalid_path = config_path if invalid_kind == "config" else policy_path
    invalid_path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(
        launch_module,
        "open_live_audit_workbench",
        lambda *_args, **_kwargs: pytest.fail("invalid CLI authority reached server launch"),
    )
    with pytest.raises(SystemExit) as exc_info:
        launch_module.main(
            _cli_arguments(root, campaign, tmp_path / "store")
            + [
                "--codex-app-server",
                str(config_path),
                "--session-policy",
                str(policy_path),
            ]
        )
    assert exc_info.value.code == 2


@pytest.mark.parametrize("capability_kind", ("missing", "version"))
def test_cli_rejects_missing_or_mismatched_installed_capability(
    tmp_path: Path, capability_kind: str, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    campaign = root / "campaign.json"
    shutil.copyfile(FIXTURE, campaign)
    executable = tmp_path / "missing-codex"
    if capability_kind == "version":
        executable = _fake_app_server(tmp_path)
        executable.write_text(
            executable.read_text(encoding="utf-8").replace("0.154.0", "0.153.0"),
            encoding="utf-8",
        )
        executable.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    config_path = tmp_path / "codex.json"
    policy_path = tmp_path / "policy.json"
    _write_cli_codex_config(config_path, executable=executable, cwd=root)
    _write_cli_policy(policy_path, root)
    with pytest.raises(SystemExit) as exc_info:
        launch_module.main(
            _cli_arguments(root, campaign, tmp_path / "store")
            + [
                "--codex-app-server",
                str(config_path),
                "--session-policy",
                str(policy_path),
            ]
        )
    assert exc_info.value.code == 2
    assert "launch denied" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("token_budget", "compute_budget"),
    ((0, 1.0), (7, 0.0)),
)
def test_cli_rejects_insufficient_live_budgets_before_server(
    tmp_path: Path,
    token_budget: int,
    compute_budget: float,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    campaign = root / "campaign.json"
    shutil.copyfile(FIXTURE, campaign)
    config_path = tmp_path / "codex.json"
    policy_path = tmp_path / "policy.json"
    _write_cli_codex_config(config_path, executable=_fake_app_server(tmp_path), cwd=root)
    _write_cli_policy(
        policy_path,
        root,
        token_budget=token_budget,
        compute_budget=compute_budget,
    )
    with pytest.raises(SystemExit) as exc_info:
        launch_module.main(
            _cli_arguments(root, campaign, tmp_path / "store")
            + [
                "--codex-app-server",
                str(config_path),
                "--session-policy",
                str(policy_path),
            ]
        )
    assert exc_info.value.code == 2
    assert "live Codex launch requires" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("malformed_kind", "expected_error"),
    (
        ("env", "env must be a JSON object"),
        ("timeout", "Codex App Server config is invalid"),
        ("policy_compute", "session policy is invalid"),
    ),
)
def test_cli_rejects_malformed_typed_authority_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    malformed_kind: str,
    expected_error: str,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    campaign = root / "campaign.json"
    shutil.copyfile(FIXTURE, campaign)
    config_path = tmp_path / "codex.json"
    policy_path = tmp_path / "policy.json"
    _write_cli_codex_config(config_path, executable=_fake_app_server(tmp_path), cwd=root)
    _write_cli_policy(policy_path, root)
    if malformed_kind == "env":
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        config_payload["app_server"]["env"] = []
        config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    elif malformed_kind == "timeout":
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        config_payload["app_server"]["request_timeout_seconds"] = 10**1000
        config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    else:
        policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
        policy_payload["compute_budget"] = 10**1000
        policy_path.write_text(json.dumps(policy_payload), encoding="utf-8")
    monkeypatch.setattr(
        launch_module,
        "open_live_audit_workbench",
        lambda *_args, **_kwargs: pytest.fail("malformed typed authority reached server launch"),
    )
    with pytest.raises(SystemExit) as exc_info:
        launch_module.main(
            _cli_arguments(root, campaign, tmp_path / "store")
            + [
                "--codex-app-server",
                str(config_path),
                "--session-policy",
                str(policy_path),
            ]
        )
    assert exc_info.value.code == 2
    assert expected_error in capsys.readouterr().err
