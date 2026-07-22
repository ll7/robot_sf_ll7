"""Direct symbol-level tests for the legacy camera-ready campaign facade.

These tests exercise the facade module's own defined symbols (the compatibility aliases and the
thin delegation wrappers in ``robot_sf.benchmark.camera_ready._legacy_campaign_facade``) without
invoking the real campaign, preflight, or route-clearance implementations. The heavy collaborators
are replaced with fakes via ``monkeypatch`` so the wrappers' delegation and global-rebind behavior
is observed in isolation.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

import robot_sf.benchmark.camera_ready._legacy_campaign_facade as camera_ready_legacy_facade
import robot_sf.benchmark.camera_ready._route_clearance as route_clearance_module
from robot_sf.benchmark.camera_ready._util import _kinematics_matrix_or_default

if TYPE_CHECKING:
    from pathlib import Path

# Sentinel ``cfg`` object. The facade wrappers do not introspect ``cfg`` before delegating to the
# (faked) implementations, so a plain sentinel is sufficient and avoids building a real
# CampaignConfig.
_SENTINEL_CFG = SimpleNamespace(label="sentinel-cfg")


def test_normalized_kinematics_matrix_is_util_alias() -> None:
    """The facade kinematics alias is object-identical to the _util default resolver."""
    assert camera_ready_legacy_facade._normalized_kinematics_matrix is _kinematics_matrix_or_default


def test_build_route_clearance_warnings_delegates_and_restores_convert_map_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The warning builder delegates to the impl while rebinding, then restores the global.

    During the delegated call the ``_route_clearance`` module global ``convert_map`` must point at
    the facade's own ``convert_map``; afterward it must be restored to its prior value. Because the
    facade and the route-clearance module both import ``convert_map`` from the same source, the
    test first swaps the route-clearance global for a distinct sentinel to make the rebind
    observable, then asserts it is restored on the success path.
    """
    prior_convert_map = object()
    monkeypatch.setattr(route_clearance_module, "convert_map", prior_convert_map)

    captured: dict[str, object] = {}

    def _fake_impl(scenarios, *, certifications=None, margin_warn_threshold_m):
        captured["scenarios"] = scenarios
        captured["certifications"] = certifications
        captured["margin_warn_threshold_m"] = margin_warn_threshold_m
        captured["convert_map_during_call"] = route_clearance_module.convert_map
        return [{"scenario": "delegate-sentinel"}]

    monkeypatch.setattr(
        camera_ready_legacy_facade, "_build_route_clearance_warnings_impl", _fake_impl
    )

    scenarios = [{"name": "smoke"}]
    result = camera_ready_legacy_facade._build_route_clearance_warnings(
        scenarios,
        certifications={"smoke": {"status": "certified_stress_geometry"}},
        margin_warn_threshold_m=0.25,
    )

    # The impl was invoked and its return value forwarded unchanged.
    assert result == [{"scenario": "delegate-sentinel"}]
    assert captured["scenarios"] is scenarios
    assert captured["certifications"] == {"smoke": {"status": "certified_stress_geometry"}}
    assert captured["margin_warn_threshold_m"] == 0.25
    # While the impl ran, the route-clearance global was rebound to the facade's convert_map.
    assert captured["convert_map_during_call"] is camera_ready_legacy_facade.convert_map
    assert captured["convert_map_during_call"] is not prior_convert_map
    # After the call, the global is restored to its prior value.
    assert route_clearance_module.convert_map is prior_convert_map


def test_build_route_clearance_warnings_restores_convert_map_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The route-clearance global is restored even when the impl raises."""
    prior_convert_map = object()
    monkeypatch.setattr(route_clearance_module, "convert_map", prior_convert_map)

    captured: dict[str, object] = {}

    class _SentinelRouteClearanceError(RuntimeError):
        """Distinct exception type used only to observe the finally restore."""

    def _raising_impl(scenarios, *, certifications=None, margin_warn_threshold_m):
        captured["convert_map_during_call"] = route_clearance_module.convert_map
        raise _SentinelRouteClearanceError("impl failure")

    monkeypatch.setattr(
        camera_ready_legacy_facade, "_build_route_clearance_warnings_impl", _raising_impl
    )

    with pytest.raises(_SentinelRouteClearanceError, match="impl failure"):
        camera_ready_legacy_facade._build_route_clearance_warnings([{"name": "smoke"}])

    # The rebind happened before the raise, then the finally block restored the global.
    assert captured["convert_map_during_call"] is camera_ready_legacy_facade.convert_map
    assert route_clearance_module.convert_map is prior_convert_map


def test_prepare_campaign_preflight_delegates_with_injected_callables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """prepare_campaign_preflight injects _validate_campaign_config and the facade warning builder."""
    captured: dict[str, object] = {}

    def _fake_impl(  # noqa: PLR0913
        cfg,
        *,
        output_root,
        label,
        campaign_id,
        invoked_command,
        validate_campaign_config,
        build_route_clearance_warnings,
        checkpoint_preflight_mode,
        checkpoint_cache_dir,
        checkpoint_registry_path,
    ):
        captured["cfg"] = cfg
        captured["output_root"] = output_root
        captured["label"] = label
        captured["campaign_id"] = campaign_id
        captured["invoked_command"] = invoked_command
        captured["validate_campaign_config"] = validate_campaign_config
        captured["build_route_clearance_warnings"] = build_route_clearance_warnings
        captured["checkpoint_preflight_mode"] = checkpoint_preflight_mode
        captured["checkpoint_cache_dir"] = checkpoint_cache_dir
        captured["checkpoint_registry_path"] = checkpoint_registry_path
        return {"preflight": "delegate-sentinel"}

    monkeypatch.setattr(camera_ready_legacy_facade, "_prepare_campaign_preflight_impl", _fake_impl)

    cache_dir = tmp_path / "cache"
    registry_path = tmp_path / "registry.json"
    result = camera_ready_legacy_facade.prepare_campaign_preflight(
        _SENTINEL_CFG,
        output_root=tmp_path / "out",
        label="issue6079",
        campaign_id="campaign-abc",
        invoked_command="pytest -q",
        checkpoint_preflight_mode="enforced_staged",
        checkpoint_cache_dir=cache_dir,
        checkpoint_registry_path=registry_path,
    )

    assert result == {"preflight": "delegate-sentinel"}
    assert captured["cfg"] is _SENTINEL_CFG
    assert captured["output_root"] == tmp_path / "out"
    assert captured["label"] == "issue6079"
    assert captured["campaign_id"] == "campaign-abc"
    assert captured["invoked_command"] == "pytest -q"
    assert captured["checkpoint_preflight_mode"] == "enforced_staged"
    assert captured["checkpoint_cache_dir"] == cache_dir
    assert captured["checkpoint_registry_path"] == registry_path
    # Injected callables are the facade's own bindings, not freshly imported copies.
    assert (
        captured["validate_campaign_config"] is camera_ready_legacy_facade._validate_campaign_config
    )
    assert (
        captured["build_route_clearance_warnings"]
        is camera_ready_legacy_facade._build_route_clearance_warnings
    )


def test_run_campaign_delegates_with_injected_callables_and_forwards_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """run_campaign injects the four collaborators and forwards the campaign kwargs."""
    captured: dict[str, object] = {}

    def _fake_impl(  # noqa: PLR0913
        cfg,
        *,
        output_root,
        label,
        campaign_id,
        skip_publication_bundle,
        invoked_command,
        prepare_campaign_preflight,
        run_batch,
        compute_aggregates_with_ci,
        export_publication_bundle,
        arm_isolation,
    ):
        captured["cfg"] = cfg
        captured["output_root"] = output_root
        captured["label"] = label
        captured["campaign_id"] = campaign_id
        captured["skip_publication_bundle"] = skip_publication_bundle
        captured["invoked_command"] = invoked_command
        captured["prepare_campaign_preflight"] = prepare_campaign_preflight
        captured["run_batch"] = run_batch
        captured["compute_aggregates_with_ci"] = compute_aggregates_with_ci
        captured["export_publication_bundle"] = export_publication_bundle
        captured["arm_isolation"] = arm_isolation
        return {"campaign": "delegate-sentinel"}

    monkeypatch.setattr(camera_ready_legacy_facade, "_run_campaign_impl", _fake_impl)

    result = camera_ready_legacy_facade.run_campaign(
        _SENTINEL_CFG,
        output_root=tmp_path / "campaign_out",
        label="issue6079",
        campaign_id="campaign-abc",
        skip_publication_bundle=True,
        invoked_command="python -m robot_sf ...",
        arm_isolation="subprocess",
    )

    assert result == {"campaign": "delegate-sentinel"}
    assert captured["cfg"] is _SENTINEL_CFG
    assert captured["output_root"] == tmp_path / "campaign_out"
    assert captured["label"] == "issue6079"
    assert captured["campaign_id"] == "campaign-abc"
    assert captured["skip_publication_bundle"] is True
    assert captured["invoked_command"] == "python -m robot_sf ..."
    assert captured["arm_isolation"] == "subprocess"
    # Injected collaborators are the facade's own module-level bindings.
    assert (
        captured["prepare_campaign_preflight"]
        is camera_ready_legacy_facade.prepare_campaign_preflight
    )
    assert captured["run_batch"] is camera_ready_legacy_facade.run_batch
    assert (
        captured["compute_aggregates_with_ci"]
        is camera_ready_legacy_facade.compute_aggregates_with_ci
    )
    assert (
        captured["export_publication_bundle"]
        is camera_ready_legacy_facade.export_publication_bundle
    )


def test_all_public_names_are_importable_attributes() -> None:
    """Every name in __all__ resolves to an attribute of the facade module."""
    for name in camera_ready_legacy_facade.__all__:
        assert hasattr(camera_ready_legacy_facade, name), name
        getattr(camera_ready_legacy_facade, name)
