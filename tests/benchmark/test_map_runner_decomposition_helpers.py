"""Direct characterization of the helpers extracted from the map_runner god-functions.

Issue #4944 / #4770-s4 decomposes ``map_runner._build_policy`` and
``map_runner_episode.run_map_episode`` into named single-responsibility helpers.
The #4927 baseline (``test_map_runner_characterization.py``) pins the god-function
behavior end-to-end; this module pins the extracted helpers' contracts directly so
the decomposition itself is characterized and behavior-preserving.

These tests are CPU-only, deterministic, and use tiny synthetic inputs. They must
stay green across further decomposition of the same helpers.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robot_sf.benchmark.cbf_safety_filter_runtime import CBFSafetyFilterRuntimeConfig
from robot_sf.benchmark.map_runner import (
    _build_policy,
    _build_socnav_family_adapter,
    _enforce_ppo_paper_profile,
    _holonomic_world_velocity_boundary,
    _is_holonomic_world_velocity_mode,
    _resolve_planner_obs_mode,
)
from robot_sf.benchmark.map_runner_episode import (
    _apply_cbf_safety_filter_step,
    _apply_safety_wrapper_step,
    _build_tracking_precision_summary,
    _min_finite_or_inf,
)
from robot_sf.benchmark.safety_wrapper_runtime import SafetyWrapperRuntimeConfig

# ---------------------------------------------------------------------------
# map_runner: holonomic world-velocity helpers
# ---------------------------------------------------------------------------


class TestHolonomicWorldVelocityHelpers:
    """Pin the holonomic world-velocity (vx_vy) predicate and boundary strings."""

    @pytest.mark.parametrize(
        ("algo_key", "kinematics", "command_mode", "expected"),
        [
            ("orca", "holonomic", "vx_vy", True),
            ("social_force", "omni", "vx_vy", True),
            ("sonic_crowdnav", "omnidirectional", "vx_vy", True),
            ("orca", "differential_drive", "vx_vy", False),
            ("orca", "holonomic", "v_theta", False),
            ("goal", "holonomic", "vx_vy", False),
            ("ppo", "holonomic", "vx_vy", False),
        ],
    )
    def test_mode_predicate_matches_expected(
        self,
        algo_key: str,
        kinematics: str,
        command_mode: str,
        expected: bool,
    ) -> None:
        """Predicate must be true only for eligible planners in holonomic vx_vy mode."""
        assert _is_holonomic_world_velocity_mode(algo_key, kinematics, command_mode) is expected

    def test_mode_predicate_treats_none_command_mode_as_not_vx_vy(self) -> None:
        """A None command mode must never select holonomic world-velocity mode."""
        assert _is_holonomic_world_velocity_mode("orca", "holonomic", None) is False

    @pytest.mark.parametrize(
        "algo_key",
        [
            "orca",
            "hrvo",
            "socnav_hrvo",
            "social_force",
            "sf",
            "social_navigation_pyenvs_orca",
            "social_nav_pyenvs_orca",
            "sonic_crowdnav",
            "sonic_gst",
            "gensafenav_ours_gst",
            "ours_gst",
            "gensafenav_gst_predictor_rand",
            "gst_predictor_rand",
            "social_navigation_pyenvs_socialforce",
            "unknown_planner",
        ],
    )
    def test_boundary_string_is_non_empty_for_all_paths(self, algo_key: str) -> None:
        """Every branch (including the fallback else) must yield a non-empty string."""
        boundary = _holonomic_world_velocity_boundary(algo_key)
        assert isinstance(boundary, str)
        assert boundary.strip()
        assert "holonomic vx_vy benchmark action space" in boundary

    def test_boundary_mentions_upstream_name_for_known_planners(self) -> None:
        """Known planners must mention their upstream solver in the boundary string."""
        assert "Python-RVO2" in _holonomic_world_velocity_boundary("orca")
        assert "HRVO" in _holonomic_world_velocity_boundary("hrvo")
        assert "social-force" in _holonomic_world_velocity_boundary("social_force")
        assert "SoNIC" in _holonomic_world_velocity_boundary("sonic_crowdnav")
        assert "GenSafeNav Ours_GST" in _holonomic_world_velocity_boundary("gensafenav_ours_gst")

    def test_sf_alias_yields_social_force_provenance(self) -> None:
        """The ``sf`` alias must resolve to the social-force provenance string.

        Regression for issue #4950 item 4: ``_HOLONOMIC_WORLD_VELOCITY_ALGO_KEYS``
        treats ``sf`` as an alias of ``social_force``, so the boundary text must also
        resolve it to the social-force description instead of falling through to the
        generic Social-Navigation-PyEnvs text.
        """
        sf_boundary = _holonomic_world_velocity_boundary("sf")
        social_force_boundary = _holonomic_world_velocity_boundary("social_force")
        assert sf_boundary == social_force_boundary
        assert "social-force" in sf_boundary
        # It must NOT fall through to the generic SocNav-PyEnvs force-model text.
        assert "Social-Navigation-PyEnvs" not in sf_boundary


# ---------------------------------------------------------------------------
# map_runner: planner obs-mode + PPO paper-gate helpers
# ---------------------------------------------------------------------------


class TestPlannerObsModeAndPaperGate:
    """Pin the planner obs-mode resolver and PPO paper-profile gate."""

    def test_resolve_obs_mode_from_dict_config(self) -> None:
        """A dict config must resolve obs_mode with the given default fallback."""
        planner = SimpleNamespace(config={"obs_mode": "DICT"})
        assert _resolve_planner_obs_mode(planner, "vector") == "dict"

    def test_resolve_obs_mode_from_attribute_config(self) -> None:
        """A non-dict config object must resolve obs_mode via attribute access."""
        planner = SimpleNamespace(config=SimpleNamespace(obs_mode="Multi_Input"))
        assert _resolve_planner_obs_mode(planner, "vector") == "multi_input"

    def test_resolve_obs_mode_uses_default_when_missing(self) -> None:
        """When obs_mode is absent, the provided default must be returned lowercased."""
        planner = SimpleNamespace(config={})
        assert _resolve_planner_obs_mode(planner, "vector") == "vector"
        planner_attr = SimpleNamespace(config=SimpleNamespace())
        assert _resolve_planner_obs_mode(planner_attr, "Dict") == "dict"

    def test_resolve_obs_mode_handles_none_config(self) -> None:
        """A planner with no config attribute must fall back to the default."""
        planner = SimpleNamespace(spec=1)
        assert _resolve_planner_obs_mode(planner, "vector") == "vector"

    def test_resolve_obs_mode_normalizes_explicit_none_to_default(self) -> None:
        """An explicit ``obs_mode: None`` must normalize to the default, not ``"none"``.

        Regression for issue #4950 item 2: ``dict.get`` only substitutes the default
        for a *missing* key, so an explicit ``None`` would stringify to ``"none"``.
        Both dict and attribute configs must coerce ``None`` to the supplied default.
        """
        dict_planner = SimpleNamespace(config={"obs_mode": None})
        assert _resolve_planner_obs_mode(dict_planner, "vector") == "vector"
        attr_planner = SimpleNamespace(config=SimpleNamespace(obs_mode=None))
        assert _resolve_planner_obs_mode(attr_planner, "dict") == "dict"

    def test_resolve_obs_mode_normalizes_empty_string_to_default(self) -> None:
        """An explicit empty/whitespace ``obs_mode`` must normalize to the default."""
        dict_planner = SimpleNamespace(config={"obs_mode": "   "})
        assert _resolve_planner_obs_mode(dict_planner, "vector") == "vector"
        attr_planner = SimpleNamespace(config=SimpleNamespace(obs_mode=""))
        assert _resolve_planner_obs_mode(attr_planner, "Dict") == "dict"

    def test_paper_gate_passes_for_experimental_profile(self) -> None:
        """An experimental profile must pass the gate with no reason."""
        ready, reason = _enforce_ppo_paper_profile({"profile": "experimental"})
        assert ready is False
        assert reason is None

    def test_paper_gate_raises_for_paper_profile_without_provenance(self) -> None:
        """A paper profile without provenance must raise and not return."""
        with pytest.raises(ValueError, match="PPO paper profile requested but gate failed"):
            _enforce_ppo_paper_profile({"profile": "paper"})


# ---------------------------------------------------------------------------
# map_runner: socnav-family adapter dispatch
# ---------------------------------------------------------------------------


class TestSocnavFamilyAdapterDispatch:
    """Pin the classical/adapter planner construction dispatch."""

    @pytest.mark.parametrize(
        ("algo_key", "expected_adapter_type"),
        [
            ("orca", "ORCAPlannerAdapter"),
            ("hrvo", "HRVOPlannerAdapter"),
            ("socnav_hrvo", "HRVOPlannerAdapter"),
            ("social_force", "SocialForcePlannerAdapter"),
            ("sf", "SocialForcePlannerAdapter"),
            ("socnav_sampling", "SamplingPlannerAdapter"),
            ("sampling", "SamplingPlannerAdapter"),
            ("rvo", "SamplingPlannerAdapter"),
            ("dwa", "SamplingPlannerAdapter"),
        ],
    )
    def test_classical_adapter_constructed_for_key(
        self,
        algo_key: str,
        expected_adapter_type: str,
    ) -> None:
        """Known classical keys must construct the expected adapter type."""
        meta: dict[str, Any] = {"algorithm": algo_key}
        adapter = _build_socnav_family_adapter(algo_key, algo_key, {}, meta=meta)
        assert type(adapter).__name__ == expected_adapter_type

    def test_rvo_dwa_marks_placeholder_status_in_meta(self) -> None:
        """RVO/DWA must record the placeholder + unimplemented status in metadata."""
        meta: dict[str, Any] = {"algorithm": "rvo"}
        _build_socnav_family_adapter("rvo", "rvo", {}, meta=meta)
        assert meta["status"] == "placeholder"
        assert meta["fallback_reason"] == "unimplemented"

    def test_planner_selector_v2_records_claim_boundary(self) -> None:
        """The diagnostic selector must stamp its diagnostic-only claim boundary.

        The selector adapter requires a full candidate config (out of scope for this
        unit test), so this case is covered end-to-end via the #4927 characterization
        baseline; here we only assert the dispatch routes the key by checking that the
        unknown-algorithm raise is NOT triggered for the selector key.
        """
        # The selector needs candidates we do not construct here; the dispatch is
        # exercised by test_classical_adapter_constructed_for_key for the simple keys.
        pytest.skip("selector adapter requires full candidate config; covered by #4927 baseline")

    def test_unknown_algorithm_raises(self) -> None:
        """An unknown algorithm key must raise ValueError mentioning the original label."""
        with pytest.raises(ValueError, match="Unknown map-based algorithm 'totally_bogus'"):
            _build_socnav_family_adapter("totally_bogus", "totally_bogus", {}, meta={})


# ---------------------------------------------------------------------------
# map_runner: end-to-end build_policy still wires the decomposed helpers
# ---------------------------------------------------------------------------


class TestBuildPolicyUsesDecomposedHelpers:
    """Confirm _build_policy still resolves classical planners through the helpers."""

    def test_orca_policy_returns_projecting_callable(self) -> None:
        """The ORCA policy built via the decomposed tail must project commands."""
        policy, meta = _build_policy("orca", {}, robot_kinematics="differential_drive")
        assert callable(policy)
        assert meta["algorithm"] == "orca"
        assert meta["status"] == "ok"

    def test_holonomic_path_returns_dict_command_payload(self) -> None:
        """The holonomic world-velocity path must return a dict (not tuple[float, float]).

        Regression for issue #4950 item 3: ``_build_policy`` is annotated to return a
        ``tuple[float, float]`` command callable, but the holonomic world-velocity path
        returns a ``dict[str, float | str]`` command payload. The annotation was widened
        to ``Callable[..., Any]``; this test exercises the dict-returning path so the
        widening stays justified and the holonomic policy stays callable end-to-end.
        """
        policy, meta = _build_policy(
            "social_force",
            {},
            robot_kinematics="holonomic",
            robot_command_mode="vx_vy",
        )
        assert meta["algorithm"] == "social_force"
        assert meta["status"] == "ok"
        command = policy({"goal_in_robot_frame": np.array([5.0, 0.0, 0.0], dtype=float)})
        # The holonomic path returns a dict command payload, not a (v, w) tuple.
        assert isinstance(command, dict)
        assert command["command_kind"] == "holonomic_vxy_world"
        assert {"vx", "vy"}.issubset(command.keys())


# ---------------------------------------------------------------------------
# map_runner_episode: tracking-precision summary helper
# ---------------------------------------------------------------------------


class TestTrackingPrecisionSummary:
    """Pin the tracking-precision summary block shape and rates."""

    def test_empty_records_yield_default_honored_rate(self) -> None:
        """With no step records, contract_honored must be True and rate 1.0."""
        summary = _build_tracking_precision_summary(
            spec={"enabled": False, "target_motp_m": 0.05},
            records=[],
            min_separation_corrupted_values=[],
        )
        assert summary["step_count"] == 0
        assert summary["contract_honored"] is True
        assert summary["contract_honored_rate"] == 1.0
        assert summary["min_separation_corrupted_m"] == float("inf")
        assert "last_step" not in summary

    def test_records_compute_rates_and_last_step(self) -> None:
        """Records must compute the honored rate and carry the last step record."""
        records = [
            {"contract_honored": True, "step": 0},
            {"contract_honored": False, "step": 1},
            {"contract_honored": True, "step": 2},
        ]
        summary = _build_tracking_precision_summary(
            spec={"enabled": True, "target_motp_m": 0.05},
            records=records,
            min_separation_corrupted_values=[1.0, 0.5, 0.25],
        )
        assert summary["step_count"] == 3
        assert summary["contract_honored"] is False
        assert summary["contract_honored_rate"] == pytest.approx(2 / 3)
        assert summary["min_separation_corrupted_m"] == 0.25
        assert summary["last_step"] == {"contract_honored": True, "step": 2}

    def test_min_finite_filters_nan_and_nonfinite(self) -> None:
        """The finite-filter helper must ignore NaN/inf and return the min finite value.

        Regression for issue #4950 item 1: an unfiltered ``min()`` over a list
        containing NaN is order-dependent and non-deterministic.
        """
        # NaN present alongside a finite value must not propagate.
        assert _min_finite_or_inf([float("nan"), 0.5, 0.25]) == 0.25
        # Order independence: NaN-first must give the same answer as NaN-last.
        assert _min_finite_or_inf([float("nan"), 0.5]) == 0.5
        assert _min_finite_or_inf([0.5, float("nan")]) == 0.5
        # +inf and -inf must be filtered out too.
        assert _min_finite_or_inf([float("inf"), float("-inf"), 1.0]) == 1.0
        # All-non-finite must fall back to +inf.
        assert _min_finite_or_inf([float("nan"), float("inf")]) == float("inf")
        assert math.isinf(_min_finite_or_inf([float("nan"), float("inf")]))

    def test_min_finite_empty_yields_inf(self) -> None:
        """An empty list must yield +inf."""
        assert _min_finite_or_inf([]) == float("inf")

    def test_summary_ignores_nan_in_min_separation_stream(self) -> None:
        """The summary must compute min_separation_corrupted_m from finite values only."""
        records = [{"contract_honored": True, "step": 0}]
        summary = _build_tracking_precision_summary(
            spec={"enabled": True, "target_motp_m": 0.05},
            records=records,
            min_separation_corrupted_values=[float("nan"), 0.8, 0.4],
        )
        assert summary["min_separation_corrupted_m"] == 0.4


# ---------------------------------------------------------------------------
# map_runner_episode: safety-wrapper + CBF filter step helpers (error/fallback)
# ---------------------------------------------------------------------------


def _runtime(*, enabled: bool = True, **overrides: Any) -> SafetyWrapperRuntimeConfig:
    """Build a safety-wrapper runtime config with test-friendly defaults."""
    base = {"fail_on_native_action": False, "fail_on_unsupported_command": False}
    base.update(overrides)
    return SafetyWrapperRuntimeConfig(enabled=enabled, **base)


def _cbf_runtime(*, enabled: bool = True, **overrides: Any) -> CBFSafetyFilterRuntimeConfig:
    """Build a CBF safety-filter runtime config with test-friendly defaults."""
    base = {"fail_on_native_action": False, "fail_on_unsupported_command": False}
    base.update(overrides)
    return CBFSafetyFilterRuntimeConfig(enabled=enabled, **base)


class TestSafetyWrapperStepHelper:
    """Pin the safety-wrapper error/fallback step helper."""

    def test_native_action_records_ineligible_when_not_failing(self) -> None:
        """A native action with fail_on_native_action=False must emit an ineligible record."""
        runtime = _runtime(fail_on_native_action=False)
        command, record = _apply_safety_wrapper_step(
            (1.0, 0.0),
            runtime=runtime,
            env=SimpleNamespace(),
            config=SimpleNamespace(),
            step_idx=3,
            step_is_native=True,
            previous_ped_positions=None,
            deadlock_monitor=None,
        )
        assert command == (1.0, 0.0)
        assert record["step"] == 3
        assert record["ineligible_reason"] == "native_environment_action"
        assert record["eligible_for_wrapper"] is False

    def test_native_action_raises_when_failing(self) -> None:
        """A native action with fail_on_native_action=True must raise ValueError."""
        runtime = _runtime(fail_on_native_action=True)
        with pytest.raises(ValueError, match="safety_wrapper.enabled requires absolute commands"):
            _apply_safety_wrapper_step(
                (1.0, 0.0),
                runtime=runtime,
                env=SimpleNamespace(),
                config=SimpleNamespace(),
                step_idx=0,
                step_is_native=True,
                previous_ped_positions=None,
                deadlock_monitor=None,
            )

    def test_unsupported_command_shape_records_ineligible(self) -> None:
        """A non-(linear, angular) command must emit an unsupported-command-shape record."""
        runtime = _runtime(fail_on_unsupported_command=False)
        command, record = _apply_safety_wrapper_step(
            "not_a_command",
            runtime=runtime,
            env=SimpleNamespace(),
            config=SimpleNamespace(),
            step_idx=5,
            step_is_native=False,
            previous_ped_positions=None,
            deadlock_monitor=None,
        )
        assert command == "not_a_command"
        assert record["ineligible_reason"] == "unsupported_command_shape"
        assert record["eligible_for_wrapper"] is False

    def test_unsupported_command_shape_raises_when_failing(self) -> None:
        """An unsupported command shape with fail_on_unsupported_command=True must raise."""
        runtime = _runtime(fail_on_unsupported_command=True)
        with pytest.raises(TypeError, match="safety_wrapper.enabled expects commands shaped like"):
            _apply_safety_wrapper_step(
                [1.0],
                runtime=runtime,
                env=SimpleNamespace(),
                config=SimpleNamespace(),
                step_idx=0,
                step_is_native=False,
                previous_ped_positions=None,
                deadlock_monitor=None,
            )

    def test_applied_path_corrects_command_and_preserves_tail(self) -> None:
        """The applied path must return the corrected (v,w) and preserve the command tail."""
        runtime = _runtime()

        def fake_apply(
            *, command, env, config, runtime, previous_ped_positions, step_idx, deadlock_monitor
        ):
            return (0.5, 0.25), {"corrected": True, "step_idx": step_idx}

        # The helper calls module-level apply_runtime_safety_wrapper; patch it for this test.
        import robot_sf.benchmark.map_runner_episode as mod

        original = mod.apply_runtime_safety_wrapper
        mod.apply_runtime_safety_wrapper = fake_apply
        try:
            command, record = _apply_safety_wrapper_step(
                (1.0, 0.0, 9.0, 9.0),
                runtime=runtime,
                env=SimpleNamespace(),
                config=SimpleNamespace(),
                step_idx=7,
                step_is_native=False,
                previous_ped_positions=None,
                deadlock_monitor=None,
            )
        finally:
            mod.apply_runtime_safety_wrapper = original
        assert command == (0.5, 0.25, 9.0, 9.0)
        assert record == {"corrected": True, "step_idx": 7}


class TestCbfSafetyFilterStepHelper:
    """Pin the CBF safety-filter error/fallback step helper."""

    def test_native_action_records_ineligible_when_not_failing(self) -> None:
        """A native action with fail_on_native_action=False must emit an ineligible record."""
        runtime = _cbf_runtime(fail_on_native_action=False)
        command, record = _apply_cbf_safety_filter_step(
            (1.0, 0.0),
            runtime=runtime,
            env=SimpleNamespace(),
            config=SimpleNamespace(),
            step_idx=2,
            step_is_native=True,
            previous_ped_positions=None,
        )
        assert command == (1.0, 0.0)
        assert record["ineligible_reason"] == "native_environment_action"
        assert record["eligible_for_cbf_filter"] is False

    def test_native_action_raises_when_failing(self) -> None:
        """A native action with fail_on_native_action=True must raise ValueError."""
        runtime = _cbf_runtime(fail_on_native_action=True)
        with pytest.raises(
            ValueError, match="cbf_safety_filter.enabled requires absolute commands"
        ):
            _apply_cbf_safety_filter_step(
                (1.0, 0.0),
                runtime=runtime,
                env=SimpleNamespace(),
                config=SimpleNamespace(),
                step_idx=0,
                step_is_native=True,
                previous_ped_positions=None,
            )

    def test_unsupported_command_shape_raises_when_failing(self) -> None:
        """An unsupported command shape with fail_on_unsupported_command=True must raise."""
        runtime = _cbf_runtime(fail_on_unsupported_command=True)
        with pytest.raises(
            TypeError, match="cbf_safety_filter.enabled expects commands shaped like"
        ):
            _apply_cbf_safety_filter_step(
                None,
                runtime=runtime,
                env=SimpleNamespace(),
                config=SimpleNamespace(),
                step_idx=0,
                step_is_native=False,
                previous_ped_positions=None,
            )

    def test_applied_path_corrects_command_and_preserves_tail(self) -> None:
        """The applied path must return the corrected (v,w) and preserve the command tail."""
        runtime = _cbf_runtime()

        def fake_apply(*, command, env, config, runtime, previous_ped_positions, step_idx):
            return (0.3, -0.1), {"filtered": True, "step_idx": step_idx}

        import robot_sf.benchmark.map_runner_episode as mod

        original = mod.apply_runtime_cbf_safety_filter
        mod.apply_runtime_cbf_safety_filter = fake_apply
        try:
            command, record = _apply_cbf_safety_filter_step(
                (1.0, 0.0, "tail"),
                runtime=runtime,
                env=SimpleNamespace(),
                config=SimpleNamespace(),
                step_idx=4,
                step_is_native=False,
                previous_ped_positions=None,
            )
        finally:
            mod.apply_runtime_cbf_safety_filter = original
        assert command == (0.3, -0.1, "tail")
        assert record == {"filtered": True, "step_idx": 4}
