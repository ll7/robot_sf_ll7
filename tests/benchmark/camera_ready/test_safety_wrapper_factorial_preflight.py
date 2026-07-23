"""CPU preflight for the issue #4830 safety-wrapper paired factorial campaign.

Issue #4830 derives a runnable camera-ready campaign from the
``configs/research/safety_wrapper_ablation_v1.yaml`` design contract. That derivation requires
the per-arm ``safety_wrapper`` field to be wired through the full campaign -> subprocess ->
worker -> ``run_batch`` -> runtime path, because on current main
``CampaignConfig``/``PlannerSpec``/``_SubprocessArmParams`` carried no such field and the isolated
worker's ``run_batch(...)`` call did not pass one.

These tests are CPU-safe (no GPU, no SLURM, no full-episode execution). They prove the binding
contract end-to-end for ALL six factorial arms (3 planners x {wrapper_off, wrapper_on}):

1. the derived config loads and enumerates exactly six arms with a stable off/on pairing;
2. each arm's ``safety_wrapper`` is resolved correctly by the campaign resolver;
3. each arm's resolved mapping reaches the runtime safety-wrapper step logic — i.e. the SAME
   ``runtime_config_from_mapping`` call that ``map_runner_episode._prepare_episode_runner`` uses
   to drive ``_apply_safety_wrapper_step`` produces a ``SafetyWrapperRuntimeConfig`` whose
   ``enabled`` matches the arm's wrapper condition;
4. the per-arm field survives the parent->worker subprocess serialization handoff;
5. the isolated worker forwards ``safety_wrapper`` to ``run_batch``;
6. ``run_batch`` forwards ``safety_wrapper`` to ``run_map_batch``.

This is a preflight/readiness contract, NOT benchmark evidence: it verifies that the campaign is
*runnable* and that the wrapper reaches the runtime step logic, without running episodes.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from robot_sf.benchmark.camera_ready._config import load_campaign_config
from robot_sf.benchmark.camera_ready.campaign import _resolve_arm_safety_wrapper
from robot_sf.benchmark.camera_ready.resource_lifecycle import (
    _serialize_subprocess_arm_params,
    _SubprocessArmParams,
)
from robot_sf.benchmark.safety_wrapper_runtime import (
    WRAPPER_OFF_ARM,
    WRAPPER_ON_ARM,
    SafetyWrapperRuntimeConfig,
    runtime_config_from_mapping,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN_CONFIG_PATH = (
    REPO_ROOT / "configs" / "benchmarks" / "issue_4830_safety_wrapper_factorial_v1.yaml"
)
DESIGN_CONTRACT_PATH = REPO_ROOT / "configs" / "research" / "safety_wrapper_ablation_v1.yaml"


def _load_campaign_config() -> object:
    """Load the derived issue #4830 factorial campaign config (cached per-test via module load)."""
    return load_campaign_config(CAMPAIGN_CONFIG_PATH)


def _arm_index(cfg) -> dict[str, object]:
    """Index planner specs by key for stable per-arm assertions."""
    return {planner.key: planner for planner in cfg.planners}


# Stable roster: 3 planners x {wrapper_off, wrapper_on}. The planner base names mirror the design
# contract's planner_groups; the arm keys encode planner + wrapper_arm for the paired report.
EXPECTED_PLANNER_BASES = ("orca", "social_force", "prediction_planner")
EXPECTED_WRAPPER_ARMS = (WRAPPER_OFF_ARM, WRAPPER_ON_ARM)


def _expected_arm_keys() -> list[str]:
    return [f"{base}__{arm}" for base in EXPECTED_PLANNER_BASES for arm in EXPECTED_WRAPPER_ARMS]


class TestSafetyWrapperFactorialConfigShape:
    """The derived config must materialize the declared factorial roster as six runnable arms."""

    def test_safety_wrapper_factorial_config_path_exists(self):
        """The derived runnable config exists at the contract-pinned path."""
        assert CAMPAIGN_CONFIG_PATH.exists(), (
            f"Derived campaign config missing at {CAMPAIGN_CONFIG_PATH}"
        )
        assert DESIGN_CONTRACT_PATH.exists(), "Design contract source must be present"

    def test_safety_wrapper_factorial_config_loads(self):
        """The derived config loads without error and parses arm_isolation: subprocess."""
        cfg = _load_campaign_config()
        assert cfg.name == "issue_4830_safety_wrapper_factorial_v1"
        # The #4826 isolation gate and submit wrapper pin subprocess isolation.
        assert cfg.arm_isolation == "subprocess"
        # Paired seeds preserved verbatim from the design contract.
        assert cfg.seed_policy.mode == "fixed-list"
        assert list(cfg.seed_policy.seeds) == [111, 112, 113]

    def test_safety_wrapper_factorial_has_six_arms(self):
        """Exactly six arms: 3 planners x {wrapper_off, wrapper_on}."""
        cfg = _load_campaign_config()
        assert len(cfg.planners) == 6
        assert _expected_arm_keys() == [planner.key for planner in cfg.planners]

    @pytest.mark.parametrize("base", EXPECTED_PLANNER_BASES)
    def test_safety_wrapper_pair_shares_planner_config(self, base):
        """Each planner's off/on pair differs ONLY in its safety_wrapper block."""
        cfg = _load_campaign_config()
        arms = _arm_index(cfg)
        off = arms[f"{base}__{WRAPPER_OFF_ARM}"]
        on = arms[f"{base}__{WRAPPER_ON_ARM}"]
        # Identical planner configuration ...
        assert off.algo == on.algo
        assert off.algo_config_path == on.algo_config_path
        assert off.benchmark_profile == on.benchmark_profile
        assert off.planner_group == on.planner_group
        assert off.socnav_missing_prereq_policy == on.socnav_missing_prereq_policy
        # ... so the ONLY difference within the pair is the wrapper itself.
        assert off.safety_wrapper != on.safety_wrapper


class TestSafetyWrapperArmBinding:
    """Each arm's safety_wrapper field resolves to the correct runtime contract.

    This is the core preflight contract: the resolved mapping must reach the runtime
    safety-wrapper step logic. ``map_runner_episode._prepare_episode_runner`` binds the wrapper
    via ``runtime_config_from_mapping(safety_wrapper)`` and the step loop branches on
    ``safety_wrapper_runtime.enabled`` (``_apply_safety_wrapper_step``). Asserting that the SAME
    call produces an enabled/disabled config matching the arm proves the runtime step logic is
    reachable for every factorial cell without regression.
    """

    @staticmethod
    def _resolved_runtime(cfg, planner) -> SafetyWrapperRuntimeConfig:
        resolved = _resolve_arm_safety_wrapper(cfg=cfg, planner=planner)
        # The real runtime entry point used by map_runner_episode._prepare_episode_runner.
        return runtime_config_from_mapping(resolved)

    @pytest.mark.parametrize(
        "base", EXPECTED_PLANNER_BASES, ids=[f"{b}-on" for b in EXPECTED_PLANNER_BASES]
    )
    def test_safety_wrapper_on_arm_reaches_runtime_step_logic(self, base):
        """wrapper_on arms bind to an enabled runtime config (reaches _apply_safety_wrapper_step)."""
        cfg = _load_campaign_config()
        planner = _arm_index(cfg)[f"{base}__{WRAPPER_ON_ARM}"]
        runtime = self._resolved_runtime(cfg, planner)
        assert runtime.enabled is True
        assert runtime.arm_key == WRAPPER_ON_ARM
        # Predeclared thresholds are fixed (no per-planner tuning) and validated by the runtime.
        assert runtime.pedestrian_caution_radius_m == pytest.approx(2.0)
        assert runtime.capped_speed_m_s == pytest.approx(0.5)
        assert runtime.ttc_veto_threshold_s == pytest.approx(1.0)
        assert runtime.clearance_veto_m == pytest.approx(0.3)

    @pytest.mark.parametrize(
        "base", EXPECTED_PLANNER_BASES, ids=[f"{b}-off" for b in EXPECTED_PLANNER_BASES]
    )
    def test_safety_wrapper_off_arm_reaches_runtime_step_logic(self, base):
        """wrapper_off arms bind to a disabled runtime config (step logic passthrough)."""
        cfg = _load_campaign_config()
        planner = _arm_index(cfg)[f"{base}__{WRAPPER_OFF_ARM}"]
        runtime = self._resolved_runtime(cfg, planner)
        assert runtime.enabled is False
        assert runtime.arm_key == WRAPPER_OFF_ARM

    def test_safety_wrapper_resolution_prefers_per_arm_over_campaign_default(self):
        """Per-arm safety_wrapper overrides a campaign-level default (factorial pairing contract)."""
        cfg = _load_campaign_config()
        on_planner = _arm_index(cfg)[f"orca__{WRAPPER_ON_ARM}"]
        # Resolve with a campaign-level default present: the per-arm value must still win.
        cfg_with_default = _cfg_with_campaign_default(
            cfg, {"enabled": False, "arm_key": "wrapper_off"}
        )
        resolved = _resolve_arm_safety_wrapper(cfg=cfg_with_default, planner=on_planner)
        runtime = runtime_config_from_mapping(resolved)
        assert runtime.enabled is True
        assert runtime.arm_key == WRAPPER_ON_ARM

    def test_safety_wrapper_resolution_falls_back_to_campaign_default(self):
        """An arm without a per-arm block falls back to the campaign-level default."""
        cfg = _load_campaign_config()
        on_planner = _arm_index(cfg)[f"orca__{WRAPPER_ON_ARM}"]
        planner_without_arm_block = _planner_without_safety_wrapper(on_planner)
        cfg_with_default = _cfg_with_campaign_default(
            cfg, {"enabled": True, "arm_key": "wrapper_on"}
        )
        resolved = _resolve_arm_safety_wrapper(
            cfg=cfg_with_default, planner=planner_without_arm_block
        )
        runtime = runtime_config_from_mapping(resolved)
        assert runtime.enabled is True
        assert runtime.arm_key == WRAPPER_ON_ARM

    def test_safety_wrapper_resolution_none_when_unset(self):
        """An arm with no block and no campaign default keeps the wrapper off (runtime default)."""
        cfg = _load_campaign_config()
        on_planner = _arm_index(cfg)[f"orca__{WRAPPER_ON_ARM}"]
        planner_without_arm_block = _planner_without_safety_wrapper(on_planner)
        resolved = _resolve_arm_safety_wrapper(cfg=cfg, planner=planner_without_arm_block)
        assert resolved is None
        # The runtime treats None as disabled (passthrough), preserving benchmark behavior.
        assert runtime_config_from_mapping(resolved).enabled is False


class TestSafetyWrapperSubprocessHandoff:
    """The per-arm safety_wrapper survives the parent->worker subprocess serialization handoff."""

    def _arm_params(self, cfg, planner) -> _SubprocessArmParams:
        resolved = _resolve_arm_safety_wrapper(cfg=cfg, planner=planner)
        return _SubprocessArmParams(
            planner_key=planner.key,
            planner_algo=planner.algo,
            planner_human_model_variant=None,
            planner_human_model_source=None,
            planner_group=planner.planner_group,
            benchmark_profile=planner.benchmark_profile,
            socnav_missing_prereq_policy="fail-fast",
            adapter_impact_eval=False,
            kinematics="differential_drive",
            observation_mode=None,
            workers=2,
            horizon=100,
            dt=0.1,
            scenario_matrix_path=Path("scenarios.yaml"),
            episodes_path=Path("/tmp/episodes.jsonl"),
            summary_path=Path("/tmp/summary.json"),
            record_forces=True,
            record_planner_decision_trace=False,
            record_simulation_step_trace=False,
            observation_noise=None,
            synthetic_actuation_profile=None,
            latency_stress_profile=None,
            snqi_weights=None,
            snqi_baseline=None,
            algo_config_path=None,
            safety_wrapper=resolved,
        )

    @pytest.mark.parametrize(
        "base,expected_enabled",
        [
            ("orca", True),
            ("social_force", True),
            ("prediction_planner", True),
        ],
        ids=["orca-on", "social_force-on", "prediction_planner-on"],
    )
    def test_safety_wrapper_on_arm_survives_serialization_round_trip(self, base, expected_enabled):
        """wrapper_on safety_wrapper round-trips through the real serializer unchanged."""
        cfg = _load_campaign_config()
        planner = _arm_index(cfg)[f"{base}__{WRAPPER_ON_ARM}"]
        params = self._arm_params(cfg, planner)
        resolved = params.safety_wrapper

        serialized = _serialize_subprocess_arm_params(params)
        payload = json.loads(serialized)
        assert payload["safety_wrapper"] == resolved

        rebuilt = _SubprocessArmParams(**payload)
        assert rebuilt.safety_wrapper == resolved
        assert runtime_config_from_mapping(rebuilt.safety_wrapper).enabled is expected_enabled

    def test_safety_wrapper_off_arm_survives_serialization_round_trip(self):
        """wrapper_off safety_wrapper round-trips through the real serializer unchanged."""
        cfg = _load_campaign_config()
        planner = _arm_index(cfg)["orca__wrapper_off"]
        params = self._arm_params(cfg, planner)
        resolved = params.safety_wrapper

        serialized = _serialize_subprocess_arm_params(params)
        rebuilt = _SubprocessArmParams(**json.loads(serialized))
        assert rebuilt.safety_wrapper == resolved
        assert runtime_config_from_mapping(rebuilt.safety_wrapper).enabled is False


class TestSafetyWrapperWorkerExecutionPath:
    """The isolated worker forwards safety_wrapper to run_batch for every factorial arm."""

    @pytest.mark.parametrize("base", EXPECTED_PLANNER_BASES, ids=EXPECTED_PLANNER_BASES)
    def test_safety_wrapper_on_arm_worker_forwards_to_run_batch(self, tmp_path, base):
        """A wrapper_on arm's worker hands the enabled safety_wrapper to run_batch."""
        cfg = _load_campaign_config()
        planner = _arm_index(cfg)[f"{base}__{WRAPPER_ON_ARM}"]
        resolved = _resolve_arm_safety_wrapper(cfg=cfg, planner=planner)
        captured = _run_worker_and_capture_run_batch_kwargs(tmp_path, safety_wrapper=resolved)
        assert captured["safety_wrapper"] == resolved
        assert runtime_config_from_mapping(captured["safety_wrapper"]).enabled is True

    def test_safety_wrapper_off_arm_worker_forwards_to_run_batch(self, tmp_path):
        """A wrapper_off arm's worker hands the disabled safety_wrapper to run_batch."""
        cfg = _load_campaign_config()
        planner = _arm_index(cfg)["orca__wrapper_off"]
        resolved = _resolve_arm_safety_wrapper(cfg=cfg, planner=planner)
        captured = _run_worker_and_capture_run_batch_kwargs(tmp_path, safety_wrapper=resolved)
        assert captured["safety_wrapper"] == resolved
        assert runtime_config_from_mapping(captured["safety_wrapper"]).enabled is False

    def test_safety_wrapper_worker_defaults_to_none_when_unset(self, tmp_path):
        """An arm with no resolved wrapper forwards None (off) to run_batch, no regression."""
        captured = _run_worker_and_capture_run_batch_kwargs(tmp_path, safety_wrapper=None)
        assert captured["safety_wrapper"] is None


class TestSafetyWrapperRunBatchForwarding:
    """run_batch forwards safety_wrapper to run_map_batch (the map-based runtime path)."""

    def test_safety_wrapper_run_batch_forwards_to_run_map_batch(self, tmp_path):
        """run_batch hands safety_wrapper to run_map_batch for map-based scenarios."""
        from robot_sf.benchmark import runner

        captured: dict[str, object] = {}
        scenario = {
            "name": "preflight_map",
            "map_file": "maps/svg_maps/francis2023/francis2023_blind_corner.svg",
            "seeds": [111],
            "simulation_config": {"max_episode_steps": 5},
            "robot_config": {"kinematics": "differential_drive"},
        }
        safety_wrapper = {
            "enabled": True,
            "arm_key": WRAPPER_ON_ARM,
            "pedestrian_caution_radius_m": 2.0,
            "capped_speed_m_s": 0.5,
            "ttc_veto_threshold_s": 1.0,
            "clearance_veto_m": 0.3,
        }

        def fake_run_map_batch(*_args, **kwargs):
            captured.update(kwargs)
            return {"status": "ok", "total_jobs": 1, "written": 1, "failures": []}

        out_path = tmp_path / "episodes.jsonl"
        with patch.object(runner, "run_map_batch", side_effect=fake_run_map_batch):
            runner.run_batch(
                [scenario],
                out_path=out_path,
                schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
                horizon=5,
                dt=0.1,
                safety_wrapper=safety_wrapper,
            )
        assert captured["safety_wrapper"] == safety_wrapper

    def test_safety_wrapper_run_batch_defaults_to_none(self, tmp_path):
        """run_batch keeps safety_wrapper off (None) when not provided: no behavior regression."""
        from robot_sf.benchmark import runner

        captured: dict[str, object] = {}
        scenario = {
            "name": "preflight_map_default",
            "map_file": "maps/svg_maps/francis2023/francis2023_blind_corner.svg",
            "seeds": [111],
            "simulation_config": {"max_episode_steps": 5},
            "robot_config": {"kinematics": "differential_drive"},
        }

        def fake_run_map_batch(*_args, **kwargs):
            captured.update(kwargs)
            return {"status": "ok", "total_jobs": 1, "written": 1, "failures": []}

        out_path = tmp_path / "episodes.jsonl"
        with patch.object(runner, "run_map_batch", side_effect=fake_run_map_batch):
            runner.run_batch(
                [scenario],
                out_path=out_path,
                schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
                horizon=5,
                dt=0.1,
            )
        assert captured["safety_wrapper"] is None


# --- Helpers ---------------------------------------------------------------------------


def _cfg_with_campaign_default(cfg, safety_wrapper: dict[str, object]):
    """Return a copy of cfg with a campaign-level safety_wrapper default set."""
    from dataclasses import replace

    return replace(cfg, safety_wrapper=dict(safety_wrapper))


def _planner_without_safety_wrapper(planner):
    """Return a copy of a planner spec with its per-arm safety_wrapper removed."""
    from dataclasses import replace

    return replace(planner, safety_wrapper=None)


def _run_worker_and_capture_run_batch_kwargs(
    tmp_path: Path, *, safety_wrapper: dict[str, object] | None
) -> dict[str, object]:
    """Run _run_single_arm_subprocess with run_batch mocked; return the captured kwargs.

    Mirrors the canonical subprocess-worker test pattern: the worker consumes a serialized
    scoped-scenarios list and forwards its parameters to ``run_batch``. Asserting
    ``safety_wrapper`` is in those kwargs proves the worker execution path reaches the runtime
    binding for the arm.
    """
    from robot_sf.benchmark.camera_ready.resource_lifecycle import _run_single_arm_subprocess

    scoped_path = tmp_path / "scoped_scenarios.json"
    scoped_path.write_text("[]", encoding="utf-8")

    base = _minimal_arm_params(tmp_path, scoped_path=scoped_path)
    params = _SubprocessArmParams(**{**base.__dict__, "safety_wrapper": safety_wrapper})

    captured: dict[str, object] = {}

    def fake_run_batch(*_args, **kwargs):
        captured.update(kwargs)
        return {"status": "ok", "total_jobs": 0, "written": 0, "failures": []}

    with (
        patch("robot_sf.benchmark.runner.run_batch", side_effect=fake_run_batch),
        patch(
            "robot_sf.benchmark.fallback_policy.summarize_benchmark_availability",
            return_value=Mock(availability_status="ok"),
        ),
        patch("robot_sf.benchmark.fallback_policy.availability_payload", return_value={}),
    ):
        _run_single_arm_subprocess(params)

    assert "safety_wrapper" in captured, "worker did not forward safety_wrapper to run_batch"
    return captured


def _minimal_arm_params(tmp_path: Path, *, scoped_path: Path) -> _SubprocessArmParams:
    """Build a minimal _SubprocessArmParams for worker-path tests."""
    return _SubprocessArmParams(
        planner_key="preflight",
        planner_algo="orca",
        planner_human_model_variant=None,
        planner_human_model_source=None,
        planner_group="core",
        benchmark_profile="baseline-safe",
        socnav_missing_prereq_policy="fail-fast",
        adapter_impact_eval=False,
        kinematics="differential_drive",
        observation_mode=None,
        workers=1,
        horizon=5,
        dt=0.1,
        scenario_matrix_path=tmp_path / "does-not-exist.yaml",
        episodes_path=tmp_path / "episodes.jsonl",
        summary_path=tmp_path / "summary.json",
        record_forces=True,
        record_planner_decision_trace=False,
        record_simulation_step_trace=False,
        observation_noise=None,
        synthetic_actuation_profile=None,
        latency_stress_profile=None,
        snqi_weights=None,
        snqi_baseline=None,
        algo_config_path=None,
        scoped_scenarios_path=scoped_path,
    )
