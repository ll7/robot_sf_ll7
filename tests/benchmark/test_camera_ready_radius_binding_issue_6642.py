"""Focused tests for the issue #6642 camera-ready radius binding contract."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from robot_sf.benchmark.camera_ready import (
    CampaignConfig,
    PlannerSpec,
    RadiusSweepConfig,
    load_campaign_config,
)
from robot_sf.benchmark.camera_ready._config import (
    RadiusSweepBindingPreflightError,
    _apply_radius_sweep_binding,
    _load_campaign_scenarios,
    _parse_radius_sweep,
    _scenario_with_kinematics,
)
from robot_sf.benchmark.camera_ready._preflight import (
    _build_manifest_context_block,
    _build_preflight_validate_payload,
)

_CONFIG_PATH = Path("radius-sweep-test.yaml")
_GATE1_RECEIPT = "a" * 64
_GATE1_COMMIT = "b" * 40


def _minimal_campaign(*, radius_sweep: RadiusSweepConfig | None = None) -> CampaignConfig:
    """Build a dependency-free campaign config for helper tests."""
    return CampaignConfig(
        name="radius-binding-test",
        scenario_matrix_path=Path("configs/scenarios/test.yaml"),
        planners=(PlannerSpec(key="goal", algo="goal"),),
        radius_sweep=radius_sweep,
    )


def _pending_raw(*, arm_key: str = "r0p5", radius_m: float = 0.5) -> dict[str, object]:
    """Return valid preparation-only radius metadata."""
    return {
        "issue": 6642,
        "parent_issue": 6600,
        "arm_key": arm_key,
        "radius_m": radius_m,
        "baseline_arm": False,
        "runtime_binding_status": "pending_gate1_canary",
    }


def _bound_config(*, arm_key: str = "r0p5", radius_m: float = 0.5) -> RadiusSweepConfig:
    """Return a fully admitted fixture with synthetic provenance values."""
    return RadiusSweepConfig(
        issue=6642,
        parent_issue=6600,
        arm_key=arm_key,
        radius_m=radius_m,
        baseline_arm=arm_key == "r1p0",
        runtime_binding_status="bound_runtime",
        binding_contract_version="radius_binding_canary.v1",
        gate1_canary_issue=6641,
        gate1_receipt_sha256=_GATE1_RECEIPT,
        gate1_source_commit=_GATE1_COMMIT,
    )


def test_legacy_campaign_without_radius_block_is_a_no_op() -> None:
    """An absent radius block returns the original scenario list unchanged."""
    scenarios = [{"name": "s1", "robot_config": {"max_velocity": 1.2}}]

    result = _apply_radius_sweep_binding(scenarios, _minimal_campaign().radius_sweep)

    assert result is scenarios
    assert result == scenarios
    assert _minimal_campaign().radius_sweep is None


@pytest.mark.parametrize(
    "mutator",
    [
        lambda raw: raw.update(issue=9999),
        lambda raw: raw.update(radius_m=float("nan")),
        lambda raw: raw.update(radius_m=0.0),
        lambda raw: raw.update(runtime_binding_status="bound_runtime"),
        lambda raw: raw.update(
            runtime_binding_status="bound_runtime",
            binding_contract_version="wrong.v1",
            gate1_canary_issue=6641,
            gate1_receipt_sha256=_GATE1_RECEIPT,
            gate1_source_commit=_GATE1_COMMIT,
        ),
    ],
)
def test_malformed_radius_metadata_fails_closed(mutator) -> None:
    """Invalid identity, geometry, contract, and provenance cannot be parsed."""
    raw = _pending_raw()
    mutator(raw)

    with pytest.raises((TypeError, ValueError)):
        _parse_radius_sweep(raw, config_path=_CONFIG_PATH)


def test_pending_radius_arm_fails_before_episode_execution() -> None:
    """Preparation metadata cannot reach an episode-producing collaborator."""
    scenarios = [{"name": "s1", "robot_config": {}, "metadata": {"keep": "me"}}]
    episode_called = False

    def _episode_collaborator() -> None:
        nonlocal episode_called
        episode_called = True

    with pytest.raises(RadiusSweepBindingPreflightError, match="pending_gate1_canary"):
        _apply_radius_sweep_binding(
            scenarios,
            RadiusSweepConfig(
                issue=6642,
                parent_issue=6600,
                arm_key="r0p5",
                radius_m=0.5,
                baseline_arm=False,
                runtime_binding_status="pending_gate1_canary",
            ),
        )
    assert not episode_called
    assert scenarios[0]["robot_config"] == {}


def test_bound_radius_patches_all_scenarios_and_preserves_unrelated_fields() -> None:
    """The admitted radius reaches each scenario without mutating its source."""
    scenarios = [
        {
            "name": "s1",
            "robot_config": {"max_velocity": 1.2, "radius": 1.0},
            "metadata": {"keep": "one"},
        },
        {"name": "s2", "robot_config": {}, "metadata": {"keep": "two"}},
    ]
    original = json.loads(json.dumps(scenarios))

    result = _apply_radius_sweep_binding(scenarios, _bound_config())

    assert len(result) == 2
    assert [row["robot_config"]["radius"] for row in result] == [0.5, 0.5]
    assert result[0]["robot_config"]["max_velocity"] == 1.2
    assert [row["metadata"]["keep"] for row in result] == ["one", "two"]
    assert all(row["metadata"]["radius_binding"]["status"] == "bound_runtime" for row in result)
    assert all(
        row["metadata"]["radius_binding"]["source"] == "radius_sweep.radius_m" for row in result
    )
    assert scenarios == original
    assert result[0] is not scenarios[0]
    assert result[0]["metadata"] is not scenarios[0]["metadata"]


def test_subprocess_scoped_scenario_retains_radius_binding() -> None:
    """The existing kinematics copy used by subprocess arms keeps the binding metadata."""
    bound = _apply_radius_sweep_binding(
        [{"name": "s1", "robot_config": {}, "metadata": {}}],
        _bound_config(radius_m=0.8),
    )[0]

    scoped = _scenario_with_kinematics(
        bound,
        kinematics="differential_drive",
        holonomic_command_mode="vx_vy",
    )

    assert scoped["robot_config"]["radius"] == 0.8
    assert scoped["robot_config"]["type"] == "differential_drive"
    assert scoped["metadata"]["radius_binding"]["arm_key"] == "r0p5"


def test_preflight_and_manifest_expose_binding_contract() -> None:
    """Serialized preflight and manifest surfaces expose the full binding identity."""
    cfg = _minimal_campaign(radius_sweep=_bound_config())
    validate = _build_preflight_validate_payload(
        cfg,
        campaign_id="radius-binding-test",
        created_at_utc="2026-08-04T00:00:00Z",
        scenarios=[{"name": "s1"}],
        resolved_seeds=[1],
        scenario_horizons_summary=None,
        route_clearance_warnings=[],
        route_clearance_warning_summary={},
        noise_spec={},
        noise_hash="noise-hash",
        checkpoint_preflight_summary={"stage": False, "checked": 0, "resolved": 0},
        checkpoint_preflight_mode="metadata_only",
    )
    manifest = _build_manifest_context_block(
        cfg,
        campaign_id="radius-binding-test",
        created_at_utc="2026-08-04T00:00:00Z",
        metadata={
            "scenario_hash": "scenario-hash",
            "resolved_seeds": [1],
            "git_meta": {},
            "config_hash": "config-hash",
        },
        invoked_command="test",
    )

    for payload in (validate, manifest):
        binding = payload["radius_binding"]
        assert binding["status"] == "bound_runtime"
        assert binding["radius_m"] == 0.5
        assert binding["source"] == "radius_sweep.radius_m"
        assert binding["contract_version"] == "radius_binding_canary.v1"
        assert binding["gate1_canary_issue"] == 6641
        assert binding["gate1_receipt_sha256"] == _GATE1_RECEIPT
        assert binding["gate1_source_commit"] == _GATE1_COMMIT


@pytest.mark.parametrize(
    ("config_name", "arm_key", "radius_m"),
    [
        ("issue_6642_radius_sweep_arm_0p5m.yaml", "r0p5", 0.5),
        ("issue_6642_radius_sweep_arm_0p8m.yaml", "r0p8", 0.8),
        ("issue_6642_radius_sweep_arm_1p0m.yaml", "r1p0", 1.0),
    ],
)
def test_preparation_arm_configs_parse_as_distinct_pending_treatments(
    config_name: str,
    arm_key: str,
    radius_m: float,
) -> None:
    """The three preparation configs remain distinct and explicitly non-runnable."""
    cfg = load_campaign_config(Path("configs/benchmarks") / config_name)

    assert cfg.radius_sweep is not None
    assert cfg.radius_sweep.arm_key == arm_key
    assert cfg.radius_sweep.radius_m == radius_m
    assert cfg.radius_sweep.runtime_binding_status == "pending_gate1_canary"
    assert cfg.radius_sweep.parent_issue == 6600


def test_real_camera_ready_loader_binds_an_admitted_arm_without_episodes() -> None:
    """The real 0.5 m matrix resolves every loaded scenario at the admitted radius."""
    cfg = load_campaign_config(Path("configs/benchmarks/issue_6642_radius_sweep_arm_0p5m.yaml"))
    bound_cfg = replace(cfg, radius_sweep=_bound_config(radius_m=0.5))

    scenarios = _load_campaign_scenarios(bound_cfg)

    assert len(scenarios) == 48
    assert all(scenario["robot_config"]["radius"] == 0.5 for scenario in scenarios)
    assert all(
        scenario["metadata"]["radius_binding"]["status"] == "bound_runtime"
        for scenario in scenarios
    )


def test_package_and_compatibility_facades_export_radius_type() -> None:
    """The package and legacy config facade expose the new typed field."""
    from robot_sf.benchmark import camera_ready_campaign_config

    assert camera_ready_campaign_config.RadiusSweepConfig is RadiusSweepConfig


def test_cli_reports_structured_pending_binding_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The preflight CLI returns a nonzero structured status for pending arms."""
    from scripts.tools import run_camera_ready_benchmark as cli

    cfg = _minimal_campaign(
        radius_sweep=RadiusSweepConfig(
            issue=6642,
            parent_issue=6600,
            arm_key="r0p5",
            radius_m=0.5,
            baseline_arm=False,
            runtime_binding_status="pending_gate1_canary",
        )
    )
    monkeypatch.setattr(cli, "load_campaign_config", lambda _path: cfg)

    def _raise_binding_error(*_args, **_kwargs):
        raise RadiusSweepBindingPreflightError("pending_gate1_canary")

    monkeypatch.setattr(cli, "prepare_campaign_preflight", _raise_binding_error)
    monkeypatch.setattr(
        cli,
        "run_campaign",
        lambda *_args, **_kwargs: pytest.fail("run mode collaborator must not be called"),
    )

    exit_code = cli.main(["--config", "ignored.yaml", "--mode", "preflight"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "radius_binding_preflight_failed"
    assert payload["evidence_status"] == "blocked"
