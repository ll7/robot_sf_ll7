"""Predictive horizon refusals must happen before campaign scenarios or submission."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.campaign.campaign_checkpoint_preflight import (
    CampaignCheckpointPreflightError,
    check_campaign_arm_checkpoints_preflight,
    iter_campaign_arm_checkpoint_references,
)
from robot_sf.planner.predictive_model import (
    PredictiveModelConfig,
    PredictiveTrajectoryModel,
    save_predictive_checkpoint,
)

ROOT = Path(__file__).resolve().parents[2]
DOORWAY = (
    ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_three_width_doorway_v2.yaml"
)


def _override_campaign(tmp_path, manifest, scenarios):
    """Bind a candidate to explicit development scenario identities without a seed inventory."""
    config_path = tmp_path / "effective_candidate.yaml"
    config_path.write_text(yaml.safe_dump(manifest))
    scenario_path = tmp_path / "effective_scenarios.yaml"
    scenario_path.write_text(yaml.safe_dump({"scenarios": scenarios}))
    cfg = load_campaign_config(DOORWAY)
    arm = replace(
        next(a for a in cfg.planners if a.algo == "predictive_mppi"), algo_config_path=config_path
    )
    return replace(cfg, planners=(arm,), scenario_matrix_path=scenario_path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"scenario_overrides": {"doorway": {"horizon_steps": 12}}},
        {"family_overrides": {"bottleneck": {"horizon_steps": 12}}},
        {
            "family_overrides": {"bottleneck": {"horizon_steps": 10}},
            "scenario_overrides": {"doorway": {"horizon_steps": 12}},
        },
    ],
)
def test_effective_context_override_refuses_long_horizon(tmp_path, predictor_registry, overrides):
    """Runtime scenario/family merges must not hide a 12-step request behind an eight-step base."""
    manifest = {
        "base_config_path": str(ROOT / "configs/algos/predictive_mppi_release_v0_0_8.yaml"),
        **overrides,
    }
    cfg = _override_campaign(
        tmp_path,
        manifest,
        [
            {"name": "doorway", "metadata": {"family": "bottleneck"}, "seeds": [1001]},
        ],
    )
    with pytest.raises(CampaignCheckpointPreflightError, match="required_horizon_steps=12"):
        check_campaign_arm_checkpoints_preflight(cfg, registry_path=predictor_registry)


def test_family_checkpoint_and_scenario_horizon_combination_refused(tmp_path, predictor_registry):
    """Two individually compatible overrides can combine into an incompatible effective binding."""
    longer = tmp_path / "twelve_steps.pt"
    save_predictive_checkpoint(
        longer,
        model=PredictiveTrajectoryModel(PredictiveModelConfig(horizon_steps=12)),
        optimizer=None,
        epoch=0,
    )
    cfg = _override_campaign(
        tmp_path,
        {
            "base_config_path": str(ROOT / "configs/algos/predictive_mppi_release_v0_0_8.yaml"),
            "params": {"predictive_checkpoint_path": str(longer)},
            "family_overrides": {
                "bottleneck": {"predictive_checkpoint_path": str(tmp_path / "eight_steps.pt")}
            },
            "scenario_overrides": {"doorway": {"horizon_steps": 12}},
        },
        [{"name": "doorway", "family": "bottleneck", "seeds": [1001]}],
    )
    with pytest.raises(CampaignCheckpointPreflightError, match="forecast_steps=8"):
        check_campaign_arm_checkpoints_preflight(cfg, registry_path=predictor_registry)


@pytest.fixture
def predictor_registry(tmp_path):
    """Use a real eight-step checkpoint without downloading or accessing evaluation seeds."""
    checkpoint = tmp_path / "eight_steps.pt"
    save_predictive_checkpoint(
        checkpoint,
        model=PredictiveTrajectoryModel(PredictiveModelConfig(horizon_steps=8)),
        optimizer=None,
        epoch=0,
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "models": [
                    {
                        "model_id": "predictive_proxy_selected_v1",
                        "local_path": str(checkpoint),
                    }
                ],
            }
        )
    )
    return registry


def test_doorway_v2_binding_refused_before_campaign(predictor_registry):
    """The unchanged doorway binding must refuse its 12-step MPPI arm on an eight-step model."""
    cfg = load_campaign_config(DOORWAY)
    arm = next(arm for arm in cfg.planners if arm.algo == "predictive_mppi")
    cfg = replace(cfg, planners=(arm,))
    with pytest.raises(CampaignCheckpointPreflightError) as exc:
        check_campaign_arm_checkpoints_preflight(cfg, registry_path=predictor_registry)
    message = str(exc.value)
    for expected in (
        "predictive_mppi",
        "predictive_mppi_camera_ready.yaml",
        "predictive_proxy_selected_v1",
        "required_horizon_steps=12",
        "forecast_steps=8",
    ):
        assert expected in message


def test_release_eight_step_binding_passes(predictor_registry):
    """The frozen main planner's authored eight-step sequence fits the saved output head."""
    cfg = load_campaign_config(DOORWAY)
    arm = next(arm for arm in cfg.planners if arm.algo == "predictive_mppi")
    arm = replace(arm, algo_config_path=ROOT / "configs/algos/predictive_mppi_release_v0_0_8.yaml")
    summary = check_campaign_arm_checkpoints_preflight(
        replace(cfg, planners=(arm,)),
        registry_path=predictor_registry,
    )
    record = summary["predictive_horizons"][0]
    assert record["required_horizon_steps"] == record["forecast_steps"] == 8


def test_stale_checkpoint_config_cannot_admit_larger_model_horizon(tmp_path, predictor_registry):
    """An edited config cannot turn a serialized eight-step output head into twelve steps."""
    import torch

    from robot_sf.benchmark.campaign.predictive_horizon_preflight import (
        PredictiveHorizonPreflightError,
    )

    checkpoint = tmp_path / "eight_steps.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["config"]["horizon_steps"] = 12
    torch.save(payload, checkpoint)
    cfg = _override_campaign(
        tmp_path,
        {"predictive_checkpoint_path": str(checkpoint), "horizon_steps": 12},
        [{"name": "doorway", "seeds": [1001]}],
    )
    with pytest.raises(PredictiveHorizonPreflightError, match="size mismatch for decoder") as exc:
        check_campaign_arm_checkpoints_preflight(cfg, registry_path=predictor_registry)
    assert exc.value.binding["unverified_reason"] == "checkpoint_error"
    assert "forecast_steps" not in exc.value.binding


@pytest.mark.parametrize(
    "section,key,value,remove",
    [
        ("config", "input_dim", None, True),
        ("config", "horizon_steps", None, True),
        (None, "feature_schema", None, True),
        (None, "feature_schema", None, False),
        (None, "feature_schema", [], False),
        (None, "feature_schema", {}, False),
        ("config", "input_dim", "4", False),
        ("config", "horizon_steps", "8", False),
        ("config", "horizon_steps", True, False),
        ("feature_schema", "input_dim", "4", False),
    ],
)
def test_required_checkpoint_metadata_fails_closed(
    tmp_path, predictor_registry, section, key, value, remove
):
    """Required metadata is never reconstructed from defaults or coerced from invalid types."""
    import torch

    from robot_sf.benchmark.campaign.predictive_horizon_preflight import (
        PredictiveHorizonPreflightError,
    )

    checkpoint = tmp_path / "eight_steps.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload if section is None else payload[section]
    if remove:
        del metadata[key]
    else:
        metadata[key] = value
    torch.save(payload, checkpoint)
    cfg = _override_campaign(
        tmp_path,
        {"predictive_checkpoint_path": str(checkpoint), "horizon_steps": 8},
        [{"name": "doorway", "seeds": [1001]}],
    )
    with pytest.raises(PredictiveHorizonPreflightError, match=key) as exc:
        check_campaign_arm_checkpoints_preflight(cfg, registry_path=predictor_registry)
    assert exc.value.binding["unverified_reason"] == "checkpoint_error"
    assert "forecast_steps" not in exc.value.binding


@pytest.mark.parametrize("saved_steps", [8, 12])
def test_checkpoint_path_override_controls_forecast_window(
    tmp_path, predictor_registry, saved_steps
):
    """Explicit checkpoint swaps win over model ID and misleading predictor YAML promises."""
    from robot_sf.benchmark.campaign.predictive_horizon_preflight import (
        check_campaign_predictive_horizons_preflight,
    )

    checkpoint = tmp_path / "swapped.pt"
    save_predictive_checkpoint(
        checkpoint,
        model=PredictiveTrajectoryModel(PredictiveModelConfig(horizon_steps=saved_steps)),
        optimizer=None,
        epoch=0,
    )
    config_path = tmp_path / "mppi.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "predictive_checkpoint_path": str(checkpoint),
                "predictive_model_id": "predictive_proxy_selected_v1",
                "predictive_horizon_steps": 24,
                "horizon_steps": 12,
                "forecast_variant": "none",
            }
        )
    )
    cfg = load_campaign_config(DOORWAY)
    arm = replace(
        next(a for a in cfg.planners if a.algo == "predictive_mppi"), algo_config_path=config_path
    )
    cfg = replace(cfg, planners=(arm,))
    refs = iter_campaign_arm_checkpoint_references(cfg)
    assert [(ref.kind, ref.value) for ref in refs] == [("model_path", str(checkpoint))]
    if saved_steps == 8:
        with pytest.raises(CampaignCheckpointPreflightError, match="forecast_steps=8"):
            check_campaign_arm_checkpoints_preflight(cfg, registry_path=predictor_registry)
    else:
        records = check_campaign_predictive_horizons_preflight(
            cfg, registry_path=predictor_registry
        )
        assert records[0]["forecast_steps"] == records[0]["required_horizon_steps"] == 12


def test_scenario_override_checked_and_disabled_arm_skipped(tmp_path, predictor_registry):
    """A safe base cannot hide a scenario override that requests a longer MPPI sequence."""
    from robot_sf.benchmark.campaign.predictive_horizon_preflight import (
        check_campaign_predictive_horizons_preflight,
    )

    cfg = _override_campaign(
        tmp_path,
        {
            "base_config_path": str(ROOT / "configs/algos/predictive_mppi_release_v0_0_8.yaml"),
            "scenario_algo_overrides": {
                "doorway": {
                    "base_config_path": str(
                        ROOT / "configs/algos/predictive_mppi_release_v0_0_8.yaml"
                    ),
                    "params": {"horizon_steps": 12},
                }
            },
        },
        [{"name": "doorway", "seeds": [1001]}],
    )
    arm = cfg.planners[0]
    with pytest.raises(CampaignCheckpointPreflightError, match="scenario=doorway"):
        check_campaign_arm_checkpoints_preflight(
            replace(cfg, planners=(arm,)), registry_path=predictor_registry
        )
    assert (
        check_campaign_predictive_horizons_preflight(
            replace(cfg, planners=(replace(arm, enabled=False),)),
            registry_path=predictor_registry,
        )
        == []
    )


def test_prediction_adaptive_horizon_retains_runtime_bound(tmp_path, predictor_registry):
    """The adaptive prediction planner's existing forecast bound differs from MPPI's horizon."""
    from robot_sf.benchmark.campaign.predictive_horizon_preflight import (
        check_campaign_predictive_horizons_preflight,
    )

    cfg = load_campaign_config(DOORWAY)
    arm = next(a for a in cfg.planners if a.algo == "predictive_mppi")
    path = tmp_path / "prediction.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "predictive_model_id": "predictive_proxy_selected_v1",
                "predictive_horizon_steps": 8,
                "predictive_adaptive_horizon_enabled": True,
                "predictive_horizon_boost_steps": 6,
            }
        )
    )
    arm = replace(arm, algo="prediction_planner", algo_config_path=path)
    records = check_campaign_predictive_horizons_preflight(
        replace(cfg, planners=(arm,)),
        registry_path=predictor_registry,
    )
    assert records[0]["required_horizon_steps"] == records[0]["forecast_steps"] == 8


def test_remote_metadata_alone_cannot_admit_predictive_window(tmp_path):
    """A downloadable reference supplies no actual output shape until staged."""
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "models": [
                    {
                        "model_id": "predictive_proxy_selected_v1",
                        "local_path": str(tmp_path / "absent.pt"),
                        "github_release": {
                            "url": "https://example.invalid/model.pt",
                            "sha256": "0" * 64,
                        },
                    }
                ],
            }
        )
    )
    cfg = load_campaign_config(DOORWAY)
    arm = next(a for a in cfg.planners if a.algo == "predictive_mppi")
    with pytest.raises(CampaignCheckpointPreflightError, match="no forecast window was admitted"):
        check_campaign_arm_checkpoints_preflight(
            replace(cfg, planners=(arm,)), registry_path=registry
        )


def test_scan_continues_after_refusal_without_loading_seed_inventory(tmp_path, predictor_registry):
    """The census reports later bindings even when the first arm refuses and seeds are unavailable."""
    from scripts.benchmark.scan_predictive_horizons import scan_predictive_horizons

    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "seed_policy": {
                    "mode": "seed-set",
                    "seed_sets_path": "absent_sealed_inventory.yaml",
                },
                "planners": [
                    {
                        "key": "bad",
                        "algo": "predictive_mppi",
                        "algo_config": str(
                            ROOT / "configs/algos/predictive_mppi_camera_ready.yaml"
                        ),
                    },
                    {
                        "key": "good",
                        "algo": "predictive_mppi",
                        "algo_config": str(
                            ROOT / "configs/algos/predictive_mppi_release_v0_0_8.yaml"
                        ),
                    },
                ],
            }
        )
    )
    scan = scan_predictive_horizons(tmp_path, registry_path=predictor_registry)
    assert scan["campaign_matrices_scanned"] == 1
    assert (scan["compatible"], scan["incompatible"], scan["unverified"]) == (1, 1, 0)
    assert [(r["planner_key"], r["status"]) for r in scan["bindings"]] == [
        ("bad", "incompatible"),
        ("good", "compatible"),
    ]


@pytest.mark.parametrize(
    "name,family,selected_family",
    [
        ("doorway", "bottleneck", "bottleneck"),
        ("francis2023_doorway", "bottleneck", "francis2023"),
        ("classic_doorway", "bottleneck", "classic"),
    ],
)
def test_scenario_override_wins_family_and_unused_overrides(
    tmp_path, predictor_registry, name, family, selected_family
):
    """Check the actual selected family and final scenario merge, rather than hypothetical bases."""
    from robot_sf.benchmark.campaign.predictive_horizon_preflight import (
        check_campaign_predictive_horizons_preflight,
    )

    cfg = _override_campaign(
        tmp_path,
        {
            "base_config_path": str(ROOT / "configs/algos/predictive_mppi_release_v0_0_8.yaml"),
            "family_overrides": {
                selected_family: {"horizon_steps": 12},
                "unused": {"horizon_steps": 30},
            },
            "scenario_overrides": {name: {"horizon_steps": 8}},
            "scenario_algo_overrides": {"absent_scenario": {"params": {"horizon_steps": 30}}},
        },
        [{"name": name, "metadata": {"family": family}, "seeds": [1001]}],
    )
    records = check_campaign_predictive_horizons_preflight(cfg, registry_path=predictor_registry)
    assert [(r["scenario"], r["family"], r["required_horizon_steps"]) for r in records] == [
        (name, selected_family, 8),
    ]


def test_scenario_algo_override_uses_its_own_config(tmp_path, predictor_registry):
    """The runtime's early algorithm override must bypass global params/family/scenario merges."""
    from robot_sf.benchmark.campaign.predictive_horizon_preflight import (
        check_campaign_predictive_horizons_preflight,
    )

    release = str(ROOT / "configs/algos/predictive_mppi_release_v0_0_8.yaml")
    cfg = _override_campaign(
        tmp_path,
        {
            "base_config_path": release,
            "params": {"horizon_steps": 12},
            "family_overrides": {"bottleneck": {"horizon_steps": 20}},
            "scenario_overrides": {"doorway": {"horizon_steps": 30}},
            "scenario_algo_overrides": {
                "doorway": {
                    "algo": "predictive_mppi",
                    "base_config_path": release,
                }
            },
        },
        [{"name": "doorway", "family": "bottleneck", "seeds": [1001]}],
    )
    records = check_campaign_predictive_horizons_preflight(cfg, registry_path=predictor_registry)
    assert len(records) == 1
    assert records[0]["required_horizon_steps"] == records[0]["forecast_steps"] == 8


def test_included_scenario_metadata_and_candidate_selection(tmp_path, predictor_registry):
    """Use expanded scenario metadata and the campaign selector, with no campaign seed lookup."""
    from robot_sf.benchmark.camera_ready._config_types import ScenarioCandidateSelection
    from robot_sf.benchmark.campaign.predictive_horizon_preflight import (
        check_campaign_predictive_horizons_preflight,
    )

    cfg = _override_campaign(
        tmp_path,
        {
            "base_config_path": str(ROOT / "configs/algos/predictive_mppi_release_v0_0_8.yaml"),
            "family_overrides": {"bottleneck": {"horizon_steps": 12}},
        },
        [{"name": "safe", "seeds": [1001]}, {"name": "doorway", "seeds": [1001]}],
    )
    included = tmp_path / "included.yaml"
    cfg.scenario_matrix_path.rename(included)
    cfg.scenario_matrix_path.write_text(
        yaml.safe_dump(
            {
                "includes": [included.name],
                "scenario_overrides_by_name": {"doorway": {"metadata": {"family": "bottleneck"}}},
            }
        )
    )
    safe_cfg = replace(cfg, scenario_candidates=ScenarioCandidateSelection(names=("safe",)))
    records = check_campaign_predictive_horizons_preflight(
        safe_cfg, registry_path=predictor_registry
    )
    assert records[0]["contexts"] == [{"scenario": "safe", "family": "nominal"}]
    with pytest.raises(CampaignCheckpointPreflightError, match="family=bottleneck"):
        check_campaign_predictive_horizons_preflight(cfg, registry_path=predictor_registry)


def test_same_scenario_name_different_families_are_checked(tmp_path, predictor_registry):
    """Family resolution must not be deduplicated by scenario identifier alone."""
    cfg = _override_campaign(
        tmp_path,
        {
            "base_config_path": str(ROOT / "configs/algos/predictive_mppi_release_v0_0_8.yaml"),
            "family_overrides": {"bottleneck": {"horizon_steps": 12}},
        },
        [
            {"name": "doorway", "family": "nominal", "seeds": [1001]},
            {"name": "doorway", "family": "bottleneck", "seeds": [1001]},
        ],
    )
    with pytest.raises(CampaignCheckpointPreflightError, match="family=bottleneck"):
        check_campaign_arm_checkpoints_preflight(cfg, registry_path=predictor_registry)


def test_census_classifies_missing_checkpoint_and_unfrozen_placeholder(
    tmp_path, predictor_registry
):
    """Unknown inputs need reproducible reasons and inventory, rather than cache-dependent totals."""
    from scripts.benchmark.scan_predictive_horizons import scan_predictive_horizons

    missing = tmp_path / "missing.yaml"
    missing.write_text(yaml.safe_dump({"predictive_checkpoint_path": str(tmp_path / "absent.pt")}))
    placeholder = tmp_path / "placeholder.yaml"
    placeholder.write_text(yaml.safe_dump({"release_parameter_freeze": {"status": "unfrozen"}}))
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "planners": [
                    {"key": "missing", "algo": "predictive_mppi", "algo_config": str(missing)},
                    {"key": "unfrozen", "algo": "predictive_mppi", "algo_config": str(placeholder)},
                    {
                        "key": "present",
                        "algo": "predictive_mppi",
                        "algo_config": str(
                            ROOT / "configs/algos/predictive_mppi_release_v0_0_8.yaml"
                        ),
                    },
                ]
            }
        )
    )
    scan = scan_predictive_horizons(tmp_path, registry_path=predictor_registry)
    assert scan["unverified_reasons"] == {"missing_checkpoint": 1, "unfrozen_placeholder": 1}
    inventory = {r["checkpoint"]: r for r in scan["checkpoint_inventory"]}
    assert inventory[str(tmp_path / "absent.pt")]["availability"] == "missing_checkpoint"
    assert inventory["predictive_proxy_selected_v1"]["availability"] == "present"
    assert len(inventory["predictive_proxy_selected_v1"]["sha256"]) == 64


@pytest.mark.parametrize(
    "name,selected_family",
    [
        ("francis2023_doorway", "francis2023"),
        ("classic_doorway", "classic"),
    ],
)
def test_runtime_family_prefix_overrides_metadata(
    tmp_path, predictor_registry, name, selected_family
):
    """Historical identifier prefixes take precedence over an authored metadata family."""
    cfg = _override_campaign(
        tmp_path,
        {
            "base_config_path": str(ROOT / "configs/algos/predictive_mppi_release_v0_0_8.yaml"),
            "family_overrides": {selected_family: {"horizon_steps": 12}},
        },
        [{"name": name, "metadata": {"family": "bottleneck"}, "seeds": [1001]}],
    )
    with pytest.raises(CampaignCheckpointPreflightError, match=f"family={selected_family}"):
        check_campaign_arm_checkpoints_preflight(cfg, registry_path=predictor_registry)


def test_census_resolves_scenario_matrix_and_candidate_selection(tmp_path, predictor_registry):
    """The scanner must use the campaign's scenario matrix, not its planner-roster YAML."""
    from scripts.benchmark.scan_predictive_horizons import scan_predictive_horizons

    cfg = _override_campaign(
        tmp_path,
        {
            "base_config_path": str(ROOT / "configs/algos/predictive_mppi_release_v0_0_8.yaml"),
            "family_overrides": {"bottleneck": {"horizon_steps": 12}},
        },
        [
            {"name": "safe", "seeds": [1001]},
            {"name": "doorway", "family": "bottleneck", "seeds": [1001]},
        ],
    )
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "scenario_matrix": cfg.scenario_matrix_path.name,
                "scenario_candidates": ["safe"],
                "planners": [
                    {
                        "key": "candidate",
                        "algo": "predictive_mppi",
                        "algo_config": cfg.planners[0].algo_config_path.name,
                    }
                ],
            }
        )
    )
    scan = scan_predictive_horizons(tmp_path, registry_path=predictor_registry)
    assert (scan["compatible"], scan["incompatible"], scan["unverified"]) == (1, 0, 0)
    assert scan["bindings"][0]["contexts"] == [{"scenario": "safe", "family": "nominal"}]
