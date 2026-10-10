"""Regression proof for prospective pinned learned-policy execution."""

from pathlib import Path

import pytest

from robot_sf.benchmark.camera_ready._config import load_campaign_config
from robot_sf.benchmark.result_provenance import (
    ProvenanceValidationError,
    build_result_provenance_manifest,
    validate_result_provenance_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_1_0.yaml"


def test_campaign_resolves_pinned_mode():
    """The new campaign must actually resolve the opted-in numerical contract."""
    cfg = load_campaign_config(CONFIG)
    assert getattr(cfg, "numerical_mode", None) == "pinned_float64_v1"
    assert cfg.seed_policy.seeds == tuple(range(1001, 1031))


def _manifest(tmp_path, evidence=None):
    out = tmp_path / "episodes.jsonl"
    out.write_text('{"episode_id":"one"}\n')
    schema = ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"
    record = {
        "episode_id": "one",
        "scenario_id": "test",
        "seed": 1001,
        "config_hash": "abc",
        "git_hash": "abc",
        "algorithm_metadata": {},
    }
    if evidence:
        record["algorithm_metadata"]["numerical_mode"] = evidence
    return build_result_provenance_manifest(
        out_path=out,
        episode_records=[record],
        schema_path=schema,
        scenario_path=ROOT / "configs/scenarios/classic_interactions.yaml",
        scenarios=[{"name": "test", "seeds": [1001]}],
        algo="ppo",
        algo_config_path=ROOT / "configs/baselines/ppo_release_robot_0_1_0_cpu.yaml",
        benchmark_profile="experimental",
        suite_key="test",
        total_jobs=1,
        written=1,
        horizon=600,
        dt=0.1,
        record_forces=True,
        active_observation_mode="dict",
        active_observation_level="full",
    )


def test_manifest_records_actor_mode(tmp_path):
    """Retained actor evidence must survive into the manifest, including dtype."""
    evidence = {"mode": "pinned_float64_v1", "inference_dtype": "float64"}
    manifest = _manifest(tmp_path, evidence)
    assert manifest["run"].get("numerical_mode") == evidence
    assert manifest["rows"][0].get("numerical_mode") == evidence


def test_manifest_rejects_false_pinned_claim(tmp_path):
    """An unpinned run cannot acquire pinned status by editing its manifest."""
    manifest = _manifest(tmp_path)
    validate_result_provenance_manifest(manifest)
    manifest["run"]["numerical_mode"] = {"mode": "pinned_float64_v1", "inference_dtype": "float64"}
    with pytest.raises(ProvenanceValidationError, match="Pinned numerical mode"):
        validate_result_provenance_manifest(manifest)


def test_float64_convolution_matches_independent_torch():
    """The expensive CNN branch must use the same padding, stride and weights."""
    import numpy as np
    import torch

    from robot_sf.baselines.pinned_actor import _compile

    rng = np.random.default_rng(1001)
    module = torch.nn.Conv2d(3, 4, 3, stride=2, padding=1).double()
    with torch.no_grad():
        module.weight.copy_(torch.from_numpy(rng.normal(size=(4, 3, 3, 3))))
        module.bias.copy_(torch.from_numpy(rng.normal(size=4)))
    data = rng.normal(size=(2, 3, 11, 13))
    with torch.no_grad():
        expected = module(torch.from_numpy(data)).numpy()
    actual = _compile(module)(data)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=2e-14)


def test_pinned_actor_rejects_unsupported_layers():
    """New architectures cannot silently bypass the pinned numerical policy."""
    import torch

    from robot_sf.baselines.pinned_actor import _compile

    with pytest.raises(ValueError, match="Unsupported pinned actor layer"):
        _compile(torch.nn.LayerNorm(4))


@pytest.mark.parametrize(
    "field,value",
    [
        ("mkldnn_enabled", True),
        ("deterministic_algorithms", False),
        ("torch_cpu_capability", "AVX512"),
        ("inference_dtype", "float32"),
        ("kernel_env", {}),
    ],
)
def test_manifest_rejects_drift_in_observed_mode(tmp_path, field, value, monkeypatch):
    """A label and float64 claim cannot hide drift in effective process settings."""
    from robot_sf import _numerical_mode

    observed = {
        "mode": "pinned_float64_v1",
        "inference_dtype": "float64",
        "kernel_env": dict(_numerical_mode.PINNED_ENV),
        "blas": [{"internal_api": "openblas", "architecture": "Haswell", "num_threads": 1}],
        "torch_cpu_capability": "DEFAULT",
        "mkldnn_enabled": False,
        "deterministic_algorithms": True,
    }
    monkeypatch.setattr(_numerical_mode, "effective_numerical_mode", lambda: dict(observed))
    manifest = _manifest(tmp_path, observed)
    validate_result_provenance_manifest(manifest)
    manifest["rows"][0]["numerical_mode"][field] = value
    with pytest.raises(ProvenanceValidationError, match="Pinned numerical mode"):
        validate_result_provenance_manifest(manifest)


def test_campaign_rejects_unpinned_learned_arm(tmp_path):
    """A campaign cannot declare pinned mode while its PPO arm uses the old config."""
    import yaml

    payload = yaml.safe_load(CONFIG.read_text())
    for planner in payload["planners"]:
        if planner["key"] == "ppo":
            planner["algo_config"] = "configs/baselines/ppo_release_robot_0_0_8_cpu.yaml"
    for key in ("scenario_matrix", "scenario_horizons"):
        payload[key] = str(ROOT / payload[key])
    for planner in payload["planners"]:
        if "algo_config" in planner:
            planner["algo_config"] = str(ROOT / planner["algo_config"])
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="numerical_mode does not match"):
        load_campaign_config(path, repository_root=ROOT)


def test_campaign_manifest_requires_retained_pinned_arm(tmp_path):
    """Run-level claims must be backed by the actual learned-arm manifest."""
    import json

    from robot_sf.benchmark.numerical_mode import validate_campaign_numerical_manifest

    arm = _manifest(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "ppo.jsonl.provenance.json").write_text(json.dumps(arm))
    payload = {
        "numerical_mode": {"mode": "pinned_float64_v1", "inference_dtype": "float64"},
        "planners": [{"algo": "ppo"}],
        "kinematics_matrix": ["differential_drive"],
    }
    with pytest.raises(ValueError, match="does not match retained arm"):
        validate_campaign_numerical_manifest(payload, tmp_path)


def test_full_float64_actor_matches_independent_torch():
    """CNN, social features, ego goal, policy MLP and action clipping agree."""
    import copy
    from types import SimpleNamespace

    import numpy as np
    import torch
    from gymnasium import spaces

    from robot_sf.baselines.pinned_actor import PinnedActor
    from robot_sf.baselines.ppo import PPO  # guarded SB3 initialization
    from robot_sf.feature_extractors.grid_socnav_extractor import GridSocNavExtractor

    assert PPO is not None
    space = spaces.Dict(
        {
            "occupancy_grid": spaces.Box(0, 1, (3, 16, 16), dtype=np.float32),
            "robot_position": spaces.Box(-100, 100, (2,), dtype=np.float32),
            "goal_next": spaces.Box(-100, 100, (2,), dtype=np.float32),
            "robot_heading": spaces.Box(-4, 4, (1,), dtype=np.float32),
        }
    )
    torch.manual_seed(1001)
    extractor = GridSocNavExtractor(
        space, grid_channels=[4], grid_kernel_sizes=[3], socnav_hidden_dims=[8]
    ).eval()
    policy_net = torch.nn.Sequential(
        torch.nn.Linear(extractor.features_dim, 8), torch.nn.Tanh()
    ).eval()
    action_net = torch.nn.Linear(8, 2).eval()
    action_space = spaces.Box(-0.1, 0.1, (2,), dtype=np.float32)
    model = SimpleNamespace(
        observation_space=space,
        action_space=action_space,
        policy=SimpleNamespace(
            pi_features_extractor=extractor,
            mlp_extractor=SimpleNamespace(policy_net=policy_net),
            action_net=action_net,
            squash_output=False,
        ),
    )
    actor = PinnedActor(model)
    refs = [copy.deepcopy(module).double().eval() for module in [extractor, policy_net, action_net]]
    rng = np.random.default_rng(1001)
    for random in [False, True]:
        obs = {
            key: (
                rng.uniform(-0.5, 0.5, value.shape).astype(np.float32)
                if random
                else np.zeros(value.shape, dtype=np.float32)
            )
            for key, value in space.spaces.items()
        }
        tensors = {
            key: torch.from_numpy(value.astype(np.float64))[None] for key, value in obs.items()
        }
        with torch.no_grad():
            expected = refs[2](refs[1](refs[0](tensors))).numpy()
        np.testing.assert_allclose(actor.mean(obs), expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(
            actor.predict(obs),
            np.clip(expected, action_space.low, action_space.high).squeeze(),
            rtol=1e-13,
            atol=1e-14,
        )
        assert actor.mean(obs).dtype == np.float64


@pytest.mark.parametrize("algo", ["prediction_planner", "future_learned_planner"])
def test_pinned_campaign_rejects_unsupported_arm(tmp_path, algo):
    """Unsupported and newly introduced learned arms cannot inherit a pinned claim."""
    import yaml

    payload = yaml.safe_load(CONFIG.read_text())
    payload["planners"].append(
        {
            "key": "unsupported",
            "algo": algo,
            "algo_config": "configs/algos/prediction_planner_camera_ready.yaml",
        }
    )
    for key in ("scenario_matrix", "scenario_horizons"):
        payload[key] = str(ROOT / payload[key])
    for planner in payload["planners"]:
        if "algo_config" in planner:
            planner["algo_config"] = str(ROOT / planner["algo_config"])
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="Unsupported pinned campaign arm"):
        load_campaign_config(path, repository_root=ROOT)


@pytest.mark.parametrize("algo", ["prediction_planner", "future_learned_planner"])
@pytest.mark.parametrize("declared", [True, False])
def test_pinned_run_rejects_unsupported_arm(tmp_path, monkeypatch, algo, declared):
    """Both planned and retained arms are checked, even with a valid PPO manifest."""
    import json

    from robot_sf import _numerical_mode
    from robot_sf.benchmark.numerical_mode import validate_campaign_numerical_manifest

    observed = {
        "mode": "pinned_float64_v1",
        "inference_dtype": "float64",
        "kernel_env": dict(_numerical_mode.PINNED_ENV),
        "blas": [{"internal_api": "openblas", "architecture": "Haswell", "num_threads": 1}],
        "torch_cpu_capability": "DEFAULT",
        "mkldnn_enabled": False,
        "deterministic_algorithms": True,
    }
    monkeypatch.setattr(_numerical_mode, "effective_numerical_mode", lambda: dict(observed))
    arm = _manifest(tmp_path, observed)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "ppo.provenance.json").write_text(json.dumps(arm))
    payload = {"numerical_mode": arm["run"]["numerical_mode"], "planners": [{"algo": "ppo"}]}
    validate_campaign_numerical_manifest(payload, tmp_path)
    if declared:
        payload["planners"].append({"algo": algo})
    else:
        arm["campaign_identity"]["algorithm"] = algo
        (runs / "unsupported.provenance.json").write_text(json.dumps(arm))
    with pytest.raises(ValueError, match="Unsupported pinned campaign arm"):
        validate_campaign_numerical_manifest(payload, tmp_path)


def test_failed_campaign_does_not_claim_pinned_execution(tmp_path, monkeypatch):
    """All three final files distinguish an unvalidated request from execution evidence."""
    import json
    from types import SimpleNamespace

    from robot_sf.benchmark.camera_ready import campaign

    claim = {"mode": "pinned_float64_v1", "inference_dtype": "float64"}
    paths = SimpleNamespace(
        campaign_root=tmp_path,
        git_meta={},
        scenario_hash="abc",
        reports_dir=tmp_path,
        manifest_payload={"numerical_mode": claim, "numerical_kernel_context": {}},
    )
    outcome = SimpleNamespace(
        benchmark_success=False,
        runtime_sec=1,
        total_episodes=0,
        campaign_finished_at_utc="2026-10-08T00:00:00Z",
    )
    monkeypatch.setattr(campaign, "_build_run_meta", lambda *args, **kwargs: {})
    tables = dict.fromkeys(
        (
            "seed_variability_json_path",
            "seed_variability_csv_path",
            "seed_episode_rows_csv_path",
            "statistical_sufficiency_json_path",
        ),
        tmp_path / "unused",
    )
    campaign._write_run_level_files(
        SimpleNamespace(numerical_mode="pinned_float64_v1"),
        paths=paths,
        outcome=outcome,
        snqi=None,
        seed_variability_payload={},
        invoked_command=None,
        table_paths=tables,
    )
    for name in ("run_meta.json", "manifest.json", "campaign_manifest.json"):
        payload = json.loads((tmp_path / name).read_text())
        assert "numerical_mode" not in payload, name
        assert "numerical_kernel_context" not in payload, name
        assert payload["requested_numerical_mode"] == claim, name
        assert payload["numerical_mode_validation"] == "unvalidated", name
