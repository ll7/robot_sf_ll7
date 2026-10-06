"""author decision of 2026-10-01 (0.0.8 ledger: plain PPO arm replaced by the release-robot retrain): the release's plain PPO slot resolves to the retrained robot policy."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from robot_sf.baselines.ppo import PPOPlanner, PPOPlannerConfig
from robot_sf.benchmark.camera_ready._config import load_campaign_config
from robot_sf.benchmark.policy_search_manifest import resolve_candidate_manifest_runtime
from robot_sf.models import get_registry_entry

MODEL_ID = "ppo_release_robot_b1002_last_20261001"
SHA256 = "764a7d88f5b608237641d973634899e05a67b25e65f8b1607cfca025459824bc"
PARENT_ID = "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200"
FORESIGHT = False
ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / (
    "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
)
DOORWAY_TEMPLATE = ROOT / (
    "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_three_width_doorway_v1.yaml"
)
AUTHORED_TEMPLATE = ROOT / (
    "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml"
)
RELEASE_PPO_PROFILE = ROOT / "configs/baselines/ppo_release_robot_0_0_8_cpu.yaml"


def test_release_ppo_registry_has_training_and_observation_contract():
    """The checkpoint carries the fixed selection digest and its inherited observation track."""
    entry = get_registry_entry(MODEL_ID)
    assert entry["github_release"]["sha256"] == SHA256
    assert entry["sha256"] == SHA256
    assert entry["commit"] == "07bd2b8037053f8e979bd97db32d096fe58e5257"
    assert entry["action_semantics"] == "velocity_delta"
    assert entry["predictive_foresight_enabled"] is FORESIGHT
    assert entry["benchmark_promotion"] == get_registry_entry(PARENT_ID)["benchmark_promotion"]
    assert (ROOT / entry["config_path"]).is_file()


def test_release_robot_model_id_decodes_signed_delta_without_override():
    """The real registered ID must add signed outputs to current physical speed."""
    planner = PPOPlanner(
        PPOPlannerConfig(model_id=MODEL_ID, action_space="unicycle", obs_mode="dict"),
        defer_model_loading=True,
    )
    planner._initialized = True
    planner._model = SimpleNamespace(predict=lambda *a, **k: (np.array([-0.25, -0.1]), None))
    assert planner.step({"robot_speed": [0.6, 0.2]}) == {
        "v": pytest.approx(0.35),
        "omega": pytest.approx(0.1),
    }
    assert planner.get_metadata()["action_semantics"] == "velocity_delta"


@pytest.mark.parametrize(
    "campaign_path",
    [TEMPLATE, DOORWAY_TEMPLATE, AUTHORED_TEMPLATE],
    ids=["main", "doorway", "authored"],
)
def test_release_campaign_resolver_binds_plain_ppo_to_release_robot(campaign_path):
    """Both freeze inputs resolve the PPO profile, model and comparability key."""
    campaign = load_campaign_config(campaign_path, repository_root=ROOT)
    arm = next(p for p in campaign.planners if p.key == "ppo")
    assert arm.algo_config_path == RELEASE_PPO_PROFILE
    raw = yaml.safe_load(arm.algo_config_path.read_text())
    algo, resolved = resolve_candidate_manifest_runtime(
        default_algo=arm.algo,
        manifest=raw,
        scenario={"name": "classic_bottleneck_low"},
        load_config=lambda path: yaml.safe_load((ROOT / path).read_text()),
    )
    assert algo == "ppo"
    assert resolved["model_id"] == MODEL_ID
    assert resolved["predictive_foresight_enabled"] is FORESIGHT
    assert resolved["fallback_to_goal"] is False
    mapping = yaml.safe_load(campaign.comparability_mapping_path.read_text())
    assert mapping["planner_key_mapping"][arm.key] == "ppo"
