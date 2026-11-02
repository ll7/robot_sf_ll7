"""Keep the historical campaign contract while routing development screening separately."""

from hashlib import sha256
from pathlib import Path

from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from scripts.tools.snapshot_campaign_horizons import campaign_admission

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = "configs/benchmarks/issue_9533_guarded_ppo_progress_escape.yaml"
DEVELOPMENT = "configs/benchmarks/issue_9533_guarded_ppo_progress_escape_dev_v2.yaml"


def test_dev_screening_preserves_historical_campaign_inventory_and_limits() -> None:
    """A separate dev route retains source limits without replacing the historical campaign."""
    development = load_campaign_config(ROOT / DEVELOPMENT, repository_root=ROOT)
    assert development.seed_policy.seed_set == "dev"
    assert (
        sha256((ROOT / HISTORICAL).read_bytes()).hexdigest()
        == "9e106cf0da01d180338e8ba88dbaad7686be9e7823b2102e8cc0802967019eab"
    )
    historical = campaign_admission(ROOT / HISTORICAL, ROOT)
    screened = campaign_admission(ROOT / DEVELOPMENT, ROOT)
    assert historical["admitted"] and screened["admitted"]
    historical_limits = {row["scenario"]: row["max_episode_steps"] for row in historical["limits"]}
    screened_limits = {row["scenario"]: row["max_episode_steps"] for row in screened["limits"]}
    assert len(screened_limits) == 16
    assert screened_limits.keys() < historical_limits.keys()
    assert all(limit == historical_limits[name] for name, limit in screened_limits.items())
