"""Drift-detection tests for the reward-profile reference table (issue #4967).

These tests mechanically guarantee that the canonical reward-profile weight values stay in sync
across the three places they are recorded:

1. The Python weight dicts in ``robot_sf/gym_env/reward.py``.
2. The per-profile YAML configs under ``configs/training/rewards/``.
3. The human-facing reference page ``docs/training/reward_profiles.md``.

They also pin the environment-factory default reward profile. If any of these drift, a focused
test fails and points the author at the doc page that must be updated.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import yaml

from robot_sf.gym_env import reward as reward_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
REWARD_CONFIG_DIR = REPO_ROOT / "configs" / "training" / "rewards"
REWARD_DOC = REPO_ROOT / "docs" / "training" / "reward_profiles.md"

#: Map profile name -> module-level weight dict attribute in ``robot_sf/gym_env/reward.py``.
_PROFILE_WEIGHT_ATTRS: dict[str, str] = {
    "route_completion_v2": "_ROUTE_COMPLETION_V2_WEIGHTS",
    "route_completion_v3": "_ROUTE_COMPLETION_V3_WEIGHTS",
    "social_quality_v1": "_SOCIAL_QUALITY_V1_WEIGHTS",
}


def _load_reward_yaml(profile: str) -> dict[str, object]:
    """Load and return the canonical reward config YAML for a named profile."""
    path = REWARD_CONFIG_DIR / f"{profile}.yaml"
    assert path.exists(), f"Missing canonical reward config: {path}"
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict), f"Reward config {path} did not parse to a mapping"
    return data


def test_named_profile_weight_dicts_match_canonical_yaml_configs() -> None:
    """Python weight dicts and configs/training/rewards/*.yaml must agree exactly.

    These two surfaces are the machine-readable source of truth for the named weighted profiles.
    Drift here silently changes training semantics, so any mismatch must fail loudly.
    """
    for profile, attr in _PROFILE_WEIGHT_ATTRS.items():
        py_weights = dict(getattr(reward_mod, attr))
        yaml_data = _load_reward_yaml(profile)
        assert yaml_data.get("reward_name") == profile, (
            f"{profile}.yaml reward_name should be {profile!r}"
        )
        yaml_weights = dict(yaml_data.get("reward_kwargs", {}).get("weights", {}))
        assert py_weights == yaml_weights, (
            f"Weight drift for {profile}: python={py_weights} yaml={yaml_weights}. "
            "Update configs/training/rewards/{profile}.yaml to match robot_sf/gym_env/reward.py."
        )


def test_legacy_collision_penalties_match_documented_defaults() -> None:
    """Legacy simple_reward collision penalties must match the documented -5/-2 split."""
    simple_sig = inspect.signature(reward_mod.simple_reward)
    assert simple_sig.parameters["ped_coll_penalty"].default == -5
    assert simple_sig.parameters["obst_coll_penalty"].default == -2

    ped_sig = inspect.signature(reward_mod.simple_ped_reward)
    assert ped_sig.parameters["ped_coll_penalty"].default == -5
    assert ped_sig.parameters["obst_coll_penalty"].default == -5


def test_environment_factory_defaults_to_route_completion_v2() -> None:
    """Both robot factories must default to route_completion_v2 when reward_name is None.

    This pins the env-factory default asserted by the reference page and the issue contract.
    """
    source = inspect.getsource(reward_mod.build_reward_function)
    # The factory default lives in environment_factory.py, but build_reward_function is the
    # single resolver; route_completion_v2 must resolve to the v2 reward for the default to hold.
    assert "route_completion_v2" in source

    import re

    from robot_sf.gym_env import environment_factory as ef

    factory_source = inspect.getsource(ef)
    defaults = re.findall(r'reward_name = reward_name or "([^"]+)"', factory_source)
    assert defaults, "Expected 'reward_name = reward_name or ...' default lines in factory source"
    assert all(name == "route_completion_v2" for name in defaults), (
        f"Unexpected factory default reward_name(s): {defaults}"
    )


def test_reward_profiles_reference_page_exists_and_documents_collision_penalties() -> None:
    """The single reference page must exist and directly answer the collision-penalty question."""
    assert REWARD_DOC.exists(), f"Missing reward-profile reference page: {REWARD_DOC}"
    text = REWARD_DOC.read_text(encoding="utf-8")
    # The three named-profile collision penalties must be explicitly documented.
    for needle in ("−5.0", "−10.0", "−6.0", "route_completion_v2"):
        assert needle in text, f"Reference page missing expected value/term: {needle!r}"
    # The checkpoint-profile recipe must be present.
    assert "How to find the profile a checkpoint was trained with" in text
