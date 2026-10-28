"""Generate the campaign configs for the matched pedestrian-perception study.

One camera-ready campaign per (stage, condition). Every campaign runs the same four
planner configurations on the same six release cells and the same seeds; only the
``observation_noise.pedestrian_perception`` block differs. Run from the repository root:

    uv run python docs/context/perception_degradation/make_configs.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import yaml

PACKET = Path("docs/context/perception_degradation")
INPUTS = PACKET / "inputs"

CELLS = [
    "francis2023_blind_corner",
    # Pilot qualification: classic_urban_crossing_medium came within 2.5 m of a pedestrian
    # in 53% of oracle pilot episodes; the medium-density classic crossing map did in 93%.
    "classic_cross_trap_medium",
    "classic_head_on_corridor_medium",
    "classic_overtaking_medium",
    "classic_doorway_medium",
    "classic_bottleneck_high",
]
# Pilot ran urban crossing as the original crossing cell plus the candidate below.
PILOT_CELLS = [*CELLS, "classic_urban_crossing_medium"]

CONDITIONS = {
    "oracle": {"condition": "oracle"},
    "noise": {"condition": "position_noise", "position_noise_std_m": 0.05},
    "delay": {"condition": "delay", "delay_steps": 2},
    "occlusion": {"condition": "static_occlusion"},
}


class Stage(TypedDict):
    """Seed and trace settings for one campaign stage."""

    seeds: list[int]
    trace: bool


STAGES: dict[str, Stage] = {
    # Smoke and pilot/qualification: development seeds 1001-1010, never pooled with main.
    "smoke": {"seeds": [1001], "trace": True},
    "pilot": {"seeds": list(range(1001, 1011)), "trace": True},
    # Main: the 50 orchestrator-allocated development seeds.
    "main": {"seeds": list(range(1121, 1131)) + list(range(1161, 1201)), "trace": False},
}

PLANNERS = [
    {
        "key": "orca",
        "algo": "orca",
        "planner_group": "core",
        "benchmark_profile": "baseline-safe",
        "algo_config": "configs/algos/orca_release_v0_0_8.yaml",
        "socnav_missing_prereq_policy": "fail-fast",
    },
    {
        "key": "hybrid_v4_continuous_guard_on",
        "algo": "hybrid_rule_local_planner",
        "planner_group": "experimental",
        "benchmark_profile": "experimental",
        "algo_config": (
            "configs/policy_search/candidates/"
            "hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release_0_0_8_frozen.yaml"
        ),
    },
    {
        "key": "hybrid_v4_continuous_guard_off",
        "algo": "hybrid_rule_local_planner",
        "planner_group": "experimental",
        "benchmark_profile": "experimental",
        "algo_config": str(INPUTS / "hybrid_v4_continuous_guard_off.yaml"),
    },
    {
        "key": "ppo",
        "algo": "ppo",
        "planner_group": "experimental",
        "benchmark_profile": "experimental",
        "algo_config": "configs/baselines/ppo_release_robot_0_0_8_cpu.yaml",
        "adapter_impact_eval": True,
    },
]


def campaign(stage: str, condition: str) -> dict:
    """Return one campaign config.

    Returns:
        Mapping written as YAML.
    """
    spec = dict(CONDITIONS[condition])
    if STAGES[stage]["trace"]:
        spec["record_trace"] = True
    return {
        "name": f"xperc_{stage}_{condition}",
        "protocol_version": "0.0.8",
        "paper_facing": False,
        "paper_profile_version": "paper-matrix-v1",
        "scenario_matrix": "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml",
        "scenario_candidates": list(PILOT_CELLS if stage == "pilot" else CELLS),
        "comparability_mapping": "configs/benchmarks/alyassi_comparability_map_v2.yaml",
        "route_clearance_certifications": "configs/benchmarks/route_clearance_certifications_v1.yaml",
        "seed_policy": {"mode": "fixed-list", "seeds": list(STAGES[stage]["seeds"])},
        "snqi_weights": None,
        "snqi_baseline": None,
        "workers": 8,
        "scenario_horizons": "configs/benchmarks/horizon_schedules/release_0_0_8_authored_v1.yaml",
        "scenario_horizons_sha256": (
            "032bbf8ad354ea9394492659aa50188e2bc40078930975ef30cd3ba77208675e"
        ),
        "dt": 0.1,
        "record_forces": True,
        "checkpoint_provenance_enforcement": "error",
        "resume": False,
        "stop_on_failure": True,
        "bootstrap_samples": 300,
        "bootstrap_confidence": 0.95,
        "bootstrap_seed": 123,
        "kinematics_matrix": ["differential_drive"],
        "export_publication_bundle": False,
        "include_videos_in_publication": False,
        "overwrite_publication_bundle": False,
        "repository_url": "https://github.com/ll7/robot_sf_ll7",
        "release_tag": "xperc-unpublished",
        "doi": "unpublished-diagnostic",
        "paper_interpretation_profile": "baseline-ready-core",
        "allow_testing_algorithms": True,
        "record_simulation_step_trace": True,
        "record_planner_decision_trace": False,
        "observation_noise": {"pedestrian_perception": spec},
        "planners": PLANNERS,
    }


def main() -> None:
    """Write every campaign config into ``inputs/``."""
    INPUTS.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        for condition in CONDITIONS:
            path = INPUTS / f"campaign_{stage}_{condition}.yaml"
            path.write_text(yaml.safe_dump(campaign(stage, condition), sort_keys=False))
            print(path)


if __name__ == "__main__":
    main()
