"""Config dataclasses for camera-ready benchmark campaign.

These pure frozen configuration dataclasses and constants are the package-local
owner for the #3385 camera-ready decomposition. Legacy modules re-export these
objects to keep earlier import paths working without behavior changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robot_sf.benchmark.latency_stress import LatencyStressProfile
    from robot_sf.benchmark.synthetic_actuation import SyntheticActuationProfile

DEFAULT_SEED_SETS_PATH = Path("configs/benchmarks/seed_sets_v1.yaml")
_AMV_DIMENSIONS = ("use_case", "context", "speed_regime", "maneuver_type")


@dataclass(frozen=True)
class AmvProfileConfig:
    """AMV paper-profile scope contract settings."""

    name: str = "amv-paper-v1"
    contract_version: str = "1"
    coverage_enforcement: str = "warn"
    required_dimensions: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict.fromkeys(_AMV_DIMENSIONS, ())
    )


@dataclass(frozen=True)
class SeedPolicy:
    """Seed selection policy for campaign scenarios."""

    mode: str = "scenario-default"
    seed_set: str | None = None
    seeds: tuple[int, ...] = ()
    seed_sets_path: Path = DEFAULT_SEED_SETS_PATH


@dataclass(frozen=True)
class ScenarioCandidateSelection:
    """Optional named scenario subset for compact benchmark slices."""

    names: tuple[str, ...] = ()
    selection_name: str | None = None


@dataclass(frozen=True)
class PlannerSpec:
    """One planner entry in a benchmark campaign matrix."""

    key: str
    algo: str
    human_model_variant: str | None = None
    human_model_source: str | None = None
    benchmark_profile: str = "baseline-safe"
    algo_config_path: Path | None = None
    socnav_missing_prereq_policy: str = "fail-fast"
    adapter_impact_eval: bool = False
    observation_mode: str | None = None
    workers_override: int | None = None
    horizon_override: int | None = None
    dt_override: float | None = None
    enabled: bool = True
    planner_group: str = "experimental"
    planner_group_explicit: bool = False


@dataclass(frozen=True)
class SnqiContractConfig:
    """SNQI paper-facing behavior contract configuration."""

    enabled: bool = True
    enforcement: str = "warn"
    rank_alignment_warn_threshold: float = 0.5
    rank_alignment_fail_threshold: float = 0.3
    outcome_separation_warn_threshold: float = 0.05
    outcome_separation_fail_threshold: float = 0.0
    max_component_dominance_warn_threshold: float = 0.24
    max_component_dominance_fail_threshold: float = 0.27
    calibration_seed: int = 123
    calibration_trials: int = 3000


@dataclass(frozen=True)
class CampaignConfig:
    """Top-level camera-ready benchmark campaign config."""

    name: str
    scenario_matrix_path: Path
    planners: tuple[PlannerSpec, ...]
    scenario_candidates: ScenarioCandidateSelection = field(
        default_factory=ScenarioCandidateSelection
    )
    scenario_amv_overrides: dict[str, dict[str, str]] = field(default_factory=dict)
    seed_policy: SeedPolicy = SeedPolicy()
    scenario_horizons_path: Path | None = None
    workers: int = 1
    horizon: int | None = None
    dt: float | None = None
    record_forces: bool = True
    record_planner_decision_trace: bool = False
    record_simulation_step_trace: bool = False
    resume: bool = True
    bootstrap_samples: int = 400
    bootstrap_confidence: float = 0.95
    bootstrap_seed: int = 123
    snqi_weights_path: Path | None = None
    snqi_baseline_path: Path | None = None
    stop_on_failure: bool = False
    export_publication_bundle: bool = True
    include_videos_in_publication: bool = False
    overwrite_publication_bundle: bool = True
    repository_url: str = "https://github.com/ll7/robot_sf_ll7"
    release_tag: str = "{release_tag}"
    doi: str = "10.5281/zenodo.<record-id>"
    paper_interpretation_profile: str = "baseline-ready-core"
    preview_scenario_limit: int = 100
    kinematics_matrix: tuple[str, ...] = ("differential_drive",)
    holonomic_command_mode: str = "vx_vy"
    observation_mode: str | None = None
    paper_facing: bool = False
    paper_profile_version: str | None = None
    amv_profile: AmvProfileConfig = field(default_factory=AmvProfileConfig)
    synthetic_actuation_profile: SyntheticActuationProfile | None = None
    latency_stress_profile: LatencyStressProfile | None = None
    comparability_mapping_path: Path | None = None
    snqi_contract: SnqiContractConfig = field(default_factory=SnqiContractConfig)
    route_clearance_certifications_path: Path | None = None
    observation_noise: dict[str, Any] | None = None
