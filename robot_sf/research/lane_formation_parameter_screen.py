"""Preregistered Stage A parameter screen for issue #6969.

This is a small space-filling mechanism screen, not a tuning loop.  Profiles
are generated from frozen bounds before any native row is executed.  The
released-default and literature-typical profiles are retained as fixed
anchors, and no profile is selected for a held-out claim in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.research.emergent_phenomena import (
    LITERATURE_CALIBRATION,
    RELEASED_DEFAULT_CALIBRATION,
    SpeedCalibration,
    released_default_config,
)
from robot_sf.research.lane_formation_reference import (
    CLAIM_BOUNDARY,
    DEFAULT_REFERENCE_SEEDS,
    DEFAULT_SAMPLING_STRIDES,
    ReferenceProtocol,
    run_native_reference,
)

if TYPE_CHECKING:
    from pysocialforce.config import SimulatorConfig

__all__ = [
    "DEFAULT_PARAMETER_SCREEN_PROFILES",
    "PARAMETER_BOUNDS",
    "ParameterProfile",
    "build_space_filling_profiles",
    "run_parameter_screen",
    "summarize_parameter_screen_rows",
]

DEFAULT_PARAMETER_SCREEN_PROFILES = 8
PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "social_force_factor": (2.55, 10.2),
    "interaction_range_gamma": (0.175, 0.7),
    "relaxation_time": (0.25, 1.0),
    "anisotropy_lambda_importance": (1.0, 4.0),
    "agent_radius_m": (0.25, 0.5),
    "desired_speed_mean_mps": (0.65, 1.3),
    "desired_speed_std_mps": (0.0, 0.2),
}


@dataclass(frozen=True)
class ParameterProfile:
    """One frozen Stage A parameter profile."""

    profile_id: str
    design_role: str
    social_force_factor: float
    interaction_range_gamma: float
    relaxation_time: float
    anisotropy_lambda_importance: float
    agent_radius_m: float
    desired_speed_mean_mps: float
    desired_speed_std_mps: float

    def as_dict(self) -> dict[str, str | float]:
        """Return profile parameters and their declared design role."""
        return {
            "profile_id": self.profile_id,
            "design_role": self.design_role,
            "social_force_factor": self.social_force_factor,
            "interaction_range_gamma": self.interaction_range_gamma,
            "relaxation_time": self.relaxation_time,
            "anisotropy_lambda_importance": self.anisotropy_lambda_importance,
            "agent_radius_m": self.agent_radius_m,
            "desired_speed_mean_mps": self.desired_speed_mean_mps,
            "desired_speed_std_mps": self.desired_speed_std_mps,
        }

    def calibration(self) -> SpeedCalibration:
        """Return the desired-speed calibration declared by this profile."""
        return SpeedCalibration(
            name=self.profile_id,
            desired_speed_mean=self.desired_speed_mean_mps,
            desired_speed_std=self.desired_speed_std_mps,
        )

    def simulator_config(self) -> SimulatorConfig:
        """Return a simulator config with only declared Stage A overrides."""
        base = released_default_config()
        scene = replace(base.scene_config, agent_radius=self.agent_radius_m)
        social = replace(
            base.social_force_config,
            factor=self.social_force_factor,
            gamma=self.interaction_range_gamma,
            lambda_importance=self.anisotropy_lambda_importance,
        )
        desired = replace(base.desired_force_config, relaxation_time=self.relaxation_time)
        return replace(
            base,
            scene_config=scene,
            social_force_config=social,
            desired_force_config=desired,
        )


def _profile_from_values(
    profile_id: str,
    design_role: str,
    values: dict[str, float],
) -> ParameterProfile:
    return ParameterProfile(
        profile_id=profile_id,
        design_role=design_role,
        **values,
    )


def _anchor_profiles() -> list[ParameterProfile]:
    base = released_default_config()
    shared = {
        "social_force_factor": float(base.social_force_config.factor),
        "interaction_range_gamma": float(base.social_force_config.gamma),
        "relaxation_time": float(base.desired_force_config.relaxation_time),
        "anisotropy_lambda_importance": float(base.social_force_config.lambda_importance),
        "agent_radius_m": float(base.scene_config.agent_radius),
    }
    return [
        _profile_from_values(
            "anchor_released_default",
            "fixed_released_default_anchor",
            {
                **shared,
                "desired_speed_mean_mps": RELEASED_DEFAULT_CALIBRATION.desired_speed_mean,
                "desired_speed_std_mps": RELEASED_DEFAULT_CALIBRATION.desired_speed_std,
            },
        ),
        _profile_from_values(
            "anchor_literature_typical",
            "fixed_literature_typical_anchor",
            {
                **shared,
                "desired_speed_mean_mps": LITERATURE_CALIBRATION.desired_speed_mean,
                "desired_speed_std_mps": LITERATURE_CALIBRATION.desired_speed_std,
            },
        ),
    ]


def build_space_filling_profiles(
    *,
    n_profiles: int = DEFAULT_PARAMETER_SCREEN_PROFILES,
    seed: int = 6969,
) -> list[ParameterProfile]:
    """Build frozen Latin-hypercube profiles plus two fixed calibration anchors.

    Returns:
        Deterministically ordered profiles.  The first profiles are the
        space-filling cells; anchors are appended and never selected by a
        response-dependent rule.
    """
    if isinstance(n_profiles, bool) or not isinstance(n_profiles, int) or n_profiles < 1:
        raise ValueError("n_profiles must be a positive integer")
    rng = np.random.default_rng(seed)
    axis_names = tuple(PARAMETER_BOUNDS)
    unit = np.empty((n_profiles, len(axis_names)), dtype=float)
    for axis_index, _axis_name in enumerate(axis_names):
        unit[:, axis_index] = (rng.permutation(n_profiles) + rng.random(n_profiles)) / n_profiles

    profiles: list[ParameterProfile] = []
    for row_index in range(n_profiles):
        values = {
            axis_name: float(
                PARAMETER_BOUNDS[axis_name][0]
                + unit[row_index, axis_index]
                * (PARAMETER_BOUNDS[axis_name][1] - PARAMETER_BOUNDS[axis_name][0])
            )
            for axis_index, axis_name in enumerate(axis_names)
        }
        profiles.append(
            _profile_from_values(f"lhs_{row_index + 1:02d}", "space_filling_stage_a", values)
        )
    profiles.extend(_anchor_profiles())
    return profiles


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def summarize_parameter_screen_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize Stage A cells without ranking or selecting candidates.

    Returns:
        Deterministically sorted per-profile summaries.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["profile"]["profile_id"], []).append(row)
    summaries: list[dict[str, Any]] = []
    for profile_id, records in sorted(grouped.items()):
        profile = records[0]["profile"]
        lsi_values = [float(record["metrics"]["lane_segregation_index"]) for record in records]
        purity_values = [float(record["metrics"]["lane_purity"]) for record in records]
        clear_hits = sum(
            bool(record["threshold_evaluations"]["lane_segregation_index>=0.5"]["meets_threshold"])
            for record in records
        )
        summaries.append(
            {
                "record_type": "lane_formation_parameter_screen_summary.v1",
                "profile": profile,
                "n_seeds": len(records),
                "seeds": sorted(int(record["seed"]) for record in records),
                "metric_stats": {
                    "lane_segregation_index": _stats(lsi_values),
                    "lane_purity": _stats(purity_values),
                },
                "clear_lane_hit_rate": clear_hits / len(records),
                "execution_status_counts": {"native:computed": len(records)},
                "selection_policy": "no_response_dependent_selection_in_stage_a",
            }
        )
    return summaries


def run_parameter_screen(
    *,
    protocol: ReferenceProtocol = ReferenceProtocol(),
    seeds: tuple[int, ...] | list[int] = DEFAULT_REFERENCE_SEEDS,
    n_profiles: int = DEFAULT_PARAMETER_SCREEN_PROFILES,
    profile_seed: int = 6969,
    sampling_strides: tuple[int, ...] | list[int] = DEFAULT_SAMPLING_STRIDES,
) -> dict[str, Any]:
    """Run native Stage A profiles against the mixed sustained-flow condition.

    Returns:
        Manifest, frozen profile table, native rows, and non-ranking summaries.
    """
    protocol.validate()
    if not seeds:
        raise ValueError("seeds must contain at least one value")
    profiles = build_space_filling_profiles(n_profiles=n_profiles, seed=profile_seed)
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        for seed in seeds:
            row = run_native_reference(
                protocol=protocol,
                condition="mixed_sustained_flow",
                seed=int(seed),
                calibration=profile.calibration(),
                sampling_strides=sampling_strides,
                sim_config=profile.simulator_config(),
            )
            row.update(
                {
                    "record_type": "lane_formation_parameter_screen_cell.v1",
                    "stage": "A",
                    "profile": profile.as_dict(),
                    "issue": "robot_sf_ll7#6969",
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
            rows.append(row)
    if any(
        row["execution"]["execution_mode"] != "native" or row["execution"]["status"] != "computed"
        for row in rows
    ):
        raise RuntimeError("parameter screen contains a non-native or non-computed row")
    manifest = {
        "schema_version": "lane_formation_parameter_screen_manifest.v1",
        "issue": "robot_sf_ll7#6969",
        "stage": "A",
        "claim_boundary": CLAIM_BOUNDARY,
        "purpose": (
            "Preregistered space-filling mechanism screen after the metric/reference audit; "
            "no response-dependent candidate selection or released-default change."
        ),
        "protocol": protocol.as_dict(),
        "seeds": [int(seed) for seed in seeds],
        "profile_seed": int(profile_seed),
        "profile_count": len(profiles),
        "profile_ids": [profile.profile_id for profile in profiles],
        "design": {
            "method": "latin_hypercube_plus_fixed_anchors",
            "bounds": PARAMETER_BOUNDS,
            "factor_mapping": {
                "social_force_factor": "SocialForceConfig.factor (A proxy)",
                "interaction_range_gamma": "SocialForceConfig.gamma (B/range proxy)",
                "relaxation_time": "DesiredForceConfig.relaxation_time (tau proxy)",
                "anisotropy_lambda_importance": "SocialForceConfig.lambda_importance",
                "agent_radius_m": "SceneConfig.agent_radius",
                "desired_speed_mean_mps": "SpeedCalibration.desired_speed_mean",
                "desired_speed_std_mps": "SpeedCalibration.desired_speed_std",
            },
            "fixed_anchors": [
                profile.profile_id
                for profile in profiles
                if profile.design_role.startswith("fixed_")
            ],
            "selection_rule": "profiles frozen before execution; no post-hoc tuning or ranking",
        },
        "execution_policy": {
            "condition": "mixed_sustained_flow",
            "allowed_success_modes": ["native:computed"],
            "fallback_degraded_unavailable_policy": "explicit_status_and_fail_closed",
            "row_count": len(rows),
        },
        "released_defaults_changed": False,
        "metric_semantics_changed": False,
    }
    return {
        "schema_version": "lane_formation_parameter_screen.v1",
        "manifest": manifest,
        "profiles": [profile.as_dict() for profile in profiles],
        "rows": rows,
        "summary": summarize_parameter_screen_rows(rows),
    }
