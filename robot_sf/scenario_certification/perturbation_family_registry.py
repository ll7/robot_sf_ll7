"""Reusable perturbation-family registry for #1610 criticality workflows.

Each family records its semantic boundary, target surface, validity constraints,
fail-closed rules, and required parameter keys so downstream writers (preflight,
criticality_summary) can stay consistent without hardcoding family knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class PerturbationFamily:
    """Reusable registry entry for one perturbation family."""

    name: str
    description: str
    target_surface: str
    semantic_boundary: str
    validity_constraints: tuple[str, ...]
    fail_closed_rules: tuple[str, ...]
    required_parameters: tuple[str, ...]
    optional_parameters: tuple[str, ...] = ()


_PERTURBATION_FAMILIES: tuple[PerturbationFamily, ...] = (
    PerturbationFamily(
        name="noop",
        description="Unperturbed baseline row for paired comparisons",
        target_surface="none",
        semantic_boundary="fixed identity; must be certified as identity variant",
        validity_constraints=("require_scenario_certification",),
        fail_closed_rules=(
            "noop must carry identical scenario to source; any parameter override is invalid",
        ),
        required_parameters=(),
    ),
    PerturbationFamily(
        name="robot_route_offset",
        description="Bounded (dx_m, dy_m) shift applied to selected robot route waypoints",
        target_surface="robot_route_waypoints",
        semantic_boundary="euclidean distance from original waypoints ≤ min(variant_max_magnitude_m, manifest_max_route_offset_m)",
        validity_constraints=(
            "max_route_offset_m",
            "waypoint_selector_all_only",
        ),
        fail_closed_rules=(
            "magnitude_m > bound excludes variant",
            "no robot routes in source map excludes variant",
            "route offset outside map bounds excludes variant",
        ),
        required_parameters=("dx_m", "dy_m", "max_magnitude_m"),
        optional_parameters=("spawn_id", "goal_id", "waypoint_selector"),
    ),
    PerturbationFamily(
        name="pedestrian_route_offset",
        description="Bounded (dx_m, dy_m) shift applied to selected pedestrian route waypoints",
        target_surface="pedestrian_route_waypoints",
        semantic_boundary="euclidean distance from original waypoints ≤ min(variant_max_magnitude_m, manifest_max_route_offset_m)",
        validity_constraints=(
            "max_route_offset_m",
            "waypoint_selector_all_only",
        ),
        fail_closed_rules=(
            "magnitude_m > bound excludes variant",
            "no pedestrian routes in source map excludes variant",
            "route offset outside map bounds excludes variant",
        ),
        required_parameters=("dx_m", "dy_m", "max_magnitude_m"),
        optional_parameters=("spawn_id", "goal_id", "waypoint_selector"),
    ),
    PerturbationFamily(
        name="single_pedestrian_start_delay_offset",
        description="Bounded start-delay offset for explicit single_pedestrians",
        target_surface="single_pedestrian_start_delay_s",
        semantic_boundary="|dt_s| ≤ min(variant_max_abs_dt_s, manifest_max_start_delay_offset_s); updated delay ≥ 0 s",
        validity_constraints=(
            "max_start_delay_offset_s",
            "pedestrian_selector_all_only",
        ),
        fail_closed_rules=(
            "abs_dt_s > bound excludes variant",
            "no single pedestrians in source scenario excludes variant",
            "updated start_delay_s < 0 excludes variant",
        ),
        required_parameters=("dt_s", "max_abs_dt_s"),
        optional_parameters=("pedestrian_id", "pedestrian_selector"),
    ),
    PerturbationFamily(
        name="single_pedestrian_speed_offset",
        description="Bounded speed offset for explicit single_pedestrians",
        target_surface="single_pedestrian_speed_m_s",
        semantic_boundary="|speed_delta_m_s| ≤ min(variant_max_abs_speed_delta_m_s, manifest_max_single_pedestrian_speed_delta_m_s); updated speed > 0 and ≤ min(optional caps)",
        validity_constraints=(
            "max_single_pedestrian_speed_delta_m_s",
            "pedestrian_selector_all_only",
        ),
        fail_closed_rules=(
            "abs_speed_delta_m_s > bound excludes variant",
            "no single pedestrians in source scenario excludes variant",
            "updated speed_m_s ≤ 0 excludes variant",
            "updated speed_m_s > max_speed_m_s excludes variant",
        ),
        required_parameters=("speed_delta_m_s", "max_abs_speed_delta_m_s"),
        optional_parameters=("pedestrian_id", "pedestrian_selector", "max_speed_m_s"),
    ),
    PerturbationFamily(
        name="single_pedestrian_wait_duration_offset",
        description="Bounded wait_at.wait_s offset for explicit single-pedestrian wait entries",
        target_surface="single_pedestrian_wait_s",
        semantic_boundary="|wait_delta_s| ≤ min(variant_max_abs_wait_delta_s, manifest_max_wait_duration_offset_s); updated wait_s ≥ 0 s",
        validity_constraints=(
            "max_wait_duration_offset_s",
            "pedestrian_selector_all_only",
        ),
        fail_closed_rules=(
            "abs_wait_delta_s > bound excludes variant",
            "no single pedestrians in source scenario excludes variant",
            "no wait_at entries for selected pedestrians excludes variant",
            "updated wait_s < 0 excludes variant",
        ),
        required_parameters=("wait_delta_s", "max_abs_wait_delta_s"),
        optional_parameters=("pedestrian_id", "pedestrian_selector"),
    ),
    PerturbationFamily(
        name="single_pedestrian_trajectory_waypoint_offset",
        description="Bounded (dx_m, dy_m) shift applied to trajectory waypoints for one explicit single_pedestrian",
        target_surface="single_pedestrian_trajectory_waypoints",
        semantic_boundary="euclidean distance from original waypoints ≤ min(variant_max_magnitude_m, manifest_max_trajectory_waypoint_offset_m); updated points inside map bounds",
        validity_constraints=(
            "max_single_pedestrian_trajectory_waypoint_offset_m",
            "waypoint_selector_all_only",
            "pedestrian_id_required",
        ),
        fail_closed_rules=(
            "magnitude_m > bound excludes variant",
            "no single pedestrians in source scenario excludes variant",
            "selected pedestrian has no trajectory excludes variant",
            "updated waypoints outside [0,width]×[0,height] excludes variant",
        ),
        required_parameters=(
            "dx_m",
            "dy_m",
            "max_magnitude_m",
            "pedestrian_id",
            "waypoint_selector",
        ),
    ),
    PerturbationFamily(
        name="pedestrian_density_offset",
        description="Bounded simulation_config.ped_density offset for scenarios with pedestrian routes",
        target_surface="route_pedestrian_density",
        semantic_boundary="|density_delta| ≤ min(variant_max_abs_density_delta, manifest_max_pedestrian_density_delta); updated density ≥ 0 and ≤ min(optional caps)",
        validity_constraints=("max_pedestrian_density_delta",),
        fail_closed_rules=(
            "abs_density_delta > bound excludes variant",
            "no pedestrian routes in source map excludes variant",
            "updated ped_density < 0 excludes variant",
            "updated ped_density > max_ped_density excludes variant",
        ),
        required_parameters=("density_delta", "max_abs_density_delta"),
        optional_parameters=("max_ped_density",),
    ),
)

_FAMILIES_BY_NAME: dict[str, PerturbationFamily] = {f.name: f for f in _PERTURBATION_FAMILIES}


def perturbation_family(name: str) -> PerturbationFamily:
    """Return a registered perturbation family by name.

    Returns:
        PerturbationFamily: Frozen family definition.

    Raises:
        ValueError: When the family is not registered.
    """
    entry = _FAMILIES_BY_NAME.get(name)
    if entry is None:
        raise ValueError(
            f"unsupported perturbation family: {name!r}; "
            f"registered: {sorted(_FAMILIES_BY_NAME.keys())}"
        )
    return entry


def supported_perturbation_families() -> frozenset[str]:
    """Return the set of registered perturbation family names."""
    return frozenset(_FAMILIES_BY_NAME.keys())


def perturbation_families() -> tuple[PerturbationFamily, ...]:
    """Return the ordered tuple of registered perturbation families."""
    return _PERTURBATION_FAMILIES


def validate_perturbation_family_parameters(
    family_name: str,
    parameters: Mapping[str, Any],
) -> tuple[list[str], PerturbationFamily]:
    """Validate variant parameters against a registered family definition.

    Returns:
        tuple[list[str], PerturbationFamily]: Fail-closed reasons and the family entry.

    Validation checks:
        - family is registered
        - required parameters are present
        - extra parameters not in required+optional are rejected
    """
    family = perturbation_family(family_name)
    reasons: list[str] = []
    missing = [key for key in family.required_parameters if key not in parameters]
    if missing:
        reasons.append(f"{family_name} missing required parameters: {sorted(missing)}")
    allowed = set(family.required_parameters).union(family.optional_parameters)
    extra = [key for key in parameters if key not in allowed]
    if extra:
        reasons.append(
            f"{family_name} unrecognized parameters: {sorted(extra)}; allowed: {sorted(allowed)}"
        )
    return reasons, family
