"""Side-effect-free readiness probe for the #5303 search-promotion timing controls.

Issue #6475 (child of #6145) requires the frozen adversarial candidate timing dimensions
``spawn_time_s`` and ``pedestrian_delay_s`` to affect a concrete runtime pedestrian instead
of remaining candidate metadata only, so a future powered #6145 preregistration can search a
non-inert space. The latest exact-head review of PR #6291 found the materialized scenario had
``pedestrian_id: null``, no ``single_pedestrians`` slot, one robot route, and zero pedestrian
routes: changing the timing dimensions changed only candidate metadata, never runtime
behavior.

This module is the missing readiness gate. It materializes a deterministic baseline candidate
plus one-at-a-time timing perturbations through the same bundle code path used by search
(:func:`robot_sf.adversarial.bundle.build_candidate_payload`) and proves each declared timing
dimension changes the *effective* runtime scenario and its canonical hash
(:func:`robot_sf.adversarial.bundle.compute_effective_scenario_hash`). The hash strips the
provenance-only ``metadata.adversarial_candidate`` block, so a dimension that survives only in
metadata hashes identically and is rejected.

It is deliberately *side-effect-free*: it builds scenario and route payloads in memory and
hashes them. It runs no search, no planner execution, no replay, no campaign, and inspects no
outcome. It makes no claim that Tree-structured Parzen Estimator search outperforms random
search, and it does not authorize #6145.

Status semantics (fail-closed on any structural gap):

- ``blocked_no_pedestrian`` -- the search space declares no ``pedestrian.id``, the scenario
  template does not expose the declared pedestrian identity, or the materialized payload lacks
  a concrete pedestrian route or ``single_pedestrians`` entry. It also covers a pedestrian-id
  override that does not exactly match the search-space declaration.
- ``blocked_missing_dimension`` -- a frozen promotion timing dimension is not declared in the
  search space.
- ``blocked_inert_dimensions`` -- at least one timing dimension is metadata-only: perturbing it
  leaves the effective runtime scenario and canonical hash unchanged, or fails to bind to the
  pedestrian.
- ``promotion_timing_ready`` -- a concrete pedestrian identity is materialized with a populated
  pedestrian route and ``single_pedestrians`` entry, and every frozen timing dimension changes
  the effective scenario and hash while bound to that pedestrian. This confirms the search space
  is non-inert; it does *not* authorize a campaign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from robot_sf.adversarial.bundle import (
    build_candidate_payload,
    compute_effective_scenario_hash,
    validate_template_pedestrian_binding,
)
from robot_sf.adversarial.config import (
    PROMOTION_TIMING_DIMENSIONS,
    CandidateSpec,
    Pose2D,
    SearchSpaceConfig,
)
from robot_sf.errors import RobotSfError

#: Output-contract schema for this readiness surface.
SCHEMA_VERSION = "issue-5303-search-promotion-preflight.v1"

#: Maximum perturbation applied to one timing dimension at a time. The probe clamps it to the
#: declared range so readiness cannot be proven using a value the search space would never sample.
PERTURBATION_DELTA_S = 1.0

#: Per-dimension runtime field the materialized ``single_pedestrians`` entry must bind the
#: timing value to. ``spawn_time_s`` maps to the pedestrian spawn delay; ``pedestrian_delay_s``
#: maps to the waypoint wait rule. Both are runtime-effective pedestrian controls. The entry
#: is matched by pedestrian identity (see :func:`_single_pedestrian_by_id`), not list position,
#: so the bound field refers to the candidate pedestrian's entry.
_DIMENSION_BOUND_FIELD = {
    "spawn_time_s": "single_pedestrians[matched_id].start_delay_s",
    "pedestrian_delay_s": "single_pedestrians[matched_id].wait_at[0].wait_s",
}

#: Declared downstream gates that keep promotion from being authorized here even when every
#: timing dimension is runtime-effective. Surfaced verbatim so an operator never mistakes
#: ``promotion_timing_ready`` for a go-ahead to run.
CAMPAIGN_GATES: tuple[str, ...] = (
    "this preflight only proves the frozen timing dimensions are runtime-effective; it runs "
    "no search, planner execution, replay, campaign, or outcome inspection",
    "#6145 remains blocked until an adequately powered preregistration is separately reviewed "
    "and approved; this surface does not authorize a promotion campaign",
)


class SearchPromotionPreflightError(RobotSfError, ValueError):
    """Raised when the search space or scenario template cannot be loaded or parsed."""


@dataclass(frozen=True, slots=True)
class TimingDimensionProbe:
    """One-at-a-time perturbation result for a single frozen timing dimension."""

    name: str
    bound_field: str
    declared: bool
    baseline_value: float
    perturbed_value: float
    baseline_hash: str
    perturbed_hash: str
    hash_changed: bool
    bound_value: float | None
    bound_to_pedestrian: bool
    status: str


@dataclass(frozen=True, slots=True)
class SearchPromotionPreflight:
    """Aggregate fail-closed readiness for the #5303 search-promotion timing controls."""

    schema_version: str
    pedestrian_id: str | None
    materialized_pedestrian_id: str | None
    single_pedestrian_populated: bool
    pedestrian_route_populated: bool
    dimensions: tuple[TimingDimensionProbe, ...]
    status: str
    blockers: tuple[str, ...]
    campaign_gates: tuple[str, ...] = CAMPAIGN_GATES

    @property
    def promotion_ready(self) -> bool:
        """True when every frozen timing dimension is runtime-effective and only gates remain."""
        return self.status == "promotion_timing_ready"


def _midpoint(range_min: float, range_max: float) -> float:
    """Return the midpoint of an inclusive range."""
    return 0.5 * (float(range_min) + float(range_max))


def _baseline_candidate(space: SearchSpaceConfig) -> CandidateSpec:
    """Build a deterministic baseline candidate from the search-space range midpoints.

    Returns:
        CandidateSpec: The baseline candidate at the midpoint of every search-space range.
    """
    return CandidateSpec(
        start=Pose2D(
            _midpoint(space.start_x.min, space.start_x.max),
            _midpoint(space.start_y.min, space.start_y.max),
        ),
        goal=Pose2D(
            _midpoint(space.goal_x.min, space.goal_x.max),
            _midpoint(space.goal_y.min, space.goal_y.max),
        ),
        spawn_time_s=_midpoint(space.spawn_time_s.min, space.spawn_time_s.max),
        pedestrian_speed_mps=_midpoint(
            space.pedestrian_speed_mps.min, space.pedestrian_speed_mps.max
        ),
        pedestrian_delay_s=_midpoint(space.pedestrian_delay_s.min, space.pedestrian_delay_s.max),
        scenario_seed=int(space.scenario_seed.min),
    )


def _bounded_perturbed_value(value: float, lower: float, upper: float) -> float:
    """Return a distinct in-range probe value when the declared range permits one."""
    value = float(value)
    lower = float(lower)
    upper = float(upper)
    if value < upper:
        return min(upper, value + PERTURBATION_DELTA_S)
    if value > lower:
        return max(lower, value - PERTURBATION_DELTA_S)
    return value


def _single_pedestrian_by_id(
    scenario: dict[str, Any], pedestrian_id: str | None
) -> dict[str, Any] | None:
    """Return the ``single_pedestrians`` entry bound to ``pedestrian_id``, if any.

    The entry is looked up by identity rather than list position so a template that already
    defines other pedestrians (listed before the candidate pedestrian) cannot cause the probe
    to inspect the wrong entry. Only the candidate pedestrian is probed, because that is the
    pedestrian the frozen timing dimensions are bound to.
    """
    if not pedestrian_id:
        return None
    entries = scenario.get("single_pedestrians")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("id") or "").strip() == pedestrian_id:
            return entry
    return None


def _pedestrian_route_by_id(
    route_payload: dict[str, Any], pedestrian_id: str | None
) -> dict[str, Any] | None:
    """Return the ``ped_routes`` mapping entry bound to ``pedestrian_id``, if any."""
    if not pedestrian_id:
        return None
    entries = route_payload.get("ped_routes")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("id") or "").strip() == pedestrian_id:
            return entry
    return None


def _materialized_binding_status(
    *,
    template: dict[str, Any],
    scenario: dict[str, Any],
    route_payload: dict[str, Any],
    pedestrian_id: str | None,
) -> tuple[str | None, bool, bool, bool, list[str]]:
    """Check that the materialized candidate has loader-bound pedestrian surfaces.

    Returns:
        tuple[str | None, bool, bool, bool, list[str]]: Materialized id, single-pedestrian
        status, route status, template-binding status, and surfaced blockers.
    """
    template_bound_ped = _single_pedestrian_by_id(template, pedestrian_id)
    bound_ped = _single_pedestrian_by_id(scenario, pedestrian_id)
    bound_route = _pedestrian_route_by_id(route_payload, pedestrian_id)
    materialized_id = (
        str(bound_ped["id"]).strip()
        if template_bound_ped and bound_ped and bound_ped.get("id")
        else None
    )
    single_pedestrian_populated = (
        bool(template_bound_ped) and bool(pedestrian_id) and materialized_id == pedestrian_id
    )
    pedestrian_route_populated = (
        bool(template_bound_ped) and bool(pedestrian_id) and bool(bound_route)
    )

    blockers: list[str] = []
    if not pedestrian_id:
        blockers.append(
            "search space declares no pedestrian.id; the frozen timing dimensions target no "
            "concrete pedestrian and the materialized scenario has no populated pedestrian "
            "route or single_pedestrians entry"
        )
    elif template_bound_ped is None:
        blockers.append(
            f"scenario template has no single_pedestrians entry for pedestrian.id={pedestrian_id!r}; "
            "the side-effect-free preflight does not load a map to infer an override target"
        )
    if pedestrian_id and not single_pedestrian_populated:
        blockers.append(
            "materialized scenario does not contain a loader-bound single_pedestrians entry "
            f"for pedestrian.id={pedestrian_id!r}"
        )
    if pedestrian_id and not pedestrian_route_populated:
        blockers.append(
            "materialized route payload does not contain a pedestrian route bound to "
            f"pedestrian.id={pedestrian_id!r}"
        )
    return (
        materialized_id,
        single_pedestrian_populated,
        pedestrian_route_populated,
        template_bound_ped is not None,
        blockers,
    )


def _extract_bound_value(
    scenario: dict[str, Any], name: str, *, pedestrian_id: str | None
) -> float | None:
    """Extract the runtime field a timing dimension must bind to from a materialized scenario.

    The lookup targets the candidate pedestrian (``pedestrian_id``) so a template that already
    defines other pedestrians cannot cause the wrong entry to be inspected.

    Returns:
        float | None: The bound runtime value, or ``None`` when the scenario does not bind the
        dimension to the candidate pedestrian.
    """
    entry = _single_pedestrian_by_id(scenario, pedestrian_id)
    if entry is None:
        return None
    if name == "spawn_time_s":
        raw = entry.get("start_delay_s")
    elif name == "pedestrian_delay_s":
        wait_at = entry.get("wait_at")
        if not isinstance(wait_at, list) or not wait_at or not isinstance(wait_at[0], dict):
            return None
        raw = wait_at[0].get("wait_s")
    else:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _probe_dimension(
    name: str,
    *,
    space: SearchSpaceConfig,
    baseline: CandidateSpec,
    baseline_hash: str,
    template_scenario: dict[str, Any],
    pedestrian_id: str | None,
    index: int,
) -> TimingDimensionProbe:
    """Materialize a one-at-a-time perturbation of one timing dimension and classify it.

    Returns:
        TimingDimensionProbe: The perturbation result, classified ``effective``,
        ``inert_metadata_only``, ``no_pedestrian``, or ``missing``.
    """
    bound_field = _DIMENSION_BOUND_FIELD.get(name, f"<unbound:{name}>")
    declared = space.timing_dimension_range(name) is not None

    if not declared:
        return TimingDimensionProbe(
            name=name,
            bound_field=bound_field,
            declared=False,
            baseline_value=0.0,
            perturbed_value=0.0,
            baseline_hash=baseline_hash,
            perturbed_hash=baseline_hash,
            hash_changed=False,
            bound_value=None,
            bound_to_pedestrian=False,
            status="missing",
        )

    timing_range = space.timing_dimension_range(name)
    if timing_range is None:
        # Keep this defensive branch aligned with the declared check above if the registry or
        # config implementation changes between the two lookups.
        return TimingDimensionProbe(
            name=name,
            bound_field=bound_field,
            declared=False,
            baseline_value=0.0,
            perturbed_value=0.0,
            baseline_hash=baseline_hash,
            perturbed_hash=baseline_hash,
            hash_changed=False,
            bound_value=None,
            bound_to_pedestrian=False,
            status="missing",
        )

    baseline_value = float(getattr(baseline, name))
    perturbed_value = _bounded_perturbed_value(
        baseline_value,
        timing_range.min,
        timing_range.max,
    )
    perturbed = replace(baseline, **{name: perturbed_value})
    perturbed_scenario, perturbed_route = build_candidate_payload(
        perturbed,
        index=index,
        template_scenario=template_scenario,
        pedestrian_id=pedestrian_id,
    )
    perturbed_hash = compute_effective_scenario_hash(perturbed_scenario, perturbed_route)
    hash_changed = perturbed_hash != baseline_hash
    bound_value = _extract_bound_value(perturbed_scenario, name, pedestrian_id=pedestrian_id)
    bound_to_pedestrian = bound_value is not None and math.isclose(
        bound_value, perturbed_value, rel_tol=0.0, abs_tol=1e-9
    )

    if pedestrian_id is None:
        status = "no_pedestrian"
    elif hash_changed and bound_to_pedestrian:
        status = "effective"
    else:
        status = "inert_metadata_only"

    return TimingDimensionProbe(
        name=name,
        bound_field=bound_field,
        declared=True,
        baseline_value=baseline_value,
        perturbed_value=perturbed_value,
        baseline_hash=baseline_hash,
        perturbed_hash=perturbed_hash,
        hash_changed=hash_changed,
        bound_value=bound_value,
        bound_to_pedestrian=bound_to_pedestrian,
        status=status,
    )


def evaluate_preflight(
    *,
    search_space: SearchSpaceConfig,
    template_scenario: dict[str, Any],
    pedestrian_id: str | None = None,
    index: int = 0,
) -> SearchPromotionPreflight:
    """Evaluate fail-closed readiness of the frozen timing controls, side-effect-free.

    Args:
        search_space: Validated adversarial search space whose timing dimensions are probed.
        template_scenario: The first scenario mapping from a scenario template, materialized
            in memory (no disk I/O required).
        pedestrian_id: Optional pedestrian identity assertion. When provided, it must match
            ``search_space.pedestrian_id``; it cannot supply or replace a missing declaration.
            A missing declaration or mismatch fails closed with ``blocked_no_pedestrian``.
        index: Candidate index used for materialization; held constant across perturbations so
            only the timing dimension varies.

    Returns:
        Aggregate preflight with a fail-closed status and surfaced blockers.
    """
    raw_resolved_id = search_space.pedestrian_id if pedestrian_id is None else pedestrian_id
    declared_id = str(search_space.pedestrian_id).strip() if search_space.pedestrian_id else None
    resolved_id = str(raw_resolved_id).strip() if raw_resolved_id is not None else None
    resolved_id = resolved_id or None
    override_mismatch = pedestrian_id is not None and resolved_id != declared_id
    template = dict(template_scenario)
    template_binding_error = validate_template_pedestrian_binding(template, declared_id)
    materialization_id = declared_id if template_binding_error is None else None
    baseline = _baseline_candidate(search_space)
    baseline_scenario, baseline_route = build_candidate_payload(
        baseline,
        index=index,
        template_scenario=template,
        pedestrian_id=materialization_id,
    )
    baseline_hash = compute_effective_scenario_hash(baseline_scenario, baseline_route)
    (
        materialized_id,
        single_pedestrian_populated,
        pedestrian_route_populated,
        template_has_pedestrian,
        blockers,
    ) = _materialized_binding_status(
        template=template,
        scenario=baseline_scenario,
        route_payload=baseline_route,
        pedestrian_id=declared_id,
    )

    if template_binding_error is not None and template_binding_error not in blockers:
        blockers.append(template_binding_error)
    if override_mismatch:
        blockers.append(
            "pedestrian_id override must match the search-space pedestrian.id exactly; "
            f"declared={declared_id!r}, requested={resolved_id!r}"
        )

    probe_pedestrian_id = (
        declared_id if template_has_pedestrian and template_binding_error is None else None
    )

    dimensions = tuple(
        _probe_dimension(
            name,
            space=search_space,
            baseline=baseline,
            baseline_hash=baseline_hash,
            template_scenario=template,
            pedestrian_id=probe_pedestrian_id,
            index=index,
        )
        for name in PROMOTION_TIMING_DIMENSIONS
    )

    for probe in dimensions:
        if probe.status == "missing":
            blockers.append(
                f"frozen timing dimension {probe.name!r} is not declared in the search space"
            )
        elif probe.status == "inert_metadata_only":
            blockers.append(
                f"timing dimension {probe.name!r} is metadata-only: perturbing it "
                f"({probe.baseline_value} -> {probe.perturbed_value}) left the effective "
                f"runtime scenario unchanged (hash_changed={probe.hash_changed}, "
                f"bound_to_pedestrian={probe.bound_to_pedestrian})"
            )

    if (
        override_mismatch
        or not declared_id
        or not template_has_pedestrian
        or template_binding_error
    ):
        status = "blocked_no_pedestrian"
    elif any(probe.status == "missing" for probe in dimensions):
        status = "blocked_missing_dimension"
    elif (
        any(probe.status == "inert_metadata_only" for probe in dimensions)
        or not single_pedestrian_populated
        or not pedestrian_route_populated
    ):
        status = "blocked_inert_dimensions"
    else:
        status = "promotion_timing_ready"

    return SearchPromotionPreflight(
        schema_version=SCHEMA_VERSION,
        pedestrian_id=declared_id,
        materialized_pedestrian_id=materialized_id,
        single_pedestrian_populated=single_pedestrian_populated,
        pedestrian_route_populated=pedestrian_route_populated,
        dimensions=dimensions,
        status=status,
        blockers=tuple(blockers),
    )


def load_template_scenario(path: str | Path) -> dict[str, Any]:
    """Load the first scenario mapping from a scenario-template YAML file.

    Raises:
        SearchPromotionPreflightError: if the file is missing, unparseable, or has no
            scenario mapping.

    Returns:
        The first scenario mapping.
    """
    template_path = Path(path)
    if not template_path.is_file():
        raise SearchPromotionPreflightError(f"scenario template not found: {template_path}")
    try:
        payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SearchPromotionPreflightError(
            f"could not load scenario template {template_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SearchPromotionPreflightError(f"scenario template must be a mapping: {template_path}")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios or not isinstance(scenarios[0], dict):
        raise SearchPromotionPreflightError(
            f"scenario template must contain a scenario mapping: {template_path}"
        )
    return dict(scenarios[0])


def evaluate_preflight_from_files(
    *,
    search_space_path: str | Path,
    scenario_template_path: str | Path,
    pedestrian_id: str | None = None,
    index: int = 0,
) -> SearchPromotionPreflight:
    """Evaluate readiness from on-disk search-space and scenario-template files (read-only).

    Raises:
        SearchPromotionPreflightError: if either input is missing or unparseable.

    Returns:
        Aggregate preflight result.
    """
    try:
        search_space = SearchSpaceConfig.from_file(search_space_path)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        raise SearchPromotionPreflightError(
            f"could not load search space {search_space_path}: {exc}"
        ) from exc
    template_scenario = load_template_scenario(scenario_template_path)
    return evaluate_preflight(
        search_space=search_space,
        template_scenario=template_scenario,
        pedestrian_id=pedestrian_id,
        index=index,
    )


def to_dict(preflight: SearchPromotionPreflight) -> dict[str, Any]:
    """Return a JSON-serializable view of the preflight result."""
    return {
        "schema_version": preflight.schema_version,
        "status": preflight.status,
        "promotion_ready": preflight.promotion_ready,
        "pedestrian_id": preflight.pedestrian_id,
        "materialized_pedestrian_id": preflight.materialized_pedestrian_id,
        "single_pedestrian_populated": preflight.single_pedestrian_populated,
        "pedestrian_route_populated": preflight.pedestrian_route_populated,
        "dimensions": [
            {
                "name": probe.name,
                "bound_field": probe.bound_field,
                "declared": probe.declared,
                "baseline_value": probe.baseline_value,
                "perturbed_value": probe.perturbed_value,
                "baseline_hash": probe.baseline_hash,
                "perturbed_hash": probe.perturbed_hash,
                "hash_changed": probe.hash_changed,
                "bound_value": probe.bound_value,
                "bound_to_pedestrian": probe.bound_to_pedestrian,
                "status": probe.status,
            }
            for probe in preflight.dimensions
        ],
        "blockers": list(preflight.blockers),
        "campaign_gates": list(preflight.campaign_gates),
    }


def render_markdown(preflight: SearchPromotionPreflight) -> str:
    """Render a compact Markdown report for the preflight result.

    Returns:
        A Markdown string leading with the claim boundary and status.
    """
    lines: list[str] = []
    lines.append("# Issue #5303 search-promotion timing-control preflight")
    lines.append("")
    lines.append(
        "Claim boundary: side-effect-free probe proving the frozen timing dimensions "
        "`spawn_time_s` and `pedestrian_delay_s` change the effective runtime scenario and "
        "its canonical hash. Runs no search, planner execution, replay, campaign, or outcome "
        "inspection; authorizes no promotion campaign."
    )
    lines.append("")
    lines.append(f"- Status: `{preflight.status}`")
    lines.append(f"- Pedestrian identity: `{preflight.pedestrian_id}`")
    lines.append(f"- Materialized pedestrian id: `{preflight.materialized_pedestrian_id}`")
    lines.append(f"- single_pedestrians populated: {preflight.single_pedestrian_populated}")
    lines.append(f"- pedestrian route populated: {preflight.pedestrian_route_populated}")
    lines.append("")
    lines.append("## Timing-dimension probes")
    lines.append("")
    lines.append(
        "| dimension | declared | baseline | perturbed | hash changed | bound to pedestrian | status |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for probe in preflight.dimensions:
        lines.append(
            f"| `{probe.name}` | {probe.declared} | {probe.baseline_value} | "
            f"{probe.perturbed_value} | {probe.hash_changed} | {probe.bound_to_pedestrian} | "
            f"`{probe.status}` |"
        )
    lines.append("")
    if preflight.blockers:
        lines.append("## Blockers (fail-closed)")
        lines.append("")
        for blocker in preflight.blockers:
            lines.append(f"- {blocker}")
        lines.append("")
    lines.append("## Declared campaign gates (remain even when timing is ready)")
    lines.append("")
    for gate in preflight.campaign_gates:
        lines.append(f"- {gate}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "CAMPAIGN_GATES",
    "PERTURBATION_DELTA_S",
    "SCHEMA_VERSION",
    "SearchPromotionPreflight",
    "SearchPromotionPreflightError",
    "TimingDimensionProbe",
    "evaluate_preflight",
    "evaluate_preflight_from_files",
    "load_template_scenario",
    "render_markdown",
    "to_dict",
]
