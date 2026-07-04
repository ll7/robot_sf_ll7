"""Config and scenario-loading helpers for camera-ready campaigns.

Extracted from ``robot_sf.benchmark.camera_ready_campaign`` for #3385 slice 7.
The public behavior stays anchored in ``camera_ready_campaign``, which
re-exports these private helpers for compatibility.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.camera_ready._util import _repo_relative
from robot_sf.benchmark.camera_ready_campaign_config import (
    _AMV_DIMENSIONS,
    DEFAULT_SEED_SETS_PATH,
    AmvProfileConfig,
    CampaignConfig,
    PlannerSpec,
    ScenarioCandidateSelection,
    SeedPolicy,
    SnqiContractConfig,
)
from robot_sf.benchmark.latency_stress import (
    load_latency_stress_profile,
    validate_latency_stress_profile,
)
from robot_sf.benchmark.synthetic_actuation import (
    SYNTHETIC_ACTUATION_CLAIM_SCOPE,
    SyntheticActuationProfile,
    validate_actuation_profile_claim_boundary,
    validate_synthetic_actuation_profile,
)
from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.training.scenario_loader import load_scenarios

_PLANNER_GROUPS = {"core", "experimental"}
_PAPER_KINEMATICS_BY_PROFILE = {
    "paper-seed-variability-v1": ("differential_drive",),
    "paper-matrix-v1": ("differential_drive",),
    "paper-cross-kinematics-v1": ("differential_drive", "bicycle_drive", "holonomic"),
}
_AMV_COVERAGE_ENFORCEMENT = {"warn", "error"}
_SNQI_CONTRACT_ENFORCEMENT = {"warn", "error", "enforce"}


def _normalize_observation_mode(raw: Any, *, label: str) -> str | None:
    """Return a normalized observation-mode override, rejecting blank strings."""
    if raw is None:
        return None
    normalized = str(raw).strip()
    if not normalized:
        raise ValueError(f"{label} cannot be empty when provided")
    return normalized


def _normalize_kinematics_matrix(raw: Any) -> tuple[str, ...]:
    """Return the campaign kinematics matrix while rejecting null entries."""
    if raw is None:
        return ("differential_drive",)
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list | tuple):
        values = list(raw)
    else:
        raise TypeError("kinematics_matrix must be a string or list of strings")

    normalized: list[str] = []
    for value in values:
        if value is None:
            raise TypeError("kinematics_matrix entries must be strings")
        text = str(value).strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _optional_synthetic_actuation_profile_mapping(
    payload: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any] | None:
    """Return optional synthetic-actuation profile metadata with fail-closed typing."""
    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"synthetic_actuation_profile.{key} must be a mapping when provided")
    return value


def _sanitize_name(name: str) -> str:
    """Normalize names for stable directory identifiers.

    Returns:
        Lowercase identifier containing only letters, digits, underscores, and hyphens.
    """
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower()).strip("_")
    return normalized or "campaign"


def _load_seed_sets(path: Path) -> dict[str, list[int]]:
    """Load seed sets file into a normalized mapping.

    Returns:
        Mapping from seed-set name to integer seed list.
    """
    if not path.exists():
        raise FileNotFoundError(f"Seed sets file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Seed sets file must be a mapping: {path}")
    out: dict[str, list[int]] = {}
    for key, value in payload.items():
        if isinstance(value, list) and value:
            out[str(key)] = [int(seed) for seed in value]
    return out


def _resolve_seed_override(policy: SeedPolicy) -> list[int] | None:
    """Resolve seed override list based on campaign seed policy.

    Returns:
        Seed list override, or ``None`` when scenario-defined seeds should be used.
    """
    mode = policy.mode.strip().lower()
    if mode == "scenario-default":
        return None
    if mode == "fixed-list":
        if not policy.seeds:
            raise ValueError("Seed policy mode 'fixed-list' requires a non-empty seeds list")
        return [int(seed) for seed in policy.seeds]
    if mode == "seed-set":
        if not policy.seed_set:
            raise ValueError("Seed policy mode 'seed-set' requires seed_set")
        seed_sets = _load_seed_sets(policy.seed_sets_path)
        if policy.seed_set not in seed_sets:
            known = ", ".join(sorted(seed_sets))
            raise ValueError(
                f"Unknown seed set '{policy.seed_set}'. Available: {known}",
            )
        return list(seed_sets[policy.seed_set])
    raise ValueError(f"Unsupported seed policy mode: {policy.mode}")


def _campaign_scenario_id(scenario: dict[str, Any]) -> str:
    """Return the stable identifier used to join scenario metadata to campaign sidecars."""
    for key in ("name", "scenario_id", "id"):
        value = scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _scenario_with_kinematics(
    scenario: dict[str, Any],
    *,
    kinematics: str,
    holonomic_command_mode: str,
) -> dict[str, Any]:
    """Return a scenario copy patched for one robot kinematics mode.

    Returns:
        dict[str, Any]: Scenario payload with ``robot_config.type`` set.
    """
    patched = dict(scenario)
    robot_cfg = (
        dict(scenario.get("robot_config")) if isinstance(scenario.get("robot_config"), dict) else {}
    )
    robot_cfg["type"] = kinematics
    if kinematics == "holonomic":
        robot_cfg.setdefault("command_mode", holonomic_command_mode)
    patched["robot_config"] = robot_cfg
    return patched


def _load_scenario_horizon_schedule(path: Path) -> dict[str, dict[str, Any]]:
    """Load and validate a scenario-horizon schedule sidecar.

    Returns:
        Mapping from scenario id to normalized horizon metadata.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Scenario horizon schedule must be a mapping: {path}")
    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, dict) or not raw_scenarios:
        raise ValueError(f"Scenario horizon schedule requires non-empty 'scenarios': {path}")

    schedule: dict[str, dict[str, Any]] = {}
    for scenario_id, raw_entry in raw_scenarios.items():
        sid = str(scenario_id).strip()
        if not sid:
            raise ValueError(f"Scenario horizon schedule contains an empty scenario id: {path}")
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Scenario horizon entry for '{sid}' must be a mapping: {path}")
        try:
            horizon_steps = int(raw_entry["recommended_horizon_steps"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Scenario horizon entry for '{sid}' requires integer recommended_horizon_steps"
            ) from exc
        if horizon_steps <= 0:
            raise ValueError(
                f"Scenario horizon entry for '{sid}' must use a positive horizon, got {horizon_steps}"
            )
        schedule[sid] = {
            "recommended_horizon_steps": horizon_steps,
            "status": str(raw_entry.get("status", "recommended")).strip() or "recommended",
            "bucket": str(raw_entry.get("bucket", "")).strip(),
        }
    return schedule


def _apply_scenario_horizon_schedule(
    scenarios: list[dict[str, Any]],
    *,
    schedule_path: Path | None,
) -> list[dict[str, Any]]:
    """Apply a scenario-specific horizon schedule to scenario max-step limits.

    Returns:
        Scenario list with patched ``simulation_config.max_episode_steps`` and provenance metadata.
    """
    if schedule_path is None:
        return scenarios

    schedule = _load_scenario_horizon_schedule(schedule_path)
    missing = [
        scenario_id
        for scenario in scenarios
        if (scenario_id := _campaign_scenario_id(scenario)) not in schedule
    ]
    if missing:
        preview = ", ".join(sorted(missing)[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise ValueError(
            "Scenario horizon schedule is missing entries for campaign scenarios: "
            f"{preview}{suffix}"
        )

    patched_scenarios: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_id = _campaign_scenario_id(scenario)
        entry = schedule[scenario_id]
        horizon_steps = int(entry["recommended_horizon_steps"])
        patched = deepcopy(scenario)
        simulation_config = patched.setdefault("simulation_config", {})
        if not isinstance(simulation_config, dict):
            raise ValueError(
                f"Scenario '{scenario_id}' simulation_config must be a mapping for horizon patching"
            )
        simulation_config["max_episode_steps"] = horizon_steps

        metadata = patched.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Scenario '{scenario_id}' metadata must be a mapping")
        metadata["scenario_horizon"] = {
            "source": _repo_relative(schedule_path),
            "recommended_horizon_steps": horizon_steps,
            "status": entry["status"],
            "bucket": entry["bucket"],
        }
        patched_scenarios.append(patched)
    return patched_scenarios


def _scenario_horizon_summary(
    scenarios: list[dict[str, Any]],
    *,
    schedule_path: Path | None,
) -> dict[str, Any] | None:
    """Summarize applied scenario-horizon metadata for preflight and manifest artifacts.

    Returns:
        Summary payload when a schedule is configured, otherwise ``None``.
    """
    if schedule_path is None:
        return None

    status_counts: dict[str, int] = {}
    horizons: list[int] = []
    for scenario in scenarios:
        metadata = scenario.get("metadata")
        horizon_meta = metadata.get("scenario_horizon") if isinstance(metadata, dict) else None
        if not isinstance(horizon_meta, dict):
            continue
        status = str(horizon_meta.get("status", "unknown")).strip() or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
        try:
            horizons.append(int(horizon_meta["recommended_horizon_steps"]))
        except (KeyError, TypeError, ValueError):
            continue

    return {
        "path": _repo_relative(schedule_path),
        "scenario_count": len(horizons),
        "min_horizon_steps": min(horizons) if horizons else None,
        "max_horizon_steps": max(horizons) if horizons else None,
        "status_counts": {key: status_counts[key] for key in sorted(status_counts)},
    }


def _filter_scenario_candidates(
    scenarios: list[dict[str, Any]],
    *,
    names: tuple[str, ...],
    matrix_path: Path,
) -> list[dict[str, Any]]:
    """Return the configured compact candidate subset.

    Returns:
        Candidate-filtered scenario list.
    """
    if not names:
        return scenarios
    requested_counts = dict.fromkeys(names, 0)
    filtered: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_name = str(
            scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or ""
        ).strip()
        if scenario_name in requested_counts:
            requested_counts[scenario_name] += 1
            filtered.append(scenario)
    missing = [name for name, count in requested_counts.items() if count <= 0]
    if missing:
        raise ValueError(
            "scenario_candidates did not resolve in "
            f"{_repo_relative(matrix_path)}: {', '.join(missing)}"
        )
    return filtered


def _scenario_name_key(scenario: Mapping[str, Any]) -> str:
    """Return the normalized scenario name key used by campaign selectors."""
    return str(
        scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or ""
    ).strip()


def _validate_scenario_amv_override_keys(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    overrides: Mapping[str, Mapping[str, str]],
    matrix_path: Path,
) -> None:
    """Fail closed when AMV overrides target scenarios outside the loaded slice."""
    if not overrides:
        return

    loaded_names = {_scenario_name_key(scenario) for scenario in scenarios}
    unmatched = sorted(name for name in overrides if name not in loaded_names)
    if unmatched:
        raise ValueError(
            "scenario_amv_overrides did not resolve in "
            f"{_repo_relative(matrix_path)}: {', '.join(unmatched)}"
        )


def _apply_scenario_amv_overrides(
    scenarios: list[dict[str, Any]],
    *,
    overrides: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Apply slice-local AMV taxonomy overrides without mutating source scenarios.

    Returns:
        Scenario list with any configured AMV taxonomy overrides merged into
        both ``scenario["amv"]`` and ``scenario["metadata"]["amv"]``.
    """
    if not overrides:
        return scenarios

    patched_scenarios: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_name = _scenario_name_key(scenario)
        taxonomy_override = overrides.get(scenario_name)
        if not isinstance(taxonomy_override, Mapping) or not taxonomy_override:
            patched_scenarios.append(scenario)
            continue

        patched = dict(scenario)
        amv = dict(scenario.get("amv")) if isinstance(scenario.get("amv"), dict) else {}
        amv.update({key: str(value) for key, value in taxonomy_override.items()})
        patched["amv"] = amv

        metadata = (
            dict(scenario.get("metadata")) if isinstance(scenario.get("metadata"), dict) else {}
        )
        metadata_amv = dict(metadata.get("amv")) if isinstance(metadata.get("amv"), dict) else {}
        metadata_amv.update(amv)
        metadata["amv"] = metadata_amv
        patched["metadata"] = metadata
        patched_scenarios.append(patched)
    return patched_scenarios


def _load_campaign_scenarios(cfg: CampaignConfig) -> list[dict[str, Any]]:
    """Load campaign scenarios and apply optional seed override.

    Returns:
        Scenario list consumable by benchmark runners.
    """
    scenarios = load_scenarios(
        cfg.scenario_matrix_path,
        base_dir=cfg.scenario_matrix_path.parent,
    )
    matrix_root = cfg.scenario_matrix_path.parent
    normalized: list[dict[str, Any]] = []
    repo_root = get_repository_root().resolve()
    for scenario in scenarios:
        patched = dict(scenario)
        map_file = patched.get("map_file")
        if isinstance(map_file, str):
            map_path = Path(map_file)
            if map_path.is_absolute():
                try:
                    patched["map_file"] = map_path.resolve().relative_to(repo_root).as_posix()
                except ValueError:
                    patched["map_file"] = map_path.resolve().as_posix()
            else:
                candidate = (matrix_root / map_path).resolve()
                if candidate.exists():
                    try:
                        patched["map_file"] = candidate.relative_to(repo_root).as_posix()
                    except ValueError:
                        patched["map_file"] = candidate.as_posix()
        normalized.append(patched)

    scenario_dicts = _filter_scenario_candidates(
        normalized,
        names=cfg.scenario_candidates.names,
        matrix_path=cfg.scenario_matrix_path,
    )
    _validate_scenario_amv_override_keys(
        scenario_dicts,
        overrides=cfg.scenario_amv_overrides,
        matrix_path=cfg.scenario_matrix_path,
    )
    scenario_dicts = _apply_scenario_amv_overrides(
        scenario_dicts,
        overrides=cfg.scenario_amv_overrides,
    )
    scenario_dicts = _apply_scenario_horizon_schedule(
        scenario_dicts,
        schedule_path=cfg.scenario_horizons_path,
    )
    seeds_override = _resolve_seed_override(cfg.seed_policy)
    if seeds_override is None:
        return scenario_dicts

    seeded: list[dict[str, Any]] = []
    for scenario in scenario_dicts:
        patched = dict(scenario)
        patched["seeds"] = list(seeds_override)
        seeded.append(patched)
    return seeded


def _resolved_seed_inventory(scenarios: list[dict[str, Any]]) -> list[int]:
    """Return sorted unique seed values actually present in campaign scenarios.

    Returns:
        Sorted list of unique integer seeds.
    """
    seeds: set[int] = set()
    for scenario in scenarios:
        scenario_seeds = scenario.get("seeds")
        if not isinstance(scenario_seeds, list):
            continue
        for value in scenario_seeds:
            try:
                seeds.add(int(value))
            except (TypeError, ValueError):
                continue
    return sorted(seeds)


def _validate_campaign_config(cfg: CampaignConfig) -> None:  # noqa: C901, PLR0912, PLR0915
    """Validate campaign-level invariants after config parsing."""
    if cfg.scenario_horizons_path is not None and not cfg.scenario_horizons_path.is_file():
        raise FileNotFoundError(
            f"Scenario horizon schedule not found: {cfg.scenario_horizons_path}"
        )
    if (
        cfg.route_clearance_certifications_path is not None
        and not cfg.route_clearance_certifications_path.is_file()
    ):
        raise FileNotFoundError(
            "Route-clearance certification file not found: "
            f"{cfg.route_clearance_certifications_path}"
        )
    if cfg.scenario_horizons_path is not None:
        if cfg.horizon is not None:
            raise ValueError("scenario_horizons cannot be combined with fixed horizon")
        planners_with_horizon_override = [
            planner.key
            for planner in cfg.planners
            if planner.enabled and planner.horizon_override is not None
        ]
        if planners_with_horizon_override:
            names = ", ".join(sorted(planners_with_horizon_override))
            raise ValueError(
                f"scenario_horizons cannot be combined with per-planner horizon overrides: {names}"
            )
    enforcement = cfg.amv_profile.coverage_enforcement
    if enforcement not in _AMV_COVERAGE_ENFORCEMENT:
        known = ", ".join(sorted(_AMV_COVERAGE_ENFORCEMENT))
        raise ValueError(f"Unsupported amv_profile.coverage_enforcement '{enforcement}'. {known}")
    for key, values in cfg.amv_profile.required_dimensions.items():
        for value in values:
            if not str(value).strip():
                raise ValueError(f"AMV required dimension '{key}' contains an empty value")
    if cfg.scenario_candidates.names and any(
        not str(name).strip() for name in cfg.scenario_candidates.names
    ):
        raise ValueError("scenario_candidates must not contain empty names")
    for scenario_name, amv_override in cfg.scenario_amv_overrides.items():
        if not str(scenario_name).strip():
            raise ValueError("scenario_amv_overrides keys must be non-empty scenario names")
        if not amv_override:
            raise ValueError(
                "scenario_amv_overrides entries must include at least one AMV taxonomy dimension"
            )
    if cfg.synthetic_actuation_profile is not None:
        if cfg.synthetic_actuation_profile.claim_scope == SYNTHETIC_ACTUATION_CLAIM_SCOPE:
            validate_synthetic_actuation_profile(cfg.synthetic_actuation_profile)
        if cfg.paper_facing:
            raise ValueError("synthetic_actuation_profile requires paper_facing=false")
        normalized_kinematics = tuple(str(value).strip().lower() for value in cfg.kinematics_matrix)
        if normalized_kinematics != ("differential_drive",):
            raise ValueError(
                "synthetic_actuation_profile requires kinematics_matrix=['differential_drive']"
            )
    if cfg.latency_stress_profile is not None:
        validate_latency_stress_profile(cfg.latency_stress_profile)
        if cfg.paper_facing:
            raise ValueError("latency_stress_profile requires paper_facing=false")
        if cfg.latency_stress_profile.action_delay_steps > 0:
            normalized_kinematics = tuple(
                str(value).strip().lower() for value in cfg.kinematics_matrix
            )
            if normalized_kinematics != ("differential_drive",):
                raise ValueError(
                    "latency_stress_profile.action_delay_steps requires "
                    "kinematics_matrix=['differential_drive']"
                )
    if cfg.snqi_contract.enforcement not in _SNQI_CONTRACT_ENFORCEMENT:
        known = ", ".join(sorted(_SNQI_CONTRACT_ENFORCEMENT))
        raise ValueError(
            f"Unsupported snqi_contract.enforcement '{cfg.snqi_contract.enforcement}'. {known}"
        )
    threshold_values = {
        "rank_alignment_warn_threshold": cfg.snqi_contract.rank_alignment_warn_threshold,
        "rank_alignment_fail_threshold": cfg.snqi_contract.rank_alignment_fail_threshold,
        "outcome_separation_warn_threshold": cfg.snqi_contract.outcome_separation_warn_threshold,
        "outcome_separation_fail_threshold": cfg.snqi_contract.outcome_separation_fail_threshold,
        "max_component_dominance_warn_threshold": (
            cfg.snqi_contract.max_component_dominance_warn_threshold
        ),
        "max_component_dominance_fail_threshold": (
            cfg.snqi_contract.max_component_dominance_fail_threshold
        ),
    }
    for field_name, value in threshold_values.items():
        if not math.isfinite(value):
            raise ValueError(f"snqi_contract.{field_name} must be a finite float")
    if (
        cfg.snqi_contract.rank_alignment_fail_threshold
        > cfg.snqi_contract.rank_alignment_warn_threshold
    ):
        raise ValueError(
            "snqi_contract.rank_alignment_fail_threshold must be <= rank_alignment_warn_threshold"
        )
    if (
        cfg.snqi_contract.outcome_separation_fail_threshold
        > cfg.snqi_contract.outcome_separation_warn_threshold
    ):
        raise ValueError(
            "snqi_contract.outcome_separation_fail_threshold must be <= outcome_separation_warn_threshold"
        )
    if (
        cfg.snqi_contract.max_component_dominance_fail_threshold
        < cfg.snqi_contract.max_component_dominance_warn_threshold
    ):
        raise ValueError(
            "snqi_contract.max_component_dominance_fail_threshold must be >= "
            "max_component_dominance_warn_threshold"
        )

    if cfg.paper_facing:
        if not cfg.paper_profile_version or not str(cfg.paper_profile_version).strip():
            raise ValueError("paper_facing=true requires non-empty paper_profile_version")
        paper_profile = str(cfg.paper_profile_version).strip()
        expected_kinematics = _PAPER_KINEMATICS_BY_PROFILE.get(paper_profile)
        if expected_kinematics is None:
            known_profiles = ", ".join(sorted(_PAPER_KINEMATICS_BY_PROFILE))
            raise ValueError(
                f"Unsupported paper_profile_version '{paper_profile}'. Expected one of: "
                f"{known_profiles}"
            )
        normalized_kinematics = tuple(str(value).strip().lower() for value in cfg.kinematics_matrix)
        if normalized_kinematics != expected_kinematics:
            raise ValueError(
                "paper_facing=true requires kinematics_matrix="
                f"{list(expected_kinematics)!r} for paper_profile_version='{paper_profile}'",
            )
        for planner in cfg.planners:
            if not planner.enabled:
                continue
            if not planner.planner_group_explicit:
                raise ValueError(
                    "paper_facing=true requires explicit planner_group for each enabled planner",
                )
        if cfg.comparability_mapping_path is None:
            raise ValueError("paper_facing=true requires comparability_mapping path")


def load_campaign_config(path: Path) -> CampaignConfig:  # noqa: C901, PLR0912, PLR0915
    """Load and validate a camera-ready benchmark campaign YAML config.

    Returns:
        Parsed campaign configuration dataclass.
    """
    # Imported lazily to avoid an import cycle: ``_run_state`` imports ``_sanitize_name``
    # from this module at import time, so a top-level import here would be circular.
    from robot_sf.benchmark.camera_ready._run_state import (  # noqa: PLC0415 - cycle break
        _resolve_observation_noise,
        _resolve_path,
    )

    config_path = path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Campaign config must be a mapping: {config_path}")

    name = str(payload.get("name") or config_path.stem)
    matrix_raw = payload.get("scenario_matrix")
    if not isinstance(matrix_raw, str) or not matrix_raw.strip():
        raise ValueError("Campaign config requires a non-empty 'scenario_matrix' string")
    scenario_matrix_path = _resolve_path(matrix_raw, base_dir=config_path.parent)
    if scenario_matrix_path is None:
        raise FileNotFoundError(
            f"Could not resolve scenario_matrix '{matrix_raw}' from config '{config_path}'.",
        )

    planners_raw = payload.get("planners")
    if not isinstance(planners_raw, list) or not planners_raw:
        raise ValueError("Campaign config requires a non-empty 'planners' list")

    planner_specs: list[PlannerSpec] = []
    for entry in planners_raw:
        if not isinstance(entry, dict):
            raise ValueError("Each planners entry must be a mapping")
        key = str(entry.get("key") or entry.get("algo") or "").strip()
        algo = str(entry.get("algo") or "").strip()
        if not key or not algo:
            raise ValueError("Planner entry requires non-empty key and algo")
        planner_group_explicit = "planner_group" in entry
        planner_group = str(entry.get("planner_group", "experimental")).strip().lower()
        if planner_group not in _PLANNER_GROUPS:
            raise ValueError(
                f"Unsupported planner_group '{planner_group}'. Expected one of: core|experimental."
            )
        planner_specs.append(
            PlannerSpec(
                key=key,
                algo=algo,
                human_model_variant=_normalize_observation_mode(
                    entry.get("human_model_variant"),
                    label="Planner entry 'human_model_variant'",
                ),
                human_model_source=_normalize_observation_mode(
                    entry.get("human_model_source"),
                    label="Planner entry 'human_model_source'",
                ),
                benchmark_profile=str(entry.get("benchmark_profile", "baseline-safe")),
                algo_config_path=_resolve_path(
                    entry.get("algo_config"), base_dir=config_path.parent
                ),
                socnav_missing_prereq_policy=str(
                    entry.get("socnav_missing_prereq_policy", "fail-fast"),
                ),
                adapter_impact_eval=bool(entry.get("adapter_impact_eval", False)),
                observation_mode=_normalize_observation_mode(
                    entry.get("observation_mode"),
                    label="Planner entry 'observation_mode'",
                ),
                workers_override=(
                    int(entry["workers"]) if entry.get("workers") is not None else None
                ),
                horizon_override=(
                    int(entry["horizon"]) if entry.get("horizon") is not None else None
                ),
                dt_override=(float(entry["dt"]) if entry.get("dt") is not None else None),
                enabled=bool(entry.get("enabled", True)),
                planner_group=planner_group,
                planner_group_explicit=planner_group_explicit,
            ),
        )

    seed_policy_raw = (
        payload.get("seed_policy") if isinstance(payload.get("seed_policy"), dict) else {}
    )
    mode = str(seed_policy_raw.get("mode", "scenario-default"))
    seed_set = seed_policy_raw.get("seed_set")
    seeds = seed_policy_raw.get("seeds") if isinstance(seed_policy_raw.get("seeds"), list) else []
    seed_sets_path_raw = seed_policy_raw.get("seed_sets_path")
    seed_sets_path = (
        _resolve_path(str(seed_sets_path_raw), base_dir=config_path.parent)
        if isinstance(seed_sets_path_raw, str) and seed_sets_path_raw.strip()
        else None
    )
    if seed_sets_path is None:
        seed_sets_path = (get_repository_root() / DEFAULT_SEED_SETS_PATH).resolve()

    snqi_weights = _resolve_path(payload.get("snqi_weights"), base_dir=config_path.parent)
    snqi_baseline = _resolve_path(payload.get("snqi_baseline"), base_dir=config_path.parent)
    scenario_horizons = _resolve_path(
        payload.get("scenario_horizons"),
        base_dir=config_path.parent,
    )
    route_clearance_certifications_path = _resolve_path(
        payload.get("route_clearance_certifications"),
        base_dir=config_path.parent,
    )
    comparability_mapping_path = _resolve_path(
        payload.get("comparability_mapping"),
        base_dir=config_path.parent,
    )
    if comparability_mapping_path is None:
        default_mapping_path = (
            get_repository_root() / "configs/benchmarks/alyassi_comparability_map_v1.yaml"
        ).resolve()
        if default_mapping_path.exists():
            comparability_mapping_path = default_mapping_path

    amv_raw = payload.get("amv_profile") if isinstance(payload.get("amv_profile"), dict) else {}
    snqi_contract_raw = (
        payload.get("snqi_contract") if isinstance(payload.get("snqi_contract"), dict) else {}
    )
    required_raw = (
        amv_raw.get("required_dimensions")
        if isinstance(amv_raw.get("required_dimensions"), dict)
        else {}
    )
    for key in required_raw:
        if key not in _AMV_DIMENSIONS:
            known = ", ".join(_AMV_DIMENSIONS)
            raise ValueError(
                f"Unsupported amv_profile.required_dimensions key '{key}'. Expected: {known}"
            )
    required_dimensions: dict[str, tuple[str, ...]] = {}
    for dimension in _AMV_DIMENSIONS:
        values = required_raw.get(dimension, [])
        if isinstance(values, (str, int, float)):
            normalized = (str(values).strip(),) if str(values).strip() else ()
        elif isinstance(values, list):
            normalized = tuple(str(value).strip() for value in values if str(value).strip())
        else:
            normalized = ()
        required_dimensions[dimension] = normalized
    scenario_candidates_raw = payload.get("scenario_candidates", [])
    if isinstance(scenario_candidates_raw, (str, int, float)):
        scenario_candidates = (str(scenario_candidates_raw).strip(),)
    elif isinstance(scenario_candidates_raw, list):
        if any(not isinstance(value, (str, int, float)) for value in scenario_candidates_raw):
            raise TypeError("scenario_candidates entries must be scalar names")
        scenario_candidates = tuple(
            str(value).strip() for value in scenario_candidates_raw if str(value).strip()
        )
    elif "scenario_candidates" in payload:
        raise TypeError("scenario_candidates must be a scalar name or list of scalar names")
    else:
        scenario_candidates = ()
    scenario_amv_overrides_raw = payload.get("scenario_amv_overrides")
    if scenario_amv_overrides_raw is None:
        scenario_amv_overrides: dict[str, dict[str, str]] = {}
    elif not isinstance(scenario_amv_overrides_raw, dict):
        raise TypeError(
            "scenario_amv_overrides must be a mapping of scenario names to AMV mappings"
        )
    else:
        scenario_amv_overrides = {}
        for raw_scenario_name, raw_taxonomy in scenario_amv_overrides_raw.items():
            scenario_name = str(raw_scenario_name).strip()
            if not scenario_name:
                raise ValueError("scenario_amv_overrides keys must be non-empty scenario names")
            if not isinstance(raw_taxonomy, dict):
                raise TypeError(
                    "scenario_amv_overrides entries must be mappings keyed by AMV dimension"
                )
            taxonomy: dict[str, str] = {}
            for raw_dimension, raw_value in raw_taxonomy.items():
                dimension = str(raw_dimension).strip()
                if dimension not in _AMV_DIMENSIONS:
                    known = ", ".join(_AMV_DIMENSIONS)
                    raise ValueError(
                        f"Unsupported scenario_amv_overrides dimension '{dimension}'. "
                        f"Expected: {known}"
                    )
                if raw_value is None:
                    raise ValueError(
                        "scenario_amv_overrides values must be non-empty strings when provided"
                    )
                value = str(raw_value).strip()
                if not value:
                    raise ValueError(
                        "scenario_amv_overrides values must be non-empty strings when provided"
                    )
                taxonomy[dimension] = value
            if not taxonomy:
                raise ValueError(
                    "scenario_amv_overrides entries must include at least one AMV taxonomy dimension"
                )
            scenario_amv_overrides[scenario_name] = taxonomy
    synthetic_actuation_raw = payload.get("synthetic_actuation_profile")
    if synthetic_actuation_raw is not None and not isinstance(synthetic_actuation_raw, dict):
        raise TypeError("synthetic_actuation_profile must be a mapping when provided")
    if synthetic_actuation_raw is not None:
        validate_actuation_profile_claim_boundary(synthetic_actuation_raw)
    latency_stress_raw = payload.get("latency_stress_profile")
    kinematics_matrix = _normalize_kinematics_matrix(
        payload.get("kinematics_matrix", ["differential_drive"])
    )

    cfg = CampaignConfig(
        name=name,
        scenario_matrix_path=scenario_matrix_path,
        planners=tuple(planner_specs),
        scenario_candidates=ScenarioCandidateSelection(names=scenario_candidates),
        scenario_amv_overrides=scenario_amv_overrides,
        scenario_horizons_path=scenario_horizons,
        seed_policy=SeedPolicy(
            mode=mode,
            seed_set=str(seed_set) if seed_set is not None else None,
            seeds=tuple(int(seed) for seed in seeds),
            seed_sets_path=seed_sets_path,
        ),
        workers=int(payload.get("workers", 1)),
        horizon=(int(payload["horizon"]) if payload.get("horizon") is not None else None),
        dt=(float(payload["dt"]) if payload.get("dt") is not None else None),
        record_forces=bool(payload.get("record_forces", True)),
        resume=bool(payload.get("resume", True)),
        bootstrap_samples=int(payload.get("bootstrap_samples", 400)),
        bootstrap_confidence=float(payload.get("bootstrap_confidence", 0.95)),
        bootstrap_seed=int(payload.get("bootstrap_seed", 123)),
        snqi_weights_path=snqi_weights,
        snqi_baseline_path=snqi_baseline,
        stop_on_failure=bool(payload.get("stop_on_failure", False)),
        export_publication_bundle=bool(payload.get("export_publication_bundle", True)),
        include_videos_in_publication=bool(payload.get("include_videos_in_publication", False)),
        overwrite_publication_bundle=bool(payload.get("overwrite_publication_bundle", True)),
        repository_url=str(payload.get("repository_url", "https://github.com/ll7/robot_sf_ll7")),
        release_tag=str(payload.get("release_tag", "{release_tag}")),
        doi=str(payload.get("doi", "10.5281/zenodo.<record-id>")),
        paper_interpretation_profile=str(
            payload.get("paper_interpretation_profile", "baseline-ready-core")
        ),
        preview_scenario_limit=int(payload.get("preview_scenario_limit", 100)),
        kinematics_matrix=kinematics_matrix,
        holonomic_command_mode=str(payload.get("holonomic_command_mode", "vx_vy")).strip(),
        observation_mode=_normalize_observation_mode(
            payload.get("observation_mode"),
            label="Campaign 'observation_mode'",
        ),
        paper_facing=bool(payload.get("paper_facing", False)),
        paper_profile_version=(
            str(payload.get("paper_profile_version")).strip()
            if payload.get("paper_profile_version") is not None
            else None
        ),
        amv_profile=AmvProfileConfig(
            name=str(amv_raw.get("name", "amv-paper-v1")).strip() or "amv-paper-v1",
            contract_version=str(amv_raw.get("contract_version", "1")).strip() or "1",
            coverage_enforcement=(
                str(amv_raw.get("coverage_enforcement", "warn")).strip().lower() or "warn"
            ),
            required_dimensions=required_dimensions,
        ),
        synthetic_actuation_profile=(
            SyntheticActuationProfile(
                name=str(synthetic_actuation_raw.get("name", "")).strip(),
                profile_version=(
                    str(synthetic_actuation_raw.get("profile_version", "v0")).strip() or "v0"
                ),
                claim_scope=(
                    str(synthetic_actuation_raw.get("claim_scope", "synthetic-only")).strip()
                    or "synthetic-only"
                ),
                claim_boundary=str(synthetic_actuation_raw.get("claim_boundary", "")).strip(),
                max_linear_accel_m_s2=float(
                    synthetic_actuation_raw.get("max_linear_accel_m_s2", 0.0)
                ),
                max_linear_decel_m_s2=float(
                    synthetic_actuation_raw.get("max_linear_decel_m_s2", 0.0)
                ),
                max_yaw_rate_rad_s=float(synthetic_actuation_raw.get("max_yaw_rate_rad_s", 0.0)),
                max_angular_accel_rad_s2=float(
                    synthetic_actuation_raw.get("max_angular_accel_rad_s2", 0.0)
                ),
                latency_mode=(
                    str(synthetic_actuation_raw.get("latency_mode", "zero-step-delay"))
                    .strip()
                    .lower()
                ),
                update_mode=(
                    str(synthetic_actuation_raw.get("update_mode", "10hz-matched")).strip().lower()
                ),
                variability_distribution=_optional_synthetic_actuation_profile_mapping(
                    synthetic_actuation_raw,
                    "variability_distribution",
                ),
                variability_sample=_optional_synthetic_actuation_profile_mapping(
                    synthetic_actuation_raw,
                    "variability_sample",
                ),
                provenance=_optional_synthetic_actuation_profile_mapping(
                    synthetic_actuation_raw,
                    "provenance",
                ),
            )
            if synthetic_actuation_raw is not None
            else None
        ),
        latency_stress_profile=load_latency_stress_profile(latency_stress_raw),
        comparability_mapping_path=comparability_mapping_path,
        route_clearance_certifications_path=route_clearance_certifications_path,
        snqi_contract=SnqiContractConfig(
            enabled=bool(snqi_contract_raw.get("enabled", True)),
            enforcement=(
                str(snqi_contract_raw.get("enforcement", "warn")).strip().lower() or "warn"
            ),
            rank_alignment_warn_threshold=float(
                snqi_contract_raw.get("rank_alignment_warn_threshold", 0.5)
            ),
            rank_alignment_fail_threshold=float(
                snqi_contract_raw.get("rank_alignment_fail_threshold", 0.3)
            ),
            outcome_separation_warn_threshold=float(
                snqi_contract_raw.get("outcome_separation_warn_threshold", 0.05)
            ),
            outcome_separation_fail_threshold=float(
                snqi_contract_raw.get("outcome_separation_fail_threshold", 0.0)
            ),
            max_component_dominance_warn_threshold=float(
                snqi_contract_raw.get("max_component_dominance_warn_threshold", 0.24)
            ),
            max_component_dominance_fail_threshold=float(
                snqi_contract_raw.get("max_component_dominance_fail_threshold", 0.27)
            ),
            calibration_seed=int(snqi_contract_raw.get("calibration_seed", 123)),
            calibration_trials=int(snqi_contract_raw.get("calibration_trials", 3000)),
        ),
        observation_noise=_resolve_observation_noise(
            payload.get("observation_noise"),
            base_dir=config_path.parent,
        ),
    )
    _validate_campaign_config(cfg)
    return cfg
