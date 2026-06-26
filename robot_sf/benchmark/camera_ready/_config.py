"""Config and scenario-loading helpers for camera-ready campaigns.

Extracted from ``robot_sf.benchmark.camera_ready_campaign`` for #3385 slice 7.
The public behavior stays anchored in ``camera_ready_campaign``, which
re-exports these private helpers for compatibility.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.benchmark.camera_ready._util import _repo_relative
from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.training.scenario_loader import load_scenarios

if TYPE_CHECKING:
    from robot_sf.benchmark.camera_ready_campaign_config import CampaignConfig, SeedPolicy

_PLANNER_GROUPS = {"core", "experimental"}
_PAPER_KINEMATICS_BY_PROFILE = {
    "paper-seed-variability-v1": ("differential_drive",),
    "paper-matrix-v1": ("differential_drive",),
    "paper-cross-kinematics-v1": ("differential_drive", "bicycle_drive", "holonomic"),
}
_AMV_COVERAGE_ENFORCEMENT = {"warn", "error"}
_SNQI_CONTRACT_ENFORCEMENT = {"warn", "error"}


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
