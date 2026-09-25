"""Capture exact scenario inputs used by one benchmark episode."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robot_sf.training.scenario_loader import resolve_map_id

EPISODE_INPUT_IDENTITY_SCHEMA = "benchmark-episode-case-input-identity.v1"
_PROVENANCE_ONLY_SCENARIO_METADATA = "adversarial_candidate"


def scenario_semantic_sha256(scenario: Mapping[str, Any], *, seed: int) -> str:
    """Hash behavior-affecting scenario fields while excluding relocatable paths.

    Returns:
        The SHA-256 digest of the canonicalized scenario identity.
    """
    identity = {key: value for key, value in scenario.items() if key not in {"seed", "seeds"}}
    identity.pop("map_file", None)
    identity.pop("route_overrides_file", None)
    metadata = identity.get("metadata")
    if isinstance(metadata, Mapping):
        metadata = dict(metadata)
        metadata.pop(_PROVENANCE_ONLY_SCENARIO_METADATA, None)
        identity["metadata"] = metadata
    simulation_config = identity.get("simulation_config")
    if isinstance(simulation_config, Mapping):
        simulation_config = dict(simulation_config)
        if simulation_config.get("route_spawn_seed") is None:
            simulation_config["route_spawn_seed"] = int(seed)
        identity["simulation_config"] = simulation_config
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def capture_episode_input_identity(
    scenario: Mapping[str, Any],
    *,
    scenario_path: str | Path,
    seed: int,
    run_id: str,
) -> dict[str, Any]:
    """Capture scenario, route, and resolved map digests before simulation.

    Returns:
        A versioned identity with ``status=unavailable`` and reason codes if any
        required file cannot be resolved or read.
    """
    source = Path(scenario_path).expanduser().resolve()
    route_digest, route_reasons = _capture_route_digest(scenario, source)
    map_assets, map_reasons = _capture_map_assets(scenario, source)
    reasons = list(route_reasons) + list(map_reasons)
    try:
        semantic_digest = scenario_semantic_sha256(scenario, seed=seed)
    except (TypeError, ValueError):
        semantic_digest = None
        reasons.append("scenario_semantic_identity_unavailable")
    identity: dict[str, Any] = {
        "schema_version": EPISODE_INPUT_IDENTITY_SCHEMA,
        "status": "unavailable",
        "run_id": run_id,
        "scenario_semantic_sha256": semantic_digest,
        "route_overrides_sha256": route_digest,
        "map_assets": map_assets,
        "reason_codes": sorted(set(reasons)),
    }
    if not reasons and semantic_digest is not None and route_digest is not None and map_assets:
        identity["status"] = "bound"
    return identity


def reconcile_consumed_map_identity(
    identity: Mapping[str, Any], *, consumed_map_sha256: str | None
) -> dict[str, Any]:
    """Reconcile the map bytes parsed for an episode with its captured identity.

    Returns:
        A copied identity marked unavailable when the parsed map digest is missing
        or differs from the captured map asset digest.
    """
    reconciled = dict(identity)
    if reconciled.get("status") != "bound":
        return reconciled
    captured_map_digests = [
        asset.get("sha256")
        for asset in reconciled.get("map_assets", [])
        if isinstance(asset, Mapping) and asset.get("role") == "map"
    ]
    if (
        len(captured_map_digests) != 1
        or not isinstance(consumed_map_sha256, str)
        or captured_map_digests[0] != consumed_map_sha256
    ):
        reconciled["status"] = "unavailable"
        reconciled["reason_codes"] = sorted(
            set(reconciled.get("reason_codes", []))
            | {"parsed_map_bytes_differ_from_captured_map_asset"}
        )
    return reconciled


def _capture_route_digest(
    scenario: Mapping[str, Any], source: Path
) -> tuple[str | None, list[str]]:
    reference = scenario.get("route_overrides_file")
    if not isinstance(reference, str) or not reference.strip():
        return None, ["route_overrides_not_declared"]
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    try:
        return _sha256_file(path.resolve(strict=True)), []
    except (OSError, RuntimeError, ValueError):
        return None, ["route_overrides_unavailable"]


def _capture_map_assets(
    scenario: Mapping[str, Any], source: Path
) -> tuple[list[dict[str, str]], list[str]]:
    map_id = scenario.get("map_id")
    map_reference = scenario.get("map_file")
    reasons: list[str] = []
    map_path: Path | None = None
    registry_path: Path | None = None
    if isinstance(map_id, str) and map_id.strip():
        try:
            map_path = resolve_map_id(map_id, source=source)
            registry = _map_registry_path()
            if registry is None:
                reasons.append("map_registry_unavailable")
            else:
                registry_path = registry.resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            reasons.append("registered_map_unavailable")
    elif isinstance(map_reference, str) and map_reference.strip():
        map_path = Path(map_reference).expanduser()
        if not map_path.is_absolute():
            map_path = source.parent / map_path
    else:
        reasons.append("map_not_declared")

    assets: list[dict[str, str]] = []
    if map_path is not None:
        _append_asset_digest(assets, reasons, role="map", path=map_path)
    if registry_path is not None:
        _append_asset_digest(assets, reasons, role="map_registry", path=registry_path)
    return sorted(assets, key=lambda item: item["role"]), reasons


def _append_asset_digest(
    assets: list[dict[str, str]], reasons: list[str], *, role: str, path: Path
) -> None:
    try:
        assets.append({"role": role, "sha256": _sha256_file(path.resolve(strict=True))})
    except (OSError, RuntimeError, ValueError):
        reasons.append(f"{role}_asset_unavailable")


def _map_registry_path() -> Path | None:
    """Return the configured registry path used by the canonical scenario loader."""
    override = os.getenv("ROBOT_SF_MAP_REGISTRY")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[2] / "maps/registry.yaml"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "EPISODE_INPUT_IDENTITY_SCHEMA",
    "capture_episode_input_identity",
    "reconcile_consumed_map_identity",
    "scenario_semantic_sha256",
]
