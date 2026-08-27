"""Fail-closed acceptance checks for a full benchmark-data release.

The camera-ready campaign runner intentionally supports partial and
``core``-only success semantics for exploratory work.  A publication release
has a stricter contract: every declared arm must complete every declared
scenario/seed cell, and no fallback or degraded row may be promoted.  This
module keeps that publication gate separate from the bounded runtime smoke.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.analysis_trace import normalize_telemetry_profile
from robot_sf.benchmark.camera_ready._config import (
    _load_campaign_scenarios,
    _scenario_with_kinematics,
)
from robot_sf.benchmark.camera_ready._preflight import (
    _config_hash_payload,
    _resolved_seed_inventory,
    _scenario_matrix_hash,
)
from robot_sf.benchmark.camera_ready._run_state import (
    _resolve_integrity_artifact_path,
    validate_campaign_integrity,
)
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.effective_algorithm_branches import (
    check_witness_coverage,
    enumerate_effective_branches,
)
from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.map_runner.map_runner_trace import _scenario_id as _producer_scenario_id
from robot_sf.benchmark.map_runner_identity import suite_key as _producer_suite_key
from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import (
    _is_policy_search_candidate_manifest,
    _parse_algo_config,
    _resolve_policy_search_candidate_runtime,
)
from robot_sf.benchmark.release_protocol import (
    STRESS_SMOKE_EXPECTED_DT,
    STRESS_SMOKE_EXPECTED_EPISODE_CELLS,
    STRESS_SMOKE_EXPECTED_HORIZON_STEPS,
    STRESS_SMOKE_EXPECTED_KINEMATICS,
    STRESS_SMOKE_EXPECTED_PLANNER_ARMS,
    STRESS_SMOKE_EXPECTED_SCENARIO_IDS,
    STRESS_SMOKE_EXPECTED_SEED,
    StressSmokeBranchWitness,
    resolve_campaign_artifact_path,
)
from robot_sf.benchmark.result_provenance import validate_result_provenance_manifest
from robot_sf.benchmark.utils import _config_hash
from robot_sf.common.artifact_paths import get_repository_root

FULL_RELEASE_SCHEMA_VERSION = "benchmark-release-manifest.v0.2"
FULL_RELEASE_EXPECTED_PLANNER_ARMS = 14
FULL_RELEASE_EXPECTED_EPISODE_CELLS = 20_160
FULL_RELEASE_EXPECTED_HORIZON_STEPS = 600
FULL_RELEASE_KINEMATICS = "differential_drive"
FULL_RELEASE_ACCEPTANCE_SCHEMA_VERSION = "benchmark-full-release-acceptance.v1"

_FORBIDDEN_STATUSES = frozenset(
    {
        "degraded",
        "error",
        "excluded",
        "failed",
        "fallback",
        "not-available",
        "not_available",
        "partial-failure",
        "partial_failure",
        "placeholder",
        "unavailable",
    }
)
_FORBIDDEN_STATUS_PREFIXES = ("predictive_foresight_model_fallback",)
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_LEGACY_EMERGENCY_MODES = frozenset({"emergency_stop", "reorient"})
_LEGACY_EMERGENCY_SOURCES = frozenset({"all_candidates_rejected", "static_reorient"})
_STRESS_RUN_SUCCESS_STATUSES = frozenset({"ok"})
_STRESS_ROW_TERMINAL_STATUSES = frozenset({"success", "collision", "failure"})
_STRESS_VALID_BENCHMARK_SUCCESS = frozenset({"true"})
_MISSING = object()
_RUNTIME_METADATA_CONTAINERS = frozenset(
    {
        "algorithm_metadata",
        "algorithm_metadata_contract",
        "checkpoint_provenance",
        "execution",
        "foresight_prediction",
        "last_decision",
        "learned_policy_contract",
        "planner_kinematics",
        "planner_runtime",
        "preflight",
        "runtime",
        "runtime_metadata",
    }
)
_DECLARATIVE_ALGORITHM_METADATA_CONTAINERS = frozenset(
    {"config", "planner_contract", "safety_shield_contract"}
)


def _normalized_status(value: Any) -> str:
    """Normalize a status marker without treating absent fields as failures.

    Returns:
        Lowercase status token, or an empty string for an absent value.
    """
    return str(value).strip().lower().replace(" ", "_") if value is not None else ""


def _strict_int(value: Any) -> int | None:
    """Parse only integral JSON values or their decimal-string representation.

    Returns:
        Parsed integer, or ``None`` for non-integral values.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def _append_blocker(blockers: list[str], message: str) -> None:
    """Add a deterministic blocker while bounding pathological row-level output."""
    if message not in blockers and len(blockers) < 100:
        blockers.append(message)


def _emergency_stop_marker(payload: Any) -> tuple[str, str] | None:  # noqa: C901
    """Reject legacy emergency-stop paths without changing fallback counters.

    Any positive ``emergency_stop_count`` is forbidden for this release.  The
    generic fallback marker remains responsible for ``fallback_count`` so its
    semantics do not change as this stricter release gate evolves.

    Returns:
        Path/value pair for the first legacy or insufficient emergency marker.
    """

    def _normalized(value: Any) -> str:
        return str(value).strip().lower().replace("-", "_")

    def _walk(  # noqa: C901, PLR0912
        value: Any, path: str, *, active: bool = False
    ) -> tuple[str, str] | None:
        if isinstance(value, Mapping):
            root_mapping = not path
            inspect_context = active or root_mapping
            for key, nested in value.items():
                key_text = str(key)
                nested_path = f"{path}.{key_text}" if path else key_text
                normalized_key = key_text.strip().lower()
                inspect_marker = inspect_context or normalized_key in _RUNTIME_METADATA_CONTAINERS
                if inspect_marker and normalized_key in {
                    "mode",
                    "planner_mode",
                    "selected_source",
                }:
                    normalized_value = _normalized(nested)
                    if (
                        normalized_key == "selected_source"
                        and normalized_value in _LEGACY_EMERGENCY_SOURCES
                    ):
                        return nested_path, normalized_value
                    if (
                        normalized_key in {"mode", "planner_mode"}
                        and normalized_value in _LEGACY_EMERGENCY_MODES
                    ):
                        return nested_path, normalized_value
                elif inspect_marker and normalized_key == "selected_source_counts":
                    if not isinstance(nested, Mapping):
                        return nested_path, "invalid"
                    for raw_source, count in nested.items():
                        source = _normalized(raw_source)
                        if source not in _LEGACY_EMERGENCY_SOURCES:
                            continue
                        count_path = f"{nested_path}.{raw_source}"
                        if str(raw_source) != source:
                            return count_path, "invalid"
                        if not isinstance(count, (int, float)) or isinstance(count, bool):
                            return count_path, "invalid"
                        try:
                            finite = math.isfinite(float(count))
                        except (OverflowError, ValueError):
                            finite = False
                        if not finite or count < 0:
                            return count_path, "invalid"
                        if count > 0:
                            return count_path, str(count)
                elif inspect_marker and normalized_key == "emergency_stop_count":
                    parsed_counter = (
                        nested if isinstance(nested, int) and not isinstance(nested, bool) else None
                    )
                    if parsed_counter is None or parsed_counter < 0:
                        return nested_path, "invalid"
                    if parsed_counter > 0:
                        return nested_path, str(nested)
                elif inspect_marker and normalized_key == "emergency_stop":
                    if not isinstance(nested, bool):
                        return nested_path, "invalid"
                    if nested is True:
                        return nested_path, "true"
                found = _walk(
                    nested,
                    nested_path,
                    active=active or normalized_key in _RUNTIME_METADATA_CONTAINERS,
                )
                if found is not None:
                    return found
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                found = _walk(nested, f"{path}[{index}]", active=active)
                if found is not None:
                    return found
        return None

    return _walk(payload, "")


def _algorithm_metadata_runtime_marker(
    metadata: Mapping[str, Any], *, expected_algorithm: str | None = None
) -> tuple[str, str] | None:
    """Scan runtime-bearing algorithm metadata without treating config as execution evidence.

    Guarded PPO's safe Risk-DWA shield command is a declared component of that composite
    planner.  Its exact ``fallback_safe`` counters are therefore native intervention telemetry;
    best-effort and uncertainty fallbacks remain forbidden.

    Returns:
        The first forbidden runtime marker, if present.
    """

    def _is_valid_native_counter(value: Any) -> bool:
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        return value >= 0

    runtime_view: dict[str, Any] = {
        str(key): value
        for key, value in metadata.items()
        if str(key) not in _DECLARATIVE_ALGORITHM_METADATA_CONTAINERS
    }
    planner_contract = metadata.get("planner_contract")
    planner_id = (
        str(planner_contract.get("planner_id", "")).strip().lower()
        if isinstance(planner_contract, Mapping)
        else ""
    )
    guarded_ppo_identity = (
        str(expected_algorithm or "").strip().lower() == "guarded_ppo"
        and str(metadata.get("canonical_algorithm", "")).strip().lower() == "guarded_ppo"
        and str(metadata.get("algorithm", "")).strip().lower() == "ppo"
        and planner_id == "guarded_ppo"
    )
    if guarded_ppo_identity:
        guard_stats = metadata.get("guard_stats")
        if isinstance(guard_stats, Mapping):
            if "fallback_safe" in guard_stats and not _is_valid_native_counter(
                guard_stats["fallback_safe"]
            ):
                return "guard_stats.fallback_safe", "invalid"
            runtime_view["guard_stats"] = {
                str(key): value
                for key, value in guard_stats.items()
                if key != "fallback_safe" or not _is_valid_native_counter(value)
            }
        shield_stats = metadata.get("shield_stats")
        if isinstance(shield_stats, Mapping):
            shield_view = dict(shield_stats)
            decision_counts = shield_stats.get("decision_counts")
            if isinstance(decision_counts, Mapping):
                if "fallback_safe" in decision_counts and not _is_valid_native_counter(
                    decision_counts["fallback_safe"]
                ):
                    return "shield_stats.decision_counts.fallback_safe", "invalid"
                shield_view["decision_counts"] = {
                    str(key): value
                    for key, value in decision_counts.items()
                    if key != "fallback_safe" or not _is_valid_native_counter(value)
                }
            last_decision = shield_stats.get("last_decision")
            if isinstance(last_decision, Mapping):
                shield_view["last_decision"] = {
                    str(key): value
                    for key, value in last_decision.items()
                    if key != "fallback_controller_state" or not isinstance(value, Mapping)
                }
            runtime_view["shield_stats"] = shield_view
    return runtime_fallback_or_degraded_marker(runtime_view)


def _status_markers(  # noqa: C901, PLR0912, PLR0915
    payload: Mapping[str, Any], prefix: str, *, expected_algorithm: str | None = None
) -> list[tuple[str, str]]:
    """Extract only execution/evidence status markers from one structured row.

    Returns:
        Path/value pairs for forbidden execution or evidence markers.
    """
    markers: list[tuple[str, str]] = []

    def _add(path: str, value: Any) -> None:
        status = _normalized_status(value)
        if status in _FORBIDDEN_STATUSES or any(
            status.startswith(prefix) for prefix in _FORBIDDEN_STATUS_PREFIXES
        ):
            markers.append((f"{prefix}.{path}", status))

    for field in ("status", "row_status", "readiness_status", "availability_status"):
        if field in payload:
            _add(field, payload[field])
    for field in ("evidence_status", "execution_status"):
        if field in payload:
            _add(field, payload[field])
    benchmark_success = payload.get("benchmark_success")
    if benchmark_success is False or (
        isinstance(benchmark_success, str)
        and benchmark_success.strip().lower() in {"false", "0", "no"}
    ):
        markers.append((f"{prefix}.benchmark_success", "false"))

    for field in (
        "fallback",
        "fallback_triggered",
        "degraded",
        "fallback_or_degraded",
        "fallback_used",
    ):
        if field not in payload:
            continue
        value = payload[field]
        if not isinstance(value, bool):
            markers.append((f"{prefix}.{field}", "invalid"))
        elif value is True:
            markers.append((f"{prefix}.{field}", "true"))
    emergency_marker = _emergency_stop_marker(payload)
    if emergency_marker is not None:
        marker_path, marker_value = emergency_marker
        markers.append((f"{prefix}.{marker_path}", marker_value))
    planner_runtime = payload.get("planner_runtime")
    if isinstance(planner_runtime, Mapping):
        runtime_marker = runtime_fallback_or_degraded_marker(planner_runtime)
        if runtime_marker is not None:
            marker_path, marker_value = runtime_marker
            markers.append((f"{prefix}.planner_runtime.{marker_path}", marker_value))
    for field in ("algorithm_metadata", "algorithm_metadata_contract"):
        metadata = payload.get(field)
        if not isinstance(metadata, Mapping):
            continue
        metadata_marker = _algorithm_metadata_runtime_marker(
            metadata, expected_algorithm=expected_algorithm
        )
        if metadata_marker is not None:
            marker_path, marker_value = metadata_marker
            markers.append((f"{prefix}.{field}.{marker_path}", marker_value))
        _add(f"{field}.status", metadata.get("status"))
        for marker_field in (
            "fallback",
            "fallback_triggered",
            "degraded",
            "fallback_or_degraded",
            "fallback_used",
        ):
            if marker_field not in metadata:
                continue
            marker_value = metadata[marker_field]
            if not isinstance(marker_value, bool):
                markers.append((f"{prefix}.{field}.{marker_field}", "invalid"))
            elif marker_value is True:
                markers.append((f"{prefix}.{field}.{marker_field}", "true"))
        planner_kinematics = metadata.get("planner_kinematics")
        if isinstance(planner_kinematics, Mapping):
            _add(
                f"{field}.planner_kinematics.execution_mode",
                planner_kinematics.get("execution_mode"),
            )
        adapter_impact = metadata.get("adapter_impact")
        if isinstance(adapter_impact, Mapping):
            _add(f"{field}.adapter_impact.execution_mode", adapter_impact.get("execution_mode"))
        runtime_marker = runtime_fallback_or_degraded_marker(metadata.get("planner_runtime"))
        if runtime_marker is not None:
            marker_path, marker_value = runtime_marker
            markers.append((f"{prefix}.{field}.planner_runtime.{marker_path}", marker_value))
        emergency_marker = _emergency_stop_marker(metadata.get("planner_runtime"))
        if emergency_marker is not None:
            marker_path, marker_value = emergency_marker
            markers.append((f"{prefix}.{field}.planner_runtime.{marker_path}", marker_value))
        foresight_marker = runtime_fallback_or_degraded_marker(metadata.get("foresight_prediction"))
        if foresight_marker is not None:
            marker_path, marker_value = foresight_marker
            markers.append((f"{prefix}.{field}.foresight_prediction.{marker_path}", marker_value))
    availability = payload.get("benchmark_availability")
    if isinstance(availability, Mapping):
        for field in ("status", "readiness_status", "availability_status", "execution_mode"):
            if field in availability:
                _add(f"benchmark_availability.{field}", availability[field])
    # The broad runtime metadata scan and the explicit compatibility fields above can
    # intentionally reach the same leaf (for example
    # ``planner_kinematics.execution_mode``).  Count each concrete marker once so the
    # acceptance receipt reports artifact facts rather than traversal-path multiplicity.
    return list(dict.fromkeys(markers))


def _read_campaign_summary(campaign_root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load the authoritative campaign summary, returning a shaped error.

    Returns:
        The parsed summary and an optional human-readable read error.
    """
    try:
        path = resolve_campaign_artifact_path(campaign_root, "reports/campaign_summary.json")
    except (OSError, ValueError) as exc:
        category = (
            "symlink is unsafe" if "symlink" in str(exc).lower() else "path is missing or unsafe"
        )
        return None, f"campaign summary cannot be read: campaign_summary.json {category}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return None, "campaign summary cannot be read"
    except json.JSONDecodeError:
        return None, "campaign summary contains invalid JSON"
    if not isinstance(payload, dict):
        return None, "campaign summary must be a JSON object"
    return payload, None


def _read_episode_rows(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Read one JSONL arm artifact without accepting malformed rows.

    Returns:
        Parsed row objects and an optional malformed-artifact error.
    """
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    return [], f"episode artifact line {line_number}: invalid JSON"
                if not isinstance(payload, dict):
                    return [], f"episode artifact line {line_number}: episode row must be an object"
                rows.append(payload)
    except OSError:
        return [], "cannot read episode artifact"
    return rows, None


def _read_campaign_object(
    campaign_root: Path, filename: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Read one fixed campaign metadata object without following symlink escapes.

    Returns:
        Parsed object and an optional shaped error.
    """
    path = campaign_root / filename
    if _path_has_symlink_component(path):
        return None, f"{filename} contains a symlink component"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return None, f"{filename} cannot be read"
    except json.JSONDecodeError:
        return None, f"{filename} contains invalid JSON"
    if not isinstance(payload, dict):
        return None, f"{filename} must contain a JSON object"
    return payload, None


def _declared_path_matches(
    raw_path: Any,
    *,
    expected_path: Path,
    campaign_root: Path,
    source_repository_root: Path | None = None,
) -> bool:
    """Match a producer path against a trusted repository asset.

    Returns:
        Whether the path resolves exactly to the expected asset.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        return False
    candidate = Path(raw_path.strip())
    expected = expected_path.resolve()
    trusted_root = Path(source_repository_root or get_repository_root()).resolve()
    candidates = (
        (candidate.resolve(),)
        if candidate.is_absolute()
        else (
            (trusted_root / candidate).resolve(),
            (campaign_root / candidate).resolve(),
        )
    )
    return any(item == expected for item in candidates)


def _source_repository_path(raw_path: Any, source_repository_root: Path) -> Path:
    """Resolve a canonical source path relative to the trusted source checkout.

    Returns:
        Absolute path anchored at ``source_repository_root``.

    Raises:
        ValueError: If an absolute path is outside both the trusted source and validator roots.
    """
    path = Path(str(raw_path))
    source_root = source_repository_root.resolve()
    validator_root = get_repository_root().resolve()
    if not path.is_absolute():
        resolved = (source_root / path).resolve()
        try:
            resolved.relative_to(source_root)
        except ValueError as exc:
            raise ValueError("source path escapes trusted repository root") from exc
        return resolved
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(source_root)
    except ValueError:
        try:
            relative = resolved.relative_to(validator_root)
        except ValueError as exc:
            raise ValueError("source path is outside trusted repositories") from exc
        return (source_root / relative).resolve()
    return (source_root / relative).resolve()


def _stress_seed_policy_payload(
    campaign_config: Any, resolved_seeds: tuple[int, ...]
) -> dict[str, Any]:
    """Return the canonical seed-policy fields emitted by camera-ready runs."""
    seed_policy = getattr(campaign_config, "seed_policy", None)
    return {
        "mode": getattr(seed_policy, "mode", None),
        "seed_set": getattr(seed_policy, "seed_set", None),
        "seeds": list(getattr(seed_policy, "seeds", ()) or ()),
        "resolved_seeds": list(resolved_seeds),
        "seed_sets_path": getattr(seed_policy, "seed_sets_path", None),
    }


def _stress_metadata_contract_blockers(  # noqa: C901, PLR0912, PLR0915
    campaign_root: Path,
    *,
    manifest: Any,
    campaign_config: Any,
    expected_source_commit: str,
    scenarios: list[dict[str, Any]],
    resolved_seeds: tuple[int, ...],
) -> tuple[list[str], dict[str, Any] | None]:
    """Bind campaign-level metadata to the release manifest and campaign config.

    Returns:
        Blockers and the parsed campaign manifest payload.
    """
    blockers: list[str] = []
    root = campaign_root.resolve()
    expected_scenario_path = Path(getattr(manifest, "scenario_matrix_path", "")).resolve()
    expected_scenario_hash = str(getattr(manifest, "scenario_matrix_sha256", "")).strip().lower()
    expected_scenario_identity = _scenario_matrix_hash(scenarios)
    expected_config_identity = _config_hash(_config_hash_payload(campaign_config))
    expected_config_path = Path(
        getattr(campaign_config, "source_config_path", None)
        or getattr(manifest, "canonical_campaign_config_path", "")
    ).resolve()
    expected_config_hash = str(getattr(manifest, "campaign_config_sha256", "")).strip().lower()
    expected_route_path = getattr(campaign_config, "route_clearance_certifications_path", None)
    expected_weights_path = getattr(campaign_config, "snqi_weights_path", None)
    expected_baseline_path = getattr(campaign_config, "snqi_baseline_path", None)
    expected_seed_policy = _stress_seed_policy_payload(campaign_config, resolved_seeds)
    expected_seed_path = Path(expected_seed_policy["seed_sets_path"]).resolve()

    if expected_scenario_path != Path(campaign_config.scenario_matrix_path).resolve():
        blockers.append("release manifest and campaign config scenario paths differ")
    try:
        observed_config_hash = sha256_file(expected_config_path)
    except OSError:
        observed_config_hash = ""
    if expected_config_hash and observed_config_hash.lower() != expected_config_hash:
        blockers.append("release manifest campaign config hash does not match the pinned file")
    if expected_scenario_hash != sha256_file(expected_scenario_path):
        blockers.append("release manifest scenario matrix hash does not match the pinned file")
    for label, path, expected_hash in (
        (
            "seed-set",
            expected_seed_path,
            getattr(manifest, "stress_smoke_seed_sets_sha256", None)
            or getattr(manifest, "seed_sets_sha256", None),
        ),
        (
            "route-certification",
            expected_route_path,
            getattr(manifest, "stress_smoke_route_certification_sha256", None)
            or getattr(manifest, "route_certification_sha256", None),
        ),
        ("SNQI weights", expected_weights_path, getattr(manifest, "snqi_weights_sha256", None)),
        ("SNQI baseline", expected_baseline_path, getattr(manifest, "snqi_baseline_sha256", None)),
    ):
        if path is None or not expected_hash:
            continue
        try:
            observed_hash = sha256_file(Path(path))
        except OSError:
            observed_hash = ""
        if observed_hash.lower() != str(expected_hash).strip().lower():
            blockers.append(f"release manifest {label} hash does not match the pinned file")

    campaign_manifest, error = _read_campaign_object(root, "campaign_manifest.json")
    if error or campaign_manifest is None:
        blockers.append(error or "campaign_manifest.json cannot be read")
    else:
        git = campaign_manifest.get("git")
        git_commit = git.get("commit") if isinstance(git, Mapping) else None
        if str(git_commit or "").strip().lower() != expected_source_commit:
            blockers.append("campaign_manifest.git.commit does not match the runtime source commit")
        if (
            str(campaign_manifest.get("config_hash", "")).strip().lower()
            != expected_config_identity
        ):
            blockers.append("campaign_manifest.config_hash does not match the campaign config")
        if not _declared_path_matches(
            campaign_manifest.get("scenario_matrix"),
            expected_path=expected_scenario_path,
            campaign_root=root,
        ):
            blockers.append("campaign_manifest.scenario_matrix is not the pinned scenario matrix")
        if (
            str(campaign_manifest.get("scenario_matrix_hash", "")).strip()
            != expected_scenario_identity
        ):
            blockers.append(
                "campaign_manifest.scenario_matrix_hash does not match resolved scenarios"
            )

        seed_block = campaign_manifest.get("seed_policy")
        if not isinstance(seed_block, Mapping):
            blockers.append("campaign_manifest.seed_policy is missing")
        else:
            for field in ("mode", "seed_set", "seeds", "resolved_seeds"):
                if seed_block.get(field) != expected_seed_policy[field]:
                    blockers.append(f"campaign_manifest.seed_policy.{field} does not match config")
            if not _declared_path_matches(
                seed_block.get("seed_sets_path"),
                expected_path=expected_seed_path,
                campaign_root=root,
            ):
                blockers.append("campaign_manifest.seed_policy.seed_sets_path is not pinned")

        if expected_route_path is not None and not _declared_path_matches(
            campaign_manifest.get("route_clearance_certifications_path"),
            expected_path=Path(expected_route_path),
            campaign_root=root,
        ):
            blockers.append("campaign_manifest route certification path is not pinned")
        for field, expected_path in (
            ("snqi_weights_path", expected_weights_path),
            ("snqi_baseline_path", expected_baseline_path),
        ):
            if expected_path is not None and not _declared_path_matches(
                campaign_manifest.get(field),
                expected_path=Path(expected_path),
                campaign_root=root,
            ):
                blockers.append(f"campaign_manifest.{field} is not pinned")

    run_meta, error = _read_campaign_object(root, "run_meta.json")
    if error or run_meta is None:
        blockers.append(error or "run_meta.json cannot be read")
    else:
        repo = run_meta.get("repo")
        repo_commit = repo.get("commit") if isinstance(repo, Mapping) else None
        if str(repo_commit or "").strip().lower() != expected_source_commit:
            blockers.append("run_meta.repo.commit does not match the runtime source commit")
        if not _declared_path_matches(
            run_meta.get("matrix_path"), expected_path=expected_scenario_path, campaign_root=root
        ):
            blockers.append("run_meta.matrix_path is not the pinned scenario matrix")
        if str(run_meta.get("scenario_matrix_hash", "")).strip() != expected_scenario_identity:
            blockers.append("run_meta.scenario_matrix_hash does not match resolved scenarios")
        seed_block = run_meta.get("seed_policy")
        if not isinstance(seed_block, Mapping):
            blockers.append("run_meta.seed_policy is missing")
        else:
            for field in ("mode", "seed_set", "seeds", "resolved_seeds"):
                if seed_block.get(field) != expected_seed_policy[field]:
                    blockers.append(f"run_meta.seed_policy.{field} does not match config")
            if not _declared_path_matches(
                seed_block.get("seed_sets_path"),
                expected_path=expected_seed_path,
                campaign_root=root,
            ):
                blockers.append("run_meta.seed_policy.seed_sets_path is not pinned")

    run_manifest, error = _read_campaign_object(root, "manifest.json")
    if error or run_manifest is None:
        blockers.append(error or "manifest.json cannot be read")
    else:
        if str(run_manifest.get("git_hash", "")).strip().lower() != expected_source_commit:
            blockers.append("manifest.git_hash does not match the runtime source commit")
        if str(run_manifest.get("scenario_matrix_hash", "")).strip() != expected_scenario_identity:
            blockers.append("manifest.scenario_matrix_hash does not match resolved scenarios")

    summary, error = _read_campaign_summary(root)
    if error or summary is None:
        blockers.append(error or "campaign summary cannot be read")
    else:
        campaign = summary.get("campaign")
        if not isinstance(campaign, Mapping):
            blockers.append("campaign summary campaign block is missing")
        else:
            if str(campaign.get("git_hash", "")).strip().lower() != expected_source_commit:
                blockers.append("campaign.git_hash does not match the runtime source commit")
            if not _declared_path_matches(
                campaign.get("scenario_matrix"),
                expected_path=expected_scenario_path,
                campaign_root=root,
            ):
                blockers.append("campaign.scenario_matrix is not the pinned scenario matrix")
            if str(campaign.get("scenario_matrix_hash", "")).strip() != expected_scenario_identity:
                blockers.append("campaign.scenario_matrix_hash does not match resolved scenarios")
            if tuple(campaign.get("kinematics_matrix", ())) != (STRESS_SMOKE_EXPECTED_KINEMATICS,):
                blockers.append("campaign.kinematics_matrix is not differential_drive-only")
            for field, expected_path, expected_hash in (
                (
                    "snqi_weights_sha256",
                    expected_weights_path,
                    getattr(manifest, "snqi_weights_sha256", None),
                ),
                (
                    "snqi_baseline_sha256",
                    expected_baseline_path,
                    getattr(manifest, "snqi_baseline_sha256", None),
                ),
            ):
                if (
                    expected_path is not None
                    and str(campaign.get(field, "")).strip().lower()
                    != str(expected_hash or "").strip().lower()
                ):
                    blockers.append(f"campaign.{field} does not match the release manifest")

    return blockers, campaign_manifest


def _stress_episode_provenance_blockers(  # noqa: C901, PLR0912, PLR0913, PLR0915
    episodes_path: Path,
    *,
    campaign_root: Path,
    source_repository_root: Path | None = None,
    planner_key: str,
    expected_algo: str,
    expected_source_commit: str,
    expected_scenario_path: Path,
    expected_scenario_hash: str,
    expected_scenario_identity: str,
    expected_algo_config_path: Path | None,
    expected_rows: list[dict[str, Any]],
) -> list[str]:
    """Require one complete, input-bound result-provenance sidecar per stress arm.

    Returns:
        Blockers for missing, stale, or mismatched sidecar bindings.
    """
    blockers: list[str] = []
    trusted_source_root = Path(source_repository_root or get_repository_root()).resolve()
    sidecar_path = episodes_path.with_name(f"{episodes_path.name}.provenance.json")
    if _path_has_symlink_component(sidecar_path) or not sidecar_path.is_file():
        return [f"planner {planner_key} episode provenance sidecar is missing"]
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [f"planner {planner_key} episode provenance sidecar cannot be read"]
    if not isinstance(payload, Mapping):
        return [f"planner {planner_key} episode provenance sidecar must be an object"]
    try:
        validate_result_provenance_manifest(payload)
    except (TypeError, ValueError):
        blockers.append(f"planner {planner_key} episode provenance schema rejected")

    run = payload.get("run")
    run = run if isinstance(run, Mapping) else {}
    if str(run.get("repo_commit", "")).strip().lower() != expected_source_commit:
        blockers.append(f"planner {planner_key} sidecar source commit is not the runtime commit")
    if run.get("runner") != "map_runner.run_map_batch":
        blockers.append(f"planner {planner_key} sidecar runner is not map_runner.run_map_batch")
    completeness = payload.get("completeness")
    completeness = completeness if isinstance(completeness, Mapping) else {}
    if completeness.get("status") != "complete":
        blockers.append(f"planner {planner_key} sidecar completeness is not complete")

    identity = payload.get("campaign_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    if str(identity.get("algorithm", "")).strip().lower() != expected_algo.lower():
        blockers.append(f"planner {planner_key} sidecar algorithm is not bound to its arm")
    expected_suite_key = _producer_suite_key(expected_scenario_path)
    if identity.get("suite_key") != expected_suite_key:
        blockers.append(f"planner {planner_key} sidecar suite key must be {expected_suite_key}")
    if identity.get("scenario_matrix_hash") != expected_scenario_identity:
        blockers.append(f"planner {planner_key} sidecar scenario identity hash is not bound")
    expected_identity_config = _config_hash(
        {
            "schema_path": str(
                trusted_source_root / "robot_sf/benchmark/schemas/episode.schema.v1.json"
            ),
            "algo": expected_algo,
            "algo_config_path": str(expected_algo_config_path)
            if expected_algo_config_path is not None
            else None,
        }
    )
    if identity.get("config_hash") != expected_identity_config:
        blockers.append(f"planner {planner_key} sidecar config identity is not bound")
    expected_count = len(expected_rows)
    for field in ("total_jobs", "written"):
        if _strict_int(identity.get(field)) != expected_count:
            blockers.append(f"planner {planner_key} sidecar {field} must be {expected_count}")

    inputs = payload.get("inputs")
    inputs = inputs if isinstance(inputs, Mapping) else {}
    schema_path = trusted_source_root / "robot_sf/benchmark/schemas/episode.schema.v1.json"
    schema_input = inputs.get("schema_path")
    schema_input = schema_input if isinstance(schema_input, Mapping) else {}
    if not _declared_path_matches(
        schema_input.get("path"),
        expected_path=schema_path,
        campaign_root=campaign_root,
        source_repository_root=trusted_source_root,
    ):
        blockers.append(f"planner {planner_key} sidecar schema path is not repository-pinned")
    try:
        schema_hash = sha256_file(schema_path)
    except OSError:
        schema_hash = ""
    if str(schema_input.get("sha256", "")).strip().lower() != schema_hash:
        blockers.append(f"planner {planner_key} sidecar schema hash is not repository-pinned")

    scenario_input = inputs.get("scenario_matrix")
    scenario_input = scenario_input if isinstance(scenario_input, Mapping) else {}
    if not _declared_path_matches(
        scenario_input.get("path"),
        expected_path=expected_scenario_path,
        campaign_root=campaign_root,
        source_repository_root=trusted_source_root,
    ):
        blockers.append(f"planner {planner_key} sidecar scenario path is not release-pinned")
    if str(scenario_input.get("sha256", "")).strip().lower() != expected_scenario_hash:
        blockers.append(f"planner {planner_key} sidecar scenario hash is not release-pinned")

    config_input = inputs.get("algo_config")
    config_input = config_input if isinstance(config_input, Mapping) else {}
    if expected_algo_config_path is None:
        if (
            config_input.get("artifact_status") != "not_provided"
            or config_input.get("path") is not None
            or config_input.get("sha256") is not None
        ):
            blockers.append(
                f"planner {planner_key} sidecar unexpectedly declares an algorithm config"
            )
    else:
        if not _declared_path_matches(
            config_input.get("path"),
            expected_path=expected_algo_config_path,
            campaign_root=campaign_root,
            source_repository_root=trusted_source_root,
        ):
            blockers.append(f"planner {planner_key} sidecar algorithm config path is not pinned")
        try:
            config_hash = sha256_file(expected_algo_config_path)
        except OSError:
            config_hash = ""
        if str(config_input.get("sha256", "")).strip().lower() != config_hash:
            blockers.append(f"planner {planner_key} sidecar algorithm config hash is not pinned")

    raw_artifacts = payload.get("raw_artifacts")
    raw_artifacts = raw_artifacts if isinstance(raw_artifacts, list) else []
    episode_artifacts = [
        item
        for item in raw_artifacts
        if isinstance(item, Mapping) and item.get("kind") == "episodes_jsonl"
    ]
    if len(episode_artifacts) != 1:
        blockers.append(f"planner {planner_key} sidecar must contain one episodes_jsonl artifact")
    else:
        artifact = episode_artifacts[0]
        if not _declared_path_matches(
            artifact.get("path"),
            expected_path=episodes_path,
            campaign_root=campaign_root,
            source_repository_root=trusted_source_root,
        ):
            blockers.append(f"planner {planner_key} sidecar raw artifact is not the run artifact")
        try:
            artifact_hash = sha256_file(episodes_path)
        except OSError:
            artifact_hash = ""
        if str(artifact.get("sha256", "")).strip().lower() != artifact_hash:
            blockers.append(f"planner {planner_key} sidecar raw artifact hash is stale")

    sidecar_rows = payload.get("rows")
    if not isinstance(sidecar_rows, list) or len(sidecar_rows) != expected_count:
        blockers.append(f"planner {planner_key} sidecar must bind every episode row")
        sidecar_rows = []
    for row_index, (row, sidecar_row) in enumerate(zip(expected_rows, sidecar_rows, strict=False)):
        if not isinstance(sidecar_row, Mapping):
            blockers.append(f"planner {planner_key} sidecar row {row_index} is not an object")
            continue
        row_provenance = row.get("result_provenance")
        row_provenance = row_provenance if isinstance(row_provenance, Mapping) else {}
        row_config = str(row_provenance.get("config_hash") or row.get("config_hash") or "").strip()
        row_commit = _source_commit(row)
        for field, expected in (
            ("episode_id", row.get("episode_id")),
            ("scenario_id", row.get("scenario_id")),
            ("seed", _strict_int(row.get("seed"))),
            ("config_hash", row_config),
            ("repo_commit", expected_source_commit),
            ("jsonl_line", row_index),
        ):
            observed = sidecar_row.get(field)
            if field in {"seed", "jsonl_line"}:
                observed = _strict_int(observed)
            elif field in {"config_hash", "repo_commit", "episode_id", "scenario_id"}:
                observed = str(observed or "").strip().lower()
                expected = str(expected or "").strip().lower()
            if observed != expected:
                blockers.append(
                    f"planner {planner_key} sidecar row {row_index} {field} is not bound"
                )
        if not row.get("episode_id"):
            blockers.append(f"planner {planner_key} episode row {row_index} has no episode_id")
        if row_commit != expected_source_commit:
            blockers.append(f"planner {planner_key} episode row {row_index} source is not bound")
        sidecar_settings = sidecar_row.get("simulator_settings")
        sidecar_settings = sidecar_settings if isinstance(sidecar_settings, Mapping) else {}
        if _strict_int(sidecar_settings.get("horizon")) != STRESS_SMOKE_EXPECTED_HORIZON_STEPS:
            blockers.append(f"planner {planner_key} sidecar row {row_index} horizon is not 600")
        try:
            sidecar_dt = float(sidecar_settings.get("dt"))
        except (TypeError, ValueError):
            sidecar_dt = float("nan")
        if sidecar_dt != STRESS_SMOKE_EXPECTED_DT:
            blockers.append(f"planner {planner_key} sidecar row {row_index} dt is not 0.1")
        if not _declared_path_matches(
            sidecar_row.get("raw_artifact"),
            expected_path=episodes_path,
            campaign_root=campaign_root,
            source_repository_root=trusted_source_root,
        ):
            blockers.append(
                f"planner {planner_key} sidecar row {row_index} raw artifact is not bound"
            )
    return blockers


def _path_has_symlink_component(path: Path) -> bool:
    """Return whether a lexical path component is a symlink."""
    lexical = Path(path.absolute())
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _resolve_stress_artifact_path(  # noqa: C901, PLR0912
    campaign_root: Path,
    raw_path: str,
    *,
    arm: tuple[str, str] | None = None,
    kind: str = "episodes",
) -> Path:
    """Resolve a stress artifact only to its campaign-root arm location.

    Camera-ready summaries historically use either ``runs/...`` paths or
    repository-relative ``output/.../campaign/runs/...`` paths.  Both forms are
    accepted only after reducing to the exact expected suffix; an absolute path
    must already name that exact file.  No generic repository-root fallback is
    permitted here because it can silently read another campaign.

    Returns:
        The existing campaign-root artifact path.
    """
    root = Path(campaign_root).absolute()
    if _path_has_symlink_component(root):
        raise ValueError("campaign root must not contain symlink components")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"campaign root is not a directory: {root}")
    raw = str(raw_path or "").strip()
    if not raw:
        raise ValueError("artifact path is empty")
    candidate_raw = Path(raw)
    if any(part == ".." for part in candidate_raw.parts):
        raise ValueError("artifact path may not contain parent traversal")

    expected_filename = "episodes.jsonl" if kind == "episodes" else "summary.json"
    if arm is not None:
        planner_key, kinematics = arm
        if not planner_key or not kinematics:
            raise ValueError("artifact arm is incomplete")
        expected_relative = Path("runs") / f"{planner_key}__{kinematics}" / expected_filename
    else:
        expected_relative = None

    if candidate_raw.is_absolute():
        candidate = candidate_raw
        if expected_relative is None:
            try:
                relative = candidate.resolve().relative_to(root)
            except ValueError as exc:
                raise ValueError("absolute artifact path is outside campaign root") from exc
            if relative.parts[:1] != ("runs",) or relative.name != expected_filename:
                raise ValueError("absolute artifact path is not a campaign run artifact")
        elif candidate.resolve() != (root / expected_relative).resolve():
            raise ValueError("absolute artifact path does not match expected campaign arm")
    else:
        parts = candidate_raw.parts
        try:
            run_index = parts.index("runs")
        except ValueError as exc:
            raise ValueError("artifact path must contain the campaign runs directory") from exc
        relative = Path(*parts[run_index:])
        if expected_relative is not None and relative != expected_relative:
            raise ValueError("artifact path does not match expected campaign arm")
        if len(relative.parts) != 3 or relative.name != expected_filename:
            raise ValueError("artifact path must be runs/<arm>/<artifact>")
        candidate = root / relative

    candidate = candidate.absolute()
    if _path_has_symlink_component(candidate):
        raise ValueError("artifact path must not contain symlink components")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact path resolves outside campaign root") from exc
    if expected_relative is not None and resolved != (root / expected_relative).resolve():
        raise ValueError("artifact path resolves to a different campaign arm")
    if not resolved.is_file():
        raise ValueError(f"artifact path is not a file: {resolved}")
    return resolved


def _nested_value(payload: Mapping[str, Any], *keys: str) -> Any:
    """Read a nested mapping value without coercing absent fields.

    Returns:
        The nested value, or the private missing sentinel.
    """
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _alias_values(
    payload: Mapping[str, Any], aliases: tuple[tuple[str, Any], ...]
) -> list[tuple[str, str]]:
    """Return non-empty normalized aliases, retaining their source labels."""
    values: list[tuple[str, str]] = []
    for label, value in aliases:
        if value is _MISSING or value is None:
            continue
        normalized = str(value).strip().lower()
        if normalized:
            values.append((label, normalized))
    return values


def _alias_blockers(
    values: list[tuple[str, str]], *, label: str, expected: str | None = None
) -> list[str]:
    """Return conflicts/mismatches for one set of provenance aliases."""
    if not values:
        return [f"{label} aliases are missing"]
    distinct = {value for _, value in values}
    blockers: list[str] = []
    if len(distinct) != 1:
        blockers.append(f"{label} aliases conflict: {sorted(distinct)!r}")
    if expected is not None and any(value != expected for _, value in values):
        blockers.append(f"{label} aliases do not match declared value {expected!r}")
    return blockers


def _explicit_success(value: Any) -> bool:
    """Accept only the producer's explicit JSON/string true spellings.

    Returns:
        Whether the value is an explicit true token.
    """
    return value is True or (
        isinstance(value, str) and value.strip().lower() in _STRESS_VALID_BENCHMARK_SUCCESS
    )


def _status_is(value: Any, allowed: frozenset[str]) -> bool:
    """Match a status against a small, explicitly accepted stress vocabulary.

    Returns:
        Whether the normalized status is in ``allowed``.
    """
    return _normalized_status(value) in allowed


def _source_commit(row: Mapping[str, Any]) -> str:
    """Resolve the preferred source commit from a row's provenance fields.

    Returns:
        Lowercase source commit token, or an empty string when absent.
    """
    provenance = row.get("result_provenance")
    if isinstance(provenance, Mapping) and provenance.get("repo_commit"):
        return str(provenance["repo_commit"]).strip().lower()
    event_ledger = row.get("event_ledger")
    if isinstance(event_ledger, Mapping) and event_ledger.get("software_commit"):
        return str(event_ledger["software_commit"]).strip().lower()
    return str(row.get("git_hash", "")).strip().lower()


def validate_diagnostic_stress_smoke_source_provenance(  # noqa: C901, PLR0912, PLR0915
    campaign_root: Path,
    *,
    expected_source_commit: str,
) -> dict[str, Any]:
    """Require campaign metadata and every episode row to name one runtime HEAD.

    Returns:
        JSON-safe source provenance admission report.
    """
    expected = str(expected_source_commit or "").strip().lower()
    blockers: list[str] = []
    if _GIT_SHA_RE.fullmatch(expected) is None:
        blockers.append("expected runtime source commit is not an exact 40-character SHA")

    observed: set[str] = set()
    observations: dict[str, str] = {}

    def _record(label: str, value: Any) -> None:
        normalized = str(value or "").strip().lower()
        if _GIT_SHA_RE.fullmatch(normalized) is None:
            blockers.append(f"{label} is missing or not an exact 40-character SHA")
            return
        observations[label] = normalized
        observed.add(normalized)

    root = campaign_root.resolve()
    for filename, label, path_getter in (
        (
            "campaign_manifest.json",
            "campaign_manifest.git.commit",
            lambda payload: (
                payload.get("git", {}).get("commit")
                if isinstance(payload.get("git"), Mapping)
                else payload.get("git_hash")
            ),
        ),
        ("manifest.json", "manifest.git_hash", lambda payload: payload.get("git_hash")),
        (
            "run_meta.json",
            "run_meta.repo.commit",
            lambda payload: (
                payload.get("repo", {}).get("commit")
                if isinstance(payload.get("repo"), Mapping)
                else None
            ),
        ),
    ):
        path = root / filename
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(f"{label} cannot be read: {exc}")
            continue
        if not isinstance(payload, Mapping):
            blockers.append(f"{filename} must contain a JSON object")
            continue
        _record(label, path_getter(payload))

    summary, summary_error = _read_campaign_summary(root)
    if summary_error is not None or summary is None:
        blockers.append(summary_error or "campaign summary cannot be read")
    else:
        campaign = summary.get("campaign")
        if not isinstance(campaign, Mapping):
            blockers.append("campaign summary campaign block is missing")
        else:
            _record("campaign_summary.campaign.git_hash", campaign.get("git_hash"))

        runs = summary.get("runs")
        if not isinstance(runs, list):
            blockers.append("campaign summary runs list is missing")
            runs = []
        for run_index, run in enumerate(runs):
            if not isinstance(run, Mapping):
                blockers.append(f"runs[{run_index}] must be an object")
                continue
            raw_path = str(run.get("episodes_path", "")).strip()
            if not raw_path:
                blockers.append(f"runs[{run_index}] is missing episodes_path")
                continue
            planner = run.get("planner")
            planner = planner if isinstance(planner, Mapping) else {}
            arm = (
                str(planner.get("key", "")).strip(),
                str(planner.get("kinematics", "")).strip(),
            )
            try:
                episodes_path = _resolve_stress_artifact_path(
                    root,
                    raw_path,
                    arm=arm if all(arm) else None,
                )
            except (OSError, ValueError) as exc:
                blockers.append(f"runs[{run_index}] episodes_path rejected: {exc}")
                continue
            rows, error = _read_episode_rows(episodes_path)
            if error:
                blockers.append(error)
                continue
            for row_index, row in enumerate(rows):
                aliases = _alias_values(
                    row,
                    (
                        ("git_hash", row.get("git_hash", _MISSING)),
                        ("repo_commit", row.get("repo_commit", _MISSING)),
                        ("provenance.git_hash", _nested_value(row, "provenance", "git_hash")),
                        (
                            "result_provenance.repo_commit",
                            _nested_value(row, "result_provenance", "repo_commit"),
                        ),
                        (
                            "event_ledger.software_commit",
                            _nested_value(row, "event_ledger", "software_commit"),
                        ),
                    ),
                )
                row_label = f"runs[{run_index}].rows[{row_index}].source_commit"
                if _nested_value(row, "event_ledger", "software_commit") is _MISSING:
                    blockers.append(f"{row_label} event_ledger.software_commit is missing")
                blockers.extend(_alias_blockers(aliases, label=row_label, expected=expected))
                if aliases:
                    _record(row_label, aliases[0][1])

    if observed != {expected}:
        blockers.append(
            "campaign provenance must contain exactly the checked-out runtime source commit"
        )
    return {
        "schema_version": "benchmark-stress-smoke-source-provenance.v1",
        "status": "valid" if not blockers else "invalid",
        "expected_source_commit": expected or None,
        "observed_source_commits": sorted(observed),
        "observations": dict(sorted(observations.items())),
        "blockers": list(dict.fromkeys(blockers)),
    }


def _episode_horizon(row: Mapping[str, Any]) -> tuple[int | None, bool]:
    """Resolve an episode horizon and whether an authoritative value was present.

    Returns:
        Tuple of parsed horizon and presence flag.
    """
    if row.get("horizon") is not None:
        return _strict_int(row["horizon"]), True
    provenance = row.get("result_provenance")
    if isinstance(provenance, Mapping):
        settings = provenance.get("simulator_settings")
        if isinstance(settings, Mapping) and settings.get("horizon") is not None:
            return _strict_int(settings["horizon"]), True
    return None, False


def _episode_dt(row: Mapping[str, Any]) -> tuple[float | None, bool]:
    """Resolve the authoritative episode time step from result provenance.

    Returns:
        The parsed time step and whether an authoritative field was present.
    """
    provenance = row.get("result_provenance")
    if isinstance(provenance, Mapping):
        settings = provenance.get("simulator_settings")
        if isinstance(settings, Mapping) and settings.get("dt") is not None:
            try:
                value = float(settings["dt"])
            except (TypeError, ValueError):
                return None, True
            return value, True
    if row.get("dt") is not None:
        try:
            return float(row["dt"]), True
        except (TypeError, ValueError):
            return None, True
    return None, False


def _stress_row_contract_blockers(  # noqa: C901, PLR0912, PLR0915
    row: Mapping[str, Any],
    *,
    prefix: str,
    planner_key: str,
    expected_algo: str,
    expected_kinematics: str,
    expected_source_commit: str,
) -> list[str]:
    """Validate one episode's explicit arm, provenance, and simulator aliases.

    Returns:
        Bounded blocker messages for this row.
    """
    blockers: list[str] = []
    if not _status_is(row.get("status"), _STRESS_ROW_TERMINAL_STATUSES):
        blockers.append(f"{prefix}.status must be a recognized scientific terminal outcome")
    if "benchmark_success" in row and not _explicit_success(row.get("benchmark_success")):
        blockers.append(f"{prefix}.benchmark_success must be explicitly true")

    result_provenance = row.get("result_provenance")
    if not isinstance(result_provenance, Mapping):
        blockers.append(f"{prefix}.result_provenance is missing")

    source_aliases = _alias_values(
        row,
        (
            ("git_hash", row.get("git_hash", _MISSING)),
            ("repo_commit", row.get("repo_commit", _MISSING)),
            ("provenance.git_hash", _nested_value(row, "provenance", "git_hash")),
            (
                "result_provenance.repo_commit",
                _nested_value(row, "result_provenance", "repo_commit"),
            ),
            (
                "event_ledger.software_commit",
                _nested_value(row, "event_ledger", "software_commit"),
            ),
        ),
    )
    if _nested_value(row, "event_ledger", "software_commit") is _MISSING:
        blockers.append(f"{prefix}: event_ledger.software_commit is missing")
    blockers.extend(
        f"{prefix}: {blocker}"
        for blocker in _alias_blockers(
            source_aliases,
            label="source",
            expected=expected_source_commit,
        )
    )

    row_config_aliases = _alias_values(
        row,
        (
            ("config_hash", row.get("config_hash", _MISSING)),
            (
                "result_provenance.config_hash",
                _nested_value(row, "result_provenance", "config_hash"),
            ),
        ),
    )
    scenario_params = row.get("scenario_params")
    if isinstance(scenario_params, Mapping):
        expected_row_config_hash = _config_hash(dict(scenario_params))
    else:
        expected_row_config_hash = None
        blockers.append(f"{prefix}: scenario_params is missing")
    blockers.extend(
        f"{prefix}: {blocker}"
        for blocker in _alias_blockers(
            row_config_aliases,
            label="row config",
            expected=expected_row_config_hash,
        )
    )
    provenance_config_hash = _nested_value(row, "provenance", "config_hash")
    analysis_trace = _nested_value(row, "algorithm_metadata", "analysis_trace")
    if isinstance(analysis_trace, Mapping):
        trace_config_aliases = _alias_values(
            row,
            (
                ("provenance.config_hash", provenance_config_hash),
                (
                    "algorithm_metadata.analysis_trace.config_hash",
                    analysis_trace.get("config_hash", _MISSING),
                ),
            ),
        )
        blockers.extend(
            f"{prefix}: {blocker}"
            for blocker in _alias_blockers(
                trace_config_aliases,
                label="analysis trace config",
            )
        )
    else:
        if analysis_trace is not _MISSING and analysis_trace is not None:
            blockers.append(f"{prefix}: algorithm_metadata.analysis_trace must be a mapping")
        blockers.extend(
            f"{prefix}: {blocker}"
            for blocker in _alias_blockers(
                _alias_values(
                    row,
                    (("provenance.config_hash", provenance_config_hash),),
                ),
                label="provenance config",
                expected=expected_row_config_hash,
            )
        )

    metadata_algorithm = _nested_value(row, "algorithm_metadata", "algorithm")
    expected_metadata_algorithm = "ppo" if expected_algo == "guarded_ppo" else expected_algo
    if metadata_algorithm is _MISSING:
        blockers.append(f"{prefix}: algorithm_metadata.algorithm is missing")
    elif str(metadata_algorithm).strip().lower() != expected_metadata_algorithm:
        blockers.append(
            f"{prefix}: algorithm_metadata.algorithm does not match declared base algorithm "
            f"'{expected_metadata_algorithm}'"
        )
    algo_aliases = _alias_values(
        row,
        (
            ("algo", row.get("algo", _MISSING)),
            (
                "algorithm_metadata.canonical_algorithm",
                _nested_value(row, "algorithm_metadata", "canonical_algorithm"),
            ),
            (
                "algorithm_metadata.planner_contract.planner_id",
                _nested_value(
                    row,
                    "algorithm_metadata",
                    "planner_contract",
                    "planner_id",
                ),
            ),
            (
                "provenance.config_identity.algo",
                _nested_value(row, "provenance", "config_identity", "algo"),
            ),
        ),
    )
    blockers.extend(
        f"{prefix}: {blocker}"
        for blocker in _alias_blockers(
            algo_aliases, label="planner algorithm", expected=expected_algo
        )
    )

    planner_key_aliases = _alias_values(
        row,
        (
            ("planner_key", row.get("planner_key", _MISSING)),
            (
                "result_provenance.planner_key",
                _nested_value(row, "result_provenance", "planner_key"),
            ),
            (
                "provenance.config_identity.planner_key",
                _nested_value(row, "provenance", "config_identity", "planner_key"),
            ),
        ),
    )
    if planner_key_aliases:
        blockers.extend(
            f"{prefix}: {blocker}"
            for blocker in _alias_blockers(
                planner_key_aliases,
                label="planner key",
                expected=planner_key.lower(),
            )
        )

    kinematics_aliases: list[tuple[str, str]] = []
    scalar_kinematics = (
        ("kinematics", row.get("kinematics", _MISSING)),
        (
            "result_provenance.simulator_settings.kinematics",
            _nested_value(row, "result_provenance", "simulator_settings", "kinematics"),
        ),
        (
            "algorithm_metadata.planner_kinematics.robot_kinematics",
            _nested_value(
                row,
                "algorithm_metadata",
                "planner_kinematics",
                "robot_kinematics",
            ),
        ),
        (
            "algorithm_metadata.planner_contract.action_contract.active_robot_kinematics",
            _nested_value(
                row,
                "algorithm_metadata",
                "planner_contract",
                "action_contract",
                "active_robot_kinematics",
            ),
        ),
    )
    kinematics_aliases.extend(_alias_values(row, scalar_kinematics))
    scenario_kinematics = _nested_value(
        row,
        "algorithm_metadata",
        "planner_kinematics",
        "scenario_kinematics",
    )
    if scenario_kinematics is not _MISSING:
        if not isinstance(scenario_kinematics, (list, tuple)) or not scenario_kinematics:
            blockers.append(f"{prefix}: scenario kinematics alias is malformed")
        else:
            kinematics_aliases.extend(
                (
                    "algorithm_metadata.planner_kinematics.scenario_kinematics",
                    str(value).strip().lower(),
                )
                for value in scenario_kinematics
            )
    # The current episode-row producer binds the arm through the strict run
    # directory and run-level planner object; it does not repeat planner
    # kinematics in every row.  Treat row-level kinematics as an optional alias
    # that must agree when present, rather than inventing a required field that
    # would reject the real job14730 schema.
    if kinematics_aliases:
        blockers.extend(
            f"{prefix}: {blocker}"
            for blocker in _alias_blockers(
                kinematics_aliases,
                label="kinematics",
                expected=expected_kinematics.lower(),
            )
        )

    scenario_id = str(row.get("scenario_id", "")).strip()
    provenance_scenario_id = _nested_value(row, "result_provenance", "scenario_id")
    if not scenario_id or provenance_scenario_id is _MISSING:
        blockers.append(f"{prefix}: scenario_id aliases are missing")
    elif scenario_id != str(provenance_scenario_id).strip():
        blockers.append(f"{prefix}: scenario_id aliases conflict")
    seed = _strict_int(row.get("seed"))
    provenance_seed = _strict_int(_nested_value(row, "result_provenance", "seed"))
    if seed is None or provenance_seed is None:
        blockers.append(f"{prefix}: seed aliases are missing or invalid")
    elif seed != provenance_seed:
        blockers.append(f"{prefix}: seed aliases conflict")

    row_horizon = _strict_int(row.get("horizon"))
    provenance_horizon = _strict_int(
        _nested_value(row, "result_provenance", "simulator_settings", "horizon")
    )
    if row_horizon != STRESS_SMOKE_EXPECTED_HORIZON_STEPS:
        blockers.append(f"{prefix}: horizon must be 600")
    if provenance_horizon != STRESS_SMOKE_EXPECTED_HORIZON_STEPS:
        blockers.append(f"{prefix}: result provenance horizon must be 600")
    episode_dt, dt_present = _episode_dt(row)
    if not dt_present or episode_dt is None or episode_dt != STRESS_SMOKE_EXPECTED_DT:
        blockers.append(f"{prefix}: result provenance dt must be 0.1")
    if row.get("dt") is not None:
        try:
            if float(row["dt"]) != STRESS_SMOKE_EXPECTED_DT:
                blockers.append(f"{prefix}: row dt must be 0.1")
        except (TypeError, ValueError):
            blockers.append(f"{prefix}: row dt is malformed")

    scenario_params = row.get("scenario_params")
    scenario_params = scenario_params if isinstance(scenario_params, Mapping) else {}
    robot_config = scenario_params.get("robot_config")
    robot_config = robot_config if isinstance(robot_config, Mapping) else {}
    robot_type = robot_config.get("type", _MISSING)
    if robot_type is _MISSING or not str(robot_type).strip():
        blockers.append(f"{prefix}: scenario_params.robot_config.type is missing")
    elif str(robot_type).strip().lower() != expected_kinematics.lower():
        blockers.append(
            f"{prefix}: scenario_params.robot_config.type must be {expected_kinematics!r}"
        )
    run_dt = scenario_params.get("run_dt", _MISSING)
    try:
        parsed_run_dt = float(run_dt) if run_dt is not _MISSING else None
    except (TypeError, ValueError):
        parsed_run_dt = None
    if parsed_run_dt is None or parsed_run_dt != STRESS_SMOKE_EXPECTED_DT:
        blockers.append(f"{prefix}: scenario_params.run_dt must be 0.1")
    return blockers


def _scenario_id(scenario: Mapping[str, Any]) -> str:
    """Resolve the stable scenario identifier used by campaign episode identity.

    Returns:
        Stable scenario identifier, or an empty string when absent.
    """
    return str(
        scenario.get("id") or scenario.get("scenario_id") or scenario.get("name") or ""
    ).strip()


def _result_provenance_scenarios(
    campaign_config: Any,
    scenarios: list[dict[str, Any]],
    *,
    kinematics: str,
) -> list[dict[str, Any]]:
    """Mirror the scenario payload hashed by map-runner result sidecars.

    The campaign runner adds kinematics and top-level telemetry. Map runner then
    adds normalized telemetry under ``metadata`` before emitting provenance.

    Returns:
        Scenarios in their producer-side result-provenance form.
    """
    effective_scenarios = [
        _scenario_with_kinematics(
            scenario,
            kinematics=kinematics,
            holonomic_command_mode=str(getattr(campaign_config, "holonomic_command_mode", "vx_vy")),
        )
        for scenario in scenarios
    ]
    telemetry = getattr(campaign_config, "telemetry", None)
    if isinstance(telemetry, Mapping):
        normalized_telemetry = normalize_telemetry_profile(dict(telemetry)).to_mapping()
        for scenario in effective_scenarios:
            scenario["telemetry"] = dict(telemetry)
            metadata = scenario.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
            metadata["telemetry"] = dict(normalized_telemetry)
            scenario["metadata"] = metadata
    return effective_scenarios


def _resolve_expected_matrix_axes(  # noqa: C901
    manifest: Any,
    campaign_config: Any | None,
    source_repository_root: Path | None = None,
) -> tuple[tuple[str, ...], tuple[int, ...], list[str]]:
    """Resolve the exact scenario/seed axes pinned by the manifest and campaign config.

    Returns:
        Scenario IDs, resolved seeds, and any axis-resolution blockers.
    """
    blockers: list[str] = []
    if campaign_config is None:
        config_path = getattr(manifest, "canonical_campaign_config_path", None)
        if config_path is not None:
            try:
                source_root = Path(source_repository_root or get_repository_root()).resolve()
                campaign_config = load_campaign_config(
                    _source_repository_path(config_path, source_root)
                )
            except (OSError, ValueError, KeyError, TypeError):
                blockers.append("canonical campaign config cannot be resolved")
    if campaign_config is not None:
        try:
            scenarios = _load_campaign_scenarios(campaign_config)
            scenario_ids = tuple(_scenario_id(scenario) for scenario in scenarios)
            seeds = tuple(_resolved_seed_inventory(scenarios))
        except (OSError, ValueError, KeyError, TypeError):
            blockers.append("resolved campaign matrix cannot be loaded")
            return (), (), blockers
        if any(not scenario_id for scenario_id in scenario_ids):
            blockers.append("resolved campaign matrix contains an empty scenario identifier")
        if len(set(scenario_ids)) != len(scenario_ids):
            blockers.append("resolved campaign matrix contains duplicate scenario identifiers")
        raw_manifest_seeds = getattr(manifest, "resolved_seeds", ())
        if not raw_manifest_seeds:
            seed_policy = getattr(manifest, "seed_policy", None)
            if isinstance(seed_policy, Mapping):
                raw_manifest_seeds = seed_policy.get("seeds", ())
        manifest_seeds = tuple(
            seed for raw_seed in raw_manifest_seeds if (seed := _strict_int(raw_seed)) is not None
        )
        if manifest_seeds != seeds:
            blockers.append("manifest resolved seeds do not match campaign config")
        return scenario_ids, seeds, blockers

    # Lightweight fixtures may provide the already-resolved axes directly. Real v0.2
    # runs always pass the loaded canonical campaign config from the release runner.
    scenario_ids = tuple(
        str(value).strip() for value in getattr(manifest, "resolved_scenario_ids", ())
    )
    seeds = tuple(
        seed
        for raw_seed in getattr(manifest, "resolved_seeds", ())
        if (seed := _strict_int(raw_seed)) is not None
    )
    if not scenario_ids or not seeds:
        blockers.append("resolved campaign scenario/seed axes are unavailable")
    return scenario_ids, seeds, blockers


def _full_release_planner_items(candidates: Any) -> tuple[tuple[Any, Any], ...]:
    """Return planner key/algo pairs from a config or test manifest mapping."""
    if isinstance(candidates, Mapping):
        return tuple(candidates.items())
    if not isinstance(candidates, (list, tuple)):
        return ()
    items: list[tuple[Any, Any]] = []
    for planner in candidates:
        if isinstance(planner, Mapping):
            items.append((planner.get("key"), planner.get("algo")))
        else:
            items.append((getattr(planner, "key", None), getattr(planner, "algo", None)))
    return tuple(items)


def _full_release_planner_candidates(
    manifest: Any,
    campaign_config: Any | None,
    source_repository_root: Path | None = None,
) -> tuple[Any, list[str]]:
    """Resolve the source of the publication planner roster.

    Returns:
        Candidate planner roster and resolution blockers.
    """
    blockers: list[str] = []
    candidates: Any = getattr(campaign_config, "planners", None)
    if candidates is not None:
        return candidates, blockers
    candidates = getattr(manifest, "planner_algorithms", None)
    if candidates is not None:
        return candidates, blockers
    config_path = getattr(manifest, "canonical_campaign_config_path", None)
    if config_path is None:
        return None, blockers
    try:
        source_root = Path(source_repository_root or get_repository_root()).resolve()
        candidates = getattr(
            load_campaign_config(_source_repository_path(config_path, source_root)),
            "planners",
            None,
        )
    except (OSError, ValueError, KeyError, TypeError):
        blockers.append("canonical campaign planner roster cannot be resolved")
    return candidates, blockers


def _full_release_campaign_config(
    manifest: Any,
    campaign_config: Any | None,
    source_repository_root: Path | None = None,
) -> tuple[Any | None, list[str]]:
    """Resolve the canonical config required for arm-bound publication provenance.

    Returns:
        The loaded campaign config and any fail-closed resolution blockers.
    """
    if campaign_config is not None:
        return campaign_config, []
    config_path = getattr(manifest, "canonical_campaign_config_path", None)
    if config_path is None:
        return None, ["canonical campaign config is required for full-release provenance"]
    try:
        source_root = Path(source_repository_root or get_repository_root()).resolve()
        return load_campaign_config(_source_repository_path(config_path, source_root)), []
    except (OSError, ValueError, KeyError, TypeError):
        return None, ["canonical campaign config cannot be resolved for provenance"]


def _full_release_algorithm_roster(
    manifest: Any,
    campaign_config: Any | None,
    planner_keys: tuple[str, ...],
    source_repository_root: Path | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Resolve the expected algorithm for every publication-grade planner arm.

    Returns:
        An arm-to-algorithm mapping and blockers for an unavailable or conflicting roster.
    """
    candidates, blockers = _full_release_planner_candidates(
        manifest, campaign_config, source_repository_root
    )
    algorithms: dict[str, str] = {}
    items = _full_release_planner_items(candidates)
    for raw_key, raw_algo in items:
        key = str(raw_key or "").strip()
        algo = str(raw_algo or "").strip().lower()
        if not key or not algo:
            blockers.append("full-release planner algorithm roster contains an empty key or algo")
            continue
        if key in algorithms and algorithms[key] != algo:
            blockers.append(f"full-release planner algorithm roster conflicts for {key!r}")
            continue
        algorithms[key] = algo

    expected_keys = set(planner_keys)
    if set(algorithms) != expected_keys:
        missing = sorted(expected_keys - set(algorithms))
        unexpected = sorted(set(algorithms) - expected_keys)
        if missing:
            blockers.append(f"full-release planner algorithm roster is missing {missing!r}")
        if unexpected:
            blockers.append(f"full-release planner algorithm roster has unexpected {unexpected!r}")
    return algorithms, blockers


def _full_release_nested_config_path(
    raw_path: Any,
    *,
    config_anchor: Path,
    source_repository_root: Path,
    label: str,
) -> Path:
    """Resolve a candidate's nested config path inside the trusted source checkout.

    The producer resolver accepts absolute paths and otherwise falls back to the process
    working directory.  That is appropriate for an interactive run, but it is not a safe
    publication provenance boundary: a validator must not resolve a nested candidate config
    from an unrelated checkout or an external file.  Prefer the producer's config-relative
    lookup when it exists, then apply the producer's repository-relative convention explicitly
    against ``source_repository_root``.

    Returns:
        The resolved regular file path inside ``source_repository_root``.

    Raises:
        ValueError: If the declaration is malformed, escapes the source root, is symlinked, or
            does not name a regular file.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{label} must be a non-empty string")
    source_root = source_repository_root.resolve()
    raw = Path(raw_path.strip())
    if raw.is_absolute():
        lexical_candidate = raw.absolute()
        if _path_has_symlink_component(lexical_candidate):
            raise ValueError(f"{label} contains a symlink component")
        candidate = raw.resolve()
    else:
        lexical_anchored = config_anchor / raw
        if _path_has_symlink_component(lexical_anchored):
            raise ValueError(f"{label} contains a symlink component")
        anchored = lexical_anchored.resolve()
        if anchored.is_file():
            candidate = anchored
        else:
            lexical_source = source_root / raw
            if _path_has_symlink_component(lexical_source):
                raise ValueError(f"{label} contains a symlink component")
            candidate = lexical_source.resolve()
    try:
        candidate.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"{label} is outside trusted source repository root") from exc
    if _path_has_symlink_component(candidate):
        raise ValueError(f"{label} contains a symlink component")
    if not candidate.is_file():
        raise ValueError(f"{label} does not name a regular file")
    return candidate


def _full_release_candidate_config(  # noqa: C901
    *,
    planner_spec: Any,
    source_repository_root: Path,
    allowed_scenario_ids: set[str] | None = None,
) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    """Load and structurally validate one arm's policy-search candidate config.

    Returns:
        The trusted top-level config path, normalized manifest, and an optional blocker.  All
        nested ``base_config_path`` declarations are rewritten to source-root absolute paths so
        the producer resolver cannot fall back to the validator process working directory.
    """
    raw_config_path = getattr(planner_spec, "algo_config_path", None)
    if raw_config_path is None:
        return None, None, None
    source_root = source_repository_root.resolve()
    try:
        config_path = _source_repository_path(raw_config_path, source_root)
        if _path_has_symlink_component(config_path) or not config_path.is_file():
            raise ValueError("candidate algorithm config is not a regular trusted source file")
        manifest = _parse_algo_config(str(config_path))
        if not _is_policy_search_candidate_manifest(manifest):
            return config_path, manifest, None

        normalized = deepcopy(manifest)
        if "base_config_path" in normalized:
            normalized["base_config_path"] = str(
                _full_release_nested_config_path(
                    normalized["base_config_path"],
                    config_anchor=config_path.parent,
                    source_repository_root=source_root,
                    label="candidate base_config_path",
                )
            )

        raw_overrides = normalized.get("scenario_algo_overrides")
        if raw_overrides is not None:
            if not isinstance(raw_overrides, Mapping):
                raise TypeError("scenario_algo_overrides must be a mapping")
            validated_overrides: dict[str, Any] = {}
            for raw_scenario_id, raw_override in raw_overrides.items():
                if not isinstance(raw_scenario_id, str) or not raw_scenario_id.strip():
                    raise TypeError("scenario_algo_overrides keys must be non-empty strings")
                scenario_id = raw_scenario_id.strip()
                if allowed_scenario_ids is not None and scenario_id not in allowed_scenario_ids:
                    raise ValueError(
                        "scenario_algo_overrides key is not in the canonical campaign matrix: "
                        f"{scenario_id!r}"
                    )
                if not isinstance(raw_override, Mapping):
                    raise TypeError(
                        f"scenario_algo_overrides entries must be mappings ({scenario_id!r})"
                    )
                override = deepcopy(dict(raw_override))
                if "algo" in override and (
                    not isinstance(override["algo"], str) or not override["algo"].strip()
                ):
                    raise TypeError(
                        f"scenario_algo_overrides[{scenario_id!r}].algo must be a non-empty string"
                    )
                if "params" in override and not isinstance(override["params"], Mapping):
                    raise TypeError(
                        f"scenario_algo_overrides[{scenario_id!r}].params must be a mapping"
                    )
                if "base_config_path" in override:
                    override["base_config_path"] = str(
                        _full_release_nested_config_path(
                            override["base_config_path"],
                            config_anchor=config_path.parent,
                            source_repository_root=source_root,
                            label=(f"scenario_algo_overrides[{scenario_id!r}].base_config_path"),
                        )
                    )
                validated_overrides[scenario_id] = override
            normalized["scenario_algo_overrides"] = validated_overrides
        return config_path, normalized, None
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        reason = str(exc)
        safe_reason = next(
            (
                marker
                for marker in (
                    "scenario_algo_overrides entries must be mappings",
                    "scenario_algo_overrides keys must be non-empty strings",
                    "scenario_algo_overrides key is not in the canonical campaign matrix",
                    "scenario_algo_overrides entries must be mappings",
                    "scenario_algo_overrides algo must be a non-empty string",
                    "scenario_algo_overrides params must be a mapping",
                    "outside trusted source repository root",
                    "contains a symlink component",
                    "does not name a regular file",
                )
                if marker in reason
            ),
            "invalid canonical policy configuration",
        )
        return None, None, f"canonical planner policy config validation failed: {safe_reason}"


def _stress_effective_branch_coverage(  # noqa: C901, PLR0912, PLR0915
    *,
    manifest: Any,
    campaign_config: Any,
    source_repository_root: Path | None = None,
) -> dict[str, Any]:
    """Check every pinned stress candidate override against manifest witnesses.

    The five executed stress scenarios intentionally remain bounded.  Branches are instead
    enumerated over the complete set of pinned hybrid candidate configs, so an override for an
    omitted scenario cannot silently evade diagnostic admission.  Witnesses carry the same
    candidate path/hash binding as the manifest pin and are checked by the shared branch helper.

    Returns:
        JSON-safe branch inventory, witness inventory, and deterministic blockers.
    """
    blockers: list[str] = []
    branch_records: list[dict[str, str]] = []
    branches: list[dict[str, str]] = []
    trusted_source_root = Path(source_repository_root or get_repository_root()).resolve()
    planners = {
        str(getattr(planner, "key", "")).strip(): planner
        for planner in getattr(campaign_config, "planners", ())
    }
    pins = tuple(getattr(manifest, "stress_smoke_hybrid_config_pins", ()))
    pins_by_arm = {
        str(getattr(pin, "planner_key", "")).strip(): pin
        for pin in pins
        if str(getattr(pin, "planner_key", "")).strip()
    }

    for pin in pins:
        arm = str(getattr(pin, "planner_key", "")).strip()
        planner_spec = planners.get(arm)
        if planner_spec is None:
            _append_blocker(blockers, f"effective branch config names unknown planner arm {arm!r}")
            continue
        config_path, candidate_config, config_error = _full_release_candidate_config(
            planner_spec=planner_spec,
            source_repository_root=trusted_source_root,
            allowed_scenario_ids=None,
        )
        if config_error is not None:
            _append_blocker(blockers, f"effective branch config for {arm!r}: {config_error}")
            continue
        if config_path is None or not isinstance(candidate_config, Mapping):
            _append_blocker(blockers, f"effective branch config for {arm!r} is unavailable")
            continue
        if config_path.resolve() != pin.path.resolve():
            _append_blocker(
                blockers,
                f"effective branch config path does not match its pin for {arm!r}",
            )
        try:
            observed_config_sha256 = sha256_file(config_path)
        except OSError:
            observed_config_sha256 = ""
            _append_blocker(blockers, f"effective branch config cannot be read for {arm!r}")
        if observed_config_sha256 != str(getattr(pin, "sha256", "")).strip().lower():
            _append_blocker(
                blockers, f"effective branch config hash does not match its pin for {arm!r}"
            )

        candidate_payload = deepcopy(dict(candidate_config))
        candidate_payload["id"] = arm
        candidate_payload.setdefault("algo", getattr(planner_spec, "algo", ""))
        raw_overrides = candidate_payload.get("scenario_algo_overrides")
        if raw_overrides is not None and not isinstance(raw_overrides, Mapping):
            _append_blocker(blockers, f"scenario_algo_overrides must be a mapping for {arm!r}")
            continue
        if isinstance(raw_overrides, Mapping):
            for scenario_id, override in raw_overrides.items():
                if not isinstance(scenario_id, str) or not scenario_id.strip():
                    _append_blocker(
                        blockers, f"scenario_algo_overrides has an invalid key for {arm!r}"
                    )
                elif not isinstance(override, Mapping):
                    _append_blocker(
                        blockers,
                        f"scenario_algo_overrides[{scenario_id!r}] must be a mapping for {arm!r}",
                    )
        enumerated = enumerate_effective_branches(candidate_payload, allowed_scenario_ids=None)
        for branch in enumerated:
            branch_copy = {
                **branch,
                "config_path": str(pin.path),
                "config_sha256": str(getattr(pin, "sha256", "")).strip().lower(),
            }
            branches.append(branch)
            branch_records.append(branch_copy)

    witness_records: list[dict[str, Any]] = []
    for index, witness in enumerate(getattr(manifest, "stress_smoke_branch_witnesses", ())):
        if isinstance(witness, StressSmokeBranchWitness):
            record = {
                "kind": witness.kind,
                "arm": witness.arm,
                "scenario": witness.scenario,
                "algorithm": witness.algorithm,
                "branch_key": witness.branch_key,
                "config_path": str(witness.config_path),
                "config_sha256": witness.config_sha256,
            }
        elif isinstance(witness, Mapping):
            record = dict(witness)
        else:
            _append_blocker(blockers, f"branch witness {index} is not a mapping")
            continue
        witness_records.append(record)
        arm = str(record.get("arm", "")).strip()
        pin = pins_by_arm.get(arm)
        if pin is None:
            _append_blocker(blockers, f"branch witness {index} names unknown planner arm {arm!r}")
            continue
        config_path = str(record.get("config_path", "")).strip()
        if config_path != str(pin.path):
            _append_blocker(
                blockers, f"branch witness {index} config path does not match its arm pin"
            )
        config_sha256 = str(record.get("config_sha256", "")).strip().lower()
        if config_sha256 != str(getattr(pin, "sha256", "")).strip().lower():
            _append_blocker(
                blockers, f"branch witness {index} config hash does not match its arm pin"
            )

    expected_keys = {
        (branch["arm"], branch["scenario"], branch["algorithm"]) for branch in branches
    }
    for witness in witness_records:
        arm = str(witness.get("arm", "")).strip()
        scenario = str(witness.get("scenario", "")).strip()
        algorithm = str(witness.get("algorithm", "")).strip()
        if arm and scenario and algorithm and (arm, scenario, algorithm) not in expected_keys:
            _append_blocker(
                blockers,
                "diagnostic witness names an unconfigured effective branch "
                f"{arm}|{scenario}|{algorithm}",
            )
    for blocker in check_witness_coverage(branches, witness_records):
        _append_blocker(blockers, blocker)

    return {
        "branches": branch_records,
        "witnesses": witness_records,
        "blockers": blockers,
    }


def _full_release_row_contract_blockers(
    row: Mapping[str, Any],
    *,
    prefix: str,
    expected_algo: str,
) -> list[str]:
    """Bind one publication row's algorithm and provenance to its containing arm.

    Returns:
        Blockers for missing or conflicting arm/provenance aliases.
    """
    blockers: list[str] = []
    if not _status_is(row.get("status"), _STRESS_ROW_TERMINAL_STATUSES):
        blockers.append(f"{prefix}.status must be a recognized scientific terminal outcome")
    normalized_algo = expected_algo.strip().lower()
    metadata_algorithm = "ppo" if normalized_algo == "guarded_ppo" else normalized_algo
    blockers.extend(
        f"{prefix}: {blocker}"
        for blocker in _alias_blockers(
            _alias_values(row, (("algo", row.get("algo", _MISSING)),)),
            label="planner algorithm",
            expected=normalized_algo,
        )
    )

    metadata = row.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        blockers.append(f"{prefix}: algorithm_metadata is missing")
    else:
        blockers.extend(
            f"{prefix}: {blocker}"
            for blocker in _alias_blockers(
                _alias_values(
                    metadata,
                    (("algorithm_metadata.algorithm", metadata.get("algorithm", _MISSING)),),
                ),
                label="algorithm metadata algorithm",
                expected=metadata_algorithm,
            )
        )
        blockers.extend(
            f"{prefix}: {blocker}"
            for blocker in _alias_blockers(
                _alias_values(
                    metadata,
                    (
                        (
                            "algorithm_metadata.canonical_algorithm",
                            metadata.get("canonical_algorithm", _MISSING),
                        ),
                    ),
                ),
                label="canonical algorithm",
                expected=normalized_algo,
            )
        )
        planner_contract = metadata.get("planner_contract")
        planner_id = (
            planner_contract.get("planner_id", _MISSING)
            if isinstance(planner_contract, Mapping)
            else _MISSING
        )
        blockers.extend(
            f"{prefix}: {blocker}"
            for blocker in _alias_blockers(
                _alias_values(
                    metadata,
                    (("algorithm_metadata.planner_contract.planner_id", planner_id),),
                ),
                label="planner contract identity",
                expected=normalized_algo,
            )
        )

    result_provenance = row.get("result_provenance")
    if not isinstance(result_provenance, Mapping):
        blockers.append(f"{prefix}: result_provenance is missing")
    else:
        source_commit = str(result_provenance.get("repo_commit", "")).strip().lower()
        blockers.extend(
            f"{prefix}: {blocker}"
            for blocker in _alias_blockers(
                _alias_values(
                    row,
                    (
                        ("git_hash", row.get("git_hash", _MISSING)),
                        ("repo_commit", row.get("repo_commit", _MISSING)),
                        (
                            "provenance.git_hash",
                            _nested_value(row, "provenance", "git_hash"),
                        ),
                        (
                            "result_provenance.repo_commit",
                            result_provenance.get("repo_commit", _MISSING),
                        ),
                        (
                            "event_ledger.software_commit",
                            _nested_value(row, "event_ledger", "software_commit"),
                        ),
                    ),
                ),
                label="source",
                expected=source_commit or None,
            )
        )
        config_hash = str(result_provenance.get("config_hash", "")).strip().lower()
        blockers.extend(
            f"{prefix}: {blocker}"
            for blocker in _alias_blockers(
                _alias_values(
                    row,
                    (
                        ("config_hash", row.get("config_hash", _MISSING)),
                        (
                            "provenance.config_hash",
                            _nested_value(row, "provenance", "config_hash"),
                        ),
                        (
                            "result_provenance.config_hash",
                            result_provenance.get("config_hash", _MISSING),
                        ),
                    ),
                ),
                label="row config",
                expected=config_hash or None,
            )
        )
        row_scenario = str(row.get("scenario_id", "")).strip()
        provenance_scenario = str(result_provenance.get("scenario_id", "")).strip()
        if not row_scenario or not provenance_scenario:
            blockers.append(f"{prefix}: scenario provenance is missing")
        elif row_scenario != provenance_scenario:
            blockers.append(f"{prefix}: scenario provenance does not match the row")
        row_seed = _strict_int(row.get("seed"))
        provenance_seed = _strict_int(result_provenance.get("seed"))
        if row_seed is None or provenance_seed is None:
            blockers.append(f"{prefix}: seed provenance is missing or invalid")
        elif row_seed != provenance_seed:
            blockers.append(f"{prefix}: seed provenance does not match the row")
        if not str(result_provenance.get("repo_commit", "")).strip():
            blockers.append(f"{prefix}: result_provenance.repo_commit is missing")
    return blockers


def _full_release_effective_algorithm(
    *,
    planner_spec: Any,
    base_algorithm: str,
    scenario: Mapping[str, Any],
    source_repository_root: Path | None = None,
    allowed_scenario_ids: set[str] | None = None,
) -> tuple[str | None, str | None]:
    """Resolve one row's expected algorithm through the producer's policy resolver.

    Policy-search arms may intentionally replace their base algorithm for a named scenario.
    The release gate must therefore compare row metadata with the effective runtime algorithm,
    while retaining the arm's base algorithm for sidecar/config binding.  Reusing the producer
    resolver keeps the two paths deterministic and makes malformed candidate manifests fail
    closed instead of silently falling back to the base algorithm.

    Returns:
        ``(algorithm, None)`` for a valid resolution, or ``(None, blocker)`` when the canonical
        planner/config/scenario binding cannot be resolved.
    """
    scenario_id = _producer_scenario_id(dict(scenario))
    if not scenario_id or scenario_id == "unknown":
        return None, "canonical scenario is missing a producer-resolvable identifier"
    try:
        config_path, candidate_config, config_error = _full_release_candidate_config(
            planner_spec=planner_spec,
            source_repository_root=Path(source_repository_root or get_repository_root()).resolve(),
            allowed_scenario_ids=allowed_scenario_ids,
        )
        if config_error is not None:
            return None, f"{config_error} for scenario {scenario_id!r}"
        effective_algorithm, _effective_config = _resolve_policy_search_candidate_runtime(
            default_algo=str(base_algorithm).strip(),
            algo_config_path=str(config_path) if config_path is not None else None,
            scenario=dict(scenario),
            algo_config=candidate_config,
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return (
            None,
            f"canonical planner policy resolution failed for scenario {scenario_id!r}",
        )
    normalized = str(effective_algorithm or "").strip().lower()
    if not normalized:
        return (
            None,
            f"canonical planner policy resolution returned an empty algorithm for {scenario_id!r}",
        )
    return normalized, None


def validate_diagnostic_stress_smoke_acceptance(  # noqa: C901, PLR0912, PLR0915
    campaign_root: Path,
    *,
    manifest: Any,
    campaign_config: Any,
    expected_source_commit: str,
) -> dict[str, Any]:
    """Validate the bounded 14-arm stress smoke without granting release success.

    Returns:
        Diagnostic-only admission report; it never grants release success.
    """
    if getattr(manifest, "release_kind", None) != "benchmark-stress-smoke":
        return {
            "schema_version": "benchmark-stress-smoke-acceptance.v1",
            "status": "not_applicable",
            "diagnostic_success": False,
            "claim_boundary": "not a diagnostic stress-smoke manifest",
            "blockers": [],
        }

    blockers: list[str] = []
    summary, summary_error = _read_campaign_summary(campaign_root.resolve())
    if summary_error is not None or summary is None:
        return {
            "schema_version": "benchmark-stress-smoke-acceptance.v1",
            "status": "invalid",
            "diagnostic_success": False,
            "claim_boundary": "diagnostic execution evidence only; no benchmark or ranking claim",
            "blockers": [summary_error or "campaign summary cannot be read"],
        }

    scenario_ids, seeds, axis_blockers = _resolve_expected_matrix_axes(manifest, campaign_config)
    blockers.extend(axis_blockers)
    resolved_scenarios: list[dict[str, Any]] = []
    if campaign_config is not None:
        try:
            resolved_scenarios = _load_campaign_scenarios(campaign_config)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            _append_blocker(blockers, f"resolved campaign matrix cannot be loaded: {exc}")
    planner_keys = tuple(str(key).strip() for key in getattr(manifest, "planner_keys", ()))
    kinematics = tuple(
        str(value).strip() for value in getattr(manifest, "expected_kinematics_matrix", ())
    )
    expected_kinematics = kinematics[0] if len(kinematics) == 1 else "differential_drive"
    expected_arms = {(key, expected_kinematics) for key in planner_keys}
    expected_ids = {
        (planner_key, expected_kinematics, scenario_id, seed)
        for planner_key in planner_keys
        for scenario_id in scenario_ids
        for seed in seeds
    }
    expected_per_arm = len(scenario_ids) * len(seeds)
    expected_cells = len(expected_ids)
    if len(planner_keys) != STRESS_SMOKE_EXPECTED_PLANNER_ARMS:
        _append_blocker(blockers, "diagnostic stress smoke must declare exactly 14 planner arms")
    if len(set(planner_keys)) != len(planner_keys):
        _append_blocker(blockers, "diagnostic stress smoke planner roster contains duplicates")
    if tuple(scenario_ids) != STRESS_SMOKE_EXPECTED_SCENARIO_IDS:
        _append_blocker(blockers, "diagnostic stress smoke must resolve the fixed five scenarios")
    if tuple(seeds) != (STRESS_SMOKE_EXPECTED_SEED,):
        _append_blocker(blockers, "diagnostic stress smoke must resolve exactly seed 116")
    if expected_cells != STRESS_SMOKE_EXPECTED_EPISODE_CELLS:
        _append_blocker(blockers, "diagnostic stress smoke must resolve exactly 70 episode cells")
    if (
        _strict_int(getattr(campaign_config, "horizon", None))
        != STRESS_SMOKE_EXPECTED_HORIZON_STEPS
    ):
        _append_blocker(blockers, "campaign config horizon must be 600")
    try:
        campaign_dt = float(getattr(campaign_config, "dt", float("nan")))
    except (TypeError, ValueError):
        campaign_dt = float("nan")
    if campaign_dt != STRESS_SMOKE_EXPECTED_DT:
        _append_blocker(blockers, "campaign config dt must be 0.1")
    if tuple(
        str(value).strip().lower() for value in getattr(campaign_config, "kinematics_matrix", ())
    ) != (STRESS_SMOKE_EXPECTED_KINEMATICS,):
        _append_blocker(blockers, "campaign config kinematics must be differential_drive only")

    if campaign_config is None:
        metadata_blockers, campaign_manifest_payload = (
            ["campaign config is unavailable for metadata binding"],
            None,
        )
    else:
        metadata_blockers, campaign_manifest_payload = _stress_metadata_contract_blockers(
            campaign_root,
            manifest=manifest,
            campaign_config=campaign_config,
            expected_source_commit=str(expected_source_commit).strip().lower(),
            scenarios=resolved_scenarios,
            resolved_seeds=seeds,
        )
    for blocker in metadata_blockers:
        _append_blocker(blockers, f"campaign metadata: {blocker}")
    # Campaign metadata uses the camera-ready 12-character structural hash;
    # result-provenance sidecars use the 16-character config hash over the
    # resolved scenario payload.  Keep these identities distinct.
    effective_scenarios = _result_provenance_scenarios(
        campaign_config,
        resolved_scenarios,
        kinematics=STRESS_SMOKE_EXPECTED_KINEMATICS,
    )
    expected_scenario_identity = _config_hash(effective_scenarios)

    for marker_path, marker in _status_markers(summary, "campaign_summary"):
        _append_blocker(blockers, f"forbidden {marker_path}={marker}")
    campaign = summary.get("campaign")
    if not isinstance(campaign, Mapping):
        _append_blocker(blockers, "campaign summary campaign block is missing")
    elif campaign.get("benchmark_success") is not True:
        _append_blocker(blockers, "campaign summary benchmark_success must be true")
    if isinstance(campaign, Mapping):
        if campaign.get("status") not in {"benchmark_success", "ok"}:
            _append_blocker(blockers, "campaign summary status must report benchmark success")
        if campaign.get("evidence_status") != "valid":
            _append_blocker(blockers, "campaign summary evidence_status must be valid")
        if campaign.get("campaign_execution_status") != "completed":
            _append_blocker(
                blockers, "campaign summary campaign_execution_status must be completed"
            )
    if isinstance(campaign, Mapping):
        for marker_path, marker in _status_markers(campaign, "campaign_summary.campaign"):
            _append_blocker(blockers, f"forbidden {marker_path}={marker}")

    runs = summary.get("runs")
    if not isinstance(runs, list):
        runs = []
        _append_blocker(blockers, "campaign summary must contain a runs list")
    planner_rows = summary.get("planner_rows")
    if not isinstance(planner_rows, list):
        planner_rows = []
        _append_blocker(blockers, "campaign summary must contain a planner_rows list")
    observed_arms: set[tuple[str, str]] = set()
    observed_planner_row_arms: set[tuple[str, str]] = set()
    identities: set[tuple[str, str, str, int]] = set()
    duplicate_identities: set[tuple[str, str, str, int]] = set()
    observed_rows = 0
    planner_specs = {
        str(getattr(planner, "key", "")).strip(): planner
        for planner in getattr(campaign_config, "planners", ())
    }
    branch_coverage = _stress_effective_branch_coverage(
        manifest=manifest,
        campaign_config=campaign_config,
    )
    for blocker in branch_coverage["blockers"]:
        _append_blocker(blockers, f"effective algorithm branches: {blocker}")
    for run_index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            _append_blocker(blockers, f"runs[{run_index}] must be an object")
            continue
        planner = run.get("planner")
        planner = planner if isinstance(planner, Mapping) else {}
        arm = (
            str(planner.get("key", "")).strip(),
            str(planner.get("kinematics", "")).strip(),
        )
        observed_arms.add(arm)
        if not _status_is(run.get("status"), _STRESS_RUN_SUCCESS_STATUSES):
            _append_blocker(blockers, f"runs[{run_index}] status is not ok")
        planner_spec = planner_specs.get(arm[0])
        expected_algo = (
            str(getattr(planner_spec, "algo", planner.get("algo", arm[0]))).strip().lower()
        )
        if arm not in expected_arms:
            _append_blocker(blockers, f"runs[{run_index}] declares an unexpected planner arm")
        if str(planner.get("algo", expected_algo)).strip().lower() != expected_algo:
            _append_blocker(blockers, f"runs[{run_index}] planner algorithm conflicts with its key")
        if arm[1].strip().lower() != STRESS_SMOKE_EXPECTED_KINEMATICS:
            _append_blocker(blockers, f"runs[{run_index}] kinematics must be differential_drive")
        if _strict_int(planner.get("horizon")) != STRESS_SMOKE_EXPECTED_HORIZON_STEPS:
            _append_blocker(blockers, f"runs[{run_index}] planner horizon is not 600")
        try:
            planner_dt = float(planner.get("dt"))
        except (TypeError, ValueError):
            planner_dt = float("nan")
        if planner_dt != STRESS_SMOKE_EXPECTED_DT:
            _append_blocker(blockers, f"runs[{run_index}] planner dt is not 0.1")
        if "benchmark_success" in run and not _explicit_success(run.get("benchmark_success")):
            _append_blocker(
                blockers, f"runs[{run_index}] benchmark_success must be explicitly true"
            )
        for marker_path, marker in _status_markers(
            run,
            f"runs[{run_index}]",
            expected_algorithm=expected_algo,
        ):
            _append_blocker(blockers, f"forbidden {marker_path}={marker}")
        run_summary = run.get("summary")
        if not isinstance(run_summary, Mapping):
            _append_blocker(blockers, f"runs[{run_index}] summary aggregate is missing")
        else:
            if not _status_is(run_summary.get("status"), _STRESS_RUN_SUCCESS_STATUSES):
                _append_blocker(blockers, f"runs[{run_index}].summary status is not ok")
            if "benchmark_success" in run_summary and not _explicit_success(
                run_summary.get("benchmark_success")
            ):
                _append_blocker(
                    blockers,
                    f"runs[{run_index}].summary benchmark_success must be explicitly true",
                )
            for field, expected in (
                ("total_jobs", expected_per_arm),
                ("written", expected_per_arm),
                ("successful_jobs", expected_per_arm),
                ("failed_jobs", 0),
                ("skipped_jobs", 0),
            ):
                if _strict_int(run_summary.get(field)) != expected:
                    _append_blocker(
                        blockers,
                        f"runs[{run_index}].summary.{field} must be {expected}",
                    )
            failures = run_summary.get("failures")
            if not isinstance(failures, list) or failures:
                _append_blocker(blockers, f"runs[{run_index}].summary failures must be empty")
            for marker_path, marker in _status_markers(
                run_summary,
                f"runs[{run_index}].summary",
                expected_algorithm=expected_algo,
            ):
                _append_blocker(blockers, f"forbidden {marker_path}={marker}")
        raw_path = str(run.get("episodes_path", "")).strip()
        if not raw_path:
            _append_blocker(blockers, f"runs[{run_index}] is missing episodes_path")
            continue
        try:
            episodes_path = _resolve_stress_artifact_path(
                campaign_root,
                raw_path,
                arm=arm,
            )
        except (OSError, ValueError) as exc:
            _append_blocker(blockers, f"runs[{run_index}] episodes_path rejected: {exc}")
            continue
        summary_path = str(run.get("summary_path", "")).strip()
        if not summary_path:
            _append_blocker(blockers, f"runs[{run_index}] is missing summary_path")
        else:
            try:
                _resolve_stress_artifact_path(
                    campaign_root,
                    summary_path,
                    arm=arm,
                    kind="summary",
                )
            except (OSError, ValueError) as exc:
                _append_blocker(blockers, f"runs[{run_index}] summary_path rejected: {exc}")
        rows, error = _read_episode_rows(episodes_path)
        if error:
            _append_blocker(blockers, error)
            continue
        algorithm_config_path = getattr(planner_spec, "algo_config_path", None)
        blockers.extend(
            f"{blocker}"
            for blocker in _stress_episode_provenance_blockers(
                episodes_path,
                campaign_root=campaign_root,
                planner_key=arm[0],
                expected_algo=expected_algo,
                expected_source_commit=str(expected_source_commit).strip().lower(),
                expected_scenario_path=Path(manifest.scenario_matrix_path),
                expected_scenario_hash=str(getattr(manifest, "scenario_matrix_sha256", ""))
                .strip()
                .lower(),
                expected_scenario_identity=expected_scenario_identity,
                expected_algo_config_path=(
                    Path(algorithm_config_path) if algorithm_config_path is not None else None
                ),
                expected_rows=rows,
            )
        )
        observed_rows += len(rows)
        if len(rows) != expected_per_arm:
            _append_blocker(
                blockers,
                f"runs[{run_index}] contains {len(rows)} rows; expected {expected_per_arm}",
            )
        for row_index, row in enumerate(rows):
            for marker_path, marker in _status_markers(
                row,
                f"runs[{run_index}].rows[{row_index}]",
                expected_algorithm=expected_algo,
            ):
                _append_blocker(blockers, f"forbidden {marker_path}={marker}")
            blockers.extend(
                f"{blocker}"
                for blocker in _stress_row_contract_blockers(
                    row,
                    prefix=f"runs[{run_index}].rows[{row_index}]",
                    planner_key=arm[0],
                    expected_algo=expected_algo,
                    expected_kinematics=expected_kinematics,
                    expected_source_commit=str(expected_source_commit).strip().lower(),
                )
            )
            scenario_id = str(row.get("scenario_id", "")).strip()
            seed = _strict_int(row.get("seed"))
            if not scenario_id or seed is None:
                _append_blocker(
                    blockers, f"runs[{run_index}].rows[{row_index}] has invalid identity"
                )
                continue
            identity = (arm[0], arm[1], scenario_id, seed)
            if identity in identities:
                duplicate_identities.add(identity)
            identities.add(identity)

    for planner_row_index, planner_row in enumerate(planner_rows):
        if not isinstance(planner_row, Mapping):
            _append_blocker(blockers, f"planner_rows[{planner_row_index}] must be an object")
            continue
        planner_row_arm = (
            str(planner_row.get("planner_key", "")).strip(),
            str(planner_row.get("kinematics", "")).strip(),
        )
        observed_planner_row_arms.add(planner_row_arm)
        if not _status_is(planner_row.get("status"), _STRESS_RUN_SUCCESS_STATUSES):
            _append_blocker(blockers, f"planner_rows[{planner_row_index}] status is not ok")
        if not _explicit_success(planner_row.get("benchmark_success")):
            _append_blocker(
                blockers,
                f"planner_rows[{planner_row_index}] benchmark_success must be explicitly true",
            )
        if _strict_int(planner_row.get("episodes")) != expected_per_arm:
            _append_blocker(
                blockers,
                f"planner_rows[{planner_row_index}] episode count is not {expected_per_arm}",
            )
        if _strict_int(planner_row.get("failed_jobs", 0)) != 0:
            _append_blocker(blockers, f"planner_rows[{planner_row_index}] failed_jobs must be 0")
        planner_spec = planner_specs.get(planner_row_arm[0])
        planner_row_expected_algo = (
            str(getattr(planner_spec, "algo", planner_row.get("algo", planner_row_arm[0])))
            .strip()
            .lower()
        )
        for marker_path, marker in _status_markers(
            planner_row,
            f"planner_rows[{planner_row_index}]",
            expected_algorithm=planner_row_expected_algo,
        ):
            _append_blocker(blockers, f"forbidden {marker_path}={marker}")

    if len(planner_rows) != len(expected_arms):
        _append_blocker(
            blockers,
            f"stress smoke must contain exactly {len(expected_arms)} planner aggregate rows",
        )
    if observed_planner_row_arms != expected_arms:
        _append_blocker(
            blockers,
            "stress-smoke planner aggregate rows do not match the manifest roster",
        )

    if duplicate_identities:
        _append_blocker(blockers, f"duplicate episode identities: {sorted(duplicate_identities)!r}")
    if observed_arms != expected_arms:
        _append_blocker(blockers, "successful stress-smoke arms do not match the manifest roster")
    if len(runs) != len(expected_arms):
        _append_blocker(
            blockers, f"stress smoke must contain exactly {len(expected_arms)} planner runs"
        )
    if observed_rows != expected_cells:
        _append_blocker(blockers, f"observed stress-smoke rows must be {expected_cells}")
    missing = expected_ids - identities
    unexpected = identities - expected_ids
    if missing or unexpected:
        _append_blocker(
            blockers, "stress-smoke episode identities do not match the exact manifest product"
        )

    source_report = validate_diagnostic_stress_smoke_source_provenance(
        campaign_root,
        expected_source_commit=expected_source_commit,
    )
    if source_report["status"] != "valid":
        for blocker in source_report["blockers"]:
            _append_blocker(blockers, f"source provenance: {blocker}")

    persisted_integrity = summary.get("campaign_integrity")
    if not isinstance(persisted_integrity, Mapping):
        _append_blocker(blockers, "campaign_integrity block is missing")
    elif persisted_integrity.get("status") != "valid":
        _append_blocker(blockers, "campaign_integrity.status must be valid")
    if campaign_manifest_payload is not None and resolved_scenarios:
        try:
            integrity_entries = [
                entry
                for entry in runs
                if isinstance(entry, Mapping) and entry.get("status") == "ok"
            ]
            recomputed_integrity = validate_campaign_integrity(
                integrity_entries,
                scenarios=resolved_scenarios,
                resolved_seeds=seeds,
                campaign_root=campaign_root.resolve(),
                campaign_manifest=campaign_manifest_payload,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            _append_blocker(blockers, f"campaign integrity recomputation failed: {exc}")
        else:
            if recomputed_integrity.get("status") != "valid":
                _append_blocker(blockers, "recomputed campaign integrity is not valid")
            if isinstance(persisted_integrity, Mapping):
                for field in (
                    "expected_identity_count",
                    "checked_arm_count",
                    "blockers",
                ):
                    if persisted_integrity.get(field) != recomputed_integrity.get(field):
                        _append_blocker(
                            blockers,
                            f"persisted campaign_integrity.{field} does not match recomputation",
                        )

    status = "valid" if not blockers else "invalid"
    return {
        "schema_version": "benchmark-stress-smoke-acceptance.v1",
        "status": status,
        "diagnostic_success": status == "valid",
        "expected_planner_arms": len(expected_arms),
        "expected_episode_cells": expected_cells,
        "observed_episode_rows": observed_rows,
        "unique_episode_identities": len(identities),
        "effective_algorithm_branches": branch_coverage["branches"],
        "diagnostic_branch_witnesses": branch_coverage["witnesses"],
        "source_provenance": source_report,
        "claim_boundary": "diagnostic execution evidence only; no benchmark, ranking, or SNQI claim",
        "blockers": blockers,
    }


def validate_full_benchmark_release_acceptance(  # noqa: C901, PLR0912, PLR0915
    campaign_root: Path,
    *,
    manifest: Any,
    campaign_config: Any | None = None,
    source_repository_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the publication-grade S30/H600 campaign contract.

    v0.1 manifests (including the one-scenario runtime smoke) return
    ``not_applicable``.  They remain useful diagnostic execution checks but
    cannot satisfy or accidentally inherit the full-release gate.

    Returns:
        JSON-safe acceptance report.  ``status=valid`` is the only status that
        permits publication of a v0.2 benchmark-data release.
    """
    if getattr(manifest, "schema_version", None) != FULL_RELEASE_SCHEMA_VERSION:
        return {
            "schema_version": FULL_RELEASE_ACCEPTANCE_SCHEMA_VERSION,
            "status": "not_applicable",
            "benchmark_success": False,
            "claim_boundary": "bounded runtime smoke or legacy release manifest; no full-release claim",
            "blockers": [],
        }

    blockers: list[str] = []
    trusted_source_root = Path(source_repository_root or get_repository_root()).resolve()
    if not trusted_source_root.is_dir():
        _append_blocker(
            blockers,
            "trusted source repository root is not a directory",
        )
    expected_cells = getattr(manifest, "expected_episode_cells", None)
    expected_horizon = getattr(manifest, "expected_horizon_steps", None)
    planner_keys = tuple(str(key) for key in getattr(manifest, "planner_keys", ()))
    kinematics = tuple(str(value) for value in getattr(manifest, "expected_kinematics_matrix", ()))
    if len(planner_keys) != FULL_RELEASE_EXPECTED_PLANNER_ARMS:
        _append_blocker(
            blockers,
            f"manifest must declare exactly {FULL_RELEASE_EXPECTED_PLANNER_ARMS} planner arms",
        )
    if expected_cells != FULL_RELEASE_EXPECTED_EPISODE_CELLS:
        _append_blocker(
            blockers,
            f"manifest expected_episode_cells must be {FULL_RELEASE_EXPECTED_EPISODE_CELLS}",
        )
    if expected_horizon != FULL_RELEASE_EXPECTED_HORIZON_STEPS:
        _append_blocker(
            blockers,
            f"manifest expected_horizon_steps must be {FULL_RELEASE_EXPECTED_HORIZON_STEPS}",
        )
    if kinematics != (FULL_RELEASE_KINEMATICS,):
        _append_blocker(
            blockers,
            f"manifest kinematics must be [{FULL_RELEASE_KINEMATICS!r}]",
        )
    resolved_campaign_config, config_resolution_blockers = _full_release_campaign_config(
        manifest, campaign_config, trusted_source_root
    )
    for blocker in config_resolution_blockers:
        _append_blocker(blockers, blocker)
    expected_algorithms, algorithm_roster_blockers = _full_release_algorithm_roster(
        manifest, resolved_campaign_config, planner_keys, trusted_source_root
    )
    for blocker in algorithm_roster_blockers:
        _append_blocker(blockers, blocker)
    scenario_ids, resolved_seeds, axis_blockers = _resolve_expected_matrix_axes(
        manifest, resolved_campaign_config, trusted_source_root
    )
    for blocker in axis_blockers:
        _append_blocker(blockers, blocker)
    if len(scenario_ids) != 48:
        _append_blocker(blockers, "manifest-resolved campaign must contain exactly 48 scenarios")
    if len(resolved_seeds) != 30:
        _append_blocker(blockers, "manifest-resolved campaign must contain exactly 30 seeds")
    if len(planner_keys) * len(scenario_ids) * len(resolved_seeds) != expected_cells:
        _append_blocker(
            blockers, "manifest-resolved planner/scenario/seed product mismatches 20,160"
        )

    planner_specs: dict[str, Any] = {}
    resolved_scenarios: list[dict[str, Any]] = []
    scenarios_by_producer_id: dict[str, dict[str, Any]] = {}
    expected_scenario_identity = ""
    expected_scenario_path: Path | None = None
    expected_scenario_hash = ""
    if resolved_campaign_config is not None:
        planner_specs = {
            str(getattr(planner, "key", "")).strip(): planner
            for planner in getattr(resolved_campaign_config, "planners", ())
        }
        try:
            resolved_scenarios = _load_campaign_scenarios(resolved_campaign_config)
        except (OSError, ValueError, KeyError, TypeError):
            _append_blocker(blockers, "campaign scenarios cannot be resolved for provenance")
        else:
            for scenario in resolved_scenarios:
                if not isinstance(scenario, Mapping):
                    _append_blocker(
                        blockers,
                        "campaign scenarios must be mappings for effective algorithm resolution",
                    )
                    continue
                producer_scenario_id = _producer_scenario_id(dict(scenario))
                if producer_scenario_id in scenarios_by_producer_id:
                    _append_blocker(
                        blockers,
                        f"campaign scenarios contain duplicate producer identifier {producer_scenario_id!r}",
                    )
                    continue
                scenarios_by_producer_id[producer_scenario_id] = dict(scenario)
            effective_scenarios = _result_provenance_scenarios(
                resolved_campaign_config,
                resolved_scenarios,
                kinematics=FULL_RELEASE_KINEMATICS,
            )
            expected_scenario_identity = _config_hash(effective_scenarios)
        try:
            expected_scenario_path = _source_repository_path(
                resolved_campaign_config.scenario_matrix_path, trusted_source_root
            )
        except ValueError:
            _append_blocker(blockers, "canonical scenario matrix path is not trusted")
        try:
            if expected_scenario_path is not None:
                expected_scenario_hash = sha256_file(expected_scenario_path)
        except OSError:
            _append_blocker(blockers, "campaign scenario matrix cannot be hashed")

    summary, summary_error = _read_campaign_summary(campaign_root.resolve())
    if summary_error:
        _append_blocker(blockers, summary_error)
        summary = {}
    campaign = summary.get("campaign") if isinstance(summary, dict) else None
    campaign = campaign if isinstance(campaign, Mapping) else {}
    for field, expected in (
        ("status", "benchmark_success"),
        ("evidence_status", "valid"),
        ("campaign_execution_status", "completed"),
    ):
        actual = campaign.get(field)
        if field == "status":
            if actual not in {"benchmark_success", "ok"}:
                _append_blocker(blockers, f"campaign.{field} must report benchmark success")
        elif actual != expected:
            _append_blocker(blockers, f"campaign.{field} must be {expected!r}")
    if campaign.get("benchmark_success") is not True:
        _append_blocker(blockers, "campaign.benchmark_success must be true")

    integrity = summary.get("campaign_integrity") if isinstance(summary, dict) else None
    if not isinstance(integrity, Mapping) or integrity.get("status") != "valid":
        _append_blocker(blockers, "campaign_integrity.status must be valid")

    row_status_summary = campaign.get("row_status_summary")
    if not isinstance(row_status_summary, Mapping):
        _append_blocker(blockers, "campaign.row_status_summary is missing")
    else:
        expected_summary = {
            "successful_evidence_rows": FULL_RELEASE_EXPECTED_PLANNER_ARMS,
            "accepted_unavailable_rows": 0,
            "unexpected_failed_rows": 0,
            "fallback_or_degraded_rows": 0,
        }
        for field, expected in expected_summary.items():
            actual = _strict_int(row_status_summary.get(field, -1))
            if actual != expected:
                _append_blocker(
                    blockers,
                    f"campaign.row_status_summary.{field} must be {expected}",
                )

    expected_arms = {(key, FULL_RELEASE_KINEMATICS) for key in planner_keys}
    expected_identities = {
        (planner_key, FULL_RELEASE_KINEMATICS, scenario_id, seed)
        for planner_key in planner_keys
        for scenario_id in scenario_ids
        for seed in resolved_seeds
    }
    runs = summary.get("runs") if isinstance(summary, dict) else None
    planner_rows = summary.get("planner_rows") if isinstance(summary, dict) else None
    if not isinstance(runs, list):
        runs = []
        _append_blocker(blockers, "campaign summary must contain a runs list")
    if not isinstance(planner_rows, list):
        planner_rows = []
        _append_blocker(blockers, "campaign summary must contain a planner_rows list")

    observed_arms: set[tuple[str, str]] = set()
    duplicate_arms: set[tuple[str, str]] = set()
    planner_row_arms: set[tuple[str, str]] = set()
    duplicate_planner_row_arms: set[tuple[str, str]] = set()
    forbidden_status_counts: Counter[str] = Counter()
    source_commits: set[str] = set()
    identities: set[tuple[str, str, str, int]] = set()
    observed_episode_rows = 0
    expected_per_arm = (
        FULL_RELEASE_EXPECTED_EPISODE_CELLS // len(expected_arms) if expected_arms else 0
    )
    expected_source = str(campaign.get("git_hash", "")).strip().lower()
    effective_algorithm_cache: dict[tuple[str, str], tuple[str | None, str | None]] = {}
    allowed_scenario_ids = set(scenarios_by_producer_id)

    for index, entry in enumerate(runs):
        if not isinstance(entry, Mapping):
            _append_blocker(blockers, f"runs[{index}] must be an object")
            continue
        planner = entry.get("planner")
        planner = planner if isinstance(planner, Mapping) else {}
        arm = (str(planner.get("key", "")).strip(), str(planner.get("kinematics", "")).strip())
        expected_algo = expected_algorithms.get(arm[0], "")
        if arm in observed_arms:
            duplicate_arms.add(arm)
        observed_arms.add(arm)
        expected_algo = expected_algorithms.get(arm[0], "")
        if str(entry.get("status", "")).strip().lower() != "ok":
            _append_blocker(blockers, f"runs[{index}] status is not ok")
        for marker_path, marker in _status_markers(
            entry, f"runs[{index}]", expected_algorithm=expected_algo
        ):
            forbidden_status_counts[marker] += 1
            _append_blocker(blockers, f"forbidden {marker_path}={marker}")
        entry_summary = entry.get("summary")
        if isinstance(entry_summary, Mapping):
            for marker_path, marker in _status_markers(
                entry_summary,
                f"runs[{index}].summary",
                expected_algorithm=expected_algo,
            ):
                forbidden_status_counts[marker] += 1
                _append_blocker(blockers, f"forbidden {marker_path}={marker}")
            failed_jobs = _strict_int(entry_summary.get("failed_jobs"))
            if failed_jobs is not None and failed_jobs > 0:
                _append_blocker(blockers, f"runs[{index}] reports {failed_jobs} failed jobs")
            failures = entry_summary.get("failures")
            if isinstance(failures, list) and failures:
                _append_blocker(blockers, f"runs[{index}] reports non-empty failures")
        entry_horizon = planner.get("horizon")
        entry_horizon_value = _strict_int(entry_horizon)
        if entry_horizon_value != FULL_RELEASE_EXPECTED_HORIZON_STEPS:
            _append_blocker(blockers, f"runs[{index}] planner horizon is not 600")
        raw_path = str(entry.get("episodes_path", "")).strip()
        if not raw_path:
            _append_blocker(blockers, f"runs[{index}] is missing episodes_path")
            continue
        try:
            episodes_path = (
                _resolve_stress_artifact_path(campaign_root, raw_path, arm=arm)
                if resolved_campaign_config is not None
                else _resolve_integrity_artifact_path(campaign_root.resolve(), raw_path)
            )
        except (OSError, ValueError):
            _append_blocker(blockers, f"runs[{index}] episodes_path rejected as missing or unsafe")
            continue
        rows, error = _read_episode_rows(episodes_path)
        if error:
            _append_blocker(blockers, error)
            continue
        observed_episode_rows += len(rows)
        declared = entry.get("summary")
        declared = declared if isinstance(declared, Mapping) else {}
        declared_count = declared.get("episodes_total", declared.get("written"))
        if len(rows) != expected_per_arm:
            _append_blocker(
                blockers,
                f"runs[{index}] contains {len(rows)} rows; expected {expected_per_arm}",
            )
        if declared_count is None or _strict_int(declared_count) != len(rows):
            _append_blocker(blockers, f"runs[{index}] declared episode count mismatches artifact")
        if resolved_campaign_config is not None:
            planner_spec = planner_specs.get(arm[0])
            algo_config_path = getattr(planner_spec, "algo_config_path", None)
            if planner_spec is None:
                _append_blocker(blockers, f"runs[{index}] has no canonical planner specification")
            elif expected_scenario_path is None or not expected_scenario_identity:
                _append_blocker(
                    blockers,
                    f"runs[{index}] cannot validate arm-bound provenance without scenarios",
                )
            else:
                try:
                    expected_algo_config_path = (
                        _source_repository_path(algo_config_path, trusted_source_root)
                        if algo_config_path is not None
                        else None
                    )
                except ValueError:
                    expected_algo_config_path = None
                    _append_blocker(
                        blockers,
                        f"runs[{index}] canonical algorithm config path is not trusted",
                    )
                for blocker in _stress_episode_provenance_blockers(
                    episodes_path,
                    campaign_root=campaign_root,
                    planner_key=arm[0],
                    expected_algo=expected_algorithms.get(arm[0], ""),
                    expected_source_commit=expected_source,
                    expected_scenario_path=expected_scenario_path,
                    expected_scenario_hash=expected_scenario_hash,
                    expected_scenario_identity=expected_scenario_identity,
                    expected_algo_config_path=expected_algo_config_path,
                    source_repository_root=trusted_source_root,
                    expected_rows=rows,
                ):
                    _append_blocker(blockers, blocker)
        arm_identities: set[tuple[str, str, str, int]] = set()
        for row_index, row in enumerate(rows):
            row_scenario_id = str(row.get("scenario_id", "")).strip()
            row_expected_algo = expected_algo
            if resolved_campaign_config is not None:
                canonical_scenario = scenarios_by_producer_id.get(row_scenario_id)
                if canonical_scenario is None:
                    _append_blocker(
                        blockers,
                        f"runs[{index}].rows[{row_index}] scenario is not in the canonical campaign matrix",
                    )
                else:
                    cache_key = (arm[0], row_scenario_id)
                    if cache_key not in effective_algorithm_cache:
                        effective_algorithm_cache[cache_key] = _full_release_effective_algorithm(
                            planner_spec=planner_specs.get(arm[0]),
                            base_algorithm=expected_algo,
                            scenario=canonical_scenario,
                            source_repository_root=trusted_source_root,
                            allowed_scenario_ids=allowed_scenario_ids,
                        )
                    row_expected_algo, resolution_error = effective_algorithm_cache[cache_key]
                    if resolution_error is not None:
                        _append_blocker(
                            blockers,
                            f"runs[{index}].rows[{row_index}] {resolution_error}",
                        )
                        row_expected_algo = expected_algo
            for blocker in _full_release_row_contract_blockers(
                row,
                prefix=f"runs[{index}].rows[{row_index}]",
                expected_algo=row_expected_algo or expected_algo,
            ):
                _append_blocker(blockers, blocker)
            for marker_path, marker in _status_markers(
                row,
                f"runs[{index}].rows[{row_index}]",
                expected_algorithm=expected_algo,
            ):
                forbidden_status_counts[marker] += 1
                _append_blocker(blockers, f"forbidden {marker_path}={marker}")
            scenario_id = str(row.get("scenario_id", "")).strip()
            if not scenario_id:
                _append_blocker(blockers, f"runs[{index}].rows[{row_index}] missing scenario_id")
            seed = _strict_int(row.get("seed"))
            if seed is None:
                _append_blocker(blockers, f"runs[{index}].rows[{row_index}] missing integer seed")
                continue
            identity = (arm[0], arm[1], scenario_id, seed)
            if identity in arm_identities or identity in identities:
                _append_blocker(blockers, f"duplicate episode identity {identity!r}")
            arm_identities.add(identity)
            identities.add(identity)
            commit = _source_commit(row)
            if not _GIT_SHA_RE.fullmatch(commit):
                _append_blocker(
                    blockers,
                    f"runs[{index}].rows[{row_index}] source commit is not a 40-character SHA",
                )
            elif commit:
                source_commits.add(commit)
            horizon, present = _episode_horizon(row)
            if not present or horizon != FULL_RELEASE_EXPECTED_HORIZON_STEPS:
                _append_blocker(blockers, f"runs[{index}].rows[{row_index}] horizon is not 600")

    if duplicate_arms:
        _append_blocker(blockers, f"duplicate planner arms: {sorted(duplicate_arms)!r}")
    if observed_arms != expected_arms:
        _append_blocker(blockers, "successful planner arms do not match the manifest roster")
    if len(runs) != FULL_RELEASE_EXPECTED_PLANNER_ARMS:
        _append_blocker(blockers, "campaign must contain exactly 14 planner arm rows")

    for index, row in enumerate(planner_rows):
        if not isinstance(row, Mapping):
            _append_blocker(blockers, f"planner_rows[{index}] must be an object")
            continue
        arm = (str(row.get("planner_key", "")).strip(), str(row.get("kinematics", "")).strip())
        if arm in planner_row_arms:
            duplicate_planner_row_arms.add(arm)
        planner_row_arms.add(arm)
        expected_algo = expected_algorithms.get(arm[0], "")
        for marker_path, marker in _status_markers(
            row,
            f"planner_rows[{index}]",
            expected_algorithm=expected_algo,
        ):
            forbidden_status_counts[marker] += 1
            _append_blocker(blockers, f"forbidden {marker_path}={marker}")
        if str(row.get("status", "")).strip().lower() != "ok":
            _append_blocker(blockers, f"planner_rows[{index}] status is not ok")
        if _strict_int(row.get("episodes", -1)) != expected_per_arm:
            _append_blocker(
                blockers, f"planner_rows[{index}] episode count is not {expected_per_arm}"
            )
        if arm not in expected_arms:
            _append_blocker(blockers, f"planner_rows[{index}] is outside the manifest roster")
    if len(planner_rows) != FULL_RELEASE_EXPECTED_PLANNER_ARMS:
        _append_blocker(blockers, "campaign must contain exactly 14 planner aggregate rows")
    if duplicate_planner_row_arms:
        _append_blocker(
            blockers,
            f"duplicate planner aggregate rows: {sorted(duplicate_planner_row_arms)!r}",
        )
    if planner_row_arms != expected_arms:
        _append_blocker(blockers, "planner aggregate rows do not match the manifest roster")

    if not _GIT_SHA_RE.fullmatch(expected_source):
        _append_blocker(blockers, "campaign.git_hash must be an exact 40-character SHA")
    if expected_source and source_commits and source_commits != {expected_source}:
        _append_blocker(blockers, "episode source commits do not match campaign.git_hash")
    if len(source_commits) != 1:
        _append_blocker(blockers, "episode provenance must name exactly one source commit")
    if observed_episode_rows != FULL_RELEASE_EXPECTED_EPISODE_CELLS:
        _append_blocker(
            blockers,
            f"observed episode rows must be {FULL_RELEASE_EXPECTED_EPISODE_CELLS}",
        )
    if len(identities) != FULL_RELEASE_EXPECTED_EPISODE_CELLS:
        _append_blocker(
            blockers,
            f"unique episode identities must be {FULL_RELEASE_EXPECTED_EPISODE_CELLS}",
        )
    missing_identities = expected_identities - identities
    unexpected_identities = identities - expected_identities
    if missing_identities or unexpected_identities:
        _append_blocker(
            blockers,
            "episode identities do not match the exact manifest-resolved planner/scenario/seed product",
        )

    status = "valid" if not blockers else "invalid"
    return {
        "schema_version": FULL_RELEASE_ACCEPTANCE_SCHEMA_VERSION,
        "status": status,
        "benchmark_success": status == "valid",
        "expected_planner_arms": FULL_RELEASE_EXPECTED_PLANNER_ARMS,
        "successful_planner_arms": sum(
            1
            for entry in runs
            if isinstance(entry, Mapping) and str(entry.get("status", "")).strip().lower() == "ok"
        ),
        "expected_episode_cells": FULL_RELEASE_EXPECTED_EPISODE_CELLS,
        "expected_scenario_count": len(scenario_ids),
        "expected_seed_count": len(resolved_seeds),
        "observed_episode_rows": observed_episode_rows,
        "unique_episode_identities": len(identities),
        "missing_episode_identities": len(missing_identities),
        "unexpected_episode_identities": len(unexpected_identities),
        "source_commits": sorted(source_commits),
        "forbidden_status_counts": dict(sorted(forbidden_status_counts.items())),
        "blockers": blockers,
        "claim_boundary": (
            "Publication-grade benchmark evidence requires all 14 arms, all 20,160 unique "
            "manifest-resolved planner/scenario/seed identities, one source commit, and zero "
            "fallback/degraded/failed/unavailable rows."
        ),
    }


__all__ = [
    "FULL_RELEASE_ACCEPTANCE_SCHEMA_VERSION",
    "FULL_RELEASE_EXPECTED_EPISODE_CELLS",
    "FULL_RELEASE_EXPECTED_HORIZON_STEPS",
    "FULL_RELEASE_EXPECTED_PLANNER_ARMS",
    "validate_diagnostic_stress_smoke_acceptance",
    "validate_diagnostic_stress_smoke_source_provenance",
    "validate_full_benchmark_release_acceptance",
]
