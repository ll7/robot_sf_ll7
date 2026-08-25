"""Fail-closed admission of the canonical 14-arm release runtime smoke."""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.camera_ready._run_state import validate_campaign_integrity
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.checkpoint_staging_receipt import (
    CheckpointStagingReceiptError,
    validate_checkpoint_staging_receipt,
)
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.release_acceptance import _status_markers
from robot_sf.benchmark.release_protocol import BENCHMARK_PROTOCOL_VERSION
from robot_sf.benchmark.result_provenance import validate_result_provenance_manifest
from robot_sf.benchmark.utils import _config_hash

RUNTIME_SMOKE_RELEASE_ID = "paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2"
RUNTIME_SMOKE_MANIFEST = Path(
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml"
)
RUNTIME_SMOKE_CONFIG = Path(
    "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_runtime_smoke.yaml"
)
RUNTIME_SMOKE_HORIZON = 600
RUNTIME_SMOKE_KINEMATICS = "differential_drive"
RUNTIME_SMOKE_MAX_AGE_HOURS = 24.0
RUNTIME_SMOKE_SUITE_KEY = "francis2023"
RUNTIME_SMOKE_SCHEMA_PATH = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")
RUNTIME_SMOKE_PLANNER_KEYS = (
    "prediction_planner",
    "goal",
    "social_force",
    "orca",
    "ppo",
    "socnav_sampling",
    "sacadrl",
    "scenario_adaptive_hybrid_orca_v2_bottleneck_yield",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
    "guarded_ppo",
    "predictive_mppi",
    "risk_dwa",
)
_RUNTIME_SMOKE_CHECKPOINT_PLANNER_KEYS = frozenset(
    {"prediction_planner", "ppo", "sacadrl", "guarded_ppo", "predictive_mppi"}
)
_FORBIDDEN_RUNTIME_STATUSES = frozenset(
    {
        "degraded",
        "error",
        "excluded",
        "failed",
        "fallback",
        "not-available",
        "not_available",
        "not-applicable",
        "not_applicable",
        "not-run",
        "not_run",
        "partial-failure",
        "partial_failure",
        "partial",
        "placeholder",
        "skipped",
        "unavailable",
        "unknown",
    }
)
_RUNTIME_STATUS_FIELDS = frozenset(
    {
        "status",
        "row_status",
        "readiness_status",
        "availability_status",
        "evidence_status",
        "execution_status",
        "load_status",
        "checkpoint_status",
        "fallback_status",
        "execution_mode",
        "mode",
        "policy",
    }
)
_RUNTIME_FLAG_FIELDS = frozenset(
    {
        "fallback_triggered",
        "fallback_used",
        "fallback_active",
        "degraded",
        "fallback_or_degraded",
    }
)
_RUNTIME_SUCCESS_FLAG_FIELDS = frozenset(
    {
        "benchmark_success",
        "load_succeeded",
        "release_benchmark_success",
    }
)
_RUNTIME_SHALLOW_CONTAINERS = frozenset(
    {
        "summary",
        "benchmark_availability",
        "campaign_integrity",
        "row_status_summary",
        "fallback_policy",
        "availability",
    }
)
_RUNTIME_DEEP_CONTAINERS = frozenset(
    {
        "algorithm_metadata",
        "algorithm_metadata_contract",
        "preflight",
        "learned_policy_contract",
        "checkpoint_provenance",
        "planner_runtime",
        "foresight_prediction",
        "planner_kinematics",
        "learned_checkpoint_observation_contract",
        "execution",
        "runtime",
    }
)
_RUNTIME_DECLARATIVE_CONTAINERS = frozenset(
    {
        "config",
        "planner_contract",
        "safety_shield_contract",
    }
)


class RuntimeSmokeAdmissionError(ValueError):
    """Raised when runtime-smoke evidence cannot admit a full release run."""


def _read_object(path: Path, label: str) -> dict[str, Any]:
    """Read one JSON mapping with a bounded public error.

    Returns:
        Parsed JSON object.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeSmokeAdmissionError(f"{label} is missing or invalid") from exc
    if not isinstance(payload, dict):
        raise RuntimeSmokeAdmissionError(f"{label} is not a JSON object")
    return payload


def _read_yaml_object(path: Path, label: str) -> dict[str, Any]:
    """Read one YAML mapping with a bounded public error.

    Returns:
        Parsed YAML object.
    """
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeSmokeAdmissionError(f"{label} is missing or invalid") from exc
    if not isinstance(payload, dict):
        raise RuntimeSmokeAdmissionError(f"{label} is not a YAML object")
    return payload


def _require_equal(problems: list[str], actual: Any, expected: Any, label: str) -> None:
    """Append a stable mismatch description."""
    if actual != expected:
        problems.append(f"{label} mismatch")


def _validate_age(run_meta: dict[str, Any], *, max_age_hours: float) -> str:
    """Return the normalized smoke completion timestamp or reject stale evidence."""
    if (
        not isinstance(max_age_hours, (int, float))
        or isinstance(max_age_hours, bool)
        or not math.isfinite(max_age_hours)
        or max_age_hours <= 0
        or max_age_hours > RUNTIME_SMOKE_MAX_AGE_HOURS
    ):
        raise RuntimeSmokeAdmissionError(
            "runtime smoke maximum age must be finite, positive, and no more than 24 hours"
        )
    raw = str(run_meta.get("finished_at_utc", "")).strip()
    try:
        finished = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeSmokeAdmissionError("runtime smoke completion timestamp is invalid") from exc
    if finished.tzinfo is None:
        finished = finished.replace(tzinfo=UTC)
    age_hours = (datetime.now(UTC) - finished.astimezone(UTC)).total_seconds() / 3600
    if age_hours < 0 or age_hours > max_age_hours:
        raise RuntimeSmokeAdmissionError("runtime smoke result is stale or future-dated")
    return finished.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _strict_int(value: Any) -> int | None:
    """Return an integer without accepting booleans or fractional values."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def _source_commit(row: dict[str, Any]) -> str:
    """Resolve the exact source commit recorded by an episode row.

    Returns:
        Normalized commit token, or an empty string when absent.
    """
    provenance = row.get("result_provenance")
    if isinstance(provenance, dict) and provenance.get("repo_commit"):
        return str(provenance["repo_commit"]).strip().lower()
    if row.get("repo_commit"):
        return str(row["repo_commit"]).strip().lower()
    return str(row.get("git_hash", "")).strip().lower()


def _episode_horizon(row: dict[str, Any]) -> int | None:
    """Resolve the authoritative episode horizon.

    Returns:
        Parsed horizon, or ``None`` when absent or malformed.
    """
    if row.get("horizon") is not None:
        return _strict_int(row["horizon"])
    provenance = row.get("result_provenance")
    if isinstance(provenance, dict):
        settings = provenance.get("simulator_settings")
        if isinstance(settings, dict):
            return _strict_int(settings.get("horizon"))
    return None


def _read_episode_rows(path: Path) -> list[dict[str, Any]]:
    """Read a smoke episode JSONL artifact without accepting malformed rows.

    Returns:
        Parsed episode row objects.
    """
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                payload = json.loads(raw_line)
                if not isinstance(payload, dict):
                    raise RuntimeSmokeAdmissionError(
                        f"runtime smoke episode row {path}:{line_number} is not an object"
                    )
                rows.append(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeSmokeAdmissionError(
            "runtime smoke episode artifact is missing or invalid"
        ) from exc
    return rows


def _runtime_marker_value(key: str, value: Any) -> str | None:  # noqa: C901
    """Return a forbidden/malformed runtime marker value, or ``None`` when safe."""
    normalized = str(value).strip().lower().replace(" ", "_")
    if key == "fallback_safe" and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        return f"invalid_{type(value).__name__}"
    is_status = key in _RUNTIME_STATUS_FIELDS or key.endswith("_status")
    if is_status:
        if not isinstance(value, str):
            return f"invalid_{type(value).__name__}"
        if normalized in _FORBIDDEN_RUNTIME_STATUSES or normalized.startswith(
            "predictive_foresight_model_fallback"
        ):
            return normalized
    if "fallback" in key and key not in _RUNTIME_SHALLOW_CONTAINERS:
        if isinstance(value, bool):
            if value or key not in _RUNTIME_FLAG_FIELDS:
                return normalized
        elif isinstance(value, (int, float)):
            if not math.isfinite(value) or value != 0:
                return normalized
        else:
            return normalized or f"invalid_{type(value).__name__}"
    if key in _RUNTIME_FLAG_FIELDS:
        if not isinstance(value, bool):
            return f"invalid_{type(value).__name__}"
        if value:
            return "true"
    if key in _RUNTIME_SUCCESS_FLAG_FIELDS and value is not True:
        return normalized or f"invalid_{type(value).__name__}"
    return None


def _is_campaign_preflight_unknown(path: str, value: Any) -> bool:
    """Allow only the producer's explicit pre-execution checkpoint placeholders.

    Returns:
        Whether the field is a canonical campaign-manifest preflight placeholder.
    """
    match = re.fullmatch(
        r"campaign_manifest\.planners\[\d+\]\.checkpoint_provenance\."
        r"(status|load_succeeded|fallback_triggered)",
        path,
    )
    if match is None:
        return False
    field = match.group(1)
    if field in {"load_succeeded", "fallback_triggered"}:
        return value is None
    return str(value).strip().lower().replace(" ", "_") in {
        "not_applicable",
        "not_run",
    }


def _is_canonical_not_applicable_status(path: str, key: str, value: Any) -> bool:
    """Allow only exact non-learned status fields emitted by the smoke producer.

    Returns:
        Whether this is a canonical non-applicable producer status.
    """
    if key != "status" or str(value).strip().lower().replace(" ", "_") != "not_applicable":
        return False
    patterns = (
        r"runs\[\d+\]\.summary(?:_artifact)?\.preflight\.learned_policy_contract\.status",
        r"runs\[\d+\]\.summary(?:_artifact)?(?:\.preflight)?\.algorithm_metadata_contract\."
        r"learned_checkpoint_observation_contract\.status",
        r"runs\[\d+\]\.rows\[\d+\]\.algorithm_metadata\."
        r"learned_checkpoint_observation_contract\.status",
    )
    return any(re.fullmatch(pattern, path) is not None for pattern in patterns)


def _is_guarded_ppo_safe_shield_marker(
    path: str, value: Any, *, normalized_marker: bool = False
) -> bool:
    """Allow only the canonical Guarded PPO native safe-shield counter.

    ``_status_markers`` normalizes a detected integer counter to text before the
    runtime-smoke deep walk sees it. Accept both the original integer and that
    normalized representation, while keeping floats and malformed values closed.

    Returns:
        Whether ``path`` identifies the fixed Guarded PPO arm's native safe counter.
    """
    if re.fullmatch(
        r"runs\[11\]\.rows\[0\]\.algorithm_metadata\."
        r"(?:guard_stats|shield_stats\.decision_counts)\.fallback_safe",
        path,
    ) is None or isinstance(value, bool):
        return False
    if normalized_marker:
        if not isinstance(value, str) or not value.isdigit():
            return False
        numeric = int(value)
    else:
        if not isinstance(value, int):
            return False
        numeric = value
    return numeric >= 0


def _is_allowed_runtime_marker(path: str, key: str, value: Any, *, parent: dict[str, Any]) -> bool:
    """Return whether a non-boolean/status token is canonical for its exact report surface.

    Returns:
        Whether the marker is an explicitly allowed producer representation.
    """
    if _is_campaign_preflight_unknown(path, value):
        return True
    # Guarded PPO is the fixed arm at index 11 in the canonical roster. Its
    # Risk-DWA safety-shield intervention is part of the declared composite
    # planner, not a missing-policy or degraded-runtime fallback. Keep the
    # exception exact: best-effort/uncertainty fallbacks and every other arm
    # continue to fail closed.
    if _is_guarded_ppo_safe_shield_marker(path, value):
        return True
    if (
        path
        == "runs[11].rows[0].algorithm_metadata.shield_stats.last_decision."
        "fallback_controller_state"
        and isinstance(value, dict)
    ):
        return True
    if path.startswith("planner_rows[") and key == "benchmark_success":
        return str(value).strip().lower() == "true"
    if (
        re.fullmatch(
            r"runs\[\d+\]\.(?:"
            r"rows\[\d+\]\.algorithm_metadata\."
            r"|summary(?:_artifact)?\.(?:preflight\.)?algorithm_metadata_contract\."
            r")(?:(?:planner_runtime)\.)?foresight_prediction\.fallback_reason",
            path,
        )
        and (value is None or value == "")
        and parent.get("fallback_used") is False
    ):
        return True
    if (
        re.fullmatch(
            r"runs\[\d+\]\.(?:"
            r"rows\[\d+\]\.algorithm_metadata\.planner_runtime"
            r"|summary(?:_artifact)?\.(?:preflight\.)?"
            r"algorithm_metadata_contract\.planner_runtime"
            r")\.fallback_reason",
            path,
        )
        and (value is None or value == "")
        and parent.get("fallback_triggered") is False
    ):
        return True
    if (
        key == "learned_policy_contract_status"
        and re.fullmatch(r"planner_rows\[\d+\]\.learned_policy_contract_status", path)
        and str(value).strip().lower().replace(" ", "_") == "not_applicable"
    ):
        return True
    return _is_canonical_not_applicable_status(path, key, value)


def _contains_symlink_component(path: Path) -> bool:
    """Return whether any existing lexical component of ``path`` is a symlink."""
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _forbidden_status_markers(  # noqa: C901
    payload: Any, prefix: str
) -> list[tuple[str, str]]:
    """Find forbidden execution markers, including nested policy summaries.

    Returns:
        Path/value pairs for every forbidden marker.
    """
    markers = (
        [
            (path, value)
            for path, value in _status_markers(payload, prefix)
            if not _is_guarded_ppo_safe_shield_marker(path, value, normalized_marker=True)
        ]
        if isinstance(payload, dict)
        else []
    )
    seen = {(path, value) for path, value in markers}

    def _walk(value: Any, path: str, *, descend_all: bool = False) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                normalized_key = str(key).strip().lower()
                marker_value = (
                    None
                    if _is_allowed_runtime_marker(child_path, normalized_key, child, parent=value)
                    else _runtime_marker_value(normalized_key, child)
                )
                if marker_value is not None:
                    marker = (child_path, marker_value)
                    if marker not in seen:
                        markers.append(marker)
                        seen.add(marker)
                if normalized_key in _RUNTIME_DECLARATIVE_CONTAINERS:
                    continue
                if descend_all or normalized_key in _RUNTIME_DEEP_CONTAINERS:
                    _walk(child, child_path, descend_all=True)
                elif normalized_key in _RUNTIME_SHALLOW_CONTAINERS:
                    _walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                _walk(child, f"{path}[{index}]", descend_all=descend_all)

    _walk(payload, prefix)
    return markers


def _resolve_campaign_artifact(  # noqa: C901, PLR0912
    *,
    raw_path: Any,
    campaign_root: Path,
    repo_root: Path,
    expected_path: Path,
    label: str,
    expected_root: Path | None = None,
    relocation_root: Path | None = None,
) -> Path:
    """Resolve an artifact only when it is this campaign's expected file.

    ``_resolve_integrity_artifact_path`` intentionally permits a repository-root
    relocation fallback for general campaign-integrity reports.  Admission of a
    release smoke is stricter: a path from an older campaign must never be
    substituted merely because a same-shaped file exists at the repository root.

    Returns:
        The resolved expected artifact path.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RuntimeSmokeAdmissionError(f"{label} path is missing")
    trusted_root = (expected_root or campaign_root).resolve()
    lexical_expected = expected_path.absolute()
    current = lexical_expected
    while True:
        if current.is_symlink():
            raise RuntimeSmokeAdmissionError(f"{label} path contains a symlink")
        if current == trusted_root:
            break
        if current.parent == current or not current.is_relative_to(trusted_root):
            raise RuntimeSmokeAdmissionError(f"{label} expected path is outside its trusted root")
        current = current.parent
    expected = expected_path.resolve()
    if not expected.is_relative_to(trusted_root):
        raise RuntimeSmokeAdmissionError(f"{label} resolves outside its trusted root")
    raw = Path(raw_path)
    candidates = [raw] if raw.is_absolute() else [campaign_root / raw, repo_root / raw]
    for candidate in candidates:
        if _contains_symlink_component(candidate):
            continue
        resolved = candidate.resolve(strict=False)
        if resolved == expected and resolved.is_file():
            return resolved
    if raw.is_absolute() and relocation_root is not None:
        if _contains_symlink_component(raw):
            raise RuntimeSmokeAdmissionError(f"{label} path contains a symlink")
        lexical_relocation_root = relocation_root.absolute()
        try:
            relative = raw.relative_to(lexical_relocation_root)
        except ValueError:
            relative = None
        if relative is not None and trusted_root == campaign_root.resolve():
            if trusted_root / relative == expected and expected.is_file():
                return expected
        elif trusted_root == repo_root.resolve():
            try:
                campaign_relative = campaign_root.resolve().relative_to(trusted_root)
                repo_relative = expected.relative_to(trusted_root)
            except ValueError:
                repo_relative = None
                campaign_relative = None
            old_repo_root = lexical_relocation_root
            if campaign_relative is not None:
                for _part in campaign_relative.parts:
                    old_repo_root = old_repo_root.parent
            if (
                repo_relative is not None
                and raw == old_repo_root / repo_relative
                and expected.is_file()
            ):
                return expected
    raise RuntimeSmokeAdmissionError(f"{label} is not bound to this campaign")


def _read_campaign_object(path: Path, *, campaign_root: Path, label: str) -> dict[str, Any]:
    """Read one canonical campaign JSON file without following symlink components.

    Returns:
        Parsed JSON object.
    """
    resolved = _resolve_campaign_artifact(
        raw_path=str(path),
        campaign_root=campaign_root,
        repo_root=campaign_root,
        expected_path=path,
        label=label,
    )
    return _read_object(resolved, label)


def _canonical_repo_artifact(path: Path, *, repo_root: Path, label: str) -> Path:
    """Resolve one canonical tracked input without following symlink components.

    Returns:
        Exact regular file inside the release worktree.
    """
    return _resolve_campaign_artifact(
        raw_path=str(path),
        campaign_root=repo_root,
        repo_root=repo_root,
        expected_path=path,
        label=label,
        expected_root=repo_root,
    )


def _canonical_scenario_matrix_hash(
    scenario_path: Path, *, repo_root: Path, scenario_id: str, seed: int
) -> str:
    """Reproduce the scoped scenario digest written by the canonical campaign producer.

    Returns:
        Stable structural digest after map, seed, and kinematics normalization.
    """
    payload = _read_yaml_object(scenario_path, "canonical runtime smoke scenario")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 1 or not isinstance(scenarios[0], dict):
        raise RuntimeSmokeAdmissionError(
            "canonical runtime smoke must resolve exactly one scenario"
        )
    scenario = dict(scenarios[0])
    observed_id = str(scenario.get("name") or scenario.get("id") or "").strip()
    if observed_id != scenario_id:
        raise RuntimeSmokeAdmissionError("canonical runtime smoke scenario identifier mismatch")
    raw_map = scenario.get("map_file")
    if isinstance(raw_map, str) and raw_map.strip():
        map_path = Path(raw_map)
        if not map_path.is_absolute():
            map_path = scenario_path.parent / map_path
        resolved_map = map_path.resolve()
        if not resolved_map.is_relative_to(repo_root) or not resolved_map.is_file():
            raise RuntimeSmokeAdmissionError("canonical runtime smoke map path is invalid")
        scenario["map_file"] = resolved_map.relative_to(repo_root).as_posix()
    scenario["seeds"] = [seed]
    robot_config = (
        dict(scenario.get("robot_config")) if isinstance(scenario.get("robot_config"), dict) else {}
    )
    robot_config["type"] = RUNTIME_SMOKE_KINEMATICS
    scenario["robot_config"] = robot_config
    return _config_hash([scenario])


def _validate_episode_provenance_sidecar(  # noqa: C901, PLR0913, PLR0915
    *,
    episodes_path: Path,
    campaign_root: Path,
    repo_root: Path,
    planner_key: str,
    algorithm: str | None,
    algorithm_config: str | None,
    scenario_path: Path,
    expected_source_commit: str,
    scenario_id: str,
    seed: int,
    expected_scenario_matrix_hash: str,
    episode_id: str,
    row_config_hash: str,
    relocation_root: Path | None,
    problems: list[str],
) -> None:
    """Bind one raw episode artifact to its exact arm and tracked inputs."""
    sidecar_path = episodes_path.with_name(f"{episodes_path.name}.provenance.json")
    if sidecar_path.is_symlink() or not sidecar_path.is_file():
        problems.append(f"planner {planner_key} episode provenance sidecar is missing")
        return
    sidecar = _read_object(sidecar_path, f"planner {planner_key} episode provenance sidecar")
    try:
        validate_result_provenance_manifest(sidecar)
    except (AttributeError, TypeError, ValueError) as exc:
        problems.append(f"planner {planner_key} episode provenance schema rejected: {exc}")
    run = sidecar.get("run")
    run = run if isinstance(run, dict) else {}
    _require_equal(
        problems,
        str(run.get("repo_commit", "")).strip().lower(),
        expected_source_commit,
        f"planner {planner_key} sidecar source commit",
    )
    _require_equal(
        problems,
        run.get("protocol_version"),
        BENCHMARK_PROTOCOL_VERSION,
        f"planner {planner_key} sidecar protocol version",
    )
    identity = sidecar.get("campaign_identity")
    identity = identity if isinstance(identity, dict) else {}
    _require_equal(
        problems,
        identity.get("algorithm"),
        algorithm,
        f"planner {planner_key} sidecar algorithm",
    )
    _require_equal(
        problems,
        identity.get("suite_key"),
        RUNTIME_SMOKE_SUITE_KEY,
        f"planner {planner_key} sidecar suite",
    )
    _require_equal(
        problems,
        identity.get("scenario_matrix_hash"),
        expected_scenario_matrix_hash,
        f"planner {planner_key} sidecar scenario identity hash",
    )
    _require_equal(problems, _strict_int(identity.get("total_jobs")), 1, "sidecar total jobs")
    _require_equal(problems, _strict_int(identity.get("written")), 1, "sidecar written rows")

    sidecar_rows = sidecar.get("rows")
    if not isinstance(sidecar_rows, list) or len(sidecar_rows) != 1:
        problems.append(f"planner {planner_key} sidecar must bind exactly one episode row")
    else:
        sidecar_row = sidecar_rows[0] if isinstance(sidecar_rows[0], dict) else {}
        _require_equal(
            problems, sidecar_row.get("scenario_id"), scenario_id, "sidecar scenario identity"
        )
        _require_equal(problems, sidecar_row.get("episode_id"), episode_id, "sidecar episode id")
        _require_equal(problems, _strict_int(sidecar_row.get("seed")), seed, "sidecar seed")
        _require_equal(
            problems, _strict_int(sidecar_row.get("jsonl_line")), 0, "sidecar JSONL line"
        )
        _require_equal(
            problems,
            str(sidecar_row.get("repo_commit", "")).strip().lower(),
            expected_source_commit,
            "sidecar row source commit",
        )
        _require_equal(
            problems,
            str(sidecar_row.get("config_hash", "")).strip(),
            row_config_hash,
            f"planner {planner_key} sidecar row config",
        )
        try:
            _resolve_campaign_artifact(
                raw_path=sidecar_row.get("raw_artifact"),
                campaign_root=campaign_root,
                repo_root=repo_root,
                expected_path=episodes_path,
                label=f"planner {planner_key} sidecar raw artifact",
                relocation_root=relocation_root,
            )
        except RuntimeSmokeAdmissionError as exc:
            problems.append(str(exc))

    if algorithm_config is not None:
        expected_config = (repo_root / algorithm_config).resolve()
        inputs = sidecar.get("inputs")
        inputs = inputs if isinstance(inputs, dict) else {}
        config_input = inputs.get("algo_config")
        config_input = config_input if isinstance(config_input, dict) else {}
        try:
            _resolve_campaign_artifact(
                raw_path=config_input.get("path"),
                campaign_root=campaign_root,
                repo_root=repo_root,
                expected_path=expected_config,
                label=f"planner {planner_key} sidecar algorithm config",
                expected_root=repo_root,
                relocation_root=relocation_root,
            )
        except RuntimeSmokeAdmissionError as exc:
            problems.append(str(exc))
        if expected_config.is_file():
            _require_equal(
                problems,
                config_input.get("sha256"),
                sha256_file(expected_config),
                f"planner {planner_key} sidecar algorithm config hash",
            )
    inputs = sidecar.get("inputs")
    inputs = inputs if isinstance(inputs, dict) else {}
    scenario_input = inputs.get("scenario_matrix")
    scenario_input = scenario_input if isinstance(scenario_input, dict) else {}
    try:
        _resolve_campaign_artifact(
            raw_path=scenario_input.get("path"),
            campaign_root=campaign_root,
            repo_root=repo_root,
            expected_path=scenario_path,
            label=f"planner {planner_key} sidecar scenario matrix",
            expected_root=repo_root,
            relocation_root=relocation_root,
        )
    except RuntimeSmokeAdmissionError as exc:
        problems.append(str(exc))
    _require_equal(
        problems,
        scenario_input.get("sha256"),
        sha256_file(scenario_path),
        f"planner {planner_key} sidecar scenario matrix hash",
    )
    raw_artifacts = sidecar.get("raw_artifacts")
    raw_artifacts = raw_artifacts if isinstance(raw_artifacts, list) else []
    episode_artifacts = [
        entry
        for entry in raw_artifacts
        if isinstance(entry, dict) and entry.get("kind") == "episodes_jsonl"
    ]
    if len(episode_artifacts) != 1:
        problems.append(f"planner {planner_key} sidecar raw episode binding is missing")
    else:
        artifact = episode_artifacts[0]
        _require_equal(
            problems,
            artifact.get("artifact_status"),
            "available",
            f"planner {planner_key} sidecar raw artifact status",
        )
        try:
            _resolve_campaign_artifact(
                raw_path=artifact.get("path"),
                campaign_root=campaign_root,
                repo_root=repo_root,
                expected_path=episodes_path,
                label=f"planner {planner_key} sidecar raw artifact inventory",
                relocation_root=relocation_root,
            )
        except RuntimeSmokeAdmissionError as exc:
            problems.append(str(exc))
        _require_equal(
            problems,
            artifact.get("sha256"),
            sha256_file(episodes_path),
            f"planner {planner_key} sidecar raw artifact hash",
        )
    inputs = sidecar.get("inputs")
    inputs = inputs if isinstance(inputs, dict) else {}
    schema_input = inputs.get("schema_path")
    schema_input = schema_input if isinstance(schema_input, dict) else {}
    expected_schema = repo_root / RUNTIME_SMOKE_SCHEMA_PATH
    try:
        _resolve_campaign_artifact(
            raw_path=schema_input.get("path"),
            campaign_root=campaign_root,
            repo_root=repo_root,
            expected_path=expected_schema,
            label=f"planner {planner_key} sidecar episode schema",
            expected_root=repo_root,
            relocation_root=relocation_root,
        )
    except RuntimeSmokeAdmissionError as exc:
        problems.append(str(exc))
    if expected_schema.is_file():
        _require_equal(
            problems,
            schema_input.get("sha256"),
            sha256_file(expected_schema),
            f"planner {planner_key} sidecar episode schema hash",
        )
    completeness = sidecar.get("completeness")
    completeness = completeness if isinstance(completeness, dict) else {}
    _require_equal(
        problems,
        completeness.get("status"),
        "complete",
        f"planner {planner_key} sidecar completeness",
    )
    expected_config_hash = _config_hash(
        {
            "schema_path": schema_input.get("path"),
            "algo": algorithm,
            "algo_config_path": config_input.get("path") if algorithm_config is not None else None,
        }
    )
    _require_equal(
        problems,
        identity.get("config_hash"),
        expected_config_hash,
        f"planner {planner_key} sidecar campaign config hash",
    )
    _require_equal(
        problems,
        run.get("runner"),
        "map_runner.run_map_batch",
        f"planner {planner_key} sidecar runner",
    )


def _canonical_smoke_contract(  # noqa: C901
    *, repo_root: Path, expected_planner_keys: tuple[str, ...]
) -> tuple[Path, Path, Path, str, int, str, dict[str, str], dict[str, str | None]]:
    """Resolve the tracked smoke axes and planner algorithms from canonical inputs.

    Returns:
        Manifest path, config path, scenario path, scenario ID, seed,
        expected scoped-scenario hash, planner-to-algorithm mapping, and
        planner-to-algorithm-config mapping.
    """
    manifest_path = _canonical_repo_artifact(
        repo_root / RUNTIME_SMOKE_MANIFEST,
        repo_root=repo_root,
        label="canonical runtime smoke manifest",
    )
    config_path = _canonical_repo_artifact(
        repo_root / RUNTIME_SMOKE_CONFIG,
        repo_root=repo_root,
        label="canonical runtime smoke config",
    )
    manifest = _read_yaml_object(manifest_path, "canonical runtime smoke manifest")
    config = _read_yaml_object(config_path, "canonical runtime smoke config")
    if tuple(expected_planner_keys) != RUNTIME_SMOKE_PLANNER_KEYS:
        raise RuntimeSmokeAdmissionError("caller planner roster is not the canonical 14-arm roster")
    if manifest.get("release_id") != RUNTIME_SMOKE_RELEASE_ID:
        raise RuntimeSmokeAdmissionError("canonical runtime smoke release identity mismatch")
    if manifest.get("campaign_config_sha256") != sha256_file(config_path):
        raise RuntimeSmokeAdmissionError("canonical runtime smoke config pin mismatch")
    manifest_planners = manifest.get("planners")
    manifest_planners = manifest_planners if isinstance(manifest_planners, dict) else {}
    config_planners = config.get("planners")
    config_planners = config_planners if isinstance(config_planners, list) else []
    enabled = [
        entry for entry in config_planners if isinstance(entry, dict) and entry.get("enabled", True)
    ]
    config_keys = tuple(str(entry.get("key", "")).strip() for entry in enabled)
    manifest_keys = tuple(str(key).strip() for key in manifest_planners.get("keys", []))
    if manifest_keys != RUNTIME_SMOKE_PLANNER_KEYS or config_keys != RUNTIME_SMOKE_PLANNER_KEYS:
        raise RuntimeSmokeAdmissionError("canonical runtime smoke planner roster mismatch")
    if _strict_int(config.get("horizon")) != RUNTIME_SMOKE_HORIZON:
        raise RuntimeSmokeAdmissionError("canonical runtime smoke horizon mismatch")
    if tuple(config.get("kinematics_matrix") or ()) != (RUNTIME_SMOKE_KINEMATICS,):
        raise RuntimeSmokeAdmissionError("canonical runtime smoke kinematics mismatch")
    manifest_kinematics = manifest.get("kinematics")
    manifest_kinematics = manifest_kinematics if isinstance(manifest_kinematics, dict) else {}
    if tuple(manifest_kinematics.get("matrix") or ()) != (RUNTIME_SMOKE_KINEMATICS,):
        raise RuntimeSmokeAdmissionError("runtime smoke manifest kinematics mismatch")
    seed_policy = config.get("seed_policy")
    seed_policy = seed_policy if isinstance(seed_policy, dict) else {}
    seeds = seed_policy.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 1 or _strict_int(seeds[0]) is None:
        raise RuntimeSmokeAdmissionError("canonical runtime smoke seed contract mismatch")
    scenario = manifest.get("scenario")
    scenario = scenario if isinstance(scenario, dict) else {}
    scenario_rel = scenario.get("matrix_path")
    if not isinstance(scenario_rel, str) or not scenario_rel.strip():
        raise RuntimeSmokeAdmissionError("canonical runtime smoke scenario path is missing")
    scenario_path = _canonical_repo_artifact(
        manifest_path.parent / scenario_rel,
        repo_root=repo_root,
        label="canonical runtime smoke scenario",
    )
    if scenario.get("matrix_sha256") != sha256_file(scenario_path):
        raise RuntimeSmokeAdmissionError("canonical runtime smoke scenario pin mismatch")
    scenario_payload = _read_yaml_object(scenario_path, "canonical runtime smoke scenario")
    scenarios = scenario_payload.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 1 or not isinstance(scenarios[0], dict):
        raise RuntimeSmokeAdmissionError(
            "canonical runtime smoke must resolve exactly one scenario"
        )
    scenario_id = str(scenarios[0].get("name") or scenarios[0].get("id") or "").strip()
    if not scenario_id:
        raise RuntimeSmokeAdmissionError("canonical runtime smoke scenario identifier is missing")
    expected_scenario_matrix_hash = _canonical_scenario_matrix_hash(
        scenario_path,
        repo_root=repo_root,
        scenario_id=scenario_id,
        seed=int(seeds[0]),
    )
    algorithms = {str(entry["key"]): str(entry.get("algo", "")) for entry in enabled}
    algorithm_configs = {
        str(entry["key"]): (
            str(entry.get("algo_config") or entry.get("algo_config_path") or "").strip() or None
        )
        for entry in enabled
    }
    return (
        manifest_path,
        config_path,
        scenario_path,
        scenario_id,
        int(seeds[0]),
        expected_scenario_matrix_hash,
        algorithms,
        algorithm_configs,
    )


def _validate_campaign_metadata(  # noqa: PLR0913
    *,
    campaign_root: Path,
    expected_source_commit: str,
    campaign_id: Any,
    scenario_id: str,
    seed: int,
    algorithms: dict[str, str],
    algorithm_configs: dict[str, str | None],
    release: dict[str, Any],
    manifest_path: Path,
    config_path: Path,
    problems: list[str],
) -> None:
    """Bind campaign/manifest metadata to the canonical smoke identity."""
    campaign_manifest = _read_campaign_object(
        campaign_root / "campaign_manifest.json",
        campaign_root=campaign_root,
        label="runtime smoke campaign manifest",
    )
    run_manifest = _read_campaign_object(
        campaign_root / "manifest.json",
        campaign_root=campaign_root,
        label="runtime smoke run manifest",
    )
    for label, payload in (
        ("campaign manifest", campaign_manifest),
        ("runtime smoke run manifest", run_manifest),
    ):
        block = payload.get("benchmark_release")
        block = block if isinstance(block, dict) else {}
        for field, expected in (
            ("release_id", RUNTIME_SMOKE_RELEASE_ID),
            ("manifest_path", RUNTIME_SMOKE_MANIFEST.as_posix()),
            ("canonical_campaign_config", RUNTIME_SMOKE_CONFIG.as_posix()),
            ("manifest_sha256", sha256_file(manifest_path)),
            ("canonical_campaign_config_sha256", sha256_file(config_path)),
        ):
            _require_equal(problems, block.get(field), expected, f"{label} {field}")
    campaign_git = campaign_manifest.get("git")
    campaign_git = campaign_git if isinstance(campaign_git, dict) else {}
    _require_equal(
        problems,
        str(campaign_git.get("commit", "")).strip().lower(),
        expected_source_commit,
        "campaign manifest source commit",
    )
    _require_equal(
        problems, campaign_manifest.get("campaign_id"), campaign_id, "campaign manifest id"
    )
    _require_equal(
        problems,
        campaign_manifest.get("scenario_matrix"),
        "configs/scenarios/single/francis2023_blind_corner.yaml",
        "campaign manifest scenario path",
    )
    seed_policy = campaign_manifest.get("seed_policy")
    seed_policy = seed_policy if isinstance(seed_policy, dict) else {}
    resolved_seeds = seed_policy.get("resolved_seeds", seed_policy.get("seeds"))
    _require_equal(problems, tuple(resolved_seeds or ()), (seed,), "campaign manifest seeds")
    planner_entries = campaign_manifest.get("planners")
    if not isinstance(planner_entries, list):
        problems.append("campaign manifest planner roster is missing")
        return
    observed_planners: list[str] = []
    for index, planner in enumerate(planner_entries):
        if not isinstance(planner, dict):
            problems.append(f"campaign manifest planner {index} is not an object")
            continue
        key = str(planner.get("key", "")).strip()
        observed_planners.append(key)
        _require_equal(
            problems,
            planner.get("algo"),
            algorithms.get(key),
            f"campaign manifest planner {index} algorithm",
        )
        _require_equal(
            problems,
            planner.get("algo_config_path"),
            algorithm_configs.get(key),
            f"campaign manifest planner {index} config",
        )
        if key in _RUNTIME_SMOKE_CHECKPOINT_PLANNER_KEYS:
            _validate_loaded_checkpoint_provenance(
                planner.get("checkpoint_provenance"),
                label=f"campaign manifest planner {index} checkpoint",
                problems=problems,
            )
        markers = _forbidden_status_markers(planner, f"campaign_manifest.planners[{index}]")
        if markers:
            problems.append(
                "campaign manifest planner contains forbidden status marker: "
                + f"{markers[0][0]}={markers[0][1]}"
            )
    _require_equal(
        problems, tuple(observed_planners), RUNTIME_SMOKE_PLANNER_KEYS, "campaign manifest roster"
    )
    run_git = str(run_manifest.get("git_hash", "")).strip().lower()
    _require_equal(problems, run_git, expected_source_commit, "run manifest source commit")
    run_manifest_release = run_manifest.get("benchmark_release")
    if isinstance(run_manifest_release, dict):
        _require_equal(
            problems,
            run_manifest_release.get("release_id"),
            release.get("release_id"),
            "run manifest release identity",
        )


def _validate_loaded_checkpoint_record(
    record: Any,
    *,
    label: str,
    problems: list[str],
    require_hash: bool,
) -> None:
    """Require one explicit successful checkpoint runtime observation."""
    if not isinstance(record, dict):
        problems.append(f"{label} is missing")
        return
    _require_equal(problems, record.get("load_succeeded"), True, f"{label} load")
    _require_equal(problems, record.get("fallback_triggered"), False, f"{label} fallback")
    _require_equal(problems, record.get("load_status"), "loaded", f"{label} status")
    if not str(record.get("model_id") or "").strip():
        problems.append(f"{label} model id is missing")
    checkpoint_hash = str(record.get("checkpoint_sha256") or "").strip().lower()
    if require_hash and re.fullmatch(r"[0-9a-f]{64}", checkpoint_hash) is None:
        problems.append(f"{label} checkpoint hash is missing or invalid")


def _validate_loaded_checkpoint_provenance(
    provenance: Any,
    *,
    label: str,
    problems: list[str],
) -> None:
    """Require complete loaded reference and runtime evidence for one learned arm."""
    if not isinstance(provenance, dict):
        problems.append(f"{label} provenance is missing")
        return
    _require_equal(problems, provenance.get("status"), "loaded", f"{label} aggregate status")
    _require_equal(problems, provenance.get("load_succeeded"), True, f"{label} aggregate load")
    _require_equal(
        problems,
        provenance.get("fallback_triggered"),
        False,
        f"{label} aggregate fallback",
    )
    for field in ("references", "runtime"):
        records = provenance.get(field)
        if not isinstance(records, list) or not records:
            problems.append(f"{label} {field} are missing")
            continue
        for record_index, record in enumerate(records):
            _validate_loaded_checkpoint_record(
                record,
                label=f"{label} {field}[{record_index}]",
                problems=problems,
                require_hash=field == "references",
            )


def validate_runtime_smoke_result(  # noqa: C901, PLR0912, PLR0915
    result_path: Path,
    *,
    repo_root: Path,
    expected_source_commit: str,
    expected_planner_keys: tuple[str, ...],
    max_age_hours: float = 24.0,
) -> dict[str, Any]:
    """Validate a byte-addressable smoke result before a full v0.2 campaign.

    Returns:
        Sanitized admission metadata suitable for release provenance and launch packets.
    """
    resolved_repo = repo_root.resolve()
    expected_planner_keys = tuple(str(key).strip() for key in expected_planner_keys)
    expected_source_commit = str(expected_source_commit).strip().lower()
    lexical_result = result_path.absolute()
    resolved_result = _resolve_campaign_artifact(
        raw_path=str(lexical_result),
        campaign_root=lexical_result.parent.parent,
        repo_root=resolved_repo,
        expected_path=lexical_result,
        label="runtime smoke result",
        expected_root=resolved_repo,
    )
    if not resolved_result.is_relative_to(resolved_repo):
        raise RuntimeSmokeAdmissionError("runtime smoke result must be inside the release worktree")
    if resolved_result.name != "release_result.json" or resolved_result.parent.name != "release":
        raise RuntimeSmokeAdmissionError(
            "runtime smoke result is not the canonical release receipt"
        )
    result = _read_object(resolved_result, "runtime smoke result")
    campaign_root = resolved_result.parent.parent
    run_meta = _read_campaign_object(
        campaign_root / "run_meta.json",
        campaign_root=campaign_root,
        label="runtime smoke run metadata",
    )
    summary = _read_campaign_object(
        campaign_root / "reports" / "campaign_summary.json",
        campaign_root=campaign_root,
        label="runtime smoke campaign summary",
    )
    (
        manifest_path,
        config_path,
        scenario_path,
        scenario_id,
        seed,
        expected_scenario_matrix_hash,
        algorithms,
        algorithm_configs,
    ) = _canonical_smoke_contract(
        repo_root=resolved_repo,
        expected_planner_keys=expected_planner_keys,
    )

    problems: list[str] = []
    release = result.get("benchmark_release")
    release = release if isinstance(release, dict) else {}
    _require_equal(problems, release.get("release_id"), RUNTIME_SMOKE_RELEASE_ID, "release_id")
    _require_equal(
        problems,
        release.get("manifest_path"),
        RUNTIME_SMOKE_MANIFEST.as_posix(),
        "runtime smoke manifest path",
    )
    _require_equal(
        problems,
        release.get("canonical_campaign_config"),
        RUNTIME_SMOKE_CONFIG.as_posix(),
        "runtime smoke config path",
    )
    _require_equal(
        problems,
        release.get("manifest_sha256"),
        sha256_file(manifest_path),
        "runtime smoke manifest hash",
    )
    _require_equal(
        problems,
        release.get("canonical_campaign_config_sha256"),
        sha256_file(config_path),
        "runtime smoke config hash",
    )
    _validate_campaign_metadata(
        campaign_root=campaign_root,
        expected_source_commit=str(expected_source_commit).strip().lower(),
        campaign_id=result.get("campaign_id"),
        scenario_id=scenario_id,
        seed=seed,
        algorithms=algorithms,
        algorithm_configs=algorithm_configs,
        release=release,
        manifest_path=manifest_path,
        config_path=config_path,
        problems=problems,
    )

    repo = run_meta.get("repo")
    repo = repo if isinstance(repo, dict) else {}
    _require_equal(problems, repo.get("commit"), expected_source_commit, "source commit")
    _require_equal(problems, run_meta.get("campaign_id"), result.get("campaign_id"), "campaign id")
    resolved = result.get("resolved_manifest")
    resolved = resolved if isinstance(resolved, dict) else {}
    planners = resolved.get("planners")
    planners = planners if isinstance(planners, dict) else {}
    _require_equal(
        problems, tuple(planners.get("keys") or ()), expected_planner_keys, "planner roster"
    )
    for field, expected in (
        ("release_id", RUNTIME_SMOKE_RELEASE_ID),
        ("canonical_campaign_config", RUNTIME_SMOKE_CONFIG.as_posix()),
    ):
        if field in resolved:
            _require_equal(problems, resolved.get(field), expected, f"resolved manifest {field}")
    resolved_scenario = resolved.get("scenario")
    if isinstance(resolved_scenario, dict):
        _require_equal(
            problems,
            resolved_scenario.get("matrix_path"),
            "configs/scenarios/single/francis2023_blind_corner.yaml",
            "resolved manifest scenario path",
        )
    resolved_seed_policy = resolved.get("seed_policy")
    if isinstance(resolved_seed_policy, dict):
        _require_equal(
            problems,
            tuple(resolved_seed_policy.get("seeds") or ()),
            (seed,),
            "resolved manifest seeds",
        )
    resolved_kinematics = resolved.get("kinematics")
    if isinstance(resolved_kinematics, dict):
        _require_equal(
            problems,
            tuple(resolved_kinematics.get("matrix") or ()),
            (RUNTIME_SMOKE_KINEMATICS,),
            "resolved manifest kinematics",
        )

    # The result is untrusted summary material too.  These fields are the
    # campaign runner's canonical admission state and must agree with the raw
    # rows below; the legacy ``release_*`` aliases are checked later as well.
    for field, expected in (
        ("benchmark_success", True),
        ("status", "benchmark_success"),
        ("evidence_status", "valid"),
        ("campaign_execution_status", "completed"),
        ("exit_code", 0),
    ):
        _require_equal(problems, result.get(field), expected, f"result {field}")
    result_markers = _forbidden_status_markers(result, "result")
    if result_markers:
        problems.append(
            "runtime smoke result contains forbidden status markers: "
            + f"{result_markers[0][0]}={result_markers[0][1]}"
        )

    expected_campaign_root = campaign_root.resolve()
    relocation_root: Path | None = None
    raw_campaign_root = result.get("campaign_root")
    if raw_campaign_root is not None:
        raw_candidate = Path(str(raw_campaign_root))
        was_absolute = raw_candidate.is_absolute()
        candidate = raw_candidate if was_absolute else resolved_repo / raw_candidate
        if candidate.resolve(strict=False) != expected_campaign_root:
            if not was_absolute or candidate.name != expected_campaign_root.name:
                problems.append("result campaign root mismatch")
            else:
                relocation_root = candidate.absolute()
    for field, raw_path in (
        ("campaign_root", raw_campaign_root),
        ("summary_json", result.get("summary_json")),
    ):
        if raw_path is None:
            continue
        if field == "campaign_root":
            continue
        try:
            _resolve_campaign_artifact(
                raw_path=raw_path,
                campaign_root=expected_campaign_root,
                repo_root=resolved_repo,
                expected_path=expected_campaign_root / "reports" / "campaign_summary.json",
                label=f"result {field}",
                relocation_root=relocation_root,
            )
        except RuntimeSmokeAdmissionError as exc:
            problems.append(str(exc))

    checkpoint = result.get("checkpoint_staging_receipt")
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    checkpoint_rel = checkpoint.get("path")
    if not isinstance(checkpoint_rel, str) or not checkpoint_rel.strip():
        problems.append("checkpoint staging receipt path is missing")
    else:
        checkpoint_relative = Path(checkpoint_rel)
        lexical_checkpoint = resolved_repo / checkpoint_relative
        checkpoint_path = lexical_checkpoint.resolve()
        if (
            checkpoint_relative.is_absolute()
            or _contains_symlink_component(lexical_checkpoint)
            or not checkpoint_path.is_relative_to(resolved_repo)
            or not checkpoint_path.is_file()
        ):
            problems.append("checkpoint staging receipt path is invalid")
        else:
            _require_equal(
                problems,
                checkpoint.get("sha256"),
                sha256_file(checkpoint_path),
                "checkpoint staging receipt hash",
            )
            try:
                cfg = load_campaign_config(config_path)
                admitted_checkpoint = validate_checkpoint_staging_receipt(
                    cfg,
                    checkpoint_path,
                    campaign_config_path=config_path,
                    max_age_hours=max_age_hours,
                )
            except (OSError, ValueError, CheckpointStagingReceiptError) as exc:
                problems.append(f"checkpoint staging receipt rejected: {exc}")
            else:
                _require_equal(
                    problems,
                    admitted_checkpoint.get("submit_safe"),
                    True,
                    "checkpoint submit_safe",
                )
                _require_equal(
                    problems,
                    checkpoint.get("submit_safe"),
                    True,
                    "release result checkpoint submit_safe",
                )

    campaign = summary.get("campaign")
    campaign = campaign if isinstance(campaign, dict) else {}
    expected_rows = len(RUNTIME_SMOKE_PLANNER_KEYS)
    for field, expected in (
        ("campaign_id", result.get("campaign_id")),
        ("git_hash", expected_source_commit),
        ("total_runs", expected_rows),
        ("successful_runs", expected_rows),
        ("total_episodes", expected_rows),
        ("non_success_runs", 0),
        ("accepted_unavailable_runs", 0),
        ("unexpected_failed_runs", 0),
        ("benchmark_success", True),
        ("status", "benchmark_success"),
        ("campaign_execution_status", "completed"),
        ("evidence_status", "valid"),
        ("exit_code", 0),
    ):
        _require_equal(problems, campaign.get(field), expected, f"campaign {field}")
    campaign_markers = _forbidden_status_markers(campaign, "campaign")
    if campaign_markers:
        problems.append(
            "runtime smoke campaign contains forbidden status markers: "
            + f"{campaign_markers[0][0]}={campaign_markers[0][1]}"
        )
    summary_integrity = summary.get("campaign_integrity")
    if isinstance(summary_integrity, dict):
        _require_equal(
            problems,
            summary_integrity.get("status"),
            "valid",
            "summary campaign integrity",
        )
        _require_equal(
            problems,
            summary_integrity.get("checked_arm_count"),
            expected_rows,
            "summary checked arm count",
        )
        _require_equal(
            problems,
            summary_integrity.get("expected_identity_count"),
            1,
            "summary expected identity count",
        )
        _require_equal(
            problems,
            summary_integrity.get("blockers"),
            [],
            "summary campaign integrity blockers",
        )
    summary_release = summary.get("benchmark_release")
    if isinstance(summary_release, dict):
        _require_equal(
            problems,
            summary_release.get("release_id"),
            RUNTIME_SMOKE_RELEASE_ID,
            "summary release identity",
        )
        _require_equal(
            problems,
            summary_release.get("manifest_sha256"),
            sha256_file(manifest_path),
            "summary release manifest hash",
        )
        _require_equal(
            problems,
            summary_release.get("canonical_campaign_config_sha256"),
            sha256_file(config_path),
            "summary release config hash",
        )

    runs = summary.get("runs")
    planner_rows = summary.get("planner_rows")
    if not isinstance(runs, list):
        runs = []
        problems.append("runtime smoke summary runs are missing")
    if not isinstance(planner_rows, list):
        planner_rows = []
        problems.append("runtime smoke summary planner_rows are missing")
    observed_arms: list[str] = []
    observed_episode_identities: list[tuple[str, str, int]] = []
    observed_episode_paths: set[Path] = set()
    integrity_entries: list[dict[str, Any]] = []
    fallback_markers: list[str] = []
    for index, entry in enumerate(runs):
        if not isinstance(entry, dict):
            problems.append(f"runtime smoke run {index} is not an object")
            continue
        planner = entry.get("planner")
        planner = planner if isinstance(planner, dict) else {}
        planner_key = str(planner.get("key", "")).strip()
        observed_arms.append(planner_key)
        _require_equal(problems, entry.get("status"), "ok", f"run {index} status")
        _require_equal(
            problems,
            planner.get("kinematics"),
            RUNTIME_SMOKE_KINEMATICS,
            f"run {index} kinematics",
        )
        _require_equal(
            problems,
            _strict_int(planner.get("horizon")),
            RUNTIME_SMOKE_HORIZON,
            f"run {index} horizon",
        )
        _require_equal(
            problems, planner.get("algo"), algorithms.get(planner_key), f"run {index} algorithm"
        )
        fallback_markers.extend(
            f"{path}={value}" for path, value in _forbidden_status_markers(entry, f"runs[{index}]")
        )
        declared = entry.get("summary")
        declared = declared if isinstance(declared, dict) else {}
        fallback_markers.extend(
            f"{path}={value}"
            for path, value in _forbidden_status_markers(declared, f"runs[{index}].summary")
        )
        expected_arm_dir = (
            campaign_root / "runs" / f"{planner_key}__{RUNTIME_SMOKE_KINEMATICS}"
            if planner_key in RUNTIME_SMOKE_PLANNER_KEYS
            else campaign_root / "runs" / "__invalid_planner_arm__"
        )
        _require_equal(
            problems,
            planner.get("algo_config_path"),
            algorithm_configs.get(planner_key),
            f"run {index} algorithm config",
        )
        try:
            episodes_path = _resolve_campaign_artifact(
                raw_path=entry.get("episodes_path"),
                campaign_root=campaign_root,
                repo_root=resolved_repo,
                expected_path=expected_arm_dir / "episodes.jsonl",
                label=f"run {index} episode artifact",
                relocation_root=relocation_root,
            )
            rows = _read_episode_rows(episodes_path)
        except (OSError, ValueError, RuntimeSmokeAdmissionError) as exc:
            problems.append(f"run {index} episode artifact rejected: {exc}")
            continue
        try:
            summary_path = _resolve_campaign_artifact(
                raw_path=entry.get("summary_path"),
                campaign_root=campaign_root,
                repo_root=resolved_repo,
                expected_path=expected_arm_dir / "summary.json",
                label=f"run {index} summary artifact",
                relocation_root=relocation_root,
            )
            arm_summary = _read_object(summary_path, f"runtime smoke arm {index} summary")
        except (OSError, ValueError, RuntimeSmokeAdmissionError) as exc:
            problems.append(f"run {index} summary artifact rejected: {exc}")
            continue
        if episodes_path in observed_episode_paths:
            problems.append(f"run {index} reuses another planner arm episode artifact")
        observed_episode_paths.add(episodes_path)
        if len(rows) != 1:
            problems.append(f"run {index} must contain exactly one episode row")
        fallback_markers.extend(
            f"{path}={value}"
            for path, value in _forbidden_status_markers(
                arm_summary, f"runs[{index}].summary_artifact"
            )
        )
        _require_equal(
            problems, arm_summary.get("status"), "ok", f"run {index} summary artifact status"
        )
        declared_count = declared.get("episodes_total", declared.get("written"))
        _require_equal(problems, _strict_int(declared_count), len(rows), f"run {index} row count")
        _require_equal(
            problems, _strict_int(declared.get("failed_jobs")), 0, f"run {index} failures"
        )
        _require_equal(problems, declared.get("failures"), [], f"run {index} failure list")
        for field in ("episodes_total", "written", "failed_jobs", "failures"):
            if field in declared and field in arm_summary:
                _require_equal(
                    problems,
                    arm_summary.get(field),
                    declared.get(field),
                    f"run {index} summary {field}",
                )
        if "out_path" in declared:
            try:
                _resolve_campaign_artifact(
                    raw_path=declared["out_path"],
                    campaign_root=campaign_root,
                    repo_root=resolved_repo,
                    expected_path=episodes_path,
                    label=f"run {index} summary output",
                    relocation_root=relocation_root,
                )
            except RuntimeSmokeAdmissionError as exc:
                problems.append(str(exc))
        integrity_entry = dict(entry)
        integrity_entry["episodes_path"] = str(episodes_path)
        integrity_entries.append(integrity_entry)
        for row_index, row in enumerate(rows):
            fallback_markers.extend(
                f"{path}={value}"
                for path, value in _forbidden_status_markers(
                    row, f"runs[{index}].rows[{row_index}]"
                )
            )
            _require_equal(
                problems,
                _source_commit(row),
                expected_source_commit,
                f"run {index} episode source commit",
            )
            _require_equal(
                problems,
                _episode_horizon(row),
                RUNTIME_SMOKE_HORIZON,
                f"run {index} episode horizon",
            )
            _require_equal(problems, row.get("scenario_id"), scenario_id, f"run {index} scenario")
            _require_equal(problems, _strict_int(row.get("seed")), seed, f"run {index} seed")
            _require_equal(
                problems,
                row.get("algo"),
                algorithms.get(planner_key),
                f"run {index} episode algorithm identity",
            )
            if "kinematics" in row:
                _require_equal(
                    problems,
                    row.get("kinematics"),
                    RUNTIME_SMOKE_KINEMATICS,
                    f"run {index} episode kinematics",
                )
            if "planner_key" in row:
                _require_equal(
                    problems,
                    row.get("planner_key"),
                    planner_key,
                    f"run {index} episode planner identity",
                )
            provenance = row.get("result_provenance")
            provenance = provenance if isinstance(provenance, dict) else {}
            _require_equal(
                problems,
                _source_commit(provenance),
                expected_source_commit,
                f"run {index} provenance source commit",
            )
            _require_equal(
                problems,
                provenance.get("scenario_id"),
                scenario_id,
                f"run {index} provenance scenario",
            )
            _require_equal(
                problems,
                _strict_int(provenance.get("seed")),
                seed,
                f"run {index} provenance seed",
            )
            row_config_hash = str(provenance.get("config_hash") or "").strip()
            if row.get("config_hash") is not None:
                _require_equal(
                    problems,
                    str(row.get("config_hash")).strip(),
                    row_config_hash,
                    f"run {index} config provenance",
                )
            if not row_config_hash:
                problems.append(f"run {index} episode config provenance is missing")
            else:
                _validate_episode_provenance_sidecar(
                    episodes_path=episodes_path,
                    campaign_root=campaign_root,
                    repo_root=resolved_repo,
                    planner_key=planner_key,
                    algorithm=algorithms.get(planner_key),
                    algorithm_config=algorithm_configs.get(planner_key),
                    scenario_path=scenario_path,
                    expected_source_commit=expected_source_commit,
                    scenario_id=scenario_id,
                    seed=seed,
                    expected_scenario_matrix_hash=expected_scenario_matrix_hash,
                    episode_id=str(row.get("episode_id", "")).strip(),
                    row_config_hash=row_config_hash,
                    relocation_root=relocation_root,
                    problems=problems,
                )
            metadata = row.get("algorithm_metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            if not metadata:
                problems.append(f"run {index} algorithm metadata is missing")
            expected_metadata_algorithm = (
                "ppo" if planner_key == "guarded_ppo" else algorithms.get(planner_key)
            )
            _require_equal(
                problems,
                metadata.get("algorithm"),
                expected_metadata_algorithm,
                f"run {index} episode algorithm",
            )
            _require_equal(
                problems,
                metadata.get("canonical_algorithm"),
                algorithms.get(planner_key),
                f"run {index} canonical algorithm",
            )
            planner_contract = metadata.get("planner_contract")
            if isinstance(planner_contract, dict) and "planner_id" in planner_contract:
                _require_equal(
                    problems,
                    planner_contract.get("planner_id"),
                    algorithms.get(planner_key),
                    f"run {index} planner contract identity",
                )
            observed_episode_identities.append((planner_key, scenario_id, seed))

    if tuple(observed_arms) != expected_planner_keys or len(set(observed_arms)) != expected_rows:
        problems.append("runtime smoke raw planner arms do not match the exact roster")
    expected_identities = {(key, scenario_id, seed) for key in expected_planner_keys}
    if (
        len(observed_episode_identities) != expected_rows
        or set(observed_episode_identities) != expected_identities
    ):
        problems.append("runtime smoke raw episode identities do not match the exact matrix")
    if fallback_markers:
        problems.append(
            "runtime smoke contains fallback or degraded markers: " + fallback_markers[0]
        )
    recomputed_integrity: dict[str, Any] | None = None
    try:
        recomputed_integrity = validate_campaign_integrity(
            integrity_entries,
            scenarios=[{"id": scenario_id, "seeds": [seed]}],
            resolved_seeds=[seed],
            campaign_root=campaign_root,
            campaign_manifest={"git": {"commit": expected_source_commit}},
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        problems.append(f"runtime smoke raw integrity recomputation failed: {exc}")
    else:
        _require_equal(
            problems,
            recomputed_integrity.get("status"),
            "valid",
            "recomputed campaign integrity",
        )
        _require_equal(
            problems,
            recomputed_integrity.get("expected_identity_count"),
            1,
            "recomputed expected identity count",
        )
        _require_equal(
            problems,
            recomputed_integrity.get("checked_arm_count"),
            expected_rows,
            "recomputed checked arm count",
        )
        _require_equal(
            problems,
            recomputed_integrity.get("blockers"),
            [],
            "recomputed campaign integrity blockers",
        )
    integrity_artifacts = summary.get("artifacts")
    integrity_artifacts = integrity_artifacts if isinstance(integrity_artifacts, dict) else {}
    raw_integrity_path = integrity_artifacts.get("campaign_integrity_json")
    if raw_integrity_path is not None:
        try:
            integrity_path = _resolve_campaign_artifact(
                raw_path=raw_integrity_path,
                campaign_root=campaign_root,
                repo_root=resolved_repo,
                expected_path=campaign_root / "reports" / "campaign_integrity.json",
                label="campaign integrity artifact",
                relocation_root=relocation_root,
            )
            persisted_integrity = _read_object(integrity_path, "campaign integrity artifact")
        except (OSError, ValueError, RuntimeSmokeAdmissionError) as exc:
            problems.append(f"campaign integrity artifact rejected: {exc}")
        else:
            if recomputed_integrity is not None:
                _require_equal(
                    problems,
                    persisted_integrity,
                    recomputed_integrity,
                    "persisted campaign integrity",
                )

    campaign_row_status = campaign.get("row_status_summary")
    campaign_row_status = campaign_row_status if isinstance(campaign_row_status, dict) else {}
    for field, expected in (
        ("successful_evidence_rows", expected_rows),
        ("accepted_unavailable_rows", 0),
        ("unexpected_failed_rows", 0),
        ("fallback_or_degraded_rows", 0),
    ):
        _require_equal(
            problems,
            _strict_int(campaign_row_status.get(field)),
            expected,
            f"campaign row status {field}",
        )

    planner_row_arms: list[str] = []
    for index, row in enumerate(planner_rows):
        if not isinstance(row, dict):
            problems.append(f"runtime smoke planner row {index} is not an object")
            continue
        planner_row_arms.append(str(row.get("planner_key", "")).strip())
        _require_equal(problems, row.get("status"), "ok", f"planner row {index} status")
        planner_success = row.get("benchmark_success")
        if planner_success is not True and planner_success != "true":
            problems.append(f"planner row {index} benchmark success mismatch")
        _require_equal(
            problems, _strict_int(row.get("episodes")), 1, f"planner row {index} episodes"
        )
        if _forbidden_status_markers(row, f"planner_rows[{index}]"):
            problems.append(f"planner row {index} contains fallback or degraded marker")
    if (
        set(planner_row_arms) != set(expected_planner_keys)
        or len(set(planner_row_arms)) != expected_rows
        or len(planner_row_arms) != expected_rows
    ):
        problems.append("runtime smoke planner aggregates do not match the exact roster")

    for field in ("total_runs", "successful_runs", "total_episodes"):
        _require_equal(problems, result.get(field), expected_rows, field)
    for field in ("non_success_runs", "accepted_unavailable_runs", "unexpected_failed_runs"):
        _require_equal(problems, result.get(field), 0, field)
    row_status = result.get("row_status_summary")
    row_status = row_status if isinstance(row_status, dict) else {}
    _require_equal(
        problems,
        row_status.get("successful_evidence_rows"),
        expected_rows,
        "successful evidence rows",
    )
    for field in (
        "accepted_unavailable_rows",
        "unexpected_failed_rows",
        "fallback_or_degraded_rows",
    ):
        _require_equal(problems, row_status.get(field), 0, field)
    integrity = result.get("campaign_integrity")
    integrity = integrity if isinstance(integrity, dict) else {}
    _require_equal(problems, integrity.get("status"), "valid", "campaign integrity")
    _require_equal(problems, integrity.get("checked_arm_count"), expected_rows, "checked arm count")
    _require_equal(problems, result.get("release_benchmark_success"), True, "release success")
    _require_equal(problems, result.get("release_status"), "ok", "release status")
    _require_equal(problems, result.get("release_exit_code"), 0, "release exit code")
    if problems:
        raise RuntimeSmokeAdmissionError("runtime smoke admission failed: " + "; ".join(problems))
    finished_at_utc = _validate_age(run_meta, max_age_hours=max_age_hours)
    return {
        "schema_version": "benchmark-runtime-smoke-admission.v1",
        "status": "admitted",
        "result_sha256": sha256_file(resolved_result),
        "checkpoint_receipt_sha256": checkpoint["sha256"],
        "source_commit": expected_source_commit,
        "campaign_id": result["campaign_id"],
        "finished_at_utc": finished_at_utc,
        "planner_arms": expected_rows,
        "episode_cells": len(observed_episode_identities),
        "fallback_or_degraded_rows": 0,
    }


__all__ = [
    "RUNTIME_SMOKE_PLANNER_KEYS",
    "RuntimeSmokeAdmissionError",
    "validate_runtime_smoke_result",
]
