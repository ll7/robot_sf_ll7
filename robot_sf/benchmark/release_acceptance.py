"""Fail-closed acceptance checks for a full benchmark-data release.

The camera-ready campaign runner intentionally supports partial and
``core``-only success semantics for exploratory work.  A publication release
has a stricter contract: every declared arm must complete every declared
scenario/seed cell, and no fallback or degraded row may be promoted.  This
module keeps that publication gate separate from the bounded runtime smoke.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.benchmark.camera_ready._preflight import _resolved_seed_inventory
from robot_sf.benchmark.camera_ready._run_state import _resolve_integrity_artifact_path
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker
from robot_sf.benchmark.release_protocol import (
    STRESS_SMOKE_EXPECTED_DT,
    STRESS_SMOKE_EXPECTED_EPISODE_CELLS,
    STRESS_SMOKE_EXPECTED_HORIZON_STEPS,
    STRESS_SMOKE_EXPECTED_KINEMATICS,
    STRESS_SMOKE_EXPECTED_PLANNER_ARMS,
    STRESS_SMOKE_EXPECTED_SCENARIO_IDS,
    STRESS_SMOKE_EXPECTED_SEED,
)

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
_STRESS_ROW_SUCCESS_STATUSES = frozenset({"success"})
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

    def _walk(value: Any, path: str, *, active: bool = False) -> tuple[str, str] | None:  # noqa: C901
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
                    if normalized_key == "selected_source" and normalized_value in _LEGACY_EMERGENCY_SOURCES:
                        return nested_path, normalized_value
                    if normalized_key in {"mode", "planner_mode"} and normalized_value in _LEGACY_EMERGENCY_MODES:
                        return nested_path, normalized_value
                elif inspect_marker and normalized_key == "emergency_stop_count":
                    parsed_counter = _strict_int(nested)
                    if parsed_counter is None or parsed_counter < 0:
                        return nested_path, "invalid"
                    if parsed_counter > 0:
                        return nested_path, str(nested)
                elif inspect_marker and normalized_key == "emergency_stop" and nested is True:
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


def _status_markers(  # noqa: C901, PLR0912
    payload: Mapping[str, Any], prefix: str
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

    for field in ("fallback_triggered", "degraded", "fallback_or_degraded"):
        if payload.get(field) is True:
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
        _add(f"{field}.status", metadata.get("status"))
        if metadata.get("fallback_or_degraded") is True:
            markers.append((f"{prefix}.{field}.fallback_or_degraded", "true"))
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
    return markers


def _read_campaign_summary(campaign_root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load the authoritative campaign summary, returning a shaped error.

    Returns:
        The parsed summary and an optional human-readable read error.
    """
    path = campaign_root / "reports" / "campaign_summary.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"campaign summary cannot be read: {exc}"
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
                except json.JSONDecodeError as exc:
                    return [], f"{path}:{line_number}: invalid JSON: {exc}"
                if not isinstance(payload, dict):
                    return [], f"{path}:{line_number}: episode row must be an object"
                rows.append(payload)
    except OSError as exc:
        return [], f"{path}: cannot read episode artifact: {exc}"
    return rows, None


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


def _alias_values(payload: Mapping[str, Any], aliases: tuple[tuple[str, Any], ...]) -> list[tuple[str, str]]:
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
                    ),
                )
                row_label = f"runs[{run_index}].rows[{row_index}].source_commit"
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


def _stress_row_contract_blockers(  # noqa: C901, PLR0912
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
    if not _status_is(row.get("status"), _STRESS_ROW_SUCCESS_STATUSES):
        blockers.append(f"{prefix}.status must be 'success'")
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
        ),
    )
    blockers.extend(
        f"{prefix}: {blocker}"
        for blocker in _alias_blockers(
            source_aliases,
            label="source",
            expected=expected_source_commit,
        )
    )

    config_aliases = _alias_values(
        row,
        (
            ("config_hash", row.get("config_hash", _MISSING)),
            (
                "result_provenance.config_hash",
                _nested_value(row, "result_provenance", "config_hash"),
            ),
            ("provenance.config_hash", _nested_value(row, "provenance", "config_hash")),
        ),
    )
    blockers.extend(
        f"{prefix}: {blocker}"
        for blocker in _alias_blockers(config_aliases, label="config")
    )

    algo_aliases = _alias_values(
        row,
        (
            ("algo", row.get("algo", _MISSING)),
            (
                "algorithm_metadata.algorithm",
                _nested_value(row, "algorithm_metadata", "algorithm"),
            ),
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
        for blocker in _alias_blockers(algo_aliases, label="planner algorithm", expected=expected_algo)
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
                ("algorithm_metadata.planner_kinematics.scenario_kinematics", str(value).strip().lower())
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
    return blockers


def _scenario_id(scenario: Mapping[str, Any]) -> str:
    """Resolve the stable scenario identifier used by campaign episode identity.

    Returns:
        Stable scenario identifier, or an empty string when absent.
    """
    return str(
        scenario.get("id") or scenario.get("scenario_id") or scenario.get("name") or ""
    ).strip()


def _resolve_expected_matrix_axes(  # noqa: C901
    manifest: Any,
    campaign_config: Any | None,
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
                campaign_config = load_campaign_config(config_path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                blockers.append(f"canonical campaign config cannot be resolved: {exc}")
    if campaign_config is not None:
        try:
            scenarios = _load_campaign_scenarios(campaign_config)
            scenario_ids = tuple(_scenario_id(scenario) for scenario in scenarios)
            seeds = tuple(_resolved_seed_inventory(scenarios))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            blockers.append(f"resolved campaign matrix cannot be loaded: {exc}")
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
    if _strict_int(getattr(campaign_config, "horizon", None)) != STRESS_SMOKE_EXPECTED_HORIZON_STEPS:
        _append_blocker(blockers, "campaign config horizon must be 600")
    try:
        campaign_dt = float(getattr(campaign_config, "dt", float("nan")))
    except (TypeError, ValueError):
        campaign_dt = float("nan")
    if campaign_dt != STRESS_SMOKE_EXPECTED_DT:
        _append_blocker(blockers, "campaign config dt must be 0.1")
    if tuple(
        str(value).strip().lower()
        for value in getattr(campaign_config, "kinematics_matrix", ())
    ) != (STRESS_SMOKE_EXPECTED_KINEMATICS,):
        _append_blocker(blockers, "campaign config kinematics must be differential_drive only")

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
            _append_blocker(blockers, "campaign summary campaign_execution_status must be completed")
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
        expected_algo = str(
            getattr(planner_spec, "algo", planner.get("algo", arm[0]))
        ).strip().lower()
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
            _append_blocker(blockers, f"runs[{run_index}] benchmark_success must be explicitly true")
        for marker_path, marker in _status_markers(run, f"runs[{run_index}]"):
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
            for marker_path, marker in _status_markers(run_summary, f"runs[{run_index}].summary"):
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
        observed_rows += len(rows)
        if len(rows) != expected_per_arm:
            _append_blocker(
                blockers,
                f"runs[{run_index}] contains {len(rows)} rows; expected {expected_per_arm}",
            )
        for row_index, row in enumerate(rows):
            for marker_path, marker in _status_markers(row, f"runs[{run_index}].rows[{row_index}]"):
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
        for marker_path, marker in _status_markers(
            planner_row, f"planner_rows[{planner_row_index}]"
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

    status = "valid" if not blockers else "invalid"
    return {
        "schema_version": "benchmark-stress-smoke-acceptance.v1",
        "status": status,
        "diagnostic_success": status == "valid",
        "expected_planner_arms": len(expected_arms),
        "expected_episode_cells": expected_cells,
        "observed_episode_rows": observed_rows,
        "unique_episode_identities": len(identities),
        "source_provenance": source_report,
        "claim_boundary": "diagnostic execution evidence only; no benchmark, ranking, or SNQI claim",
        "blockers": blockers,
    }


def validate_full_benchmark_release_acceptance(  # noqa: C901, PLR0912, PLR0915
    campaign_root: Path,
    *,
    manifest: Any,
    campaign_config: Any | None = None,
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
    scenario_ids, resolved_seeds, axis_blockers = _resolve_expected_matrix_axes(
        manifest, campaign_config
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

    for index, entry in enumerate(runs):
        if not isinstance(entry, Mapping):
            _append_blocker(blockers, f"runs[{index}] must be an object")
            continue
        planner = entry.get("planner")
        planner = planner if isinstance(planner, Mapping) else {}
        arm = (str(planner.get("key", "")).strip(), str(planner.get("kinematics", "")).strip())
        if arm in observed_arms:
            duplicate_arms.add(arm)
        observed_arms.add(arm)
        if str(entry.get("status", "")).strip().lower() != "ok":
            _append_blocker(blockers, f"runs[{index}] status is not ok")
        for marker_path, marker in _status_markers(entry, f"runs[{index}]"):
            forbidden_status_counts[marker] += 1
            _append_blocker(blockers, f"forbidden {marker_path}={marker}")
        entry_summary = entry.get("summary")
        if isinstance(entry_summary, Mapping):
            for marker_path, marker in _status_markers(entry_summary, f"runs[{index}].summary"):
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
            episodes_path = _resolve_integrity_artifact_path(campaign_root.resolve(), raw_path)
        except (OSError, ValueError) as exc:
            _append_blocker(blockers, f"runs[{index}] episodes_path rejected: {exc}")
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
        arm_identities: set[tuple[str, str, str, int]] = set()
        for row_index, row in enumerate(rows):
            for marker_path, marker in _status_markers(row, f"runs[{index}].rows[{row_index}]"):
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
        for marker_path, marker in _status_markers(row, f"planner_rows[{index}]"):
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

    expected_source = str(campaign.get("git_hash", "")).strip().lower()
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
