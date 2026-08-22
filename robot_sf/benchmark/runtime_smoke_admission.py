"""Fail-closed admission of the canonical 14-arm release runtime smoke."""

from __future__ import annotations

import json
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

RUNTIME_SMOKE_RELEASE_ID = "paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2"
RUNTIME_SMOKE_MANIFEST = Path(
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml"
)
RUNTIME_SMOKE_CONFIG = Path(
    "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_runtime_smoke.yaml"
)
RUNTIME_SMOKE_HORIZON = 600
RUNTIME_SMOKE_KINEMATICS = "differential_drive"
RUNTIME_SMOKE_PLANNER_KEYS = (
    "prediction_planner",
    "goal",
    "social_force",
    "orca",
    "ppo",
    "socnav_sampling",
    "sacadrl",
    "scenario_adaptive_hybrid_orca_v1",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
    "guarded_ppo",
    "predictive_mppi",
    "risk_dwa",
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
        "partial-failure",
        "partial_failure",
        "placeholder",
        "unavailable",
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


def _forbidden_status_markers(  # noqa: C901
    payload: Any, prefix: str
) -> list[tuple[str, str]]:
    """Find forbidden execution markers, including nested policy summaries.

    Returns:
        Path/value pairs for every forbidden marker.
    """
    markers = list(_status_markers(payload, prefix)) if isinstance(payload, dict) else []
    seen = {(path, value) for path, value in markers}
    status_fields = {
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
    flag_fields = {
        "fallback_triggered",
        "fallback_used",
        "fallback_active",
        "degraded",
        "fallback_or_degraded",
    }
    shallow_containers = {
        "summary",
        "benchmark_availability",
        "campaign_integrity",
        "row_status_summary",
        "fallback_policy",
        "availability",
    }
    deep_containers = {
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
    declarative_containers = {
        "config",
        "planner_contract",
        "safety_shield_contract",
    }

    def _walk(value: Any, path: str, *, descend_all: bool = False) -> None:  # noqa: C901
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                normalized_key = str(key).strip().lower()
                normalized_value = str(child).strip().lower().replace(" ", "_")
                is_status_key = normalized_key in status_fields or normalized_key.endswith(
                    "_status"
                )
                if is_status_key and (
                    normalized_value in _FORBIDDEN_RUNTIME_STATUSES
                    or normalized_value.startswith("predictive_foresight_model_fallback")
                ):
                    marker = (child_path, normalized_value)
                    if marker not in seen:
                        markers.append(marker)
                        seen.add(marker)
                if (
                    "fallback" in normalized_key
                    and isinstance(child, (int, float))
                    and not isinstance(child, bool)
                    and child > 0
                ):
                    marker = (child_path, str(child))
                    if marker not in seen:
                        markers.append(marker)
                        seen.add(marker)
                if normalized_key in flag_fields and child is True:
                    marker = (child_path, "true")
                    if marker not in seen:
                        markers.append(marker)
                        seen.add(marker)
                if normalized_key in declarative_containers:
                    continue
                if descend_all or normalized_key in deep_containers:
                    _walk(child, child_path, descend_all=True)
                elif normalized_key in shallow_containers:
                    _walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                _walk(child, f"{path}[{index}]", descend_all=descend_all)

    _walk(payload, prefix)
    return markers


def _resolve_campaign_artifact(
    *,
    raw_path: Any,
    campaign_root: Path,
    repo_root: Path,
    expected_path: Path,
    label: str,
    expected_root: Path | None = None,
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
        if candidate.is_symlink():
            continue
        resolved = candidate.resolve(strict=False)
        if resolved == expected and resolved.is_file():
            return resolved
    raise RuntimeSmokeAdmissionError(f"{label} is not bound to this campaign")


def _validate_episode_provenance_sidecar(  # noqa: PLR0913
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
    row_config_hash: str,
    problems: list[str],
) -> None:
    """Bind one raw episode artifact to its exact arm and tracked inputs."""
    sidecar_path = episodes_path.with_name(f"{episodes_path.name}.provenance.json")
    if sidecar_path.is_symlink() or not sidecar_path.is_file():
        problems.append(f"planner {planner_key} episode provenance sidecar is missing")
        return
    sidecar = _read_object(sidecar_path, f"planner {planner_key} episode provenance sidecar")
    run = sidecar.get("run")
    run = run if isinstance(run, dict) else {}
    _require_equal(
        problems,
        str(run.get("repo_commit", "")).strip().lower(),
        expected_source_commit,
        f"planner {planner_key} sidecar source commit",
    )
    identity = sidecar.get("campaign_identity")
    identity = identity if isinstance(identity, dict) else {}
    _require_equal(
        problems,
        identity.get("algorithm"),
        algorithm,
        f"planner {planner_key} sidecar algorithm",
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
        _require_equal(problems, _strict_int(sidecar_row.get("seed")), seed, "sidecar seed")
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
        try:
            _resolve_campaign_artifact(
                raw_path=artifact.get("path"),
                campaign_root=campaign_root,
                repo_root=repo_root,
                expected_path=episodes_path,
                label=f"planner {planner_key} sidecar raw artifact inventory",
            )
        except RuntimeSmokeAdmissionError as exc:
            problems.append(str(exc))
        _require_equal(
            problems,
            artifact.get("sha256"),
            sha256_file(episodes_path),
            f"planner {planner_key} sidecar raw artifact hash",
        )


def _canonical_smoke_contract(  # noqa: C901
    *, repo_root: Path, expected_planner_keys: tuple[str, ...]
) -> tuple[Path, Path, Path, str, int, dict[str, str], dict[str, str | None]]:
    """Resolve the tracked smoke axes and planner algorithms from canonical inputs.

    Returns:
        Manifest path, config path, scenario path, scenario ID, seed,
        planner-to-algorithm mapping, and planner-to-algorithm-config mapping.
    """
    manifest_path = repo_root / RUNTIME_SMOKE_MANIFEST
    config_path = repo_root / RUNTIME_SMOKE_CONFIG
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
    scenario_path = (manifest_path.parent / scenario_rel).resolve()
    if not scenario_path.is_file():
        raise RuntimeSmokeAdmissionError("canonical runtime smoke scenario is missing")
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
    campaign_manifest = _read_object(
        campaign_root / "campaign_manifest.json", "runtime smoke campaign manifest"
    )
    run_manifest = _read_object(campaign_root / "manifest.json", "runtime smoke run manifest")
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
    resolved_result = result_path.resolve()
    if not resolved_result.is_relative_to(resolved_repo):
        raise RuntimeSmokeAdmissionError("runtime smoke result must be inside the release worktree")
    if resolved_result.name != "release_result.json" or resolved_result.parent.name != "release":
        raise RuntimeSmokeAdmissionError(
            "runtime smoke result is not the canonical release receipt"
        )
    result = _read_object(resolved_result, "runtime smoke result")
    campaign_root = resolved_result.parent.parent
    run_meta = _read_object(campaign_root / "run_meta.json", "runtime smoke run metadata")
    summary = _read_object(
        campaign_root / "reports" / "campaign_summary.json", "runtime smoke campaign summary"
    )
    (
        manifest_path,
        config_path,
        scenario_path,
        scenario_id,
        seed,
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
    for field, raw_path in (
        ("campaign_root", result.get("campaign_root")),
        ("summary_json", result.get("summary_json")),
    ):
        if raw_path is None:
            continue
        if field == "campaign_root":
            candidate = Path(str(raw_path))
            if not candidate.is_absolute():
                candidate = resolved_repo / candidate
            _require_equal(
                problems,
                candidate.resolve(),
                expected_campaign_root,
                "result campaign root",
            )
            continue
        try:
            _resolve_campaign_artifact(
                raw_path=raw_path,
                campaign_root=expected_campaign_root,
                repo_root=resolved_repo,
                expected_path=expected_campaign_root / "reports" / "campaign_summary.json",
                label=f"result {field}",
            )
        except RuntimeSmokeAdmissionError as exc:
            problems.append(str(exc))

    checkpoint = result.get("checkpoint_staging_receipt")
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    checkpoint_rel = checkpoint.get("path")
    if not isinstance(checkpoint_rel, str) or not checkpoint_rel.strip():
        problems.append("checkpoint staging receipt path is missing")
    else:
        checkpoint_path = (resolved_repo / checkpoint_rel).resolve()
        if not checkpoint_path.is_relative_to(resolved_repo) or not checkpoint_path.is_file():
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
                    row_config_hash=row_config_hash,
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
