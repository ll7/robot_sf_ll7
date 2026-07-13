"""Run-state and provenance helpers for camera-ready campaigns.

Extracted from ``robot_sf.benchmark.camera_ready_campaign`` as a bounded
package-decomposition slice for issue #3385. The legacy module re-exports these
helpers so existing imports keep working while orchestration is split gradually.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from robot_sf.benchmark.camera_ready._config import _sanitize_name
from robot_sf.benchmark.fallback_policy import (
    resolve_execution_mode as _resolve_benchmark_execution_mode,
)
from robot_sf.benchmark.observation_noise import (
    load_observation_noise_spec,
    normalize_observation_noise_spec,
)
from robot_sf.benchmark.utils import _git_hash_fallback
from robot_sf.common.artifact_paths import get_repository_root

if TYPE_CHECKING:
    from robot_sf.benchmark.camera_ready._config_types import CampaignConfig


def _campaign_success_counters(
    run_entries: Sequence[Mapping[str, Any]],
    *,
    expected_core_runs: int | None = None,
) -> dict[str, Any]:
    """Return campaign success counters, anchoring success on planners present."""
    total_runs = 0
    successful_runs = 0
    core_total_runs = 0
    core_successful_runs = 0

    for entry in run_entries:
        total_runs += 1
        is_ok = str(entry.get("status", "")) == "ok"
        if is_ok:
            successful_runs += 1

        planner_group = str((entry.get("planner") or {}).get("planner_group", "")).strip().lower()
        if planner_group == "core":
            core_total_runs += 1
            if is_ok:
                core_successful_runs += 1

    if expected_core_runs is None:
        expected_core_runs = core_total_runs

    if core_total_runs:
        success_basis = "core"
        benchmark_success = core_successful_runs == core_total_runs == expected_core_runs
    else:
        success_basis = "all"
        benchmark_success = total_runs > 0 and successful_runs == total_runs

    return {
        "benchmark_success": benchmark_success,
        "benchmark_success_basis": success_basis,
        "total_runs": total_runs,
        "successful_runs": successful_runs,
        "core_total_runs": core_total_runs,
        "core_successful_runs": core_successful_runs,
    }


def _episode_identity(record: Mapping[str, Any]) -> tuple[str, int | str | None]:
    """Return the stable logical identity used by aggregate-integrity checks."""
    scenario_id = str(record.get("scenario_id", "")).strip()
    seed = record.get("seed")
    try:
        seed = int(seed) if seed is not None else None
    except (TypeError, ValueError):
        seed = str(seed)
    return scenario_id, seed


def _expected_episode_identities(
    scenarios: Sequence[Mapping[str, Any]],
    resolved_seeds: Sequence[int],
) -> set[tuple[str, int]]:
    """Build the expected scenario/seed denominator from the prepared campaign matrix.

    Returns:
        Set of logical scenario/seed identities expected in every successful arm.
    """
    expected: set[tuple[str, int]] = set()
    fallback_seeds = [int(seed) for seed in resolved_seeds]
    for scenario in scenarios:
        scenario_id = str(
            scenario.get("id") or scenario.get("scenario_id") or scenario.get("name") or ""
        ).strip()
        raw_seeds = scenario.get("seeds")
        seeds = (
            raw_seeds if isinstance(raw_seeds, Sequence) and not isinstance(raw_seeds, str) else ()
        )
        if not seeds:
            seeds = fallback_seeds
        for seed in seeds:
            try:
                expected.add((scenario_id, int(seed)))
            except (TypeError, ValueError):
                continue
    return expected


def _resolve_integrity_artifact_path(campaign_root: Path, raw_path: str) -> Path:
    """Resolve a campaign-relative or repository-relative artifact path.

    Returns:
        Resolved artifact path.
    """
    path = Path(raw_path)
    if path.is_absolute():
        return path
    repo_candidate = (get_repository_root() / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (campaign_root / path).resolve()


def _integrity_blocker(arm: str, invariant: str, **details: Any) -> dict[str, Any]:
    """Build one deterministic, machine-readable aggregate-integrity blocker.

    Returns:
        Blocker payload with arm, invariant, and structured details.
    """
    return {"arm": arm, "invariant": invariant, "details": details}


def validate_campaign_integrity(  # noqa: C901, PLR0912, PLR0915
    run_entries: Sequence[Mapping[str, Any]],
    *,
    scenarios: Sequence[Mapping[str, Any]],
    resolved_seeds: Sequence[int],
    campaign_root: Path,
    campaign_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate final arm aggregates without modifying or deduplicating their rows.

    The checker intentionally treats an appended row as evidence of contamination.  It never
    selects a favorable slice: every successful arm must have exact logical coverage and enough
    row-level provenance to establish one compatible public commit/config attempt.

    Returns:
        Machine-readable integrity verdict and blockers.
    """
    expected = _expected_episode_identities(scenarios, resolved_seeds)
    blockers: list[dict[str, Any]] = []
    manifest_git = str(
        (campaign_manifest.get("git") or {}).get("commit")
        if isinstance(campaign_manifest.get("git"), Mapping)
        else campaign_manifest.get("git_hash", "")
    ).strip()

    for entry in run_entries:
        if str(entry.get("status", "")) != "ok":
            continue
        planner = entry.get("planner")
        planner = planner if isinstance(planner, Mapping) else {}
        arm = f"{planner.get('key', 'unknown')} ({planner.get('kinematics', 'unknown')})"
        raw_path = str(entry.get("episodes_path", "")).strip()
        if not raw_path:
            blockers.append(_integrity_blocker(arm, "missing_episode_artifact"))
            continue
        episodes_path = _resolve_integrity_artifact_path(campaign_root, raw_path)
        try:
            records: list[dict[str, Any]] = []
            with episodes_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("episode row must be an object")
                    records.append(payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            blockers.append(_integrity_blocker(arm, "unreadable_episode_artifact", error=str(exc)))
            continue

        observed = [_episode_identity(record) for record in records]
        observed_set = set(observed)
        duplicate_counts = Counter(observed)
        duplicates = sorted(
            [identity for identity, count in duplicate_counts.items() if count > 1],
            key=str,
        )
        if duplicates:
            blockers.append(
                _integrity_blocker(
                    arm,
                    "duplicate_logical_coverage",
                    identities=[list(identity) for identity in duplicates],
                )
            )

        summary = entry.get("summary")
        summary = summary if isinstance(summary, Mapping) else {}
        declared = summary.get("episodes_total")
        if declared is None:
            declared = summary.get("written")
        count_details = {
            "expected": len(expected),
            "observed": len(records),
            "declared": declared,
            "missing_identities": [list(identity) for identity in sorted(expected - observed_set)],
            "unexpected_identities": [
                list(identity) for identity in sorted(observed_set - expected)
            ],
        }
        declared_count: int | None = None
        if declared is not None:
            try:
                declared_count = int(declared)
            except (TypeError, ValueError):
                declared_count = None
        if len(records) != len(expected) or (
            declared is not None and declared_count != len(records)
        ):
            blockers.append(_integrity_blocker(arm, "count_mismatch", **count_details))

        commits: set[str] = set()
        configs_by_identity: dict[tuple[str, int | str | None], set[str]] = {}
        for record in records:
            result_provenance = record.get("result_provenance")
            result_provenance = result_provenance if isinstance(result_provenance, Mapping) else {}
            row_config = str(
                result_provenance.get("config_hash") or record.get("config_hash") or ""
            ).strip()
            row_commit = str(
                result_provenance.get("repo_commit") or record.get("git_hash") or ""
            ).strip()
            if not row_config or not row_commit:
                blockers.append(
                    _integrity_blocker(
                        arm,
                        "missing_provenance",
                        identity=list(_episode_identity(record)),
                        missing=[
                            field
                            for field, value in (
                                ("config_hash", row_config),
                                ("commit", row_commit),
                            )
                            if not value
                        ],
                    )
                )
            if row_commit:
                commits.add(row_commit)
            identity = _episode_identity(record)
            configs_by_identity.setdefault(identity, set()).add(row_config)
            if result_provenance:
                if _episode_identity(result_provenance) != identity:
                    blockers.append(
                        _integrity_blocker(
                            arm,
                            "row_provenance_identity_mismatch",
                            identity=list(identity),
                            provenance_identity=list(_episode_identity(result_provenance)),
                        )
                    )

        if manifest_git and commits - {manifest_git}:
            blockers.append(
                _integrity_blocker(
                    arm,
                    "mixed_commit_provenance",
                    expected=manifest_git,
                    observed=sorted(commits),
                )
            )
        elif len(commits) > 1:
            blockers.append(
                _integrity_blocker(arm, "mixed_commit_provenance", observed=sorted(commits))
            )
        mixed_configs = {
            identity: sorted(values)
            for identity, values in configs_by_identity.items()
            if len(values) > 1
        }
        if mixed_configs:
            blockers.append(
                _integrity_blocker(
                    arm,
                    "mixed_config_provenance",
                    identities={
                        str(identity): values for identity, values in mixed_configs.items()
                    },
                )
            )

    status = "valid" if not blockers else "invalid"
    return {
        "schema_version": "benchmark-camera-ready-integrity.v1",
        "status": status,
        "benchmark_success_allowed": status == "valid",
        "expected_identity_count": len(expected),
        "checked_arm_count": sum(str(entry.get("status", "")) == "ok" for entry in run_entries),
        "blockers": blockers,
        "claim_boundary": (
            "A derived clean slice is diagnostic-only unless it was predeclared and "
            "provenance-complete."
        ),
    }


_MAX_FIRST_ERROR_LEN = 200


def _build_arm_rollup(run_entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build per-arm rollup for the top-level campaign summary.

    One entry per arm with planner key, kinematics, status, episode counts,
    and for failed/partial arms the first error string (truncated) plus the
    count of distinct error signatures from per-job failures.

    Returns:
        List of arm rollup dicts, one per run entry.
    """
    rollup: list[dict[str, Any]] = []
    for entry in run_entries:
        planner_info = entry.get("planner") or {}
        summary = entry.get("summary") or {}
        failures = summary.get("failures") or []
        status = str(entry.get("status", "unknown"))
        written_value = summary.get("written")
        if written_value is None:
            written_value = summary.get("episodes_total")
        episodes_written = int(written_value) if written_value is not None else 0
        failed_jobs_value = summary.get("failed_jobs")
        episodes_failed = int(failed_jobs_value) if failed_jobs_value is not None else 0

        first_error: str | None = None
        distinct_error_count = 0
        if status not in {"ok", "not_available"}:
            error_signatures: set[str] = set()
            first_error_raw: str | None = None
            for failure in failures:
                error_value = failure.get("error")
                error_str = str(error_value) if error_value is not None else ""
                if error_str:
                    if first_error_raw is None:
                        first_error_raw = error_str
                    error_signatures.add(error_str)
            if error_signatures:
                first_error = first_error_raw[:_MAX_FIRST_ERROR_LEN] if first_error_raw else None
                distinct_error_count = len(error_signatures)
            else:
                summary_error = summary.get("error")
                if summary_error:
                    first_error = str(summary_error)[:_MAX_FIRST_ERROR_LEN]

        arm_entry: dict[str, Any] = {
            "planner_key": str(planner_info.get("key", "unknown")),
            "algo": str(planner_info.get("algo", "unknown")),
            "kinematics": str(planner_info.get("kinematics", "unknown")),
            "status": status,
            "episodes_written": episodes_written,
            "episodes_failed": episodes_failed,
        }
        if first_error is not None:
            arm_entry["first_error"] = first_error
            arm_entry["distinct_error_count"] = distinct_error_count
        rollup.append(arm_entry)
    return rollup


def _resolve_execution_mode(algorithm_metadata_contract: Any) -> str:
    """Resolve execution mode from algorithm metadata payload with legacy fallbacks.

    Returns:
        Resolved execution mode string, or ``"unknown"`` when unavailable.
    """
    return _resolve_benchmark_execution_mode(algorithm_metadata_contract)


def _sanitize_git_remote(remote: str) -> str:
    """Remove credentials from git remote URLs before persisting provenance metadata.

    Returns:
        Credential-free remote URL when parseable, otherwise original input.
    """
    if not remote or "://" not in remote:
        return remote
    try:
        parsed = urlsplit(remote)
    except ValueError:
        return remote
    if not parsed.hostname:
        return remote
    netloc = parsed.hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _git_context() -> dict[str, str]:
    """Collect lightweight git metadata for campaign provenance.

    Returns:
        Mapping with ``commit``, ``branch``, and sanitized ``remote`` fields.
    """

    def _run(args: list[str]) -> str:
        """Run a git command and degrade to ``unknown`` when provenance is unavailable.

        Returns:
            Decoded command output, or ``"unknown"`` on command failure.
        """
        try:
            out = subprocess.check_output(args, stderr=subprocess.DEVNULL)
            return out.decode("utf-8", errors="replace").strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return "unknown"

    return {
        "commit": _git_hash_fallback(),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "remote": _sanitize_git_remote(_run(["git", "config", "--get", "remote.origin.url"])),
    }


def _campaign_id(cfg: CampaignConfig, *, label: str | None = None) -> str:
    """Build a unique campaign identifier from config name and wall-clock timestamp.

    Returns:
        Campaign identifier used for output directories and manifests.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _sanitize_name(cfg.name)
    if label:
        suffix = _sanitize_name(label)
        return f"{base}_{suffix}_{stamp}"
    return f"{base}_{stamp}"


def _resolve_campaign_id(
    cfg: CampaignConfig,
    *,
    label: str | None = None,
    campaign_id: str | None = None,
) -> str:
    """Resolve the output campaign identifier.

    Returns:
        Explicit sanitized campaign id when provided, otherwise timestamped id.
    """
    if campaign_id is not None:
        normalized = _sanitize_name(campaign_id)
        if not normalized:
            raise ValueError("campaign_id must contain at least one alphanumeric character")
        return normalized
    return _campaign_id(cfg, label=label)


def _resolve_path(raw_path: str | None, *, base_dir: Path) -> Path | None:
    """Resolve paths relative to ``base_dir``.

    Returns:
        Absolute resolved path, or ``None`` when no path was provided.
    """
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path

    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate

    repo_candidate = (get_repository_root() / path).resolve()
    if repo_candidate.exists():
        return repo_candidate

    return candidate


def _resolve_observation_noise(raw: Any, *, base_dir: Path) -> dict[str, Any] | None:
    """Resolve an optional inline or file-backed observation-noise config.

    Returns:
        Normalized noise spec, or ``None`` when no profile was configured.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return normalize_observation_noise_spec(raw)
    if isinstance(raw, str) and raw.strip():
        path = _resolve_path(raw, base_dir=base_dir)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Could not resolve observation_noise '{raw}'")
        return load_observation_noise_spec(path)
    raise ValueError("observation_noise must be mapping or YAML file path")
