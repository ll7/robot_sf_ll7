"""Run-state and provenance helpers for camera-ready campaigns.

Extracted from ``robot_sf.benchmark.camera_ready_campaign`` as a bounded
package-decomposition slice for issue #3385. The legacy module re-exports these
helpers so existing imports keep working while orchestration is split gradually.
"""

from __future__ import annotations

import subprocess
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
    from collections.abc import Mapping, Sequence

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
        episodes_written = int(summary.get("written", summary.get("episodes_total", 0)))
        episodes_failed = int(summary.get("failed_jobs", 0))

        first_error: str | None = None
        distinct_error_count = 0
        if status not in {"ok", "not_available"}:
            error_signatures: set[str] = set()
            for failure in failures:
                error_str = str(failure.get("error", ""))
                if error_str:
                    error_signatures.add(error_str)
            if error_signatures:
                first_error = sorted(error_signatures)[0][:_MAX_FIRST_ERROR_LEN]
                distinct_error_count = len(error_signatures)
            elif str(summary.get("error", "")):
                first_error = str(summary["error"])[:_MAX_FIRST_ERROR_LEN]

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
