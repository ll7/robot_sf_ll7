"""Result provenance helpers for map-based benchmark runs."""

from __future__ import annotations

import platform
import shlex
import sys
import uuid
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.utils import _config_hash, _git_hash_fallback

if TYPE_CHECKING:
    from pathlib import Path


def map_result_provenance(  # noqa: PLR0913
    *,
    schema_path: str | Path,
    scenario_path: Path,
    scenarios: list[dict[str, Any]],
    algo: str,
    algo_config_path: str | None,
    benchmark_profile: str,
    suite_key: str,
    total_jobs: int,
    written: int,
    artifact_pointer_status: str,
    metric_affecting_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return self-describing provenance for map-runner summary artifacts.

    The optional ``metric_affecting_config`` is the metric-affecting run-config
    block (LiDAR ``scan_noise`` and the collision-handling regime; see
    ``robot_sf/benchmark/run_config_provenance.py``). When provided it is embedded
    under ``config_identity`` so the result artifact is self-describing about the
    settings that change metric semantics (issue #3701). When ``None`` the field
    is omitted for backwards compatibility.
    """
    from robot_sf.benchmark.release_protocol import BENCHMARK_PROTOCOL_VERSION  # noqa: PLC0415

    config_identity: dict[str, Any] = {
        "schema_path": str(schema_path),
        "scenario_path": str(scenario_path),
        "scenario_count": len(scenarios),
        "scenario_matrix_hash": _config_hash(scenarios),
        "algo": str(algo),
        "algo_config_path": str(algo_config_path) if algo_config_path is not None else None,
        "benchmark_profile": str(benchmark_profile),
    }
    if metric_affecting_config is not None:
        config_identity["metric_affecting_config"] = metric_affecting_config

    provenance: dict[str, Any] = {
        "protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "commit_hash": _git_hash_fallback(),
        "run_id": uuid.uuid4().hex,
        "python_version": platform.python_version(),
        "artifact_pointer_status": artifact_pointer_status,
        "config_identity": config_identity,
        "seed_identity": {
            "suite_key": suite_key,
            "total_jobs": int(total_jobs),
            "written": int(written),
        },
    }
    if hasattr(sys, "argv") and sys.argv:
        provenance["invocation"] = shlex.join(sys.argv)
    return provenance
