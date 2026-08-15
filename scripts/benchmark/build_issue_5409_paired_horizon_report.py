#!/usr/bin/env python3
"""Build the fail-closed paired analysis handoff for issue #5409.

The issue #5409 launch packet names three post-run artifacts but intentionally does
not let the launch packet produce them.  This command consumes the two completed
camera-ready campaign roots and writes:

* ``matched_key_completeness.json``;
* ``paired_horizon_deltas.json``; and
* ``paired_uncertainty_summary.json``.

The comparison is valid only when both arms contain exactly one row for every
``(planner_key, scenario_id, seed)`` identity, use the expected scenario matrix and
seed set, preserve campaign provenance, and report only ``native`` or ``adapter``
execution.  Missing, duplicate, fallback, degraded, unavailable, failed, partial,
or provenance-invalid rows produce blocked artifacts and no numeric comparison.

This is a nominal fixed-ablation analysis handoff.  It does not promote a result to
paper-grade evidence and does not infer a horizon finding from the presence of an
output file.

Example::

    uv run python scripts/benchmark/build_issue_5409_paired_horizon_report.py \
        --h500-root /path/to/issue5409_horizon_ablation_h500 \
        --h600-root /path/to/issue5409_horizon_ablation_h600 \
        --output-dir /path/to/issue5409_horizon_ablation_pair
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from robot_sf.benchmark.utils import episode_metric_value

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

try:
    from scripts.benchmark.validate_horizon_ablation_pair import (
        validate_horizon_ablation_pair,
    )
except ImportError:  # pragma: no cover - supports direct script execution from another cwd.
    validate_horizon_ablation_pair = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H500_CONFIG = REPO_ROOT / "configs/benchmarks/issue_5409_horizon_ablation_h500.yaml"
DEFAULT_H600_CONFIG = REPO_ROOT / "configs/benchmarks/issue_5409_horizon_ablation_h600.yaml"

COMPLETENESS_SCHEMA = "issue-5409-matched-key-completeness.v1"
DELTA_SCHEMA = "issue-5409-paired-horizon-deltas.v1"
UNCERTAINTY_SCHEMA = "issue-5409-paired-uncertainty-summary.v1"
METRICS: tuple[str, ...] = (
    "success",
    "collisions",
    "near_misses",
    "time_to_goal_norm",
    "snqi",
)
VALID_EXECUTION_MODES = frozenset({"native", "adapter"})
DEFAULT_SCENARIO_COUNT = 48
DEFAULT_SEEDS = (111, 112, 113)
DEFAULT_ROWS_PER_ARM = 1728
DEFAULT_SCENARIO_MATRIX_HASH = "c10df617a87c"
DEFAULT_BOOTSTRAP_SAMPLES = 300
DEFAULT_BOOTSTRAP_SEED = 123
DEFAULT_CONFIDENCE = 0.95
DEFAULT_CAMPAIGN_IDS = (
    "issue5409_horizon_ablation_h500",
    "issue5409_horizon_ablation_h600",
)

Key = tuple[str, str, int]


@dataclass(frozen=True)
class EpisodeCell:
    """One normalized episode row and the values used by the paired analysis."""

    key: Key
    scenario_family: str
    metrics: dict[str, float]
    execution_mode: str


@dataclass
class ArmInspection:
    """Collected evidence and blockers for one horizon arm."""

    role: str
    root: Path
    expected_horizon: int
    expected_planners: tuple[str, ...]
    expected_scenarios: tuple[str, ...] | None
    expected_seeds: tuple[int, ...]
    expected_rows: int
    metadata: dict[str, Any]
    rows: dict[Key, EpisodeCell]
    duplicate_keys: list[Key]
    metric_missing: dict[str, list[Key]]
    blockers: list[str]

    @property
    def unique_key_count(self) -> int:
        """Return the number of unique normalized episode identities."""
        return len(self.rows)


def _unique(values: Iterable[str]) -> list[str]:
    """Return non-empty strings in first-seen order without duplicates."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _nested(payload: Any, *keys: str) -> Any:
    """Read a nested mapping value without raising on malformed artifacts."""
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _finite_float(value: Any) -> float | None:
    """Coerce a finite numeric scalar."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _strict_seed(value: Any) -> int | None:
    """Coerce an integer seed without accepting booleans or truncating floats."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("+-").isdigit():
            try:
                return int(text, 10)
            except ValueError:
                return None
    return None


def _key_text(key: Key) -> str:
    """Render a stable key for blocker and artifact messages."""
    planner, scenario, seed = key
    return f"{planner}|{scenario}|{seed}"


def _key_payload(key: Key) -> dict[str, Any]:
    """Render a normalized identity as JSON."""
    planner, scenario, seed = key
    return {"planner_key": planner, "scenario_id": scenario, "seed": seed}


def _sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, blockers: list[str], *, label: str) -> dict[str, Any]:
    """Read a required JSON object and append a precise blocker on failure."""
    if not path.is_file():
        blockers.append(f"missing {label}: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(f"unreadable {label}: {path} ({exc})")
        return {}
    if not isinstance(payload, dict):
        blockers.append(f"{label} must contain a JSON object: {path}")
        return {}
    return payload


def _read_jsonl(path: Path, blockers: list[str], *, label: str) -> list[dict[str, Any]]:
    """Read JSONL episode objects, failing closed on malformed rows."""
    if not path.is_file():
        blockers.append(f"missing {label}: {path}")
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    blockers.append(f"malformed {label} line {line_number}: {exc}")
                    continue
                if not isinstance(payload, dict):
                    blockers.append(f"{label} line {line_number} is not a JSON object")
                    continue
                rows.append(payload)
    except OSError as exc:
        blockers.append(f"unreadable {label}: {path} ({exc})")
    return rows


def _load_config(path: Path) -> dict[str, Any]:
    """Load a raw YAML config for roster and horizon provenance."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config root is not a mapping: {path}")
    return payload


def _planner_keys_from_config(payload: dict[str, Any]) -> tuple[str, ...]:
    """Extract the ordered planner roster from a campaign config."""
    planners = payload.get("planners")
    if not isinstance(planners, list) or not planners:
        raise ValueError("config has no planners list")
    keys: list[str] = []
    for entry in planners:
        if not isinstance(entry, dict):
            raise ValueError("planner entries must be mappings")
        key = str(entry.get("key") or entry.get("algo") or "").strip()
        if not key:
            raise ValueError(f"planner entry has no key: {entry!r}")
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError("config planner roster contains duplicate keys")
    return tuple(keys)


def _scenario_ids_from_preflight(root: Path) -> tuple[str, ...] | None:
    """Read the complete scenario inventory when the campaign persisted one."""
    candidates = (
        root / "preflight" / "validate_config.json",
        root / "preflight" / "preview_scenarios.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if path.name == "validate_config.json":
            resolved = _nested(payload, "scenario_candidates", "resolved")
            if isinstance(resolved, list):
                values = tuple(_unique(str(value) for value in resolved))
                if values:
                    return values
        scenarios = payload.get("scenarios") if isinstance(payload, dict) else None
        if isinstance(scenarios, list):
            values = tuple(
                _unique(
                    str(item.get("name"))
                    for item in scenarios
                    if isinstance(item, dict) and item.get("name")
                )
            )
            if values:
                return values
    return None


def _scenario_family(record: dict[str, Any]) -> str:
    """Resolve an explicit scenario family, with a conservative id fallback."""
    candidates = (
        record.get("scenario_family"),
        _nested(record, "scenario_params", "scenario_family"),
        _nested(record, "scenario_params", "metadata", "scenario_family"),
        _nested(record, "scenario_params", "metadata", "archetype"),
        _nested(record, "scenario_params", "metadata", "family"),
        _nested(record, "scenario_params", "family"),
        record.get("scenario_id"),
    )
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return "unknown"


def _execution_mode(*payloads: dict[str, Any]) -> str:
    """Resolve execution mode from the strongest available provenance block."""
    paths = (
        ("benchmark_availability", "execution_mode"),
        ("algorithm_metadata_contract", "planner_kinematics", "execution_mode"),
        ("algorithm_metadata", "planner_kinematics", "execution_mode"),
        ("algorithm_metadata", "adapter_impact", "execution_mode"),
    )
    for payload in payloads:
        for path in paths:
            value = _nested(payload, *path)
            text = str(value or "").strip().lower()
            if text:
                return text
    return ""


def _checkpoint_gate_is_submit_safe(checkpoint: dict[str, Any]) -> bool:
    """Accept the known submit-safe checkpoint receipt shapes.

    Older campaign packets reported ``status=ok``.  The current enforced staging
    gate reports ``mode=enforced_staged`` plus equal checked/resolved counts and
    ``submit_safe=true`` instead.  Both shapes are accepted only when their
    explicit submit-safe evidence is present.
    """
    if checkpoint.get("submit_safe") is not True:
        return False
    status = str(checkpoint.get("status") or "").strip().lower()
    if status in {"ok", "staged"}:
        return True
    checked = checkpoint.get("checked")
    resolved = checkpoint.get("resolved")
    return (
        checkpoint.get("mode") == "enforced_staged"
        and checkpoint.get("stage") is True
        and isinstance(checked, int)
        and not isinstance(checked, bool)
        and isinstance(resolved, int)
        and not isinstance(resolved, bool)
        and checked == resolved
        and checked >= 0
    )


def _run_entry_index(campaign_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index campaign summary run entries by planner key."""
    entries = campaign_summary.get("runs")
    if not isinstance(entries, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        planner = entry.get("planner")
        key = str(planner.get("key") or "").strip() if isinstance(planner, dict) else ""
        if key and key not in result:
            result[key] = entry
    return result


def _run_directory(root: Path, planner_key: str, blockers: list[str]) -> Path | None:
    """Resolve exactly one planner run directory."""
    runs_root = root / "runs"
    if not runs_root.is_dir():
        blockers.append(f"missing runs directory: {runs_root}")
        return None
    candidates = sorted(
        path
        for path in runs_root.iterdir()
        if path.is_dir() and (path.name == planner_key or path.name.startswith(f"{planner_key}__"))
    )
    if len(candidates) != 1:
        blockers.append(
            f"expected one run directory for planner {planner_key!r}, found "
            f"{[path.name for path in candidates]}"
        )
        return candidates[0] if candidates else None
    return candidates[0]


def _validate_arm_metadata(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    role: str,
    root: Path,
    expected_horizon: int,
    expected_planners: tuple[str, ...],
    expected_scenarios: tuple[str, ...] | None,
    expected_seeds: tuple[int, ...],
    expected_scenario_count: int,
    expected_scenario_matrix_hash: str,
    expected_campaign_id: str,
    config_payload: dict[str, Any] | None,
    config_path: Path | None,
    manifest: dict[str, Any],
    campaign_summary: dict[str, Any],
    matrix_summary: dict[str, Any],
    comparability: dict[str, Any],
    checkpoint: dict[str, Any],
    validate_config: dict[str, Any],
    amv_summary: dict[str, Any],
    blockers: list[str],
) -> dict[str, Any]:
    """Validate campaign-level identity and return compact provenance."""
    campaign_id = str(manifest.get("campaign_id") or "").strip()
    if campaign_id != expected_campaign_id:
        blockers.append(f"{role}: campaign_id={campaign_id!r}, expected {expected_campaign_id!r}")

    if manifest.get("schema_version") != "benchmark-camera-ready-campaign.v1":
        blockers.append(f"{role}: unsupported campaign manifest schema")

    scenario_hash = str(manifest.get("scenario_matrix_hash") or "").strip()
    if scenario_hash != expected_scenario_matrix_hash:
        blockers.append(
            f"{role}: scenario_matrix_hash={scenario_hash!r}, expected "
            f"{expected_scenario_matrix_hash!r}"
        )
    summary_campaign = campaign_summary.get("campaign")
    if isinstance(summary_campaign, dict):
        summary_hash = str(summary_campaign.get("scenario_matrix_hash") or "").strip()
        if summary_hash and summary_hash != scenario_hash:
            blockers.append(f"{role}: campaign summary scenario matrix hash drift")
        summary_id = str(summary_campaign.get("campaign_id") or "").strip()
        if summary_id and summary_id != campaign_id:
            blockers.append(f"{role}: campaign summary campaign_id drift")

    seed_policy = manifest.get("seed_policy")
    resolved_seeds = tuple(
        seed
        for seed in (
            _strict_seed(value) for value in (_nested(seed_policy, "resolved_seeds") or [])
        )
        if seed is not None
    )
    if resolved_seeds != expected_seeds:
        blockers.append(
            f"{role}: resolved seeds={list(resolved_seeds)!r}, expected {list(expected_seeds)!r}"
        )

    manifest_git = str(_nested(manifest, "git", "commit") or "").strip()
    if not manifest_git:
        blockers.append(f"{role}: campaign manifest git.commit is missing")

    mapping_hash = str(
        manifest.get("comparability_mapping_hash") or comparability.get("mapping_hash") or ""
    ).strip()
    report_mapping_hash = str(comparability.get("mapping_hash") or "").strip()
    if not mapping_hash or not report_mapping_hash:
        blockers.append(f"{role}: comparability mapping hash is missing")
    elif mapping_hash != report_mapping_hash:
        blockers.append(f"{role}: comparability mapping hash drift")

    noise_hash = str(manifest.get("observation_noise_hash") or "").strip()
    if not noise_hash:
        blockers.append(f"{role}: observation_noise_hash is missing")

    matrix_rows = matrix_summary.get("rows")
    if not isinstance(matrix_rows, list):
        blockers.append(f"{role}: reports/matrix_summary.json has no rows list")
        matrix_rows = []
    matrix_by_planner: dict[str, dict[str, Any]] = {}
    for row in matrix_rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("planner_key") or "").strip()
        if not key:
            continue
        if key in matrix_by_planner:
            blockers.append(f"{role}: duplicate matrix summary planner row {key!r}")
        matrix_by_planner[key] = row

    observed_planners = tuple(matrix_by_planner)
    if set(observed_planners) != set(expected_planners):
        blockers.append(
            f"{role}: matrix planner roster differs; observed={sorted(observed_planners)!r}, "
            f"expected={sorted(expected_planners)!r}"
        )
    for planner_key in expected_planners:
        row = matrix_by_planner.get(planner_key)
        if row is None:
            continue
        if int(row.get("scenario_count", -1) or -1) != expected_scenario_count:
            blockers.append(f"{role}: planner {planner_key!r} has wrong scenario_count")
        matrix_hash = str(row.get("scenario_matrix_hash") or "").strip()
        if matrix_hash != expected_scenario_matrix_hash:
            blockers.append(f"{role}: planner {planner_key!r} has scenario hash drift")
        row_seeds = tuple(
            seed
            for seed in (_strict_seed(value) for value in (row.get("resolved_seeds") or []))
            if seed is not None
        )
        if row_seeds != expected_seeds:
            blockers.append(f"{role}: planner {planner_key!r} has seed-policy drift")
        row_horizon = _strict_seed(row.get("horizon"))
        if row_horizon is not None and row_horizon != expected_horizon:
            blockers.append(f"{role}: planner {planner_key!r} has horizon drift")

    preflight_scenario_count = _strict_seed(validate_config.get("scenario_count"))
    if preflight_scenario_count is not None and preflight_scenario_count != expected_scenario_count:
        blockers.append(f"{role}: preflight scenario_count is not {expected_scenario_count}")
    amv_scenario_count = _strict_seed(amv_summary.get("scenario_count"))
    if amv_scenario_count is not None and amv_scenario_count != expected_scenario_count:
        blockers.append(f"{role}: AMV coverage scenario_count is not {expected_scenario_count}")

    if not _checkpoint_gate_is_submit_safe(checkpoint):
        blockers.append(f"{role}: checkpoint staging receipt does not satisfy the submit-safe gate")

    horizon_from_config = _strict_seed(config_payload.get("horizon")) if config_payload else None
    if horizon_from_config is not None and horizon_from_config != expected_horizon:
        blockers.append(f"{role}: config horizon is not {expected_horizon}")
    config_sha256 = ""
    if config_path is not None:
        if not config_path.is_file():
            blockers.append(f"{role}: config is missing: {config_path}")
        else:
            try:
                config_sha256 = _sha256(config_path)
            except OSError as exc:
                blockers.append(f"{role}: config cannot be hashed ({exc})")

    matrix_config_hashes = _unique(
        str(row.get("config_hash") or "") for row in matrix_rows if isinstance(row, dict)
    )
    manifest_config_hash = str(manifest.get("config_hash") or "").strip()
    if (
        manifest_config_hash
        and matrix_config_hashes
        and matrix_config_hashes != [manifest_config_hash]
    ):
        blockers.append(f"{role}: matrix config_hash differs from campaign manifest")

    return {
        "role": role,
        "campaign_root": str(root),
        "campaign_id": campaign_id,
        "horizon": expected_horizon,
        "scenario_matrix": str(manifest.get("scenario_matrix") or ""),
        "scenario_matrix_hash": scenario_hash,
        "resolved_seeds": list(resolved_seeds),
        "planner_keys": list(expected_planners),
        "scenario_count": expected_scenario_count,
        "expected_rows": len(expected_planners) * expected_scenario_count * len(expected_seeds),
        "manifest_git_commit": manifest_git,
        "manifest_config_hash": manifest_config_hash,
        "config_sha256": config_sha256,
        "comparability_mapping_hash": mapping_hash,
        "comparability_report_scenario_matrix_hash": str(
            comparability.get("scenario_matrix_hash") or ""
        ).strip(),
        "observation_noise_hash": noise_hash,
        "checkpoint_gate": {
            "status": checkpoint.get("status"),
            "submit_safe": checkpoint.get("submit_safe"),
            "checked": checkpoint.get("checked"),
            "resolved": checkpoint.get("resolved"),
        },
        "preflight_scenario_count": preflight_scenario_count,
        "amv_coverage_status": amv_summary.get("status"),
        "matrix_planner_row_count": len(matrix_rows),
    }


def _inspect_arm(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    role: str,
    root: Path,
    expected_horizon: int,
    expected_planners: tuple[str, ...],
    expected_scenarios: tuple[str, ...] | None,
    expected_seeds: tuple[int, ...],
    expected_rows: int,
    expected_scenario_count: int,
    expected_scenario_matrix_hash: str,
    expected_campaign_id: str,
    config_payload: dict[str, Any] | None,
    config_path: Path | None,
) -> ArmInspection:
    """Inspect one campaign root and normalize its episode rows."""
    root = root.expanduser().resolve()
    blockers: list[str] = []
    required_files = {
        "campaign_manifest": root / "campaign_manifest.json",
        "validate_config": root / "preflight" / "validate_config.json",
        "checkpoint_staging": root / "preflight" / "checkpoint_staging.json",
        "matrix_summary": root / "reports" / "matrix_summary.json",
        "comparability_matrix": root / "reports" / "comparability_matrix.json",
        "amv_coverage_summary": root / "reports" / "amv_coverage_summary.json",
        "campaign_table": root / "reports" / "campaign_table.csv",
        "campaign_summary": root / "reports" / "campaign_summary.json",
    }
    artifacts = {
        name: _read_json(path, blockers, label=name)
        for name, path in required_files.items()
        if path.suffix == ".json"
    }
    for name, path in required_files.items():
        if path.suffix != ".json" and not path.is_file():
            blockers.append(f"missing {name}: {path}")

    metadata = _validate_arm_metadata(
        role=role,
        root=root,
        expected_horizon=expected_horizon,
        expected_planners=expected_planners,
        expected_scenarios=expected_scenarios,
        expected_seeds=expected_seeds,
        expected_scenario_count=expected_scenario_count,
        expected_scenario_matrix_hash=expected_scenario_matrix_hash,
        expected_campaign_id=expected_campaign_id,
        config_payload=config_payload,
        config_path=config_path,
        manifest=artifacts["campaign_manifest"],
        campaign_summary=artifacts["campaign_summary"],
        matrix_summary=artifacts["matrix_summary"],
        comparability=artifacts["comparability_matrix"],
        checkpoint=artifacts["checkpoint_staging"],
        validate_config=artifacts["validate_config"],
        amv_summary=artifacts["amv_coverage_summary"],
        blockers=blockers,
    )

    rows: dict[Key, EpisodeCell] = {}
    duplicate_keys: list[Key] = []
    metric_missing: dict[str, list[Key]] = defaultdict(list)
    raw_row_count = 0
    run_entries = _run_entry_index(artifacts["campaign_summary"])
    for planner_key in expected_planners:
        run_dir = _run_directory(root, planner_key, blockers)
        if run_dir is None:
            continue
        entry = run_entries.get(planner_key)
        if entry is None:
            blockers.append(f"{role}: campaign summary has no run entry for {planner_key!r}")
            entry = {}
        run_summary = _read_json(
            run_dir / "summary.json",
            blockers,
            label=f"{role} {planner_key} run summary",
        )
        entry_summary = entry.get("summary") if isinstance(entry, dict) else {}
        if not isinstance(entry_summary, dict):
            entry_summary = {}
        summary_for_status = {**entry_summary, **run_summary}
        availability = summary_for_status.get("benchmark_availability")
        availability_status = (
            str(availability.get("availability_status") if isinstance(availability, dict) else "")
            .strip()
            .lower()
        )
        readiness_status = (
            str(availability.get("readiness_status") if isinstance(availability, dict) else "")
            .strip()
            .lower()
        )
        if availability_status and availability_status != "available":
            blockers.append(
                f"{role}: planner {planner_key!r} availability_status={availability_status!r}"
            )
        if readiness_status and readiness_status not in VALID_EXECUTION_MODES:
            blockers.append(
                f"{role}: planner {planner_key!r} readiness_status={readiness_status!r}"
            )
        if summary_for_status.get("benchmark_success") is False:
            blockers.append(f"{role}: planner {planner_key!r} benchmark_success=false")
        for count_name in ("failed_jobs", "skipped_jobs"):
            count = _finite_float(summary_for_status.get(count_name))
            if count is not None and count > 0:
                blockers.append(f"{role}: planner {planner_key!r} has {count_name}={int(count)}")

        episodes_path = run_dir / "episodes.jsonl"
        episode_rows = _read_jsonl(
            episodes_path,
            blockers,
            label=f"{role} {planner_key} episodes",
        )
        raw_row_count += len(episode_rows)
        if len(episode_rows) != expected_scenario_count * len(expected_seeds):
            blockers.append(
                f"{role}: planner {planner_key!r} has {len(episode_rows)} rows, expected "
                f"{expected_scenario_count * len(expected_seeds)}"
            )

        run_modes: set[str] = set()
        fallback_rows: list[str] = []
        for record in episode_rows:
            scenario_id = str(record.get("scenario_id") or "").strip()
            seed = _strict_seed(record.get("seed"))
            if not scenario_id or seed is None:
                blockers.append(f"{role}: {planner_key!r} has a row with missing scenario_id/seed")
                continue
            key = (planner_key, scenario_id, seed)
            summary_mode = _execution_mode(summary_for_status)
            record_mode = _execution_mode(record)
            if summary_mode and record_mode and summary_mode != record_mode:
                blockers.append(
                    f"{role}: {_key_text(key)} execution mode {record_mode!r} "
                    f"disagrees with run summary {summary_mode!r}"
                )
            mode = record_mode or summary_mode
            run_modes.add(mode)
            if mode not in VALID_EXECUTION_MODES:
                fallback_rows.append(_key_text(key))
            if expected_scenarios is not None and scenario_id not in expected_scenarios:
                blockers.append(f"{role}: unexpected scenario_id={scenario_id!r}")
            if seed not in expected_seeds:
                blockers.append(f"{role}: unexpected seed={seed!r} for planner {planner_key!r}")
            row_horizon = _strict_seed(_nested(record, "scenario_params", "run_horizon"))
            if row_horizon is not None and row_horizon != expected_horizon:
                blockers.append(
                    f"{role}: {_key_text(key)} has run_horizon={row_horizon}, "
                    f"expected {expected_horizon}"
                )
            if key in rows:
                duplicate_keys.append(key)
                continue
            values: dict[str, float] = {}
            for metric in METRICS:
                value = _finite_float(episode_metric_value(record, metric))
                if value is None:
                    metric_missing[metric].append(key)
                else:
                    values[metric] = value
            rows[key] = EpisodeCell(
                key=key,
                scenario_family=_scenario_family(record),
                metrics=values,
                execution_mode=mode,
            )
        if len(run_modes) != 1:
            blockers.append(
                f"{role}: planner {planner_key!r} has execution modes {sorted(run_modes)!r}"
            )
        if fallback_rows:
            blockers.append(
                f"{role}: planner {planner_key!r} has non-native/adapter rows "
                f"({fallback_rows[:5]!r})"
            )

    if len(rows) != expected_rows:
        blockers.append(f"{role}: unique row count={len(rows)}, expected {expected_rows}")
    if duplicate_keys:
        blockers.append(f"{role}: duplicate matched keys={len(duplicate_keys)}")
    for metric, keys in metric_missing.items():
        if keys:
            blockers.append(f"{role}: metric {metric!r} missing for {len(keys)} rows")

    metadata.update(
        {
            "row_count": raw_row_count,
            "unique_key_count": len(rows),
            "duplicate_key_count": len(duplicate_keys),
            "execution_modes": sorted(
                {cell.execution_mode for cell in rows.values() if cell.execution_mode}
            ),
            "metric_missing_counts": {
                metric: len(keys) for metric, keys in sorted(metric_missing.items()) if keys
            },
        }
    )
    return ArmInspection(
        role=role,
        root=root,
        expected_horizon=expected_horizon,
        expected_planners=expected_planners,
        expected_scenarios=expected_scenarios,
        expected_seeds=expected_seeds,
        expected_rows=expected_rows,
        metadata=metadata,
        rows=rows,
        duplicate_keys=sorted(set(duplicate_keys)),
        metric_missing={metric: sorted(set(keys)) for metric, keys in metric_missing.items()},
        blockers=_unique(blockers),
    )


def _expected_keys(
    planners: Sequence[str],
    scenarios: Sequence[str] | None,
    seeds: Sequence[int],
) -> set[Key] | None:
    """Build the declared identity denominator when scenario IDs are known."""
    if scenarios is None:
        return None
    return {
        (planner, scenario, seed)
        for planner in planners
        for scenario in scenarios
        for seed in seeds
    }


def _key_completeness(
    arm: ArmInspection,
    expected_keys: set[Key] | None,
) -> dict[str, Any]:
    """Return machine-readable missing, extra, and duplicate identity details."""
    observed = set(arm.rows)
    expected = expected_keys or set()
    return {
        "raw_row_count": arm.metadata.get("row_count", 0),
        "unique_key_count": len(observed),
        "expected_key_count": len(expected) if expected_keys is not None else arm.expected_rows,
        "missing_keys": [_key_payload(key) for key in sorted(expected - observed)],
        "extra_keys": [_key_payload(key) for key in sorted(observed - expected)]
        if expected_keys is not None
        else [],
        "duplicate_keys": [_key_payload(key) for key in arm.duplicate_keys],
        "metric_missing_counts": arm.metadata.get("metric_missing_counts", {}),
    }


def _pair_blockers(  # noqa: C901
    h500: ArmInspection,
    h600: ArmInspection,
    *,
    expected_keys: set[Key] | None,
    expected_horizon_pair: tuple[int, int],
    pair_validation: dict[str, Any] | None,
) -> list[str]:
    """Collect all cross-arm blockers without weakening arm-level failures."""
    blockers = [*h500.blockers, *h600.blockers]
    if pair_validation is not None and not pair_validation.get("is_valid", False):
        blockers.append("config pair validation failed")
        blockers.extend(str(value) for value in pair_validation.get("mismatches", []))
    if (h500.expected_horizon, h600.expected_horizon) != expected_horizon_pair:
        blockers.append("horizon roles do not match the declared h500/h600 pair")

    for field in (
        "scenario_matrix_hash",
        "manifest_git_commit",
        "comparability_mapping_hash",
        "observation_noise_hash",
    ):
        left = h500.metadata.get(field)
        right = h600.metadata.get(field)
        if not left or not right or left != right:
            blockers.append(f"cross-arm provenance drift: {field}")
    if h500.expected_planners != h600.expected_planners:
        blockers.append("cross-arm planner roster order differs")
    if h500.expected_seeds != h600.expected_seeds:
        blockers.append("cross-arm seed order differs")
    if expected_keys is not None:
        for label, arm in (("h500", h500), ("h600", h600)):
            observed = set(arm.rows)
            missing = expected_keys - observed
            extra = observed - expected_keys
            if missing:
                blockers.append(f"{label}: missing {len(missing)} declared matched keys")
            if extra:
                blockers.append(f"{label}: has {len(extra)} undeclared matched keys")
    if set(h500.rows) != set(h600.rows):
        blockers.append("h500 and h600 matched-key sets differ")
    for metric in METRICS:
        if h500.metadata.get("metric_missing_counts", {}).get(metric) or h600.metadata.get(
            "metric_missing_counts", {}
        ).get(metric):
            blockers.append(f"metric completeness failed: {metric}")
    return _unique(blockers)


def _point_estimates(
    rows: Sequence[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Aggregate paired arm values without uncertainty or ranking claims."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field) or "") for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items()):
        payload = dict(zip(group_fields, group_key, strict=True))
        payload["paired_row_count"] = len(group_rows)
        payload["metrics"] = {}
        for metric in METRICS:
            h500_values = [float(row["metrics"][metric]["h500"]) for row in group_rows]
            h600_values = [float(row["metrics"][metric]["h600"]) for row in group_rows]
            deltas = [float(row["metrics"][metric]["delta_h600_minus_h500"]) for row in group_rows]
            payload["metrics"][metric] = {
                "h500_mean": float(np.mean(h500_values)),
                "h600_mean": float(np.mean(h600_values)),
                "delta_mean_h600_minus_h500": float(np.mean(deltas)),
            }
        output.append(payload)
    return output


def _bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Compute a deterministic percentile bootstrap interval over seed means."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty sequence")
    if array.size == 1 or samples <= 0:
        mean = float(np.mean(array))
        return mean, mean
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(int(samples), array.size), replace=True).mean(axis=1)
    alpha = min(0.999999, max(0.0, float(confidence)))
    low = float(np.quantile(sampled, (1.0 - alpha) / 2.0, method="linear"))
    high = float(np.quantile(sampled, 1.0 - (1.0 - alpha) / 2.0, method="linear"))
    return low, high


def _uncertainty_rows(
    rows: Sequence[dict[str, Any]],
    group_fields: tuple[str, ...],
    *,
    confidence: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Compute paired delta uncertainty from per-seed means."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field) or "") for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items()):
        payload = dict(zip(group_fields, group_key, strict=True))
        payload["paired_row_count"] = len(group_rows)
        payload["metrics"] = {}
        for metric_index, metric in enumerate(METRICS):
            by_seed: dict[int, list[float]] = defaultdict(list)
            for row in group_rows:
                by_seed[int(row["seed"])].append(
                    float(row["metrics"][metric]["delta_h600_minus_h500"])
                )
            seed_means = [
                {"seed": seed, "mean_delta": float(np.mean(values))}
                for seed, values in sorted(by_seed.items())
            ]
            values = [entry["mean_delta"] for entry in seed_means]
            status = "supported" if len(values) >= 2 else "insufficient_seed_support"
            if values:
                mean_delta = float(np.mean(values))
                ci_low, ci_high = _bootstrap_ci(
                    values,
                    confidence=confidence,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + metric_index,
                )
                std = float(np.std(np.asarray(values, dtype=float), ddof=0))
            else:  # pragma: no cover - matched rows always provide values.
                mean_delta = None
                ci_low = None
                ci_high = None
                std = None
            payload["metrics"][metric] = {
                "status": status,
                "seed_count": len(seed_means),
                "seed_means": seed_means,
                "mean_delta_h600_minus_h500": mean_delta,
                "std_across_seed_means": std,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        output.append(payload)
    return output


def _blocked_payload(schema_version: str, blockers: Sequence[str], **extra: Any) -> dict[str, Any]:
    """Build a blocked artifact without fake numeric placeholders."""
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "status": "blocked",
        "benchmark_success_allowed": False,
        "claim_boundary": (
            "No numeric horizon comparison is valid until all declared paired keys, "
            "provenance, and native/adapter execution gates pass."
        ),
        "blockers": list(_unique(str(value) for value in blockers)),
    }
    payload.update(extra)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON with finite-number enforcement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def analyze_pair(  # noqa: C901, PLR0912, PLR0913, PLR0915
    h500_root: Path,
    h600_root: Path,
    *,
    output_dir: Path,
    h500_config: Path | None = DEFAULT_H500_CONFIG,
    h600_config: Path | None = DEFAULT_H600_CONFIG,
    expected_planners: Sequence[str] | None = None,
    expected_scenarios: Sequence[str] | None = None,
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
    expected_rows_per_arm: int = DEFAULT_ROWS_PER_ARM,
    expected_scenario_count: int = DEFAULT_SCENARIO_COUNT,
    expected_scenario_matrix_hash: str = DEFAULT_SCENARIO_MATRIX_HASH,
    expected_horizons: tuple[int, int] = (500, 600),
    expected_campaign_ids: tuple[str, str] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    validate_config_pair: bool = True,
) -> dict[str, Any]:
    """Build all three issue #5409 handoff artifacts and return their statuses."""
    seeds = tuple(int(seed) for seed in expected_seeds)
    campaign_ids = tuple(expected_campaign_ids or DEFAULT_CAMPAIGN_IDS)
    if len(campaign_ids) != 2:
        raise ValueError("expected_campaign_ids must contain exactly two values: h500 and h600")
    scenario_ids = tuple(str(value) for value in expected_scenarios) if expected_scenarios else None
    config_payloads: list[dict[str, Any] | None] = []
    config_paths = [
        path.expanduser().resolve() if path is not None else None
        for path in (h500_config, h600_config)
    ]
    config_errors: list[str] = []
    for path in config_paths:
        if path is None:
            config_payloads.append(None)
            continue
        try:
            config_payloads.append(_load_config(path.resolve()))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            config_payloads.append(None)
            config_errors.append(f"config load failed for {path}: {exc}")

    if expected_planners is None:
        roster_candidates = [
            _planner_keys_from_config(payload) for payload in config_payloads if payload is not None
        ]
        if roster_candidates:
            expected_planner_tuple = roster_candidates[0]
        else:
            expected_planner_tuple = ()
    else:
        expected_planner_tuple = tuple(str(value) for value in expected_planners)
    if not expected_planner_tuple:
        config_errors.append("no expected planner roster is available")
    for role, payload in zip(("h500", "h600"), config_payloads, strict=True):
        if payload is None:
            continue
        try:
            config_roster = _planner_keys_from_config(payload)
        except ValueError as exc:
            config_errors.append(f"{role}: planner roster could not be read ({exc})")
        else:
            if config_roster != expected_planner_tuple:
                config_errors.append(f"{role}: config planner roster differs from expected roster")

    if scenario_ids is None:
        scenario_ids = _scenario_ids_from_preflight(h500_root.resolve())
        if scenario_ids is None:
            scenario_ids = _scenario_ids_from_preflight(h600_root.resolve())
    if scenario_ids is not None and len(scenario_ids) != expected_scenario_count:
        config_errors.append(
            f"scenario inventory has {len(scenario_ids)} IDs, expected {expected_scenario_count}"
        )

    pair_validation: dict[str, Any] | None = None
    if validate_config_pair and config_paths[0] is not None and config_paths[1] is not None:
        if validate_horizon_ablation_pair is None:
            config_errors.append("horizon pair validator could not be imported")
        else:
            pair_validation = validate_horizon_ablation_pair(
                config_paths[0], config_paths[1]
            ).to_payload()

    h500 = _inspect_arm(
        role="h500",
        root=h500_root,
        expected_horizon=expected_horizons[0],
        expected_planners=expected_planner_tuple,
        expected_scenarios=scenario_ids,
        expected_seeds=seeds,
        expected_rows=expected_rows_per_arm,
        expected_scenario_count=expected_scenario_count,
        expected_scenario_matrix_hash=expected_scenario_matrix_hash,
        expected_campaign_id=campaign_ids[0],
        config_payload=config_payloads[0],
        config_path=config_paths[0],
    )
    h600 = _inspect_arm(
        role="h600",
        root=h600_root,
        expected_horizon=expected_horizons[1],
        expected_planners=expected_planner_tuple,
        expected_scenarios=scenario_ids,
        expected_seeds=seeds,
        expected_rows=expected_rows_per_arm,
        expected_scenario_count=expected_scenario_count,
        expected_scenario_matrix_hash=expected_scenario_matrix_hash,
        expected_campaign_id=campaign_ids[1],
        config_payload=config_payloads[1],
        config_path=config_paths[1],
    )
    expected_keys = _expected_keys(expected_planner_tuple, scenario_ids, seeds)
    blockers = [*config_errors]
    blockers.extend(
        _pair_blockers(
            h500,
            h600,
            expected_keys=expected_keys,
            expected_horizon_pair=expected_horizons,
            pair_validation=pair_validation,
        )
    )
    if expected_rows_per_arm != len(expected_planner_tuple) * expected_scenario_count * len(seeds):
        blockers.append(
            "expected_rows_per_arm does not equal planner_count * scenario_count * seed_count"
        )
    blockers = _unique(blockers)
    ready = not blockers

    completeness: dict[str, Any] = {
        "schema_version": COMPLETENESS_SCHEMA,
        "status": "ready" if ready else "blocked",
        "benchmark_success_allowed": ready,
        "claim_boundary": (
            "Nominal fixed h500-vs-h600 comparison only; this handoff is not paper-grade "
            "evidence and does not by itself establish a horizon finding."
        ),
        "expected": {
            "planner_count": len(expected_planner_tuple),
            "planner_keys": list(expected_planner_tuple),
            "scenario_count": expected_scenario_count,
            "scenario_ids": list(scenario_ids) if scenario_ids is not None else None,
            "resolved_seeds": list(seeds),
            "rows_per_horizon": expected_rows_per_arm,
            "rows_total": expected_rows_per_arm * 2,
            "matched_key": ["planner_key", "scenario_id", "seed"],
            "scenario_matrix_hash": expected_scenario_matrix_hash,
            "horizons": {"h500": expected_horizons[0], "h600": expected_horizons[1]},
            "campaign_ids": {"h500": campaign_ids[0], "h600": campaign_ids[1]},
        },
        "arms": {
            "h500": {
                "metadata": h500.metadata,
                "key_completeness": _key_completeness(h500, expected_keys),
                "blockers": h500.blockers,
            },
            "h600": {
                "metadata": h600.metadata,
                "key_completeness": _key_completeness(h600, expected_keys),
                "blockers": h600.blockers,
            },
        },
        "pair_validation": pair_validation,
        "blockers": blockers,
    }
    _write_json(output_dir / "matched_key_completeness.json", completeness)

    if not ready:
        deltas = _blocked_payload(
            DELTA_SCHEMA,
            blockers,
            expected_rows_per_horizon=expected_rows_per_arm,
            rows=[],
        )
        uncertainty = _blocked_payload(
            UNCERTAINTY_SCHEMA,
            blockers,
            confidence={
                "method": "bootstrap_mean_over_seed_means",
                "confidence": confidence,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed,
            },
            planner_rows=[],
            scenario_family_rows=[],
        )
    else:
        paired_rows: list[dict[str, Any]] = []
        for key in sorted(h500.rows):
            left = h500.rows[key]
            right = h600.rows[key]
            metrics = {
                metric: {
                    "h500": left.metrics[metric],
                    "h600": right.metrics[metric],
                    "delta_h600_minus_h500": right.metrics[metric] - left.metrics[metric],
                }
                for metric in METRICS
            }
            paired_rows.append(
                {
                    **_key_payload(key),
                    "scenario_family": left.scenario_family,
                    "execution_mode_h500": left.execution_mode,
                    "execution_mode_h600": right.execution_mode,
                    "metrics": metrics,
                }
            )
        deltas = {
            "schema_version": DELTA_SCHEMA,
            "status": "ready",
            "benchmark_success_allowed": True,
            "claim_boundary": completeness["claim_boundary"],
            "provenance": {
                "h500": h500.metadata,
                "h600": h600.metadata,
                "matched_key": ["planner_key", "scenario_id", "seed"],
            },
            "rows": paired_rows,
            "planner_point_estimates": _point_estimates(paired_rows, ("planner_key",)),
            "scenario_family_point_estimates": _point_estimates(
                paired_rows, ("planner_key", "scenario_family")
            ),
        }
        uncertainty = {
            "schema_version": UNCERTAINTY_SCHEMA,
            "status": "ready",
            "benchmark_success_allowed": True,
            "claim_boundary": completeness["claim_boundary"],
            "provenance": {
                "h500": h500.metadata,
                "h600": h600.metadata,
                "matched_key": ["planner_key", "scenario_id", "seed"],
            },
            "confidence": {
                "method": "bootstrap_mean_over_seed_means",
                "confidence": confidence,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed,
            },
            "planner_rows": _uncertainty_rows(
                paired_rows,
                ("planner_key",),
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "scenario_family_rows": _uncertainty_rows(
                paired_rows,
                ("planner_key", "scenario_family"),
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
        }
    _write_json(output_dir / "paired_horizon_deltas.json", deltas)
    _write_json(output_dir / "paired_uncertainty_summary.json", uncertainty)
    return {
        "status": completeness["status"],
        "benchmark_success_allowed": completeness["benchmark_success_allowed"],
        "output_dir": str(output_dir.expanduser().resolve()),
        "artifacts": {
            "matched_key_completeness": str(
                (output_dir / "matched_key_completeness.json").resolve()
            ),
            "paired_horizon_deltas": str((output_dir / "paired_horizon_deltas.json").resolve()),
            "paired_uncertainty_summary": str(
                (output_dir / "paired_uncertainty_summary.json").resolve()
            ),
        },
        "blocker_count": len(blockers),
        "blockers": blockers,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h500-root", type=Path, required=True)
    parser.add_argument("--h600-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--h500-config", type=Path, default=DEFAULT_H500_CONFIG)
    parser.add_argument("--h600-config", type=Path, default=DEFAULT_H600_CONFIG)
    parser.add_argument("--expected-rows-per-arm", type=int, default=DEFAULT_ROWS_PER_ARM)
    parser.add_argument("--expected-scenario-count", type=int, default=DEFAULT_SCENARIO_COUNT)
    parser.add_argument(
        "--expected-seed",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        dest="expected_seeds",
    )
    parser.add_argument(
        "--expected-scenario-matrix-hash",
        default=DEFAULT_SCENARIO_MATRIX_HASH,
    )
    parser.add_argument(
        "--expected-campaign-id",
        nargs=2,
        metavar=("H500_ID", "H600_ID"),
        default=None,
        dest="expected_campaign_ids",
        help=(
            "Expected campaign IDs for the h500 and h600 arms. Use this for a "
            "reviewed rerun suffix without weakening other provenance checks."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--skip-config-pair-validation", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the paired analysis and return a fail-closed exit code."""
    args = _build_parser().parse_args(argv)
    result = analyze_pair(
        args.h500_root,
        args.h600_root,
        output_dir=args.output_dir,
        h500_config=args.h500_config,
        h600_config=args.h600_config,
        expected_seeds=args.expected_seeds,
        expected_rows_per_arm=args.expected_rows_per_arm,
        expected_scenario_count=args.expected_scenario_count,
        expected_scenario_matrix_hash=args.expected_scenario_matrix_hash,
        expected_campaign_ids=tuple(args.expected_campaign_ids)
        if args.expected_campaign_ids is not None
        else None,
        confidence=args.confidence,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        validate_config_pair=not args.skip_config_pair_validation,
    )
    if args.json_output:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"issue #5409 paired analysis: {result['status']}")
        for blocker in result.get("blockers", []):
            print(f"- {blocker}", file=sys.stderr)
        for path in result["artifacts"].values():
            print(path)
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
