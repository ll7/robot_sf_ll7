#!/usr/bin/env python3
"""Validate heterogeneous-population shards or build their final comparison report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.heterogeneous_population_ablation import (
    assess_mean_matched_episode_records,
    build_per_archetype_ablation_report,
)
from robot_sf.benchmark.heterogeneous_population_metrics import (
    cvar,
    pedestrian_metric_observations_from_control_trace,
)
from robot_sf.benchmark.heterogeneous_rank_sensitivity import (
    compute_bootstrap_rank_sensitivity,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = "output/issue_3574_mean_matched_harness/manifest.json"
DEFAULT_RECORDS = "output/issue_3574_mean_matched_harness/episode_records.jsonl"
DEFAULT_OUTPUT = "output/issue_3574_mean_matched_harness"
DEFAULT_DURABLE = "output/issue_3574_mean_matched_harness/durable_evidence"
COMBINED_REPORT_MIN_PLANNERS = 2
SHARD_RECEIPT_SCHEMA = "heterogeneous_population_ablation_shard_receipt.v3"
FINALIZATION_PROVENANCE_SCHEMA = "heterogeneous_population_ablation_finalization.v3"
SHARD_CLAIM_BOUNDARY = (
    "This receipt validates one planner shard against the paired manifest. "
    "It makes no cross-planner rank, comparison, or paper-grade claim."
)
MANIFEST_COMPATIBILITY_FIELDS = (
    "schema_version",
    "issue",
    "paired_arms",
    "scenario_rows",
    "planner_rows",
    "seed_rows",
    "response_law_fractions",
    "trace_metric_keys",
    "expected_episode_output_keys",
)
PER_ARCHETYPE_METRIC_HIGHER_IS_SAFER = {
    "clearance_m": True,
    "near_field_exposure_s": False,
}

# Test-only seam: replacement after open must not change the bytes hashed and parsed from that fd.
_AFTER_SOURCE_OPEN_HOOK: Callable[[Path, str], None] | None = None


def metric_higher_is_safer(metric_key: str) -> bool:
    """Return the declared safety direction or fail closed for an unknown trace metric."""
    try:
        return PER_ARCHETYPE_METRIC_HIGHER_IS_SAFER[metric_key]
    except KeyError as exc:
        raise ValueError(
            f"No higher_is_safer direction is declared for trace metric {metric_key!r}"
        ) from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--durable-dir", default=DEFAULT_DURABLE)
    parser.add_argument(
        "--mode",
        choices=("combined", "shard", "finalize"),
        default="combined",
        help=(
            "combined: direct cross-planner report; shard: one non-comparative receipt; "
            "finalize: verify repeated --shard-receipt inputs and compare them."
        ),
    )
    parser.add_argument(
        "--shard-receipt",
        action="append",
        default=[],
        help="Repeat for each independently validated planner when --mode finalize is used.",
    )
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        help=(
            "Controller-supplied root containing relocated receipt source bundles. Repeat to "
            "search multiple roots; SHA-256 still decides the accepted source."
        ),
    )
    parser.add_argument(
        "--source-artifact-head",
        "--expected-head",
        dest="source_artifact_head",
        help=(
            "Executing git commit for shard provenance. When omitted, an explicit manifest head "
            "is used, otherwise the active repository HEAD is captured."
        ),
    )
    return parser.parse_args()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _invoke_open_hook(path: Path, label: str) -> None:
    if _AFTER_SOURCE_OPEN_HOOK is not None:
        _AFTER_SOURCE_OPEN_HOOK(path, label)


def _portable_identity(
    path: Path,
    *,
    argument: str,
    bundle_root: Path,
    size_bytes: int,
    sha256: str,
) -> dict[str, Any]:
    resolved = path.resolve()
    root = bundle_root.resolve()
    try:
        portable_path = str(resolved.relative_to(root))
    except ValueError:
        portable_path = resolved.name
        root = resolved.parent
    return {
        "argument": argument,
        "portable_path": portable_path,
        "source_root_hint": str(root),
        "resolved_path": str(resolved),
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def _read_json_source(
    path: Path, *, argument: str, bundle_root: Path, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash and parse JSON from the same opened file descriptor."""
    with path.open("rb") as source:
        _invoke_open_hook(path, label)
        data = source.read()
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    identity = _portable_identity(
        path,
        argument=argument,
        bundle_root=bundle_root,
        size_bytes=len(data),
        sha256=_sha256_bytes(data),
    )
    return payload, identity


def _hash_source(path: Path, *, argument: str, bundle_root: Path, label: str) -> dict[str, Any]:
    """Hash a provenance-only source through one stable opened descriptor."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        _invoke_open_hook(path, label)
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return _portable_identity(
        path,
        argument=argument,
        bundle_root=bundle_root,
        size_bytes=size,
        sha256=digest.hexdigest(),
    )


def _rewrite_manifest_status(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Persist captured-runtime status before the manifest is bound into a receipt."""
    manifest["status"] = "ready"
    manifest["claim_boundary"] = "captured_runtime_ready"
    try:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise OSError(f"Failed to rewrite manifest at {manifest_path}") from exc


def _normalized_cell_key(row: dict[str, Any]) -> str:
    """Canonicalize a full manifest row after removing only planner identity."""
    normalized = dict(row)
    normalized.pop("planner", None)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _planner_artifact_key(planner: str) -> str:
    """Reject path traversal, normalized aliases, separators, and unsafe components."""
    normalized = os.path.normpath(planner)
    if (
        planner in {".", ".."}
        or Path(planner).is_absolute()
        or normalized != planner
        or "/" in planner
        or "\\" in planner
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", planner)
    ):
        raise ValueError(f"unsafe planner artifact component: {planner!r}")
    return planner


def _manifest_planner_order(manifest: dict[str, Any]) -> list[str]:
    return [
        str(row.get("planner", row.get("key")))
        for row in manifest.get("planner_rows", [])
        if isinstance(row, dict) and ("planner" in row or "key" in row)
    ]


def _csv_row(record: dict[str, Any]) -> dict[str, Any]:
    trace = record["algorithm_metadata"]["pedestrian_control_trace"]
    mean_value = ""
    cvar_value = ""
    try:
        observations = pedestrian_metric_observations_from_control_trace(
            trace, "clearance_m", reducer="mean"
        )
        values = [observation.value for observation in observations]
        if values:
            mean_value = f"{sum(values) / len(values):.4f}"
            cvar_value = f"{cvar(values, 0.2, higher_is_safer=True):.4f}"
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        pass
    fraction_value = record.get("response_law_fraction")
    return {
        "scenario_id": record["scenario_id"],
        "seed": int(record["seed"]),
        "planner": record["planner"],
        "arm": record["population_arm"],
        "response_law_fraction": float(0.0 if fraction_value is None else fraction_value),
        "mean_clearance_m": mean_value,
        "cvar_clearance_m": cvar_value,
    }


def _build_per_archetype_reports(
    records: list[dict[str, Any]], metric_keys: list[str]
) -> dict[str, dict[str, Any]]:
    """Use the established paired-arm report builder for each campaign group."""
    groups: dict[tuple[str, int, str, float], dict[str, dict[str, Any]]] = {}
    for record in records:
        fraction_value = record.get("response_law_fraction")
        fraction = float(0.0 if fraction_value is None else fraction_value)
        key = (
            str(record["scenario_id"]),
            int(record["seed"]),
            str(record["planner"]),
            fraction,
        )
        groups.setdefault(key, {})[str(record["population_arm"])] = record["algorithm_metadata"][
            "pedestrian_control_trace"
        ]

    reports: dict[str, dict[str, Any]] = {}
    for metric_key in metric_keys:
        metric_reports: dict[str, Any] = {}
        for (scenario, seed, planner, fraction), traces_by_arm in groups.items():
            if {
                "heterogeneous",
                "mean_matched_homogeneous",
            }.issubset(traces_by_arm):
                metric_reports[
                    f"{scenario}/seed_{seed}/{planner}/response_law_fraction_{fraction:g}"
                ] = build_per_archetype_ablation_report(
                    control_traces_by_arm=traces_by_arm,
                    metric_key=metric_key,
                    higher_is_safer=metric_higher_is_safer(metric_key),
                    cvar_alpha=0.2,
                    reducer="mean",
                )
        reports[metric_key] = metric_reports
    return reports


def _reduced_rank_record(record: dict[str, Any]) -> dict[str, Any]:
    """Retain only fields consumed by the unchanged rank-sensitivity implementation."""
    return {
        "scenario_id": record["scenario_id"],
        "planner": record["planner"],
        "seed": int(record["seed"]),
        "population_arm": record["population_arm"],
        "response_law_fraction": record.get("response_law_fraction"),
        "metrics": {"mean_clearance": record["metrics"]["mean_clearance"]},
    }


def _load_reduce_records(
    path: Path,
    *,
    argument: str,
    bundle_root: Path,
    manifest: dict[str, Any],
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate one complete shard, then retain only reduced report inputs."""
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as source:
        _invoke_open_hook(path, label)
        for line_number, raw_line in enumerate(source, start=1):
            digest.update(raw_line)
            size_bytes += len(raw_line)
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if not isinstance(record, dict):
                raise ValueError(f"episode_records[{line_number}] must be a mapping")
            records.append(record)

    readiness = assess_mean_matched_episode_records(manifest, records)
    metric_keys = [str(key) for key in manifest.get("trace_metric_keys", [])]
    ready_records = records if readiness["ready"] else []
    reduced = {
        "integration_readiness": readiness,
        "rank_records": [_reduced_rank_record(record) for record in ready_records],
        "csv_rows": [_csv_row(record) for record in ready_records],
        "per_archetype_metric_reports": _build_per_archetype_reports(ready_records, metric_keys),
        "planner_order": _manifest_planner_order(manifest),
        "normalized_cells": {
            _normalized_cell_key(row)
            for row in manifest.get("manifest_rows", [])
            if isinstance(row, dict)
        },
        "record_count": len(records),
        "streaming_stats": {
            "processing_model": "sequential_whole_shard",
            "max_live_raw_shards": 1 if records else 0,
            "max_live_raw_records": len(records),
            "raw_shards_retained": 0,
            "raw_records_retained": 0,
            "reduced_rank_record_count": len(ready_records),
        },
    }
    identity = _portable_identity(
        path,
        argument=argument,
        bundle_root=bundle_root,
        size_bytes=size_bytes,
        sha256=digest.hexdigest(),
    )
    return reduced, identity


def _config_identity(manifest: dict[str, Any]) -> dict[str, Any] | None:
    raw_path = manifest.get("config_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = _resolve_path(raw_path)
    if not path.is_file():
        return {
            "argument": raw_path,
            "portable_path": raw_path,
            "source_root_hint": str(REPO_ROOT),
            "resolved_path": str(path),
            "available": False,
        }
    return {
        **_hash_source(
            path,
            argument=raw_path,
            bundle_root=REPO_ROOT,
            label="config",
        ),
        "available": True,
    }


def _manifest_head(manifest: dict[str, Any]) -> str | None:
    for key in ("expected_head", "git_head", "commit_sha", "source_head"):
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _source_artifact_head(manifest: dict[str, Any], explicit: str | None) -> str:
    """Resolve campaign provenance without comparing it to the recovery checkout."""
    manifest_head = _manifest_head(manifest)
    if explicit and manifest_head and explicit != manifest_head:
        raise ValueError("--source-artifact-head does not match manifest head")
    source_head = explicit or manifest_head
    if not source_head:
        raise ValueError(
            "shard mode requires a nonempty source artifact head in the manifest "
            "or --source-artifact-head"
        )
    return source_head


def _receipt_builder_head() -> str:
    """Capture recovery-code provenance independently from source provenance."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("receipt builder git HEAD cannot be captured") from exc


def _write_json(destinations: Iterable[Path], filename: str, payload: dict[str, Any]) -> None:
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _write_csv(destinations: Iterable[Path], rows: list[dict[str, Any]]) -> None:
    fields = [
        "scenario_id",
        "seed",
        "planner",
        "arm",
        "response_law_fraction",
        "mean_clearance_m",
        "cvar_clearance_m",
    ]
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "ablation_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def _receipt_command_identity(
    *,
    planner: str,
    source_artifact_head: str,
    receipt_builder_head: str,
    manifest_identity: dict[str, Any],
    records_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "mode": "shard",
        "schema_version": SHARD_RECEIPT_SCHEMA,
        "planner": planner,
        "source_artifact_head": source_artifact_head,
        "receipt_builder_head": receipt_builder_head,
        "manifest_sha256": manifest_identity["sha256"],
        "records_sha256": records_identity["sha256"],
    }


def _build_shard_receipt(
    *,
    planner: str,
    source_artifact_head: str,
    receipt_builder_head: str,
    manifest: dict[str, Any],
    manifest_identity: dict[str, Any],
    records_identity: dict[str, Any],
    reduced: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SHARD_RECEIPT_SCHEMA,
        "status": "validated",
        "artifact_kind": "single_planner_shard_validation_receipt",
        "evidence_status": "diagnostic-only",
        "claim_boundary": SHARD_CLAIM_BOUNDARY,
        "planner": planner,
        "planner_count": 1,
        "episode_record_count": reduced["record_count"],
        "integration_readiness": reduced["integration_readiness"],
        "normalized_cell_count": len(reduced["normalized_cells"]),
        "per_archetype_metric_report_counts": {
            key: len(value) for key, value in reduced["per_archetype_metric_reports"].items()
        },
        "streaming_stats": reduced["streaming_stats"],
        "cross_planner_rank_comparison": {
            "status": "not_run",
            "claim_made": False,
            "reason": "single_planner_shard",
            "minimum_planners_for_final_combined_report": COMBINED_REPORT_MIN_PLANNERS,
        },
        "provenance": {
            "manifest": manifest_identity,
            "records": records_identity,
            "config": _config_identity(manifest),
            "source_artifact_head": source_artifact_head,
            "receipt_builder_head": receipt_builder_head,
            "command_identity": _receipt_command_identity(
                planner=planner,
                source_artifact_head=source_artifact_head,
                receipt_builder_head=receipt_builder_head,
                manifest_identity=manifest_identity,
                records_identity=records_identity,
            ),
            "raw_invocation": {
                "executable": sys.executable,
                "argv": list(sys.argv),
            },
        },
    }


def _receipt_markdown(receipt: dict[str, Any]) -> str:
    return f"""# Single-Planner Shard Validation Receipt

## Claim Boundary

- Evidence status: `{receipt["evidence_status"]}`.
- Planner: `{receipt["planner"]}`.
- No cross-planner rank or comparison claim was made.
- This is not paper-grade evidence.

## Validation Result

- Status: `{receipt["status"]}`.
- Episode records: {receipt["episode_record_count"]}.
- Expected cells: {receipt["integration_readiness"]["expected_row_count"]}.
- Source artifact head: `{receipt["provenance"]["source_artifact_head"]}`.
- Receipt builder head: `{receipt["provenance"]["receipt_builder_head"]}`.
"""


def _candidate_paths(identity: dict[str, Any], source_roots: list[Path]) -> list[Path]:
    portable = Path(str(identity["portable_path"]))
    candidates = [
        *(root / portable for root in source_roots),
        Path(str(identity["source_root_hint"])) / portable,
        Path(str(identity["resolved_path"])),
    ]
    unique: list[Path] = []
    for candidate in sorted({path.resolve() for path in candidates}):
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _identity_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return actual["size_bytes"] == expected.get("size_bytes") and actual["sha256"] == expected.get(
        "sha256"
    )


def _verified_manifest_source(
    expected: dict[str, Any], source_roots: list[Path]
) -> tuple[dict[str, Any], dict[str, Any]]:
    for candidate in _candidate_paths(expected, source_roots):
        if not candidate.is_file():
            continue
        payload, actual = _read_json_source(
            candidate,
            argument=str(expected.get("argument", candidate)),
            bundle_root=candidate.parent,
            label="verified-manifest",
        )
        if _identity_matches(actual, expected):
            return payload, actual
    raise ValueError("manifest source identity mismatch or digest verification failed")


def _verified_records_source(
    expected: dict[str, Any],
    source_roots: list[Path],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    for candidate in _candidate_paths(expected, source_roots):
        if not candidate.is_file():
            continue
        reduced, actual = _load_reduce_records(
            candidate,
            argument=str(expected.get("argument", candidate)),
            bundle_root=candidate.parent,
            manifest=manifest,
            label="verified-records",
        )
        if _identity_matches(actual, expected):
            return reduced, actual
    raise ValueError("records source identity mismatch or digest verification failed")


def _verify_aux_source(expected: dict[str, Any], source_roots: list[Path], *, label: str) -> None:
    """Verify a provenance-only source such as the benchmark config."""
    for candidate in _candidate_paths(expected, source_roots):
        if not candidate.is_file():
            continue
        actual = _hash_source(
            candidate,
            argument=str(expected.get("argument", candidate)),
            bundle_root=candidate.parent,
            label=label,
        )
        if _identity_matches(actual, expected):
            return
    raise ValueError(f"{label} source identity mismatch or digest verification failed")


def _read_receipt(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return _read_json_source(
        path,
        argument=str(path),
        bundle_root=path.parent,
        label="receipt",
    )


def _verify_receipt(  # noqa: C901
    receipt_path: Path, source_roots: list[Path]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    receipt, receipt_identity = _read_receipt(receipt_path)
    fixed = {
        "schema_version": SHARD_RECEIPT_SCHEMA,
        "status": "validated",
        "artifact_kind": "single_planner_shard_validation_receipt",
        "evidence_status": "diagnostic-only",
        "claim_boundary": SHARD_CLAIM_BOUNDARY,
        "planner_count": 1,
    }
    for key, value in fixed.items():
        if receipt.get(key) != value:
            raise ValueError(f"receipt {receipt_path} has invalid {key}")
    planner = receipt.get("planner")
    if not isinstance(planner, str):
        raise ValueError(f"receipt {receipt_path} has no planner")
    _planner_artifact_key(planner)
    provenance = receipt.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"receipt {receipt_path} has no provenance")
    source_artifact_head = provenance.get("source_artifact_head")
    if not isinstance(source_artifact_head, str) or not source_artifact_head:
        raise ValueError(f"receipt {receipt_path} has no source artifact head")
    receipt_builder_head = provenance.get("receipt_builder_head")
    if not isinstance(receipt_builder_head, str) or not receipt_builder_head:
        raise ValueError(f"receipt {receipt_path} has no receipt builder head")
    manifest, actual_manifest = _verified_manifest_source(provenance["manifest"], source_roots)
    reduced, actual_records = _verified_records_source(
        provenance["records"], source_roots, manifest
    )
    config_identity = provenance.get("config")
    if isinstance(config_identity, dict) and config_identity.get("available") is True:
        if config_identity.get("argument") != manifest.get("config_path"):
            raise ValueError(f"receipt {receipt_path} config does not match its manifest")
        _verify_aux_source(config_identity, source_roots, label="config")
    manifest_planners = {str(row.get("planner")) for row in manifest.get("manifest_rows", [])}
    record_planners = {str(row["planner"]) for row in reduced["rank_records"]}
    if manifest_planners != {planner} or record_planners != {planner}:
        raise ValueError(f"receipt {receipt_path} planner does not match its sources")
    if _manifest_head(manifest) not in {None, source_artifact_head}:
        raise ValueError(f"receipt {receipt_path} source artifact head mismatches its manifest")
    if not reduced["integration_readiness"]["ready"]:
        raise ValueError(f"receipt {receipt_path} sources no longer pass readiness")
    comparisons = {
        "episode_record_count": reduced["record_count"],
        "integration_readiness": reduced["integration_readiness"],
        "normalized_cell_count": len(reduced["normalized_cells"]),
        "per_archetype_metric_report_counts": {
            key: len(value) for key, value in reduced["per_archetype_metric_reports"].items()
        },
        "streaming_stats": reduced["streaming_stats"],
    }
    for key, value in comparisons.items():
        if receipt.get(key) != value:
            raise ValueError(f"receipt {receipt_path} {key} does not match its sources")
    expected_command = _receipt_command_identity(
        planner=planner,
        source_artifact_head=source_artifact_head,
        receipt_builder_head=receipt_builder_head,
        manifest_identity=provenance["manifest"],
        records_identity=provenance["records"],
    )
    if provenance.get("command_identity") != expected_command:
        raise ValueError(f"receipt {receipt_path} command identity mismatch")
    verified = {
        "planner": planner,
        "source_artifact_head": source_artifact_head,
        "receipt_builder_head": receipt_builder_head,
        "receipt": receipt_identity,
        "manifest": actual_manifest,
        "records": actual_records,
        "streaming_stats": reduced["streaming_stats"],
    }
    return manifest, reduced, verified


def _merge_reduced(
    shards: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    first_manifest = shards[0][0]
    planner_order_list = _manifest_planner_order(first_manifest)
    planner_order = {planner: index for index, planner in enumerate(planner_order_list)}
    ordered_shards = sorted(
        shards,
        key=lambda item: (
            planner_order.get(item[2]["planner"], len(planner_order)),
            item[2]["planner"],
        ),
    )
    rank_records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    reports = {key: {} for key in first_manifest["trace_metric_keys"]}
    row_readiness: list[dict[str, Any]] = []
    expected_count = 0
    observed_count = 0
    for _, reduced, _ in ordered_shards:
        rank_records.extend(reduced["rank_records"])
        csv_rows.extend(reduced["csv_rows"])
        for metric_key, metric_reports in reduced["per_archetype_metric_reports"].items():
            reports[metric_key].update(metric_reports)
        readiness = reduced["integration_readiness"]
        row_readiness.extend(readiness["row_readiness"])
        expected_count += readiness["expected_row_count"]
        observed_count += readiness["observed_row_count"]
    row_readiness.sort(
        key=lambda row: (
            str(row["scenario_id"]),
            str(row["planner"]),
            int(row["seed"]),
            str(row["population_arm"]),
        )
    )
    combined_readiness = dict(ordered_shards[0][1]["integration_readiness"])
    combined_readiness["expected_row_count"] = expected_count
    combined_readiness["observed_row_count"] = observed_count
    combined_readiness["row_readiness"] = row_readiness
    combined_readiness["blockers"] = [
        blocker
        for _, reduced, _ in ordered_shards
        for blocker in reduced["integration_readiness"]["blockers"]
    ]
    return {
        "integration_readiness": combined_readiness,
        "rank_records": rank_records,
        "csv_rows": csv_rows,
        "per_archetype_metric_reports": reports,
        "planner_order": planner_order_list,
        "record_count": sum(reduced["record_count"] for _, reduced, _ in ordered_shards),
        "streaming_stats": {
            "processing_model": "sequential_whole_shard",
            "shard_count": len(ordered_shards),
            "max_live_raw_shards": 1,
            "max_live_raw_records": max(
                reduced["streaming_stats"]["max_live_raw_records"]
                for _, reduced, _ in ordered_shards
            ),
            "raw_shards_retained": 0,
            "raw_records_retained": 0,
            "combined_raw_record_list_allocated": False,
            "reduced_rank_record_count": len(rank_records),
        },
    }


def _finalize_receipts(
    receipt_arguments: list[str], source_root_arguments: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(receipt_arguments) < COMBINED_REPORT_MIN_PLANNERS:
        raise ValueError("finalize mode requires at least two --shard-receipt inputs")
    receipt_paths = sorted({_resolve_path(value) for value in receipt_arguments})
    if len(receipt_paths) != len(receipt_arguments):
        raise ValueError("duplicate shard receipt paths")
    source_roots = sorted({_resolve_path(value) for value in source_root_arguments})
    shards = sorted(
        (_verify_receipt(path, source_roots) for path in receipt_paths),
        key=lambda item: item[2]["planner"],
    )
    planners = [item[2]["planner"] for item in shards]
    if len(set(planners)) != len(planners):
        raise ValueError("finalize mode requires distinct planner receipts")
    source_heads = [item[2]["source_artifact_head"] for item in shards]
    if any(not head for head in source_heads) or len(set(source_heads)) != 1:
        raise ValueError("all shard source artifact heads must be present and equal")
    builder_heads = sorted({item[2]["receipt_builder_head"] for item in shards})
    first_manifest = shards[0][0]
    for manifest, _, _ in shards[1:]:
        for field in MANIFEST_COMPATIBILITY_FIELDS:
            if manifest.get(field) != first_manifest.get(field):
                raise ValueError(f"shard manifest mismatch for {field!r}")
    baseline_cells = shards[0][1]["normalized_cells"]
    for _, reduced, verified in shards[1:]:
        cells = reduced["normalized_cells"]
        if cells != baseline_cells:
            missing = len(baseline_cells - cells)
            extra = len(cells - baseline_cells)
            raise ValueError(
                f"normalized cell coverage mismatch for {verified['planner']}: "
                f"missing={missing}, extra={extra}"
            )
    merged = _merge_reduced(shards)
    receipt_digests = sorted(item[2]["receipt"]["sha256"] for item in shards)
    provenance = {
        "schema_version": FINALIZATION_PROVENANCE_SCHEMA,
        "status": "verified",
        "mode": "finalize",
        "source_artifact_head": source_heads[0],
        "receipt_builder_heads": builder_heads,
        "planners": sorted(planners),
        "planner_count": len(planners),
        "receipt_count": len(shards),
        "receipts": [item[2] for item in shards],
        "streaming_stats": merged["streaming_stats"],
        "command_identity": {
            "mode": "finalize",
            "schema_version": FINALIZATION_PROVENANCE_SCHEMA,
            "source_artifact_head": source_heads[0],
            "receipt_builder_heads": builder_heads,
            "receipt_sha256": receipt_digests,
        },
    }
    return merged, provenance


def _write_blocker(output_dir: Path, filename: str, reason: str, error: str) -> None:
    _write_json(
        (output_dir,),
        filename,
        {"status": "blocked", "reason": reason, "error": error},
    )


def _write_combined_report(  # noqa: C901
    reduced: dict[str, Any],
    *,
    destinations: tuple[Path, Path],
    finalization_provenance: dict[str, Any] | None,
) -> None:
    planners = sorted({record["planner"] for record in reduced["rank_records"]})
    if len(planners) < COMBINED_REPORT_MIN_PLANNERS:
        raise ValueError("final combined report requires at least two distinct planners")
    rank_sensitivity = compute_bootstrap_rank_sensitivity(
        reduced["rank_records"],
        metric_key="mean_clearance",
        planners=planners,
        higher_is_safer=True,
        num_bootstrap=1000,
        seed=3574,
    )
    reports = reduced["per_archetype_metric_reports"]
    summary = {
        "schema_version": "heterogeneous_population_ablation.v1",
        "integration_readiness": reduced["integration_readiness"],
        "rank_sensitivity": rank_sensitivity,
        "per_archetype_metric_reports": reports,
        "ablation_reports": reports.get("clearance_m", {}),
    }
    if finalization_provenance is not None:
        summary["finalization_provenance"] = finalization_provenance
        _write_json(destinations, "finalization_provenance.json", finalization_provenance)
    planner_order = {planner: index for index, planner in enumerate(reduced["planner_order"])}
    csv_rows = sorted(
        reduced["csv_rows"],
        key=lambda row: (
            str(row["scenario_id"]),
            float(row["response_law_fraction"]),
            planner_order.get(str(row["planner"]), len(planner_order)),
            int(row["seed"]),
            str(row["arm"]),
        ),
    )
    _write_json(destinations, "summary.json", summary)
    _write_json(destinations, "rank_sensitivity.json", rank_sensitivity)
    _write_csv(destinations, csv_rows)
    trace_metrics = ", ".join(reduced["integration_readiness"]["trace_metric_keys"])
    markdown = f"""# Issue #3574 Heterogeneous Population Ablation Report

This report compiles supplied paired episode records into the issue #3574 integration schema.

## Claim Boundary

- Evidence status: `diagnostic-only` until a separately reviewed campaign provides attributable
  paired records and appropriate confidence bounds.
- This report does not run a benchmark and does not establish a heterogeneous-population effect,
  planner rank-stability result, realism claim, or sim-to-real claim.
- Any ranking text below describes only the supplied records; it is not an empirical conclusion by
  itself.

## Executive Summary
We compared planners across two paired population arms:
1. **Heterogeneous**: A mixture of cautious (25%), standard (50%), and hurried (25%) pedestrians.
2. **Mean-Matched Homogeneous**: A homogeneous population with speed/radius set to the weighted means of the mixture.

We evaluated the safety metric **clearance_m** (distance between robot and nearest pedestrian) using **mean** and **CVaR (alpha=0.2)** (tail safety of the 20% closest encounters).

## Per-Archetype Trace Metrics

The JSON summary contains per-archetype reports for every metric required by the paired manifest:
`{trace_metrics}`. These reports preserve both arms for
each scenario/seed/planner triplet and are only rendered after the fail-closed integration-readiness
check passes.

## Rank-Order Sensitivity Analysis
We ran paired bootstrap resampling (1000 iterations) over seeds to compute the probability of one planner beating another under both arms.

### Rankings (metric: clearance_m, higher is safer)
"""
    if rank_sensitivity["status"] == "ready":
        for arm_name in sorted(rank_sensitivity["arms"]):
            arm_data = rank_sensitivity["arms"][arm_name]
            markdown += f"\n### Arm: {arm_name}\n"
            markdown += f"- **Rank Order**: {', '.join(arm_data['ranking'])}\n"
            markdown += "- **Observed Means**:\n"
            for planner, mean in arm_data["observed_means"].items():
                markdown += f"  - {planner}: {mean:.4f} m\n"
            markdown += "- **Pairwise Bootstrap Probabilities**:\n"
            for pair_name, probability in arm_data["pairwise_probabilities"].items():
                label = pair_name.replace("_beats_", " beats ")
                markdown += f"  - P({label}) = {probability:.2%}\n"

        markdown += "\n### Rank Reversals\n"
        reversals = rank_sensitivity.get("reversals", [])
        if reversals:
            for reversal in reversals:
                markdown += f"- **WARNING**: Reversal detected! {reversal['description']}\n"
        else:
            markdown += (
                "- **No rank reversals detected.** The planner ranking was stable "
                "across both arms.\n"
            )
    else:
        markdown += f"\nRank sensitivity calculation blocked: {rank_sensitivity.get('blockers')}\n"

    markdown += """
## Detailed Ablation Results
Below is the table of the clearance metrics per seed and arm:

| Scenario | Seed | Planner | Arm | Mean Clearance (m) | CVaR Clearance (m) |
|---|---|---|---|---|---|
"""
    for row in csv_rows:
        markdown += (
            f"| {row['scenario_id']} | {row['seed']} | {row['planner']} | {row['arm']} | "
            f"{row['mean_clearance_m']} | {row['cvar_clearance_m']} |\n"
        )
    markdown += """
## Non-Reactive Mixture Sweeps Caveats
- Since this is a CPU-level smoke validation run on a small slice, rank sensitivity estimates carry higher uncertainty.
- In full runs, a larger sample of seeds and scenarios is required to establish statistical significance.
"""
    for destination in destinations:
        (destination / "analysis.md").write_text(markdown, encoding="utf-8")


def _direct_sources(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
    Path,
]:
    manifest_path = _resolve_path(args.manifest)
    records_path = _resolve_path(args.records)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not records_path.is_file():
        raise FileNotFoundError(f"Episode records not found: {records_path}")
    bundle_root = Path(os.path.commonpath((manifest_path.parent, records_path.parent))).resolve()
    manifest, _ = _read_json_source(
        manifest_path,
        argument=args.manifest,
        bundle_root=bundle_root,
        label="manifest",
    )
    reduced, records_identity = _load_reduce_records(
        records_path,
        argument=args.records,
        bundle_root=bundle_root,
        manifest=manifest,
        label="records",
    )
    return (
        manifest,
        reduced,
        {},
        records_identity,
        manifest_path,
        records_path,
        bundle_root,
    )


def main() -> int:  # noqa: C901
    """Run shard validation, direct comparison, or receipt-driven finalization."""
    args = parse_args()
    output_dir = _resolve_path(args.output_dir)
    durable_dir = _resolve_path(args.durable_dir)

    if args.mode == "finalize":
        try:
            reduced, provenance = _finalize_receipts(
                list(args.shard_receipt), list(args.source_root)
            )
            _write_combined_report(
                reduced,
                destinations=(output_dir, durable_dir),
                finalization_provenance=provenance,
            )
            _write_json(
                (output_dir,),
                "finalization_diagnostics.json",
                {
                    "canonical": False,
                    "purpose": "raw_controller_invocation",
                    "executable": sys.executable,
                    "argv": list(sys.argv),
                },
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            _write_blocker(
                output_dir,
                "finalization_blocked.json",
                "shard_receipt_verification_failed",
                str(exc),
            )
            print(f"Blocked: {exc}")
            return 2
        print(f"Final report generation complete: {output_dir}")
        return 0

    try:
        (
            manifest,
            reduced,
            _,
            records_identity,
            manifest_path,
            _,
            bundle_root,
        ) = _direct_sources(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}")
        return 1
    readiness = reduced["integration_readiness"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "integration_readiness.json").write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not readiness["ready"]:
        print(f"Blocked: episode records failed readiness; see {output_dir}")
        return 2

    try:
        _rewrite_manifest_status(manifest_path, manifest)
        manifest, manifest_identity = _read_json_source(
            manifest_path,
            argument=args.manifest,
            bundle_root=bundle_root,
            label="manifest-final",
        )
    except OSError as exc:
        _write_json(
            (output_dir,),
            "manifest_rewrite_failure.json",
            {
                "status": "blocked",
                "reason": "manifest_rewrite_failed",
                "manifest_path": str(manifest_path),
                "error": str(exc),
            },
        )
        return 2

    planners = sorted({record["planner"] for record in reduced["rank_records"]})
    if args.mode == "shard":
        if len(planners) != 1:
            _write_blocker(
                output_dir,
                "shard_receipt_blocked.json",
                "single_planner_shard_receipt_requires_exactly_one_planner",
                f"planners={planners}",
            )
            return 2
        try:
            planner = _planner_artifact_key(planners[0])
            source_artifact_head = _source_artifact_head(manifest, args.source_artifact_head)
            receipt_builder_head = _receipt_builder_head()
        except ValueError as exc:
            _write_blocker(
                output_dir, "shard_receipt_blocked.json", "invalid_shard_identity", str(exc)
            )
            return 2
        destinations = (
            output_dir / "shards" / planner,
            durable_dir / "shards" / planner,
        )
        receipt = _build_shard_receipt(
            planner=planner,
            source_artifact_head=source_artifact_head,
            receipt_builder_head=receipt_builder_head,
            manifest=manifest,
            manifest_identity=manifest_identity,
            records_identity=records_identity,
            reduced=reduced,
        )
        _write_json(destinations, "shard_receipt.json", receipt)
        _write_csv(destinations, reduced["csv_rows"])
        markdown = _receipt_markdown(receipt)
        for destination in destinations:
            (destination / "shard_receipt.md").write_text(markdown, encoding="utf-8")
        print(f"Shard validation complete: {destinations[0]}")
        return 0

    try:
        _write_combined_report(
            reduced,
            destinations=(output_dir, durable_dir),
            finalization_provenance=None,
        )
    except ValueError as exc:
        _write_blocker(
            output_dir,
            "combined_report_blocked.json",
            "cross_planner_rank_comparison_requires_at_least_two_planners",
            str(exc),
        )
        return 2
    print(f"Report generation complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
