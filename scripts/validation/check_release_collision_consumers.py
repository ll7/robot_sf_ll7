#!/usr/bin/env python3
"""Fail-closed audit of collision-derived consumers in a publication bundle.

The checker reads a release ``.tar.gz`` without extracting it, verifies both the
outer archive digest and every embedded payload checksum, and then reconciles
per-episode collision outcomes against success and the canonical Social
Navigation Quality Index (SNQI) implementation.

This is a narrow metric-integrity audit. A pass is not a release-wide benchmark,
planner-ranking, SNQI-validity, or paper-facing claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tarfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from robot_sf.benchmark.event_ledger import (
    EPISODE_EVENT_LEDGER_SCHEMA_VERSION,
    reconcile_event_ledger,
)
from robot_sf.benchmark.metrics import snqi
from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    load_baseline_mapping,
    load_weight_mapping,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "configs/benchmarks/snqi_weights_camera_ready_v3.json"
DEFAULT_BASELINE = ROOT / "configs/benchmarks/snqi_baseline_camera_ready_v3.json"
SCHEMA_VERSION = "release_collision_consumer_reconciliation.v1"
CLAIM_BOUNDARY = (
    "Diagnostic-only reconciliation of exact per-episode collision outcomes with collision "
    "counts, the binary-success collision gate, and the canonical SNQI collision term. A pass is "
    "not release-wide benchmark success, planner-ranking evidence, SNQI contract validity, or a "
    "paper claim."
)
_EPISODE_PATH = re.compile(r"^runs/(?P<arm>[^/]+)/episodes\.jsonl$")
_MAX_REPORTED_VIOLATIONS = 100


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_checksum_manifest(data: bytes) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(data.decode("utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"checksums.sha256:{line_number}: malformed checksum entry")
        path = parts[1].lstrip("*")
        if path in entries:
            raise ValueError(f"checksums.sha256:{line_number}: duplicate path {path!r}")
        entries[path] = parts[0].lower()
    if not entries:
        raise ValueError("checksums.sha256 contains no entries")
    return entries


def _json_object(data: bytes, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: expected a JSON object")
    return payload


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


class _AuditState:
    def __init__(self) -> None:
        self.violation_counts: Counter[str] = Counter()
        self.violations: list[dict[str, str]] = []
        self.rows = 0
        self.collision_rows = 0
        self.goal_timeout_rows = 0
        self.success_rows = 0
        self.snqi_rows = 0
        self.per_arm_rows: Counter[str] = Counter()
        self.episode_commits: Counter[str] = Counter()
        self.identities: set[tuple[str, str]] = set()

    def add(self, code: str, message: str) -> None:
        self.violation_counts[code] += 1
        if len(self.violations) < _MAX_REPORTED_VIOLATIONS:
            self.violations.append({"code": code, "message": message})


def _check_episode(  # noqa: C901, PLR0912, PLR0915
    record: Mapping[str, Any],
    *,
    arm: str,
    source: str,
    weights: dict[str, float],
    baseline: dict[str, dict[str, float]],
    state: _AuditState,
) -> None:
    state.rows += 1
    state.per_arm_rows[arm] += 1
    episode_id = record.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        state.add("episode_identity", f"{source}: episode_id must be a non-empty string")
        episode_id = f"<missing:{state.rows}>"
    identity = (arm, episode_id)
    if identity in state.identities:
        state.add("duplicate_episode", f"{source}: duplicate episode identity {identity!r}")
    state.identities.add(identity)

    metrics = record.get("metrics")
    ledger = record.get("event_ledger")
    outcome = record.get("outcome")
    if not isinstance(metrics, Mapping):
        state.add("metrics_shape", f"{source}: metrics must be an object")
        return
    if not isinstance(ledger, Mapping):
        state.add("ledger_shape", f"{source}: event_ledger must be an object")
        return
    if not isinstance(outcome, Mapping):
        state.add("outcome_shape", f"{source}: outcome must be an object")
        return

    exact = ledger.get("exact_events")
    reconciliation = ledger.get("reconciliation")
    collision_events = ledger.get("collision_events")
    if ledger.get("schema_version") != EPISODE_EVENT_LEDGER_SCHEMA_VERSION:
        state.add(
            "ledger_schema",
            f"{source}: expected {EPISODE_EVENT_LEDGER_SCHEMA_VERSION}",
        )
    if not isinstance(exact, Mapping):
        state.add("exact_events_shape", f"{source}: exact_events must be an object")
        return
    if not isinstance(reconciliation, Mapping):
        state.add("reconciliation_shape", f"{source}: reconciliation must be an object")
        return
    if not isinstance(collision_events, list):
        state.add("collision_events_shape", f"{source}: collision_events must be a list")
        return

    exact_fields = ("collision", "goal_reached", "timeout", "invalid_run")
    if any(not isinstance(exact.get(field), bool) for field in exact_fields):
        state.add("exact_events_type", f"{source}: exact event fields must all be booleans")
        return
    collision = bool(exact["collision"])
    if exact["goal_reached"] and exact["timeout"]:
        state.goal_timeout_rows += 1

    ledger_violations = reconcile_event_ledger(ledger)
    for violation in ledger_violations:
        state.add("ledger_reconciliation", f"{source}: {violation}")
    if reconciliation.get("audit_result") != "pass":
        state.add("ledger_audit_result", f"{source}: reconciliation.audit_result is not pass")

    if outcome.get("collision_event") is not collision:
        state.add("outcome_collision", f"{source}: outcome collision disagrees with exact ledger")
    if bool(collision_events) is not collision:
        state.add(
            "typed_collision_event",
            f"{source}: exact collision and typed collision-event presence disagree",
        )

    required_metrics = (
        "collisions",
        "total_collision_count",
        "ped_collision_count",
        "obstacle_collision_count",
        "agent_collision_count",
        "time_to_goal_norm",
        "near_misses",
        "comfort_exposure",
        "force_exceed_events",
        "jerk_mean",
        "curvature_mean",
        "snqi",
    )
    try:
        values = {
            field: _finite_number(metrics.get(field), field=f"{source}: metrics.{field}")
            for field in required_metrics
        }
    except ValueError as exc:
        state.add("metric_value", str(exc))
        return
    metric_success = metrics.get("success")
    if not isinstance(metric_success, bool):
        state.add("success_type", f"{source}: metrics.success must be boolean")
        return
    if collision and metric_success:
        state.add(
            "collision_success",
            f"{source}: exact collision must force metrics.success=false",
        )

    total = values["total_collision_count"]
    for component in (
        "collisions",
        "ped_collision_count",
        "obstacle_collision_count",
        "agent_collision_count",
    ):
        if values[component] < 0.0 or not values[component].is_integer():
            state.add(
                "collision_count_domain",
                f"{source}: metrics.{component} must be integer >= 0",
            )
    components = (
        values["ped_collision_count"]
        + values["obstacle_collision_count"]
        + values["agent_collision_count"]
    )
    if total < 0.0 or not total.is_integer():
        state.add("collision_count_domain", f"{source}: total collision count must be integer >= 0")
    if not math.isclose(values["collisions"], total, abs_tol=1e-12):
        state.add("collision_alias", f"{source}: collisions != total_collision_count")
    if not math.isclose(components, total, abs_tol=1e-12):
        state.add("collision_components", f"{source}: collision components do not sum to total")
    if (total > 0.0) is not collision:
        state.add("exact_collision_count", f"{source}: exact collision and total count disagree")
    try:
        ledger_collision_value = _finite_number(
            reconciliation.get("collision_metric_value"),
            field=f"{source}: reconciliation.collision_metric_value",
        )
    except ValueError as exc:
        state.add("ledger_collision_value", str(exc))
    else:
        if not math.isclose(ledger_collision_value, total, abs_tol=1e-12):
            state.add("ledger_collision_value", f"{source}: ledger collision value != total")
    if reconciliation.get("collision_metric_source") != "metrics.total_collision_count":
        state.add(
            "ledger_collision_source",
            f"{source}: ledger collision source must be metrics.total_collision_count",
        )

    metric_values = dict(metrics)
    actual_snqi = values["snqi"]
    recomputed_snqi = snqi(metric_values, weights, baseline)
    if not math.isclose(actual_snqi, recomputed_snqi, rel_tol=1e-12, abs_tol=1e-12):
        state.add(
            "snqi_recomputation",
            f"{source}: stored SNQI {actual_snqi!r} != canonical {recomputed_snqi!r}",
        )
    else:
        state.snqi_rows += 1

    if collision:
        state.collision_rows += 1
        counterfactual = dict(metric_values)
        counterfactual["collisions"] = 0.0
        collision_penalty = snqi(counterfactual, weights, baseline) - recomputed_snqi
        if not collision_penalty > 0.0:
            state.add(
                "snqi_collision_penalty",
                f"{source}: exact collision does not activate a positive SNQI penalty",
            )
    if metric_success:
        state.success_rows += 1

    software_commit = ledger.get("software_commit")
    if not isinstance(software_commit, str) or not software_commit:
        state.add("episode_commit", f"{source}: ledger software_commit is missing")
    else:
        state.episode_commits[software_commit] += 1


def _check_episode_member(
    handle: BinaryIO,
    *,
    arm: str,
    source: str,
    weights: dict[str, float],
    baseline: dict[str, dict[str, float]],
    state: _AuditState,
) -> str:
    digest = hashlib.sha256()
    for line_number, raw_line in enumerate(handle, start=1):
        digest.update(raw_line)
        if not raw_line.strip():
            continue
        row_source = f"{source}:{line_number}"
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            state.add("episode_json", f"{row_source}: invalid JSON: {exc}")
            continue
        if not isinstance(record, Mapping):
            state.add("episode_shape", f"{row_source}: expected a JSON object")
            continue
        _check_episode(
            record,
            arm=arm,
            source=row_source,
            weights=weights,
            baseline=baseline,
            state=state,
        )
    return digest.hexdigest()


def _release_warnings(
    metadata: Mapping[str, dict[str, Any]],
    *,
    episode_commits: Counter[str],
    goal_timeout_rows: int,
) -> list[str]:
    warnings: list[str] = []
    release_result = metadata.get("release/release_result.json", {})
    campaign_summary = metadata.get("reports/campaign_summary.json", {})
    campaign = campaign_summary.get("campaign")
    campaign = campaign if isinstance(campaign, Mapping) else {}
    publication = metadata.get("publication_manifest.json", {})
    provenance = publication.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    repository = provenance.get("repository")
    repository = repository if isinstance(repository, Mapping) else {}

    comparisons = (
        ("status", release_result.get("status"), campaign.get("status")),
        ("evidence_status", release_result.get("evidence_status"), campaign.get("evidence_status")),
        ("total_episodes", release_result.get("total_episodes"), campaign.get("total_episodes")),
        ("successful_runs", release_result.get("successful_runs"), campaign.get("successful_runs")),
    )
    mismatched = [name for name, old, rebuilt in comparisons if old != rebuilt]
    if mismatched:
        warnings.append(
            "release/release_result.json disagrees with the rebuilt campaign summary for: "
            + ", ".join(mismatched)
        )
    tag_commit = repository.get("commit")
    if tag_commit and set(episode_commits) != {tag_commit}:
        warnings.append(
            "episode-ledger software commits do not equal the publication manifest/tag commit"
        )
    if campaign.get("snqi_contract_status") != "pass":
        warnings.append(
            f"campaign SNQI contract status is {campaign.get('snqi_contract_status')!r}; "
            "this audit proves only collision-term consumption"
        )
    if goal_timeout_rows:
        warnings.append(
            f"{goal_timeout_rows} non-collision row(s) record both goal_reached and timeout; "
            "the bundle lacks reached_goal_step/horizon inputs needed to recompute that separate "
            "success-timing condition"
        )
    channels = publication.get("publication_channels")
    channels = channels if isinstance(channels, Mapping) else {}
    if any(
        "<record-id>" in str(value) or "{release_tag}" in str(value) for value in channels.values()
    ):
        warnings.append("publication metadata retains unresolved URL/DOI placeholders")
    return warnings


def audit_bundle(  # noqa: C901, PLR0912, PLR0913, PLR0915
    bundle: Path,
    *,
    expected_bundle_sha256: str,
    expected_release_tag: str,
    expected_rows: int,
    expected_arms: int,
    expected_rows_per_arm: int,
    weights_path: Path = DEFAULT_WEIGHTS,
    baseline_path: Path = DEFAULT_BASELINE,
    source_url: str | None = None,
) -> dict[str, Any]:
    """Audit one publication bundle and return a machine-readable report."""
    state = _AuditState()
    metadata: dict[str, dict[str, Any]] = {}
    bundle_sha256 = _sha256_path(bundle)
    weights_sha256 = _sha256_path(weights_path)
    baseline_sha256 = _sha256_path(baseline_path)
    weights = load_weight_mapping(weights_path)
    baseline = load_baseline_mapping(baseline_path)
    checksums: dict[str, str] = {}
    payload_hashes: dict[str, str] = {}
    roots: set[str] = set()

    if bundle_sha256 != expected_bundle_sha256.lower():
        state.add(
            "bundle_sha256",
            f"archive sha256 {bundle_sha256} != expected {expected_bundle_sha256.lower()}",
        )
    else:
        try:
            with tarfile.open(bundle, mode="r:gz") as archive:
                for member in archive:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or ".." in path.parts or not path.parts:
                        state.add("unsafe_archive_path", f"unsafe archive member {member.name!r}")
                        continue
                    roots.add(path.parts[0])
                    if member.isdir():
                        continue
                    if not member.isfile():
                        state.add(
                            "unsafe_archive_type", f"non-regular archive member {member.name!r}"
                        )
                        continue
                    handle = archive.extractfile(member)
                    if handle is None:
                        state.add("archive_read", f"cannot read archive member {member.name!r}")
                        continue
                    logical_path_parts = path.parts[1:]
                    logical_path = PurePosixPath(*logical_path_parts).as_posix()
                    if logical_path == "checksums.sha256":
                        try:
                            checksums = _parse_checksum_manifest(handle.read())
                        except ValueError as exc:
                            state.add("checksum_manifest", str(exc))
                        continue
                    if logical_path == "publication_manifest.json":
                        try:
                            metadata[logical_path] = _json_object(
                                handle.read(), source=logical_path
                            )
                        except ValueError as exc:
                            state.add("metadata_json", str(exc))
                        continue
                    if not logical_path_parts or logical_path_parts[0] != "payload":
                        _sha256_stream(handle)
                        continue

                    payload_path = PurePosixPath(*logical_path_parts[1:]).as_posix()
                    episode_match = _EPISODE_PATH.fullmatch(payload_path)
                    selected_json = payload_path in {
                        "release/release_manifest.resolved.json",
                        "release/release_result.json",
                        "reports/campaign_summary.json",
                    }
                    if episode_match:
                        digest = _check_episode_member(
                            handle,
                            arm=episode_match.group("arm"),
                            source=payload_path,
                            weights=weights,
                            baseline=baseline,
                            state=state,
                        )
                    elif selected_json:
                        data = handle.read()
                        digest = hashlib.sha256(data).hexdigest()
                        try:
                            metadata[payload_path] = _json_object(data, source=payload_path)
                        except ValueError as exc:
                            state.add("metadata_json", str(exc))
                    else:
                        digest = _sha256_stream(handle)
                    payload_hashes[payload_path] = digest
                    expected_digest = checksums.get(payload_path)
                    if expected_digest is None:
                        state.add("unsigned_payload", f"payload file is unsigned: {payload_path}")
                    elif digest != expected_digest:
                        state.add("payload_sha256", f"payload checksum mismatch: {payload_path}")
        except (OSError, tarfile.TarError) as exc:
            state.add("archive_read", f"cannot read archive: {exc}")

    if len(roots) != 1:
        state.add("archive_root", f"expected one archive root, found {sorted(roots)!r}")
    missing_payloads = sorted(set(checksums) - set(payload_hashes))
    for path in missing_payloads:
        state.add("missing_payload", f"signed payload file is missing: {path}")

    publication = metadata.get("publication_manifest.json", {})
    channels = publication.get("publication_channels")
    channels = channels if isinstance(channels, Mapping) else {}
    if channels.get("release_tag") != expected_release_tag:
        state.add("release_tag", "publication manifest release tag does not match expected tag")
    published_files = publication.get("files")
    if isinstance(published_files, list):
        published_hashes = {
            str(entry.get("path")): str(entry.get("sha256"))
            for entry in published_files
            if isinstance(entry, Mapping)
        }
        if published_hashes != checksums:
            state.add("publication_file_manifest", "publication file manifest != checksums.sha256")
    else:
        state.add("publication_file_manifest", "publication manifest files must be a list")

    release_manifest = metadata.get("release/release_manifest.resolved.json", {})
    if release_manifest.get("release_tag") != expected_release_tag:
        state.add("release_tag", "resolved release manifest tag does not match expected tag")
    metric_contract = release_manifest.get("metrics")
    metric_contract = metric_contract if isinstance(metric_contract, Mapping) else {}
    if metric_contract.get("snqi_weights_sha256") != weights_sha256:
        state.add("weights_sha256", "local SNQI weights do not match release manifest")
    if metric_contract.get("snqi_baseline_sha256") != baseline_sha256:
        state.add("baseline_sha256", "local SNQI baseline does not match release manifest")
    campaign_summary = metadata.get("reports/campaign_summary.json", {})
    campaign = campaign_summary.get("campaign")
    campaign = campaign if isinstance(campaign, Mapping) else {}
    if campaign.get("snqi_weights_sha256") not in {None, weights_sha256}:
        state.add("weights_sha256", "local SNQI weights do not match campaign summary")
    if campaign.get("snqi_baseline_sha256") not in {None, baseline_sha256}:
        state.add("baseline_sha256", "local SNQI baseline does not match campaign summary")

    if state.rows != expected_rows:
        state.add("row_count", f"expected {expected_rows} rows, found {state.rows}")
    if len(state.per_arm_rows) != expected_arms:
        state.add("arm_count", f"expected {expected_arms} arms, found {len(state.per_arm_rows)}")
    for arm, count in sorted(state.per_arm_rows.items()):
        if count != expected_rows_per_arm:
            state.add(
                "rows_per_arm",
                f"arm {arm!r}: expected {expected_rows_per_arm} rows, found {count}",
            )
    if state.collision_rows == 0:
        state.add("collision_support", "bundle contains no exact collision rows")

    warnings = _release_warnings(
        metadata,
        episode_commits=state.episode_commits,
        goal_timeout_rows=state.goal_timeout_rows,
    )
    status = "pass" if not state.violation_counts else "fail"
    publication_provenance = publication.get("provenance")
    publication_provenance = (
        publication_provenance if isinstance(publication_provenance, Mapping) else {}
    )
    publication_repository = publication_provenance.get("repository")
    publication_repository = (
        publication_repository if isinstance(publication_repository, Mapping) else {}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "review_marker": "AI-GENERATED NEEDS-REVIEW",
        "status": status,
        "result_classification": (
            "collision_consumer_reconciled" if status == "pass" else "not_reconciled"
        ),
        "evidence_grade": "diagnostic-only",
        "diagnostic_only": True,
        "benchmark_promotion": False,
        "paper_facing": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "source": {
            "release_tag": expected_release_tag,
            "release_asset_url": source_url,
            "bundle_name": bundle.name,
            "bundle_sha256": bundle_sha256,
        },
        "integrity": {
            "outer_bundle_sha256_matches": bundle_sha256 == expected_bundle_sha256.lower(),
            "signed_payload_files": len(checksums),
            "verified_payload_files": len(payload_hashes),
            "snqi_weights_sha256": weights_sha256,
            "snqi_baseline_sha256": baseline_sha256,
        },
        "counts": {
            "arms": len(state.per_arm_rows),
            "rows": state.rows,
            "rows_per_arm": dict(sorted(state.per_arm_rows.items())),
            "exact_collision_rows": state.collision_rows,
            "goal_and_timeout_rows": state.goal_timeout_rows,
            "success_rows": state.success_rows,
            "snqi_recomputed_rows": state.snqi_rows,
            "unique_episode_identities": len(state.identities),
        },
        "provenance": {
            "episode_software_commits": dict(sorted(state.episode_commits.items())),
            "publication_repository_commit": publication_repository.get("commit"),
        },
        "release_wide_warnings": warnings,
        "violation_count": sum(state.violation_counts.values()),
        "violation_counts": dict(sorted(state.violation_counts.items())),
        "violations": state.violations,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--expected-release-tag", required=True)
    parser.add_argument("--expected-rows", required=True, type=int)
    parser.add_argument("--expected-arms", required=True, type=int)
    parser.add_argument("--expected-rows-per-arm", required=True, type=int)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--source-url")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line collision-consumer audit."""
    args = _parser().parse_args(argv)
    try:
        report = audit_bundle(
            args.bundle,
            expected_bundle_sha256=args.expected_bundle_sha256,
            expected_release_tag=args.expected_release_tag,
            expected_rows=args.expected_rows,
            expected_arms=args.expected_arms,
            expected_rows_per_arm=args.expected_rows_per_arm,
            weights_path=args.weights,
            baseline_path=args.baseline,
            source_url=args.source_url,
        )
    except (OSError, ValueError) as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "review_marker": "AI-GENERATED NEEDS-REVIEW",
            "status": "fail",
            "result_classification": "not_reconciled",
            "evidence_grade": "diagnostic-only",
            "diagnostic_only": True,
            "benchmark_promotion": False,
            "paper_facing": False,
            "claim_boundary": CLAIM_BOUNDARY,
            "violation_count": 1,
            "violation_counts": {"input_error": 1},
            "violations": [{"code": "input_error", "message": str(exc)}],
        }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
