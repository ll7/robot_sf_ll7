#!/usr/bin/env python3
"""Fail-closed release-gate self-check: per-episode SNQI field vs diagnostics basis.

Release 0.0.3 exposed a silent drift between *two* SNQI surfaces in the publication
bundle (issue #5580):

- the per-episode ``metrics["snqi"]`` field, baked at episode-capture time by
  ``robot_sf.benchmark.metrics.snqi`` (the campaign-aware scalarizer that includes the
  ``curvature_mean`` / ``w_curvature`` term); and
- the ``snqi_diagnostics.json`` ``planner_ordering`` / ``mean_snqi``, computed by
  ``robot_sf.benchmark.snqi.compute.compute_snqi_v0`` (which has **no** curvature term).

Those two formulas are not the same scalarization, so the per-episode field matches the
curvature-aware recomputation at ~100% but the ``compute_snqi_v0`` diagnostics basis at
only ~33.9%, and they can elect a *different* SNQI-best arm. This checker closes that
gate: it recomputes every per-episode ``snqi`` with the **same curvature-aware basis the
stored field actually uses** and asserts the two surfaces stay aligned — both per-episode
and at the aggregated planner-ordering level — so they can never drift silently again.

This is a narrow bundle-consistency audit. A pass is not a release-wide benchmark,
planner-ranking, SNQI-contract, or paper claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tarfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.metrics import snqi as curvature_aware_snqi
from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    load_baseline_mapping,
    load_weight_mapping,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "configs/benchmarks/snqi_weights_camera_ready_v3.json"
DEFAULT_BASELINE = ROOT / "configs/benchmarks/snqi_baseline_camera_ready_v3.json"
SCHEMA_VERSION = "release_snqi_field_consistency.v1"
CLAIM_BOUNDARY = (
    "Diagnostic-only reconciliation of the per-episode metrics.snqi field against the "
    "snqi_diagnostics.json planner_ordering basis, recomputed with the same curvature-aware "
    "SNQI scalarizer that produced the stored field. A pass is not release-wide benchmark "
    "success, planner-ranking evidence, SNQI contract validity, or a paper claim."
)
_EPISODE_PATH = re.compile(r"^[^/]+/payload/runs/(?P<arm>[^/]+)/episodes\.jsonl$")
_RECOMPUTE_RTOL = 1e-9
_RECOMPUTE_ATOL = 1e-9
_MAX_REPORTED_VIOLATIONS = 100


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(data: bytes, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: expected a JSON object")
    return payload


class _AuditState:
    def __init__(self) -> None:
        self.violation_counts: dict[str, int] = defaultdict(int)
        self.violations: list[dict[str, str]] = []
        self.rows = 0
        self.episode_field_present = 0
        self.snqi_recomputed_rows = 0
        self.per_arm_field_sum: dict[str, float] = defaultdict(float)
        self.per_arm_recomputed_sum: dict[str, float] = defaultdict(float)
        self.per_arm_field_count: dict[str, int] = defaultdict(int)
        self.per_arm_recomputed_count: dict[str, int] = defaultdict(int)

    def add(self, code: str, message: str) -> None:
        self.violation_counts[code] += 1
        if len(self.violations) < _MAX_REPORTED_VIOLATIONS:
            self.violations.append({"code": code, "message": message})


def _planner_key(arm: str) -> str:
    """Return the canonical ``planner_key::kinematics`` token used by the diagnostics ordering.

    The bundle's per-arm directory is ``<planner>__<kinematics>``; the diagnostics
    ``planner_ordering`` enumerates the same arms as ``planner_key`` / ``kinematics``.
    """
    planner, _, kinematics = arm.partition("__")
    return f"{planner}::{kinematics}"


def _check_episode(
    record: Mapping[str, Any],
    *,
    arm: str,
    source: str,
    weights: dict[str, float],
    baseline: dict[str, dict[str, float]],
    state: _AuditState,
) -> None:
    state.rows += 1
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        state.add("metrics_shape", f"{source}: metrics must be an object")
        return
    if "snqi" not in metrics:
        state.add("snqi_field_absent", f"{source}: metrics.snqi field is absent")
        return
    state.episode_field_present += 1
    try:
        stored_snqi = float(metrics["snqi"])
    except (TypeError, ValueError) as exc:
        state.add("snqi_field_type", f"{source}: metrics.snqi is not a finite number: {exc}")
        return
    metric_values = dict(metrics)
    try:
        recomputed_snqi = curvature_aware_snqi(metric_values, weights, baseline_stats=baseline)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        state.add("snqi_recompute_failed", f"{source}: curvature-aware recompute failed: {exc}")
        return
    state.snqi_recomputed_rows += 1
    state.per_arm_field_sum[arm] += stored_snqi
    state.per_arm_field_count[arm] += 1
    state.per_arm_recomputed_sum[arm] += recomputed_snqi
    state.per_arm_recomputed_count[arm] += 1
    if not math.isclose(
        stored_snqi, recomputed_snqi, rel_tol=_RECOMPUTE_RTOL, abs_tol=_RECOMPUTE_ATOL
    ):
        state.add(
            "snqi_field_recompute_mismatch",
            f"{source}: stored SNQI {stored_snqi!r} != curvature-aware recompute "
            f"{recomputed_snqi!r}",
        )


def read_snqi_diagnostics(archive: tarfile.TarFile) -> dict[str, Any] | None:
    """Return the ``snqi_diagnostics.json`` payload, or ``None`` if absent in the bundle."""
    for member in archive.getmembers():
        if member.name.endswith("payload/reports/snqi_diagnostics.json") and member.isfile():
            handle = archive.extractfile(member)
            if handle is None:
                return None
            return _json_object(handle.read(), source=member.name)
    return None


def _diagnostics_ordering(diagnostics: Mapping[str, Any]) -> dict[str, int]:
    """Map ``planner_key::kinematics`` -> rank from the diagnostics planner_ordering."""
    ordering: dict[str, int] = {}
    for row in diagnostics.get("planner_ordering", []):
        planner = str(row.get("planner_key", "unknown"))
        kinematics = str(row.get("kinematics", "unknown"))
        rank = int(row.get("rank", 0))
        ordering[f"{planner}::{kinematics}"] = rank
    return ordering


def audit_bundle(  # noqa: C901
    bundle: Path,
    *,
    expected_bundle_sha256: str,
    expected_release_tag: str,
    weights_path: Path = DEFAULT_WEIGHTS,
    baseline_path: Path = DEFAULT_BASELINE,
    source_url: str | None = None,
) -> dict[str, Any]:
    """Audit one publication bundle and return a machine-readable consistency report."""
    state = _AuditState()
    bundle_sha256 = _sha256_path(bundle)
    weights_sha256 = _sha256_path(weights_path)
    baseline_sha256 = _sha256_path(baseline_path)
    weights = load_weight_mapping(weights_path)
    baseline = load_baseline_mapping(baseline_path)

    if bundle_sha256 != expected_bundle_sha256.lower():
        state.add(
            "bundle_sha256",
            f"archive sha256 {bundle_sha256} != expected {expected_bundle_sha256.lower()}",
        )
        return _finalize(
            state,
            inputs=_FinalizeInput(
                bundle=bundle,
                bundle_sha256=bundle_sha256,
                weights_sha256=weights_sha256,
                baseline_sha256=baseline_sha256,
                diagnostics_ordering={},
                field_ordering={},
                prov_reconcile=None,
                source_url=source_url,
            ),
        )

    diagnostics_ordering: dict[str, int] = {}
    field_ordering: dict[str, int] = {}
    prov_reconcile: dict[str, Any] | None = None
    with tarfile.open(bundle, mode="r:gz") as archive:
        diagnostics = read_snqi_diagnostics(archive)
        if diagnostics is None:
            state.add(
                "snqi_diagnostics_absent", "bundle has no payload/reports/snqi_diagnostics.json"
            )
        else:
            diagnostics_ordering = _diagnostics_ordering(diagnostics)
            prov_reconcile = {
                "weights_version": diagnostics.get("weights_version"),
                "baseline_version": diagnostics.get("baseline_version"),
                "weights_sha256": diagnostics.get("weights_sha256"),
                "baseline_sha256": diagnostics.get("baseline_sha256"),
                "weights_sha256_matches_local": diagnostics.get("weights_sha256") == weights_sha256,
                "baseline_sha256_matches_local": diagnostics.get("baseline_sha256")
                == baseline_sha256,
            }
            if diagnostics.get("weights_sha256") != weights_sha256:
                state.add(
                    "weights_sha256",
                    "local SNQI weights do not match snqi_diagnostics.json weights_sha256",
                )
            if diagnostics.get("baseline_sha256") != baseline_sha256:
                state.add(
                    "baseline_sha256",
                    "local SNQI baseline does not match snqi_diagnostics.json baseline_sha256",
                )

        for member in archive.getmembers():
            match = _EPISODE_PATH.fullmatch(member.name)
            if not match or not member.isfile():
                continue
            arm = match.group("arm")
            handle = archive.extractfile(member)
            if handle is None:
                continue
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                row_source = f"{member.name}:{line_number}"
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

    # Aggregate planner-ordering from the per-episode field basis vs the diagnostics basis.
    field_means = {
        arm: (state.per_arm_field_sum[arm] / state.per_arm_field_count[arm])
        if state.per_arm_field_count[arm]
        else 0.0
        for arm in state.per_arm_field_sum
    }
    recomputed_means = {
        arm: (state.per_arm_recomputed_sum[arm] / state.per_arm_recomputed_count[arm])
        if state.per_arm_recomputed_count[arm]
        else 0.0
        for arm in state.per_arm_recomputed_sum
    }
    field_ranked = sorted(field_means, key=lambda a: (-field_means[a], a))
    recomputed_ranked = sorted(recomputed_means, key=lambda a: (-recomputed_means[a], a))
    # Map arm -> diagnostics-style key for ordering comparison.
    field_ordering = {_planner_key(a): rank for rank, a in enumerate(field_ranked, start=1)}
    recomputed_ordering = {
        _planner_key(a): rank for rank, a in enumerate(recomputed_ranked, start=1)
    }
    # The per-episode field and the curvature-aware recompute must agree (they use the
    # same scalarizer); and that ordering must match the diagnostics planner_ordering.
    if field_ordering != recomputed_ordering:
        state.add(
            "field_vs_recomputed_ordering",
            "per-episode field arm ordering disagrees with curvature-aware recompute ordering",
        )
    if diagnostics_ordering and field_ordering != diagnostics_ordering:
        state.add(
            "field_vs_diagnostics_ordering",
            "per-episode field arm ordering disagrees with snqi_diagnostics.json planner_ordering",
        )

    return _finalize(
        state,
        inputs=_FinalizeInput(
            bundle=bundle,
            bundle_sha256=bundle_sha256,
            weights_sha256=weights_sha256,
            baseline_sha256=baseline_sha256,
            diagnostics_ordering=diagnostics_ordering,
            field_ordering=field_ordering,
            prov_reconcile=prov_reconcile,
            source_url=source_url,
        ),
    )


@dataclass(frozen=True)
class _FinalizeInput:
    """Inputs needed to assemble the consistency audit report."""

    bundle: Path
    bundle_sha256: str
    weights_sha256: str
    baseline_sha256: str
    diagnostics_ordering: dict[str, int]
    field_ordering: dict[str, int]
    prov_reconcile: dict[str, Any] | None
    source_url: str | None


def _finalize(state: _AuditState, *, inputs: _FinalizeInput) -> dict[str, Any]:
    status = "pass" if not state.violation_counts else "fail"
    return {
        "schema_version": SCHEMA_VERSION,
        "review_marker": "AI-GENERATED NEEDS-REVIEW",
        "status": status,
        "result_classification": (
            "snqi_field_consistent" if status == "pass" else "snqi_field_inconsistent"
        ),
        "evidence_grade": "diagnostic-only",
        "diagnostic_only": True,
        "benchmark_promotion": False,
        "paper_facing": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "source": {
            "bundle_name": inputs.bundle.name,
            "bundle_sha256": inputs.bundle_sha256,
            "release_asset_url": inputs.source_url,
        },
        "integrity": {
            "snqi_weights_sha256": inputs.weights_sha256,
            "snqi_baseline_sha256": inputs.baseline_sha256,
            "provenance_reconcile": inputs.prov_reconcile,
        },
        "counts": {
            "rows": state.rows,
            "episode_field_present": state.episode_field_present,
            "snqi_recomputed_rows": state.snqi_recomputed_rows,
            "arms": len(state.per_arm_field_count),
        },
        "ordering": {
            "field_planner_ordering": inputs.field_ordering,
            "diagnostics_planner_ordering": inputs.diagnostics_ordering,
        },
        "violation_count": sum(state.violation_counts.values()),
        "violation_counts": dict(sorted(state.violation_counts.items())),
        "violations": state.violations,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--expected-release-tag", required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--source-url")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line SNQI-field consistency audit."""
    args = _parser().parse_args(argv)
    try:
        report = audit_bundle(
            args.bundle,
            expected_bundle_sha256=args.expected_bundle_sha256,
            expected_release_tag=args.expected_release_tag,
            weights_path=args.weights,
            baseline_path=args.baseline,
            source_url=args.source_url,
        )
    except (OSError, ValueError) as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "review_marker": "AI-GENERATED NEEDS-REVIEW",
            "status": "fail",
            "result_classification": "snqi_field_inconsistent",
            "evidence_grade": "diagnostic-only",
            "diagnostic_only": True,
            "benchmark_promotion": False,
            "paper_facing": False,
            "claim_boundary": CLAIM_BOUNDARY,
            "violations": [{"code": "audit_error", "message": str(exc)}],
            "violation_count": 1,
            "violation_counts": {"audit_error": 1},
        }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
