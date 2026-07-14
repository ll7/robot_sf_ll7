"""Tests for the publication-bundle SNQI field-consistency release gate (issue #5580)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.metrics import snqi as curvature_aware_snqi
from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    load_baseline_mapping,
    load_weight_mapping,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts/validation/check_release_snqi_field_consistency.py"
WEIGHTS_PATH = ROOT / "configs/benchmarks/snqi_weights_camera_ready_v3.json"
BASELINE_PATH = ROOT / "configs/benchmarks/snqi_baseline_camera_ready_v3.json"

_SPEC = importlib.util.spec_from_file_location("check_release_snqi_field_consistency", SCRIPT_PATH)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

audit_bundle = _MODULE.audit_bundle


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True) + "\n").encode()


def _episode(
    *, collision: bool, seed: int, weights: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    success = not collision
    total = int(collision)
    metrics: dict[str, Any] = {
        "collisions": total,
        "success": success,
        "time_to_goal_norm": 0.5 if success else 1.0,
        "near_misses": 0,
        "comfort_exposure": 0.0,
        "force_exceed_events": 0,
        "jerk_mean": 0.0,
        "curvature_mean": 0.1,
    }
    # The stored per-episode field is the curvature-aware scalarizer used at capture time.
    metrics["snqi"] = curvature_aware_snqi(metrics, weights, baseline_stats=baseline)
    return {
        "episode_id": f"scenario--{seed}",
        "scenario_id": "scenario",
        "seed": seed,
        "metrics": metrics,
    }


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def _build_bundle(
    tmp_path: Path,
    *,
    rows_by_arm: dict[str, list[dict[str, Any]]],
    diag_ordering: list[dict[str, Any]],
) -> tuple[Path, str]:
    weights_sha256 = hashlib.sha256(WEIGHTS_PATH.read_bytes()).hexdigest()
    baseline_sha256 = hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest()
    payloads: dict[str, bytes] = {
        "reports/campaign_summary.json": _json_bytes(
            {"campaign": {"snqi_contract_status": "pass"}}
        ),
    }
    for arm, rows in rows_by_arm.items():
        payloads[f"runs/{arm}/episodes.jsonl"] = b"".join(_json_bytes(r) for r in rows)
    diag = {
        "weights_version": "snqi_weights_camera_ready_v3",
        "baseline_version": "snqi_baseline_camera_ready_v3",
        "weights_sha256": weights_sha256,
        "baseline_sha256": baseline_sha256,
        "planner_ordering": diag_ordering,
    }
    payloads["reports/snqi_diagnostics.json"] = _json_bytes(diag)

    signed = {path: _sha256(data) for path, data in payloads.items()}
    publication_manifest = {
        "publication_channels": {"release_tag": "test-v1"},
        "files": [{"path": path, "sha256": digest} for path, digest in sorted(signed.items())],
    }
    checksum_bytes = "".join(
        f"{digest}  {path}\n" for path, digest in sorted(signed.items())
    ).encode()
    bundle = tmp_path / "fixture.tar.gz"
    root = "fixture"
    with tarfile.open(bundle, "w:gz") as archive:
        _add_bytes(archive, f"{root}/checksums.sha256", checksum_bytes)
        for path, data in payloads.items():
            _add_bytes(archive, f"{root}/payload/{path}", data)
        _add_bytes(archive, f"{root}/publication_manifest.json", _json_bytes(publication_manifest))
    return bundle, hashlib.sha256(bundle.read_bytes()).hexdigest()


def test_consistent_bundle_passes(tmp_path: Path) -> None:
    """A bundle whose per-episode field matches the curvature-aware diagnostics basis passes."""
    weights = load_weight_mapping(WEIGHTS_PATH)
    baseline = load_baseline_mapping(BASELINE_PATH)
    rows = [_episode(collision=False, seed=1, weights=weights, baseline=baseline)]
    mean = sum(r["metrics"]["snqi"] for r in rows) / len(rows)
    diag_ordering = [
        {
            "planner_key": "planner",
            "kinematics": "differential_drive",
            "episode_count": len(rows),
            "mean_snqi": mean,
            "rank": 1,
        }
    ]
    bundle, digest = _build_bundle(
        tmp_path, rows_by_arm={"planner__differential_drive": rows}, diag_ordering=diag_ordering
    )

    report = audit_bundle(bundle, expected_bundle_sha256=digest, expected_release_tag="test-v1")

    assert report["status"] == "pass"
    assert report["violation_count"] == 0
    assert report["counts"]["rows"] == 1
    assert report["counts"]["episode_field_present"] == 1
    assert report["counts"]["snqi_recomputed_rows"] == 1


def test_stored_field_mismatch_with_recompute_fails(tmp_path: Path) -> None:
    """A corrupted stored field that no longer matches the curvature-aware recompute fails."""
    weights = load_weight_mapping(WEIGHTS_PATH)
    baseline = load_baseline_mapping(BASELINE_PATH)
    rows = [_episode(collision=False, seed=1, weights=weights, baseline=baseline)]
    rows[0]["metrics"]["snqi"] += 0.5
    mean = sum(r["metrics"]["snqi"] for r in rows) / len(rows)
    diag_ordering = [
        {
            "planner_key": "planner",
            "kinematics": "differential_drive",
            "episode_count": len(rows),
            "mean_snqi": mean,
            "rank": 1,
        }
    ]
    bundle, digest = _build_bundle(
        tmp_path, rows_by_arm={"planner__differential_drive": rows}, diag_ordering=diag_ordering
    )

    report = audit_bundle(bundle, expected_bundle_sha256=digest, expected_release_tag="test-v1")

    assert report["status"] == "fail"
    assert report["violation_counts"]["snqi_field_recompute_mismatch"] == 1


def test_field_vs_diagnostics_ordering_drift_fails(tmp_path: Path) -> None:
    """When diagnostics ordering contradicts the per-episode field basis, the gate fails.

    This reproduces issue #5580's core concern: the per-episode ``snqi`` field and the
    ``snqi_diagnostics.json`` ``planner_ordering`` must share one SNQI basis. If the
    diagnostics claims a different SNQI-best arm than the field aggregation, the gate fails
    fail-closed so the two surfaces can never drift silently again.
    """
    weights = load_weight_mapping(WEIGHTS_PATH)
    baseline = load_baseline_mapping(BASELINE_PATH)
    a_rows = [_episode(collision=False, seed=1, weights=weights, baseline=baseline)]
    b_rows = [_episode(collision=False, seed=2, weights=weights, baseline=baseline)]
    # Make arm B's per-episode field strictly higher than arm A's.
    b_rows[0]["metrics"]["snqi"] += 0.3
    a_mean = a_rows[0]["metrics"]["snqi"]
    b_mean = b_rows[0]["metrics"]["snqi"]
    # Deliberately inverted planner_ordering: claims A > B, contradicting the field basis.
    diag_ordering = [
        {
            "planner_key": "planner_a",
            "kinematics": "differential_drive",
            "episode_count": 1,
            "mean_snqi": a_mean,
            "rank": 1,
        },
        {
            "planner_key": "planner_b",
            "kinematics": "differential_drive",
            "episode_count": 1,
            "mean_snqi": b_mean,
            "rank": 2,
        },
    ]
    bundle, digest = _build_bundle(
        tmp_path,
        rows_by_arm={
            "planner_a__differential_drive": a_rows,
            "planner_b__differential_drive": b_rows,
        },
        diag_ordering=diag_ordering,
    )

    report = audit_bundle(bundle, expected_bundle_sha256=digest, expected_release_tag="test-v1")

    assert report["status"] == "fail"
    assert "field_vs_diagnostics_ordering" in report["violation_counts"]
    assert report["ordering"]["field_planner_ordering"]["planner_b::differential_drive"] == 1
    assert report["ordering"]["diagnostics_planner_ordering"]["planner_a::differential_drive"] == 1


@pytest.mark.skipif(
    not os.environ.get("ROBOT_SF_0_0_3_BUNDLE"),
    reason="requires the locally-downloaded 0.0.3 publication bundle (ROBOT_SF_0_0_3_BUNDLE)",
)
def test_real_0_0_3_bundle_reproduces_issue_5580_drift() -> None:
    """Reproduce issue #5580 against the real 0.0.3 release bundle.

    The per-episode ``snqi`` field (curvature-aware ``metrics.snqi``) and the
    ``snqi_diagnostics.json`` ``planner_ordering`` (historically ``compute_snqi_v0``) use two
    different SNQI formulas. The field is internally consistent (0 per-episode mismatches) but
    elects a different SNQI-best arm than the diagnostics ordering, so the gate fails closed on
    ``field_vs_diagnostics_ordering``. Run manually after downloading the asset:

        gh release download 0.0.3 -p 'paper_experiment_matrix_v2_h600_s30_extended_release_v0_0_3_final_publication_bundle.tar.gz' -D /tmp
        ROBOT_SF_0_0_3_BUNDLE=/tmp/<asset> uv run pytest tests/validation/test_check_release_snqi_field_consistency.py::test_real_0_0_3_bundle_reproduces_issue_5580_drift
    """
    bundle = Path(os.environ["ROBOT_SF_0_0_3_BUNDLE"])
    # sha256 of the 0.0.3 final publication bundle asset (recorded in issue #5097 evidence).
    expected_sha = "3cfefaaa39aab6cae541cece9573848a7e0afc5e1d9e4c9a7bbf48df2330b1a7"
    report = audit_bundle(
        bundle,
        expected_bundle_sha256=expected_sha,
        expected_release_tag="0.0.3",
    )
    assert report["counts"]["rows"] == 20160
    assert report["counts"]["episode_field_present"] == 20160
    # Per-episode field is internally consistent with the curvature-aware recompute.
    assert report["violation_counts"].get("snqi_field_recompute_mismatch", 0) == 0
    # But it disagrees with the diagnostics planner_ordering basis (the reported drift).
    assert "field_vs_diagnostics_ordering" in report["violation_counts"]
    # The two bases elect different SNQI-best arms.
    field_best = min(
        report["ordering"]["field_planner_ordering"],
        key=report["ordering"]["field_planner_ordering"].get,
    )
    diag_best = min(
        report["ordering"]["diagnostics_planner_ordering"],
        key=report["ordering"]["diagnostics_planner_ordering"].get,
    )
    assert field_best != diag_best
