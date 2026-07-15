"""Tests for trace-level predicate export lane (issue #5593).

Tests the export contract, CLI round-trip, fail-closed behavior, deterministic
output, and negative-control coverage reporting. The real-campaign smoke test runs
end-to-end against an existing evidence bundle under output/benchmarks/.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.identity.hash_utils import read_jsonl as _read_jsonl
from robot_sf.benchmark.trace_predicate_export import (
    DEGRADED_PREDICATE_STATUSES,
    EXPORT_STATUS_DEGRADED,
    EXPORT_STATUS_EXPORTED,
    EXPORT_STATUS_MISSING,
    MOTIVATED_TRACE_PREDICATES,
    TRACE_PREDICATE_COVERAGE_SCHEMA_VERSION,
    TRACE_PREDICATE_EXPORT_SCHEMA_VERSION,
    TRACE_PREDICATE_MANIFEST_SCHEMA_VERSION,
    TracePredicateExportError,
    build_trace_predicate_coverage_report,
    build_trace_predicate_export,
    build_trace_predicate_manifest,
    validate_trace_predicate_export,
    write_trace_predicate_export,
)
from scripts.tools.export_trace_predicates import main as export_cli_main

REPO_ROOT = Path(__file__).resolve().parents[2]


def _record_with_all_predicates() -> dict[str, Any]:
    """Episode record with all three predicate producers populated."""
    return {
        "episode_id": "ep-001",
        "scenario_id": "test-scenario",
        "seed": 42,
        "algo": "goal",
        "git_hash": "abc123",
        "safety_predicates": {
            "oscillatory_control_predicate": {
                "schema_version": "safety_predicate.oscillatory_control.v1",
                "predicate": "oscillatory_control",
                "evidence_kind": "diagnostic_proxy",
                "oscillation": True,
                "fields": {"heading_rate_sign_changes": 6, "progress_ratio": 0.1},
                "thresholds": {"min_heading_rate_sign_changes": 4, "max_progress_ratio": 0.5},
            },
            "late_evasive_predicate": {
                "schema_version": "safety_predicate.late_evasive.v2",
                "predicate": "late_evasive",
                "evidence_kind": "diagnostic_proxy",
                "late_evasive": True,
                "fields": {
                    "first_hazard_visible_step": 10,
                    "minimum_distance_m": 0.3,
                    "response_latency_s": None,
                    "latency_unavailable_reason": "no_clearance_restoring_action",
                },
                "thresholds": {},
            },
            "occlusion_near_miss_predicate": {
                "schema_version": "safety_predicate.occlusion_near_miss.v1",
                "predicate": "occlusion_near_miss",
                "evidence_kind": "diagnostic_proxy",
                "occlusion_near_miss": False,
                "status": "false",
                "status_reason": None,
                "fields": {"was_occluded_before_min": False, "emergence_step": None},
                "thresholds": {},
            },
        },
    }


def _record_with_no_predicates() -> dict[str, Any]:
    """Episode record with no safety predicates block."""
    return {
        "episode_id": "ep-002",
        "scenario_id": "test-scenario",
        "seed": 43,
        "algo": "goal",
        "git_hash": "abc123",
    }


def _record_with_degraded_predicate() -> dict[str, Any]:
    """Episode record with a degraded (unavailable) predicate."""
    return {
        "episode_id": "ep-003",
        "scenario_id": "test-scenario",
        "seed": 44,
        "algo": "goal",
        "git_hash": "abc123",
        "safety_predicates": {
            "occlusion_near_miss_predicate": {
                "schema_version": "safety_predicate.occlusion_near_miss.v1",
                "predicate": "occlusion_near_miss",
                "evidence_kind": "diagnostic_proxy",
                "occlusion_near_miss": False,
                "status": "unavailable",
                "status_reason": "missing_visibility_evidence",
                "fields": {
                    "was_occluded_before_min": None,
                    "emergence_step": None,
                    "visibility_evidence_status": "unavailable",
                },
                "thresholds": {},
            }
        },
    }


class TestBuildExport:
    """Single-record and batch export contract."""

    def test_record_with_all_predicates(self) -> None:
        """All three predicates should be exported with clean status."""
        rows = build_trace_predicate_export([_record_with_all_predicates()])
        assert len(rows) == 1
        row = rows[0]
        assert row["scenario_id"] == "test-scenario"
        assert row["seed"] == 42
        assert row["planner_id"] == "goal"
        assert row["episode_id"] == "ep-001"
        assert row["software_commit"] == "abc123"
        for predicate, spec in ((p["predicate"], p) for p in MOTIVATED_TRACE_PREDICATES):
            block = row["predicates"][predicate]
            assert block["export_status"] == EXPORT_STATUS_EXPORTED
            assert block["schema_version"] == spec["schema_version"]
            assert "fields" in block

    def test_record_without_predicates_marks_missing(self) -> None:
        """A record without safety_predicates must mark each predicate missing."""
        rows = build_trace_predicate_export([_record_with_no_predicates()])
        row = rows[0]
        for predicate in (s["predicate"] for s in MOTIVATED_TRACE_PREDICATES):
            block = row["predicates"][predicate]
            assert block["export_status"] == EXPORT_STATUS_MISSING
            assert block["reason"] == "predicate_record_absent"

    def test_degraded_predicate_is_flagged(self) -> None:
        """A predicate with a degraded status must be flagged, not silently exported."""
        rows = build_trace_predicate_export([_record_with_degraded_predicate()])
        block = rows[0]["predicates"]["occlusion_near_miss"]
        assert block["export_status"] == EXPORT_STATUS_DEGRADED
        assert block["status"] == "unavailable"
        assert block["status_reason"] == "missing_visibility_evidence"

    def test_missing_episode_identity_fails_closed(self) -> None:
        """Required query identity must not be replaced with synthetic defaults."""
        record = _record_with_all_predicates()
        record.pop("scenario_id")
        with pytest.raises(TracePredicateExportError, match="scenario_id"):
            build_trace_predicate_export([record])

    def test_malformed_identity_types_fail_closed(self) -> None:
        """Required identity fields must be non-empty strings."""
        # scenario_id = 0 (integer)
        record = _record_with_all_predicates()
        record["scenario_id"] = 0
        with pytest.raises(TracePredicateExportError, match="scenario_id must be a string"):
            build_trace_predicate_export([record])

        # algo = False (boolean)
        record = _record_with_all_predicates()
        record["algo"] = False
        with pytest.raises(TracePredicateExportError, match="planner must be a string"):
            build_trace_predicate_export([record])

        # episode_id = dict (mapping)
        record = _record_with_all_predicates()
        record["episode_id"] = {"id": "ep-001"}
        with pytest.raises(TracePredicateExportError, match="episode_id must be a string"):
            build_trace_predicate_export([record])

        # blank scenario_id
        record = _record_with_all_predicates()
        record["scenario_id"] = "   "
        with pytest.raises(
            TracePredicateExportError, match="scenario_id must be a non-empty string"
        ):
            build_trace_predicate_export([record])

    def test_malformed_predicate_record_fails_closed(self) -> None:
        """Malformed producer payloads must not be promoted as exported evidence."""
        record = _record_with_all_predicates()
        record["safety_predicates"]["late_evasive_predicate"]["fields"] = None
        with pytest.raises(TracePredicateExportError, match="fields"):
            build_trace_predicate_export([record])

        record = _record_with_all_predicates()
        record["safety_predicates"]["occlusion_near_miss_predicate"]["status"] = "wat"
        with pytest.raises(TracePredicateExportError, match="invalid status"):
            build_trace_predicate_export([record])

    def test_batch_deterministic_order(self) -> None:
        """Batch export is deterministically ordered by run/scenario/seed/episode."""
        records = [
            _record_with_all_predicates(),
            _record_with_degraded_predicate(),
            _record_with_no_predicates(),
        ]
        rows = build_trace_predicate_export(records, run_id="goal__dd")
        keys = [(r["run_id"], r["scenario_id"], r["seed"], r["episode_id"]) for r in rows]
        assert keys == sorted(keys)


class TestManifestAndCoverage:
    """Manifest gap tracking and coverage report (exported vs motivated)."""

    def test_manifest_records_gaps(self) -> None:
        """Manifest must count present/degraded/missing and list every gap."""
        rows = build_trace_predicate_export(
            [
                _record_with_all_predicates(),
                _record_with_no_predicates(),
                _record_with_degraded_predicate(),
            ]
        )
        manifest = build_trace_predicate_manifest(
            export_rows=rows, sources=["bundle.jsonl"], release="test-release"
        )
        assert manifest["schema_version"] == TRACE_PREDICATE_MANIFEST_SCHEMA_VERSION
        assert manifest["episode_count"] == 3
        assert manifest["complete"] is False
        occ = next(
            p for p in manifest["predicate_types"] if p["predicate"] == "occlusion_near_miss"
        )
        # One clean (present) episode means the predicate is exported at the release level;
        # the degraded/missing episodes are tracked as gaps instead.
        assert occ["episodes_present"] == 1
        assert occ["episodes_degraded"] == 1
        assert occ["export_status"] == EXPORT_STATUS_EXPORTED
        missing = [g for g in manifest["gaps"] if g["export_status"] == EXPORT_STATUS_MISSING]
        # ep-002 has all three missing; ep-003 has two missing (occlusion is degraded).
        assert len(missing) == 5

    def test_coverage_reports_exported_vs_motivated(self) -> None:
        """Coverage report enumerates exported vs motivated-not-exported."""
        rows = build_trace_predicate_export(
            [
                _record_with_all_predicates(),
                _record_with_no_predicates(),
                _record_with_degraded_predicate(),
            ]
        )
        manifest = build_trace_predicate_manifest(
            export_rows=rows, sources=["bundle.jsonl"], release="test-release"
        )
        coverage = build_trace_predicate_coverage_report(manifest=manifest)
        assert coverage["schema_version"] == TRACE_PREDICATE_COVERAGE_SCHEMA_VERSION
        assert coverage["summary"]["motivated_count"] == len(MOTIVATED_TRACE_PREDICATES)
        # 1 present + 1 degraded + 1 missing, per predicate: all three motivated
        # predicates have at least one exported (present) episode.
        assert set(coverage["exported_predicates"]) == {
            p["predicate"] for p in MOTIVATED_TRACE_PREDICATES
        }
        assert coverage["motivated_not_exported"] == []

    def test_negative_control_deliberately_omitted(self) -> None:
        """A fixture omitting two families must flag them as not-exported.

        Negative-control acceptance criterion: the coverage report must correctly
        flag at least one deliberately-omitted predicate, verifying the 'flag what is
        missing' behavior rather than only the happy path.
        """
        partial = {
            "episode_id": "ep-partial",
            "scenario_id": "s",
            "seed": 1,
            "algo": "goal",
            "safety_predicates": {
                "oscillatory_control_predicate": {
                    "schema_version": "safety_predicate.oscillatory_control.v1",
                    "predicate": "oscillatory_control",
                    "evidence_kind": "diagnostic_proxy",
                    "oscillation": True,
                    "fields": {"heading_rate_sign_changes": 6, "progress_ratio": 0.1},
                    "thresholds": {},
                }
            },
        }
        rows = build_trace_predicate_export([partial])
        manifest = build_trace_predicate_manifest(
            export_rows=rows, sources=["bundle.jsonl"], release="partial-release"
        )
        coverage = build_trace_predicate_coverage_report(manifest=manifest)

        assert "oscillatory_control" in coverage["exported_predicates"]
        assert "late_evasive" in coverage["motivated_not_exported"]
        assert "occlusion_near_miss" in coverage["motivated_not_exported"]
        assert coverage["summary"]["exported_count"] == 1
        assert coverage["summary"]["motivated_not_exported_count"] == 2


class TestDeterminism:
    """Deterministic, checksum/byte-identical output."""

    def test_deterministic_jsonl(self) -> None:
        """Same rows serialize to byte-identical JSON-lines."""
        rows = build_trace_predicate_export(
            [
                _record_with_all_predicates(),
                _record_with_degraded_predicate(),
                _record_with_no_predicates(),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "a.jsonl"
            p2 = Path(tmp) / "b.jsonl"
            m1 = {
                "schema_version": "x",
                "release": "r",
                "sources": ["s"],
                "episode_count": 3,
                "predicate_types": [],
                "gaps": [],
                "complete": True,
            }
            c1 = {
                "schema_version": "c",
                "release": "r",
                "motivated_predicates": [],
                "exported_predicates": [],
                "motivated_not_exported": [],
                "per_predicate": {},
                "summary": {},
            }
            write_trace_predicate_export(
                export_rows=rows,
                manifest=m1,
                coverage=c1,
                export_jsonl=p1,
                manifest_json=Path(tmp) / "m1.json",
                coverage_json=Path(tmp) / "c1.json",
            )
            write_trace_predicate_export(
                export_rows=rows,
                manifest=m1,
                coverage=c1,
                export_jsonl=p2,
                manifest_json=Path(tmp) / "m2.json",
                coverage_json=Path(tmp) / "c2.json",
            )
            assert p1.read_bytes() == p2.read_bytes()


class TestValidation:
    """Contract validation fails closed on bad rows."""

    def test_valid_row_has_no_violations(self) -> None:
        rows = build_trace_predicate_export([_record_with_all_predicates()])
        assert validate_trace_predicate_export(rows[0]) == []

    def test_missing_required_field_fails(self) -> None:
        bad = dict(build_trace_predicate_export([_record_with_all_predicates()])[0].items())
        del bad["episode_id"]
        violations = validate_trace_predicate_export(bad)
        assert any("episode_id" in v for v in violations)

    def test_missing_predicate_requires_reason(self) -> None:
        bad = build_trace_predicate_export([_record_with_no_predicates()])[0]
        bad["predicates"]["oscillatory_control"].pop("reason", None)
        violations = validate_trace_predicate_export(bad)
        assert any(
            "missing predicate oscillatory_control must carry a reason" in v for v in violations
        )

    def test_degraded_predicate_requires_status(self) -> None:
        bad = build_trace_predicate_export([_record_with_degraded_predicate()])[0]
        bad["predicates"]["occlusion_near_miss"].pop("status", None)
        violations = validate_trace_predicate_export(bad)
        assert any(
            "degraded predicate occlusion_near_miss must carry a status" in v for v in violations
        )


class TestCLIIntegration:
    """End-to-end CLI test with a synthetic JSONL bundle."""

    def test_cli_exports_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "episodes.jsonl"
            bundle.write_text(
                "\n".join(
                    json.dumps(r)
                    for r in (
                        _record_with_all_predicates(),
                        _record_with_degraded_predicate(),
                        _record_with_no_predicates(),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            out = Path(tmp) / "out"
            code = export_cli_main(
                ["--bundle", str(bundle), "--release", "cli-test", "--output-dir", str(out)]
            )
            assert code == 0
            assert (out / "trace_predicate_export.jsonl").exists()
            assert (out / "trace_predicate_manifest.json").exists()
            assert (out / "trace_predicate_coverage.json").exists()

            rows = [
                json.loads(line)
                for line in (out / "trace_predicate_export.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            assert len(rows) == 3
            manifest = json.loads(
                (out / "trace_predicate_manifest.json").read_text(encoding="utf-8")
            )
            assert manifest["complete"] is False
            coverage = json.loads(
                (out / "trace_predicate_coverage.json").read_text(encoding="utf-8")
            )
            assert coverage["summary"]["exported_count"] == 3

    def test_cli_missing_bundle_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = export_cli_main(
                [
                    "--bundle",
                    str(Path(tmp) / "nope.jsonl"),
                    "--release",
                    "r",
                    "--output-dir",
                    str(Path(tmp) / "o"),
                ]
            )
            assert code == 1

    def test_cli_rejects_partial_bundle_export(self) -> None:
        """A malformed requested source must not produce successful partial artifacts."""
        with tempfile.TemporaryDirectory() as tmp:
            valid = Path(tmp) / "valid.jsonl"
            valid.write_text(json.dumps(_record_with_all_predicates()) + "\n", encoding="utf-8")
            malformed = Path(tmp) / "malformed.jsonl"
            malformed.write_text("{not-json}\n", encoding="utf-8")
            out = Path(tmp) / "out"

            code = export_cli_main(
                [
                    "--bundle",
                    str(valid),
                    "--bundle",
                    str(malformed),
                    "--release",
                    "partial-input",
                    "--output-dir",
                    str(out),
                ]
            )

            assert code == 1
            assert not out.exists()


class TestRealCampaignSmoke:
    """Real-campaign smoke test against an existing evidence bundle.

    Runs the export end-to-end on one existing campaign bundle that already carries a
    ``safety_predicates`` block, asserting the lane produces a valid export, a manifest
    with the expected predicate presence, and a coverage report.
    """

    RELATIVE_BUNDLE = Path(
        "output/benchmarks/camera_ready/issue4206_trace_capable_h600_rerun_20260704/"
        "runs/socnav_sampling__differential_drive/episodes.jsonl"
    )

    @classmethod
    def _bundle_path(cls) -> Path | None:
        """Resolve the real campaign bundle, checking the current repo and parent checkout.

        ``output/`` is gitignored and therefore absent from fresh worktrees; the bundle
        commonly lives in the main checkout. Return ``None`` when unavailable so tests skip.
        """
        for root in (REPO_ROOT,):
            bundle = root / cls.RELATIVE_BUNDLE
            if bundle.is_file():
                return bundle
        # Sibling main checkout. The worktree container here is
        # ``.../robot_sf_ll7.worktrees/<branch>/``; the main checkout is ``.../robot_sf_ll7``.
        container = REPO_ROOT.parent
        if container.name.endswith(".worktrees"):
            main_root = container.with_name(container.name[: -len(".worktrees")])
            alt = main_root / cls.RELATIVE_BUNDLE
            if alt.is_file():
                return alt
        return None

    def test_real_bundle_export_succeeds(self) -> None:
        import pytest

        bundle = self._bundle_path()
        if bundle is None:
            pytest.skip("real campaign bundle not present")
        episodes = _read_jsonl(bundle)
        assert episodes, "bundle must contain episode records"

        rows = build_trace_predicate_export(episodes, run_id="socnav_sampling__differential_drive")
        assert len(rows) == len(episodes)
        for row in rows:
            violations = validate_trace_predicate_export(row)
            assert violations == [], f"{row['episode_id']} failed validation: {violations}"
            # Every motivated predicate block must be present and fail-closed typed.
            for predicate in (s["predicate"] for s in MOTIVATED_TRACE_PREDICATES):
                assert row["predicates"][predicate]["export_status"] in (
                    EXPORT_STATUS_EXPORTED,
                    EXPORT_STATUS_DEGRADED,
                    EXPORT_STATUS_MISSING,
                )

        manifest = build_trace_predicate_manifest(
            export_rows=rows,
            sources=[str(bundle)],
            release="issue4206_trace_capable_h600_rerun_20260704",
        )
        coverage = build_trace_predicate_coverage_report(manifest=manifest)
        # The real bundle has all three predicate producers present for most episodes.
        assert coverage["summary"]["exported_count"] == 3
        assert coverage["summary"]["motivated_not_exported_count"] == 0
        occ = next(
            p for p in manifest["predicate_types"] if p["predicate"] == "occlusion_near_miss"
        )
        # Real data shows degraded (not_applicable) occlusion predicates -> gaps exist.
        assert occ["episodes_degraded"] >= 1
        assert manifest["complete"] is False
        assert manifest["gaps"]

    def test_real_bundle_cli_roundtrip(self) -> None:
        import pytest

        bundle = self._bundle_path()
        if bundle is None:
            pytest.skip("real campaign bundle not present")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code = export_cli_main(
                [
                    "--bundle",
                    str(bundle),
                    "--release",
                    "issue4206_trace_capable_h600_rerun_20260704",
                    "--output-dir",
                    str(out),
                ]
            )
            assert code == 0
            rows = [
                json.loads(line)
                for line in (out / "trace_predicate_export.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            assert rows
            coverage = json.loads(
                (out / "trace_predicate_coverage.json").read_text(encoding="utf-8")
            )
            assert coverage["summary"]["exported_count"] == 3


def test_degraded_status_constant_matches_producers() -> None:
    """Degraded statuses must include the statuses the producers actually emit."""
    assert "not_applicable" in DEGRADED_PREDICATE_STATUSES
    assert "unavailable" in DEGRADED_PREDICATE_STATUSES


def test_schema_versions_versioned() -> None:
    """Export/manifest/coverage schema versions follow the vN convention."""
    assert TRACE_PREDICATE_EXPORT_SCHEMA_VERSION == "trace_predicate_export.v1"
    assert TRACE_PREDICATE_MANIFEST_SCHEMA_VERSION == "trace_predicate_manifest.v1"
    assert TRACE_PREDICATE_COVERAGE_SCHEMA_VERSION == "trace_predicate_coverage.v1"
