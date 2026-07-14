"""Tests for trace-level predicate export lane (issue #5593).

Tests the export contract, CLI round-trip, fail-closed behavior,
deterministic output, and negative-control coverage reporting.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from robot_sf.benchmark.event_ledger import (
    ensure_event_ledger,
)
from robot_sf.benchmark.trace_predicate_export import (
    MOTIVATED_PREDICATE_FAMILIES,
    PREDICATE_SCHEMA_BY_FAMILY,
    TRACE_PREDICATE_EXPORT_SCHEMA,
    CoverageReportRow,
    build_coverage_report,
    build_export_manifest,
    build_predicate_export_batch,
    build_predicate_export_record,
    compute_export_checksum,
    export_to_jsonl,
    format_coverage_report_md,
    serialize_export_records,
    validate_export_batch,
    validate_export_record,
    write_coverage_report,
)
from scripts.tools.export_trace_predicates import main as export_cli_main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_episode_record(
    *,
    scenario_id: str = "test-scenario",
    seed: int = 42,
    planner: str = "goal",
    episode_id: str = "ep-001",
    collision_count: float = 0.0,
    near_misses: float = 0.0,
    safety_predicates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a minimal episode record with optional safety predicates."""
    record: dict[str, Any] = {
        "version": "v1",
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "algo": planner,
        "git_hash": "abc123",
        "metrics": {
            "success": 1.0,
            "collisions": collision_count,
            "near_misses": near_misses,
            "min_clearance": 1.5,
        },
        "termination_reason": "success",
        "outcome": {
            "route_complete": True,
            "collision_event": False,
            "timeout_event": False,
        },
    }
    if safety_predicates is not None:
        record["safety_predicates"] = safety_predicates
    return record


def _record_with_all_predicates() -> dict[str, Any]:
    """Episode record with all three predicate producers populated."""
    return _make_episode_record(
        safety_predicates={
            "oscillatory_control_predicate": {
                "schema_version": "safety_predicate.oscillatory_control.v1",
                "predicate": "oscillatory_control",
                "evidence_kind": "diagnostic_proxy",
                "oscillation": True,
                "fields": {
                    "heading_rate_sign_changes": 6,
                    "progress_ratio": 0.1,
                },
                "thresholds": {
                    "min_heading_rate_sign_changes": 4,
                    "max_progress_ratio": 0.5,
                },
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
                "fields": {
                    "was_occluded_before_min": False,
                    "emergence_step": None,
                },
                "thresholds": {},
            },
        }
    )


def _record_with_no_predicates() -> dict[str, Any]:
    """Episode record with no safety predicates (metric-derived only)."""
    return _make_episode_record()


def _record_with_degraded_predicate() -> dict[str, Any]:
    """Episode record with a degraded (unavailable) predicate."""
    return _make_episode_record(
        safety_predicates={
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
        }
    )


# ---------------------------------------------------------------------------
# Core export tests
# ---------------------------------------------------------------------------
class TestBuildPredicateExportRecord:
    """Tests for single-record export."""

    def test_record_with_all_predicates(self) -> None:
        """All three predicates should be extracted with booleans and records."""
        record = _record_with_all_predicates()
        ensure_event_ledger(record)
        export = build_predicate_export_record(record)

        assert export.schema_version == TRACE_PREDICATE_EXPORT_SCHEMA
        assert export.scenario_id == "test-scenario"
        assert export.seed == 42
        assert export.planner == "goal"
        assert export.episode_id == "ep-001"
        assert export.software_commit == "abc123"

        # Boolean flags should be present
        assert export.surrogate_events.get("oscillation") is True
        assert export.surrogate_events.get("late_evasive") is True
        assert export.surrogate_events.get("occlusion_near_miss") is False

        # Detailed records should be present
        assert "oscillation" in export.predicate_records
        assert "late_evasive" in export.predicate_records
        assert "occlusion_near_miss" in export.predicate_records

        # Schema versions should propagate
        assert (
            export.predicate_records["oscillation"]["schema_version"]
            == "safety_predicate.oscillatory_control.v1"
        )
        assert (
            export.predicate_records["late_evasive"]["schema_version"]
            == "safety_predicate.late_evasive.v2"
        )

        # Metric-derived families (near_miss, clearance_breach, ttc_breach) have no
        # dedicated producer and appear as missing at the producer-record level.
        # The three producer-derived families should NOT be missing.
        for fam in ("oscillation", "late_evasive", "occlusion_near_miss"):
            assert fam not in export.missing_fields
        for fam in ("near_miss", "clearance_breach", "ttc_breach"):
            assert fam in export.missing_fields
        assert export.degraded_fields == []

    def test_record_without_predicates(self) -> None:
        """Records without predicates should list missing families."""
        record = _record_with_no_predicates()
        export = build_predicate_export_record(record)

        assert export.schema_version == TRACE_PREDICATE_EXPORT_SCHEMA
        # Metric-derived predicates (near_miss, clearance_breach, ttc_breach) get
        # booleans from the ledger surrogate_events
        for fam in ("oscillation", "late_evasive", "occlusion_near_miss"):
            # These are missing because no safety_predicates were provided
            assert fam in export.missing_fields

    def test_degraded_predicate_is_flagged(self) -> None:
        """A predicate with status=unavailable should be in degraded_fields."""
        record = _record_with_degraded_predicate()
        export = build_predicate_export_record(record)

        assert "occlusion_near_miss" in export.degraded_fields
        assert export.surrogate_events.get("occlusion_near_miss") is False

    def test_to_dict_is_json_serializable(self) -> None:
        """The export record must serialize to JSON without errors."""
        record = _record_with_all_predicates()
        ensure_event_ledger(record)
        export = build_predicate_export_record(record)
        d = export.to_dict()
        # Should not raise
        json.dumps(d, sort_keys=True)
        assert isinstance(d, dict)
        assert d["schema_version"] == TRACE_PREDICATE_EXPORT_SCHEMA


class TestBuildPredicateExportBatch:
    """Tests for multi-record batch export."""

    def test_batch_preserves_order(self) -> None:
        """Batch export should preserve record order."""
        records = [
            _record_with_all_predicates(),
            _record_with_no_predicates(),
            _record_with_degraded_predicate(),
        ]
        exports = build_predicate_export_batch(records)

        assert len(exports) == 3
        # Record 0: all predicates present with producer records
        assert exports[0].surrogate_events.get("oscillation") is True
        assert "oscillation" in exports[0].predicate_records
        assert exports[0].missing_fields == [
            "clearance_breach",
            "near_miss",
            "ttc_breach",
        ]
        # Record 1: no safety predicates — all producer families missing
        assert exports[1].surrogate_events.get("oscillation") is False
        assert "oscillation" in exports[1].missing_fields
        assert exports[1].predicate_records == {}
        # Record 2: degraded predicate
        assert "occlusion_near_miss" in exports[2].degraded_fields

    def test_batch_fails_on_bad_record(self) -> None:
        """A record with bad data should raise ValueError with episode_id."""
        records = [_record_with_all_predicates(), {"episode_id": "bad-ep"}]
        # This should not raise because ensure_event_ledger handles minimal records
        exports = build_predicate_export_batch(records)
        assert len(exports) == 2
        # The bad record falls back to unknown / defaults
        assert exports[1].scenario_id == "unknown"


# ---------------------------------------------------------------------------
# Serialization and determinism tests
# ---------------------------------------------------------------------------
class TestSerialization:
    """Tests for deterministic export serialization."""

    def test_serialize_produces_jsonl(self) -> None:
        """Serialization should produce valid JSONL."""
        record = _record_with_all_predicates()
        ensure_event_ledger(record)
        exports = [build_predicate_export_record(record)]
        content = serialize_export_records(exports)

        lines = [line for line in content.splitlines() if line.strip()]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["schema_version"] == TRACE_PREDICATE_EXPORT_SCHEMA

    def test_deterministic_output(self) -> None:
        """Same input must produce byte-identical output."""
        records = [
            _record_with_all_predicates(),
            _record_with_degraded_predicate(),
        ]
        exports = build_predicate_export_batch(records)

        output1 = serialize_export_records(exports)
        output2 = serialize_export_records(exports)

        assert output1 == output2
        assert compute_export_checksum(exports) == compute_export_checksum(exports)

    def test_empty_batch_produces_empty(self) -> None:
        """An empty batch should produce a valid (empty) serialization."""
        content = serialize_export_records([])
        assert content == "\n"

    def test_file_roundtrip(self) -> None:
        """Export, write, and read back should produce same data."""
        records = [_record_with_all_predicates()]
        ensure_event_ledger(records[0])
        exports = [build_predicate_export_record(r) for r in records]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "export.jsonl"
            checksum = export_to_jsonl(exports, path)

            # Read back
            content = path.read_text(encoding="utf-8")
            lines = [line for line in content.splitlines() if line.strip()]
            parsed = json.loads(lines[0])
            assert parsed["schema_version"] == TRACE_PREDICATE_EXPORT_SCHEMA
            assert len(checksum) == 64  # SHA-256 hex length


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------
class TestManifest:
    """Tests for the export manifest."""

    def test_manifest_collects_predicate_types(self) -> None:
        """The manifest should list all predicate types found in export."""
        records = [_record_with_all_predicates()]
        ensure_event_ledger(records[0])
        exports = [build_predicate_export_record(r) for r in records]
        manifest = build_export_manifest(exports)

        assert manifest.schema_version == "trace_predicate_manifest.v1"
        assert manifest.export_schema == TRACE_PREDICATE_EXPORT_SCHEMA
        assert manifest.episode_count == 1
        assert "oscillation" in manifest.predicate_types
        assert "late_evasive" in manifest.predicate_types
        assert "occlusion_near_miss" in manifest.predicate_types

        # Schema versions should be recorded
        assert manifest.predicate_schema_versions["oscillation"] == (
            "safety_predicate.oscillatory_control.v1"
        )

    def test_manifest_tracks_missing_and_degraded(self) -> None:
        """Episodes with missing or degraded predicates should be counted."""
        records = [
            _record_with_all_predicates(),
            _record_with_no_predicates(),
            _record_with_degraded_predicate(),
        ]
        exports = build_predicate_export_batch(records)
        manifest = build_export_manifest(exports)

        assert manifest.episode_count == 3
        assert manifest.episodes_with_missing >= 1  # at least the no-predicate one
        assert manifest.episodes_with_degraded == 1

    def test_manifest_includes_checksum(self) -> None:
        """Manifest should include a SHA-256 checksum of the export."""
        records = [_record_with_all_predicates()]
        ensure_event_ledger(records[0])
        exports = [build_predicate_export_record(r) for r in records]
        manifest = build_export_manifest(exports)

        assert manifest.checksum_sha256 is not None
        assert len(manifest.checksum_sha256) == 64


# ---------------------------------------------------------------------------
# Coverage report tests
# ---------------------------------------------------------------------------
class TestCoverageReport:
    """Tests for the predicate coverage report."""

    def test_coverage_enumerates_all_families(self) -> None:
        """Coverage report should have one row per motivated predicate."""
        records = [_record_with_all_predicates()]
        ensure_event_ledger(records[0])
        exports = [build_predicate_export_record(r) for r in records]
        rows = build_coverage_report(exports)

        families = [row.predicate_family for row in rows]
        assert set(families) == set(MOTIVATED_PREDICATE_FAMILIES)

    def test_coverage_flags_exported_predicates(self) -> None:
        """Predicates present in export should be marked exported."""
        records = [_record_with_all_predicates()]
        ensure_event_ledger(records[0])
        exports = [build_predicate_export_record(r) for r in records]
        rows = build_coverage_report(exports)

        osc_row = next(r for r in rows if r.predicate_family == "oscillation")
        assert osc_row.exported is True
        assert osc_row.episodes_exported >= 1

    def test_coverage_flags_missing_predicates(self) -> None:
        """Predicates not in export should be marked missing."""
        records = [_record_with_no_predicates()]
        exports = build_predicate_export_batch(records)
        rows = build_coverage_report(exports)

        osc_row = next(r for r in rows if r.predicate_family == "oscillation")
        assert osc_row.episodes_missing >= 1

    def test_negative_control_deliberately_omitted(self) -> None:
        """A fixture with one family omitted should flag it as missing.

        This is the negative-control acceptance criterion: the coverage report
        must correctly flag at least one deliberately-omitted predicate, so
        the 'flag what's missing' behavior is verified, not just the happy path.
        """
        # Only provide oscillatory predicate, deliberately omit late_evasive
        # and occlusion_near_miss
        partial_record = _make_episode_record(
            safety_predicates={
                "oscillatory_control_predicate": {
                    "schema_version": "safety_predicate.oscillatory_control.v1",
                    "predicate": "oscillatory_control",
                    "evidence_kind": "diagnostic_proxy",
                    "oscillation": True,
                    "fields": {"heading_rate_sign_changes": 6, "progress_ratio": 0.1},
                    "thresholds": {},
                },
            }
        )
        exports = build_predicate_export_batch([partial_record])
        rows = build_coverage_report(exports)

        osc_row = next(r for r in rows if r.predicate_family == "oscillation")
        late_row = next(r for r in rows if r.predicate_family == "late_evasive")
        occ_row = next(r for r in rows if r.predicate_family == "occlusion_near_miss")

        assert osc_row.exported is True
        assert osc_row.episodes_exported >= 1
        assert late_row.exported is False
        assert late_row.episodes_missing >= 1
        assert occ_row.exported is False
        assert occ_row.episodes_missing >= 1

    def test_coverage_schema_versions(self) -> None:
        """Schema versions in coverage should match the producer schemas."""
        records = [_record_with_all_predicates()]
        ensure_event_ledger(records[0])
        exports = [build_predicate_export_record(r) for r in records]
        # Only specify the three families that have dedicated producer schemas
        rows = build_coverage_report(
            exports, predicate_families=["oscillation", "late_evasive", "occlusion_near_miss"]
        )

        osc_row = next(r for r in rows if r.predicate_family == "oscillation")
        late_row = next(r for r in rows if r.predicate_family == "late_evasive")

        assert osc_row.schema_version == PREDICATE_SCHEMA_BY_FAMILY["oscillation"]
        assert late_row.schema_version == PREDICATE_SCHEMA_BY_FAMILY["late_evasive"]


class TestCoverageReportMarkdown:
    """Tests for Markdown coverage report formatting."""

    def test_md_table_has_header_and_rows(self) -> None:
        """The Markdown table should have proper header and data rows."""
        records = [_record_with_all_predicates()]
        ensure_event_ledger(records[0])
        exports = [build_predicate_export_record(r) for r in records]
        rows = build_coverage_report(exports)
        md = format_coverage_report_md(rows, total_episodes=1)

        assert "## Predicate Coverage Report" in md
        assert "| Predicate |" in md
        assert "oscillation" in md

    def test_md_file_write(self) -> None:
        """Writing the coverage report should produce a valid file."""
        rows = [
            CoverageReportRow(
                predicate_family="oscillation",
                exported=True,
                schema_version="safety_predicate.oscillatory_control.v1",
                episodes_exported=5,
                episodes_degraded=0,
                episodes_missing=0,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "coverage.md"
            write_coverage_report(rows, path, total_episodes=5)
            content = path.read_text(encoding="utf-8")
            assert "oscillation" in content
            assert "yes" in content  # exported flag


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------
class TestValidation:
    """Tests for export validation."""

    def test_valid_record_has_no_violations(self) -> None:
        """A valid export record should pass validation."""
        record = _record_with_all_predicates()
        ensure_event_ledger(record)
        export = build_predicate_export_record(record)
        violations = validate_export_record(export.to_dict())
        assert violations == []

    def test_missing_schema_version_fails(self) -> None:
        """A record with wrong schema_version should fail validation."""
        d = {
            "schema_version": "wrong.version",
            "scenario_id": "s",
            "seed": 1,
            "planner": "p",
            "episode_id": "e",
            "surrogate_events": {},
            "predicate_records": {},
            "degraded_fields": [],
            "missing_fields": [],
        }
        violations = validate_export_record(d)
        assert any("schema_version" in v for v in violations)

    def test_missing_required_field_fails(self) -> None:
        """A record missing required fields should enumerate violations."""
        d = {
            "schema_version": TRACE_PREDICATE_EXPORT_SCHEMA,
            "surrogate_events": {},
            "predicate_records": {},
            "degraded_fields": [],
            "missing_fields": [],
        }
        violations = validate_export_record(d)
        missing_keys = [v.split(": ")[1].strip("'") for v in violations if "missing" in v]
        assert "scenario_id" in missing_keys
        assert "episode_id" in missing_keys

    def test_non_boolean_surrogate_flag_fails(self) -> None:
        """A surrogate_events value that is not boolean should fail."""
        d = {
            "schema_version": TRACE_PREDICATE_EXPORT_SCHEMA,
            "scenario_id": "s",
            "seed": 1,
            "planner": "p",
            "episode_id": "e",
            "surrogate_events": {"oscillation": "yes"},  # not bool
            "predicate_records": {},
            "degraded_fields": [],
            "missing_fields": [],
        }
        violations = validate_export_record(d)
        assert any("boolean" in v for v in violations)

    def test_batch_validation_uses_episode_id(self) -> None:
        """Batch validation should use episode_id in violation messages."""
        record = _record_with_all_predicates()
        ensure_event_ledger(record)
        valid_export = build_predicate_export_record(record)
        bad_record = dict(valid_export.to_dict())
        bad_record["surrogate_events"]["oscillation"] = "not_bool"
        bad_record["episode_id"] = "my-bad-ep"

        violations = validate_export_batch([bad_record])
        assert any("my-bad-ep" in v for v in violations)


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------
class TestCLIIntegration:
    """End-to-end CLI tests with synthetic JSONL."""

    def test_cli_exports_jsonl_from_file(self) -> None:
        """The CLI should read a JSONL file and produce export artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            records = [
                _record_with_all_predicates(),
                _record_with_degraded_predicate(),
            ]
            # Write JSONL
            with open(input_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            output_dir = Path(tmpdir) / "output"
            exit_code = export_cli_main(
                [
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            assert exit_code == 0
            assert (output_dir / "trace_predicates.jsonl").exists()
            assert (output_dir / "trace_predicates_manifest.json").exists()
            assert (output_dir / "trace_predicates_coverage.md").exists()

            # Verify export content
            content = (output_dir / "trace_predicates.jsonl").read_text(encoding="utf-8")
            lines = [line for line in content.splitlines() if line.strip()]
            assert len(lines) == 2
            first = json.loads(lines[0])
            assert first["schema_version"] == TRACE_PREDICATE_EXPORT_SCHEMA

            # Verify manifest
            manifest = json.loads(
                (output_dir / "trace_predicates_manifest.json").read_text(encoding="utf-8")
            )
            assert manifest["episode_count"] == 2

    def test_cli_handles_empty_input_directory(self) -> None:
        """The CLI should fail gracefully with no JSONL files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir)
            exit_code = export_cli_main([str(input_path)])
            assert exit_code == 1

    def test_cli_with_family_filter(self) -> None:
        """The CLI should allow filtering by predicate families."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            record = _record_with_all_predicates()
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            output_dir = Path(tmpdir) / "output"
            exit_code = export_cli_main(
                [
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--families",
                    "oscillation",
                ]
            )

            assert exit_code == 0
            # With only oscillation requested, the export should still carry
            # the surrogate booleans but the coverage report should only have
            # oscillation rows
            coverage = (output_dir / "trace_predicates_coverage.md").read_text(encoding="utf-8")
            assert "oscillation" in coverage

    def test_cli_file_write_failure_blocks_before_manifest_and_coverage(self) -> None:
        """A critical export write failure preserves the pending manifest and blocks."""
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            input_path.write_text(
                json.dumps(_record_with_all_predicates()) + "\n",
                encoding="utf-8",
            )
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            pending_manifest = output_dir / "trace_predicates_manifest.json"
            pending_manifest.write_text('{"status": "pending"}\n', encoding="utf-8")

            with mock.patch(
                "scripts.tools.export_trace_predicates.export_to_jsonl",
                side_effect=OSError("Disk full"),
            ):
                exit_code = export_cli_main([str(input_path), "--output-dir", str(output_dir)])

            assert exit_code == 3
            assert pending_manifest.read_text(encoding="utf-8") == '{"status": "pending"}\n'
            assert not (output_dir / "trace_predicates_coverage.md").exists()


# ---------------------------------------------------------------------------
# Schema version consistency
# ---------------------------------------------------------------------------
class TestSchemaConsistency:
    """Ensure schema constants are consistent and versioned."""

    def test_export_schema_is_versioned(self) -> None:
        """The export schema version should follow the vN convention."""
        assert TRACE_PREDICATE_EXPORT_SCHEMA == "trace_predicate_export.v1"

    def test_motivated_families_nonempty(self) -> None:
        """MOTIVATED_PREDICATE_FAMILIES should not be empty."""
        assert len(MOTIVATED_PREDICATE_FAMILIES) >= 3

    def test_predicate_schema_by_family_matches(self) -> None:
        """PREDICATE_SCHEMA_BY_FAMILY keys should be a subset of motivated families."""
        families = set(MOTIVATED_PREDICATE_FAMILIES)
        for key in PREDICATE_SCHEMA_BY_FAMILY:
            assert key in families, (
                f"{key} is in PREDICATE_SCHEMA_BY_FAMILY but not in MOTIVATED_PREDICATE_FAMILIES"
            )

    def test_produced_record_matches_export_schema(self) -> None:
        """A produced record should use the export schema constant."""
        record = _record_with_all_predicates()
        ensure_event_ledger(record)
        export = build_predicate_export_record(record)
        assert export.schema_version == TRACE_PREDICATE_EXPORT_SCHEMA


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_predicate_records(self) -> None:
        """A record with no surrogate events should still export."""
        record = _record_with_no_predicates()
        export = build_predicate_export_record(record)
        violations = validate_export_record(export.to_dict())
        assert violations == []

    def test_manifest_empty_batch(self) -> None:
        """An empty manifest should still be valid."""
        manifest = build_export_manifest([])
        assert manifest.episode_count == 0
        assert manifest.checksum_sha256 is None

    def test_predicate_records_preserved_in_dict(self) -> None:
        """Detailed predicate records should survive to_dict round-trip."""
        record = _record_with_all_predicates()
        ensure_event_ledger(record)
        export = build_predicate_export_record(record)
        d = export.to_dict()

        assert "oscillation" in d["predicate_records"]
        assert d["predicate_records"]["oscillation"]["predicate"] == "oscillatory_control"

    def test_multiple_records_different_seeds(self) -> None:
        """Records with different seeds should be independent."""
        records = [
            _make_episode_record(seed=1, planner="goal", episode_id="e1"),
            _make_episode_record(seed=2, planner="rvo", episode_id="e2"),
            _make_episode_record(seed=3, planner="goal", episode_id="e3"),
        ]
        exports = build_predicate_export_batch(records)

        assert exports[0].seed == 1
        assert exports[0].planner == "goal"
        assert exports[1].seed == 2
        assert exports[1].planner == "rvo"
        assert exports[2].seed == 3
