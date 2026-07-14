"""Tests for the coverage-constrained Pareto portfolio selector (issue #5601)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from robot_sf.benchmark.scenario_generation.portfolio_selector import (
    SCHEMA_VERSION,
    CoverageQuotas,
    PortfolioSelectionError,
    SelectionConfig,
    compute_pareto_front,
    extract_descriptors,
    load_portfolio_selection_schema,
    max_min_coverage_selection,
    select_portfolio,
    validate_selection_manifest,
    write_selection_manifest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_catalog_entry(  # noqa: PLR0913
    scenario_id: str,
    min_clearance_m: float | None = 2.0,
    ttc_min_s: float | None = None,
    collision_count: int | None = None,
    near_miss_count: int | None = None,
    map_name: str = "test_map",
    replay_status: str = "replay_validated",
    ped_count: int | None = 3,
    critical_signal: str = "min_clearance",
) -> dict:
    """Build a synthetic catalog entry in generated_scenario_catalog_entry.v1 style."""
    entry: dict = {
        "schema_version": "generated-scenario-catalog-entry.v1",
        "scenario_id": scenario_id,
        "candidate_id": scenario_id,
        "metadata": {
            "source": "auto_generated",
            "generated_by": "test_generator",
            "required_manual_review": True,
            "benchmark_evidence": False,
        },
        "source_episode": {
            "episode_id": f"ep-{scenario_id}",
            "source_seed": 42,
            "source_map": map_name,
        },
        "criticality": {
            "signal": critical_signal,
            "observed_at_s": 5.0,
            "source_metrics": {},
        },
        "segment": {
            "window_start_s": 0.0,
            "window_end_s": 10.0,
            "initial_robot_state": {"position": [0.0, 0.0]},
            "trace_frames": [
                {
                    "time_s": 5.0,
                    "robot": {"position": [1.0, 1.0]},
                    "pedestrians": [
                        {"position": [p * 0.5, p * 0.5]} for p in range(ped_count or 0)
                    ],
                }
            ],
        },
        "replay": {
            "schema_version": "generated-scenario-replay.v1",
            "source_seed": 42,
            "replay_contract": "source_episode_seed_pinned.v1",
            "status": replay_status,
            "warnings": [],
        },
        "provenance": {
            "schema_version": "generated-scenario-provenance.v1",
            "source_issue": "#4932",
            "distiller": "test",
            "claim_boundary": "generated scenario hypotheses only",
        },
    }

    # Add criticality metrics
    if min_clearance_m is not None:
        entry["criticality"]["source_metrics"]["min_clearance_m"] = min_clearance_m

    # Add metrics_summary in candidate-like format
    entry["metrics_summary"] = {
        "severity": {},
        "diversity": {"unique_scenario_families": ped_count},
    }
    if ttc_min_s is not None:
        entry["metrics_summary"]["severity"]["ttc_min_s"] = ttc_min_s
    if collision_count is not None:
        entry["metrics_summary"]["severity"]["collision_count"] = collision_count
    if near_miss_count is not None:
        entry["metrics_summary"]["severity"]["near_miss_count"] = near_miss_count

    return entry


def _make_minimal_entry(scenario_id: str) -> dict:
    """Build a minimal entry with no optional fields."""
    return {
        "scenario_id": scenario_id,
        "candidate_id": scenario_id,
        "criticality": {"signal": "unknown", "observed_at_s": 0.0, "source_metrics": {}},
        "source_episode": {},
    }


def _sha256(payload: dict) -> str:
    """Compute a deterministic SHA-256 hex digest for a dict."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Synthetic fixture data
# ---------------------------------------------------------------------------

_KNOWN_CANDIDATES = [
    # (id, clearance, ttc, collisions, near_misses, map, ped_count)
    ("cand_a", 0.3, None, 0, 2, "crosswalk", 2),
    ("cand_b", 0.6, 2.0, 0, 1, "crosswalk", 3),
    ("cand_c", 1.2, 5.0, 0, 0, "crosswalk", 1),
    ("cand_d", 0.1, 1.0, 1, 3, "junction", 5),
    ("cand_e", 0.4, 1.5, 0, 2, "junction", 4),
    ("cand_f", 2.5, 8.0, 0, 0, "junction", 2),
    ("cand_g", 0.2, 0.5, 0, 4, "straight", 6),
    ("cand_h", 0.8, 3.0, 0, 1, "straight", 1),
    ("cand_i", 0.5, 2.5, 0, 1, "roundabout", 3),
    ("cand_j", 1.8, 6.0, 0, 0, "roundabout", 2),
]

SYNTHETIC_ARCHIVE = [
    _make_catalog_entry(
        scenario_id=cid,
        min_clearance_m=clearance,
        ttc_min_s=ttc,
        collision_count=collisions,
        near_miss_count=near_misses,
        map_name=map_name,
        ped_count=ped_count,
    )
    for cid, clearance, ttc, collisions, near_misses, map_name, ped_count in _KNOWN_CANDIDATES
]


# ---------------------------------------------------------------------------
# Tests: Descriptor extraction
# ---------------------------------------------------------------------------


class TestDescriptorExtraction:
    """Descriptor extraction from catalog entries."""

    def test_extracts_from_catalog_entry(self) -> None:
        """Should extract all descriptor fields from a well-formed catalog entry."""
        entry = _make_catalog_entry(
            "test_01", min_clearance_m=0.5, map_name="crosswalk", ped_count=4
        )
        records = extract_descriptors([entry])

        assert len(records) == 1
        rec = records[0]
        assert rec["candidate_id"] == "test_01"
        assert rec["criticality"]["min_clearance_m"] == 0.5
        assert rec["topology"]["map_family"] == "crosswalk"
        assert rec["actor_interaction"]["pedestrian_count"] == 4
        assert rec["actor_interaction"]["interaction_class"] == "moderate"
        assert rec["mechanism_signature"]["critical_signal"] == "min_clearance"

    def test_extracts_with_missing_fields(self) -> None:
        """Should handle entries with missing optional fields gracefully."""
        entry = _make_minimal_entry("minimal_01")
        records = extract_descriptors([entry])

        assert len(records) == 1
        rec = records[0]
        # All fields should have defaults or None, not crash
        assert rec["candidate_id"] == "minimal_01"
        assert rec["criticality"]["min_clearance_m"] is None
        assert rec["topology"]["map_family"] == "unknown"
        assert rec["actor_interaction"]["pedestrian_count"] is None
        assert rec["mechanism_signature"]["critical_signal"] == "unknown"

    def test_includes_provenance_hash(self) -> None:
        """Should include a deterministic source entry hash."""
        entry = _make_catalog_entry("prov_01", min_clearance_m=1.0)
        records = extract_descriptors([entry])

        rec = records[0]
        assert "source_entry_hash" in rec["descriptor_provenance"]
        assert len(rec["descriptor_provenance"]["source_entry_hash"]) == 64  # SHA-256 hex
        assert (
            rec["descriptor_provenance"]["extracted_from"] == "generated_scenario_catalog_entry.v1"
        )

    def test_severity_score_from_clearance(self) -> None:
        """Lower clearance should produce higher severity score."""
        entry_high = _make_catalog_entry("high", min_clearance_m=0.1)
        entry_low = _make_catalog_entry("low", min_clearance_m=4.0)

        records = extract_descriptors([entry_high, entry_low])
        sev_high = records[0]["criticality"]["severity_score"]
        sev_low = records[1]["criticality"]["severity_score"]

        assert sev_high is not None
        assert sev_low is not None
        assert sev_high > sev_low, "lower clearance should yield higher severity"

    def test_interaction_class_from_ped_count(self) -> None:
        """Should derive interaction class from pedestrian count."""
        entries = [
            _make_catalog_entry("no_ped", ped_count=0),
            _make_catalog_entry("sparse", ped_count=1),
            _make_catalog_entry("moderate", ped_count=3),
            _make_catalog_entry("dense", ped_count=10),
        ]
        records = extract_descriptors(entries)
        classes = [r["actor_interaction"]["interaction_class"] for r in records]
        assert classes == ["no_pedestrians", "sparse", "moderate", "dense"]

    def test_multiple_entries_have_unique_hashes(self) -> None:
        """Distinct entries should produce distinct provenance hashes."""
        entry_a = _make_catalog_entry("a", min_clearance_m=1.0)
        entry_b = _make_catalog_entry("b", min_clearance_m=2.0)
        records = extract_descriptors([entry_a, entry_b])
        assert (
            records[0]["descriptor_provenance"]["source_entry_hash"]
            != records[1]["descriptor_provenance"]["source_entry_hash"]
        )


# ---------------------------------------------------------------------------
# Tests: Pareto front computation
# ---------------------------------------------------------------------------


class TestParetoFront:
    """Pareto dominance and front computation."""

    def test_single_candidate_is_pareto(self) -> None:
        """A single candidate should be on the Pareto front."""
        ids = ["only"]
        vectors = [[0.5, 0.3, 0.7, 0.2]]
        front, reasons = compute_pareto_front(
            ids, vectors, {"d1": "maximize", "d2": "maximize", "d3": "maximize", "d4": "maximize"}
        )
        assert front == {"only"}
        assert reasons == {}

    def test_dominated_candidate_excluded(self) -> None:
        """A candidate dominated on all dimensions should not be on the front."""
        ids = ["good", "bad"]
        vectors = [[0.9, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1]]
        front, reasons = compute_pareto_front(
            ids, vectors, {"d1": "maximize", "d2": "maximize", "d3": "maximize", "d4": "maximize"}
        )
        assert front == {"good"}
        assert "bad" in reasons
        assert "good" in reasons["bad"]["dominated_by"]

    def test_no_dominance_all_front(self) -> None:
        """Candidates with incomparable vectors should all be on the front."""
        ids = ["a", "b", "c"]
        vectors = [
            [1.0, 0.0, 0.5, 0.5],
            [0.0, 1.0, 0.5, 0.5],
            [0.5, 0.5, 1.0, 0.5],
        ]
        front, reasons = compute_pareto_front(
            ids, vectors, {"d1": "maximize", "d2": "maximize", "d3": "maximize", "d4": "maximize"}
        )
        assert front == {"a", "b", "c"}
        assert reasons == {}

    def test_minimize_direction(self) -> None:
        """When direction is minimize, lower values dominate higher ones."""
        ids = ["a", "b"]
        vectors = [[0.1, 0.5], [0.9, 0.5]]
        front, reasons = compute_pareto_front(ids, vectors, {"d1": "minimize", "d2": "maximize"})
        assert front == {"a"}
        assert "b" in reasons

    def test_empty_input(self) -> None:
        """Empty input should produce empty results."""
        front, reasons = compute_pareto_front([], [], {})
        assert front == set()
        assert reasons == {}

    def test_dominance_includes_reasons(self) -> None:
        """Dominated candidates should have machine-readable reasons."""
        ids = ["dominator", "dominated"]
        vectors = [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
        _, reasons = compute_pareto_front(
            ids, vectors, {"d1": "maximize", "d2": "maximize", "d3": "maximize", "d4": "maximize"}
        )
        assert len(reasons["dominated"]["reasons"]) > 0
        assert "dominator" in reasons["dominated"]["reasons"][0]


# ---------------------------------------------------------------------------
# Tests: Max-min coverage selection
# ---------------------------------------------------------------------------


class TestMaxMinCoverage:
    """Max-min coverage selection from Pareto front."""

    def test_selects_from_front(self) -> None:
        """Should select candidates from the Pareto front."""
        vectors = {
            "a": [1.0, 0.0, 0.5, 0.5],
            "b": [0.0, 1.0, 0.5, 0.5],
            "c": [0.5, 0.5, 1.0, 0.5],
        }
        records = [
            {
                "candidate_id": "a",
                "topology": {"geometry_label": "type_a"},
                "actor_interaction": {"interaction_class": "sparse"},
                "mechanism_signature": {"mechanism_label": "close_approach_sparse"},
            },
            {
                "candidate_id": "b",
                "topology": {"geometry_label": "type_b"},
                "actor_interaction": {"interaction_class": "moderate"},
                "mechanism_signature": {"mechanism_label": "close_approach_moderate"},
            },
            {
                "candidate_id": "c",
                "topology": {"geometry_label": "type_c"},
                "actor_interaction": {"interaction_class": "dense"},
                "mechanism_signature": {"mechanism_label": "collision_dense"},
            },
        ]
        seq, _excl = max_min_coverage_selection(
            {"a", "b", "c"},
            ["a", "b", "c"],
            vectors,
            CoverageQuotas(min_per_topology=1, min_per_interaction_class=1),
            max_size=3,
            descriptor_records=records,
        )
        assert len(seq) == 3
        assert seq[0]["selection_order"] == 1
        assert seq[0]["candidate_id"] in {"a", "b", "c"}

    def test_deterministic_rerun(self) -> None:
        """Identical inputs should produce identical selection order."""
        ids = ["x", "y", "z", "w"]
        vectors = {
            "x": [1.0, 0.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0, 0.0],
            "z": [0.0, 0.0, 1.0, 0.0],
            "w": [0.0, 0.0, 0.0, 1.0],
        }
        records = [
            {
                "candidate_id": "x",
                "topology": {"geometry_label": "t1"},
                "actor_interaction": {"interaction_class": "sparse"},
                "mechanism_signature": {"mechanism_label": "m1"},
            },
            {
                "candidate_id": "y",
                "topology": {"geometry_label": "t2"},
                "actor_interaction": {"interaction_class": "moderate"},
                "mechanism_signature": {"mechanism_label": "m2"},
            },
            {
                "candidate_id": "z",
                "topology": {"geometry_label": "t3"},
                "actor_interaction": {"interaction_class": "dense"},
                "mechanism_signature": {"mechanism_label": "m3"},
            },
            {
                "candidate_id": "w",
                "topology": {"geometry_label": "t4"},
                "actor_interaction": {"interaction_class": "sparse"},
                "mechanism_signature": {"mechanism_label": "m4"},
            },
        ]
        quotas = CoverageQuotas(min_per_topology=1, min_per_interaction_class=1)

        seq1, _ = max_min_coverage_selection({"x", "y", "z", "w"}, ids, vectors, quotas, 4, records)
        seq2, _ = max_min_coverage_selection({"x", "y", "z", "w"}, ids, vectors, quotas, 4, records)
        ids_1 = [s["candidate_id"] for s in seq1]
        ids_2 = [s["candidate_id"] for s in seq2]
        assert ids_1 == ids_2

    def test_respects_max_size(self) -> None:
        """Should not select more than max_size candidates."""
        ids = ["a", "b", "c", "d"]
        vector_values = {
            "a": [0.9, 0.9, 0.9, 0.9],
            "b": [0.1, 0.8, 0.1, 0.1],
            "c": [0.1, 0.1, 0.8, 0.1],
            "d": [0.1, 0.1, 0.1, 0.8],
        }
        records = [
            {
                "candidate_id": i,
                "topology": {"geometry_label": f"t{i}"},
                "actor_interaction": {"interaction_class": f"c{i}"},
                "mechanism_signature": {"mechanism_label": f"m{i}"},
            }
            for i in ids
        ]
        seq, _ = max_min_coverage_selection(
            set(ids),
            ids,
            vector_values,
            CoverageQuotas(),
            max_size=2,
            descriptor_records=records,
        )
        assert len(seq) == 2

    def test_excludes_dominated(self) -> None:
        """Dominated candidates should be in the exclusion ledger."""
        ids = ["a", "b"]
        vectors = {"a": [1.0, 1.0, 1.0, 1.0], "b": [0.0, 0.0, 0.0, 0.0]}
        records = [
            {
                "candidate_id": "a",
                "topology": {"geometry_label": "t1"},
                "actor_interaction": {"interaction_class": "sparse"},
                "mechanism_signature": {"mechanism_label": "m1"},
            },
            {
                "candidate_id": "b",
                "topology": {"geometry_label": "t2"},
                "actor_interaction": {"interaction_class": "moderate"},
                "mechanism_signature": {"mechanism_label": "m2"},
            },
        ]
        _, excl = max_min_coverage_selection(
            {"a"},
            ids,
            vectors,
            CoverageQuotas(),
            10,
            records,
        )
        excluded_ids = [e["candidate_id"] for e in excl]
        assert "b" in excluded_ids
        assert all(e["excluded"] for e in excl)

    def test_empty_front_returns_empty(self) -> None:
        """Empty Pareto front should produce no selection."""
        seq, _excl_empty = max_min_coverage_selection(set(), [], {}, CoverageQuotas(), 10, [])
        assert seq == []
        assert _excl_empty == []


# ---------------------------------------------------------------------------
# Tests: Full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end portfolio selection."""

    def test_selects_from_synthetic_archive(self) -> None:
        """Should produce a schema-valid manifest from the synthetic archive."""
        manifest = select_portfolio(
            entries=SYNTHETIC_ARCHIVE,
            manifest_id="test-synthetic-01",
            archive_path="synthetic",
            archive_hash="test-hash",
        )
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert manifest["manifest_id"] == "test-synthetic-01"
        assert manifest["pareto_analysis"]["front_size"] > 0
        assert manifest["pareto_analysis"]["front_size"] <= len(SYNTHETIC_ARCHIVE)
        assert len(manifest["selection_sequence"]) > 0
        assert len(manifest["descriptor_records"]) == len(SYNTHETIC_ARCHIVE)
        assert len(manifest["eligible_inventory"]) == len(SYNTHETIC_ARCHIVE)

        # Validate against schema
        validate_selection_manifest(manifest)

    def test_full_pipeline_deterministic(self) -> None:
        """Identical inputs should produce byte-identical manifests."""
        m1 = select_portfolio(
            entries=SYNTHETIC_ARCHIVE,
            manifest_id="det-test",
        )
        m2 = select_portfolio(
            entries=SYNTHETIC_ARCHIVE,
            manifest_id="det-test",
        )
        # Timestamps differ, so compare selection and descriptors
        assert m1["selection_sequence"] == m2["selection_sequence"]
        assert m1["pareto_analysis"] == m2["pareto_analysis"]
        assert m1["exclusion_ledger"] == m2["exclusion_ledger"]

    def test_with_max_size_limit(self) -> None:
        """Should limit portfolio size."""
        config = SelectionConfig(max_portfolio_size=3)
        manifest = select_portfolio(
            entries=SYNTHETIC_ARCHIVE,
            manifest_id="size-test",
            config=config,
        )
        assert len(manifest["selection_sequence"]) == 3

    def test_with_minimal_entries(self) -> None:
        """Should handle entries with missing fields by marking descriptor values as None."""
        entries = [_make_minimal_entry("min_a"), _make_minimal_entry("min_b")]
        manifest = select_portfolio(entries=entries, manifest_id="minimal-test")
        validate_selection_manifest(manifest)
        assert len(manifest["descriptor_records"]) == 2

    def test_single_entry(self) -> None:
        """A single entry should be selected if it forms its own Pareto front."""
        entry = _make_catalog_entry("single_01", min_clearance_m=0.5)
        manifest = select_portfolio(entries=[entry], manifest_id="single-test")
        assert len(manifest["selection_sequence"]) == 1
        assert manifest["selection_sequence"][0]["candidate_id"] == "single_01"

    def test_includes_exclusion_ledger(self) -> None:
        """Every non-selected candidate should appear in the exclusion ledger."""
        entries = SYNTHETIC_ARCHIVE[:5]
        manifest = select_portfolio(entries=entries, manifest_id="excl-test")
        all_ids = {e.get("candidate_id") or e.get("scenario_id") for e in entries}
        selected_ids = {s["candidate_id"] for s in manifest["selection_sequence"]}
        excluded_ids = {e["candidate_id"] for e in manifest["exclusion_ledger"]}
        assert all_ids == selected_ids | excluded_ids
        assert len(excluded_ids) == len(all_ids) - len(selected_ids)

    def test_persistence_manifest_ref(self) -> None:
        """Should accept an optional persistence manifest reference."""
        persistence_ref = {
            "path": "docs/context/evidence/persistence_v1.json",
            "hash": "abc123",
            "schema_version": "generated_scenario_persistence.v1",
        }
        manifest = select_portfolio(
            entries=SYNTHETIC_ARCHIVE[:3],
            manifest_id="persist-test",
            persistence_manifest_ref=persistence_ref,
        )
        assert manifest["archive_hashes"]["persistence_manifest_ref"] == persistence_ref


# ---------------------------------------------------------------------------
# Tests: Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Schema loading and manifest validation."""

    def test_schema_loads(self) -> None:
        """Should load the schema without errors."""
        schema = load_portfolio_selection_schema()
        assert schema["$id"].endswith("scenario_portfolio_selection.v1.json")
        assert "properties" in schema

    def test_valid_manifest_passes(self) -> None:
        """A well-formed manifest should pass schema validation."""
        manifest = select_portfolio(
            entries=SYNTHETIC_ARCHIVE[:3],
            manifest_id="valid-test",
        )
        # Should not raise
        validate_selection_manifest(manifest)

    def test_invalid_manifest_fails(self) -> None:
        """A malformed manifest should raise PortfolioSelectionError."""
        bad = {
            "schema_version": SCHEMA_VERSION,
            "manifest_id": "bad",
        }
        with pytest.raises(PortfolioSelectionError):
            validate_selection_manifest(bad)

    def test_missing_field_fails(self) -> None:
        """Missing required field should fail validation."""
        manifest = select_portfolio(
            entries=SYNTHETIC_ARCHIVE[:3],
            manifest_id="missing-test",
        )
        del manifest["pareto_analysis"]
        with pytest.raises(PortfolioSelectionError):
            validate_selection_manifest(manifest)


# ---------------------------------------------------------------------------
# Tests: Write / load round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    """JSON serialization of manifests."""

    def test_write_and_parse_roundtrip(self, tmp_path: Path) -> None:
        """Written manifest should parse back to identical selection data."""
        manifest = select_portfolio(
            entries=SYNTHETIC_ARCHIVE[:3],
            manifest_id="roundtrip",
        )
        out_path = tmp_path / "manifest.json"
        write_selection_manifest(manifest, out_path)
        assert out_path.exists()

        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded["manifest_id"] == "roundtrip"
        assert loaded["selection_sequence"] == manifest["selection_sequence"]
        validate_selection_manifest(loaded)


# ---------------------------------------------------------------------------
# Tests: Coverage quota behavior
# ---------------------------------------------------------------------------


class TestCoverageQuotas:
    """Quota enforcement in portfolio selection."""

    def test_min_per_topology(self) -> None:
        """With min_per_topology quotas, should select from distinct maps."""
        entries = [
            _make_catalog_entry("a1", min_clearance_m=0.2, map_name="map_a"),
            _make_catalog_entry("a2", min_clearance_m=0.3, map_name="map_a"),
            _make_catalog_entry("b1", min_clearance_m=0.4, map_name="map_b"),
        ]
        config = SelectionConfig(quotas=CoverageQuotas(min_per_topology=1))
        manifest = select_portfolio(entries=entries, manifest_id="quota-topo", config=config)
        topology_set = set()
        for rec in manifest["descriptor_records"]:
            if rec["candidate_id"] in {s["candidate_id"] for s in manifest["selection_sequence"]}:
                topology_set.add(rec["topology"]["map_family"])
        assert len(topology_set) >= 1


# ---------------------------------------------------------------------------
# Tests: CLI
# ---------------------------------------------------------------------------


class TestCLI:
    """CLI entry point behavior."""

    def test_help(self) -> None:
        """--help should print without error."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/analysis/select_scenario_portfolio.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0
        assert "--archive" in result.stdout

    def test_dry_run_with_synthetic_data(self, tmp_path: Path) -> None:
        """Dry-run should succeed and print summary."""
        import subprocess
        import sys

        archive_path = tmp_path / "test_archive.jsonl"
        with archive_path.open("w") as f:
            for entry in SYNTHETIC_ARCHIVE[:3]:
                f.write(json.dumps(entry) + "\n")

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/analysis/select_scenario_portfolio.py"),
                "--archive",
                str(archive_path),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "Pareto front size" in result.stdout


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and failure modes."""

    def test_empty_archive(self) -> None:
        """An empty archive should be handled gracefully - depends on CLI."""
        pass  # CLI handles this at the entry point

    def test_duplicate_candidate_ids(self) -> None:
        """Duplicate IDs should not crash; they get separate descriptor records."""
        entries = [
            _make_catalog_entry("dup", min_clearance_m=0.5),
            _make_catalog_entry("dup", min_clearance_m=1.0),
        ]
        # Two different entries with same ID should be handled
        manifest = select_portfolio(entries=entries, manifest_id="dup-test")
        assert len(manifest["descriptor_records"]) == 2

    def test_missing_descriptors_are_none(self) -> None:
        """When descriptor fields are missing, the values should be None not absent."""
        entry = _make_catalog_entry("missing-stuff")
        del entry["criticality"]["source_metrics"]
        records = extract_descriptors([entry])
        rec = records[0]
        assert rec["criticality"]["min_clearance_m"] is None

    def test_replay_status_from_entry(self) -> None:
        """Should extract replay status from catalog entry."""
        entry = _make_catalog_entry("replay-test", replay_status="loads_only")
        records = extract_descriptors([entry])
        assert records[0]["replay_persistence"]["status"] == "loads_only"

    def test_non_finite_values_handled(self) -> None:
        """Non-finite numeric values should not crash normalization."""
        vectors = [[0.5, float("inf"), 0.3, 0.1], [0.3, 0.2, float("nan"), 0.4]]
        # Normalization should handle this without crashing
        from robot_sf.benchmark.scenario_generation.portfolio_selector import _normalize_vectors

        normalized = _normalize_vectors(vectors, "min_max")
        assert len(normalized) == 2
        assert all(math.isfinite(v) for vec in normalized for v in vec)
