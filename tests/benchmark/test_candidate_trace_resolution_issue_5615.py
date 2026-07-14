"""Synthetic-fixture tests for the candidate-to-trace resolution layer (issue #5615).

The fixtures encode *known* answers along the three required fail-closed paths:
``resolved`` (candidate -> campaign row -> valid trace -> predicates/intervals),
``trace-missing`` (campaign row present but artifact cannot be located), and
``schema-mismatch`` (artifact located but fails ``simulation_trace_export.v1``
validation). A ``provenance-incomplete`` path (candidate with no matching
campaign row) is also exercised. Determinism and JSON-Schema validation are
checked directly.

Test ids intentionally contain ``candidate_trace_resolution`` plus a status word
so they match the issue's ``-k`` selectors
(``resolved``, ``trace_missing``, ``schema_mismatch``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import robot_sf.benchmark.candidate_trace_resolution as candidate_trace_resolution_module
from robot_sf.benchmark.candidate_trace_resolution import (
    SCHEMA_VERSION,
    CandidateTraceResolutionError,
    load_campaign_result_store,
    resolve_candidate_to_episode,
    resolve_candidate_trace_resolution,
    validate_candidate_trace_resolution,
)
from robot_sf.benchmark.seed_flip_mining import mine_seed_flip_inversion_candidates
from robot_sf.benchmark.utils import coerce_optional_id
from scripts.tools.campaign_result_store import write_result_store

_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "benchmark" / "issue_5615"
_TRACE_FIXTURES = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
)


def _row(scenario: str, planner: str, seed: int, episode: str, *, artifact_uri: str | None) -> dict:
    """Build one campaign episode row with full provenance."""
    return {
        "run_id": f"run-{scenario}-{planner}-{seed}",
        "episode_id": episode,
        "planner": planner,
        "scenario_id": scenario,
        "scenario_family": scenario,
        "seed": seed,
        "row_status": "native",
        "artifact_uri": artifact_uri or "",
        "artifact_sha256": "deadbeef" * 8,
    }


def _write_campaign_store(tmp_path: Path) -> Path:
    """Write a small pinned campaign result store with one resolvable row."""
    store_dir = tmp_path / "campaign_store"
    # Point the artifact_uri at a real, schema-valid trace fixture.
    trace_path = _TRACE_FIXTURES / "minimal_trace.json"
    write_result_store(
        store_dir,
        [
            _row(
                "classic_bottleneck_medium",
                "hybrid_rule_v0_minimal",
                111,
                "fixture_episode_001",
                artifact_uri=str(trace_path),
            )
        ],
        study_id="study-5615",
        command="uv run python scripts/tools/campaign_result_store.py out",
        source_commit="abc123",
    )
    return store_dir


def _candidate_manifest() -> dict:
    """Mine a tiny candidate manifest containing a single seed-flip candidate.

    The candidate's scenario/planner/seed/episode match the resolvable campaign
    row so it resolves against the store produced by :func:`_write_campaign_store`.
    """
    rows = [
        {
            "episode_id": f"fixture_episode_{i:03d}",
            "scenario_id": "classic_bottleneck_medium",
            "seed": 111 + i,
            "config_hash": "cfg-abc",
            "repo_commit": "abc123",
            "execution_mode": "native",
            "collision_semantics": "typed",
            "scenario_params": {"algo": "hybrid_rule_v0_minimal"},
            "metrics": {"success": float(s)},
        }
        for i, s in enumerate((1.0, 0.0))
    ]
    manifest = mine_seed_flip_inversion_candidates(rows)
    # Ensure the candidate carries the fields the resolver joins on.
    cand = next(c for c in manifest["candidates"] if c["archetype"] == "seed_flip")
    cand["scenario_id"] = "classic_bottleneck_medium"
    cand["planner"] = "hybrid_rule_v0_minimal"
    cand["seed"] = 111
    cand["episode_id"] = "fixture_episode_001"
    cand["config_hash"] = "cfg-abc"
    return manifest


def _schema_mismatch_trace(tmp_path: Path) -> Path:
    """Write a trace artifact that fails ``simulation_trace_export.v1`` validation."""
    bad = tmp_path / "bad_trace.json"
    # Missing required top-level fields -> schema mismatch.
    bad.write_text(json.dumps({"trace_id": "bad"}), encoding="utf-8")
    return bad


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (0, 0),
        (" 111 ", 111),
        (111.0, 111),
        ("", None),
        (111.5, None),
        (float("nan"), None),
        (float("inf"), None),
        (True, None),
    ],
)
def test_candidate_trace_resolution_optional_id_coercion(
    value: object, expected: int | None
) -> None:
    """Optional identifiers preserve valid falsy values and reject malformed input."""
    assert coerce_optional_id(value) == expected


def test_campaign_result_store_ignores_rows_with_null_required_identifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing store identifiers must not become literal ``None`` key components."""
    store_dir = tmp_path / "campaign_store"
    store_dir.mkdir()
    (store_dir / "summary.json").write_text('{"study_id": "study-5615"}', encoding="utf-8")
    (store_dir / "episodes.parquet").touch()

    class _Frame:
        def to_dict(self, *, orient: str) -> list[dict[str, object]]:
            assert orient == "records"
            return [
                {
                    "run_id": "run-bad",
                    "scenario_id": None,
                    "planner": "planner",
                    "episode_id": "episode",
                    "seed": 111,
                    "artifact_uri": "trace.json",
                }
            ]

    monkeypatch.setattr(candidate_trace_resolution_module, "read_parquet_frame", lambda _: _Frame())
    store = load_campaign_result_store(store_dir)

    assert store.episodes == {}
    result = resolve_candidate_to_episode(
        {
            "scenario_id": "None",
            "planner": "planner",
            "episode_id": "episode",
            "seed": 111,
        },
        store,
    )
    assert result["resolution_status"] == "provenance-incomplete"


def test_candidate_trace_resolution_resolved_joins_trace_and_signals() -> None:
    """A candidate with a real pinned trace resolves with predicate availability."""
    import tempfile

    manifest = _candidate_manifest()
    with tempfile.TemporaryDirectory() as td:
        store_dir = _write_campaign_store(Path(td))
        result = resolve_candidate_trace_resolution(
            manifest,
            campaign_store_dir=store_dir,
            trace_search_roots=[_TRACE_FIXTURES],
        )
    assert result["schema_version"] == SCHEMA_VERSION
    row = result["rows"][0]
    assert row["resolution_status"] == "resolved"
    assert row["trace_schema_version"] == "simulation_trace_export.v1"
    assert row["trace_content_hash"] is not None
    # The minimal trace has at least one clearance/collision predicate family.
    assert row["predicate_rows_available"] is not None
    assert result["summary"]["n_resolved"] == 1
    assert result["summary"]["n_trace_missing"] == 0
    assert result["summary"]["n_schema_mismatch"] == 0


def test_candidate_trace_resolution_trace_missing_has_reason_code() -> None:
    """A campaign row whose artifact cannot be located is trace-missing."""
    import tempfile

    manifest = _candidate_manifest()
    with tempfile.TemporaryDirectory() as td:
        # Artifact URI points at a non-existent file and no search root matches.
        store_dir = Path(td) / "store"
        from scripts.tools.campaign_result_store import write_result_store

        write_result_store(
            store_dir,
            [
                _row(
                    "classic_bottleneck_medium",
                    "hybrid_rule_v0_minimal",
                    111,
                    "fixture_episode_001",
                    artifact_uri="/nonexistent/trace.json",
                )
            ],
            study_id="study-5615",
            command="echo",
            source_commit="abc123",
        )
        result = resolve_candidate_trace_resolution(
            manifest,
            campaign_store_dir=store_dir,
            trace_search_roots=[],
        )
    row = result["rows"][0]
    assert row["resolution_status"] == "trace-missing"
    assert row["reason_code"] == "trace_artifact_not_found"
    assert row["trace_artifact_uri"] is None
    assert result["summary"]["n_trace_missing"] == 1


def test_candidate_trace_resolution_schema_mismatch_has_reason_code() -> None:
    """A located artifact that fails trace validation is schema-mismatch."""
    import tempfile

    manifest = _candidate_manifest()
    with tempfile.TemporaryDirectory() as td:
        bad = _schema_mismatch_trace(Path(td))
        store_dir = Path(td) / "store"
        from scripts.tools.campaign_result_store import write_result_store

        write_result_store(
            store_dir,
            [
                _row(
                    "classic_bottleneck_medium",
                    "hybrid_rule_v0_minimal",
                    111,
                    "fixture_episode_001",
                    artifact_uri=str(bad),
                )
            ],
            study_id="study-5615",
            command="echo",
            source_commit="abc123",
        )
        result = resolve_candidate_trace_resolution(
            manifest,
            campaign_store_dir=store_dir,
            trace_search_roots=[],
        )
    row = result["rows"][0]
    assert row["resolution_status"] == "schema-mismatch"
    assert row["reason_code"].startswith("unexpected_trace_schema")
    assert result["summary"]["n_schema_mismatch"] == 1


def test_candidate_trace_resolution_provenance_incomplete_without_store() -> None:
    """Without a campaign store, candidates are provenance-incomplete."""
    manifest = _candidate_manifest()
    result = resolve_candidate_trace_resolution(manifest, campaign_store_dir=None)
    row = result["rows"][0]
    assert row["resolution_status"] == "provenance-incomplete"
    assert row["reason_code"] == "no_campaign_store_provided"
    assert result["summary"]["n_provenance_incomplete"] == 1


def test_candidate_trace_resolution_every_candidate_appears_once() -> None:
    """Every candidate appears exactly once with an explicit status."""
    import tempfile

    manifest = _candidate_manifest()
    n_candidates = len(manifest["candidates"])
    with tempfile.TemporaryDirectory() as td:
        store_dir = _write_campaign_store(Path(td))
        result = resolve_candidate_trace_resolution(
            manifest,
            campaign_store_dir=store_dir,
            trace_search_roots=[_TRACE_FIXTURES],
        )
    assert len(result["rows"]) == n_candidates
    ids = [r["candidate_id"] for r in result["rows"]]
    assert len(ids) == len(set(ids))  # no duplicates
    statuses = {r["resolution_status"] for r in result["rows"]}
    assert statuses <= {
        "resolved",
        "trace-missing",
        "schema-mismatch",
        "provenance-incomplete",
    }


def test_candidate_trace_resolution_deterministic_byte_identical() -> None:
    """A re-run on the same inputs is byte-identical (no clocks in rows)."""
    import tempfile

    manifest = _candidate_manifest()
    with tempfile.TemporaryDirectory() as td:
        store_dir = _write_campaign_store(Path(td))
        r1 = resolve_candidate_trace_resolution(
            manifest, campaign_store_dir=store_dir, trace_search_roots=[_TRACE_FIXTURES]
        )
        r2 = resolve_candidate_trace_resolution(
            manifest, campaign_store_dir=store_dir, trace_search_roots=[_TRACE_FIXTURES]
        )
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_candidate_trace_resolution_validates_against_published_schema() -> None:
    """The resolved manifest validates against candidate_trace_resolution.v1."""
    import tempfile

    manifest = _candidate_manifest()
    with tempfile.TemporaryDirectory() as td:
        store_dir = _write_campaign_store(Path(td))
        result = resolve_candidate_trace_resolution(
            manifest, campaign_store_dir=store_dir, trace_search_roots=[_TRACE_FIXTURES]
        )
    outcome = validate_candidate_trace_resolution(result)
    assert outcome["ok"], outcome["errors"]


def test_candidate_trace_resolution_fails_closed_on_empty_candidates() -> None:
    """A candidate manifest with no candidates fails closed."""
    manifest = _candidate_manifest()
    manifest["candidates"] = []
    with pytest.raises(CandidateTraceResolutionError):
        resolve_candidate_trace_resolution(manifest)


def test_candidate_trace_resolution_fails_closed_on_wrong_schema() -> None:
    """A non-candidate manifest fails closed."""
    with pytest.raises(CandidateTraceResolutionError):
        resolve_candidate_trace_resolution({"schema_version": "something.else", "candidates": []})
