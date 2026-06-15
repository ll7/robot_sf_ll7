"""Tests for the manifest lineage graph report builder."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.manifest_lineage_graph import (
    LineageStatus,
    build_manifest_lineage_graph,
    format_lineage_graph_markdown,
    write_manifest_lineage_graph_report,
)

FIXTURE_DIR = Path(__file__).resolve().parents[0] / "fixtures" / "manifest_lineage_graph"


def _load_fixture(name: str) -> dict:
    """Load a JSON fixture by name."""
    path = FIXTURE_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_connected_manifest_trace_is_connected() -> None:
    """A complete manifest yields a connected trace."""
    manifest_path = FIXTURE_DIR / "connected_manifest.json"
    candidates = [
        {
            "artifact_id": "dissertation_table_1",
            "artifact_kind": "table",
            "source_manifest_path": str(manifest_path),
            "claim_boundary": "dissertation-chapter-4",
        }
    ]

    graph = build_manifest_lineage_graph(
        [manifest_path],
        artifact_candidates=candidates,
    )

    assert graph.schema_version == "manifest_lineage_graph.v1"
    trace = next(t for t in graph.traces if t.artifact_id == "dissertation_table_1")
    assert trace.lineage_status == LineageStatus.CONNECTED
    assert "all mandatory lineage fields" in trace.reason
    assert all(s == LineageStatus.CONNECTED for s in trace.field_statuses.values())


def test_missing_manifest_trace_is_missing() -> None:
    """A manifest missing all lineage fields yields a missing trace."""
    manifest_path = FIXTURE_DIR / "missing_manifest.json"
    candidates = [
        {
            "artifact_id": "release_table_1",
            "artifact_kind": "table",
            "source_manifest_path": str(manifest_path),
            "claim_boundary": "release-v1.0",
        }
    ]

    graph = build_manifest_lineage_graph(
        [manifest_path],
        artifact_candidates=candidates,
    )

    trace = next(t for t in graph.traces if t.artifact_id == "release_table_1")
    assert trace.lineage_status == LineageStatus.MISSING
    assert any(s == LineageStatus.MISSING for s in trace.field_statuses.values())


def test_ambiguous_manifest_trace_is_ambiguous() -> None:
    """A manifest with conflicting proxy values yields an ambiguous trace."""
    manifest_path = FIXTURE_DIR / "ambiguous_manifest.json"
    candidates = [
        {
            "artifact_id": "release_figure_1",
            "artifact_kind": "figure",
            "source_manifest_path": str(manifest_path),
            "claim_boundary": "release-v1.0",
        }
    ]

    graph = build_manifest_lineage_graph(
        [manifest_path],
        artifact_candidates=candidates,
    )

    trace = next(t for t in graph.traces if t.artifact_id == "release_figure_1")
    assert trace.lineage_status == LineageStatus.AMBIGUOUS
    assert any(s == LineageStatus.AMBIGUOUS for s in trace.field_statuses.values())


def test_orphan_candidate_is_inconclusive() -> None:
    """A candidate pointing at a manifest not in the input set is inconclusive."""
    manifest_path = FIXTURE_DIR / "connected_manifest.json"
    candidates = [
        {
            "artifact_id": "orphan_artifact",
            "artifact_kind": "table",
            "source_manifest_path": "nonexistent_manifest.json",
            "claim_boundary": "unknown",
        }
    ]

    graph = build_manifest_lineage_graph(
        [manifest_path],
        artifact_candidates=candidates,
    )

    trace = next(t for t in graph.traces if t.artifact_id == "orphan_artifact")
    assert trace.lineage_status == LineageStatus.INCONCLUSIVE


def test_graph_nodes_include_contract_manifest_and_fields() -> None:
    """The graph contains the validation contract, manifest, and field nodes."""
    manifest_path = FIXTURE_DIR / "connected_manifest.json"

    graph = build_manifest_lineage_graph([manifest_path])

    node_kinds = {node.kind for node in graph.nodes}
    assert "validation_contract" in node_kinds
    assert "manifest" in node_kinds
    assert "lineage_field" in node_kinds

    assert any(edge.kind == "validates_with" for edge in graph.edges)
    assert any(edge.kind == "has_field" for edge in graph.edges)


def test_markdown_report_contains_summary_and_traces() -> None:
    """Markdown report includes the summary and artifact trace table."""
    manifest_path = FIXTURE_DIR / "connected_manifest.json"
    candidates = [
        {
            "artifact_id": "dissertation_table_1",
            "artifact_kind": "table",
            "source_manifest_path": str(manifest_path),
            "claim_boundary": "dissertation-chapter-4",
        }
    ]

    graph = build_manifest_lineage_graph(
        [manifest_path],
        artifact_candidates=candidates,
    )
    markdown = format_lineage_graph_markdown(graph)

    assert "# Manifest Lineage Graph Report" in markdown
    assert "## Summary" in markdown
    assert "## Artifact traces" in markdown
    assert "dissertation_table_1" in markdown
    assert "connected" in markdown


def test_write_report_creates_json_and_markdown(tmp_path: Path) -> None:
    """write_manifest_lineage_graph_report emits both output files."""
    manifest_path = FIXTURE_DIR / "connected_manifest.json"
    graph = build_manifest_lineage_graph([manifest_path])

    json_path = tmp_path / "lineage.json"
    md_path = tmp_path / "lineage.md"
    written = write_manifest_lineage_graph_report(
        graph,
        json_path=json_path,
        markdown_path=md_path,
    )

    assert written["json"] == json_path
    assert written["markdown"] == md_path
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "manifest_lineage_graph.v1"
    assert "nodes" in payload
    assert "edges" in payload


def test_cli_runs_and_writes_outputs(tmp_path: Path) -> None:
    """The CLI script emits JSON and Markdown reports."""
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "benchmark"
        / "build_manifest_lineage_graph.py"
    )
    json_path = tmp_path / "out.json"
    md_path = tmp_path / "out.md"

    import subprocess

    result = subprocess.run(
        [
            "python",
            str(script),
            "--manifest",
            str(FIXTURE_DIR / "connected_manifest.json"),
            "--manifest",
            str(FIXTURE_DIR / "ambiguous_manifest.json"),
            "--artifact-candidates",
            str(FIXTURE_DIR / "candidates.json"),
            "--out-json",
            str(json_path),
            "--out-md",
            str(md_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads(result.stdout)
    assert summary["manifest_count"] == 2
    assert summary["artifact_candidate_count"] == 4
    assert json_path.exists()
    assert md_path.exists()


def test_repo_relative_paths_in_graph(tmp_path: Path) -> None:
    """Manifest paths in the graph are repository-relative."""
    manifest_path = FIXTURE_DIR / "connected_manifest.json"
    graph = build_manifest_lineage_graph([manifest_path])

    manifest_node = next(node for node in graph.nodes if node.kind == "manifest")
    assert not Path(manifest_node.path).is_absolute()
    assert manifest_node.path.startswith("tests/benchmark/fixtures/manifest_lineage_graph/")


def test_missing_field_produces_missing_edges() -> None:
    """Missing lineage fields produce missing field nodes and no proxy edges."""
    manifest_path = FIXTURE_DIR / "missing_manifest.json"
    graph = build_manifest_lineage_graph([manifest_path])

    field_nodes = [node for node in graph.nodes if node.kind == "lineage_field"]
    assert len(field_nodes) == 8
    missing_field = next(
        node
        for node in field_nodes
        if node.label == "source" and node.status == LineageStatus.MISSING
    )
    assert missing_field.status == LineageStatus.MISSING


def test_ambiguous_field_produces_ambiguous_edges() -> None:
    """Ambiguous lineage fields produce ambiguous_between edges."""
    manifest_path = FIXTURE_DIR / "ambiguous_manifest.json"
    graph = build_manifest_lineage_graph([manifest_path])

    ambiguous_edges = [edge for edge in graph.edges if edge.kind == "ambiguous_between"]
    assert len(ambiguous_edges) >= 1
    labels = {edge.source.split("__")[-1] for edge in ambiguous_edges}
    assert "validator_version" in labels
