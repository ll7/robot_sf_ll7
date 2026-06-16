#!/usr/bin/env python3
"""Build a compact manifest lineage graph report for local fixture inputs.

Emits a JSON graph and a Markdown adjacency report. The graph traces artifact
candidates (e.g. dissertation/release tables) back to source manifests and the
shared validation contract, marking missing or ambiguous lineage as blocked or
inconclusive.

Example::

    python scripts/benchmark/build_manifest_lineage_graph.py \
        --manifest tests/benchmark/fixtures/manifest_lineage_graph/connected_manifest.json \
        --artifact-candidates tests/benchmark/fixtures/manifest_lineage_graph/candidates.json \
        --out-json output/manifest_lineage_graph.json \
        --out-md output/manifest_lineage_graph.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_sf.benchmark.manifest_lineage_graph import (  # noqa: E402
    _parse_generated_at_utc,
    build_manifest_lineage_graph,
    write_manifest_lineage_graph_report,
)


def _cli_generated_at_utc(value: str) -> str:
    """CLI type wrapper that turns ValueError into argparse-friendly output."""
    try:
        return _parse_generated_at_utc(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        required=True,
        help="Manifest JSON files to include in the lineage graph.",
    )
    parser.add_argument(
        "--artifact-candidates",
        type=Path,
        default=None,
        help="Optional JSON file with artifact candidate list or {'candidates': [...]} object.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="Output path for the JSON graph report.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Optional output path for the Markdown adjacency report.",
    )
    parser.add_argument(
        "--generated-at-utc",
        type=_cli_generated_at_utc,
        default=None,
        metavar="TIMESTAMP",
        help=(
            "Optional ISO-8601 UTC timestamp for the report "
            "(e.g. 2026-06-15T00:00:00+00:00). Defaults to current UTC time."
        ),
    )
    return parser.parse_args(argv)


def _resolve_manifest_paths(paths: list[Path]) -> list[Path]:
    """Resolve and validate manifest paths from CLI arguments."""
    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"manifest not found: {path}")
        if not path.is_file():
            raise ValueError(f"manifest path is not a file: {path}")
        resolved_path = path.resolve()
        if resolved_path in seen:
            continue
        resolved.append(resolved_path)
        seen.add(resolved_path)
    return resolved


def main(argv: list[str] | None = None) -> int:
    """Run the manifest lineage graph report builder."""
    args = _parse_args(argv)
    manifest_paths = _resolve_manifest_paths(args.manifest)

    # Load candidates after resolving manifest paths so the source path is
    # available for relative manifest resolution.
    candidates: list[dict[str, Any]] = []
    candidate_source_path: Path | None = None
    if args.artifact_candidates is not None:
        if not args.artifact_candidates.exists():
            raise FileNotFoundError(
                f"artifact candidates file not found: {args.artifact_candidates}"
            )
        candidate_payload = json.loads(args.artifact_candidates.read_text(encoding="utf-8"))
        if isinstance(candidate_payload, dict):
            candidates = list(candidate_payload.get("candidates", []))
        elif isinstance(candidate_payload, list):
            candidates = list(candidate_payload)
        else:
            raise ValueError("artifact candidates file must contain a JSON list or object")
        candidate_source_path = args.artifact_candidates

    graph = build_manifest_lineage_graph(
        manifest_paths,
        artifact_candidates=candidates,
        candidate_source_path=candidate_source_path,
        generated_at_utc=args.generated_at_utc,
    )

    written = write_manifest_lineage_graph_report(
        graph,
        json_path=args.out_json,
        markdown_path=args.out_md,
    )

    summary = {
        "json_path": str(written["json"]),
        "markdown_path": str(written.get("markdown", "")),
        "manifest_count": graph.summary.get("manifest_count", 0),
        "artifact_candidate_count": graph.summary.get("artifact_candidate_count", 0),
        "node_count": graph.summary.get("node_count", 0),
        "edge_count": graph.summary.get("edge_count", 0),
        "trace_count": graph.summary.get("trace_count", 0),
    }
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
