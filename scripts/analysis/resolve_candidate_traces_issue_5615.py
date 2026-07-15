#!/usr/bin/env python3
"""Deterministic candidate-to-trace resolver CLI (issue #5615).

Reads a ``seed_flip_inversion_candidates.v1`` candidate manifest (the issue
#5446 miner output), joins every candidate to its pinned campaign episode, its
``simulation_trace_export.v1`` artifact, its per-trace
``trace_failure_predicates.v1`` rows, and its ``critical-intervals.v1`` records,
and emits a schema-versioned ``candidate_trace_resolution.v1`` manifest.

It is a composition layer only: it consumes the pinned campaign result store
read-only and never launches campaigns, mutates artifacts, or adds analysis.

Examples
--------
    # Resolve candidates against a pinned campaign store, writing the manifest.
    uv run python scripts/analysis/resolve_candidate_traces_issue_5615.py \
        --candidates candidates.json \
        --campaign-store output/campaign \
        --trace-roots tests/fixtures/analysis_workbench/simulation_trace_export_v1 \
        --json out/resolution.json --validate

    # Re-run idempotence check (byte-identical manifest on same input).
    uv run python scripts/analysis/resolve_candidate_traces_issue_5615.py \
        --candidates candidates.json --check-determinism
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.candidate_trace_resolution import (
    SCHEMA_VERSION,
    CandidateTraceResolutionError,
    load_episode_mapping,
    load_episode_requests,
    resolve_candidate_trace_resolution,
    resolve_episode_requests,
    validate_candidate_trace_resolution,
)


def _load_json(path: Path) -> Any:
    """Load a JSON payload from ``path``.

    Returns:
        The parsed JSON object.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _load_critical_interval_config(path: Path | None) -> dict[str, Any] | None:
    """Load an optional critical-intervals config (JSON or YAML).

    Returns:
        The parsed config dict, or ``None`` when no path is supplied.
    """
    if path is None:
        return None
    import yaml

    text = path.read_text(encoding="utf-8")
    if text.strip().startswith("{"):
        return json.loads(text)
    return yaml.safe_load(text) or {}


def build_parser() -> argparse.ArgumentParser:
    """Build the resolver CLI argument parser.

    Returns:
        The configured parser.
    """
    p = argparse.ArgumentParser(
        prog="resolve_candidate_traces_issue_5615",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--candidates",
        type=Path,
        help="Validated seed_flip_inversion_candidates.v1 manifest JSON (issue #5446).",
    )
    input_group.add_argument(
        "--episode-requests",
        type=Path,
        help="Concrete issue_5446_trace_reexport_list.v1 request manifest (issue #5756).",
    )
    p.add_argument(
        "--episode-mapping",
        "--episode-map",
        dest="episode_mapping",
        type=Path,
        default=None,
        help="Read-only rerun mapping with episode identity, outcome, and trace URI.",
    )
    p.add_argument(
        "--campaign-store",
        type=Path,
        default=None,
        help="Pinned campaign-result-store.v1 directory (read-only).",
    )
    p.add_argument(
        "--trace-roots",
        type=Path,
        nargs="*",
        default=[],
        help="Roots to search for trace export artifacts when artifact_uri is absent.",
    )
    p.add_argument(
        "--critical-interval-config",
        type=Path,
        default=None,
        help="Optional critical-intervals.v1 YAML/JSON config for interval discovery.",
    )
    p.add_argument(
        "--json", type=Path, default=None, help="Write the resolution manifest JSON here."
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Validate the manifest against candidate_trace_resolution.v1 JSON Schema.",
    )
    p.add_argument(
        "--check-determinism",
        action="store_true",
        help="Run the resolver twice and assert byte-identical output (idempotence).",
    )
    return p


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0912
    """Run the resolver CLI.

    Returns:
        Process exit code: 0 ok, 2 fail-closed, 3 validation/determinism failure.
    """
    args = build_parser().parse_args(argv)
    if args.episode_requests is not None:
        if args.episode_mapping is None:
            print("error: --episode-mapping is required with --episode-requests", file=sys.stderr)
            return 2
        try:
            request_manifest, _ = load_episode_requests(args.episode_requests)
            episode_mapping = load_episode_mapping(args.episode_mapping)
        except CandidateTraceResolutionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        manifest = resolve_episode_requests(
            request_manifest,
            episode_mapping,
            trace_search_roots=[Path(r) for r in args.trace_roots],
        )
        candidate_manifest = None
    else:
        try:
            candidate_manifest = _load_json(args.candidates)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        critical_interval_config = _load_critical_interval_config(args.critical_interval_config)

        kwargs: dict[str, Any] = {
            "trace_search_roots": [Path(r) for r in args.trace_roots],
            "critical_interval_config": critical_interval_config,
        }
        if args.campaign_store is not None:
            kwargs["campaign_store_dir"] = args.campaign_store

        try:
            manifest = resolve_candidate_trace_resolution(candidate_manifest, **kwargs)
        except CandidateTraceResolutionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if args.check_determinism:
        if args.episode_requests is not None:
            manifest2 = resolve_episode_requests(
                request_manifest,
                episode_mapping,
                trace_search_roots=[Path(r) for r in args.trace_roots],
            )
        else:
            manifest2 = resolve_candidate_trace_resolution(candidate_manifest, **kwargs)
        if json.dumps(manifest, sort_keys=True) != json.dumps(manifest2, sort_keys=True):
            print("error: determinism check failed; re-run was not byte-identical", file=sys.stderr)
            return 3

    payload = json.dumps(manifest, indent=2, sort_keys=False)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

    s = manifest["summary"]
    print(
        "resolved="
        f"{s['n_resolved']} trace-missing={s['n_trace_missing']} "
        f"schema-mismatch={s['n_schema_mismatch']} "
        f"provenance-incomplete={s['n_provenance_incomplete']} "
        f"of {s['n_candidates']} candidates",
        file=sys.stderr,
    )

    if args.validate:
        result = validate_candidate_trace_resolution(manifest)
        if not result["ok"]:
            print("error: manifest failed schema validation:", file=sys.stderr)
            for error in result["errors"]:
                print(f"  {error}", file=sys.stderr)
            return 3
        print(f"validated against {SCHEMA_VERSION}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
