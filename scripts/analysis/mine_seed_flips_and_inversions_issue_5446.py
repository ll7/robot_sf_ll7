#!/usr/bin/env python3
"""Deterministic miner CLI for reproducible seed flips and held-out planner upsets.

Issue #5446. Reads benchmark result rows (an ``episodes.jsonl`` or a JSON list),
applies the evidence gates and candidate mining in
:mod:`robot_sf.benchmark.seed_flip_mining`, and emits a schema-versioned
candidate manifest (JSON) plus an optional compact Markdown summary.

This is analysis tooling only: it selects *candidates worth confirming* and
records the uncertainty and every exclusion behind each. It is not a benchmark
metric and makes no planner-ranking claim. Confirmation runs are a separate
exact-compute packet.

Examples
--------
    # Mine from a campaign episodes.jsonl, write the manifest to stdout.
    uv run python scripts/analysis/mine_seed_flips_and_inversions_issue_5446.py \
        --input output/campaign/episodes.jsonl

    # Write a manifest + Markdown summary, consuming sibling-issue signals.
    uv run python scripts/analysis/mine_seed_flips_and_inversions_issue_5446.py \
        --input rows.jsonl --external signals.json \
        --json out/candidates.json --md out/candidates.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.seed_flip_mining import (
    DEFAULT_CONF,
    DEFAULT_MIN_HELDOUT_CELLS,
    DEFAULT_MIN_SEEDS,
    DEFAULT_TRIAGE_MAX_SEEDS,
    SeedFlipMiningError,
    mine_seed_flip_inversion_candidates,
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """Load rows from a ``.jsonl`` (one object per line) or a JSON list/dict."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict) and "rows" in data:
        return list(data["rows"])
    if isinstance(data, dict) and "episodes" in data:
        return list(data["episodes"])
    if isinstance(data, list):
        return data
    raise SystemExit(
        f"unrecognized rows payload in {path}: expected JSONL, list, or {{'rows': [...]}}"
    )


def _render_markdown(manifest: dict[str, Any], top: int) -> str:
    """Compact, human-readable summary of the candidate manifest."""
    s = manifest["summary"]
    lines = [
        "# Seed-flip & held-out planner-inversion candidates (issue #5446)",
        "",
        f"- schema: `{manifest['schema_version']}`",
        f"- rows: {s['n_rows']} | eligible cells: {s['n_eligible_cells']} | "
        f"excluded rows: {s['n_excluded_rows']}",
        f"- candidates: {s['n_candidates']} "
        f"(seed-flip {s['n_seed_flip_candidates']}, upset {s['n_planner_upset_candidates']}) "
        f"| Pareto-selected: {s['n_selected']}",
        "",
        f"> {manifest['claim_boundary']}",
        "",
        "## Archetype availability",
    ]
    for name, info in manifest["archetype_availability"].items():
        mark = "available" if info.get("available") else "unavailable"
        lines.append(f"- `{name}`: {mark}")
    lines.append("")
    lines.append("## Selected candidates (Pareto frontier)")
    selected = [c for c in manifest["candidates"] if c.get("selected")]
    if not selected:
        lines.append("_none selected_")
    for cand in selected[:top]:
        bits = []
        flip = cand.get("seed_flip")
        if flip:
            bits.append(
                f"flip entropy={flip['entropy_bits']:.3f} bits, "
                f"n_seeds={flip['effective_denominator']}, "
                f"CI={flip['interval'][0]:.2f}-{flip['interval'][1]:.2f}"
            )
        if cand.get("heldout_planner_skill_gap") is not None:
            bits.append(f"heldout skill gap={cand['heldout_planner_skill_gap']:.3f}")
        if cand.get("cross_planner_disagreement_entropy") is not None:
            bits.append(f"disagreement={cand['cross_planner_disagreement_entropy']:.3f} bits")
        triage = " _(triage-only)_" if cand.get("triage_only") else ""
        lines.append(
            f"- **{cand['archetype']}** `{cand['scenario_id']}` / `{cand['planner']}`{triage}: "
            + "; ".join(bits)
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the miner CLI."""
    p = argparse.ArgumentParser(
        description=(
            "Mine reproducible seed flips and held-out planner upsets from benchmark result "
            "rows (issue #5446). Analysis tooling only; not a benchmark metric or ranking claim."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path, help="Rows file (.jsonl or .json list).")
    p.add_argument(
        "--external",
        type=Path,
        default=None,
        help="Optional JSON of sibling-issue signal tables "
        "(oracle_regret #5302, transfer #5303, quality_diversity #5308, multiplicity #5351).",
    )
    p.add_argument("--outcome-metric", default="success", help="Binary outcome metric name.")
    p.add_argument("--group-by", default=None, help="Planner-identifying dotted path override.")
    p.add_argument("--seed-field", default="seed", help="Dotted path to the seed field.")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Wilson interval confidence.")
    p.add_argument("--min-seeds", type=int, default=DEFAULT_MIN_SEEDS, help="Min seeds to flip.")
    p.add_argument(
        "--triage-max-seeds",
        type=int,
        default=DEFAULT_TRIAGE_MAX_SEEDS,
        help="Cells at/under this seed count are flagged triage-only.",
    )
    p.add_argument(
        "--min-heldout-cells",
        type=int,
        default=DEFAULT_MIN_HELDOUT_CELLS,
        help="Min leave-one-out cells for a trusted planner strength.",
    )
    p.add_argument(
        "--allow-non-native",
        action="store_true",
        help="Do NOT fail closed on non-native execution rows (diagnostic only).",
    )
    p.add_argument("--json", type=Path, default=None, help="Write the manifest JSON here.")
    p.add_argument("--md", type=Path, default=None, help="Write a Markdown summary here.")
    p.add_argument("--top", type=int, default=20, help="Max selected candidates in the summary.")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the miner CLI; returns a process exit code (0 ok, 2 fail-closed)."""
    args = build_parser().parse_args(argv)
    rows = _load_rows(args.input)
    external = None
    if args.external is not None:
        external = json.loads(args.external.read_text(encoding="utf-8"))

    kwargs: dict[str, Any] = {
        "outcome_metric": args.outcome_metric,
        "seed_field": args.seed_field,
        "conf": args.conf,
        "min_seeds": args.min_seeds,
        "triage_max_seeds": args.triage_max_seeds,
        "min_heldout_cells": args.min_heldout_cells,
        "require_native": not args.allow_non_native,
        "external": external,
    }
    if args.group_by is not None:
        kwargs["group_by"] = args.group_by

    try:
        manifest = mine_seed_flip_inversion_candidates(rows, **kwargs)
    except SeedFlipMiningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(manifest, indent=2, sort_keys=False)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    if args.md is not None:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(_render_markdown(manifest, args.top), encoding="utf-8")
    if args.json is None and args.md is None:
        print(payload)
    else:
        s = manifest["summary"]
        print(
            f"mined {s['n_candidates']} candidates "
            f"({s['n_selected']} selected) from {s['n_eligible_cells']} eligible cells; "
            f"excluded {s['n_excluded_rows']} rows"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
