#!/usr/bin/env python3
"""Build the pinned Chapter 7 case-capsule manifest (issue #5447).

Reads a *validated* ``seed_flip_inversion_candidates.v1`` candidate manifest (the
issue #5446 miner output) plus optional causal / online-risk report bundles, and
emits a schema-versioned ``ch7_case_capsule_manifest.v1`` selecting diverse
worked-example archetypes for Chapter 7. It is synthesis/selection tooling only:
it never fabricates a capsule, labels unavailable archetypes ``unavailable``,
grades descriptive-only unless a validated causal report is supplied, and fails
closed on empty/wrong-schema input.

This CLI does **not** render figures. Actual vector-figure generation from pinned
episode trajectories is a downstream step (reuses ``robot_sf.benchmark.figures``)
that operates on this manifest; the manifest records the exact input hashes so
that step is reproducible.

Examples
--------
    # Build the capsule manifest from a frozen #5446 candidate manifest.
    uv run python scripts/analysis/build_ch7_case_capsules_issue_5447.py \
        --candidates docs/context/evidence/issue_5446_*/candidates.json

    # Include validated causal + risk reports and write the manifest + validation.
    uv run python scripts/analysis/build_ch7_case_capsules_issue_5447.py \
        --candidates candidates.json --causal causal.json --risk risk.json \
        --json out/ch7_capsules.json --validate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.case_capsules import (
    DEFAULT_MAX_CAPSULES,
    DEFAULT_MIN_CAPSULES,
    CaseCapsuleError,
    build_ch7_case_capsule_manifest,
    validate_ch7_case_capsule_manifest,
)


def _load_json(path: Path) -> Any:
    """Load a JSON payload from ``path``.

    Returns:
        The parsed JSON object.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        The configured parser.
    """
    p = argparse.ArgumentParser(
        prog="build_ch7_case_capsules_issue_5447",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--candidates",
        required=True,
        type=Path,
        help="Validated seed_flip_inversion_candidates.v1 manifest JSON (issue #5446).",
    )
    p.add_argument(
        "--causal",
        type=Path,
        default=None,
        help="Optional validated causal report bundle (JSON dict keyed by candidate/scenario).",
    )
    p.add_argument(
        "--risk",
        type=Path,
        default=None,
        help="Optional validated online-risk report bundle (JSON dict keyed by candidate/scenario).",
    )
    p.add_argument(
        "--min-capsules",
        type=int,
        default=DEFAULT_MIN_CAPSULES,
        help="Honest floor; below it the manifest is insufficient_evidence.",
    )
    p.add_argument(
        "--max-capsules",
        type=int,
        default=DEFAULT_MAX_CAPSULES,
        help="Ceiling on admitted capsules.",
    )
    p.add_argument(
        "--allow-triage",
        action="store_true",
        help="Permit triage-only (<=3-seed) candidates as capsule sources.",
    )
    p.add_argument("--json", type=Path, default=None, help="Write the capsule manifest JSON here.")
    p.add_argument(
        "--validate",
        action="store_true",
        help="Also run the structural validator and print its result.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the capsule-builder CLI.

    Returns:
        Process exit code: 0 ok, 2 fail-closed build error, 3 structural
        validation failure.
    """
    args = build_parser().parse_args(argv)
    candidate_manifest = _load_json(args.candidates)
    causal_reports = _load_json(args.causal) if args.causal is not None else None
    risk_reports = _load_json(args.risk) if args.risk is not None else None

    try:
        manifest = build_ch7_case_capsule_manifest(
            candidate_manifest,
            causal_reports=causal_reports,
            risk_reports=risk_reports,
            min_capsules=args.min_capsules,
            max_capsules=args.max_capsules,
            allow_triage=args.allow_triage,
        )
    except CaseCapsuleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(manifest, indent=2, sort_keys=False)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

    s = manifest["summary"]
    print(
        f"status={manifest['status']} admitted={s['n_admitted']}/{s['n_capsules_requested']} "
        f"unavailable={s['n_unavailable']} "
        f"admitted_archetypes={','.join(s['admitted_archetypes']) or '-'}",
        file=sys.stderr,
    )

    if args.validate:
        result = validate_ch7_case_capsule_manifest(manifest)
        print(json.dumps(result.as_dict(), indent=2), file=sys.stderr)
        if not result.ok:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
