#!/usr/bin/env python3
"""Reproduce the powered issue #5303 six-seed promotion contract hash and frozen manifest.

This is the side-effect-free, check-only invocation of the frozen powered #5303 contract
(schema ``issue_5303_search_promotion_contract.v2``). It does not execute planners, run a
search campaign, replay or confirm anything, submit Slurm jobs, or read evaluation
outcomes. It only loads the frozen contract, the #6139 recertification receipt, the
preregistration manifest, and statically parsed handoff sources; it recomputes SHA-256
hashes; asserts the frozen design, exact cluster-level inference, and outcome-free
sensitivity analysis; proves the timing dimensions are runtime-effective while the
historical inert/no-pedestrian mode stays rejected; and rejects the historical v1 contract
for promotion-capable execution.

With ``--identities`` it emits exactly 768 scheduled search identities (method x search
seed x candidate index) as a deterministic JSON manifest and performs no planner execution
and no outcome read. It exits non-zero unless exactly 768 unique identities are emitted.

Examples:
    # Full powered-contract check (exit 0 only when the frozen contract verifies).
    uv run python scripts/tools/check_issue_5303_search_promotion_contract_v2.py

    # Emit the deterministic 768-identity scheduled-search manifest.
    uv run python scripts/tools/check_issue_5303_search_promotion_contract_v2.py --identities
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.issue_5303_search_promotion_preregistration_v2 import (
    DEFAULT_CONTRACT_PATH,
    DEFAULT_MANIFEST_PATH,
    EXPECTED_TOTAL_SCHEDULED_ATTEMPTS,
    dump_preflight_payload,
    identity_manifest_bytes,
    preflight_issue_5303_powered_contract,
    scheduled_search_identities,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the powered contract-check CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  uv run python scripts/tools/check_issue_5303_search_promotion_contract_v2.py"
        ),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="Frozen powered issue #5303 search-promotion contract YAML.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Override the contract-declared issue #6139 corrected recertification receipt JSON.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Frozen-contract hash manifest for the powered preregistration.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root for resolving relative paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the check report.",
    )
    parser.add_argument(
        "--identities",
        action="store_true",
        help=(
            "Emit exactly 768 scheduled search identities as a deterministic JSON manifest "
            "instead of running the full check. Performs no planner execution and no outcome "
            "read."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the powered contract check (or identity emission) and return the exit code."""
    args = parse_args(argv)

    if args.identities:
        identities = scheduled_search_identities()
        if len(identities) != EXPECTED_TOTAL_SCHEDULED_ATTEMPTS:
            print(
                f"FAILED: expected exactly {EXPECTED_TOTAL_SCHEDULED_ATTEMPTS} scheduled "
                f"search identities, derived {len(identities)}",
                file=sys.stderr,
            )
            return 1
        unique_identities = {identity["identity_sha256"] for identity in identities}
        if len(unique_identities) != EXPECTED_TOTAL_SCHEDULED_ATTEMPTS:
            print(
                f"FAILED: scheduled search identities are not unique "
                f"({len(unique_identities)} unique of {len(identities)})",
                file=sys.stderr,
            )
            return 1
        sys.stdout.buffer.write(identity_manifest_bytes())
        return 0

    try:
        result = preflight_issue_5303_powered_contract(
            args.contract,
            receipt_path=args.receipt,
            manifest_path=args.manifest,
            repo_root=args.repo_root,
        )
    except OSError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2
    dump_preflight_payload(result, args.output)
    print(json.dumps(result.to_payload(), sort_keys=True))
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
