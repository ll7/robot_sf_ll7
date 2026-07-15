#!/usr/bin/env python3
"""One-scenario / one-seed / two-suite cross-benchmark canary runner (issue #5783).

Materializes the runnable Robot SF -> SocNavBench export, runs both suites for one pinned
policy over one mapped scenario and one seed, and emits a machine-checkable joint receipt.

Scope and claim boundary
------------------------
This is a **diagnostic sim-to-sim canary**. It is not a benchmark campaign, a training run,
or a simulator-equivalence / policy-superiority claim. The joint receipt proves that the same
pinned policy identity and the same cross-suite metric can be produced on both sides, with
suite-specific denominators preserved and recorded.

Fail-closed contract
--------------------
The canary exits nonzero when any of these gates fire:
  * the licensed SocNavBench ETH asset is not staged (unless ``--allow-synthetic-traversible``
    is passed explicitly for the no-licensed-data test path -- this is a recorded fallback;
    the real canary never sets it);
  * placeholder metadata (``tbd`` / ``blocked_prerequisite`` / ``to_be_selected`` / ``None``)
    appears in the pinned policy or scenario mapping;
  * the policy identity differs between the two suite receipts (fallback detected);
  * the suite-specific denominators differ between suites.

Example:
    uv run python scripts/tools/cross_benchmark_canary.py
    uv run python scripts/tools/cross_benchmark_canary.py --out-dir /tmp/canary --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.socnavbench_canary import (
    CANARY_NAME,
    CANARY_VERSION,
    CanaryError,
    run_canary,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "output" / "cross_benchmark_canary"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the cross-suite canary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=(
            "Directory to write the SocNavBench export and the joint canary receipt JSON "
            f"(default: {DEFAULT_OUT_DIR.as_posix()})."
        ),
    )
    parser.add_argument(
        "--allow-synthetic-traversible",
        action="store_true",
        default=False,
        help=(
            "TEST-ONLY escape for environments without the licensed SocNavBench ETH asset. "
            "Never use for real canary runs; enabling it is a fallback and is recorded in the receipt."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-checkable joint receipt JSON to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the canary; return 0 on a valid joint receipt, 1 on any fail-closed gate.

    A nonzero exit signals a blocked gate (missing asset, placeholder metadata, policy
    mismatch, fallback, or denominator drift). It is never evidence of a benchmark result.
    """
    args = _parse_args(argv)
    try:
        receipt = run_canary(
            out_dir=args.out_dir,
            allow_synthetic_traversible=args.allow_synthetic_traversible,
        )
    except CanaryError as exc:
        # Diagnostics go to stderr so stdout stays reserved for the receipt/summary.
        print(f"canary blocked: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        suites = receipt["suites"]
        robot_sf = next(s for s in suites if s["suite"] == "Robot SF")
        socnavbench = next(s for s in suites if s["suite"] == "SocNavBench")
        policy = receipt["policy_identity"]
        print(f"Cross-suite canary {CANARY_NAME} v{CANARY_VERSION} (issue #5783)")
        print(f"  policy: {policy['policy_id']}@{policy['version']} ({policy['algo']})")
        print(
            f"  scenario: {receipt['scenario_mapping']['robot_sf_scenario_id']} <-> "
            f"{receipt['scenario_mapping']['socnavbench_scenario_id']} seed={receipt['seed']}"
        )
        print(f"  metric: {robot_sf['metric_id']}")
        print(
            f"  Robot SF value:   {robot_sf['value']:.6f} "
            f"(denominator={robot_sf['denominator']:.6f} {robot_sf['denominator_kind']})"
        )
        print(
            f"  SocNavBench value:{socnavbench['value']:.6f} "
            f"(denominator={socnavbench['denominator']:.6f} {socnavbench['denominator_kind']})"
        )
        print(f"  policy identity match: {receipt['policy_identity_match']}")
        print(f"  denominators preserved: {receipt['denominators_preserved']}")
        print(f"  fallback forbidden: {receipt['fallback_forbidden']}")
        print(
            f"  external asset staged: {receipt['external_asset_staged']} "
            f"({receipt['external_asset_id']})"
        )
        print(f"  receipt: {receipt.get('receipt_path')}")
        print(f"  claim boundary: {receipt['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
