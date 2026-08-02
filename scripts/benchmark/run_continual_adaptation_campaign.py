#!/usr/bin/env python3
"""Wire continual-adaptation campaign evidence for the promotion gate (issue #6657).

Builds nominal/shift/forgetting result references and a versioned evidence
bundle naming the validator-derived adapted-policy identifier, then validates
the promotion gate via ``check_continual_adaptation_run``.

This script is metadata-only: it does not launch training, run evaluations,
write checkpoints, mutate the safety wrapper, or promote a policy.  Fallback
or degraded execution fails closed.

Example::

    uv run python scripts/benchmark/run_continual_adaptation_campaign.py \\
        --manifest configs/benchmark/continual_adaptation_promotion_fixture.yaml \\
        --validate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.continual_adaptation_campaign import (
    build_continual_adaptation_evidence,
    prepare_promotion_manifest,
    validate_promotion_readiness,
    write_evidence_bundle,
    write_promotion_manifest,
)
from robot_sf.research.continual_adaptation_protocol import (
    check_continual_adaptation_run,
    load_continual_adaptation_run,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a continual_adaptation_run.v1 YAML manifest.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the manifest against the promotion gate and exit.",
    )
    parser.add_argument(
        "--evidence-out",
        type=Path,
        help="Directory to write the evidence bundle YAML.",
    )
    parser.add_argument(
        "--promotion-manifest-out",
        type=Path,
        help="Path to write the promotion-ready manifest YAML.",
    )
    parser.add_argument(
        "--execution-mode",
        default="native",
        help="Execution mode label (fallback/degraded fails closed).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the continual-adaptation campaign evidence builder."""
    args = parse_args()

    manifest = load_continual_adaptation_run(args.manifest)

    if args.validate:
        report = check_continual_adaptation_run(manifest, source=args.manifest)
        print(json.dumps(report.to_dict(), indent=2))
        if report.protocol_status != "valid":
            print(f"FAIL: protocol_status={report.protocol_status}", file=sys.stderr)
            return 1
        if report.promotion_decision == "promote" and not report.promotion_ready:
            print("FAIL: promotion gate not satisfied", file=sys.stderr)
            return 1
        print("OK: protocol check passed", file=sys.stderr)
        return 0

    evidence = build_continual_adaptation_evidence(
        manifest,
        nominal_uri=f"docs/context/evidence/issue_{manifest['issue']}_continual_adaptation_campaign/nominal_result.json",
        shift_uri=f"docs/context/evidence/issue_{manifest['issue']}_continual_adaptation_campaign/shift_result.json",
        forgetting_uri=f"docs/context/evidence/issue_{manifest['issue']}_continual_adaptation_campaign/forgetting_result.json",
        evidence_bundle_uri=f"docs/context/evidence/issue_{manifest['issue']}_continual_adaptation_campaign/evidence_bundle.yaml",
        evidence_bundle_identifier=f"continual_adaptation_evidence_{manifest['issue']}_v1",
        execution_mode=args.execution_mode,
        source=args.manifest,
    )

    if args.evidence_out:
        path = write_evidence_bundle(evidence, args.evidence_out, overwrite=args.overwrite)
        print(f"Evidence bundle written: {path}", file=sys.stderr)

    if args.promotion_manifest_out:
        promoted = prepare_promotion_manifest(manifest, evidence)
        path = write_promotion_manifest(
            promoted, args.promotion_manifest_out, overwrite=args.overwrite
        )
        print(f"Promotion manifest written: {path}", file=sys.stderr)

        validation = validate_promotion_readiness(promoted, source=args.promotion_manifest_out)
        if not validation.is_promotion_ready:
            print(
                f"FAIL: promotion gate not satisfied: {validation.blockers}",
                file=sys.stderr,
            )
            return 1
        print("OK: promotion gate satisfied", file=sys.stderr)

    print(json.dumps(evidence.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
