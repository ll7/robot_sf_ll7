#!/usr/bin/env python3
"""Fail-closed research-lane continual-adaptation diagnostic launcher (#6659).

Loads a ``continual_adaptation_run.v1`` manifest, refuses to proceed unless the
protocol contract reports ``protocol_status == 'valid'``, and writes the bounded
adaptation plus the nominal/shift/forgetting evaluation surfaces as DIAGNOSTIC
outputs under ``output/``. It emits no promotion decision, generates no evidence
bundle, and makes no benchmark or paper claim.

Example:
    uv run python scripts/research/run_continual_adaptation_diagnostics.py \
        --manifest configs/training/continual_adaptation_run_issue_6582.yaml
"""

from __future__ import annotations

import argparse
import json
import sys

from robot_sf.research.continual_adaptation_launcher import (
    render_markdown,
    run_continual_adaptation_diagnostics,
)
from robot_sf.research.continual_adaptation_protocol import (
    ContinualAdaptationProtocolError,
    load_continual_adaptation_run,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to a continual_adaptation_run.v1 manifest (JSON or YAML).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Diagnostic output directory (defaults to "
            "output/continual_adaptation_diagnostics/<run_id>)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-readable diagnostic report JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the diagnostic launcher and return a shell-friendly exit code.

    Returns:
        ``0`` when the manifest is protocol-valid and diagnostics were written;
        ``1`` when the launcher fails closed on a schema violation or an invalid
        protocol status.
    """
    args = build_arg_parser().parse_args(argv)
    try:
        manifest = load_continual_adaptation_run(args.manifest)
        report = run_continual_adaptation_diagnostics(
            manifest, source=args.manifest, output_dir=args.output_dir
        )
    except ContinualAdaptationProtocolError as exc:
        # Fail closed: an invalid or schema-violating manifest writes no output.
        print(f"continual-adaptation launcher failed closed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
