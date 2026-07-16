"""Package verified issue #5756 release/rerun rows into deterministic public traces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robot_sf.benchmark.trace_reexport_packaging import (
    TraceReexportPackagingError,
    package_trace_reexport,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-bundle", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--canary-output", type=Path, required=True)
    parser.add_argument("--ppo-output", type=Path, required=True)
    parser.add_argument("--goal-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the fail-closed packager and return a shell-friendly status."""
    args = _parser().parse_args(argv)
    try:
        output = package_trace_reexport(
            release_bundle=args.release_bundle,
            request_manifest=args.request_manifest,
            canary_output=args.canary_output,
            ppo_output=args.ppo_output,
            goal_output=args.goal_output,
            output_dir=args.output_dir,
        )
    except (OSError, TraceReexportPackagingError, ValueError) as exc:
        print(f"trace re-export packaging failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote complete issue #5756 trace package to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
