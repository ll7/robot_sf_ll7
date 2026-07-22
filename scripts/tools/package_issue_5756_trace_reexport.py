"""Package verified issue #5756 release/rerun rows into deterministic public traces.

The default ``package`` command runs the fail-closed packager against the frozen release
bundle and the three pinned rerun outputs, materializing a complete 90-trace package.

The ``to-resolver-mapping`` command adapts an already-complete package directory into the
``issue_5756_trace_mapping_receipt.v1`` mapping receipt consumed by the #5615 candidate
resolver and the #5756 worked-example renderer, so the two independently-merged contracts
join end to end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robot_sf.benchmark.trace_reexport_packaging import (
    TraceReexportPackagingError,
    build_resolver_mapping_receipt,
    package_trace_reexport,
)


def _add_package_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = sub.add_parser(
        "package",
        help="Run the fail-closed packager (default when no subcommand is given).",
    )
    parser.add_argument("--release-bundle", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--canary-output", type=Path, required=True)
    parser.add_argument("--ppo-output", type=Path, required=True)
    parser.add_argument("--goal-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--emit-resolver-receipt",
        action="store_true",
        help="Also write a resolver-ready mapping receipt into the package directory.",
    )


def _add_resolver_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = sub.add_parser(
        "to-resolver-mapping",
        help="Adapt a complete package into the resolver's mapping receipt.",
    )
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path (default: <package-dir>/resolver_mapping_receipt.json).",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    _add_package_parser(sub)
    _add_resolver_parser(sub)
    return parser


def _run_package(args: argparse.Namespace) -> int:
    try:
        output = package_trace_reexport(
            release_bundle=args.release_bundle,
            request_manifest=args.request_manifest,
            canary_output=args.canary_output,
            ppo_output=args.ppo_output,
            goal_output=args.goal_output,
            output_dir=args.output_dir,
        )
        if args.emit_resolver_receipt:
            receipt_path = output / "resolver_mapping_receipt.json"
            build_resolver_mapping_receipt(output, output_path=receipt_path)
            print(f"wrote resolver mapping receipt to {receipt_path}")
    except (OSError, TraceReexportPackagingError, ValueError) as exc:
        print(f"trace re-export packaging failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote complete issue #5756 trace package to {output}")
    return 0


def _run_resolver_mapping(args: argparse.Namespace) -> int:
    output_path = args.output or (args.package_dir / "resolver_mapping_receipt.json")
    try:
        build_resolver_mapping_receipt(args.package_dir, output_path=output_path)
    except (OSError, TraceReexportPackagingError, ValueError) as exc:
        print(f"resolver mapping conversion failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote resolver mapping receipt to {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the packager or adapter and return a shell-friendly status.

    For backward compatibility, when no subcommand is supplied the original flat
    ``package`` arguments are accepted directly.
    """
    raw = list(argv if argv is not None else sys.argv[1:])
    if raw and raw[0] == "to-resolver-mapping":
        args = _parser().parse_args(raw)
        return _run_resolver_mapping(args)
    args = _parser().parse_args(["package", *raw])
    return _run_package(args)


if __name__ == "__main__":
    raise SystemExit(main())
