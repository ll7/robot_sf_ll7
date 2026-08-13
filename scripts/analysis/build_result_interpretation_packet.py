#!/usr/bin/env python3
"""Build and validate a result interpretation packet (issue #7029).

Loads a packet JSON file, validates it against the
``result_interpretation_packet.v1`` contract, and optionally writes the
deterministic output.  The packet is a contract-only slice: it does not
re-run experiments or infer values from filenames or plots.

Usage::

    uv run python scripts/analysis/build_result_interpretation_packet.py \\
        --input packet.json --output validated.json

    uv run python scripts/analysis/build_result_interpretation_packet.py \\
        --input packet.json --validate-only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the repository root is on the path so that ``robot_sf`` is importable
# even when invoked outside a virtualenv activation.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot_sf.benchmark.result_interpretation_packet import (  # noqa: E402
    compute_packet_digest,
    compute_post_review_digest,
    load_result_interpretation_packet,
    write_caption,
    write_checksum_manifest,
    write_deterministic_json,
    write_review_report,
)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="build_result_interpretation_packet",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input packet JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the validated output JSON.  Defaults to <input>.validated.json.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Validate the input and exit without writing output.",
    )
    parser.add_argument(
        "--show-digest",
        action="store_true",
        default=False,
        help="Print the packet digest and exit.",
    )
    parser.add_argument(
        "--caption-output",
        type=Path,
        default=None,
        help="Optional path for a deterministic caption text file.",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=None,
        help="Optional path for a deterministic review report JSON.",
    )
    parser.add_argument(
        "--checksum-output",
        type=Path,
        default=None,
        help="Optional path for a checksum manifest covering generated outputs.",
    )
    args = parser.parse_args(argv)

    try:
        packet = load_result_interpretation_packet(args.input)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    digest = compute_packet_digest(packet)
    if args.show_digest:
        print(f"packet_digest: {digest}")
        return 0

    if args.validate_only:
        print(f"packet {packet.packet_id!r} is valid (digest: {digest[:16]}...)")
        return 0

    output = args.output or args.input.with_suffix(".validated.json")
    write_deterministic_json(packet.to_dict(), output)
    generated = {"packet.json": output}
    if args.caption_output is not None:
        write_caption(packet, args.caption_output)
        generated["caption.txt"] = args.caption_output
    if args.review_output is not None:
        write_review_report(packet, args.review_output)
        generated["review.json"] = args.review_output
    if args.checksum_output is not None:
        write_checksum_manifest(generated, args.checksum_output, packet=packet)
    post_review = compute_post_review_digest(packet)
    print(f"written {output}")
    print(f"packet_digest: {digest}")
    print(f"post_review_digest: {post_review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
