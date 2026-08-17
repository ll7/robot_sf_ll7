"""Verify a Chapter 7 v2 build-provenance receipt fail-closed."""

from __future__ import annotations

import argparse
from pathlib import Path

from jsonschema import ValidationError

from scripts.analysis.ch7_evidence_build_receipt import (
    DEFAULT_RECEIPT,
    Ch7EvidenceBuildReceiptError,
    verify_receipt,
)


def main(argv: list[str] | None = None) -> int:
    """Verify a receipt and return a CLI status code."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="verify recorded/current hashes without running two fresh package builds",
    )
    args = parser.parse_args(argv)
    try:
        result = verify_receipt(
            repository=args.repository,
            receipt_path=args.receipt,
            rebuild=not args.no_rebuild,
        )
    except (Ch7EvidenceBuildReceiptError, OSError, ValidationError) as exc:
        print(f"ch7 build receipt unavailable: {exc}")
        return 2
    print(
        f"ch7 build receipt verification: {result['status']} ({result['receipt_payload_sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
