"""Create the durable Chapter 7 v2 build-provenance receipt."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from jsonschema import ValidationError

from scripts.analysis.ch7_evidence_build_receipt import (
    DEFAULT_RECEIPT,
    Ch7EvidenceBuildReceiptError,
    build_receipt,
)


def main(argv: list[str] | None = None) -> int:
    """Build a receipt and return a CLI status code."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument(
        "--source-commit",
        help="Git commit that produced the package; defaults to the current HEAD",
    )
    args = parser.parse_args(argv)
    try:
        receipt: dict[str, Any] = build_receipt(
            repository=args.repository,
            output=args.output,
            source_commit=args.source_commit,
        )
    except (Ch7EvidenceBuildReceiptError, OSError, ValidationError) as exc:
        print(f"ch7 build receipt unavailable: {exc}")
        return 2
    print(
        "ch7 build receipt status: "
        f"{receipt['status']} ({receipt['integrity']['receipt_payload_sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
