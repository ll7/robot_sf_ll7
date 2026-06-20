#!/usr/bin/env python3
"""CLI entry point for adversarial candidate-batch certification."""

from __future__ import annotations

import sys

from robot_sf.adversarial.batch_certification import main as certification_main


def main(argv: list[str] | None = None) -> int:
    """Run as a script with arguments from ``sys.argv``."""
    return certification_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
