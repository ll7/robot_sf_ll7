#!/usr/bin/env python3
"""CLI entry point for adversarial manifest quality summaries."""

from __future__ import annotations

import sys

from robot_sf.adversarial.manifest_quality import main as summarize_main


def main(argv: list[str] | None = None) -> int:
    """Run as a script with arguments from ``sys.argv``."""
    return summarize_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
