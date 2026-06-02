#!/usr/bin/env python3
"""Validate figure artifacts with deterministic QA checks."""

from __future__ import annotations

from robot_sf.benchmark.figure_qa import main

if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
