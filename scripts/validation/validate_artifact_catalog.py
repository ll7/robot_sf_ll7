#!/usr/bin/env python3
"""Validate ``artifact_catalog.v1`` files."""

from __future__ import annotations

from robot_sf.benchmark.artifact_catalog import main

if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
