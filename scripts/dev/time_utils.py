#!/usr/bin/env python3
"""Shared UTC timestamp contract for repository scripts.

Canonical owner for the ``datetime.now(UTC).isoformat().replace("+00:00", "Z")``
pattern duplicated across validation, benchmark-admission, and receipt scripts
(issue #9250, family 3). The single format keeps machine-readable timestamps
comparable across receipts, badges, and admission packets: ISO-8601 with a
``Z`` suffix, no microseconds ambiguity beyond what ``datetime.isoformat``
emits.

Importable with ``from scripts.dev.time_utils import utc_now_iso`` when the
script runs from the repository root (the established ``uv run`` practice for
dev/validation tooling). Standalone plain-Python scripts that cannot import
``scripts.*`` (see issue #9248) keep their local copy and are out of scope.
"""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 ``Z``-suffixed string."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
