#!/usr/bin/env python3
"""Invocation-local evidence paths for the wheel-install smoke (issue #8917).

The wheel-install smoke persists optional-extra install and probe logs so every
run leaves inspectable evidence. Those logs previously used fixed paths under
``output/validation``, so concurrent invocations with distinct
``ROBOT_SF_WHEEL_INSTALL_SMOKE_REPORT`` destinations overwrote each other's
logs. This helper derives a per-invocation log directory from the report's
output directory plus an explicit run identity and binds log paths and SHA-256
digests into the structured report.
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

RUN_ID_ENV = "ROBOT_SF_WHEEL_INSTALL_SMOKE_RUN_ID"
LOG_ROOT_NAME = "wheel-install-smoke-logs"


def run_id_from_env(environ: Mapping[str, str] | None = None) -> str:
    """Return the explicit run identity, or a timestamp-plus-pid fallback."""
    source = os.environ if environ is None else environ
    explicit = source.get(RUN_ID_ENV, "").strip()
    if explicit:
        return explicit
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{os.getpid()}"


def log_dir_for_report(report_path: Path | str, run_id: str) -> Path:
    """Return the invocation-local log directory for one report path."""
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise ValueError(f"invalid run id {run_id!r}")
    return Path(report_path).resolve().parent / LOG_ROOT_NAME / run_id


def extra_log_paths(log_dir: Path | str, extra: str) -> tuple[Path, Path]:
    """Return the ``(install_log, probe_log)`` paths for one optional extra."""
    base = Path(log_dir) / f"wheel-extra-{extra}"
    return (
        base.with_name(f"{base.name}-install.log"),
        base.with_name(f"{base.name}-probe.log"),
    )


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enrich_extras_rows(
    rows: Sequence[Mapping[str, object]], log_dir: Path | str
) -> list[dict[str, object]]:
    """Bind log paths and digests into extras rows, failing closed on missing logs.

    Each row must identify its ``extra``; the returned rows are copies with
    ``log_dir``, ``install_log``, ``probe_log``, ``install_log_sha256``, and
    ``probe_log_sha256`` attached. A missing or unreadable log raises so a
    report can never claim evidence it did not persist.
    """
    directory = Path(log_dir)
    enriched: list[dict[str, object]] = []
    for row in rows:
        extra = str(row.get("extra", ""))
        if not extra:
            raise ValueError("extras row is missing the 'extra' field")
        install_log, probe_log = extra_log_paths(directory, extra)
        missing = [str(path) for path in (install_log, probe_log) if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing extras evidence log(s): " + ", ".join(missing))
        enriched.append(
            {
                **row,
                "log_dir": str(directory),
                "install_log": str(install_log),
                "probe_log": str(probe_log),
                "install_log_sha256": _sha256(install_log),
                "probe_log_sha256": _sha256(probe_log),
            }
        )
    return enriched
