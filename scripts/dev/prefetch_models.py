#!/usr/bin/env python3
"""Pre-fetch registered model artifacts into the runtime cache before benchmarks run.

Why this exists (issue #6189): the exact-repeat benchmark test downloads a
predictive-proxy model from a GitHub release at runtime, inside the timed repeat
loop. A transient network blip that fails one repeat but succeeds another makes
the three trajectory fingerprints differ and turns the ``main`` CI checkmark
flaky. This script moves that fetch into a dedicated, bounded-retry *setup*
phase so the timed loop only ever loads a present, checksum-verified cache file.

It is the CI-visible entry point for the *same* reusable model preflight that
scientific campaigns already use through
:func:`robot_sf.models.registry.resolve_model_path` /
:func:`robot_sf.models.registry.prefetch_model`: a single registry path provides
bounded retries, atomic cache replacement, and a registry-pinned SHA-256
verification. CI and campaigns therefore resolve models identically.

Usage::

    # Pre-fetch the model the exact-repeat test depends on:
    uv run python scripts/dev/prefetch_models.py \\
        --model-id predictive_proxy_selected_v2_full

    # Pin a non-default registry / cache location:
    uv run python scripts/dev/prefetch_models.py --model-id <id> \\
        --registry model/registry.yaml --cache-dir output/model_cache

Exit code is non-zero when any model cannot be staged (loud setup failure); a
present, checksum-matching cache entry is reused without any network access, so
re-runs are idempotent and free.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.models.registry import prefetch_model

MANIFEST_SCHEMA_VERSION = "model_prefetch_manifest.v1"
DEFAULT_CACHE_DIR = Path("output/model_cache")
DEFAULT_MANIFEST_PATH = DEFAULT_CACHE_DIR / ".prefetch_manifest.json"
# Precise failures prefetch_model / resolve_model_path can raise. Catching these
# explicitly (rather than ``Exception``) keeps the repo's broad-exception
# ratchet unchanged while still surfacing every setup failure loudly.
_PREFETCH_ERRORS: tuple[type[BaseException], ...] = (
    KeyError,
    RuntimeError,
    ValueError,
    FileNotFoundError,
    OSError,
)


def _now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the model preflight."""
    parser = argparse.ArgumentParser(
        description=(
            "Pre-fetch registered model artifacts into the runtime cache before "
            "benchmarks run (issue #6189)."
        ),
    )
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        default=[],
        required=True,
        help=(
            "Registry model id to pre-fetch (repeatable). The exact-repeat "
            "benchmark test depends on 'predictive_proxy_selected_v2_full'."
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to model/registry.yaml (default: the bundled registry).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory for staged artifacts (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=(f"Path to write a JSON summary manifest (default: {DEFAULT_MANIFEST_PATH})."),
    )
    parser.add_argument(
        "--format",
        choices=("friendly", "json"),
        default="friendly",
        help="Output format (default: friendly).",
    )
    return parser


def prefetch_models(
    model_ids: list[str],
    *,
    registry: str | Path | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> list[dict[str, Any]]:
    """Pre-fetch every requested model id and return per-model result rows.

    Each row records the model id, resolved path, observed SHA-256, whether the
    cache entry was reused without a download, and the outcome status. A row
    whose ``ok`` is ``False`` carries a human-readable ``error`` and the
    exception type so CI surfaces a clear, actionable setup failure.

    Args:
        model_ids: Registry model ids to pre-fetch.
        registry: Optional model-registry path override.
        cache_dir: Cache directory to stage artifacts into.

    Returns:
        list[dict[str, Any]]: One result row per requested model id, in order.
    """
    results: list[dict[str, Any]] = []
    cache_root = Path(cache_dir)
    for model_id in model_ids:
        # Reuse detection: a present cache entry before prefetch means the
        # bounded-retry download path will short-circuit on the pinned checksum.
        reused = _existing_cache_path(model_id, cache_root) is not None
        row: dict[str, Any] = {
            "model_id": model_id,
            "ok": False,
            "cached_reused": reused,
            "status": "failed",
        }
        try:
            path, observed_sha256 = prefetch_model(
                model_id,
                registry_path=registry,
                cache_dir=cache_root,
            )
        except _PREFETCH_ERRORS as exc:
            row["error"] = str(exc)
            row["error_type"] = type(exc).__name__
            row["status"] = "prefetch_failed"
            results.append(row)
            continue
        row.update(
            {
                "ok": True,
                "status": "present" if reused else "downloaded",
                "path": str(path),
                "sha256": observed_sha256,
            }
        )
        results.append(row)
    return results


def _existing_cache_path(model_id: str, cache_root: Path) -> Path | None:
    """Return the present cache asset for ``model_id`` when one exists, else None."""
    if cache_root is None:
        return None
    model_cache_dir = cache_root / model_id
    if not model_cache_dir.is_dir():
        return None
    candidates = [path for path in model_cache_dir.iterdir() if path.is_file()]
    return candidates[0] if candidates else None


def _write_manifest(
    manifest_path: Path,
    results: list[dict[str, Any]],
) -> None:
    """Write a JSON summary manifest describing the prefetch outcomes."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": _now_iso(),
        "results": results,
        "all_ok": all(row.get("ok") for row in results),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _render_friendly(results: list[dict[str, Any]]) -> str:
    """Return a one-line-per-model human-readable summary."""
    lines = ["Model preflight results:"]
    for row in results:
        status = row["status"]
        if row.get("ok"):
            reused = " (cached, reused)" if row.get("cached_reused") else " (downloaded)"
            lines.append(
                f"  - {row['model_id']}: ok{reused} sha256={row['sha256']} -> {row['path']}"
            )
        else:
            lines.append(
                f"  - {row['model_id']}: {status} ({row.get('error_type', 'error')}): "
                f"{row.get('error', '')}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: pre-fetch models, write manifest, return a shell exit code."""
    args = _build_parser().parse_args(argv)
    results = prefetch_models(
        args.model_ids,
        registry=args.registry,
        cache_dir=args.cache_dir,
    )
    _write_manifest(args.manifest, results)
    all_ok = all(row.get("ok") for row in results)
    if args.format == "json":
        print(
            json.dumps(
                {"schema": MANIFEST_SCHEMA_VERSION, "results": results, "all_ok": all_ok},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(_render_friendly(results))
        print(f"Wrote prefetch manifest to {args.manifest}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
