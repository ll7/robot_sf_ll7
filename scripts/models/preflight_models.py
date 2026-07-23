"""Stage required model assets into the local cache before a timed run (issue #6189).

This is the shared, campaign- and CI-callable entrypoint for the model preflight.
It resolves every registry model id it is given -- or every model id a config
would resolve at runtime -- to a checksum-verified local cache path *before* any
worker loop starts, so the timed execution never performs a network download.

Usage::

    # Explicit model ids (used by the CI setup step before the exact-repeat test):
    uv run python scripts/models/preflight_models.py predictive_proxy_selected_v2_full

    # Or derive the required model ids from a campaign/planner config:
    uv run python scripts/models/preflight_models.py --config configs/baselines/ppo_issue_791_eval_aligned_large_capacity_cpu.yaml

Exit codes are distinct so a CI step / sbatch wrapper can branch mechanically:

- ``0`` -- every required asset is present and checksum-verified in the cache.
- ``2`` -- a config file was requested but is missing or unreadable.
- ``3`` -- an asset could not be staged after bounded retries (fail-closed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from robot_sf.models.preflight import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    ModelPreflightError,
    preflight_models,
    required_model_ids_for_config,
)

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_BLOCKED = 3


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Stage required model assets into the cache before a timed run (issue #6189).",
    )
    parser.add_argument(
        "model_ids",
        nargs="*",
        help="Registry model ids to stage. Combined with any --config-derived ids.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="PATH",
        help="Config YAML whose runtime-resolved model ids should also be staged. Repeatable.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override the model cache directory (defaults to output/model_cache).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Bounded retry attempts per asset (default: {DEFAULT_MAX_ATTEMPTS}).",
    )
    parser.add_argument(
        "--backoff-seconds",
        type=float,
        default=DEFAULT_BACKOFF_SECONDS,
        help=f"Base retry backoff in seconds (default: {DEFAULT_BACKOFF_SECONDS}).",
    )
    return parser


def _config_model_ids(config_paths: list[str]) -> list[str]:
    """Return the union of runtime-resolved model ids across the given configs.

    Raises:
        FileNotFoundError: If a requested config path does not exist.
    """
    collected: list[str] = []
    for raw in config_paths:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"config not found: {path}")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        collected.extend(required_model_ids_for_config(config))
    return collected


def main(argv: list[str] | None = None) -> int:
    """Run the model preflight CLI.

    Returns:
        int: Process exit code (see module docstring).
    """
    args = build_arg_parser().parse_args(argv)

    try:
        model_ids = list(args.model_ids) + _config_model_ids(args.config)
    except (FileNotFoundError, yaml.YAMLError) as exc:
        print(f"[preflight] config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    if not model_ids:
        print(
            "[preflight] no model ids given; pass model ids and/or --config PATH",
            file=sys.stderr,
        )
        return EXIT_CONFIG_ERROR

    try:
        resolved = preflight_models(
            model_ids,
            cache_dir=args.cache_dir,
            max_attempts=args.max_attempts,
            backoff_seconds=args.backoff_seconds,
        )
    except ModelPreflightError as exc:
        print(f"[preflight] BLOCKED: {exc}", file=sys.stderr)
        return EXIT_BLOCKED

    for model_id, path in resolved.items():
        print(f"[preflight] OK {model_id} -> {path}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
