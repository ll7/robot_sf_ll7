#!/usr/bin/env python3
"""Build the Chapter 7 worked-example portfolio manifest (issue #6789)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.case_portfolio import (
    CasePortfolioError,
    build_ch7_worked_example_portfolio,
    file_sha256,
    finalize_manifest,
    read_json_or_gzip,
    validate_ch7_worked_example_portfolio,
    write_deterministic_json,
)

DEFAULT_CONFIG = Path("configs/analysis/ch7_worked_example_portfolio.v1.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CasePortfolioError(f"config must parse to a dict: {path}")
    return payload


def _resolve_candidate_manifest_path(config: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override
    pinned = config.get("pinned_inputs", {})
    if not isinstance(pinned, dict) or not pinned.get("candidate_manifest_rel"):
        raise CasePortfolioError(
            "candidate manifest is required: pass --candidate-manifest or configure "
            "pinned_inputs.candidate_manifest_rel"
        )
    return Path(str(pinned["candidate_manifest_rel"]))


def _load_candidate_manifest(config: dict[str, Any], override: Path | None) -> dict[str, Any]:
    path = _resolve_candidate_manifest_path(config, override)
    digest = file_sha256(path)
    expected = None
    pinned = config.get("pinned_inputs", {})
    if isinstance(pinned, dict):
        expected = pinned.get("candidate_manifest_gz_sha256") or pinned.get(
            "candidate_manifest_raw_sha256"
        )
    if expected and digest != expected:
        raise CasePortfolioError(
            f"candidate manifest raw SHA-256 mismatch for {path}: expected {expected}, got {digest}"
        )
    payload = read_json_or_gzip(path)
    if not isinstance(payload, dict):
        raise CasePortfolioError(f"candidate manifest must be a dict: {path}")
    if payload.get("schema_version") != "seed_flip_inversion_candidates.v1":
        raise CasePortfolioError(
            "candidate manifest schema mismatch: expected "
            f"'seed_flip_inversion_candidates.v1', got {payload.get('schema_version')!r}"
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="build_ch7_worked_example_portfolio",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Chapter 7 worked-example portfolio YAML config.",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=None,
        help="Optional seed_flip_inversion_candidates.v1 JSON or JSON.GZ inventory.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write manifest JSON here.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run structural validation and print the validation summary to stderr.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the portfolio-builder CLI."""
    args = build_parser().parse_args(argv)
    try:
        config = _load_yaml(args.config)
        candidate_manifest = _load_candidate_manifest(config, args.candidate_manifest)
        manifest = finalize_manifest(
            build_ch7_worked_example_portfolio(config, candidate_manifest=candidate_manifest)
        )
    except (OSError, json.JSONDecodeError, yaml.YAMLError, CasePortfolioError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json is not None:
        write_deterministic_json(manifest, args.json)
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))

    summary = manifest["summary"]
    print(
        f"status={manifest['status']} selected={summary['n_selected']} "
        f"eligible={summary['n_eligible']}/{summary['n_inventory']} "
        f"uncovered_roles={','.join(summary['uncovered_roles']) or '-'}",
        file=sys.stderr,
    )

    if args.validate:
        result = validate_ch7_worked_example_portfolio(manifest)
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True), file=sys.stderr)
        if not result.ok:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
