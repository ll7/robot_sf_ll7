#!/usr/bin/env python3
"""Run the fixture-only leakage-safe parametric curriculum diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.training.parametric_curriculum import (
    PARAMETRIC_CURRICULUM_DIAGNOSTIC_SCHEMA,
    build_parameter_space,
    build_parametric_curriculum_report,
)


def _parser() -> argparse.ArgumentParser:
    """Build the config-first command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Diagnostic YAML config.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the diagnostic YAML mapping."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("curriculum diagnostic config must be a mapping")
    if payload.get("schema_version") != PARAMETRIC_CURRICULUM_DIAGNOSTIC_SCHEMA:
        raise ValueError("curriculum diagnostic config schema_version is invalid")
    return payload


def run(config_path: Path) -> dict[str, Any]:
    """Build, validate, and return one deterministic curriculum report."""

    config = _load_config(config_path)
    space = build_parameter_space(config.get("parameter_space"))
    report = build_parametric_curriculum_report(
        space,
        seed=int(config["seed"]),
        train_count=int(config["train_sample_count"]),
        evaluation_count=int(config["evaluation_sample_count"]),
    )
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "robot_sf"
        / "benchmark"
        / "schemas"
        / "parametric_curriculum_diagnostic.v1.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(report), key=lambda error: list(error.path)
    )
    if errors:
        raise ValueError(f"curriculum diagnostic schema validation failed: {errors[0].message}")
    return report


def main() -> int:
    """Run the diagnostic and optionally write its JSON report."""

    args = _parser().parse_args()
    report = run(args.config)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "ok",
                    "output": str(args.output),
                    "parameter_space_digest": report["parameter_space"]["digest"],
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
