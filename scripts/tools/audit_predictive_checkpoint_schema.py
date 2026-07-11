"""CPU-only schema-compatibility audit for predictive-planner campaign arms (issue #5241).

For every predictive planner arm in a camera-ready campaign config, this tool loads *only* the
checkpoint metadata / feature spec (never the full model, never a GPU) and reuses the exact
runtime schema comparison (``validate_predictive_feature_schema_metadata``) to classify each arm
COMPAT or INCOMPAT. With ``--emit-filtered-config`` it writes a copy of the campaign config with
incompatible arms removed and a ``schema_excluded_arms`` provenance list, so exclusion is
recorded rather than silent.

This unblocks the gap-prediction and mppi-social comparison campaigns (job 13194: the
``prediction_planner_v2_xl_ego`` checkpoint declared ``predictive_ego_v1`` while the runtime
expected ``predictive_legacy_v1`` -> ``ObstacleFeatureSchemaError`` under ``stop_on_failure``).

Modes:

- default: classify arms whose checkpoint is present locally; report ``UNSTAGED`` for the rest.
  Network-free. Run this for the audit table / filtered config when checkpoints are cached.
- ``--stage``: stage registry-backed checkpoints into the durable cache (CPU download; still no
  model instantiation / no GPU) before reading metadata, so every arm gets a definitive
  COMPAT/INCOMPAT classification.

Exit codes:

- ``0`` -- audit completed (regardless of how many arms are INCOMPAT; the filtered config records
  them). Check the table / report for INCOMPAT rows.
- ``2`` -- the campaign config file is missing or unreadable.
- ``3`` -- an unexpected error while staging or reading checkpoints (fail-closed).

Examples:

    # Audit + emit filtered config (checkpoints already cached locally):
    uv run python scripts/tools/audit_predictive_checkpoint_schema.py \\
        --config configs/benchmarks/paper_experiment_matrix_v1_gap_prediction_compare.yaml \\
        --emit-filtered-config configs/benchmarks/paper_experiment_matrix_v1_gap_prediction_compare_schema_filtered.yaml

    # Stage registry checkpoints first so every arm is classified:
    uv run python scripts/tools/audit_predictive_checkpoint_schema.py --stage \\
        --config configs/benchmarks/paper_experiment_matrix_v1_mppi_social_compare.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.predictive_checkpoint_schema_audit import (
    audit_predictive_checkpoint_schema_from_config,
    emit_schema_filtered_config,
    format_schema_audit_table,
)

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_AUDIT_ERROR = 3


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="CPU-only predictive-checkpoint schema audit + auto-filtered campaign configs "
        "(issue #5241).",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Camera-ready campaign config YAML to audit.",
    )
    parser.add_argument(
        "--emit-filtered-config",
        metavar="OUT.yaml",
        default=None,
        help="Write a copy of the campaign config with INCOMPAT arms removed and a "
        "schema_excluded_arms provenance list.",
    )
    parser.add_argument(
        "--stage",
        action="store_true",
        help="Stage registry-backed checkpoints into the durable cache (CPU download, no GPU) "
        "before reading metadata so every arm is classified COMPAT/INCOMPAT.",
    )
    parser.add_argument(
        "--report-path",
        metavar="PATH",
        default=None,
        help="Optional path to write a JSON manifest of the per-arm audit.",
    )
    parser.add_argument(
        "--registry-path",
        default=None,
        help="Optional model-registry path override (default: model/registry.yaml).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory override for staged registry downloads.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the predictive-checkpoint schema audit CLI.

    Returns:
        int: Process exit code (0 ok, 2 config error, 3 audit error).
    """
    args = build_arg_parser().parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"error: campaign config not found: {config_path}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    try:
        result = audit_predictive_checkpoint_schema_from_config(
            config_path,
            stage=bool(args.stage),
            registry_path=args.registry_path,
            cache_dir=args.cache_dir,
        )
    except FileNotFoundError as exc:
        print(f"error: campaign config unreadable: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: schema audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_AUDIT_ERROR

    print(format_schema_audit_table(result))
    print()
    if result.incompatible_arms:
        print(
            f"Found {len(result.incompatible_arms)} INCOMPAT arm(s); "
            "use --emit-filtered-config to drop them with provenance."
        )

    if args.emit_filtered_config:
        out = emit_schema_filtered_config(config_path, result, args.emit_filtered_config)
        print(f"Wrote filtered config: {out}")

    if args.report_path:
        report = Path(args.report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(result.to_manifest(), indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"Wrote audit manifest: {report}")

    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
