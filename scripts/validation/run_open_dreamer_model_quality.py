"""Run the fail-closed issue #6318 Step 3 model-quality gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from robot_sf.research.open_dreamer_model_quality import (
    ModelQualityConfig,
    evaluate_model_quality,
    write_model_quality_report,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the config-first CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for the JSON report; it is created when absent.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Optional dataset override for a bounded diagnostic run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate the configured quality gate and return a fail-closed status code."""
    args = build_arg_parser().parse_args(argv)
    config = ModelQualityConfig.from_yaml(args.config)
    if args.dataset_path is not None:
        config = replace(config, dataset_path=args.dataset_path.resolve())
    report = evaluate_model_quality(config)
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "open_dreamer_model_quality.v1.json"
    write_model_quality_report(report, report_path)
    payload = report.to_dict()
    print(
        json.dumps(
            {
                "report_path": str(report_path),
                "status": payload["status"],
                "reason": payload["reason"],
                "evidence_boundary": payload["evidence_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
