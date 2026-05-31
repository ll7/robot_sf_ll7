"""Validate expert trajectory datasets from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf import common
from robot_sf.benchmark.validation.trajectory_dataset import TrajectoryDatasetValidator

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the trajectory dataset validator CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-id",
        "--dataset",
        dest="dataset_id",
        help="Dataset id under the canonical trajectory dataset directory.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Explicit dataset path. Use instead of --dataset-id for ad-hoc files.",
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=200,
        help="Minimum episode count required for a validated-quality dataset.",
    )
    parser.add_argument(
        "--fail-on-quarantine",
        action="store_true",
        help="Return exit code 2 when validation classifies the dataset as quarantined.",
    )
    parser.add_argument(
        "--require-decision-transformer-fields",
        action="store_true",
        help=(
            "Require reward, terminated, truncated, and return_to_go arrays even when the "
            "dataset metadata does not declare the Decision Transformer preflight schema."
        ),
    )
    return parser


def _resolve_dataset_path(dataset_id: str | None, path: Path | None) -> Path:
    """Resolve the dataset path from CLI arguments.

    Returns:
        Concrete dataset path for validation.

    Raises:
        ValueError: If neither or both path selectors are provided.
    """
    if bool(dataset_id) == bool(path):
        raise ValueError("Provide exactly one of --dataset-id/--dataset or --path.")
    if path is not None:
        return path
    assert dataset_id is not None
    return common.get_trajectory_dataset_path(dataset_id)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a trajectory dataset and print a JSON report.

    Returns:
        Process exit code.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        dataset_path = _resolve_dataset_path(args.dataset_id, args.path)
    except ValueError as exc:
        parser.error(str(exc))

    result = TrajectoryDatasetValidator(dataset_path).validate(
        minimum_episodes=int(args.min_episodes),
        require_decision_transformer_fields=(
            True if args.require_decision_transformer_fields else None
        ),
    )
    payload = {
        "dataset_id": result.dataset_id,
        "dataset_path": str(result.dataset_path),
        "episode_count": result.episode_count,
        "scenario_coverage": result.scenario_coverage,
        "quality_status": result.quality_status.value,
        "integrity_report": result.integrity_report,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.fail_on_quarantine and result.quality_status == common.TrajectoryQuality.QUARANTINED:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
