"""Deterministic validation report for collision-cause attribution (issue #5443).

Loads the frozen ground-truth fixture manifest and, when supplied, a set of
analyser attribution verdicts, then emits a schema-tagged JSON report. It runs no
simulation and makes no benchmark or paper-grade claim.

Fail-closed behaviour: with no ``--verdicts`` file the report status is
``analyser_unavailable`` — the manifest is validated and the matrix coverage
confirmed, but accuracy is not scored, because the analyser under test
(#5441/#5442) does not yet exist to emit verdicts. Providing a verdicts file
scores attribution accuracy and applies the issue stop rule.

Usage::

    uv run python scripts/analysis/validate_collision_cause_attribution_issue_5443.py \
        --manifest tests/benchmark/fixtures/collision_cause_attribution_manifest_5443.json \
        [--verdicts path/to/analyser_verdicts.json] [--out output/report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.collision_cause_attribution import (
    REPORT_STATUS_SCORED,
    VERDICT_PASS,
    CollisionCauseAttributionError,
    build_validation_report,
)

_DEFAULT_MANIFEST = Path("tests/benchmark/fixtures/collision_cause_attribution_manifest_5443.json")


def _load_fixtures(manifest_path: Path) -> list[dict[str, Any]]:
    """Read the ``fixtures`` list from a manifest JSON file.

    Returns:
        The list of fixture mappings.

    Raises:
        CollisionCauseAttributionError: If the manifest lacks a ``fixtures`` list.
    """
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixtures = data.get("fixtures")
    if not isinstance(fixtures, list):
        raise CollisionCauseAttributionError(
            f"manifest {manifest_path} must contain a 'fixtures' list"
        )
    return fixtures


def _load_verdicts(verdicts_path: Path) -> list[dict[str, Any]]:
    """Read analyser verdicts from a JSON file (a list, or an object with ``verdicts``).

    Returns:
        The list of verdict mappings.

    Raises:
        CollisionCauseAttributionError: If the verdicts payload is malformed.
    """
    data = json.loads(verdicts_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    verdicts = data.get("verdicts") if isinstance(data, dict) else None
    if not isinstance(verdicts, list):
        raise CollisionCauseAttributionError(
            f"verdicts file {verdicts_path} must be a list or an object with a 'verdicts' list"
        )
    return verdicts


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="Path to the frozen ground-truth fixture manifest JSON.",
    )
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=None,
        help="Optional analyser verdicts JSON; omit to fail closed as analyser_unavailable.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the report JSON (also printed to stdout).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the validation report.

    Returns:
        Process exit code: 0 on ``analyser_unavailable`` or a scored ``pass``;
        1 when a scored report returns the ``revise`` verdict.
    """
    args = _parse_args(argv)
    try:
        fixtures = _load_fixtures(args.manifest)
        verdicts = _load_verdicts(args.verdicts) if args.verdicts is not None else None
        report = build_validation_report(fixtures, verdicts)
    except (CollisionCauseAttributionError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2

    payload = report.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)

    if report.status == REPORT_STATUS_SCORED and report.report is not None:
        return 0 if report.report.verdict == VERDICT_PASS else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
