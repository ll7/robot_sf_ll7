"""Post-hoc interaction-validity annotation for an issue #4207 evidence packet.

This tool re-reads the per-cell aggregate metrics already recorded in a certification-transfer
``summary.json`` and classifies whether each cell exercised the robot-pedestrian near field
(see :func:`robot_sf.benchmark.certification_transfer.classify_interaction_status`). It does NOT
run a new simulation; it only annotates an existing recorded run so a reviewer can tell a genuine
``stable_pass``/``stable_fail`` (model swap exercised, no flip) apart from a vacuous one (the robot
never entered the pedestrian near field, so the SFM/HSFM swap was never exercised).

Outputs ``interaction_validity.csv`` and ``interaction_validity.md`` next to the summary. The
verdict is diagnostic-only and adds no deployment, safety, or paper/dissertation claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from robot_sf.benchmark.certification_transfer import (
    INTERACTION_NEAR_FIELD_M,
    classify_interaction_status,
)
from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256_file

CELL_COLUMNS = (
    "planner_key",
    "structural_class",
    "evaluation_model",
    "gate_status",
    "interaction_status",
    "min_clearance_m",
    "robot_ped_within_5m_frac",
    "episodes",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to a recorded certification-transfer summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for annotation artifacts (defaults to the summary directory).",
    )
    return parser.parse_args(argv)


def annotate(summary_path: Path, output_dir: Path) -> dict[str, Any]:
    """Classify interaction status for each recorded gate cell.

    Returns:
        Mapping with the per-cell rows and aggregate interaction-status counts.
    """

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for cell in summary.get("gate_cells", []):
        metrics = cell.get("metrics") or {}
        rows.append(
            {
                "planner_key": cell.get("planner_key"),
                "structural_class": cell.get("structural_class"),
                "evaluation_model": cell.get("evaluation_model"),
                "gate_status": cell.get("gate_status"),
                "interaction_status": classify_interaction_status(metrics),
                "min_clearance_m": metrics.get("min_clearance_m"),
                "robot_ped_within_5m_frac": metrics.get("robot_ped_within_5m_frac"),
                "episodes": cell.get("episodes"),
            }
        )
    counts = dict(Counter(row["interaction_status"] for row in rows))
    model_sensitivity_exercised = counts.get("interacting", 0) > 0
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "interaction_validity.csv"
    md_path = output_dir / "interaction_validity.md"
    _write_csv(csv_path, rows)
    md_path.write_text(
        _markdown(summary_path, rows, counts, model_sensitivity_exercised, csv_path),
        encoding="utf-8",
    )
    return {
        "rows": rows,
        "interaction_status_counts": counts,
        "model_sensitivity_exercised": model_sensitivity_exercised,
        "csv": str(csv_path),
        "md": str(md_path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CELL_COLUMNS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in CELL_COLUMNS})


def _csv_value(value: Any) -> Any:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def _markdown(
    summary_path: Path,
    rows: list[dict[str, Any]],
    counts: dict[str, int],
    model_sensitivity_exercised: bool,
    csv_path: Path,
) -> str:
    verdict = (
        "Model sensitivity WAS exercised: at least one cell entered the pedestrian near field."
        if model_sensitivity_exercised
        else "Model sensitivity was NOT exercised: every recorded cell stayed outside the "
        f"{INTERACTION_NEAR_FIELD_M:g} m pedestrian near field, so the certification-transfer "
        "statuses are vacuous and do NOT demonstrate certification robustness."
    )
    lines = [
        "# Issue #4207 Certification-Transfer Interaction-Validity Annotation",
        "",
        "Post-hoc, diagnostic-only annotation. This does NOT run a new simulation; it classifies "
        "the interaction status of each cell from the aggregate metrics already recorded in "
        f"`{summary_path.name}`.",
        "",
        f"- Near-field band: `{INTERACTION_NEAR_FIELD_M:g}` m "
        "(`social_force_default` and `hsfm_total_force_v1` only diverge inside it).",
        f"- Interaction status counts: `{counts}`",
        f"- Model sensitivity exercised: `{model_sensitivity_exercised}`",
        "",
        f"**Verdict:** {verdict}",
        "",
        "## Per-cell interaction status",
        "",
        "| planner_key | evaluation_model | gate_status | interaction_status | "
        "min_clearance_m | robot_ped_within_5m_frac | episodes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {planner_key} | {evaluation_model} | {gate_status} | {interaction_status} | "
            "{min_clearance_m} | {within} | {episodes} |".format(
                planner_key=row["planner_key"],
                evaluation_model=row["evaluation_model"],
                gate_status=row["gate_status"],
                interaction_status=row["interaction_status"],
                min_clearance_m=_csv_value(row["min_clearance_m"]),
                within=_csv_value(row["robot_ped_within_5m_frac"]),
                episodes=row["episodes"],
            )
        )
    lines += [
        "",
        "## Provenance",
        "",
        f"- Source summary: `{summary_path.name}` (sha256 `{_sha256_file(summary_path)}`)",
        f"- Companion CSV: `{csv_path.name}`",
        "- Claim boundary: diagnostic certification-transfer interaction validity; no deployment, "
        "safety, benchmark-strength, or paper/dissertation claim.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Annotate a recorded certification-transfer packet with interaction validity."""

    args = parse_args(argv)
    summary_path = args.summary.resolve()
    output_dir = (args.output_dir or summary_path.parent).resolve()
    result = annotate(summary_path, output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
