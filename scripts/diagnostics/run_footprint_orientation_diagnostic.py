"""Runner for the footprint-orientation diagnostic (issue #4762).

Loads ``configs/diagnostics/footprint_orientation_v1.yaml``, runs the
diagnostic on the five self-contained scenario-family fixtures, and writes a
JSON or Markdown report. CPU-only; no campaigns, training, or benchmark
claims.

Examples:
    python scripts/diagnostics/run_footprint_orientation_diagnostic.py
    python scripts/diagnostics/run_footprint_orientation_diagnostic.py --format markdown
    python scripts/diagnostics/run_footprint_orientation_diagnostic.py --format json --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.nav.footprint_diagnostic import (
    build_diagnostic_report,
    build_diagnostic_scenarios,
    load_footprint_orientation_config,
    parse_diagnostic_parameters,
    parse_footprints,
)

DEFAULT_CONFIG_PATH = Path("configs") / "diagnostics" / "footprint_orientation_v1.yaml"


def _resolve_config_path(config_path: Path) -> Path:
    """Resolve a config path against the repository root when relative."""

    path = Path(config_path)
    if path.is_absolute() and path.exists():
        return path
    repo_root = get_repository_root().resolve()
    candidate = path if path.is_absolute() else (repo_root / path)
    if not candidate.exists():
        raise FileNotFoundError(f"footprint-orientation config not found: {path}")
    return candidate


def build_markdown_report(report: dict) -> str:
    """Render a diagnostic report dict as a compact Markdown document."""

    lines: list[str] = []
    lines.append(f"# Footprint-orientation diagnostic ({report['profile_id']})")
    lines.append("")
    lines.append(f"> {report['claim_boundary_note']}")
    lines.append("")
    params = report["diagnostic_parameters"]
    lines.append(
        f"**Parameters:** sample_step={params['sample_step_m']} m, "
        f"max_samples={params['max_samples']}"
    )
    lines.append("")
    lines.append(
        "| Scenario | Mechanism | Footprint | Kind | Centerline (m) | "
        "Footprint-aware (m) | Status |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for scenario in report["scenarios"]:
        first = True
        for row in scenario["results"]:
            scenario_cell = scenario["display_name"] if first else ""
            mechanism_cell = scenario["mechanism"] if first else ""
            lines.append(
                "| {scenario} | {mechanism} | {footprint} | {kind} | {cl} | {fa} | {status} |".format(
                    scenario=scenario_cell,
                    mechanism=mechanism_cell,
                    footprint=row["footprint_id"],
                    kind=row["kind"],
                    cl=_fmt(row["centerline_clearance_m"]),
                    fa=_fmt(row["footprint_aware_clearance_m"]),
                    status=row["status"],
                )
            )
            first = False
    lines.append("")
    lines.append(
        "Centerline clearance ignores the footprint; footprint-aware clearance "
        "orients a rigid footprint along the local route tangent. A `collision` "
        "status means the oriented footprint intersects an obstacle. This is a "
        "diagnostic proxy, not a full SE(2) planner."
    )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the footprint-orientation diagnostic runner."""
    parser = argparse.ArgumentParser(
        description="Run the footprint-orientation diagnostic (issue #4762).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the footprint-orientation diagnostic YAML config.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: stdout).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the footprint-orientation diagnostic and emit a JSON or Markdown report."""
    args = parse_args(argv)
    config_path = _resolve_config_path(args.config)
    payload = load_footprint_orientation_config(config_path)
    footprints = parse_footprints(payload)
    params = parse_diagnostic_parameters(payload)
    scenarios = build_diagnostic_scenarios()
    report = build_diagnostic_report(
        scenarios,
        footprints,
        params["sample_step_m"],
        params["max_samples"],
        pass_threshold_m=params["pass_threshold_m"],
        profile_id=str(payload.get("profile_id", "footprint_orientation_diagnostic_v1")),
    )
    if args.format == "markdown":
        output_text = build_markdown_report(report)
    else:
        output_text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        Path(args.output).write_text(output_text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
