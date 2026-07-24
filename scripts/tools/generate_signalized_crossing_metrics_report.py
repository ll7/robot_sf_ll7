"""Generate fixture-based signalized crossing metrics report for issue #2753.

Constructs four bounded fixture rows exercising the canonical row types
defined in ``robot_sf.benchmark.signal_metrics``:

- ``red_required_stop``: observable red-phase crossing, benchmark_evidence=true
- ``green_proceed``: observable green-phase crossing, benchmark_evidence=true
- ``unavailable_no_claim``: no signal metadata, denominator-excluded
- ``proxy_only_denominator_excluded``: proxy diagnostic planner, denominator-excluded

This is fixture/report-table evidence only.  It does **not** run simulator
or runtime traces, therefore it does **not** establish compliance claims
beyond the report-row contract.  No claim-matrix updates are produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.signal_metrics import (
    render_report_rows_markdown,
    signal_metrics_report_rows,
)

_EVIDENCE_DIR = Path("docs/context/evidence/issue_2753_signalized_crossing_metrics")

_ROWS_SPEC: list[dict[str, Any]] = [
    {
        "episode_id": "fixture_red_required_stop",
        "robot_pos": np.array(
            [
                [0.0, 0.0],
                [11.0, 0.0],
                [12.0, 0.0],
                [13.0, 0.0],
                [14.0, 0.0],
                [15.0, 0.0],
            ]
        ),
        "peds_pos": np.zeros((6, 0, 2)),
        "dt": 1.0,
        "episode_metadata": {
            "signal_state": {
                "contract_state": "planner_observable",
                "benchmark_evidence": True,
                "timeline": [
                    {"state": "red", "duration": 5.0},
                    {"state": "green", "duration": 5.0},
                ],
                "stop_line": [[10.0, 10.0], [10.0, -10.0]],
                "crosswalk_polygon": [
                    [11.0, 10.0],
                    [15.0, 10.0],
                    [15.0, -10.0],
                    [11.0, -10.0],
                ],
            }
        },
    },
    {
        "episode_id": "fixture_green_proceed",
        "robot_pos": np.array(
            [
                [-1.0, 0.0],
                [-0.5, 0.0],
                [0.5, 0.0],
                [1.5, 0.0],
                [2.5, 0.0],
            ]
        ),
        "peds_pos": np.zeros((5, 0, 2)),
        "dt": 0.1,
        "episode_metadata": {
            "signal_state": {
                "contract_state": "planner_observable",
                "benchmark_evidence": True,
                "timeline": [
                    {"state": "red", "duration": 0.05},
                    {"state": "green", "duration": 5.0},
                ],
                "stop_line": [[0.0, 1.0], [0.0, -1.0]],
                "crosswalk_polygon": [
                    [1.0, 1.0],
                    [5.0, 1.0],
                    [5.0, -1.0],
                    [1.0, -1.0],
                ],
            }
        },
    },
    {
        "episode_id": "fixture_unavailable_no_claim",
        "robot_pos": np.zeros((10, 2)),
        "peds_pos": np.zeros((10, 0, 2)),
        "dt": 0.1,
        "episode_metadata": None,
    },
    {
        "episode_id": "fixture_proxy_only_denominator_excluded",
        "robot_pos": np.zeros((10, 2)),
        "peds_pos": np.zeros((10, 0, 2)),
        "dt": 0.1,
        "episode_metadata": {
            "signal_state": {
                "contract_state": "proxy_diagnostic",
            }
        },
    },
]


class _FixtureEpisode:
    """Minimal episode adapter for fixture construction."""

    def __init__(
        self,
        robot_pos: np.ndarray,
        peds_pos: np.ndarray,
        dt: float,
        episode_metadata: dict[str, Any] | None,
    ):
        self.robot_pos = robot_pos
        self.peds_pos = peds_pos
        self.dt = dt
        self.episode_metadata = episode_metadata


def _build_episodes() -> list[tuple[str, _FixtureEpisode]]:
    episodes: list[tuple[str, _FixtureEpisode]] = []
    for spec in _ROWS_SPEC:
        ep = _FixtureEpisode(
            robot_pos=spec["robot_pos"],
            peds_pos=spec["peds_pos"],
            dt=spec["dt"],
            episode_metadata=spec["episode_metadata"],
        )
        episodes.append((spec["episode_id"], ep))
    return episodes


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observable = [r for r in rows if r["signal_compliance_eligible"]]
    excluded = [r for r in rows if not r["signal_compliance_eligible"]]
    row_types = [r["row_type"] for r in rows]
    return {
        "issue": 2753,
        "title": "Signalized crossing metrics report-row fixture evidence",
        "claim_boundary": (
            "Fixture/report-table evidence only. "
            "No simulator or runtime traces were executed. "
            "No compliance claim is established beyond the report-row contract."
        ),
        "total_rows": len(rows),
        "row_types_present": sorted(set(row_types)),
        "observable_count": len(observable),
        "excluded_count": len(excluded),
        "eligible_rows": [
            {
                "episode_id": r["episode_id"],
                "row_type": r["row_type"],
                "planner_observable": r["planner_observable"],
                "benchmark_evidence": r["benchmark_evidence"],
                "signal_metrics_denominator": r["signal_metrics_denominator"],
            }
            for r in observable
        ],
        "excluded_rows": [
            {
                "episode_id": r["episode_id"],
                "row_type": r["row_type"],
                "planner_observable": r["planner_observable"],
                "benchmark_evidence": r["benchmark_evidence"],
                "signal_metrics_denominator": r["signal_metrics_denominator"],
                "exclusion_reason": r["exclusion_reason"],
            }
            for r in excluded
        ],
        "excluded_denominator_zero": all(r["signal_metrics_denominator"] == 0 for r in excluded),
        "excluded_not_compliance_eligible": all(
            not r["signal_compliance_eligible"] for r in excluded
        ),
    }


def _build_readme() -> str:
    return """\
# Issue #2753 Signalized Crossing Metrics Report-Row Fixture Evidence 2026-06-13

**Date:** 2026-06-13
**Commit:** (generated, see manifest)

## Scope

Fixture/report-table evidence for the four canonical signalized crossing
metric row types defined in `robot_sf/benchmark/signal_metrics.py`:

| Row type | planner_observable | benchmark_evidence | denominator | compliance_eligible |
|---|---|---|---|---|
| `red_required_stop` | true | true | 1 | true |
| `green_proceed` | true | true | 1 | true |
| `unavailable_no_claim` | false | false | 0 | false |
| `proxy_only_denominator_excluded` | false | false | 0 | false |

## Claim Boundary

This is fixture/report-table evidence only.  The script constructs
synthetic episodes and feeds them through `signal_metrics_report_rows`
to verify the report-row contract.  No simulator or runtime traces were
executed, so no compliance claim is established beyond the report-row
structure. Simulator-backed runtime evidence is deferred to issue #2799.

## Files

- `summary.json`: machine-readable fixture summary
- `report.md`: rendered Markdown report table
- `README.md`: this file

## Reproduction

```bash
uv run python scripts/tools/generate_signalized_crossing_metrics_report.py
```
"""


def generate(output_dir: Path) -> None:
    """Build fixture rows and write evidence artifacts."""
    episodes = _build_episodes()
    rows = signal_metrics_report_rows(episodes)
    summary = _build_summary(rows)
    markdown = render_report_rows_markdown(rows)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    report_path = output_dir / "report.md"
    report_md = (
        "# Issue #2753 Signalized Crossing Metrics Report-Row Fixture 2026-06-13\n\n"
        "## Claim Boundary\n\n"
        "Fixture/report-table evidence only.  No simulator or runtime traces.\n\n"
        "## Report Table\n\n"
        f"{markdown}\n"
    )
    report_path.write_text(report_md)

    readme_path = output_dir / "README.md"
    readme_path.write_text(_build_readme())


def main(argv: list[str] | None = None) -> int:
    """Entry point for the generator CLI."""
    parser = argparse.ArgumentParser(
        description="Generate signalized crossing metrics fixture report for issue #2753."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_EVIDENCE_DIR,
        help="Output directory for evidence artifacts.",
    )
    args = parser.parse_args(argv)
    generate(args.output_dir)
    print(f"Wrote evidence artifacts to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
