#!/usr/bin/env python
"""Build the Issue #2946 diagnostic mechanism-evidence figure pack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "issue-2946-mechanism-figure-pack"
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_GENERATED_AT = "2026-06-19T00:00:00+00:00"


@dataclass
class InputArtifact:
    """Tracked source artifact used to build the figure pack."""

    name: str
    path: Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from a tracked input path."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Load CSV rows from a tracked input path."""
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def finite_float(value: str | None) -> float | None:
    """Parse a finite float value; return None for blank, NaN, or infinite cells."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    parsed = float(text)
    if not math.isfinite(parsed):
        return None
    return parsed


def copy_panel(src: Path, dst: Path) -> dict[str, Any]:
    """Copy a tracked PNG panel into the pack and return output provenance."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    dst.write_bytes(data)
    return {
        "source": str(src),
        "destination": str(dst),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def build_seed_delta_chart(
    rows: list[dict[str, str]],
    out_path: Path,
) -> dict[str, Any]:
    """Render the Issue #2432 seed-pair delta breakdown."""
    labels: list[str] = []
    deltas: list[float] = []
    skipped_non_finite_rows = 0
    for row in rows:
        delta = finite_float(row["per_frame_max_abs_delta"])
        if delta is None:
            skipped_non_finite_rows += 1
            continue
        labels.append(f"{row['scenario_id']}:{row['seed']}")
        deltas.append(delta)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, deltas)
    ax.set_title("Issue #2432 frame-level delta by seed")
    ax.set_xlabel("scenario:seed")
    ax.set_ylabel("per_frame_max_abs_delta")
    ax.set_ylim(0, max(deltas + [0.0]) * 1.1 if any(deltas) else 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=30)

    zero_count = sum(1 for d in deltas if d == 0.0)
    summary = {
        "input_rows": len(rows),
        "plotted_rows": len(deltas),
        "skipped_non_finite_rows": skipped_non_finite_rows,
        "zero_delta_rows": zero_count,
        "nonzero_delta_rows": len(deltas) - zero_count,
    }
    if not deltas:
        ax.text(
            0.5,
            0.95,
            "No finite seed-level frame deltas in this slice",
            ha="center",
            va="top",
            transform=ax.transAxes,
        )
    elif summary["zero_delta_rows"] == len(deltas):
        ax.text(
            0.5,
            0.95,
            "No non-zero seed-level frame deltas in this slice",
            ha="center",
            va="top",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, metadata={"Date": DEFAULT_GENERATED_AT})
    plt.close(fig)
    return {
        "figure": str(out_path),
        "summary": summary,
    }


def build_scenario_max_delta_chart(
    rows: list[dict[str, str]],
    out_path: Path,
) -> dict[str, Any]:
    """Render the Issue #2434 scenario sweep max-delta summary."""
    metric_fields = [
        "per_frame_max_abs_delta",
        "robot_max_abs_delta",
        "selected_action_max_abs_delta",
        "event_max_abs_delta",
        "ammv_force_vectors_max_abs_delta",
    ]
    scenario_metrics: dict[str, dict[str, float]] = defaultdict(
        lambda: dict.fromkeys(metric_fields, 0.0)
    )
    skipped_non_finite_cells = 0

    for row in rows:
        scenario = row["scenario_id"]
        for field in metric_fields:
            value = finite_float(row[field])
            if value is None:
                skipped_non_finite_cells += 1
                continue
            scenario_metrics[scenario][field] = max(scenario_metrics[scenario][field], value)

    scenarios = sorted(scenario_metrics.keys())
    x = range(len(scenarios))
    width = 0.16

    fig, ax = plt.subplots(figsize=(11, 4))
    for idx, field in enumerate(metric_fields):
        values = [scenario_metrics[scenario][field] for scenario in scenarios]
        offset = (idx - (len(metric_fields) - 1) / 2) * width
        ax.bar([i + offset for i in x], values, width, label=field)

    ax.set_title("Issue #2434 scenario sweep max deltas (all zero in this classic slice)")
    ax.set_xlabel("scenario_id")
    ax.set_ylabel("max delta")
    ax.set_xticks(list(x))
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    max_value = (
        max(value for scenario in scenarios for value in scenario_metrics[scenario].values())
        if scenarios
        else 0.0
    )
    ax.set_ylim(0, max(1e-6, max_value) * 1.1)
    if not scenarios:
        ax.text(
            0.5,
            0.95,
            "No finite scenario sweep deltas in this slice",
            ha="center",
            va="top",
            transform=ax.transAxes,
        )

    zero_count = sum(
        1 for scenario in scenarios for value in scenario_metrics[scenario].values() if value == 0.0
    )
    fig.tight_layout()
    fig.savefig(out_path, metadata={"Date": DEFAULT_GENERATED_AT})
    plt.close(fig)
    return {
        "figure": str(out_path),
        "scenarios": scenarios,
        "input_rows": len(rows),
        "skipped_non_finite_cells": skipped_non_finite_cells,
        "zero_metric_cells": zero_count,
        "metric_count": len(metric_fields) * len(scenarios),
    }


def build_row_type_bars(
    fixture: dict[str, Any],
    runtime: dict[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    """Render signalized row-type eligibility and exclusion counts."""
    row_types = sorted(set(fixture["row_types_present"]).union(runtime["row_types_present"]))
    buckets = [
        "eligible",
        "excluded",
    ]

    def row_buckets(summary: dict[str, Any]) -> dict[tuple[str, str], int]:
        bucketed: dict[tuple[str, str], int] = defaultdict(int)
        for row in summary.get("eligible_rows", []):
            bucketed[(row["row_type"], "eligible")] += 1
        for row in summary.get("excluded_rows", []):
            bucketed[(row["row_type"], "excluded")] += 1
        return bucketed

    fixture_b = row_buckets(fixture)
    runtime_b = row_buckets(runtime)

    x = list(range(len(row_types)))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, bucket in enumerate(buckets):
        fixture_values = [fixture_b.get((rt, bucket), 0) for rt in row_types]
        runtime_values = [runtime_b.get((rt, bucket), 0) for rt in row_types]
        base = [p + i * width for p in x]
        ax.bar([p - width / 2 for p in base], fixture_values, width, label=f"fixture {bucket}")
        ax.bar([p + width / 2 for p in base], runtime_values, width, label=f"runtime {bucket}")

    ax.set_title("Signalized row-type inclusion vs exclusion (fixture vs runtime)")
    ax.set_xlabel("row_type")
    ax.set_ylabel("count")
    ax.set_xticks(x)
    ax.set_xticklabels(row_types)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, metadata={"Date": DEFAULT_GENERATED_AT})
    plt.close(fig)

    return {
        "figure": str(out_path),
        "fixture_row_type_counts": {
            k: dict(v)
            for k, v in {
                "eligible": {rt: fixture_b.get((rt, "eligible"), 0) for rt in row_types},
                "excluded": {rt: fixture_b.get((rt, "excluded"), 0) for rt in row_types},
            }.items()
        },
        "runtime_row_type_counts": {
            "eligible": {rt: runtime_b.get((rt, "eligible"), 0) for rt in row_types},
            "excluded": {rt: runtime_b.get((rt, "excluded"), 0) for rt in row_types},
        },
    }


def main() -> int:
    """Build figures, manifest, and evidence README."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/context/evidence/issue_2946_mechanism_figure_pack_2026-06-19"),
    )
    parser.add_argument(
        "--generated-at",
        default=DEFAULT_GENERATED_AT,
        help="Stable timestamp written to tracked provenance outputs.",
    )
    args = parser.parse_args()

    input_artifacts = {
        "issue_2432": InputArtifact(
            "issue_2432 candidate pair CSV",
            Path(
                "docs/context/evidence/issue_2432_ammv_trace_selection_2026-06-06/candidate_pair_comparison.csv"
            ),
        ),
        "issue_2432_summary": InputArtifact(
            "issue_2432 summary",
            Path("docs/context/evidence/issue_2432_ammv_trace_selection_2026-06-06/summary.json"),
        ),
        "issue_2434": InputArtifact(
            "issue_2434 scenario sweep CSV",
            Path(
                "docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/candidate_pair_comparison.csv"
            ),
        ),
        "issue_2434_summary": InputArtifact(
            "issue_2434 summary",
            Path("docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/summary.json"),
        ),
        "issue_2428_default_panel": InputArtifact(
            "issue_2428 default panel PNG",
            Path(
                "docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/panels/trajectory_panels/trajectory_panel_default_social_force_classic_head_on_corridor_low_other_classic_head_on_corridor_low--111--b05ccbf52ac7ab9b.png"
            ),
        ),
        "issue_2428_ammv_panel": InputArtifact(
            "issue_2428 ammv panel PNG",
            Path(
                "docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/panels/trajectory_panels/trajectory_panel_ammv_social_force_classic_head_on_corridor_low_other_classic_head_on_corridor_low--111--5fde302d5414e476.png"
            ),
        ),
        "issue_2753": InputArtifact(
            "issue_2753 fixture row summary",
            Path("docs/context/evidence/issue_2753_signalized_crossing_metrics/summary.json"),
        ),
        "issue_2799": InputArtifact(
            "issue_2799 runtime row summary",
            Path("docs/context/evidence/issue_2799_signalized_runtime/summary.json"),
        ),
    }

    for key, artifact in input_artifacts.items():
        if not artifact.path.exists():
            raise FileNotFoundError(f"Missing required input {key}: {artifact.path}")

    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows_2432 = read_csv(input_artifacts["issue_2432"].path)
    rows_2434 = read_csv(input_artifacts["issue_2434"].path)
    fixture = read_json(input_artifacts["issue_2753"].path)
    runtime = read_json(input_artifacts["issue_2799"].path)

    figure_catalog = {}

    default_panel_copy = copy_panel(
        input_artifacts["issue_2428_default_panel"].path,
        figures_dir / "01_panel_default_social_force.png",
    )
    figure_catalog["01_panel_default_social_force"] = {
        "type": "png",
        "path": str(figures_dir / "01_panel_default_social_force.png"),
        "title": "Issue #2428 default Social Force trajectory panel",
        "claim_boundary": "Single-row, scenario-seeded diagnostic panel; does not imply general AMMV advantage.",
        "sources": [str(input_artifacts["issue_2428_default_panel"].path)],
        "provenance": default_panel_copy,
        "output_sha256": default_panel_copy["sha256"],
    }

    ammv_panel_copy = copy_panel(
        input_artifacts["issue_2428_ammv_panel"].path,
        figures_dir / "02_panel_ammv_social_force.png",
    )
    figure_catalog["02_panel_ammv_social_force"] = {
        "type": "png",
        "path": str(figures_dir / "02_panel_ammv_social_force.png"),
        "title": "Issue #2428 AMMV-aware Social Force trajectory panel",
        "claim_boundary": "Single-row, scenario-seeded diagnostic panel; demonstrates AMMV trace renderability only.",
        "sources": [str(input_artifacts["issue_2428_ammv_panel"].path)],
        "provenance": ammv_panel_copy,
        "output_sha256": ammv_panel_copy["sha256"],
    }

    seed_delta_info = build_seed_delta_chart(
        rows_2432, figures_dir / "03_seed_pair_delta_breakdown.svg"
    )
    figure_catalog["03_seed_pair_delta_breakdown"] = {
        "type": "svg",
        "path": str(figures_dir / "03_seed_pair_delta_breakdown.svg"),
        "title": "Issue #2432 per-pair frame delta summary",
        "claim_boundary": "Adapter-mode local head-on seed slice only; all rows in this slice are numerically identical at frame level.",
        "sources": [
            str(input_artifacts["issue_2432"].path),
            str(input_artifacts["issue_2432_summary"].path),
        ],
        "metrics": seed_delta_info,
    }

    scenario_delta_info = build_scenario_max_delta_chart(
        rows_2434,
        figures_dir / "04_scenario_sweep_delta_summary.svg",
    )
    figure_catalog["04_scenario_sweep_delta_summary"] = {
        "type": "svg",
        "path": str(figures_dir / "04_scenario_sweep_delta_summary.svg"),
        "title": "Issue #2434 multi-scenario deltas (classic families)",
        "claim_boundary": "Adapter-mode compact 5-scenario sweep; no per-scenario frame/metric delta > 0 was found in the recorded outputs.",
        "sources": [
            str(input_artifacts["issue_2434"].path),
            str(input_artifacts["issue_2434_summary"].path),
        ],
        "metrics": scenario_delta_info,
    }

    row_type_info = build_row_type_bars(
        fixture, runtime, figures_dir / "05_signalized_row_type_counts.svg"
    )
    figure_catalog["05_signalized_row_type_counts"] = {
        "type": "svg",
        "path": str(figures_dir / "05_signalized_row_type_counts.svg"),
        "title": "Signalized row-type eligibility and exclusion",
        "claim_boundary": "Compliance denominator semantics only; two compatible observables and two denominator-zero excluded rows per source.",
        "sources": [
            str(input_artifacts["issue_2753"].path),
            str(input_artifacts["issue_2799"].path),
        ],
        "metrics": row_type_info,
    }

    source_metadata = {
        "issue": 2946,
        "generated_at": args.generated_at,
        "schema_version": "issue_2946_mechanism_figure_pack.v1",
        "inputs": {
            key: {
                "path": str(value.path),
                "name": value.name,
                "sha256": sha256_file(value.path),
            }
            for key, value in input_artifacts.items()
        },
        "output_dir": str(output_dir),
        "source_issue_lineage": [
            2159,
            2227,
            2428,
            2430,
            2432,
            2434,
            2444,
            2753,
            2754,
            2799,
            2923,
            2924,
            2946,
        ],
        "source_issue_notes": {
            "2444": (
                "Existing follow-up lane for a nonzero AMMV/default mechanism-divergence pair; "
                "no tracked evidence bundle from that issue was available as a direct input here."
            ),
            "2754": (
                "Existing follow-up lane for a signalized-crossing failure-case pack; current "
                "inputs use its predecessor denominator/row-status summaries only."
            ),
            "2924": (
                "Existing follow-up lane for counterfactual scenario-pair runner work; no tracked "
                "evidence bundle from that issue was available as a direct input here."
            ),
        },
        "existing_follow_up_issue_lanes": [2444, 2754, 2924],
        "figures": figure_catalog,
        "validation_commands": [
            "python -m json.tool docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/panels/trajectory_panel_manifest.json",
            "python -m json.tool docs/context/evidence/issue_2432_ammv_trace_selection_2026-06-06/summary.json",
            "python - <<'PY'\nimport csv\nfrom pathlib import Path\nrows = list(csv.DictReader(Path('docs/context/evidence/issue_2432_ammv_trace_selection_2026-06-06/candidate_pair_comparison.csv').open()))\nprint(f\"issue_2432 candidate rows: {len(rows)}\")\nPY",
            "python -m json.tool docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/summary.json",
            "python - <<'PY'\nimport csv\nfrom pathlib import Path\nrows = list(csv.DictReader(Path('docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/candidate_pair_comparison.csv').open()))\nprint(f\"issue_2434 candidate rows: {len(rows)}\")\nPY",
            "python -m json.tool docs/context/evidence/issue_2753_signalized_crossing_metrics/summary.json",
            "python -m json.tool docs/context/evidence/issue_2799_signalized_runtime/summary.json",
        ],
    }

    figure_manifest_path = output_dir / "figure_pack_manifest.json"
    with figure_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(source_metadata, f, indent=2)

    output_manifest = {
        "output_dir": str(output_dir),
        "figures_dir": str(figures_dir),
        "figure_count": len(figure_catalog),
        "first_run_manifest": str(figure_manifest_path),
        "command": "uv run python scripts/analysis/build_issue_2946_mechanism_figure_pack.py",
    }
    with (output_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Issue #2946 Mechanism-Evidence Figure Pack\n")
        f.write(f"Generated: {source_metadata['generated_at']}\n\n")
        f.write("## Figures\n\n")
        for key, value in figure_catalog.items():
            f.write(f"- {key}: `{Path(value['path']).name}`\n")
            f.write(f"  - title: {value['title']}\n")
            f.write(f"  - claim_boundary: {value['claim_boundary']}\n")
            f.write(f"  - sources: {', '.join(value['sources'])}\n\n")
        f.write("## Reproducibility\n\n")
        f.write("Run: `uv run python scripts/analysis/build_issue_2946_mechanism_figure_pack.py`\n")
        f.write(f"Manifest: `{figure_manifest_path}`\n")
        f.write("\n## Follow-up Boundaries\n\n")
        f.write(
            "This pack closes the first diagnostic figure-pack request for Issue #2946. "
            "Broader mechanism-evidence claims remain routed to existing follow-up lanes "
            "Issue #2444, Issue #2754, and Issue #2924.\n"
        )
    with (output_dir / "figure_pack_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(output_manifest, f, indent=2)

    print(f"Wrote figure pack to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
