#!/usr/bin/env python3
"""Per-planner control-wandering + hybrid command-source-switch profile.

Parses oscillation predicates from the retained issue4206 trace-capable h600 rerun
(6291 of 6480 episodes carry a populated oscillatory_control_predicate) and produces:
  1. Per-planner percentile CSV for progress_ratio, heading_rate_sign_changes,
     command_source_changes (with N per cell).
  2. Per-scenario-family wandering + command-source-switch profile for hybrid planners.
  3. Two PNGs: progress_ratio distribution (planners overlaid) and per-planner
     command-source-change rate bar chart.

CPU-only, no simulation, no GPU, no campaign/Slurm, no metric writes.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_RUNS = "output/benchmarks/camera_ready/issue4206_trace_capable_h600_rerun_20260704/runs"
OUTPUT_DIR = "output/oscillatory_control_profile"

FIELDS = [
    "progress_ratio",
    "command_source_changes",
    "heading_rate_sign_changes",
    "linear_velocity_sign_changes",
    "mean_abs_jerk",
    "path_length_m",
    "net_progress_m",
]

HYBRID_PLANNERS = [
    "scenario_adaptive_hybrid_orca_v1",
    "hybrid_rule_v3_fast_progress_static_escape",
]


def scenario_family_from_id(scenario_id: str) -> str:
    """Extract scenario_family from scenario_id."""
    if scenario_id.startswith("classic_"):
        # Remove 'classic_' prefix and difficulty suffix (_high, _low, _medium)
        parts = scenario_id[len("classic_"):].rsplit("_", 1)
        return parts[0]
    elif scenario_id.startswith("francis2023_"):
        return scenario_id[len("francis2023_"):]
    return scenario_id


def load_episodes(base_runs: str) -> list[dict]:
    """Load all episodes from all planners."""
    episodes = []
    for planner_dir in sorted(os.listdir(base_runs)):
        episodes_file = os.path.join(base_runs, planner_dir, "episodes.jsonl")
        if not os.path.exists(episodes_file):
            continue
        # Planner name is directory without __differential_drive suffix
        planner = planner_dir.replace("__differential_drive", "")
        with open(episodes_file) as f:
            for line in f:
                ep = json.loads(line)
                fields = (
                    ep.get("safety_predicates", {})
                    .get("oscillatory_control_predicate", {})
                    .get("fields", {})
                )
                if not fields:
                    continue
                row = {
                    "planner": planner,
                    "scenario_id": ep.get("scenario_id", ""),
                    "scenario_family": scenario_family_from_id(
                        ep.get("scenario_id", "")
                    ),
                    "seed": ep.get("seed"),
                }
                for field in FIELDS:
                    row[field] = fields.get(field)
                episodes.append(row)
    return episodes


def percentile_table(
    episodes: list[dict], group_col: str, value_col: str
) -> dict[str, dict]:
    """Compute percentile stats per group."""
    grouped = defaultdict(list)
    for ep in episodes:
        val = ep.get(value_col)
        if val is not None:
            grouped[ep[group_col]].append(val)
    result = {}
    for group, values in sorted(grouped.items()):
        arr = np.array(values, dtype=float)
        result[group] = {
            "N": len(arr),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "mean": float(np.mean(arr)),
        }
    return result


def write_per_planner_csv(episodes: list[dict], output_path: Path):
    """Write per-planner percentile CSV for key metrics."""
    rows = []
    planners = sorted({ep["planner"] for ep in episodes})
    for planner in planners:
        pl_eps = [ep for ep in episodes if ep["planner"] == planner]
        row = {"planner": planner, "N": len(pl_eps)}
        for col in ["progress_ratio", "heading_rate_sign_changes", "command_source_changes"]:
            vals = [ep[col] for ep in pl_eps if ep[col] is not None]
            if vals:
                arr = np.array(vals, dtype=float)
                row[f"{col}_mean"] = round(float(np.mean(arr)), 4)
                row[f"{col}_p10"] = round(float(np.percentile(arr, 10)), 4)
                row[f"{col}_p25"] = round(float(np.percentile(arr, 25)), 4)
                row[f"{col}_p50"] = round(float(np.percentile(arr, 50)), 4)
                row[f"{col}_p75"] = round(float(np.percentile(arr, 75)), 4)
                row[f"{col}_p90"] = round(float(np.percentile(arr, 90)), 4)
        rows.append(row)

    headers = list(rows[0].keys())
    with open(output_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")
    print(f"Wrote {output_path} ({len(rows)} planners)")


def write_hybrid_scenario_csv(episodes: list[dict], output_path: Path):
    """Write per-scenario-family profile for hybrid planners."""
    hybrid_eps = [ep for ep in episodes if ep["planner"] in HYBRID_PLANNERS]
    families = sorted({ep["scenario_family"] for ep in hybrid_eps})

    rows = []
    for family in families:
        fam_eps = [ep for ep in hybrid_eps if ep["scenario_family"] == family]
        for planner in HYBRID_PLANNERS:
            pl_eps = [ep for ep in fam_eps if ep["planner"] == planner]
            if not pl_eps:
                continue
            row = {
                "scenario_family": family,
                "planner": planner,
                "N": len(pl_eps),
            }
            for col in ["progress_ratio", "command_source_changes"]:
                vals = [ep[col] for ep in pl_eps if ep[col] is not None]
                if vals:
                    arr = np.array(vals, dtype=float)
                    row[f"{col}_median"] = round(float(np.median(arr)), 4)
                    row[f"{col}_mean"] = round(float(np.mean(arr)), 4)
            rows.append(row)

    headers = list(rows[0].keys()) if rows else []
    with open(output_path, "w") as f:
        if headers:
            f.write(",".join(headers) + "\n")
            for row in rows:
                f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")
    print(f"Wrote {output_path} ({len(rows)} rows)")


def plot_progress_ratio_distribution(episodes: list[dict], output_path: Path):
    """Plot overlaid progress_ratio histograms per planner."""
    planners = sorted({ep["planner"] for ep in episodes})
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.linspace(0, 1.05, 50)
    for planner in planners:
        vals = [
            ep["progress_ratio"]
            for ep in episodes
            if ep["planner"] == planner and ep["progress_ratio"] is not None
        ]
        if vals:
            ax.hist(vals, bins=bins, alpha=0.4, label=planner, density=True)
    ax.set_xlabel("progress_ratio")
    ax.set_ylabel("Density")
    ax.set_title("Progress Ratio Distribution by Planner")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_command_source_changes(episodes: list[dict], output_path: Path):
    """Bar chart of mean command_source_changes per planner."""
    planners = sorted({ep["planner"] for ep in episodes})
    means = []
    for planner in planners:
        vals = [
            ep["command_source_changes"]
            for ep in episodes
            if ep["planner"] == planner and ep["command_source_changes"] is not None
        ]
        means.append(float(np.mean(vals)) if vals else 0.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(planners, means, color="steelblue")
    ax.set_xlabel("Planner")
    ax.set_ylabel("Mean command_source_changes")
    ax.set_title("Mean Command Source Changes per Planner")
    ax.tick_params(axis="x", rotation=45)
    # Highlight hybrid planners
    for i, planner in enumerate(planners):
        if planner in HYBRID_PLANNERS:
            bars[i].set_color("darkorange")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def _planner_progress_table(episodes: list[dict]) -> list[str]:
    """Build per-planner progress_ratio table lines."""
    lines = ["### Per-Planner Progress Ratio\n"]
    lines.append("| Planner | N | p10 | p25 | Median | p75 | p90 | Mean |")
    lines.append("|---------|---|-----|-----|--------|-----|-----|------|")
    planners = sorted({ep["planner"] for ep in episodes})
    for planner in planners:
        vals = [
            ep["progress_ratio"]
            for ep in episodes
            if ep["planner"] == planner and ep["progress_ratio"] is not None
        ]
        if vals:
            arr = np.array(vals, dtype=float)
            lines.append(
                f"| {planner} | {len(arr)} | "
                f"{np.percentile(arr, 10):.3f} | "
                f"{np.percentile(arr, 25):.3f} | "
                f"{np.median(arr):.3f} | "
                f"{np.percentile(arr, 75):.3f} | "
                f"{np.percentile(arr, 90):.3f} | "
                f"{np.mean(arr):.3f} |"
            )
    return lines


def _worst_wanderers(episodes: list[dict]) -> list[str]:
    """Build worst wanderers section."""
    lines = ["\n### Worst Wanderers (lowest median progress_ratio)\n"]
    planners = sorted({ep["planner"] for ep in episodes})
    planner_medians = {}
    for planner in planners:
        vals = [
            ep["progress_ratio"]
            for ep in episodes
            if ep["planner"] == planner and ep["progress_ratio"] is not None
        ]
        if vals:
            planner_medians[planner] = float(np.median(vals))
    worst = sorted(planner_medians.items(), key=lambda x: x[1])[:3]
    for planner, median in worst:
        lines.append(f"- **{planner}**: median progress_ratio = {median:.3f}")
    return lines


def _hybrid_command_source_table(episodes: list[dict]) -> list[str]:
    """Build hybrid planner command source switches table."""
    lines = ["\n### Hybrid Planner Command Source Switches\n"]
    lines.append("| Planner | N | Mean | Median | p90 |")
    lines.append("|---------|---|------|--------|-----|")
    for planner in HYBRID_PLANNERS:
        vals = [
            ep["command_source_changes"]
            for ep in episodes
            if ep["planner"] == planner and ep["command_source_changes"] is not None
        ]
        if vals:
            arr = np.array(vals, dtype=float)
            lines.append(
                f"| {planner} | {len(arr)} | "
                f"{np.mean(arr):.2f} | "
                f"{np.median(arr):.1f} | "
                f"{np.percentile(arr, 90):.1f} |"
            )
    return lines


def _hybrid_scenario_family_table(episodes: list[dict]) -> list[str]:
    """Build hybrid planners per-scenario-family table."""
    lines = ["\n### Hybrid Planners: Per-Scenario-Family Profile\n"]
    lines.append("| Scenario Family | Planner | N | progress_ratio median | command_source_changes mean |")
    lines.append("|-----------------|---------|---|----------------------|----------------------------|")
    hybrid_eps = [ep for ep in episodes if ep["planner"] in HYBRID_PLANNERS]
    families = sorted({ep["scenario_family"] for ep in hybrid_eps})
    for family in families:
        for planner in HYBRID_PLANNERS:
            vals_pr = [
                ep["progress_ratio"]
                for ep in hybrid_eps
                if ep["scenario_family"] == family
                and ep["planner"] == planner
                and ep["progress_ratio"] is not None
            ]
            vals_csc = [
                ep["command_source_changes"]
                for ep in hybrid_eps
                if ep["scenario_family"] == family
                and ep["planner"] == planner
                and ep["command_source_changes"] is not None
            ]
            if vals_pr and vals_csc:
                lines.append(
                    f"| {family} | {planner} | {len(vals_pr)} | "
                    f"{np.median(vals_pr):.3f} | "
                    f"{np.mean(vals_csc):.2f} |"
                )
    return lines


def build_summary(episodes: list[dict]) -> str:
    """Build summary text for issue comment."""
    lines = ["## Oscillatory Control Profile Summary\n"]
    lines.extend(_planner_progress_table(episodes))
    lines.extend(_worst_wanderers(episodes))
    lines.extend(_hybrid_command_source_table(episodes))
    lines.extend(_hybrid_scenario_family_table(episodes))
    return "\n".join(lines)


def main():
    """Run the oscillatory control profile analysis."""
    print("Loading episodes...")
    episodes = load_episodes(BASE_RUNS)
    print(f"Loaded {len(episodes)} episodes")

    if not episodes:
        print("ERROR: No episodes found", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_dir = Path(OUTPUT_DIR)

    # 1. Per-planner CSV
    write_per_planner_csv(episodes, output_dir / "per_planner_percentiles.csv")

    # 2. Hybrid per-scenario-family CSV
    write_hybrid_scenario_csv(
        episodes, output_dir / "hybrid_scenario_family_profile.csv"
    )

    # 3. Plots
    plot_progress_ratio_distribution(
        episodes, output_dir / "progress_ratio_distribution.png"
    )
    plot_command_source_changes(
        episodes, output_dir / "command_source_changes_rate.png"
    )

    # 4. Summary text
    summary = build_summary(episodes)
    summary_path = output_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Wrote {summary_path}")

    # Print summary to stdout
    print("\n" + summary)


if __name__ == "__main__":
    main()
