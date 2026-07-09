"""Issue #4904: per-planner response-latency + decel-demand intervention profile.

Parses episodes.jsonl from the trace-capable h600 rerun, extracts
response_latency_s and required_deceleration_m_s2 for episodes with
interaction_exposure_steps > 0, computes per-planner percentile tables,
and flags the goal-planner latency-null anomaly.

CPU-only, no simulation, no GPU, no campaign/Slurm.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN_DIR = Path("output/benchmarks/camera_ready/issue4206_trace_capable_h600_rerun_20260704/runs")
OUTPUT_DIR = Path("output")


def load_episodes(run_dir: Path) -> list[dict]:
    """Load all episodes from all planners."""
    episodes = []
    for planner_dir in sorted(run_dir.iterdir()):
        if not planner_dir.is_dir():
            continue
        planner_name = planner_dir.name.replace("__differential_drive", "")
        episodes_file = planner_dir / "episodes.jsonl"
        if not episodes_file.exists():
            continue
        with open(episodes_file) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_planner"] = planner_name
                episodes.append(record)
    return episodes


def _is_finite_number(value) -> bool:
    """True only for a real finite number.

    json.loads silently accepts ``NaN``/``Infinity``/``-Infinity`` tokens, so a
    non-finite field value would otherwise poison np.percentile/np.median. Booleans
    (a subclass of int) are excluded too — a stray ``true`` must not read as ``1.0``.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def extract_profiles(episodes: list[dict]) -> dict:
    """Extract latency/decel profiles per planner, only exposure > 0."""
    profiles = defaultdict(
        lambda: {
            "latencies": [],
            "decels": [],
            "late_evasive_true": 0,
            "latency_populated": 0,
            "total_exposure": 0,
        }
    )

    for ep in episodes:
        exposure = ep.get("interaction_exposure", {})
        exposure_steps = exposure.get("interaction_exposure_steps", 0)
        if exposure_steps <= 0:
            continue

        planner = ep["_planner"]
        prof = profiles[planner]
        prof["total_exposure"] += 1

        sp = ep.get("safety_predicates", {}).get("late_evasive_predicate", {})
        late_evasive = sp.get("late_evasive", False)
        if late_evasive:
            prof["late_evasive_true"] += 1

        fields = sp.get("fields", {})
        latency = fields.get("response_latency_s")
        decel = fields.get("required_deceleration_m_s2")

        if _is_finite_number(latency):
            prof["latency_populated"] += 1
            prof["latencies"].append(float(latency))

        if _is_finite_number(decel):
            prof["decels"].append(float(decel))

    return dict(profiles)


def compute_percentiles(values: list[float]) -> dict:
    """Compute p10/p25/p50/p75/p90/max for a list of values."""
    if not values:
        return {
            "N": 0,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    arr = np.array(values)
    pcts = np.percentile(arr, [10, 25, 50, 75, 90])
    return {
        "N": len(arr),
        "p10": round(float(pcts[0]), 6),
        "p25": round(float(pcts[1]), 6),
        "p50": round(float(pcts[2]), 6),
        "p75": round(float(pcts[3]), 6),
        "p90": round(float(pcts[4]), 6),
        "max": round(float(np.max(arr)), 6),
    }


def write_percentile_csv(profiles: dict, output_path: Path) -> None:
    """Write per-planner percentile CSV."""
    fieldnames = [
        "planner",
        "exposure_episodes",
        "late_evasive_true",
        "latency_populated",
        "lat_N",
        "lat_p10",
        "lat_p25",
        "lat_p50",
        "lat_p75",
        "lat_p90",
        "lat_max",
        "decel_N",
        "decel_p10",
        "decel_p25",
        "decel_p50",
        "decel_p75",
        "decel_p90",
        "decel_max",
        "late_evasive_rate",
        "latency_populated_rate",
        "anomaly_gap",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for planner in sorted(profiles):
            prof = profiles[planner]
            lat_pct = compute_percentiles(prof["latencies"])
            dec_pct = compute_percentiles(prof["decels"])
            total = prof["total_exposure"]
            le_rate = prof["late_evasive_true"] / total if total > 0 else 0
            lp_rate = prof["latency_populated"] / total if total > 0 else 0
            writer.writerow(
                {
                    "planner": planner,
                    "exposure_episodes": total,
                    "late_evasive_true": prof["late_evasive_true"],
                    "latency_populated": prof["latency_populated"],
                    "lat_N": lat_pct["N"],
                    "lat_p10": lat_pct["p10"],
                    "lat_p25": lat_pct["p25"],
                    "lat_p50": lat_pct["p50"],
                    "lat_p75": lat_pct["p75"],
                    "lat_p90": lat_pct["p90"],
                    "lat_max": lat_pct["max"],
                    "decel_N": dec_pct["N"],
                    "decel_p10": dec_pct["p10"],
                    "decel_p25": dec_pct["p25"],
                    "decel_p50": dec_pct["p50"],
                    "decel_p75": dec_pct["p75"],
                    "decel_p90": dec_pct["p90"],
                    "decel_max": dec_pct["max"],
                    "late_evasive_rate": round(le_rate, 4),
                    "latency_populated_rate": round(lp_rate, 4),
                    "anomaly_gap": round(le_rate - lp_rate, 4),
                }
            )


def plot_latency_histogram(profiles: dict, output_path: Path) -> None:
    """Plot overlaid response-latency histograms per planner."""
    fig, ax = plt.subplots(figsize=(10, 6))
    planners_sorted = sorted(profiles, key=lambda p: -len(profiles[p]["latencies"]))
    for planner in planners_sorted:
        lats = profiles[planner]["latencies"]
        if not lats:
            continue
        ax.hist(lats, bins=30, alpha=0.5, label=f"{planner} (n={len(lats)})")
    ax.set_xlabel("Response Latency (s)")
    ax.set_ylabel("Episode Count")
    ax.set_title("Per-Planner Response Latency Distribution (exposure>0, non-null)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_decel_histogram(profiles: dict, output_path: Path) -> None:
    """Plot overlaid decel-demand histograms per planner."""
    fig, ax = plt.subplots(figsize=(10, 6))
    planners_sorted = sorted(profiles, key=lambda p: -len(profiles[p]["decels"]))
    for planner in planners_sorted:
        decels = profiles[planner]["decels"]
        if not decels:
            continue
        ax.hist(decels, bins=30, alpha=0.5, label=f"{planner} (n={len(decels)})")
    ax.set_xlabel("Required Deceleration (m/s²)")
    ax.set_ylabel("Episode Count")
    ax.set_title("Per-Planner Decel-Demand Distribution (exposure>0, non-null)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_summary(profiles: dict) -> None:
    """Print summary stats for the issue comment."""
    # Find fastest/slowest reactors (by median latency among planners with latency data)
    median_latencies = {}
    for planner, prof in profiles.items():
        if prof["latencies"]:
            median_latencies[planner] = float(np.median(prof["latencies"]))

    if median_latencies:
        fastest = min(median_latencies, key=median_latencies.get)
        slowest = max(median_latencies, key=median_latencies.get)
        print("\n=== Response Latency (median, among planners with latency data) ===")
        print(f"  Fastest reactor: {fastest} (median={median_latencies[fastest]:.3f}s)")
        print(f"  Slowest reactor: {slowest} (median={median_latencies[slowest]:.3f}s)")

    # Find hardest brakers (by median decel)
    median_decels = {}
    for planner, prof in profiles.items():
        if prof["decels"]:
            median_decels[planner] = float(np.median(prof["decels"]))

    if median_decels:
        hardest = max(median_decels, key=median_decels.get)
        softest = min(median_decels, key=median_decels.get)
        print("\n=== Required Deceleration (median, among planners with decel data) ===")
        print(f"  Hardest braker: {hardest} (median={median_decels[hardest]:.6f} m/s²)")
        print(f"  Softest braker: {softest} (median={median_decels[softest]:.6f} m/s²)")

    # Goal-planner anomaly
    print("\n=== Goal-Planner Latency-Null Anomaly ===")
    for planner in sorted(profiles):
        prof = profiles[planner]
        total = prof["total_exposure"]
        le_true = prof["late_evasive_true"]
        lat_pop = prof["latency_populated"]
        anomaly = le_true > 0 and lat_pop == 0
        if anomaly or planner == "goal":
            print(
                f"  {planner}: exposure>0={total}, late_evasive_true={le_true}, latency_populated={lat_pop}, anomaly={anomaly}"
            )


def main() -> None:
    """Run the full analysis: parse, compute percentiles, plot, and summarize."""
    print(f"Loading episodes from {RUN_DIR}...")
    episodes = load_episodes(RUN_DIR)
    print(f"Loaded {len(episodes)} episodes total.")

    profiles = extract_profiles(episodes)
    print(f"Planners with exposure>0 episodes: {len(profiles)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "issue_4904_latency_decel_percentiles.csv"
    write_percentile_csv(profiles, csv_path)
    print(f"Wrote percentile CSV: {csv_path}")

    lat_plot = OUTPUT_DIR / "issue_4904_response_latency_histogram.png"
    plot_latency_histogram(profiles, lat_plot)
    print(f"Wrote latency histogram: {lat_plot}")

    dec_plot = OUTPUT_DIR / "issue_4904_decel_demand_histogram.png"
    plot_decel_histogram(profiles, dec_plot)
    print(f"Wrote decel histogram: {dec_plot}")

    print_summary(profiles)
    print("\nDone.")


if __name__ == "__main__":
    main()
