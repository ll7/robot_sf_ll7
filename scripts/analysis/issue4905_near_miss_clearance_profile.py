"""Per-planner near-miss + minimum-clearance geometry profile from retained h600 trace predicates.

Reads episodes.jsonl files from the issue4206 trace-capable h600 rerun and produces:
- Per-planner min-separation percentiles CSV
- Per-planner occlusion-at-minimum share and clearance-vs-near-miss cross-tab CSV
- Two PNGs: min-separation distribution (planners overlaid); min-separation vs near-miss-rate

Pure jsonl parse + numpy/matplotlib. No recompute, no simulation, no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if (REPO_ROOT / ".git").is_file():
    _gitdir_text = (REPO_ROOT / ".git").read_text().strip().replace("gitdir: ", "")
    _gitdir = Path(_gitdir_text)
    _main_checkout = _gitdir.parent.parent.parent
else:
    _main_checkout = REPO_ROOT
RUNS_DIR = (
    _main_checkout
    / "output/benchmarks/camera_ready/issue4206_trace_capable_h600_rerun_20260704/runs"
)
OUTPUT_DIR = Path("output/issue4905_near_miss_clearance_profile")
SENTINEL = 1e9


def derive_scenario_family(scenario_id: str) -> str:
    """Extract scenario family by removing trailing difficulty level."""
    parts = scenario_id.rsplit("_", 1)
    if len(parts) > 1 and parts[-1] in ("low", "medium", "high"):
        return parts[0]
    return scenario_id


def load_episodes(runs_dir: Path) -> list[dict]:
    """Load all episodes from planner directories under runs_dir."""
    episodes = []
    for planner_dir in sorted(runs_dir.iterdir()):
        if not planner_dir.is_dir():
            continue
        episodes_file = planner_dir / "episodes.jsonl"
        if not episodes_file.exists():
            continue
        planner_name = planner_dir.name.replace("__differential_drive", "")
        for line in episodes_file.read_text().splitlines():
            if not line.strip():
                continue
            ep = json.loads(line)
            ep["_planner_dir"] = planner_name
            episodes.append(ep)
    return episodes


def extract_episode_fields(ep: dict) -> dict | None:
    """Extract relevant fields from episode, returning None if exposure<=0."""
    ie = ep.get("interaction_exposure", {})
    exposure_steps = ie.get("interaction_exposure_steps", 0)
    if not exposure_steps or exposure_steps <= 0:
        return None

    sp = ep.get("safety_predicates", {})
    late = sp.get("late_evasive_predicate", {}).get("fields", {})
    occl = sp.get("occlusion_near_miss_predicate", {}).get("fields", {})
    el = ep.get("event_ledger", {})
    se = el.get("surrogate_events", {})

    min_dist = late.get("minimum_distance_m", SENTINEL)
    actual_min_sep = occl.get("actual_minimum_separation_m", SENTINEL)
    min_sep_step = occl.get("min_separation_step", 0)
    was_occluded = occl.get("was_occluded_before_min")
    near_miss = se.get("near_miss", False)
    clearance_breach = se.get("clearance_breach", False)

    algo = ep.get("algo", ep.get("_planner_dir", "unknown"))
    scenario_id = ep.get("scenario_id", "unknown")
    scenario_family = ep.get("scenario_family") or derive_scenario_family(scenario_id)

    return {
        "planner": algo,
        "scenario_id": scenario_id,
        "scenario_family": scenario_family,
        "minimum_distance_m": min_dist,
        "actual_minimum_separation_m": actual_min_sep,
        "min_separation_step": min_sep_step,
        "was_occluded_before_min": was_occluded,
        "near_miss": near_miss,
        "clearance_breach": clearance_breach,
        "exposure_steps": exposure_steps,
    }


def build_dataframe(episodes: list[dict]) -> pd.DataFrame:
    """Build DataFrame from episodes with exposure>0."""
    rows = []
    for ep in episodes:
        row = extract_episode_fields(ep)
        if row is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def compute_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-planner min-separation percentiles for non-sentinel values."""
    mask = df["actual_minimum_separation_m"] < SENTINEL
    valid = df[mask]
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    result = []
    for planner, group in valid.groupby("planner"):
        vals = group["actual_minimum_separation_m"]
        row = {"planner": planner, "N": len(vals)}
        for p in percentiles:
            label = "min" if p == 0 else ("max" if p == 100 else f"p{p}")
            row[label] = np.percentile(vals, p)
        result.append(row)
    return pd.DataFrame(result).sort_values("planner")


def compute_occlusion_share(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-planner share of episodes where was_occluded_before_min is true."""
    mask = df["actual_minimum_separation_m"] < SENTINEL
    valid = df[mask]
    result = []
    for planner, group in valid.groupby("planner"):
        n = len(group)
        occluded_true = group["was_occluded_before_min"].fillna(False).astype(bool).sum()
        occluded_none = group["was_occluded_before_min"].isna().sum()
        result.append(
            {
                "planner": planner,
                "N": n,
                "occluded_true": int(occluded_true),
                "occluded_share": occluded_true / n if n > 0 else 0.0,
                "occluded_none": int(occluded_none),
            }
        )
    return pd.DataFrame(result).sort_values("planner")


def compute_clearance_nearmiss_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-tab of min-separation bins vs near-miss/clearance-breach rates."""
    mask = df["actual_minimum_separation_m"] < SENTINEL
    valid = df[mask].copy()
    valid["sep_bin"] = pd.cut(
        valid["actual_minimum_separation_m"],
        bins=[0, 0.3, 0.5, 1.0, 1.5, 2.0, float("inf")],
        labels=["<0.3m", "0.3-0.5m", "0.5-1.0m", "1.0-1.5m", "1.5-2.0m", ">2.0m"],
        right=False,
    )
    ctab = (
        valid.groupby(["planner", "scenario_family", "sep_bin"], observed=False)
        .agg(
            n_episodes=("near_miss", "count"),
            near_miss_rate=("near_miss", "mean"),
            clearance_breach_rate=("clearance_breach", "mean"),
        )
        .reset_index()
    )
    return ctab


def plot_min_separation_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Plot overlaid histograms of min-separation per planner."""
    mask = df["actual_minimum_separation_m"] < SENTINEL
    valid = df[mask]
    fig, ax = plt.subplots(figsize=(12, 6))
    planners = sorted(valid["planner"].unique())
    for planner in planners:
        subset = valid[valid["planner"] == planner]["actual_minimum_separation_m"]
        ax.hist(subset, bins=50, alpha=0.5, label=planner, density=True)
    ax.set_xlabel("Actual Minimum Separation (m)")
    ax.set_ylabel("Density")
    ax.set_title("Per-Planner Minimum Separation Distribution (exposure>0)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_separation_vs_nearmiss(df: pd.DataFrame, output_path: Path) -> None:
    """Plot min-separation vs near-miss rate by planner."""
    mask = df["actual_minimum_separation_m"] < SENTINEL
    valid = df[mask].copy()
    valid["sep_bin"] = pd.cut(
        valid["actual_minimum_separation_m"],
        bins=np.arange(0, 4.1, 0.2),
        right=False,
    )
    summary = (
        valid.groupby(["planner", "sep_bin"], observed=False)
        .agg(near_miss_rate=("near_miss", "mean"), n=("near_miss", "count"))
        .reset_index()
    )
    summary["sep_mid"] = summary["sep_bin"].apply(lambda x: x.mid if hasattr(x, "mid") else np.nan)
    fig, ax = plt.subplots(figsize=(12, 6))
    planners = sorted(summary["planner"].unique())
    for planner in planners:
        subset = summary[summary["planner"] == planner].dropna(subset=["sep_mid"])
        subset = subset[subset["n"] >= 3]
        ax.plot(subset["sep_mid"], subset["near_miss_rate"], "o-", label=planner, markersize=4)
    ax.set_xlabel("Actual Minimum Separation bin midpoint (m)")
    ax.set_ylabel("Near-Miss Rate")
    ax.set_title("Minimum Separation vs Near-Miss Rate by Planner (exposure>0)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the analysis pipeline: load, extract, compute, save, report."""
    print(f"Loading episodes from {RUNS_DIR}")
    episodes = load_episodes(RUNS_DIR)
    print(f"Loaded {len(episodes)} total episodes")

    df = build_dataframe(episodes)
    print(f"Episodes with exposure>0: {len(df)}")

    if df.empty:
        print("ERROR: No episodes with exposure>0 found. Exiting.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pct_df = compute_percentiles(df)
    pct_path = OUTPUT_DIR / "min_separation_percentiles.csv"
    pct_df.to_csv(pct_path, index=False)
    print(f"Saved percentiles CSV: {pct_path}")

    occl_df = compute_occlusion_share(df)
    occl_path = OUTPUT_DIR / "occlusion_at_minimum_share.csv"
    occl_df.to_csv(occl_path, index=False)
    print(f"Saved occlusion share CSV: {occl_path}")

    ctab_df = compute_clearance_nearmiss_crosstab(df)
    ctab_path = OUTPUT_DIR / "clearance_nearmiss_crosstab.csv"
    ctab_df.to_csv(ctab_path, index=False)
    print(f"Saved cross-tab CSV: {ctab_path}")

    hist_path = OUTPUT_DIR / "min_separation_distribution.png"
    plot_min_separation_distribution(df, hist_path)
    print(f"Saved histogram: {hist_path}")

    scatter_path = OUTPUT_DIR / "separation_vs_nearmiss_rate.png"
    plot_separation_vs_nearmiss(df, scatter_path)
    print(f"Saved scatter: {scatter_path}")

    print("\n=== SUMMARY ===")
    print("\nPer-planner min-separation percentiles (exposure>0, non-sentinel):")
    print(pct_df.to_string(index=False))

    print("\nPer-planner occlusion-at-minimum share:")
    print(occl_df.to_string(index=False))
    if occl_df["occluded_none"].sum() == occl_df["N"].sum():
        print(
            "NOTE: was_occluded_before_min is None for ALL exposure>0 episodes "
            "(visibility observation disabled). Occlusion share is reported as 0.0."
        )

    print("\n=== COLLISION-ADJACENT PLANNERS ===")
    merged = pct_df.merge(occl_df[["planner", "occluded_share"]], on="planner")
    merged["near_miss_episodes"] = 0
    for planner in merged["planner"]:
        mask = (df["planner"] == planner) & (df["actual_minimum_separation_m"] < SENTINEL)
        merged.loc[merged["planner"] == planner, "near_miss_episodes"] = int(
            df[mask]["near_miss"].sum()
        )
    merged = merged.sort_values("p50")
    print(
        "Planners ranked by median min-separation (lower = tighter clearance, "
        "more collision-adjacent):"
    )
    print(
        merged[
            ["planner", "N", "min", "p10", "p25", "p50", "p75", "p90", "near_miss_episodes"]
        ].to_string(index=False)
    )

    print("\nAll outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
