#!/usr/bin/env python3
"""Build the mandatory planner safety–efficiency tradeoff figure for the AMV paper.

## Purpose
Produces a scatter plot of collision rate (safety, x-axis) versus success rate
(efficiency, y-axis) for all planners in the canonical publication bundle. This
is the planner tradeoff figure required in the Results section per:

    context/MDPI/FutureTransportation/mandatory-figures-tables.md
    (item 5: "Planner tradeoff figure (Results)", status: pending → done)

The figure ties directly to the same frozen campaign bundle used for the
headline tables in ``paper/sections/results.tex`` so that numbers and
visual are always in sync.

## Reproducibility contract
- One canonical bundle path drives all inputs (``--bundle-path``).
- Bootstrap CIs use the same policy as the manuscript text:
  per-seed means → 400 resamples → 2.5th / 97.5th percentile.
- The script prints the bundle run-id and per-planner point values so
  the provenance is auditable from the terminal output alone.
- Headline planners (orca, ppo) get error bars and colored markers.
  The diagnostic control (goal) uses a hollow marker.
  Experimental planners use gray triangles without error bars.

## Usage
    # Standard run (uses default bundle path inside this repo)
    .venv-paper/bin/python tools/python/scripts/build_planner_tradeoff_figure.py

    # Explicit bundle path (e.g. after a new campaign run)
    .venv-paper/bin/python tools/python/scripts/build_planner_tradeoff_figure.py \\
        --bundle-path /path/to/some_other_publication_bundle

    # Dry-run (print values, skip writing files)
    .venv-paper/bin/python tools/python/scripts/build_planner_tradeoff_figure.py --dry-run

## Agent instructions
To regenerate the figure after a new campaign:
1.  Copy the new publication bundle into ``artifacts/robot_sf_ll7/``.
2.  Run the script with ``--bundle-path artifacts/robot_sf_ll7/<new_bundle>``.
3.  The output PNG/PDF lands in
    ``artifacts/robot_sf_ll7/paper_tools/tradeoff_figure/``.
4.  The LaTeX figure block in ``paper/sections/results.tex`` references a
    fixed asset path (``artifacts/robot_sf_ll7/paper_tools/tradeoff_figure/
    planner_tradeoff_safety_efficiency.png``), so the manuscript picks up
    the new figure automatically on the next ``pdflatex`` run.
5.  Update ``context/MDPI/FutureTransportation/mandatory-figures-tables.md``
    if the bundle run-id changes.

## Output
    artifacts/robot_sf_ll7/paper_tools/tradeoff_figure/
        planner_tradeoff_safety_efficiency.png   (300 dpi, 5.2 × 3.6 in)
        planner_tradeoff_safety_efficiency.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

# Default canonical bundle — update this constant when the campaign is re-run
# and the bundle is copied to artifacts/.
_DEFAULT_BUNDLE_NAME = (
    "paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle"
)
DEFAULT_BUNDLE_PATH = REPO_ROOT / "artifacts" / "robot_sf_ll7" / _DEFAULT_BUNDLE_NAME

DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "robot_sf_ll7" / "paper_tools" / "tradeoff_figure"

# ---------------------------------------------------------------------------
# Planner display metadata
# ---------------------------------------------------------------------------

# Maps planner_key → (display label, role)
# role: "headline" | "control" | "experimental"
PLANNER_META: dict[str, tuple[str, str]] = {
    "goal": ("goal", "control"),
    "orca": ("orca", "headline"),
    "ppo": ("ppo", "headline"),
    "social_force": ("social_force", "experimental"),
    "prediction_planner": ("prediction", "experimental"),
    "sacadrl": ("sacadrl", "experimental"),
    "socnav_sampling": ("socnav_samp.", "experimental"),
}

# Palette for headline planners and control (experimental rows share one gray)
ROLE_COLOR: dict[str, str] = {
    "headline_orca": "#2166ac",  # blue
    "headline_ppo": "#d6604d",  # red-orange
    "control": "#888888",  # mid gray
    "experimental": "#626262",  # dark gray
}

# Bootstrap parameters — must match manuscript text (results.tex)
N_BOOTSTRAP_RESAMPLES = 400
CI_LEVEL = 0.95

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------


def apply_style() -> None:
    """Apply publication-ready matplotlib/seaborn style matching other figures."""
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "figure.figsize": (5.2, 3.6),
            "axes.grid": True,
            "grid.alpha": 0.2,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.titlesize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_campaign_table(bundle_path: Path) -> pd.DataFrame:
    """Load aggregate campaign metrics from the canonical CSV."""
    csv_path = bundle_path / "payload" / "reports" / "campaign_table.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"campaign_table.csv not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    # Normalise column names: the April-14 bundle uses 'collisions_mean';
    # older bundles used 'collision_mean'. Accept both.
    if "collisions_mean" in df.columns and "collision_mean" not in df.columns:
        df = df.rename(columns={"collisions_mean": "collision_mean"})
    return df


def _extract_episode_value(ep: dict, metric: str) -> float | None:
    """Extract a scalar metric value from one episode record.

    Metric resolution order (handles the April-14 bundle layout):
    - ``"success"``    → ``bool(ep["metrics"]["success"])``
    - ``"collisions"`` → ``float(ep["outcome"]["collision_event"])``
      Note: ``metrics.collisions`` (the raw count field) is always 0 in this
      bundle; the actual collision indicator is ``outcome.collision_event``.
    - Any other key    → ``ep["metrics"][metric]`` coerced to float.
    """
    if metric == "success":
        val = ep["metrics"].get("success")
        return float(val) if val is not None else None
    if metric == "collisions":
        # The per-episode collision *rate* in the campaign table is derived from
        # outcome.collision_event, not from metrics.collisions (which is 0 for
        # all episodes in this bundle — the count field tracks something else).
        val = ep.get("outcome", {}).get("collision_event")
        return float(val) if val is not None else None
    val = ep["metrics"].get(metric)
    return float(val) if val is not None else None


def bootstrap_ci_for_planner(
    episodes_jsonl: Path,
    metric: str,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    ci_level: float = CI_LEVEL,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Compute bootstrap CI for *metric* from episode-level data.

    Strategy (matches manuscript text):
    1.  Load all episodes for one planner.
    2.  Group episodes by seed; compute per-seed mean of *metric*.
    3.  Resample those seed-level means ``n_resamples`` times.
    4.  Return (ci_low, ci_high) as percentile bounds.

    Parameters
    ----------
    episodes_jsonl:
        Path to the planner's ``episodes.jsonl`` file.
    metric:
        Logical metric name. Supported: ``"success"`` (from metrics field),
        ``"collisions"`` (from outcome.collision_event — see note in
        ``_extract_episode_value``), or any key in ``episode["metrics"]``.
    n_resamples:
        Number of bootstrap draws (default 400, matching manuscript).
    ci_level:
        Confidence level (default 0.95 → 2.5/97.5 percentiles).
    rng:
        Optional numpy Generator for reproducibility.

    Returns
    -------
    (ci_low, ci_high)  floats
    """
    if rng is None:
        rng = np.random.default_rng(42)

    records: list[dict] = []
    with episodes_jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No episodes found in {episodes_jsonl}")

    # Build (seed, metric_value) pairs
    rows: list[tuple[int, float]] = []
    for ep in records:
        value = _extract_episode_value(ep, metric)
        if value is None:
            continue
        rows.append((int(ep["seed"]), value))

    if not rows:
        raise ValueError(
            f"Metric '{metric}' could not be extracted from episodes at {episodes_jsonl}"
        )

    df = pd.DataFrame(rows, columns=["seed", "value"])
    seed_means: np.ndarray = df.groupby("seed")["value"].mean().to_numpy()

    n_seeds = len(seed_means)
    # Resample seed-level means with replacement
    boot_means = np.array(
        [rng.choice(seed_means, size=n_seeds, replace=True).mean() for _ in range(n_resamples)]
    )

    alpha = (1.0 - ci_level) / 2.0
    ci_low = float(np.percentile(boot_means, 100 * alpha))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return ci_low, ci_high


def load_all_metrics(
    bundle_path: Path,
) -> pd.DataFrame:
    """Return a DataFrame with per-planner means and 95% bootstrap CIs.

    Columns returned:
        planner_key, success_mean, collision_mean,
        success_ci_low, success_ci_high,
        collision_ci_low, collision_ci_high
    """
    campaign = load_campaign_table(bundle_path)
    # Keep only the columns we need
    keep = ["planner_key", "success_mean", "collision_mean"]
    missing = [c for c in keep if c not in campaign.columns]
    if missing:
        raise ValueError(f"Missing columns in campaign table: {missing}")
    df = campaign[keep].copy()

    runs_dir = bundle_path / "payload" / "runs"
    rng = np.random.default_rng(42)

    ci_rows: list[dict] = []
    for _, row in df.iterrows():
        pkey = row["planner_key"]
        # Run directories are named "{planner_key}__differential_drive"
        # but fall back to an exact match as well.
        candidates = list(runs_dir.glob(f"{pkey}__*")) + [runs_dir / pkey]
        episodes_path: Path | None = None
        for cand in candidates:
            ep_file = cand / "episodes.jsonl"
            if ep_file.exists():
                episodes_path = ep_file
                break

        if episodes_path is None:
            print(
                f"  [warn] No episodes.jsonl found for planner '{pkey}'; CIs will be NaN.",
                file=sys.stderr,
            )
            ci_rows.append(
                {
                    "planner_key": pkey,
                    "success_ci_low": float("nan"),
                    "success_ci_high": float("nan"),
                    "collision_ci_low": float("nan"),
                    "collision_ci_high": float("nan"),
                }
            )
            continue

        s_lo, s_hi = bootstrap_ci_for_planner(episodes_path, "success", rng=rng)
        c_lo, c_hi = bootstrap_ci_for_planner(episodes_path, "collisions", rng=rng)
        ci_rows.append(
            {
                "planner_key": pkey,
                "success_ci_low": s_lo,
                "success_ci_high": s_hi,
                "collision_ci_low": c_lo,
                "collision_ci_high": c_hi,
            }
        )

    ci_df = pd.DataFrame(ci_rows)
    result = df.merge(ci_df, on="planner_key", how="left")
    return result


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def build_figure(
    metrics: pd.DataFrame,
    run_id: str,
) -> plt.Figure:
    """Build and return the tradeoff scatter figure.

    The figure shows Success Rate (y) vs Collision Rate (x) for all planners.
    Headline planners carry 95% bootstrap CI error bars.
    """
    apply_style()
    fig, ax = plt.subplots()

    texts: list = []

    for _, row in metrics.iterrows():
        pkey = str(row["planner_key"])
        label, role = PLANNER_META.get(pkey, (pkey, "experimental"))

        x = float(row["collision_mean"])
        y = float(row["success_mean"])

        if role == "headline":
            color = ROLE_COLOR[f"headline_{pkey}"]
            # Error bars: half-widths for asymmetric CI
            x_lo = x - float(row["collision_ci_low"])
            x_hi = float(row["collision_ci_high"]) - x
            y_lo = y - float(row["success_ci_low"])
            y_hi = float(row["success_ci_high"]) - y
            ax.errorbar(
                x,
                y,
                xerr=[[x_lo], [x_hi]],
                yerr=[[y_lo], [y_hi]],
                fmt="o",
                color=color,
                markersize=7,
                capsize=3,
                linewidth=1.2,
                zorder=4,
            )
        elif role == "control":
            color = ROLE_COLOR["control"]
            ax.plot(
                x,
                y,
                "o",
                color="white",
                markeredgecolor=color,
                markeredgewidth=1.4,
                markersize=7,
                zorder=4,
            )
        else:
            # experimental
            color = ROLE_COLOR["experimental"]
            ax.plot(
                x,
                y,
                "^",
                color=color,
                markersize=6,
                alpha=0.9,
                zorder=3,
            )

        label_x = x
        label_y = max(y, 0.012) if role == "experimental" else y
        t = ax.text(
            label_x,
            label_y,
            f"  {label}",
            fontsize=8.0 if role == "experimental" else 7.5,
            color=color if role != "control" else ROLE_COLOR["control"],
            alpha=0.95 if role == "experimental" else 0.9,
            zorder=5,
        )
        texts.append(t)

    # Auto-adjust text labels to avoid overlap
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.7),
        expand=(1.3, 1.5),
        force_text=(0.3, 0.4),
    )

    # Axis labels and formatting
    ax.set_xlabel("Collision rate (lower is safer)")
    ax.set_ylabel("Success rate (higher is better)")
    ax.set_xlim(left=-0.02)
    ax.set_ylim(bottom=-0.02)

    # Preferred direction (upper-left = both safer and more successful) is stated
    # in the figure caption; keep it out of the rendered image to avoid duplicating
    # the caption inside the data area.

    # Legend: only headline + control entries; experimental as a group
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ROLE_COLOR["headline_orca"],
            markersize=6,
            label="orca  (main)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ROLE_COLOR["headline_ppo"],
            markersize=6,
            label="ppo  (main)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor=ROLE_COLOR["control"],
            markeredgewidth=1.2,
            markersize=6,
            label="goal  (diagnostic)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=ROLE_COLOR["experimental"],
            markersize=6,
            label="experimental planners",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    sns.despine(ax=ax)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the mandatory planner safety–efficiency tradeoff figure "
            "for the AMV benchmark paper."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=DEFAULT_BUNDLE_PATH,
        help=(
            f"Path to the canonical publication bundle directory. Default: {DEFAULT_BUNDLE_PATH}"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for PNG/PDF. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print computed values and exit without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_path: Path = args.bundle_path.resolve()

    if not bundle_path.exists():
        print(f"ERROR: bundle path does not exist: {bundle_path}", file=sys.stderr)
        print(
            "Copy the publication bundle to artifacts/robot_sf_ll7/ first, "
            "or pass --bundle-path <path>.",
            file=sys.stderr,
        )
        return 1

    # --- Load run-id from manifest ---
    manifest_path = bundle_path / "publication_manifest.json"
    run_id = bundle_path.name  # fallback
    if manifest_path.exists():
        with manifest_path.open() as fh:
            manifest = json.load(fh)
        run_id = manifest.get("provenance", {}).get("run_id", run_id)

    print(f"Bundle: {bundle_path.name}")
    print(f"Run-id: {run_id}")
    print()

    # --- Load metrics ---
    print("Computing bootstrap CIs from episode-level data ...")
    metrics = load_all_metrics(bundle_path)

    # --- Print provenance table ---
    print("\nPer-planner values used for figure:")
    header = f"{'planner':<22} {'success':>8} {'coll':>8} {'succ_CI':>18} {'coll_CI':>18}"
    print(header)
    print("-" * len(header))
    for _, row in metrics.iterrows():
        pkey = row["planner_key"]
        s = row["success_mean"]
        c = row["collision_mean"]
        s_ci = f"[{row['success_ci_low']:.3f}, {row['success_ci_high']:.3f}]"
        c_ci = f"[{row['collision_ci_low']:.3f}, {row['collision_ci_high']:.3f}]"
        print(f"{pkey:<22} {s:>8.4f} {c:>8.4f} {s_ci:>18} {c_ci:>18}")

    if args.dry_run:
        print("\n[dry-run] Skipping figure output.")
        return 0

    # --- Build figure ---
    fig = build_figure(metrics, run_id)

    # --- Save ---
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "planner_tradeoff_safety_efficiency"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"\nWrote: {png_path}")
    print(f"Wrote: {pdf_path}")
    print()
    print("Next steps:")
    print("  1. Check the figure visually.")
    print("  2. Run 'pdflatex' (or the full build) to verify LaTeX integration.")
    print("  3. Update mandatory-figures-tables.md if the run-id changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
