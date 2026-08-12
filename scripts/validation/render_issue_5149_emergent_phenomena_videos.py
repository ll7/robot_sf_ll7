#!/usr/bin/env python3
"""Render per-phenomenon replay videos for the issue #5149 multi-seed campaign.

Issue #6960 (maintainer-requested 2026-08-12): the delivered multi-seed
emergent-phenomena campaign bundle
(``docs/context/evidence/issue_5149_emergent_phenomena_multiseed_2026-08/``,
PR #6952) archives order-parameter figures but no replay videos, so the
measured phenomena cannot be reviewed visually. This script renders one short
animated GIF per phenomenon x speed calibration (lane formation in the
bidirectional corridor; oscillation at the narrow doorway), plus one cheap
two-row "grid" GIF per phenomenon that shows both calibrations side by side.

Representative-seed rule (deterministic, defensible): for each scenario x
calibration group the script reads the campaign's committed ``runs.jsonl``,
restricts to the seeds whose per-seed verdict equals the group's majority
verdict (weaker-verdict tie-break, as in the campaign aggregation), and picks
the seed with the median primary order parameter (lower seed on ties). The
rendered replay therefore shows a *typical* measured run, not a cherry-picked
best case.

Determinism: the replay trajectories are re-simulated with the same pinned
harness (``robot_sf.research.emergent_phenomena.run_scenario``) and seeds as
the campaign, so the animation replays exactly the dynamics the campaign
measured (same platform/environment caveat as the campaign bundle). GIF
encoding via matplotlib's PillowWriter embeds no timestamps; re-runs on the
same platform/environment are byte-stable given ``--generated-at``.

Claim boundary: visualization of existing measured face-validity (smoke-tier)
evidence; the videos add no new quantitative claims beyond the campaign bundle.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")  # headless rendering; no display required
import matplotlib.pyplot as plt
import numpy as np
import pysocialforce as pysf
from matplotlib.animation import PillowWriter

from robot_sf.evidence.writers import (
    register_evidence,
    write_json,
    write_review_sidecar,
    write_sha256sums,
)
from robot_sf.research.emergent_phenomena import (
    LITERATURE_CALIBRATION,
    RELEASED_DEFAULT_CALIBRATION,
    ScenarioResult,
    default_scenario_set,
    released_default_config,
    run_scenario,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_DIR = Path("docs/context/evidence/issue_5149_emergent_phenomena_multiseed_2026-08")
ISSUE_REF = "robot_sf_ll7#6960"
CAMPAIGN_ISSUE_REF = "robot_sf_ll7#5149"
GENERATION_COMMAND = (
    "uv run python scripts/validation/render_issue_5149_emergent_phenomena_videos.py"
)

# The two phenomena requested by issue #6960. The exit-arching diagnostic is a
# static density pattern already legible in the archived trajectory figures.
VIDEO_SCENARIOS: tuple[str, ...] = ("bidirectional_corridor", "narrow_doorway")
PHENOMENON_BY_SCENARIO = {
    "bidirectional_corridor": "lane_formation",
    "narrow_doorway": "doorway_oscillation",
}
# Mirrors PRIMARY_ORDER_PARAMETER in build_issue_5149_emergent_phenomena_campaign.py
# for the two video scenarios.
PRIMARY_ORDER_PARAMETER = {
    "bidirectional_corridor": "lane_segregation_index",
    "narrow_doorway": "oscillation_flips",
}
CALIBRATIONS = {
    "released_default": RELEASED_DEFAULT_CALIBRATION,
    "literature_typical": LITERATURE_CALIBRATION,
}
# Verdict labels ordered weakest-first (matches emergent_phenomena_campaign).
VERDICT_SEVERITY: tuple[str, ...] = (
    "absent_or_negligible",
    "weak_partial",
    "clearly_present",
)

DEFAULT_FRAME_STRIDE = 2  # render every 2nd step (dt=0.1 s -> 0.2 s per frame)
DEFAULT_FPS = 10  # 2x real time at the default stride
TRAIL_STEPS = 30  # trajectory trail length in simulation steps


def _git_commit() -> str:
    """Return the current commit hash, or ``unknown`` outside git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_run_records(bundle_dir: Path) -> list[dict[str, Any]]:
    """Load the campaign's per-seed run records from ``runs.jsonl``.

    Returns:
        List of run-record dicts.
    """
    runs_path = bundle_dir / "runs.jsonl"
    lines = runs_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def majority_verdict(verdicts: list[str]) -> str:
    """Most common verdict, tie-breaking toward the weaker label.

    Returns:
        The majority verdict label.
    """
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1

    def sort_key(item: tuple[str, int]) -> tuple[int, int]:
        label, count = item
        severity = (
            VERDICT_SEVERITY.index(label) if label in VERDICT_SEVERITY else len(VERDICT_SEVERITY)
        )
        return (-count, severity)

    return min(counts.items(), key=sort_key)[0]


def select_representative_record(
    records: list[dict[str, Any]], scenario: str, calibration: str
) -> dict[str, Any]:
    """Pick the representative run record for one scenario x calibration group.

    Rule: restrict to records whose verdict equals the group's majority
    verdict (weaker-verdict tie-break), then take the record with the median
    primary order parameter; ties break toward the lower seed. Deterministic
    given ``runs.jsonl``.

    Args:
        records: All campaign run records.
        scenario: Scenario name.
        calibration: Calibration name.

    Returns:
        The selected run record.

    Raises:
        ValueError: If the group has no records.
    """
    group = [r for r in records if r["scenario"] == scenario and r["calibration"] == calibration]
    if not group:
        raise ValueError(f"no run records for {scenario} x {calibration}")
    majority = majority_verdict([r["phenomenon_verdict"] for r in group])
    pool = [r for r in group if r["phenomenon_verdict"] == majority]
    param = PRIMARY_ORDER_PARAMETER[scenario]
    ordered = sorted(pool, key=lambda r: (float(r["order_parameters"][param]), int(r["seed"])))
    return ordered[(len(ordered) - 1) // 2]


def _draw_static_geometry(ax: Axes, result: ScenarioResult) -> None:
    """Draw walls, door/exit markers, and axis furniture for one scenario."""
    scenario = result.scenario
    length = scenario.length
    hw = scenario.half_width
    ax.plot([-1.0, length + 1.0], [hw, hw], color="black", lw=1.5)
    ax.plot([-1.0, length + 1.0], [-hw, -hw], color="black", lw=1.5)
    if scenario.name == "narrow_doorway":
        door_x = float(scenario.extra.get("door_x", length / 2.0))
        door_half = float(scenario.extra.get("door_half_width", 0.6))
        ax.plot([door_x, door_x], [door_half, hw], color="black", lw=2.5)
        ax.plot([door_x, door_x], [-door_half, -hw], color="black", lw=2.5)
        ax.axvline(door_x, color="red", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlim(-0.5, length + 0.5)
    ax.set_ylim(-hw - 0.4, hw + 0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.2)


class _ReplayPanel:
    """One animated scenario panel (scatter + fading trails) on a matplotlib axis."""

    def __init__(self, ax: Axes, result: ScenarioResult, label: str) -> None:
        self.ax = ax
        self.result = result
        self.label = label
        pos = result.trajectory.positions
        dirs = result.trajectory.desired_directions[:, 0]
        self.plus = dirs > 0
        self.minus = ~self.plus
        _draw_static_geometry(ax, result)
        self.trail_plus: list = []
        self.trail_minus: list = []
        for mask, store, color in (
            (self.plus, self.trail_plus, "tab:blue"),
            (self.minus, self.trail_minus, "tab:orange"),
        ):
            for i in np.where(mask)[0]:
                (line,) = ax.plot(pos[:1, i, 0], pos[:1, i, 1], color=color, lw=0.7, alpha=0.35)
                store.append((i, line))
        self.scat_plus = ax.scatter(
            pos[0, self.plus, 0], pos[0, self.plus, 1], s=26, color="tab:blue", zorder=3
        )
        self.scat_minus = ax.scatter(
            pos[0, self.minus, 0], pos[0, self.minus, 1], s=26, color="tab:orange", zorder=3
        )

    def update(self, t: int) -> None:
        """Advance the panel to simulation step ``t``."""
        pos = self.result.trajectory.positions
        t = min(t, pos.shape[0] - 1)
        start = max(0, t - TRAIL_STEPS)
        for i, line in self.trail_plus:
            line.set_data(pos[start : t + 1, i, 0], pos[start : t + 1, i, 1])
        for i, line in self.trail_minus:
            line.set_data(pos[start : t + 1, i, 0], pos[start : t + 1, i, 1])
        self.scat_plus.set_offsets(pos[t, self.plus, :])
        self.scat_minus.set_offsets(pos[t, self.minus, :])
        sim_t = t * self.result.trajectory.dt
        self.ax.set_title(f"{self.label} | t = {sim_t:5.1f} s", fontsize=9)


def _panel_label(result: ScenarioResult, record: dict[str, Any]) -> str:
    """Build the per-panel provenance label from a run record."""
    param = PRIMARY_ORDER_PARAMETER[result.scenario.name]
    value = float(record["order_parameters"][param])
    return (
        f"{result.scenario.name} | {result.calibration.name} | seed {result.scenario.seed} | "
        f"{param}={value:.3f} ({record['phenomenon_verdict']})"
    )


def _figsize_for(scenario_name: str, n_rows: int) -> tuple[float, float]:
    """Return a figure size proportional to the scenario geometry."""
    if scenario_name == "bidirectional_corridor":
        return (9.0, 2.6 * n_rows)
    return (7.5, 2.6 * n_rows)


def render_replay_gif(
    results_with_records: list[tuple[ScenarioResult, dict[str, Any]]],
    out_path: Path,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    fps: int = DEFAULT_FPS,
    dpi: int = 100,
) -> int:
    """Render one animated GIF with one row per (result, record) pair.

    Args:
        results_with_records: One or more replay rows (single video: one row;
            grid video: one row per calibration).
        out_path: Output GIF path.
        frame_stride: Render every ``frame_stride``-th simulation step.
        fps: GIF frame rate.
        dpi: Render DPI.

    Returns:
        Number of rendered frames.
    """
    n_rows = len(results_with_records)
    scenario_name = results_with_records[0][0].scenario.name
    fig, axes = plt.subplots(n_rows, 1, figsize=_figsize_for(scenario_name, n_rows), squeeze=False)
    panels = [
        _ReplayPanel(axes[row][0], result, _panel_label(result, record))
        for row, (result, record) in enumerate(results_with_records)
    ]
    fig.tight_layout()
    n_steps = max(r.trajectory.positions.shape[0] for r, _ in results_with_records)
    frames = list(range(0, n_steps, frame_stride))
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(out_path), dpi=dpi):
        for t in frames:
            for panel in panels:
                panel.update(t)
            writer.grab_frame()
    plt.close(fig)
    return len(frames)


def build_videos(
    bundle_dir: Path,
    generated_at_override: str | None = None,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    fps: int = DEFAULT_FPS,
) -> Path:
    """Render all replay videos into the campaign bundle and refresh integrity.

    Args:
        bundle_dir: The committed campaign bundle directory (must contain
            ``runs.jsonl`` and ``manifest.json``).
        generated_at_override: Optional pinned ISO-8601 UTC timestamp for
            byte-stable manifest re-runs.
        frame_stride: Render every ``frame_stride``-th simulation step.
        fps: GIF frame rate.

    Returns:
        The bundle directory.
    """
    records = load_run_records(bundle_dir)
    campaign_manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))

    try:
        register_evidence(bundle_dir, area="benchmark_evidence")
    except (FileNotFoundError, ValueError):
        pass  # outside the repo evidence tree (e.g. tests writing to tmp)

    commit = _git_commit()
    short = commit[:10] if commit != "unknown" else "unknown"
    if generated_at_override:
        generated_at = generated_at_override
    else:
        generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    scenarios = {s.name: s for s in default_scenario_set()}
    sim_config = released_default_config()

    video_rows: list[dict[str, Any]] = []
    video_names: list[str] = []
    for scenario_name in VIDEO_SCENARIOS:
        per_calibration: list[tuple[ScenarioResult, dict[str, Any]]] = []
        for cal_name, calibration in CALIBRATIONS.items():
            record = select_representative_record(records, scenario_name, cal_name)
            seed = int(record["seed"])
            scenario_cfg = replace(scenarios[scenario_name], seed=seed)
            result = run_scenario(scenario_cfg, calibration, sim_config=sim_config)
            per_calibration.append((result, record))
            name = f"{scenario_name}__{cal_name}__seed{seed}__git{short}.gif"
            n_frames = render_replay_gif(
                [(result, record)], bundle_dir / name, frame_stride=frame_stride, fps=fps
            )
            video_names.append(name)
            video_rows.append(
                {
                    "file": name,
                    "kind": "single",
                    "phenomenon": PHENOMENON_BY_SCENARIO[scenario_name],
                    "scenario": scenario_name,
                    "calibration": cal_name,
                    "seed": seed,
                    "seed_selection": "median primary order parameter among majority-verdict "
                    "seeds (lower seed on ties)",
                    "phenomenon_verdict": record["phenomenon_verdict"],
                    "primary_order_parameter": PRIMARY_ORDER_PARAMETER[scenario_name],
                    "primary_order_parameter_value": float(
                        record["order_parameters"][PRIMARY_ORDER_PARAMETER[scenario_name]]
                    ),
                    "source_commit": commit,
                    "n_frames": n_frames,
                    "fps": fps,
                    "frame_stride": frame_stride,
                }
            )
        # Cheap grid variant: both calibrations stacked, trajectories reused.
        seeds_part = "__".join(
            f"{cal}_seed{r['seed']}"
            for (_, r), cal in zip(per_calibration, CALIBRATIONS, strict=True)
        )
        grid_name = f"{scenario_name}__grid__{seeds_part}__git{short}.gif"
        n_frames = render_replay_gif(
            per_calibration, bundle_dir / grid_name, frame_stride=frame_stride, fps=fps
        )
        video_names.append(grid_name)
        video_rows.append(
            {
                "file": grid_name,
                "kind": "grid",
                "phenomenon": PHENOMENON_BY_SCENARIO[scenario_name],
                "scenario": scenario_name,
                "calibrations": {
                    cal: int(r["seed"])
                    for (_, r), cal in zip(per_calibration, CALIBRATIONS, strict=True)
                },
                "source_commit": commit,
                "n_frames": n_frames,
                "fps": fps,
                "frame_stride": frame_stride,
            }
        )

    manifest = {
        "schema": "issue_5149_emergent_phenomena_videos_manifest.v1",
        "issue": ISSUE_REF,
        "campaign_issue": CAMPAIGN_ISSUE_REF,
        "generated_at_utc": generated_at,
        "git_head": commit,
        "generation_command": GENERATION_COMMAND,
        "campaign_bundle_git_head": campaign_manifest.get("git_head", "unknown"),
        "campaign_runs_jsonl": "runs.jsonl",
        "harness_modules": [
            "robot_sf/research/emergent_phenomena.py",
        ],
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "packages": {
            "pysocialforce": getattr(pysf, "__version__", "unknown"),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "claim_boundary": "replay visualization of the measured multi-seed campaign "
        "(smoke-tier face-validity evidence); adds no quantitative claims beyond the "
        "campaign bundle",
        "determinism_note": "Trajectories re-simulated with the pinned harness and seeds; "
        "GIF encoding embeds no timestamps. Byte-stable re-runs require --generated-at "
        "plus the same platform/environment.",
        "videos": video_rows,
    }
    write_json(bundle_dir / "videos_manifest.json", manifest)

    for name in video_names:
        write_review_sidecar(bundle_dir / name)

    # Integrity manifest last so it covers every file in the bundle.
    write_sha256sums(bundle_dir)
    return bundle_dir


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_BUNDLE_DIR,
        help=f"Campaign bundle directory (default: {DEFAULT_BUNDLE_DIR})",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=DEFAULT_FRAME_STRIDE,
        help=f"Render every Nth simulation step (default: {DEFAULT_FRAME_STRIDE})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"GIF frame rate (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--generated-at",
        type=str,
        default=None,
        help="Optional pinned ISO-8601 UTC timestamp for byte-stable manifest re-runs "
        "(default: current wall-clock time).",
    )
    args = parser.parse_args(argv)
    out = build_videos(
        args.bundle_dir,
        generated_at_override=args.generated_at,
        frame_stride=args.frame_stride,
        fps=args.fps,
    )
    manifest = json.loads((out / "videos_manifest.json").read_text(encoding="utf-8"))
    print(f"Wrote {len(manifest['videos'])} replay videos into {out}")
    for row in manifest["videos"]:
        print(f"  {row['file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
