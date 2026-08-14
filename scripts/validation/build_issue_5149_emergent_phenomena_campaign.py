#!/usr/bin/env python3
"""Build the multi-seed emergent-phenomena campaign bundle for issue #5149.

Maintainer authorization 2026-08-11
(https://github.com/ll7/robot_sf_ll7/issues/5149#issuecomment-5264374182):
elevate the pinned single-seed exhibit
(``docs/context/evidence/issue_5149_emergent_phenomena_2026-07/``) to
*measured* face-validity evidence. This script runs the same three canonical
crowd-dynamics scenarios (bidirectional corridor, narrow doorway, high-density
exit) at the released-default and literature-typical speed calibrations across
a pinned seed list (default: 5149..5158), and archives:

- ``runs.jsonl`` -- one episode-record-style row per scenario x calibration x
  seed (order parameters + verdict),
- ``summary.json`` -- per-scenario/per-calibration aggregate statistics and
  verdict distributions,
- ``manifest.json`` -- full provenance manifest (command, git head, runtime,
  package versions, seeds, scenario/calibration/simulator parameters),
- one trajectory plot (PNG) per scenario x calibration at the representative
  first seed, and one order-parameter-by-seed figure per scenario,
- a human-readable ``README.md`` with the honest interpretation, and
- a ``SHA256SUMS`` integrity manifest.

Claim boundary: **measured face-validity (smoke-tier) evidence** for THIS
implementation at the pinned parameterizations. Multi-seed measurement, not
benchmark-matrix evidence and not paper-grade validation against real human
trajectory data (that is issue #4975).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless rendering; no display required
import matplotlib.pyplot as plt
import numpy as np
import pysocialforce as pysf

from robot_sf.evidence.writers import (
    register_evidence,
    write_json,
    write_review_sidecar,
    write_sha256sums,
    write_text,
)
from robot_sf.research.emergent_phenomena import (
    ARCH_DENSITY_RATIO_CLEAR,
    LANE_SEGREGATION_CLEAR,
    LANE_SEGREGATION_WEAK,
    LITERATURE_CALIBRATION,
    OSCILLATION_FLIPS_CLEAR,
    RELEASED_DEFAULT_CALIBRATION,
    ScenarioResult,
    default_scenario_set,
    released_default_config,
    simulator_config_snapshot,
)
from robot_sf.research.emergent_phenomena_campaign import (
    DEFAULT_CAMPAIGN_SEEDS,
    aggregate_run_records,
    result_to_run_record,
    run_multiseed_campaign,
)
from robot_sf.research.representative_selection import primary_order_parameter

# evidence-writer-exempt: runs.jsonl is an intentionally immutable JSONL artifact; the shared
# write_json helper cannot preserve one sorted JSON document per line, so the artifact carries
# its required marker through a generated review sidecar.

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_5149_emergent_phenomena_multiseed_2026-08")
ISSUE_REF = "robot_sf_ll7#5149"
AUTHORIZATION_COMMENT = "https://github.com/ll7/robot_sf_ll7/issues/5149#issuecomment-5264374182"
GENERATION_COMMAND = (
    "uv run python scripts/validation/build_issue_5149_emergent_phenomena_campaign.py"
)

# Threshold reference lines drawn on the by-seed figures.
THRESHOLD_LINES = {
    "bidirectional_corridor": [
        (LANE_SEGREGATION_CLEAR, "clearly_present"),
        (LANE_SEGREGATION_WEAK, "weak_partial"),
    ],
    "narrow_doorway": [(float(OSCILLATION_FLIPS_CLEAR), "clearly_present")],
    "high_density_exit": [(ARCH_DENSITY_RATIO_CLEAR, "clearly_present")],
}


def _git_commit() -> str:
    """Return the current commit hash, or ``unknown`` outside git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _trajectory_plot(result: ScenarioResult, out_path: Path) -> None:
    """Render a trajectory plot for one scenario x calibration x seed run."""
    pos = result.trajectory.positions  # (T, N, 2)
    dirs = result.trajectory.desired_directions[:, 0]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plus = dirs > 0
    minus = ~plus
    for i in np.where(plus)[0]:
        ax.plot(pos[:, i, 0], pos[:, i, 1], color="tab:blue", lw=0.5, alpha=0.55)
    for i in np.where(minus)[0]:
        ax.plot(pos[:, i, 0], pos[:, i, 1], color="tab:orange", lw=0.5, alpha=0.55)
    ax.set_title(
        f"{result.scenario.name} | {result.calibration.name} | seed {result.scenario.seed} "
        f"(v_des~{result.max_speeds.mean():.2f} m/s)"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    if result.scenario.name == "narrow_doorway":
        door_x = float(result.scenario.extra.get("door_x", result.scenario.length / 2.0))
        ax.axvline(door_x, color="red", lw=1.0, ls="--", alpha=0.5, label="door")
    elif result.scenario.name == "high_density_exit":
        ax.axvline(result.scenario.length, color="red", lw=1.0, ls="--", alpha=0.5, label="exit")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _by_seed_plot(
    scenario_name: str, records: list[dict[str, Any]], seeds: list[int], out_path: Path
) -> None:
    """Render the primary order parameter per seed for both calibrations."""
    param = primary_order_parameter(scenario_name)
    calibrations = ["released_default", "literature_typical"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_seeds = max(1, len(seeds))
    for c_idx, cal in enumerate(calibrations):
        recs = sorted(
            (r for r in records if r["scenario"] == scenario_name and r["calibration"] == cal),
            key=lambda r: r["seed"],
        )
        values = [r["order_parameters"][param] for r in recs]
        # Deterministic per-seed jitter around the calibration position.
        xs = [c_idx + (s_idx - (n_seeds - 1) / 2.0) * 0.03 for s_idx in range(len(recs))]
        ax.scatter(xs, values, s=28, alpha=0.8, label=f"{cal} (per seed)")
        if values:
            mean = float(np.mean(values))
            ax.hlines(mean, c_idx - 0.22, c_idx + 0.22, color="black", lw=1.4)
            ax.annotate(
                f"mean={mean:.3f}",
                (c_idx + 0.24, mean),
                fontsize=8,
                va="center",
            )
    for threshold, label in THRESHOLD_LINES[scenario_name]:
        ax.axhline(threshold, color="red", lw=0.9, ls=":", alpha=0.7)
        ax.annotate(f"{label} >= {threshold:g}", (-0.45, threshold), fontsize=7, color="red")
    ax.set_xticks(range(len(calibrations)))
    ax.set_xticklabels(calibrations)
    ax.set_xlim(-0.5, len(calibrations) - 0.3)
    ax.set_ylabel(param)
    ax.set_title(f"{scenario_name}: {param} across {n_seeds} seeds")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _write_runs_jsonl(out_path: Path, records: list[dict[str, Any]]) -> None:
    """Write one sorted, deterministic JSON line per run record."""
    # evidence-writer-exempt: JSONL is intentionally emitted one sorted record per line;
    # the shared write_json helper writes a single JSON document, so the immutable JSONL
    # artifact carries its required marker through the generated review sidecar below.
    ordered = sorted(records, key=lambda r: (r["scenario"], r["calibration"], r["seed"]))
    lines = [json.dumps(rec, sort_keys=True) for rec in ordered]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_readme(
    out_dir: Path,
    aggregates: list[dict[str, Any]],
    seeds: list[int],
    substrate_version: str,
    commit: str,
    generated_at: str,
    figure_names: list[str],
) -> None:
    """Write the human-readable README with the honest interpretation."""
    lines: list[str] = []
    lines.append("")
    lines.append("# Issue #5149: Multi-Seed Emergent-Phenomena Campaign (Measured Face-Validity)")
    lines.append("")
    lines.append(
        "Plain-language summary: this bundle measures whether THIS repository's "
        "pedestrian simulator (the bundled `fast-pysf` / PySocialForce Social Force "
        "model) reproduces the canonical crowd-dynamics emergent phenomena "
        "(lane formation in bidirectional flow, doorway oscillation, and an exit "
        f"arching diagnostic) across {len(seeds)} seeds per scenario and speed "
        "calibration, elevating the pinned single-seed exhibit "
        "(`issue_5149_emergent_phenomena_2026-07/`) to measured evidence with "
        "seed-level dispersion."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append(f"- Generated at (UTC): `{generated_at}`")
    lines.append(f"- Git head: `{commit}`")
    lines.append(f"- Substrate: `pysocialforce=={substrate_version}`")
    lines.append(f"- Generation command: `{GENERATION_COMMAND}`")
    lines.append(
        "- Harness modules: `robot_sf/research/emergent_phenomena.py`, "
        "`robot_sf/research/emergent_phenomena_campaign.py`"
    )
    lines.append(f"- Maintainer authorization: {AUTHORIZATION_COMMENT}")
    lines.append(f"- Seeds: `{seeds}`")
    lines.append("- Full machine-readable provenance: `manifest.json`.")
    lines.append("")
    lines.append("## Claim boundary")
    lines.append(
        "This is **measured face-validity (smoke-tier) evidence**: per-seed order "
        "parameters with dispersion across a pinned seed list, at the released "
        "parameterization and a literature-typical speed calibration. It is NOT "
        "benchmark-matrix evidence and NOT paper-grade validation against real "
        "human trajectory datasets (tracked separately in issue #4975). Verdicts "
        "are conservative threshold labels on simple order parameters, suitable "
        "as a behavioral-validity exhibit and regression anchor for force-model "
        "changes (#4972 speed recalibration, #4973 anticipatory variant)."
    )
    lines.append("")
    lines.append("## Results (aggregated across seeds)")
    lines.append("")
    lines.append(
        "| Scenario | Calibration | n seeds | Primary order parameter "
        "(mean +/- std [min, max]) | Verdicts |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for agg in aggregates:
        param = primary_order_parameter(agg["scenario"])
        stats = agg["order_parameter_stats"][param]
        verdicts = ", ".join(f"{k}: {v}" for k, v in agg["verdict_counts"].items())
        lines.append(
            f"| {agg['scenario']} | {agg['calibration']} | {agg['n_seeds']} | "
            f"{param} = {stats['mean']:.3f} +/- {stats['std']:.3f} "
            f"[{stats['min']:.3f}, {stats['max']:.3f}] | {verdicts} "
            f"(majority: {agg['majority_verdict']}) |"
        )
    lines.append("")
    lines.append(
        "Secondary order parameters (lane purity, throughput, burst length, arch "
        "lateral spread) are in `summary.json`; per-seed rows are in `runs.jsonl`."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "Read the verdict counts literally: they are per-seed conservative "
        "threshold labels, and the majority verdict tie-breaks toward the weaker "
        "label so a split seed population never overclaims. The expected pattern "
        "from the 2026-07 single-seed exhibit is that doorway oscillation and "
        "exit arching emerge clearly at both calibrations while lane formation "
        "is weak at the slow released default and somewhat stronger at the "
        "literature-typical speed; this campaign measures how stable that "
        "pattern is across seeds rather than asserting it from one run."
    )
    lines.append("")
    lines.append("## Thresholds (documented, conservative)")
    lines.append(
        f"- Lane formation `clearly_present` if `lane_segregation_index >= "
        f"{LANE_SEGREGATION_CLEAR}`; `weak_partial` if `>= {LANE_SEGREGATION_WEAK}`."
    )
    lines.append(
        f"- Doorway oscillation `clearly_present` if `oscillation_flips >= "
        f"{OSCILLATION_FLIPS_CLEAR}`."
    )
    lines.append(
        f"- Exit arching `clearly_present` if `exit_density_ratio >= {ARCH_DENSITY_RATIO_CLEAR}`."
    )
    lines.append("")
    lines.append("## Reproducibility")
    lines.append(
        "Re-run with the generation command above from the repository root. "
        "Output is deterministic given the pinned seed list and the released "
        "force parameters; pass `--generated-at` with the timestamp above for a "
        "byte-stable re-run on the same platform/environment (`manifest.json` "
        "records the runtime; cross-platform floating-point drift is possible). "
        "File integrity is in `SHA256SUMS`."
    )
    lines.append("")
    lines.append("## Files")
    lines.append("- `README.md` — this file.")
    lines.append("- `manifest.json` — full provenance manifest.")
    lines.append("- `summary.json` — aggregate statistics + verdict distributions.")
    lines.append("- `runs.jsonl` — one record per scenario x calibration x seed.")
    for name in figure_names:
        lines.append(f"- `{name}` — figure.")
    lines.append("- `SHA256SUMS` — integrity manifest for the bundle.")
    content = "\n".join(lines) + "\n"
    write_text(out_dir / "README.md", content, issue_ref=ISSUE_REF, marker_date=generated_at[:10])


def build_campaign_bundle(
    output_dir: Path,
    seeds: list[int],
    generated_at_override: str | None = None,
) -> Path:
    """Run the multi-seed campaign and write the full evidence bundle.

    Args:
        output_dir: Directory to write the bundle into.
        seeds: Campaign seed list.
        generated_at_override: Optional pinned ISO-8601 UTC timestamp for
            byte-stable re-runs.

    Returns:
        The output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Register the bundle directory up front so the per-file writers do not
    # add redundant file-level catalog rows.
    try:
        register_evidence(output_dir, area="benchmark_evidence")
    except (FileNotFoundError, ValueError):
        pass  # outside the repo evidence tree (e.g. tests writing to tmp)

    scenarios = default_scenario_set()
    sim_config = released_default_config()
    calibrations = [RELEASED_DEFAULT_CALIBRATION, LITERATURE_CALIBRATION]
    results = run_multiseed_campaign(
        seeds=seeds, scenarios=scenarios, calibrations=calibrations, sim_config=sim_config
    )
    records = [result_to_run_record(r) for r in results]
    aggregates = aggregate_run_records(records)

    # Figures: trajectory plot at the representative first seed, plus one
    # order-parameter-by-seed figure per scenario.
    figure_names: list[str] = []
    representative_seed = seeds[0]
    for result in results:
        if result.scenario.seed != representative_seed:
            continue
        stem = f"{result.scenario.name}__{result.calibration.name}__seed{representative_seed}.png"
        _trajectory_plot(result, output_dir / stem)
        figure_names.append(stem)
    for scenario in scenarios:
        stem = f"{scenario.name}__order_parameter_by_seed.png"
        _by_seed_plot(scenario.name, records, seeds, output_dir / stem)
        figure_names.append(stem)

    _write_runs_jsonl(output_dir / "runs.jsonl", records)

    commit = _git_commit()
    if generated_at_override:
        generated_at = generated_at_override
    else:
        generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    substrate_version = getattr(pysf, "__version__", "unknown")

    summary = {
        "issue": ISSUE_REF,
        "generated_at_utc": generated_at,
        "git_head": commit,
        "substrate": "pysocialforce",
        "substrate_version": substrate_version,
        "claim_boundary": "measured face-validity (smoke-tier) evidence across a pinned "
        "seed list; not benchmark-matrix evidence and not paper-grade validation "
        "against real trajectory data (#4975)",
        "evidence_status": "smoke evidence",
        "seeds": list(seeds),
        "aggregates": aggregates,
    }
    write_json(output_dir / "summary.json", summary)

    manifest = {
        "schema": "issue_5149_emergent_phenomena_multiseed_manifest.v1",
        "issue": ISSUE_REF,
        "maintainer_authorization": AUTHORIZATION_COMMENT,
        "generated_at_utc": generated_at,
        "git_head": commit,
        "generation_command": GENERATION_COMMAND,
        "harness_modules": [
            "robot_sf/research/emergent_phenomena.py",
            "robot_sf/research/emergent_phenomena_campaign.py",
        ],
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "packages": {
            "pysocialforce": substrate_version,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "seeds": list(seeds),
        "n_runs": len(records),
        "scenarios": [
            {
                "name": s.name,
                "length": s.length,
                "half_width": s.half_width,
                "n_pedestrians": s.n_pedestrians,
                "n_steps": s.n_steps,
                "extra": dict(s.extra),
            }
            for s in scenarios
        ],
        "calibrations": [
            {
                "name": c.name,
                "desired_speed_mean_mps": c.desired_speed_mean,
                "desired_speed_std_mps": c.desired_speed_std,
            }
            for c in calibrations
        ],
        "simulator_config": simulator_config_snapshot(sim_config),
        "verdict_thresholds": {
            "lane_segregation_clear": LANE_SEGREGATION_CLEAR,
            "lane_segregation_weak": LANE_SEGREGATION_WEAK,
            "oscillation_flips_clear": OSCILLATION_FLIPS_CLEAR,
            "arch_density_ratio_clear": ARCH_DENSITY_RATIO_CLEAR,
        },
        "outputs": {
            "runs_jsonl": "runs.jsonl",
            "summary": "summary.json",
            "readme": "README.md",
            "figures": figure_names,
            "integrity": "SHA256SUMS",
        },
        "determinism_note": "Deterministic given the seed list and pinned parameters; "
        "byte-stable re-runs require --generated-at plus the same platform/environment "
        "(cross-platform floating-point drift is possible).",
        "predecessor_bundle": "docs/context/evidence/issue_5149_emergent_phenomena_2026-07",
    }
    write_json(output_dir / "manifest.json", manifest)

    _write_readme(
        output_dir, aggregates, list(seeds), substrate_version, commit, generated_at, figure_names
    )

    # Binary figures and the line-oriented run archive must preserve their exact bytes for
    # SHA256SUMS and therefore carry the evidence marker in same-bundle review sidecars.
    for figure_name in figure_names:
        write_review_sidecar(output_dir / figure_name)
    write_review_sidecar(output_dir / "runs.jsonl")

    # Integrity manifest last so it covers every other file in the bundle.
    write_sha256sums(output_dir)
    return output_dir


def _parse_seeds(raw: str) -> list[int]:
    """Parse a comma-separated seed list.

    Returns:
        List of unique integer seeds in the given order.
    """
    seeds = [int(part) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("seed list must not be empty")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seed list must not contain duplicates")
    return seeds


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=list(DEFAULT_CAMPAIGN_SEEDS),
        help=f"Comma-separated seed list (default: {','.join(map(str, DEFAULT_CAMPAIGN_SEEDS))})",
    )
    parser.add_argument(
        "--generated-at",
        type=str,
        default=None,
        help="Optional pinned ISO-8601 UTC timestamp for byte-stable re-runs "
        "(default: current wall-clock time).",
    )
    args = parser.parse_args(argv)
    out = build_campaign_bundle(args.output_dir, args.seeds, args.generated_at)

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    print(f"Wrote multi-seed emergent-phenomena campaign bundle to {out}")
    print(f"Substrate: pysocialforce=={summary['substrate_version']}")
    print(f"Generated at (UTC): {summary['generated_at_utc']}")
    print(f"Seeds: {summary['seeds']}")
    print("")
    print(f"{'scenario':24s} {'calibration':20s} {'majority':22s} verdicts")
    for agg in summary["aggregates"]:
        verdicts = ", ".join(f"{k}:{v}" for k, v in agg["verdict_counts"].items())
        print(
            f"{agg['scenario']:24s} {agg['calibration']:20s} "
            f"{agg['majority_verdict']:22s} {verdicts}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
