#!/usr/bin/env python3
"""Build the registered emergent-phenomena demonstration artifact for issue #5149.

This is the pinned generation command for the pedestrian substrate's
behavioral-validity exhibit. It runs three canonical crowd-dynamics scenarios
(bidirectional corridor, narrow doorway, high-density exit) directly on the
released ``fast-pysf`` (PySocialForce) substrate, at the released-default speed
calibration (~0.65 m/s desired) and at a literature-typical calibration
(~1.3 m/s), records trajectories, computes simple order parameters, and exports:

- one trajectory plot (PNG) per scenario x calibration,
- a ``summary.json`` with provenance + order parameters,
- a human-readable ``README.md`` with the honest interpretation, and
- a ``SHA256SUMS`` manifest.

The harness lives in ``robot_sf.research.emergent_phenomena``; this script is a
thin orchestrator that writes the pinned evidence bundle.

Claim boundary: this is **diagnostic behavioral-validity evidence** (smoke-tier
emergent-pattern demonstration), NOT paper-grade validation against real human
trajectory datasets (that is tracked separately in issue #4975). It establishes
that THIS implementation, at the released parameterization, does or does not
reproduce the canonical phenomena, and pins a regression anchor for force-model
changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless rendering; no display required
import matplotlib.pyplot as plt
import numpy as np

from robot_sf.evidence.writers import write_json, write_sha256sums
from robot_sf.research.emergent_phenomena import (
    EmergentPhenomenaReport,
    ScenarioResult,
    run_emergent_phenomena_demo,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_5149_emergent_phenomena_2026-07")
ISSUE_REF = "robot_sf_ll7#5149"

# How long to label a phenomenon as "clearly present" vs "weak/partial" vs
# "absent". These thresholds are documented in the README and are intentionally
# conservative so the demonstration does not overclaim.
LANE_SEGREGATION_CLEAR = 0.5
LANE_SEGREGATION_WEAK = 0.15
OSCILLATION_FLIPS_CLEAR = 2
ARCH_DENSITY_RATIO_CLEAR = 2.0


def _git_commit() -> str:
    """Return the current commit hash, or ``unknown`` outside git."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _lane_verdict(value: float) -> str:
    if value >= LANE_SEGREGATION_CLEAR:
        return "clearly_present"
    if value >= LANE_SEGREGATION_WEAK:
        return "weak_partial"
    return "absent_or_negligible"


def _oscillation_verdict(flips: float) -> str:
    return "clearly_present" if flips >= OSCILLATION_FLIPS_CLEAR else "absent_or_negligible"


def _arch_verdict(ratio: float) -> str:
    return "clearly_present" if ratio >= ARCH_DENSITY_RATIO_CLEAR else "absent_or_negligible"


def _scenario_plot(result: ScenarioResult, out_path: Path) -> None:
    """Render a trajectory plot for one scenario x calibration."""
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
        f"{result.scenario.name} | {result.calibration.name} "
        f"(v_des~{result.max_speeds.mean():.2f} m/s)"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    # Mark door/exit for the bottleneck scenarios.
    if result.scenario.name == "narrow_doorway":
        door_x = float(result.scenario.extra.get("door_x", result.scenario.length / 2.0))
        ax.axvline(door_x, color="red", lw=1.0, ls="--", alpha=0.5, label="door")
    elif result.scenario.name == "high_density_exit":
        ax.axvline(result.scenario.length, color="red", lw=1.0, ls="--", alpha=0.5, label="exit")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _result_to_record(result: ScenarioResult) -> dict[str, Any]:
    """Serialize one scenario result into a JSON-friendly record."""
    ops = {k: float(v) for k, v in result.order_parameters.items()}
    verdict = _derive_verdict(result.scenario.name, ops)
    return {
        "scenario": result.scenario.name,
        "calibration": result.calibration.name,
        "desired_speed_mean_mps": float(result.calibration.desired_speed_mean),
        "desired_speed_std_mps": float(result.calibration.desired_speed_std),
        "realized_mean_max_speed_mps": float(result.max_speeds.mean()),
        "n_pedestrians": int(result.scenario.n_pedestrians),
        "n_steps": int(result.scenario.n_steps),
        "duration_secs": float(result.scenario.n_steps * result.trajectory.dt),
        "dt_secs": float(result.trajectory.dt),
        "seed": int(result.scenario.seed),
        "scenario_extra": dict(result.scenario.extra),
        "order_parameters": ops,
        "phenomenon_verdict": verdict,
    }


def _derive_verdict(scenario: str, ops: dict[str, float]) -> str:
    """Map the order parameters to a coarse phenomenon-present label."""
    if scenario == "bidirectional_corridor":
        return _lane_verdict(ops.get("lane_segregation_index", 0.0))
    if scenario == "narrow_doorway":
        return _oscillation_verdict(ops.get("oscillation_flips", 0.0))
    if scenario == "high_density_exit":
        return _arch_verdict(ops.get("exit_density_ratio", 0.0))
    return "unknown"


def _write_readme(
    out_dir: Path,
    records: list[dict[str, Any]],
    report: EmergentPhenomenaReport,
    commit: str,
    generated_at: str,
) -> None:
    """Write the human-readable README with the honest interpretation."""
    lines: list[str] = []
    lines.append(f"<!-- AI-GENERATED ({ISSUE_REF}, {generated_at[:10]}) - NEEDS-REVIEW -->")
    lines.append("")
    lines.append(
        "# Issue #5149: Emergent-Phenomena Demonstration for the Released Pedestrian Substrate"
    )
    lines.append("")
    lines.append(
        "Plain-language summary: this bundle demonstrates whether THIS repository's "
        "pedestrian simulator (the bundled `fast-pysf` / PySocialForce Social Force "
        "model) reproduces the three canonical crowd-dynamics emergent phenomena "
        "(lane formation in bidirectional flow, doorway oscillation, and exit "
        "arching/clogging), run at the released-default speed calibration (~0.65 m/s "
        "desired) and at a literature-typical calibration (~1.3 m/s)."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append(f"- Generated at (UTC): `{generated_at}`")
    lines.append(f"- Git head: `{commit}`")
    lines.append(f"- Substrate: `pysocialforce=={report.substrate_version}`")
    lines.append(
        "- Generation command: "
        "`uv run python scripts/validation/build_issue_5149_emergent_phenomena_demo.py`"
    )
    lines.append("- Harness module: `robot_sf/research/emergent_phenomena.py`")
    lines.append("")
    lines.append("## Claim boundary")
    lines.append(
        "This is **diagnostic behavioral-validity (smoke-tier) evidence**, not "
        "paper-grade validation against real human trajectory datasets (tracked "
        "separately in issue #4975). It establishes whether the phenomena are "
        "reproducible in this implementation at the released parameterization, and "
        "pins a regression anchor for force-model changes (e.g. the anticipatory "
        "variant in #4973, speed recalibration in #4972)."
    )
    lines.append("")
    lines.append("## Speed calibrations")
    lines.append(
        "- `released_default`: desired speed ~0.65 m/s, reproducing the released "
        "default regime (`initial_speed=0.5`, `max_speed_multiplier=1.3`; see #4972)."
    )
    lines.append(
        "- `literature_typical`: desired speed ~1.3 m/s (Moussaid et al. 2010, "
        "doi:10.1371/journal.pone.0010047)."
    )
    lines.append(
        "- The desired speed is realized through the released substrate's own "
        "speed-derivation logic (`max_speeds = max_speed_multiplier * "
        "initial_speeds`); the harness sets spawn velocity magnitude to "
        "`desired / max_speed_multiplier` along the goal direction rather than "
        "patching the force stack."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Scenario | Calibration | Order parameters | Phenomenon verdict |")
    lines.append("| --- | --- | --- | --- |")
    for rec in records:
        ops_str = ", ".join(f"{k}={v:.3f}" for k, v in rec["order_parameters"].items())
        lines.append(
            f"| {rec['scenario']} | {rec['calibration']} | {ops_str} | "
            f"{rec['phenomenon_verdict']} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "Read the verdict column literally: `clearly_present` means the order "
        "parameter crossed a conservative documented threshold; `weak_partial` "
        "means a detectable but non-robust signal; `absent_or_negligible` means "
        "the phenomenon did not emerge at these parameters in this run. Lane "
        "formation in particular is expected to be the weakest signal at the slow "
        "released-default regime (the issue itself flags this as genuinely open); "
        "the literature-typical speed is expected to strengthen it. Each scenario "
        "is a single seeded run (deterministic given the seed), so a verdict is a "
        "regression anchor, not a population statistic."
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
        "Output is deterministic given the pinned seed (`5149`) and the released "
        "force parameters. Trajectory plots are PNG; numeric results are in "
        "`summary.json`; file integrity is in `SHA256SUMS`."
    )
    lines.append("")
    lines.append("## Files")
    lines.append("- `README.md` — this file.")
    lines.append("- `summary.json` — provenance + per-scenario order parameters.")
    for rec in records:
        stem = f"{rec['scenario']}__{rec['calibration']}"
        lines.append(f"- `{stem}.png` — trajectory plot.")
    lines.append("- `SHA256SUMS` — integrity manifest for the bundle.")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_packet(output_dir: Path, generated_at_override: str | None = None) -> Path:
    """Run the demonstration and write the full evidence bundle to ``output_dir``.

    Args:
        output_dir: Directory to write the bundle into.
        generated_at_override: Optional pinned ISO-8601 UTC timestamp for
            byte-stable re-runs. When ``None`` the current wall-clock time is
            used (matches the repo evidence convention; pass a pinned value to
            reproduce a bundle exactly).

    Returns:
        The output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run_emergent_phenomena_demo()
    records = [_result_to_record(r) for r in report.results]

    # Trajectory plots.
    for rec, result in zip(records, report.results, strict=True):
        stem = f"{rec['scenario']}__{rec['calibration']}"
        _scenario_plot(result, output_dir / f"{stem}.png")

    # Summary JSON with provenance.
    commit = _git_commit()
    if generated_at_override:
        generated_at = generated_at_override
    else:
        generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    summary = {
        "issue": ISSUE_REF,
        "generated_at_utc": generated_at,
        "git_head": commit,
        "substrate": "pysocialforce",
        "substrate_version": report.substrate_version,
        "claim_boundary": "diagnostic behavioral-validity (smoke-tier) evidence; "
        "not paper-grade validation against real trajectory data (#4975)",
        "evidence_status": "diagnostic-only",
        "released_default_config": report.config_json,
        "results": records,
    }
    write_json(output_dir / "summary.json", summary)

    # README.
    _write_readme(output_dir, records, report, commit, generated_at)

    # Integrity manifest (written last so it covers README + summary + plots).
    write_sha256sums(output_dir)
    return output_dir


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
        "--generated-at",
        type=str,
        default=None,
        help="Optional pinned ISO-8601 UTC timestamp for byte-stable re-runs "
        "(default: current wall-clock time).",
    )
    args = parser.parse_args(argv)
    out = build_packet(args.output_dir, generated_at_override=args.generated_at)
    # Print a compact result table for the operator.
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"Wrote emergent-phenomena demonstration bundle to {out}")
    print(f"Substrate: pysocialforce=={summary['substrate_version']}")
    print(f"Generated at (UTC): {summary['generated_at_utc']}")
    print("")
    print(f"{'scenario':24s} {'calibration':20s} verdict")
    for rec in summary["results"]:
        ops = rec["order_parameters"]
        ops_str = ", ".join(f"{k}={v:.3f}" for k, v in ops.items())
        print(
            f"{rec['scenario']:24s} {rec['calibration']:20s} "
            f"{rec['phenomenon_verdict']:24s} {ops_str}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
