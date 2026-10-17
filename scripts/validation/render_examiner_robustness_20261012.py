"""Render portable tables from the examiner study's reconciled diagnostic data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "docs/context/robsweep"


def number(value: float | None) -> str:
    """Format an optional coefficient without inventing a rank for ties."""
    return "undefined" if value is None else f"{value:.3f}"


def interval(values: list[float] | None) -> str:
    """Render a paired seed-bootstrap percentile interval."""
    return "undefined" if values is None else f"[{values[0]:.3f}, {values[1]:.3f}]"


def crowd_summary(lines: list[str]) -> None:
    """Present actual window definitions and physically clear exit inputs."""
    probes = json.loads((BUNDLE / "evidence/crowd_probes.json").read_text())
    sanity = json.loads((BUNDLE / "evidence/sanity.json").read_text())
    lines.extend(
        [
            "",
            "Crowd sanity (30 dev seeds; 40 pedestrians; preferred speed 0.65 m/s).",
            "",
            "| Probe | Result | Interpretation |",
            "|---|---|---|",
        ]
    )
    for window in ([7.5, 15.0], [15.0, 30.0], [30.0, 60.0]):
        rows = [r for r in probes["lane_windows"] if r["actual_window_s"] == window]
        segregation = np.mean([r["segregation"] for r in rows])
        purity = np.mean([r["purity"] for r in rows])
        weak = sum(r["segregation"] >= 0.15 for r in rows)
        clear = sum(r["segregation"] >= 0.5 for r in rows)
        lines.append(
            f"| Bidirectional corridor, {window[0]:g}–{window[1]:g} s | segregation {segregation:.3f}; purity {purity:.3f}; weak {weak}/30; clear {clear}/30 | mainly weak transient separation; sustained lanes not established |"
        )
    bidirectional = [r for r in sanity["rows"] if r["scenario"] == "narrow_doorway"]
    doors = probes["unidirectional_door"]
    lines.append(
        f"| Bidirectional 1.2 m door | {np.mean([r['throughput_person_s'] for r in bidirectional]):.4f} people/s; {min(r['valid_unique_crossings'] for r in bidirectional)}–{max(r['valid_unique_crossings'] for r in bidirectional)}/40 crossing in 60 s | opposing-flow bottleneck largely blocks |"
    )
    lines.append(
        f"| Unidirectional, goal-aligned 1.2 m door | {np.mean([r['throughput_person_s'] for r in doors]):.3f} people/s; {min(r['valid_unique_crossings'] for r in doors)}–{max(r['valid_unique_crossings'] for r in doors)}/40 crossings in 60 s | passage emerges |"
    )
    lines.append(
        f"| Unidirectional actors with no wall overlap at any recorded step | {np.mean([r['wall_clear_unique_crossings'] / 60 for r in doors]):.3f} people/s | {sum(r['wall_overlap_ped_steps'] for r in doors):,} wall-overlap pedestrian-steps remain in the complete population |"
    )
    lines.extend(
        [
            "",
            "Final probes explicitly use the release's wrapped_v2 pair kernel and historical legacy_shifted_gradient_v1 wall law. All corridor/door resets are independently clear for 0.4 m pedestrian bodies. Force radius remains 0.35 m, while the physical check uses 0.4 m. Earlier legacy-kernel, invalid-reset, failed rejection and off-axis doorway probes are retained as excluded diagnostics. Throughput counts each actor once at the door plane, interpolates lateral position, and requires its disc to fit inside the aperture. The wall-clear subset excludes any actor with a sampled wall overlap during the run; it does not prove swept or pedestrian-pair collision freedom. These finite cohorts do not test continuous-flow capacity or prove that sustained lanes cannot emerge under other densities or boundaries.",
            "",
            "The model produces mainly weak transient lane sorting and some one-way door flow in these finite-population fixtures. It does not establish stable lanes, realistic counterflow, or empirically calibrated throughput. Known social-force phenomena are discussed in [Helbing and Molnár (1995)](https://doi.org/10.1103/PhysRevE.51.4282). These model-specific diagnostics are not empirical crowd validation.",
        ]
    )


def sensitivity_tables(lines: list[str], summary: dict) -> None:
    """Show top-set ties and family difficulty across the complete five-arm grid."""
    variants = [r for r in summary["rankings"] if r["treatment"].startswith("r")]
    if len(variants) != 9 or not all(r.get("complete_matched_roster") for r in variants):
        return
    lines.extend(
        [
            "",
            "Five-planner aggregate success rates (300 episodes each). Bold values mark the success-leading set, retaining ties.",
            "",
            "| Radius / preferred speed | ORCA | Social force | Static hybrid v4 | Goal | PPO |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    planners = ["orca", "social_force", "hybrid_rule_v4_fast_progress_static_escape", "goal", "ppo"]
    for r in variants:
        outcomes = {p["planner"]: p for p in r["planner_outcomes"]}
        leader = max(p["success"] for p in outcomes.values())
        values = []
        for planner in planners:
            value = f"{outcomes[planner]['success']:.1%}"
            if outcomes[planner]["success"] == leader:
                value = f"**{value}**"
            values.append(value)
        lines.append(f"| {r['treatment']} | " + " | ".join(values) + " |")
    control = next(r for r in variants if r["treatment"] == "r1_v0.65")
    families = [f["family"] for f in control["family_outcomes"]]
    difficulty = {
        r["treatment"]: dict(
            zip(
                families,
                rankdata([f["success"] for f in r["family_outcomes"]], method="min"),
                strict=True,
            )
        )
        for r in variants
    }
    lines.extend(
        [
            "",
            "Family difficulty across the five sweep planners (150 episodes per family/treatment). Hardness rank 1 is lowest success; ties retain the minimum rank. Ranges cover the nine design settings, not confidence intervals.",
            "",
            "| Family | Release-setting success | Success range | Release-setting collision | Collision range | Hardness rank baseline | Hardness rank range |",
            "|---|---:|---|---:|---|---:|---|",
        ]
    )
    for family in families:
        records = [next(f for f in r["family_outcomes"] if f["family"] == family) for r in variants]
        base = next(f for f in control["family_outcomes"] if f["family"] == family)
        ranks = [difficulty[r["treatment"]][family] for r in variants]
        lines.append(
            f"| {family} | {base['success']:.1%} | {min(f['success'] for f in records):.1%}–{max(f['success'] for f in records):.1%} | {base['collision']:.1%} | {min(f['collision'] for f in records):.1%}–{max(f['collision'] for f in records):.1%} | {difficulty[control['treatment']][family]:g} | {min(ranks):g}–{max(ranks):g} |"
        )


def render() -> None:
    """Write a complete report, visibly labeling any pending acquisition."""
    summary = json.loads((BUNDLE / "evidence/summary.json").read_text())
    study = json.loads((BUNDLE / "inputs/study.json").read_text())
    cells = list(csv.DictReader((BUNDLE / "evidence/cells.csv").open()))
    lookup = {(r["treatment"], r["planner"], r["family"]): r for r in cells}
    status = (
        "complete"
        if summary["missing"] == 0 and not summary["excluded"]
        else "acquisition incomplete; no final robustness verdict"
    )
    lines = [
        "# Examiner robustness study — 2026-10-12",
        "",
        f"**Diagnostic-only development evidence; {status}.** Requested {summary['requested']:,}; recorded {summary['recorded']:,}; eligible {summary['eligible']:,}; missing {summary['missing']:,}; excluded {len(summary['excluded']):,}.",
        "",
        "Seeds 1061–1090 only; host CPUs; differential drive; frozen model weights. One medium cell for each multi-cell family and five singleton Francis cells from Figure 7.3. These ten cells are a bounded sample, not the full release matrix or all parameterizations of those families. Narrow doorway is outside that figure. No manuscript claim is admitted by this report.",
        "",
        "The comparator is a new matched control using 0.0.8 configurations on unchanged runtime source main `33a4735900153e742828227cc2171436ea09cd90`. It is not a comparison against historical release episode scores. Radius changes retain planner thresholds and PPO weights; they may also affect radius-aware spawn/goal sampling. Speed treatment is deterministic preferred speed, not a guarantee of actual walking speed during holds or interactions.",
        "",
        "The [measured verdict](VERDICT.md) distinguishes robust leading-set and family conclusions from design-dependent strict rankings. [Reproduction](REPRODUCE.md), [study contract](PLAN.md), and [analysis test-value proof](TESTS.md) define the execution and interpretation boundaries.",
        "",
        "Verified release facts:",
        "",
        "| Quantity | Verified value / correction |",
        "|---|---|",
        "| Robot envelope / drive | 1.0 m radius disc; differential drive |",
        "| Physical speed | default 2.0 m/s; pedestrian-overtaking scenario explicitly uses 0.7 m/s |",
        "| Pedestrian speed | default preferred cap 1.3 × 0.5 = 0.65 m/s; actual speed can be lower. Authored actors also use 0.3, 0.7 and 0.8 m/s initial/role speeds. Therefore 0.65 is not universal across the release. |",
        "| Narrow doorway | measured SVG gap 2.0 m; radius 1.0 gives zero diameter margin and no strictly positive-clearance crossing. The release requires 0.1 m safety margin per side. An ideal exactly tangent centerline is a mathematical limiting case, not a valid traversal under that contract. |",
        "| Radius 0.5 / 0.3 doorway geometry | diameter margins 1.0 / 1.4 m; geometrically feasible centerline, without implying planner/crowd success |",
        "",
        "| Planner | Release global command cap (m/s) | Requested swap (m/s) |",
        "|---|---:|---:|",
    ]
    for p in study["roster"]:
        cap = p["diagnostic_release_cap"]
        lines.append(f"| {p['key']} | {cap:g} | {2.0 if cap != 2.0 else 1.2:g} |")
    lines.extend(
        [
            "",
            "The listed caps are confirmed, with two further qualifications: SA-CADRL's preferred command speed is 1.0 m/s despite its 2.0 ceiling; hybrid/guarded/predictive controllers retain conditional lower caps. The requested swap still has two cap groups. It is a cap-sensitivity check, not a fully equal-cap comparison. Swap configs change both the plant ceiling and planner ceiling; DWA's candidate count and relative spacing are retained while its velocity grid expands to the new ceiling.",
            "",
            "Planner-rank correlations versus the matched legacy control. Radius/speed correlations use five planners; the cap swap uses all fourteen. Success and collision incidence are ranked separately; no SNQI robustness claim is made. Kendall tau-b retains ties. Brackets are paired seed-bootstrap 95% percentile intervals (1,000 resamples), conditional on defined tau draws (counts in summary.json); these quantify seed variation in these fixed cells, not scenario-selection uncertainty. Undefined means a constant/tied ordering has no tau, or the complete paired roster is unavailable.",
            "",
            "| Radius / preferred speed or cap arm | Success tau-b | 95% CI | Collision tau-b | 95% CI | Family success tau-b |",
            "|---|---:|---|---:|---|---:|",
        ]
    )
    for r in summary["rankings"]:
        if r["treatment"] == "legacy":
            continue
        lines.append(
            f"| {r['treatment']} | {number(r.get('success_tau_b'))} | {interval(r.get('success_tau_ci95'))} | {number(r.get('collision_tau_b'))} | {interval(r.get('collision_tau_ci95'))} | {number(r.get('family_success_tau_b'))} |"
        )
    swap = next(r for r in summary["rankings"] if r["treatment"] == "swap")
    sensitivity_tables(lines, summary)
    if swap.get("complete_matched_roster"):
        outcomes = swap["planner_outcomes"]
        before = rankdata([-r["baseline_success"] for r in outcomes], method="min")
        after = rankdata([-r["success"] for r in outcomes], method="min")
        lines.extend(
            [
                "",
                "Complete-roster cap swap (each planner: 300 matched episodes; success ranks retain ties).",
                "",
                "| Planner | Success baseline → swap | Collision baseline → swap | Success rank baseline → swap |",
                "|---|---|---|---|",
            ]
        )
        for i, r in enumerate(outcomes):
            lines.append(
                f"| {r['planner']} | {r['baseline_success']:.1%} → {r['success']:.1%} | {r['baseline_collision']:.1%} → {r['collision']:.1%} | {before[i]:g} → {after[i]:g} |"
            )
    planners = ["orca", "social_force", "hybrid_rule_v4_fast_progress_static_escape", "goal", "ppo"]
    for treatment in ["legacy"] + [
        r["treatment"] for r in summary["rankings"] if r["treatment"].startswith("r")
    ]:
        lines.extend(
            [
                "",
                f"**{treatment}: per-cell success/collision episode counts (requested 30 each).**",
                "",
                "| Family / selected cell | ORCA | Social force | Static hybrid v4 | Goal | PPO |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for family, scenario in study["families"].items():
            values = []
            for planner in planners:
                row = lookup[treatment, planner, family]
                value = f"{row['successes']}/{row['collision_episodes']}"
                if row["eligible"] != "30":
                    value += f" (n={row['eligible']}; missing={row['missing']}; excluded={row['excluded']})"
                values.append(value)
            lines.append(f"| {family}: `{scenario}` | " + " | ".join(values) + " |")
    crowd_summary(lines)
    lines.extend(
        [
            "",
            "Detailed outcomes for **every** requested planner/treatment/family cell, including the full cap roster, typed collision incidence and missingness: [cells.csv](evidence/cells.csv). Machine-readable paired correlations and planner/family aggregates: [summary.json](evidence/summary.json). Geometry proof: [door_geometry.json](evidence/door_geometry.json). All-release fact audit: [release_fact_audit.json](evidence/release_fact_audit.json).",
            "",
            "Custody: compressed canonical episode records (checkout-prefix normalization only), complete crowd trajectories, canary traces, exact acquisition driver and input bytes are retained. Producer sidecar references and file checksums are rebound to portable compressed artifacts in the [custody ledger](evidence/portable_custody.json); scientific values remain unchanged. [Source identity](evidence/acquisition_source.json) records an administrative Git-epoch change during acquisition; runtime/planner bytes and every acquisition input remained unchanged against the pinned base. Later tooling cleanup is separate from the archived acquisition driver. This is diagnostic provenance, not a frozen-release replay certificate.",
            "",
            "Archived YAML files under `inputs/` preserve exact bytes at their original relative-path layout; regenerate execution configs under `configs/benchmarks/robsweep/` before running them. [Input locations](evidence/input_locations.json) map each archived artifact to that original config path. [SHA256SUMS](evidence/SHA256SUMS) is checked from the repository root. No frozen config, release artifact, benchmark default, model weight, or locked thesis text is changed.",
        ]
    )
    (BUNDLE / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    render()
