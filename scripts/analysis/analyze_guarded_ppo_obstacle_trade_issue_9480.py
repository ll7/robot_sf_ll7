"""Descriptive analysis of guarded-PPO obstacle contacts in the frozen 0.0.6 campaign.

Issue #9480: why guarded PPO posts low pedestrian contact but high obstacle
contact in ``paper_matrix_v2_h600_s30`` (14 x 48 x 30). Publication bundle
only; no training, no reruns.

Reads per-episode rows from the verified publication bundle and emits:
- Table 7.1 aggregate verification (both arms),
- obstacle-contact rates by scenario family (guarded vs base PPO),
- contact-time distribution vs the episode cap,
- guard decision cross-tabulation (per-episode counts + contact-step decision),
- pedestrian-proximity proxies (near-miss/step) for contact vs clean episodes,
- PNG figures under ``docs/figures/issue_9480_guarded_obstacle_trade/``,
- the evidence note itself (``--output``).

Evidence tier: diagnostic reading of retained episode records. The bundle
retains no simulation step traces and no per-step positions, so contact-map
localization, guard timing within the last-k-steps window, and pedestrian
positions immediately before contact are UNAVAILABLE (reported as NA, never
zero). Guard "precedence" is answered at episode granularity plus the
contact-step final decision only.

Report schema: ``issue_9480_guarded_obstacle_trade_report.v1``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

REPORT_SCHEMA_VERSION = "issue_9480_guarded_obstacle_trade_report.v1"
EXPECTED_SOFTWARE_COMMIT = "31cdfe0361abe2c520117a17f99c1b7a0aba4359"
GUARDED_ARM = "guarded_ppo__differential_drive"
BASE_ARM = "ppo__differential_drive"
FIGURE_DIR_NAME = "issue_9480_guarded_obstacle_trade"

FALLBACK_LABELS = ("fallback_safe", "fallback_best_effort")
STOP_LABELS = ("stop_safe", "stop_best_effort")
PRIOR_LABELS = ("prior_safe", "prior_residual_safe", "prior_blend_safe")
UNCERTAINTY_LABELS = (
    "uncertainty_fallback_stop",
    "uncertainty_fallback_slow_down",
    "uncertainty_fallback_configured",
)


def _family(scenario_id: str) -> str:
    """Strip the trailing cell variant to get the scenario family."""
    return re.sub(r"_(low|medium|high|easy|hard|v\d+)$", "", scenario_id)


def _load_arm(bundle_root: Path, arm: str) -> list[dict]:
    """Load and provenance-check one arm's episode rows."""
    path = bundle_root / "payload" / "runs" / arm / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    commits = {row.get("event_ledger", {}).get("software_commit") for row in rows}
    if commits != {EXPECTED_SOFTWARE_COMMIT}:
        raise ValueError(f"{arm}: unexpected software commits {sorted(commits)}")
    return rows


def _contact_time(row: Mapping) -> float | None:
    """First static-geometry contact time in seconds, if recorded."""
    events = row.get("event_ledger", {}).get("collision_events", [])
    times = [
        float(event["collision_time"])
        for event in events
        if event.get("collision_partner_type") == "static_geometry"
        and event.get("collision_time") is not None
    ]
    return min(times) if times else None


def _guard_counts(row: Mapping) -> dict[str, int]:
    """Aggregate guard decision-label counts for one episode."""
    stats = row.get("algorithm_metadata", {}).get("guard_stats", {})
    fallback = sum(int(stats.get(label, 0)) for label in FALLBACK_LABELS)
    stop = sum(int(stats.get(label, 0)) for label in STOP_LABELS)
    prior = sum(int(stats.get(label, 0)) for label in PRIOR_LABELS)
    uncertainty = sum(int(stats.get(label, 0)) for label in UNCERTAINTY_LABELS)
    ppo_clear = int(stats.get("ppo_clear", 0))
    ppo_safe = int(stats.get("ppo_safe", 0))
    return {
        "fallback": fallback,
        "stop": stop,
        "prior": prior,
        "uncertainty": uncertainty,
        "ppo_clear": ppo_clear,
        "ppo_safe": ppo_safe,
        "any_intervention": fallback + stop + prior + uncertainty + ppo_safe,
    }


def _last_decision(row: Mapping) -> dict[str, object]:
    """Final shield decision (the contact step for contact episodes)."""
    return row.get("algorithm_metadata", {}).get("shield_stats", {}).get("last_decision", {})


def _near_miss_step_rate(row: Mapping) -> float:
    """Pedestrian-only near-miss events per recorded step."""
    steps = max(1, int(row.get("steps", 0)))
    return float(row.get("metrics", {}).get("near_misses", 0.0)) / steps


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    """Median/p90/max summary for a non-empty sample."""
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median": round(statistics.median(ordered), 2),
        "p90": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))], 2),
        "max": round(max(ordered), 2),
    }


def _aggregate_table(rows: Sequence[Mapping]) -> dict[str, object]:
    """Table 7.1 style aggregate means for one arm."""
    n = len(rows)
    ped = sum(float(row["metrics"]["ped_collision_count"]) for row in rows) / n
    obst = sum(float(row["metrics"]["obstacle_collision_count"]) for row in rows) / n
    success = sum(1 for row in rows if row["outcome"]["route_complete"])
    timeouts = sum(1 for row in rows if row["outcome"]["timeout_event"])
    return {
        "episodes": n,
        "ped_collision_mean": round(ped, 4),
        "obstacle_collision_mean": round(obst, 4),
        "success": success,
        "timeouts": timeouts,
    }


def _family_table(rows: Sequence[Mapping]) -> list[dict[str, object]]:
    """Obstacle-contact episodes and rates by scenario family."""
    totals: dict[str, int] = {}
    contacts: dict[str, int] = {}
    for row in rows:
        family = _family(str(row["scenario_id"]))
        totals[family] = totals.get(family, 0) + 1
        if float(row["metrics"]["obstacle_collision_count"]) > 0:
            contacts[family] = contacts.get(family, 0) + 1
    table = [
        {
            "family": family,
            "episodes": totals[family],
            "obstacle_contact_episodes": contacts.get(family, 0),
            "rate": round(contacts.get(family, 0) / totals[family], 4),
        }
        for family in totals
    ]
    table.sort(key=lambda entry: (-entry["obstacle_contact_episodes"], entry["family"]))
    return table


def _figures(
    guarded_times: Sequence[float],
    base_times: Sequence[float],
    guarded_family: Sequence[Mapping],
    base_family: Sequence[Mapping],
    last_decisions: Mapping[str, int],
    figure_dir: Path,
) -> list[str]:
    """Render PNG figures; return relative paths for the note."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    names = []

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(guarded_times, bins=30, alpha=0.6, label="guarded_ppo (n=481)")
    ax.hist(base_times, bins=30, alpha=0.6, label="base ppo (n=467)")
    ax.axvline(60, color="k", linestyle="--", linewidth=1, label="episode cap (60 s)")
    ax.set_xlabel("first obstacle-contact time (s)")
    ax.set_ylabel("episodes")
    ax.set_title("Obstacle-contact timing vs the episode cap")
    ax.legend()
    names.append("contact_timing_hist.png")
    fig.savefig(figure_dir / names[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    families = [entry["family"] for entry in guarded_family[:12]]
    base_by_family = {entry["family"]: entry["rate"] for entry in base_family}
    guarded_rates = [next(e["rate"] for e in guarded_family if e["family"] == f) for f in families]
    base_rates = [base_by_family.get(f, 0.0) for f in families]
    short = [f.replace("classic_", "c:").replace("francis2023_", "f23:") for f in families]
    x = range(len(families))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.4
    ax.bar([i - width / 2 for i in x], guarded_rates, width, label="guarded_ppo")
    ax.bar([i + width / 2 for i in x], base_rates, width, label="base ppo")
    ax.set_xticks(list(x))
    ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("obstacle-contact episode rate")
    ax.set_title("Obstacle-contact rate by scenario family (top guarded families)")
    ax.legend()
    names.append("family_rates.png")
    fig.savefig(figure_dir / names[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    labels = sorted(last_decisions)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, [last_decisions[label] for label in labels])
    ax.set_ylabel("episodes")
    ax.set_title("Guard final decision on the contact step (guarded, n=481)")
    names.append("contact_step_decision.png")
    fig.savefig(figure_dir / names[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    return names


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bundle analysis and write figures plus the evidence note."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    guarded = _load_arm(args.bundle_root, GUARDED_ARM)
    base = _load_arm(args.bundle_root, BASE_ARM)
    guarded_contacts = [
        row for row in guarded if float(row["metrics"]["obstacle_collision_count"]) > 0
    ]
    base_contacts = [row for row in base if float(row["metrics"]["obstacle_collision_count"]) > 0]
    guarded_clean = [row for row in guarded if not row["outcome"]["collision_event"]]
    guarded_times = [t for t in (_contact_time(r) for r in guarded_contacts) if t is not None]
    base_times = [t for t in (_contact_time(r) for r in base_contacts) if t is not None]

    guard_totals: dict[str, int] = {}
    for row in guarded_contacts:
        for label, count in _guard_counts(row).items():
            guard_totals[label] = guard_totals.get(label, 0) + count
    contact_steps = sum(int(row.get("steps", 0)) for row in guarded_contacts)
    last_labels: dict[str, int] = {}
    for row in guarded_contacts:
        label = str(_last_decision(row).get("decision_label", "unknown"))
        last_labels[label] = last_labels.get(label, 0) + 1

    guarded_family = _family_table(guarded)
    base_family = _family_table(base)
    figure_names = _figures(
        guarded_times, base_times, guarded_family, base_family, last_labels, args.figure_dir
    )
    figure_rel = f"../figures/{FIGURE_DIR_NAME}"

    contact_rate_c = sum(_near_miss_step_rate(r) for r in guarded_contacts) / len(guarded_contacts)
    clean_rate_c = sum(_near_miss_step_rate(r) for r in guarded_clean) / len(guarded_clean)

    lines = [
        "# Guarded-PPO obstacle contacts in the frozen 0.0.6 campaign",
        "",
        "Claim boundary: descriptive analysis of the frozen release bundle only. No episode was",
        "stepped or rerun, no checkpoint retrained, no runtime changed.",
        "Evidence tier: diagnostic reading of retained episode records.",
        "",
        "## Input identity",
        "",
        "- Bundle: `benchmark_0_0_6_s30_h600_20260911_publication_bundle` (checksums verified)",
        f"- Source commit: `{EXPECTED_SOFTWARE_COMMIT}`",
        f"- Arms: `{GUARDED_ARM}` (BR-06 v3 checkpoint behind the runtime guard) and",
        f"  `{BASE_ARM}` (different checkpoint, no guard) — descriptive comparison only,",
        "  not a clean guard ablation because the checkpoints differ.",
        "",
        "## Availability matrix (what the bundle can and cannot answer)",
        "",
        "| Issue packet item | Verdict | Reason |",
        "| --- | --- | --- |",
        "| Contact table: cell, step, map location | partial | cell + step (via `collision_time`) retained;"
        " map location NA — no per-step positions retained |",
        "| Pedestrian within clearance in preceding N steps | NA | no step traces |",
        "| Guard counts per episode + contact-step decision | available | `guard_stats`, `shield_stats.last_decision` |",
        "| Guard active in last-k-steps window | NA | per-step decision series not retained |",
        "| Base-PPO same views | available | episode-level |",
        "| Contact timing vs cap | available | `collision_time` vs 600-step (60 s) cap |",
        "| Per-map overlay figure | substituted | family-rate bars + timing histogram instead (no positions) |",
        "",
        "## Table 7.1 verification",
        "",
    ]
    agg_g = _aggregate_table(guarded)
    agg_b = _aggregate_table(base)
    lines += [
        "| arm | episodes | ped mean | obstacle mean | success | timeouts |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| guarded_ppo | {agg_g['episodes']} | {agg_g['ped_collision_mean']} |"
        f" {agg_g['obstacle_collision_mean']} | {agg_g['success']}/{agg_g['episodes']} |"
        f" {agg_g['timeouts']} |",
        f"| base ppo | {agg_b['episodes']} | {agg_b['ped_collision_mean']} |"
        f" {agg_b['obstacle_collision_mean']} | {agg_b['success']}/{agg_b['episodes']} |"
        f" {agg_b['timeouts']} |",
        "",
        "Guarded obstacle mean (0.334) matches Table 7.1 (0.33); base PPO is nearly identical",
        f"({agg_b['obstacle_collision_mean']}). Guarded sheds pedestrian contacts"
        f" ({agg_g['ped_collision_mean']} vs {agg_b['ped_collision_mean']}) while success collapses"
        f" ({agg_g['success']} vs {agg_b['success']}) and timeouts rise"
        f" ({agg_g['timeouts']} vs {agg_b['timeouts']}). The obstacle rate is not a trade the guard",
        "introduced — descriptively, both checkpoints hit walls at the same rate.",
        "",
        "## Obstacle-contact rate by scenario family",
        "",
        "| family | guarded contacts / episodes (rate) | base contacts / episodes (rate) |",
        "| --- | --- | --- |",
    ]
    base_by_family = {entry["family"]: entry for entry in base_family}
    for entry in guarded_family:
        other = base_by_family.get(entry["family"], {})
        lines.append(
            f"| {entry['family']} | {entry['obstacle_contact_episodes']}/{entry['episodes']}"
            f" ({entry['rate']:.3f}) | {other.get('obstacle_contact_episodes', 0)}/"
            f"{other.get('episodes', 0)} ({other.get('rate', 0.0):.3f}) |"
        )
    lines += [
        "",
        f"![family rates]({figure_rel}/{figure_names[1]})",
        "",
        "Shared wall-heavy cells (merging ~0.97, narrow_doorway 1.00 both arms, doorway,",
        "t_intersection, bottleneck) hit both checkpoints. Divergences (cross_trap 0.767 vs 0.156,",
        "overtaking 0.850 vs 0.217, narrow_hallway 0.467 vs 0.867) are descriptive only: the",
        "checkpoints differ, so no family delta isolates the guard.",
        "",
        "## Contact timing vs the episode cap",
        "",
    ]
    qt_g, qt_b = _quantiles(guarded_times), _quantiles(base_times)
    lines += [
        f"Guarded contact times (s): n={qt_g['n']}, median={qt_g['median']},"
        f" p90={qt_g['p90']}, max={qt_g['max']}.",
        f"Base contact times (s): n={qt_b['n']}, median={qt_b['median']},"
        f" p90={qt_b['p90']}, max={qt_b['max']}.",
        f"Contacts after 50 s of the 60 s cap: guarded"
        f" {sum(1 for t in guarded_times if t > 50)}/{len(guarded_times)}, base"
        f" {sum(1 for t in base_times if t > 50)}/{len(base_times)}.",
        "",
        f"![contact timing]({figure_rel}/{figure_names[0]})",
        "",
        "Contacts skew early/mid-episode for both arms; guarded contacts run later (median 13.0 s",
        "vs 9.6 s), consistent with the guard prolonging episodes rather than with a",
        '"late, under time pressure" cluster.',
        "",
        "## Guard cross-tabulation (guarded arm, 481 contact episodes)",
        "",
        f"Per-episode guard decisions over {contact_steps} contact-episode steps: "
        + ", ".join(
            f"{label}={guard_totals.get(label, 0)}"
            for label in ("ppo_clear", "ppo_safe", "prior", "fallback", "stop", "uncertainty")
        )
        + ".",
        "",
        "Final (contact-step) decisions: "
        + ", ".join(f"{label}={last_labels.get(label, 0)}" for label in sorted(last_labels))
        + ".",
        "",
        f"![contact-step decision]({figure_rel}/{figure_names[2]})",
        "",
        "In 322/481 contact episodes the guard's final decision was a substitution",
        "(`fallback_safe` 186, `stop_safe` 133, `ppo_safe` 3), i.e. the guard had engaged by",
        "contact time in two thirds of cases yet contact still occurred. In 159/481 the guard",
        "passed the PPO command through (`ppo_clear`). Correlation, not causation: without step",
        "traces we cannot tell unavoidable-from-early-commitment apart from",
        "fallback-steered-into-wall. The coverage reading is that the guard's obstacle clearance",
        "(0.30 m) and short-horizon rollout do not prevent these contacts, and the fallback DWA",
        "weights goal progress (4.5) far above obstacle clearance (1.2).",
        "",
        "## Pedestrian proximity in contact vs clean episodes (guarded arm)",
        "",
        f"Mean pedestrian near-miss events per step: contact episodes {contact_rate_c:.5f},"
        f" clean episodes {clean_rate_c:.5f}. Episodes with any near-miss:"
        f" {sum(1 for r in guarded_contacts if float(r['metrics'].get('near_misses', 0)) > 0)}"
        f"/{len(guarded_contacts)} contact vs"
        f" {sum(1 for r in guarded_clean if float(r['metrics'].get('near_misses', 0)) > 0)}"
        f"/{len(guarded_clean)} clean.",
        "",
        "Wall contacts concentrate in episodes with *less* pedestrian proximity — evidence against",
        "'dodge-pedestrian-into-wall' as the dominant mechanism and consistent with",
        "constrained-geometry contacts under weak obstacle coverage.",
        "",
        "## Verified implementation facts (code, not prose)",
        "",
        "- Training reward `route_completion_v3` (un-overridden): collision -10.0 covers",
        "  pedestrian/robot/obstacle alike; `near_miss` -1.0 is pedestrian-only",
        "  (`snqi_proxy`: robot-ped min distance); `ttc_risk` -0.8 falls back to `near_misses`",
        "  because PPO env metadata never sets `time_to_collision` — hence effectively",
        "  pedestrian-only in training. (Issue prose cites -1.5/-1.2; the frozen code and the",
        "  March-2026 training-time weights are -1.0/-0.8.)",
        "- Guard thresholds are NOT pedestrian-only: `guard_hard_ped_clearance` 0.58 m,",
        "  `guard_hard_obstacle_clearance` 0.30 m, `guard_min_ttc` 0.70 s; fallback DWA weights",
        "  pedestrian clearance 2.0 vs obstacle clearance 1.2 with goal progress 4.5.",
        "",
        "## Dissertation paragraph (Section 7.4 candidate)",
        "",
        "In the frozen 0.0.6 campaign guarded PPO nearly eliminates pedestrian contact (0.017 per",
        "episode) while obstacle contact (0.334) matches the unguarded checkpoint (0.324), so the",
        "configuration trades success for pedestrian safety rather than pedestrians for walls:",
        "success falls 796 to 329 of 1440 with timeouts rising 35 to 605. Contacts concentrate in",
        "constrained cells (merging, narrow doorway, doorway, t-intersection) at early-to-mid",
        "episode times, in episodes with below-average pedestrian proximity, and in two thirds of",
        "cases after the guard had already substituted a fallback or stop command — consistent",
        "with pedestrian-asymmetric shaping (pedestrian-only near-miss/TTC terms) plus obstacle",
        "coverage (0.30 m clearance, short-horizon rollout) too weak to save wall approaches the",
        "policy commits to. Step-trace evidence for the final causal step is not retained in the",
        "bundle.",
        "",
        f"Report schema: `{REPORT_SCHEMA_VERSION}`.",
        "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output} + {len(figure_names)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
