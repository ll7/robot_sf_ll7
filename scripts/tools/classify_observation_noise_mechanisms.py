#!/usr/bin/env python3
"""Mechanism-classification report for observation-noise envelope evidence.

Reads an existing observation-noise envelope ``summary.json`` and emits a
mechanism-layer classification report as compact JSON and Markdown.

Usage::

    uv run python scripts/tools/classify_observation_noise_mechanisms.py \\
        --evidence docs/context/evidence/issue_2755_observation_noise_envelope_2026-06-13/summary.json \\
        --output-dir docs/context/evidence/issue_2782_observation_noise_mechanisms

When ``--evidence`` is omitted the script falls back to the default
envelope evidence path.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
from typing import Any

from robot_sf.benchmark.observation_noise_mechanism_classifier import (
    MECHANISM_LABELS,
    classify_all_conditions,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE = (
    REPO_ROOT
    / "docs/context/evidence/issue_2755_observation_noise_envelope_2026-06-13"
    / "summary.json"
)


def _git_head() -> str:
    """Return the short git HEAD, or empty string on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=5,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _relative_to_repo(path: pathlib.Path) -> str:
    """Return path relative to REPO_ROOT, or the original path string."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_evidence(path: pathlib.Path) -> dict[str, Any]:
    """Load the envelope summary JSON."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _build_mechanism_report(
    classified: list[dict[str, Any]],
    envelope_meta: dict[str, Any],
    repro: dict[str, Any],
) -> dict[str, Any]:
    """Build the mechanism-classification JSON report."""
    mechanism_summary: dict[str, list[str]] = {}
    for r in classified:
        label = r["mechanism"]["label"]
        mechanism_summary.setdefault(label, []).append(r["condition"])
    assigned_labels = sorted(mechanism_summary)

    return {
        "schema_version": "observation_noise_mechanism.v1",
        "issue": 2782,
        "claim_boundary": (
            "Diagnostic mechanism-layer classification of observation-noise "
            "envelope evidence. Not paper-facing benchmark proof. Classifies "
            "each perturbation condition by the pipeline layer at which it "
            "had (or did not have) an effect."
        ),
        "source_evidence": envelope_meta.get("source_path", ""),
        "reproducibility": repro,
        "fixture": {
            "trace_path": envelope_meta.get("trace_path", ""),
            "scenario_id": envelope_meta.get("scenario_id", ""),
            "first_visible_step": envelope_meta.get("first_visible_step"),
        },
        "mechanism_labels_valid": sorted(MECHANISM_LABELS),
        "mechanism_labels_assigned": assigned_labels,
        "conditions": [
            {
                "condition": r["condition"],
                "spec": r.get("spec", {}),
                "mechanism": r["mechanism"],
                "condition_classification": r.get("classification", {}),
                "response_delay_steps": r.get("response_delay_steps"),
                "closest_distance_m": r.get("closest_distance_m"),
                "action_proxy_changes": r.get("action_proxy_changes", {}),
            }
            for r in classified
        ],
        "summary": {
            "total_conditions": len(classified),
            "mechanism_distribution": {k: len(v) for k, v in mechanism_summary.items()},
            "mechanism_to_conditions": mechanism_summary,
        },
    }


def _generate_markdown(
    classified: list[dict[str, Any]],
    envelope_meta: dict[str, Any],
    repro: dict[str, Any],
) -> str:
    """Generate the Markdown mechanism report."""
    lines: list[str] = [
        "# Observation-Noise Mechanism Classification",
        "",
        "## Claim Boundary",
        "",
        "**Diagnostic mechanism-layer classification only. Not paper-facing benchmark proof.**",
        "This classifies each perturbation condition from the observation-noise "
        "envelope by the pipeline layer at which it had (or did not have) an "
        "effect: observation source, command, trajectory, or timing.",
        "The label table is the valid vocabulary; the distribution section shows "
        "which labels were actually assigned for this evidence bundle.",
        "",
        "## Reproducibility",
        "",
        f"- **Issue:** #{repro['issue']}",
        f"- **Generated at (UTC):** {repro['generated_at_utc']}",
        f"- **Command:** `{repro['command']}`",
        f"- **Repo HEAD:** `{repro['repo_head']}`",
        f"- **Source evidence:** `{envelope_meta.get('source_path', '')}`",
        f"- **Fixture:** `{envelope_meta.get('trace_path', '')}`",
        f"- **Scenario:** {envelope_meta.get('scenario_id', '')}",
        f"- **First visible step:** {envelope_meta.get('first_visible_step', '?')}",
        "",
        "## Mechanism Label Vocabulary",
        "",
        "| Label | Pipeline layer | Meaning |",
        "|---|---|---|",
        "| `noise_stayed_below_decision_threshold` | Decision | Noise present but did not cross the decision boundary |",
        "| `observation_affected_source_but_not_command` | Observation -> Command | Observation perturbed but command unchanged |",
        "| `observation_did_not_affect_selected_source` | Observation | Perturbation did not reach the policy input |",
        "| `command_changed_but_trajectory_did_not` | Command -> Trajectory | Command changed but trajectory unchanged |",
        "| `delay_shifted_stop_timing` | Timing | Delay shifted stop/yield decision timing |",
        "| `occlusion_changed_first_actionable_frame` | Observation timing | Occlusion changed when policy first saw the pedestrian |",
        "| `scenario_had_no_actionable_conflict` | Scenario | No actionable conflict for noise testing |",
        "| `stored_action_proxy_prevents_live_conclusion` | Proxy boundary | Action proxies from stored trace; live replay required |",
        "| `diagnostic_only` | Reference | Baseline or mixed-effects reference |",
        "| `inconclusive` | Fallback | Insufficient data for classification |",
        "",
        "## Conditions",
        "",
    ]

    for r in classified:
        m = r["mechanism"]
        spec = r.get("spec", {})
        lines.append(f"### {r['condition']}")
        lines.append("")
        lines.append(f"- **Mechanism label:** `{m['label']}`")
        lines.append(f"  - {m['rationale']}")
        lines.append(
            f"- **Prior classification:** `{r.get('classification', {}).get('label', '?')}`"
        )
        lines.append(f"- **Noise profile:** {spec.get('noise_profile', '?')}")
        if r.get("response_delay_steps") is not None:
            lines.append(f"- **Response delay:** {r['response_delay_steps']} steps")
        lines.append(f"- **Closest distance:** {r.get('closest_distance_m', 'N/A')} m")
        lines.append("")

    # Distribution summary
    mechanism_summary: dict[str, list[str]] = {}
    for r in classified:
        label = r["mechanism"]["label"]
        mechanism_summary.setdefault(label, []).append(r["condition"])

    lines.extend(
        [
            "## Mechanism Distribution",
            "",
            "| Mechanism label | Conditions | Count |",
            "|---|---|---|",
        ]
    )
    for label in sorted(mechanism_summary.keys()):
        conditions = mechanism_summary[label]
        lines.append(f"| `{label}` | {', '.join(conditions)} | {len(conditions)} |")

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Single deterministic fixture (seed=111), single scenario family.",
            "- Action proxies are from the stored trace, not re-executed; "
            "command -> trajectory conclusions require live replay.",
            "- Mechanism labels are rule-based heuristics, not causal proof.",
            "- Not paper-facing benchmark evidence.",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Run mechanism classification and write reports."""
    parser = argparse.ArgumentParser(
        description="Classify observation-noise envelope by mechanism layer."
    )
    parser.add_argument(
        "--evidence",
        type=str,
        default=str(DEFAULT_EVIDENCE),
        help="Path to the envelope summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/context/evidence/issue_2782_observation_noise_mechanisms",
        help="Output directory for mechanism reports.",
    )
    args = parser.parse_args()

    evidence_path = pathlib.Path(args.evidence)
    if not evidence_path.exists():
        raise SystemExit(f"Evidence file not found: {evidence_path}")

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    envelope = _load_evidence(evidence_path)

    fixture_meta = {
        "trace_path": envelope.get("fixture", {}).get("trace_path", ""),
        "scenario_id": envelope.get("fixture", {}).get("scenario_id", ""),
        "first_visible_step": envelope.get("fixture", {}).get("first_visible_step"),
        "source_path": _relative_to_repo(evidence_path),
    }

    classified = classify_all_conditions(
        envelope["conditions"],
        fixture_meta,
    )

    repo_head = _git_head()
    generated_at = datetime.datetime.now(datetime.UTC).isoformat()
    command = (
        f"uv run python scripts/tools/classify_observation_noise_mechanisms.py "
        f"--evidence {args.evidence} --output-dir {args.output_dir}"
    )

    repro = {
        "issue": 2782,
        "generated_at_utc": generated_at,
        "command": command,
        "repo_head": repo_head,
    }

    report = _build_mechanism_report(classified, fixture_meta, repro)

    json_path = output_dir / "mechanism_report.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    md_content = _generate_markdown(classified, fixture_meta, repro)
    md_path = output_dir / "README.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md_content)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    dist = report["summary"]["mechanism_distribution"]
    for label in sorted(dist):
        print(f"  [{label:>50s}]  {dist[label]} condition(s)")


if __name__ == "__main__":
    main()
