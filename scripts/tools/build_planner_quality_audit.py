#!/usr/bin/env python3
"""Build a planner quality and paper-faithfulness audit from benchmark campaigns."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _repository_root() -> Path:
    """Resolve the repository root from this script location."""
    return Path(__file__).resolve().parents[2]


def _planner_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = summary.get("planner_rows")
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = row.get("planner_key")
        if isinstance(key, str) and key:
            out[key] = row
    return out


def _runs_by_planner(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runs = summary.get("runs")
    if not isinstance(runs, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        planner = run.get("planner") or {}
        key = planner.get("key")
        if isinstance(key, str) and key:
            out[key] = run
    return out


def _format_source_list(value: Any) -> str:
    """Format one or more source paths for markdown output."""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if isinstance(item, str) and item)
    if isinstance(value, str):
        return value
    return ""


def _load_termination_counts(run_entry: dict[str, Any], campaign_root: Path) -> dict[str, int]:
    episodes_path_value = run_entry.get("episodes_path")
    if not isinstance(episodes_path_value, str) or not episodes_path_value:
        return {}
    episodes_path = Path(episodes_path_value)
    if not episodes_path.is_absolute():
        campaign_candidate = (campaign_root / episodes_path).resolve()
        repo_candidate = (_repository_root() / episodes_path).resolve()
        if campaign_candidate.exists():
            episodes_path = campaign_candidate
        elif repo_candidate.exists():
            episodes_path = repo_candidate
        else:
            episodes_path = campaign_candidate
    episodes = _read_jsonl(episodes_path)
    counts: Counter[str] = Counter()
    for row in episodes:
        reason = row.get("termination_reason")
        counts[str(reason or "unknown")] += 1
    return dict(counts)


def _primary_failure_mode(termination_counts: dict[str, int]) -> str:
    non_success = {k: v for k, v in termination_counts.items() if k != "success" and v > 0}
    if not non_success:
        return "none"
    return max(sorted(non_success), key=lambda key: non_success[key])


def _headline_recommendation_bucket(value: str) -> str:
    mapping = {
        "keep": "headline_suite",
        "control-only": "control_only",
        "non-headline": "non_headline",
    }
    return mapping.get(value, "non_headline")


def _build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Planner Quality Audit",
        "",
        f"- Hard matrix campaign: `{payload['hard_matrix_campaign_id']}`",
        f"- Sanity campaign: `{payload['sanity_campaign_id']}`",
        f"- Policy version: `{payload['policy_version']}`",
        "",
        "## Decision Table",
        "",
        "| planner | classification | headline | hard success | hard collision | hard max_steps | hard SNQI | sanity success | runtime(s) | primary failure | interpretation |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["planner_audit_rows"]:
        lines.append(
            "| "
            f"{row['planner_key']} | {row['classification']} | {row['headline_recommendation']} | "
            f"{row['hard_matrix']['success_mean']:.4f} | {row['hard_matrix']['collisions_mean']:.4f} | "
            f"{row['hard_matrix']['max_steps_rate']:.4f} | {row['hard_matrix']['snqi_mean']:.4f} | "
            f"{row['sanity_matrix']['success_mean']:.4f} | {row['hard_matrix']['runtime_sec']:.2f} | "
            f"{row['hard_matrix']['primary_failure_mode']} | {row['interpret_result_as']} |"
        )
    lines.extend(
        [
            "",
            "## Headline Suite Recommendation",
            "",
            f"- Headline suite: `{', '.join(payload['headline_suite']['headline_suite'])}`",
            f"- Control only: `{', '.join(payload['headline_suite']['control_only'])}`",
            f"- Non-headline: `{', '.join(payload['headline_suite']['non_headline'])}`",
            "",
            "## Paper-Faithfulness Notes",
            "",
        ]
    )
    for row in payload["planner_audit_rows"]:
        lines.extend(
            [
                f"### {row['planner_key']}",
                "",
                f"- Paper-reference family: {row['paper_reference_family']}",
                f"- What the paper/repo evaluates: {row['paper_evaluates']}",
                f"- What we currently implement: {row['current_implementation']}",
                f"- Missing for a fair comparison: {row['missing_for_fair_comparison']}",
                f"- Interpretation: {row['interpret_result_as']}",
                "",
            ]
        )
    external_candidates = payload.get("external_candidates", [])
    if external_candidates:
        lines.extend(["## External Candidate Parity Gaps", ""])
        for item in external_candidates:
            lines.extend(
                [
                    f"### {item['label']}",
                    "",
                    f"- Closest local proxies: `{', '.join(item['closest_local_proxies'])}`",
                    f"- Observation-contract gap: {item['observation_contract_gap']}",
                    f"- Action-contract gap: {item['action_contract_gap']}",
                    f"- Scenario-assumption gap: {item['scenario_assumption_gap']}",
                    f"- Evaluation-harness gap: {item['evaluation_harness_gap']}",
                    f"- Interpretation: {item['interpretation']}",
                    "",
                ]
            )
        lines.extend(["## External Reproduction Priority", ""])
    for item in payload["reproduction_priority"]:
        lines.extend(
            [
                f"### {item['label']}",
                "",
                f"- Rationale: {item['rationale']}",
                f"- Exact policy/config source: `{_format_source_list(item['exact_policy_or_config_source'])}`",
                f"- Expected observation/action contract: {item['expected_observation_action_contract']}",
                f"- Expected scenario/eval protocol: {item['expected_scenario_and_eval_protocol']}",
                f"- Wrapper strategy: {item['wrapper_strategy']}",
                f"- Acceptance threshold: {item['acceptance_threshold']}",
                "",
            ]
        )
    return "\n".join(lines)


def build_audit(
    hard_matrix_root: Path, sanity_root: Path, parity_config_path: Path
) -> dict[str, Any]:
    """Build planner audit payload from corrected hard-matrix and sanity campaigns."""
    hard_summary = _read_json(hard_matrix_root / "reports" / "campaign_summary.json")
    sanity_summary = _read_json(sanity_root / "reports" / "campaign_summary.json")
    parity_config = _read_yaml(parity_config_path)

    hard_rows = _planner_rows(hard_summary)
    sanity_rows = _planner_rows(sanity_summary)
    hard_runs = _runs_by_planner(hard_summary)
    planner_config = parity_config.get("planners") or {}
    external_candidates = parity_config.get("external_candidates") or []
    reproduction_priority = parity_config.get("reproduction_priority") or []

    planner_keys = sorted(set(hard_rows) | set(sanity_rows))
    planner_audit_rows: list[dict[str, Any]] = []
    headline_suite = {"headline_suite": [], "control_only": [], "non_headline": []}

    for planner_key in planner_keys:
        hard_row = hard_rows.get(planner_key, {})
        sanity_row = sanity_rows.get(planner_key, {})
        run_entry = hard_runs.get(planner_key, {})
        term_counts = _load_termination_counts(run_entry, hard_matrix_root)
        total_episodes = sum(term_counts.values())
        max_steps_rate = 0.0
        if total_episodes > 0:
            max_steps_rate = float(term_counts.get("max_steps", 0) / total_episodes)
        policy = planner_config.get(planner_key) or {}
        headline_recommendation = str(policy.get("headline_recommendation", "non-headline"))
        headline_suite[_headline_recommendation_bucket(headline_recommendation)].append(planner_key)
        planner_audit_rows.append(
            {
                "planner_key": planner_key,
                "classification": str(
                    policy.get("classification", "weak but honest local implementation")
                ),
                "headline_recommendation": headline_recommendation,
                "paper_reference_family": str(policy.get("paper_reference_family", "unknown")),
                "paper_evaluates": str(policy.get("paper_evaluates", "")),
                "current_implementation": str(policy.get("current_implementation", "")),
                "missing_for_fair_comparison": str(policy.get("missing_for_fair_comparison", "")),
                "interpret_result_as": str(
                    policy.get("interpret_result_as", "implementation-level local evidence")
                ),
                "hard_matrix": {
                    "success_mean": _safe_float(hard_row.get("success_mean")) or 0.0,
                    "collisions_mean": _safe_float(hard_row.get("collisions_mean")) or 0.0,
                    "snqi_mean": _safe_float(hard_row.get("snqi_mean")) or 0.0,
                    "runtime_sec": _safe_float(hard_row.get("runtime_sec")) or 0.0,
                    "episodes_per_second": _safe_float(hard_row.get("episodes_per_second")) or 0.0,
                    "max_steps_rate": max_steps_rate,
                    "termination_reason_counts": term_counts,
                    "primary_failure_mode": _primary_failure_mode(term_counts),
                },
                "sanity_matrix": {
                    "success_mean": _safe_float(sanity_row.get("success_mean")) or 0.0,
                    "collisions_mean": _safe_float(sanity_row.get("collisions_mean")) or 0.0,
                    "runtime_sec": _safe_float(sanity_row.get("runtime_sec")) or 0.0,
                    "status": str(sanity_row.get("status", "missing")),
                },
            }
        )

    return {
        "schema_version": "planner-quality-audit.v1",
        "policy_version": str(parity_config.get("version", "unknown")),
        "hard_matrix_campaign_id": str(
            (hard_summary.get("campaign") or {}).get("campaign_id", hard_matrix_root.name)
        ),
        "hard_matrix_campaign_root": str(hard_matrix_root),
        "sanity_campaign_id": str(
            (sanity_summary.get("campaign") or {}).get("campaign_id", sanity_root.name)
        ),
        "sanity_campaign_root": str(sanity_root),
        "planner_audit_rows": planner_audit_rows,
        "headline_suite": headline_suite,
        "external_candidates": external_candidates,
        "reproduction_priority": reproduction_priority,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-matrix-root", type=Path, required=True)
    parser.add_argument("--sanity-matrix-root", type=Path, required=True)
    parser.add_argument(
        "--parity-config",
        type=Path,
        default=Path("configs/benchmarks/planner_quality_audit_v1.yaml"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main() -> int:
    """CLI entry point."""
    args = _build_parser().parse_args()
    payload = build_audit(
        args.hard_matrix_root.resolve(),
        args.sanity_matrix_root.resolve(),
        args.parity_config.resolve(),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(_build_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {"output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
