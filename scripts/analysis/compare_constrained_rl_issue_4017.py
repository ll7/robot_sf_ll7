"""Build the issue #4017 constrained-RL diagnostic comparison report.

The report consumes paired CPU-smoke training manifests from
``scripts/training/train_constrained_rl.py`` plus the constrained trace JSONL. It
does not run a benchmark campaign or promote paper-facing safety claims.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import yaml

_CONFIG_KEYS = {
    "schema_version",
    "issue",
    "evidence_tier",
    "claim_boundary",
    "baseline_manifest",
    "constrained_manifest",
    "output_json",
    "output_markdown",
}
_EXPECTED_SCHEMA = "issue_4017.constrained_rl_diagnostic.v1"
_EXPECTED_EVIDENCE_TIER = "diagnostic-only"


def build_report_from_config(config_path: str | Path) -> dict[str, object]:
    """Build and write the diagnostic comparison report described by ``config_path``."""

    config_path = Path(config_path).resolve()
    config = _load_config(config_path)
    baseline_path = _resolve_path(config_path, config["baseline_manifest"])
    constrained_path = _resolve_path(config_path, config["constrained_manifest"])
    output_json = _resolve_output_path(config_path, config["output_json"])
    output_markdown = _resolve_output_path(config_path, config["output_markdown"])

    baseline_manifest = _read_json_object(baseline_path, "baseline_manifest")
    constrained_manifest = _read_json_object(constrained_path, "constrained_manifest")
    baseline = _run_summary(baseline_manifest, role="baseline", manifest_path=baseline_path)
    constrained = _run_summary(
        constrained_manifest,
        role="constrained",
        manifest_path=constrained_path,
    )
    trace_summary = _constraint_trace_summary(constrained_manifest, constrained_path)

    blockers = _comparison_blockers(baseline, constrained, trace_summary)
    report = {
        "schema_version": _EXPECTED_SCHEMA,
        "issue": 4017,
        "evidence_tier": _EXPECTED_EVIDENCE_TIER,
        "claim_boundary": str(config["claim_boundary"]),
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "baseline_manifest": str(baseline_path),
            "constrained_manifest": str(constrained_path),
            "constrained_trace": trace_summary["trace_path"],
        },
        "paired_seed_count": 1 if baseline["seed"] == constrained["seed"] else 0,
        "fallback_or_degraded": bool(
            baseline["fallback_or_degraded"] or constrained["fallback_or_degraded"]
        ),
        "baseline": baseline,
        "constrained": constrained,
        "constraint_trace": trace_summary,
        "constraint_effect": _constraint_effect(baseline, constrained, trace_summary),
        "blockers": blockers,
        "status": "diagnostic_ready" if not blockers else "diagnostic_blocked",
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_markdown_report(report), encoding="utf-8")
    return report


def _load_config(config_path: Path) -> dict[str, object]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("comparison config must be a mapping.")
    config = dict(raw)
    unknown = sorted(set(config) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(f"comparison config contains unsupported keys: {', '.join(unknown)}")
    missing = sorted(_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"comparison config missing required keys: {', '.join(missing)}")
    if config["schema_version"] != _EXPECTED_SCHEMA:
        raise ValueError(f"schema_version must be {_EXPECTED_SCHEMA}.")
    if int(config["issue"]) != 4017:
        raise ValueError("comparison config issue must be 4017.")
    if config["evidence_tier"] != _EXPECTED_EVIDENCE_TIER:
        raise ValueError(f"evidence_tier must be {_EXPECTED_EVIDENCE_TIER}.")
    return config


def _run_summary(
    manifest: Mapping[str, object],
    *,
    role: str,
    manifest_path: Path,
) -> dict[str, object]:
    required = (
        "policy_id",
        "algorithm",
        "evidence_tier",
        "claim_boundary",
        "seed",
        "total_timesteps",
        "constraints_enabled",
        "fallback_or_degraded",
    )
    missing = [key for key in required if key not in manifest]
    runtime_seconds = manifest.get("runtime_seconds")
    runtime_status = "recorded" if runtime_seconds is not None else "missing_from_manifest"
    if runtime_seconds is not None:
        runtime_seconds = _finite_float(runtime_seconds)
        if runtime_seconds is None:
            runtime_status = "non_finite_ignored"
    return {
        "role": role,
        "manifest_path": str(manifest_path),
        "policy_id": manifest.get("policy_id"),
        "algorithm": manifest.get("algorithm"),
        "evidence_tier": manifest.get("evidence_tier"),
        "claim_boundary": manifest.get("claim_boundary"),
        "dry_run": bool(manifest.get("dry_run", False)),
        "seed": manifest.get("seed"),
        "total_timesteps": manifest.get("total_timesteps"),
        "num_envs": manifest.get("num_envs"),
        "device": manifest.get("device"),
        "constraints_enabled": bool(manifest.get("constraints_enabled", False)),
        "constraints": list(manifest.get("constraints") or ()),
        "runtime_seconds": runtime_seconds,
        "runtime_status": runtime_status,
        "fallback_or_degraded": bool(manifest.get("fallback_or_degraded", True)),
        "missing_fields": missing,
    }


def _constraint_trace_summary(
    constrained_manifest: Mapping[str, object],
    constrained_manifest_path: Path,
) -> dict[str, object]:
    trace_path_raw = constrained_manifest.get("constraint_trace_path")
    if not trace_path_raw:
        raise ValueError("constrained manifest missing constraint_trace_path.")
    trace_path = Path(str(trace_path_raw))
    if not trace_path.is_absolute():
        trace_path = (constrained_manifest_path.parent / trace_path).resolve()
        if not trace_path.exists():
            trace_path = (Path.cwd() / str(trace_path_raw)).resolve()
    if not trace_path.exists():
        raise FileNotFoundError(f"constraint trace does not exist: {trace_path}")

    constraints = list(constrained_manifest.get("constraints") or ())
    by_name = {
        str(item.get("name")): {
            "budget_per_episode": float(item.get("budget_per_episode", 0.0)),
            "source_key": item.get("source_key"),
            "multiplier_init": float(item.get("multiplier_init", 0.0)),
            "multiplier_lr": float(item.get("multiplier_lr", 0.0)),
            "multiplier_max": float(item.get("multiplier_max", 0.0)),
        }
        for item in constraints
        if isinstance(item, Mapping)
    }
    records = _read_jsonl_objects(trace_path)
    summaries = {
        name: {
            "source_key": spec["source_key"],
            "budget_per_episode": spec["budget_per_episode"],
            "episode_count": 0,
            "positive_violation_count": 0,
            "max_violation": None,
            "latest_violation": None,
            "total_cost": 0.0,
            "latest_multiplier": spec["multiplier_init"],
            "multiplier_init": spec["multiplier_init"],
            "multiplier_lr": spec["multiplier_lr"],
            "multiplier_max": spec["multiplier_max"],
        }
        for name, spec in by_name.items()
    }
    trajectory = []
    for record in records:
        costs = _float_mapping(record.get("costs", {}))
        violations = _float_mapping(record.get("violations", {}))
        multipliers = _float_mapping(record.get("multipliers_after_update", {}))
        trajectory.append(
            {
                "timesteps": record.get("timesteps"),
                "episode_steps": record.get("episode_steps"),
                "violations": violations,
                "multipliers_after_update": multipliers,
            }
        )
        for name, summary in summaries.items():
            violation = violations.get(name, 0.0)
            summary["episode_count"] = int(summary["episode_count"]) + 1
            summary["positive_violation_count"] = int(summary["positive_violation_count"]) + int(
                violation > 0.0
            )
            previous_max = summary["max_violation"]
            summary["max_violation"] = (
                violation if previous_max is None else max(float(previous_max), violation)
            )
            summary["latest_violation"] = violation
            summary["total_cost"] = float(summary["total_cost"]) + costs.get(name, 0.0)
            if name in multipliers:
                summary["latest_multiplier"] = multipliers[name]

    return {
        "trace_path": str(trace_path),
        "record_count": len(records),
        "constraints": summaries,
        "multiplier_trajectory": trajectory,
    }


def _constraint_effect(
    baseline: Mapping[str, object],
    constrained: Mapping[str, object],
    trace_summary: Mapping[str, object],
) -> dict[str, object]:
    constraints = trace_summary["constraints"]
    assert isinstance(constraints, Mapping)
    violated = [
        name
        for name, summary in constraints.items()
        if isinstance(summary, Mapping) and int(summary["positive_violation_count"]) > 0
    ]
    multiplier_changed = [
        name
        for name, summary in constraints.items()
        if isinstance(summary, Mapping)
        and float(summary["latest_multiplier"]) != float(summary["multiplier_init"])
    ]
    runtime_delta = None
    if baseline["runtime_seconds"] is not None and constrained["runtime_seconds"] is not None:
        runtime_delta = float(constrained["runtime_seconds"]) - float(baseline["runtime_seconds"])
    return {
        "interpretation": "diagnostic_only",
        "matched_seed": baseline["seed"] == constrained["seed"],
        "matched_total_timesteps": baseline["total_timesteps"] == constrained["total_timesteps"],
        "constrained_runtime_delta_seconds": runtime_delta,
        "budget_violation_constraints": violated,
        "multiplier_changed_constraints": multiplier_changed,
        "benchmark_safety_claim": False,
    }


def _comparison_blockers(
    baseline: Mapping[str, object],
    constrained: Mapping[str, object],
    trace_summary: Mapping[str, object],
) -> list[str]:
    blockers = []
    for summary in (baseline, constrained):
        blockers.extend(_run_blockers(summary))
    if baseline["seed"] != constrained["seed"]:
        blockers.append("baseline and constrained manifests use different seeds")
    if baseline["total_timesteps"] != constrained["total_timesteps"]:
        blockers.append("baseline and constrained manifests use different total_timesteps")
    if not constrained["constraints_enabled"]:
        blockers.append("constrained manifest has constraints_enabled=false")
    if baseline["constraints_enabled"]:
        blockers.append("baseline manifest has constraints_enabled=true")
    if int(trace_summary["record_count"]) == 0:
        blockers.append("constrained trace has no completed episode multiplier records")
    return blockers


def _run_blockers(summary: Mapping[str, object]) -> list[str]:
    role = summary["role"]
    blockers = [f"{role} manifest missing {field_name}" for field_name in summary["missing_fields"]]
    if summary["fallback_or_degraded"]:
        blockers.append(f"{role} manifest reports fallback_or_degraded=true")
    if summary["evidence_tier"] != "smoke":
        blockers.append(f"{role} manifest evidence_tier is not smoke")
    if summary["runtime_status"] != "recorded":
        blockers.append(f"{role} manifest missing runtime_seconds")
    return blockers


def _markdown_report(report: Mapping[str, object]) -> str:
    baseline = report["baseline"]
    constrained = report["constrained"]
    trace = report["constraint_trace"]
    assert isinstance(baseline, Mapping)
    assert isinstance(constrained, Mapping)
    assert isinstance(trace, Mapping)
    blockers = list(report["blockers"])
    lines = [
        "# Issue #4017 constrained-RL diagnostic comparison",
        "",
        "This is a diagnostic smoke comparison only. It is not benchmark, paper-grade, "
        "or dissertation safety evidence.",
        "",
        f"- Status: `{report['status']}`",
        f"- Evidence tier: `{report['evidence_tier']}`",
        f"- Paired seed count: `{report['paired_seed_count']}`",
        f"- Fallback or degraded: `{str(report['fallback_or_degraded']).lower()}`",
        "",
        "## Runs",
        "",
        "| role | policy_id | algorithm | seed | timesteps | runtime_seconds | degraded |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        _run_markdown_row(baseline),
        _run_markdown_row(constrained),
        "",
        "## Constraint Trace",
        "",
        f"- Trace records: `{trace['record_count']}`",
        "",
        "| constraint | budget | positive violations | max violation | latest multiplier |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    constraints = trace["constraints"]
    assert isinstance(constraints, Mapping)
    for name, summary in constraints.items():
        assert isinstance(summary, Mapping)
        lines.append(
            "| {name} | {budget} | {count} | {max_violation} | {multiplier} |".format(
                name=name,
                budget=summary["budget_per_episode"],
                count=summary["positive_violation_count"],
                max_violation=summary["max_violation"],
                multiplier=summary["latest_multiplier"],
            )
        )
    lines.extend(["", "## Blockers", ""])
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- None for diagnostic manifest comparison.")
    lines.append("")
    return "\n".join(lines)


def _run_markdown_row(summary: Mapping[str, object]) -> str:
    return (
        f"| {summary['role']} | {summary['policy_id']} | {summary['algorithm']} | "
        f"{summary['seed']} | {summary['total_timesteps']} | {summary['runtime_seconds']} | "
        f"{str(summary['fallback_or_degraded']).lower()} |"
    )


def _read_json_object(path: Path, field_name: str) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must contain a JSON object.")
    return dict(raw)


def _read_jsonl_objects(path: Path) -> list[dict[str, object]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}:{line_number} must contain a JSON object.")
        records.append(dict(raw))
    return records


def _float_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): numeric
        for key, raw in value.items()
        if (numeric := _finite_float(raw)) is not None
    }


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _resolve_path(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    candidates = ((config_path.parent / path).resolve(), (Path.cwd() / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_output_path(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve() if not (config_path.parent / path).is_absolute() else path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Diagnostic comparison YAML config.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison report CLI."""

    args = _parse_args(argv)
    build_report_from_config(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
