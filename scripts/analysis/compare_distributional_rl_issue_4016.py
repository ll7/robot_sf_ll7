"""Build issue #4016 distributional-RL diagnostic comparison reports.

The report compares matched QR-DQN mean-return and lower-tail risk manifests.
It is a diagnostic integration surface only: fallback/degraded rows are excluded
from evidence, and the output never promotes a benchmark or safety claim.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

_EXPECTED_SCHEMA = "issue_4016.distributional_rl_risk_comparison.v1"
_EXPECTED_EVIDENCE_TIER = "diagnostic-only"
_EXPECTED_CLAIM_BOUNDARY = "risk-selection diagnostic only; not benchmark evidence"
_CONFIG_KEYS = {
    "schema_version",
    "issue",
    "evidence_tier",
    "claim_boundary",
    "mean_manifest",
    "risk_manifest",
    "output_json",
    "output_markdown",
}
_METRIC_KEYS = (
    "success_rate",
    "collision_rate",
    "near_miss_rate",
    "mean_min_clearance",
    "mean_path_efficiency",
)


def build_report_from_config(config_path: str | Path) -> dict[str, Any]:
    """Build and write the diagnostic comparison report described by ``config_path``."""

    config_path = Path(config_path).resolve()
    config = _load_config(config_path)
    mean_path = _resolve_path(config_path, config["mean_manifest"])
    risk_path = _resolve_path(config_path, config["risk_manifest"])
    output_json = _resolve_output_path(config_path, config["output_json"])
    output_markdown = _resolve_output_path(config_path, config["output_markdown"])

    mean = _summarize_manifest(
        _read_json_object(mean_path, "mean_manifest"), mean_path, role="mean"
    )
    risk = _summarize_manifest(
        _read_json_object(risk_path, "risk_manifest"), risk_path, role="risk"
    )
    blockers = _comparison_blockers(mean, risk)
    effect = _effect_summary(mean, risk, blockers)

    report: dict[str, Any] = {
        "schema_version": _EXPECTED_SCHEMA,
        "issue": 4016,
        "evidence_tier": _EXPECTED_EVIDENCE_TIER,
        "claim_boundary": _EXPECTED_CLAIM_BOUNDARY,
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": _portable_path(config_path),
        "inputs": {
            "mean": mean,
            "risk": risk,
        },
        "matched_context": _matched_context(mean, risk),
        "effect": effect,
        "blockers": blockers,
        "fallback_degraded_rows": {
            "excluded": int(mean["fallback_or_degraded"]) + int(risk["fallback_or_degraded"]),
            "included_as_non_evidence": 0,
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_markdown.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _load_config(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("distributional RL comparison config must be a mapping")
    unknown = set(payload) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    missing = _CONFIG_KEYS - set(payload)
    if missing:
        raise ValueError(f"missing config keys: {sorted(missing)}")
    if payload["schema_version"] != _EXPECTED_SCHEMA:
        raise ValueError(f"schema_version must be {_EXPECTED_SCHEMA!r}")
    if int(payload["issue"]) != 4016:
        raise ValueError("issue must be 4016")
    if payload["evidence_tier"] != _EXPECTED_EVIDENCE_TIER:
        raise ValueError("evidence_tier must be diagnostic-only")
    if str(payload["claim_boundary"]).strip() != _EXPECTED_CLAIM_BOUNDARY:
        raise ValueError(f"claim_boundary must be {_EXPECTED_CLAIM_BOUNDARY!r}")
    return payload


def _summarize_manifest(
    manifest: Mapping[str, Any], manifest_path: Path, *, role: str
) -> dict[str, Any]:
    required = (
        "policy_id",
        "algorithm",
        "evidence_tier",
        "claim_boundary",
        "risk_objective",
        "risk_alpha",
        "seed",
        "total_timesteps",
        "metrics",
    )
    missing = [key for key in required if key not in manifest]
    metrics = _metric_summary(manifest.get("metrics"))
    diagnostics = _diagnostic_summary(manifest.get("risk_selection_diagnostics"))
    fallback_or_degraded = bool(manifest.get("fallback_or_degraded", False))
    return {
        "role": role,
        "manifest_path": _portable_path(manifest_path),
        "policy_id": manifest.get("policy_id"),
        "algorithm": manifest.get("algorithm"),
        "evidence_tier": manifest.get("evidence_tier"),
        "claim_boundary": manifest.get("claim_boundary"),
        "risk_objective": manifest.get("risk_objective"),
        "risk_alpha": _optional_finite_float(manifest.get("risk_alpha")),
        "seed": manifest.get("seed"),
        "total_timesteps": manifest.get("total_timesteps"),
        "checkpoint_path": manifest.get("checkpoint_path"),
        "fallback_or_degraded": fallback_or_degraded,
        "fallback_or_degraded_reason": manifest.get("fallback_or_degraded_reason"),
        "metrics": metrics,
        "risk_selection_diagnostics": diagnostics,
        "missing_fields": missing,
    }


def _metric_summary(payload: object) -> dict[str, float | None]:
    if not isinstance(payload, Mapping):
        return dict.fromkeys(_METRIC_KEYS)
    return {key: _optional_finite_float(payload.get(key)) for key in _METRIC_KEYS}


def _diagnostic_summary(payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {
            "record_count": 0,
            "mean_selected_count": 0,
            "risk_selected_count": 0,
            "mean_risk_disagreement_count": 0,
        }
    return {
        "record_count": _nonnegative_int(payload.get("record_count")),
        "mean_selected_count": _nonnegative_int(payload.get("mean_selected_count")),
        "risk_selected_count": _nonnegative_int(payload.get("risk_selected_count")),
        "mean_risk_disagreement_count": _nonnegative_int(
            payload.get("mean_risk_disagreement_count")
        ),
    }


def _comparison_blockers(mean: Mapping[str, Any], risk: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for summary in (mean, risk):
        role = str(summary["role"])
        missing = summary["missing_fields"]
        if missing:
            blockers.append(f"{role}_manifest_missing_fields:{','.join(missing)}")
        if summary["algorithm"] not in {"qr_dqn", "distributional_rl"}:
            blockers.append(f"{role}_algorithm_not_distributional_rl")
        if summary["evidence_tier"] != _EXPECTED_EVIDENCE_TIER:
            blockers.append(f"{role}_evidence_tier_not_diagnostic_only")
        if summary["fallback_or_degraded"]:
            blockers.append(f"{role}_fallback_or_degraded_excluded")
    if mean["risk_objective"] != "mean":
        blockers.append("mean_manifest_risk_objective_not_mean")
    if risk["risk_objective"] not in {"cvar_lower", "var_lower", "cvar_blend"}:
        blockers.append("risk_manifest_risk_objective_not_lower_tail")
    for key in ("seed", "total_timesteps", "checkpoint_path"):
        if mean.get(key) != risk.get(key):
            blockers.append(f"unmatched_{key}")
    return blockers


def _effect_summary(
    mean: Mapping[str, Any],
    risk: Mapping[str, Any],
    blockers: Sequence[str],
) -> dict[str, Any]:
    mean_metrics = mean["metrics"]
    risk_metrics = risk["metrics"]
    assert isinstance(mean_metrics, Mapping)
    assert isinstance(risk_metrics, Mapping)
    deltas = {
        f"{key}_delta": _delta(risk_metrics.get(key), mean_metrics.get(key)) for key in _METRIC_KEYS
    }
    mean_diag = mean["risk_selection_diagnostics"]
    risk_diag = risk["risk_selection_diagnostics"]
    assert isinstance(mean_diag, Mapping)
    assert isinstance(risk_diag, Mapping)
    return {
        "interpretation": "diagnostic_only",
        "comparison_status": "blocked" if blockers else "valid_diagnostic",
        "benchmark_safety_claim": False,
        "metric_deltas": deltas,
        "risk_selection": {
            "mean_records": mean_diag["record_count"],
            "risk_records": risk_diag["record_count"],
            "risk_mean_disagreement_count": risk_diag["mean_risk_disagreement_count"],
        },
    }


def _matched_context(mean: Mapping[str, Any], risk: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "matched_seed": mean["seed"] == risk["seed"],
        "matched_total_timesteps": mean["total_timesteps"] == risk["total_timesteps"],
        "matched_checkpoint_path": mean["checkpoint_path"] == risk["checkpoint_path"],
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    effect = report["effect"]
    assert isinstance(effect, Mapping)
    blockers = report["blockers"]
    assert isinstance(blockers, Sequence)
    status = effect["comparison_status"]
    lines = [
        "# Issue #4016 Distributional RL Risk Comparison",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Evidence tier: `{report['evidence_tier']}`",
        f"- Claim boundary: {report['claim_boundary']}",
        f"- Status: `{status}`",
        "- Benchmark safety claim: `false`",
        "",
        "## Blockers",
        "",
    ]
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- None for diagnostic comparison.")
    lines.extend(["", "## Metric Deltas", ""])
    metric_deltas = effect["metric_deltas"]
    assert isinstance(metric_deltas, Mapping)
    for key, value in metric_deltas.items():
        rendered = "null" if value is None else f"{value:.6g}"
        lines.append(f"- `{key}`: {rendered}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This report is diagnostic-only. It compares matched smoke manifests and excludes "
            "fallback or degraded rows; it is not benchmark or paper-facing evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OSError(f"could not read {label} at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse {label} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _resolve_path(config_path: Path, raw: object) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _resolve_output_path(config_path: Path, raw: object) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _portable_path(path: object) -> str:
    raw_path = Path(str(path))
    try:
        return str(raw_path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(raw_path)


def _optional_finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nonnegative_int(value: object) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _delta(risk_value: object, mean_value: object) -> float | None:
    risk_float = _optional_finite_float(risk_value)
    mean_float = _optional_finite_float(mean_value)
    if risk_float is None or mean_float is None:
        return None
    return risk_float - mean_float


def main(argv: Sequence[str] | None = None) -> int:
    """Run the issue #4016 diagnostic comparison report CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML comparison config path")
    args = parser.parse_args(argv)
    report = build_report_from_config(args.config)
    print(
        json.dumps(
            {"status": report["effect"]["comparison_status"], "output": report["config_path"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
