#!/usr/bin/env python3
"""Run an analysis-only counterfactual pair for Issue #2924.

This runner consumes already tracked compact observations plus mechanism_trace.v1 inputs.
It does not run a benchmark or simulation. Fallback, degraded, failed, partial-failure, and
not_available rows fail closed before any survived/falsified verdict is emitted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jsonschema
import yaml

from robot_sf.benchmark.counterfactual_pair import (
    PairHypothesis,
    evaluate_counterfactual_pair,
)
from robot_sf.benchmark.mechanism_trace import generate_mechanism_trace_report

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMA_VERSION = "counterfactual_pair_run.v1"
MANIFEST_SCHEMA_VERSION = "counterfactual_pair.v1"
OBSERVATION_SCHEMA_VERSION = "counterfactual_pair_observation.v1"
_INVARIANT_FIELDS = ("scenario", "seed", "planner", "artifact")
_ACTIVE_CLASSIFICATIONS = {"active-but-irrelevant", "slice-local", "revise", "stop"}
_EFFECT_CLASSIFICATIONS = {"slice-local", "revise", "stop"}
_FAIL_CLOSED_VALUES = {
    "fallback",
    "degraded",
    "partial-failure",
    "failed",
    "not_available",
    "unknown",
}


class CounterfactualPairRunnerError(ValueError):
    """Raised when the Issue #2924 runner cannot produce valid analysis evidence."""


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/research/issue_2924_counterfactual_pair.yaml"),
        help="counterfactual_pair.v1 manifest YAML/JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON result path. Defaults to stdout only.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Optional Markdown report path.",
    )
    parser.add_argument(
        "--panel-output-dir",
        type=Path,
        help=(
            "Optional directory for deterministic trajectory panels. Requires both manifest "
            "configs to provide trace_export_path."
        ),
    )
    return parser


def load_counterfactual_pair_schema() -> dict[str, Any]:
    """Load the counterfactual_pair.v1 JSON schema."""

    path = (
        Path(__file__).resolve().parents[2]
        / "robot_sf/benchmark/schemas/counterfactual_pair.v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def validate_counterfactual_pair_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a counterfactual_pair.v1 manifest."""

    schema = load_counterfactual_pair_schema()
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(payload)
    except jsonschema.ValidationError as exc:
        raise CounterfactualPairRunnerError(
            f"manifest schema validation failed: {exc.message}"
        ) from exc
    return payload


def _load_mapping(path: Path) -> dict[str, Any]:
    """Load a JSON or YAML mapping."""

    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CounterfactualPairRunnerError(f"failed to read {path}: {exc}") from exc
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CounterfactualPairRunnerError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CounterfactualPairRunnerError(f"expected mapping payload in {path}")
    return payload


def _resolve_path(raw_path: str, *, manifest_dir: Path) -> Path:
    """Resolve a manifest path as absolute, repo-relative/cwd-relative, then manifest-relative."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return manifest_dir / path


def _assert_invariants(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return matched invariants or raise with the mismatched field name."""

    expected = manifest["invariant_fields"]
    baseline = manifest["baseline_config"]["invariant_fields"]
    intervention = manifest["intervention_config"]["invariant_fields"]
    for field in _INVARIANT_FIELDS:
        values = {
            "manifest": expected.get(field),
            "baseline": baseline.get(field),
            "intervention": intervention.get(field),
        }
        if len(set(values.values())) != 1:
            raise CounterfactualPairRunnerError(
                f"invariant mismatch for {field}: "
                f"manifest={values['manifest']!r}, baseline={values['baseline']!r}, "
                f"intervention={values['intervention']!r}"
            )
    return {field: expected[field] for field in _INVARIANT_FIELDS}


def _assert_observation_invariants(
    *,
    role: str,
    config: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> None:
    """Ensure a loaded observation agrees with its manifest run-config invariants."""

    observed = observation.get("invariant_fields")
    if not isinstance(observed, Mapping):
        raise CounterfactualPairRunnerError(f"{role} observation is missing invariant_fields")
    expected = config["invariant_fields"]
    for field in _INVARIANT_FIELDS:
        if observed.get(field) != expected.get(field):
            raise CounterfactualPairRunnerError(
                f"invariant mismatch for {role}.{field}: "
                f"manifest={expected.get(field)!r}, observation={observed.get(field)!r}"
            )


def _fail_closed_status_reasons(observation: Mapping[str, Any]) -> list[str]:
    """Return fail-closed status reasons for an observation."""

    reasons: list[str] = []
    for field in ("row_status", "execution_mode", "readiness_status", "availability_status"):
        value = observation.get(field)
        if isinstance(value, str) and value in _FAIL_CLOSED_VALUES:
            reasons.append(f"{field}={value}")
    return reasons


def _assert_success_capable_observation(role: str, observation: Mapping[str, Any]) -> None:
    """Fail closed for fallback, degraded, unavailable, failed, or partial rows."""

    schema_version = observation.get("schema_version")
    if schema_version != OBSERVATION_SCHEMA_VERSION:
        raise CounterfactualPairRunnerError(
            f"{role} observation schema_version must be {OBSERVATION_SCHEMA_VERSION!r}"
        )
    reasons = _fail_closed_status_reasons(observation)
    if reasons:
        raise CounterfactualPairRunnerError(
            f"fail-closed {role} observation status: {', '.join(reasons)}"
        )
    metrics = observation.get("metrics")
    if not isinstance(metrics, Mapping):
        raise CounterfactualPairRunnerError(f"{role} observation is missing metrics")


def _activation_summary(
    trace_payload: dict[str, Any], *, expected_mechanism: str
) -> dict[str, Any]:
    """Build an activation summary from mechanism_trace.v1 rows."""

    report = generate_mechanism_trace_report(trace_payload)
    rows = [row for row in report["rows"] if row.get("mechanism_id") == expected_mechanism]
    active_rows = [row for row in rows if row.get("classification") in _ACTIVE_CLASSIFICATIONS]
    effect_rows = [row for row in rows if row.get("classification") in _EFFECT_CLASSIFICATIONS]
    return {
        "mechanism_id": expected_mechanism,
        "row_count": len(rows),
        "active_count": len(active_rows),
        "effect_count": len(effect_rows),
        "activated": bool(active_rows),
        "classifications": {
            state: sum(1 for row in rows if row.get("classification") == state)
            for state in sorted(_ACTIVE_CLASSIFICATIONS | {"inactive"})
        },
    }


def _observation_for_evaluator(
    observation: Mapping[str, Any],
    *,
    activation: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the shape expected by robot_sf.benchmark.counterfactual_pair."""

    return {
        "mechanism_activated": bool(activation["activated"]),
        "metrics": observation["metrics"],
    }


def _round_delta(value: float) -> float:
    """Round report deltas without hiding small exact fixture changes."""

    return round(float(value), 10)


def _git_commit() -> str:
    """Return the current short commit for provenance when available."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _generate_panels_if_requested(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    panel_output_dir: Path | None,
) -> dict[str, Any] | None:
    """Generate trajectory panels when requested and trace exports are configured."""

    if panel_output_dir is None:
        return None
    trace_paths: list[Path] = []
    for role in ("baseline_config", "intervention_config"):
        raw_path = manifest[role].get("trace_export_path")
        if not raw_path:
            raise CounterfactualPairRunnerError(
                f"--panel-output-dir requires {role}.trace_export_path"
            )
        trace_paths.append(_resolve_path(str(raw_path), manifest_dir=manifest_path.parent))

    from robot_sf.benchmark.trajectory_panels import generate_trajectory_panel_bundle

    command = (
        "python scripts/analysis/run_counterfactual_pair_issue_2924.py "
        f"--manifest {manifest_path.as_posix()} --panel-output-dir {panel_output_dir.as_posix()}"
    )
    bundle = generate_trajectory_panel_bundle(
        trace_paths,
        output_dir=panel_output_dir,
        command=command,
        commit=_git_commit(),
    )
    return {
        "output_dir": bundle.output_dir.as_posix(),
        "selection_csv": bundle.selection_csv.as_posix(),
        "manifest_path": bundle.manifest_path.as_posix(),
        "captions_path": bundle.captions_path.as_posix(),
        "artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "png_path": artifact.png_path.as_posix(),
                "pdf_path": artifact.pdf_path.as_posix(),
                "category": artifact.category,
                "panel_type": artifact.panel_type,
            }
            for artifact in bundle.artifacts
        ],
    }


def _write_markdown_report(payload: Mapping[str, Any], path: Path) -> None:
    """Write a compact analysis-only Markdown report."""

    outcome = payload["outcome_delta"]
    lines = [
        "# Issue #2924 Counterfactual Pair Report",
        "",
        "## Boundary",
        "",
        "- Evidence tier: `analysis_only`",
        f"- Claim boundary: `{payload['claim_boundary']}`",
        "- Fallback/degraded/not_available rows: fail closed before verdict calculation.",
        "",
        "## Pair",
        "",
        f"- Pair id: `{payload['pair_id']}`",
        f"- Scenario: `{payload['invariant_fields']['scenario']}`",
        f"- Seed: `{payload['invariant_fields']['seed']}`",
        f"- Planner: `{payload['invariant_fields']['planner']}`",
        f"- Artifact invariant: `{payload['invariant_fields']['artifact']}`",
        "",
        "## Result",
        "",
        f"- Expected mechanism: `{payload['expected_mechanism']}`",
        f"- Activation delta: `{payload['activation_delta']['active_count_delta']}` active rows",
        (
            f"- Outcome delta: `{outcome['metric']}` "
            f"`{outcome['baseline']}` -> `{outcome['intervention']}` "
            f"(delta `{outcome['delta']}`; expected `{outcome['expected_direction']}`)"
        ),
        f"- Hypothesis verdict: `{payload['hypothesis_verdict']}`",
        f"- Reason: {payload['pair_result']['reason']}",
        "",
    ]
    trace_panel = payload.get("trace_panel")
    if isinstance(trace_panel, Mapping):
        lines.extend(
            [
                "## Trace Panels",
                "",
                f"- Panel manifest: `{trace_panel['manifest_path']}`",
                f"- Captions: `{trace_panel['captions_path']}`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Trace Panels",
                "",
                "- Not generated by this command. The configured trace exports are panel-compatible, "
                "but no `--panel-output-dir` was supplied.",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_counterfactual_pair(
    *,
    manifest_path: Path,
    output_path: Path | None = None,
    markdown_output_path: Path | None = None,
    panel_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one counterfactual_pair.v1 manifest and optionally write reports."""

    manifest = validate_counterfactual_pair_manifest(_load_mapping(manifest_path))
    invariants = _assert_invariants(manifest)
    expected_mechanism = str(manifest["expected_mechanism"])
    direction = manifest["expected_metric_direction"]
    metric = str(direction["metric"])
    expected_direction = str(direction["direction"])
    min_delta = float(direction.get("min_delta", 0.0))

    loaded: dict[str, dict[str, Any]] = {}
    activations: dict[str, dict[str, Any]] = {}
    for role in ("baseline", "intervention"):
        config = manifest[f"{role}_config"]
        result_path = _resolve_path(str(config["result_path"]), manifest_dir=manifest_path.parent)
        trace_path = _resolve_path(
            str(config["mechanism_trace_path"]),
            manifest_dir=manifest_path.parent,
        )
        observation = _load_mapping(result_path)
        _assert_observation_invariants(role=role, config=config, observation=observation)
        _assert_success_capable_observation(role, observation)
        trace_payload = _load_mapping(trace_path)
        loaded[role] = observation
        activations[role] = _activation_summary(
            trace_payload,
            expected_mechanism=expected_mechanism,
        )

    hypothesis = PairHypothesis(
        expected_mechanism=expected_mechanism,
        outcome_metric=metric,
        expected_direction=expected_direction,
    )
    pair_result = evaluate_counterfactual_pair(
        _observation_for_evaluator(loaded["baseline"], activation=activations["baseline"]),
        _observation_for_evaluator(
            loaded["intervention"],
            activation=activations["intervention"],
        ),
        hypothesis,
        min_outcome_delta=min_delta,
    )
    pair_result_payload = pair_result.to_dict()
    outcome_delta = {
        "metric": metric,
        "baseline": pair_result.outcome_baseline,
        "intervention": pair_result.outcome_intervention,
        "delta": _round_delta(pair_result.outcome_delta),
        "expected_direction": expected_direction,
        "min_delta": min_delta,
    }
    activation_delta = {
        "active_count_delta": (
            activations["intervention"]["active_count"] - activations["baseline"]["active_count"]
        ),
        "effect_count_delta": (
            activations["intervention"]["effect_count"] - activations["baseline"]["effect_count"]
        ),
        "activated_bool_delta": pair_result.activation_delta,
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "pair_id": manifest["pair_id"],
        "evidence_tier": manifest["evidence_tier"],
        "claim_boundary": manifest["claim_boundary"],
        "invariant_fields": invariants,
        "expected_mechanism": expected_mechanism,
        "baseline": {
            "config_id": manifest["baseline_config"]["config_id"],
            "activation": activations["baseline"],
            "metrics": loaded["baseline"]["metrics"],
        },
        "intervention": {
            "config_id": manifest["intervention_config"]["config_id"],
            "activation": activations["intervention"],
            "metrics": loaded["intervention"]["metrics"],
        },
        "activation_delta": activation_delta,
        "outcome_delta": outcome_delta,
        "hypothesis_verdict": pair_result.verdict,
        "pair_result": pair_result_payload,
        "trace_panel": _generate_panels_if_requested(
            manifest=manifest,
            manifest_path=manifest_path,
            panel_output_dir=panel_output_dir,
        ),
        "diagnostic_limitations": manifest.get("diagnostic_limitations", []),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if markdown_output_path is not None:
        _write_markdown_report(payload, markdown_output_path)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Issue #2924 counterfactual-pair CLI."""

    args = _build_parser().parse_args(argv)
    try:
        payload = run_counterfactual_pair(
            manifest_path=args.manifest,
            output_path=args.output,
            markdown_output_path=args.markdown_output,
            panel_output_dir=args.panel_output_dir,
        )
    except CounterfactualPairRunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.output is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
