"""Validate learned-prediction readiness before training is unblocked.

This validator checks the fail-closed contract defined in
docs/context/issue_2768_learned_prediction_readiness.md. It exits with code 0 only when
all prerequisites are satisfied; otherwise it exits with code 2 and a structured error
message.

Checks performed:
  1. Readiness doc exists and contains the required section headers.
  2. Trace registry file exists (if referenced) with at least one source entry.
  3. Split manifest exists and prevents leakage.
  4. Baseline evidence exists for a named target and has ADE/FDE metrics.
  5. Calibration and transferability evidence are present.
  6. Closed-loop coupling gate is continue.
  7. Horizon and timestep recommendations are documented in the readiness contract.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "blocked"
DEFAULT_BASELINE_TARGET = "constant_velocity"
FORECAST_CALIBRATION_REPORT_SCHEMA_VERSION = "ForecastCalibrationReport.v1"
FORECAST_DATASET_SCHEMA_VERSION = "forecast_dataset.v1"
FORECAST_TRANSFERABILITY_STRESS_MATRIX_SCHEMA_VERSION = "ForecastTransferabilityStressMatrix.v1"


def _load_json(path: Path) -> dict | list | None:
    """Load a JSON file, returning None on read/parse failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_yaml(path: Path) -> dict | list | None:
    """Load a YAML file, returning None if unavailable."""
    try:
        import yaml
    except ImportError:
        return None
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except OSError:
        return None


def _check_readiness_doc(doc_path: Path) -> tuple[str, list[str]]:
    """Verify the readiness document exists and contains required sections."""
    if not doc_path.exists():
        return STATUS_BLOCKED, [f"readiness doc not found: {doc_path}"]

    content = doc_path.read_text(encoding="utf-8")
    errors: list[str] = []
    required_sections = [
        "Trace Dataset Registry",
        "Train / Validation / Test Split Metadata",
        "Target Horizon Definition",
        "Dynamic Actor Types",
        "Semantic Input Contract",
        "Calibration Metrics",
        "Collision-Relevance Metrics",
        "Deterministic / Semantic Baselines",
        "Comparison Protocol",
        "Training Block Conditions",
    ]
    for section in required_sections:
        if section not in content:
            errors.append(f"missing required section: {section}")

    required_fields = {
        "horizon_seconds",
        "horizon_steps",
        "dt_seconds",
        "horizon_recommendation",
        "timestep_recommendation",
    }
    for field in required_fields:
        if not re.search(rf"(?im)^\s*-?\s*{re.escape(field)}\s*:", content):
            errors.append(f"readiness doc missing required field: {field}")

    if errors:
        return STATUS_FAILED, errors
    return STATUS_PASSED, []


def _check_trace_registry(registry_path: Path) -> tuple[str, list[str]]:  # noqa: C901
    """Verify trace registry exists and has at least one viable source entry."""
    if not registry_path.exists():
        return STATUS_BLOCKED, [f"trace registry not found: {registry_path}"]

    data = _load_yaml(registry_path)
    if data is None:
        return STATUS_FAILED, [f"trace registry could not be loaded: {registry_path}"]

    if isinstance(data, list):
        sources = data
    elif isinstance(data, dict):
        sources = data.get("sources", [])
    else:
        return STATUS_FAILED, ["trace registry must be a dictionary or a list"]

    if not sources:
        return STATUS_FAILED, ["trace registry has no source entries"]

    viable = 0
    for src in sources:
        if not isinstance(src, dict):
            return STATUS_FAILED, ["trace registry source entries must be dictionaries"]
        count = src.get("episode_count", 0)
        actors = src.get("actor_types", [])
        if not isinstance(count, (int, float)):
            return STATUS_FAILED, ["trace registry source episode_count must be numeric"]
        if count > 0 and actors:
            viable += 1

    if viable == 0:
        return (
            STATUS_FAILED,
            ["trace registry has no source with episode_count > 0 and non-empty actor_types"],
        )
    return STATUS_PASSED, []


def _check_split_manifest_legacy(manifest: dict) -> tuple[str, list[str]]:
    """Validate a legacy split manifest using explicit fractions."""
    errors: list[str] = []
    train_frac = manifest.get("train_fraction")
    val_frac = manifest.get("validation_fraction")
    test_frac = manifest.get("test_fraction")

    if train_frac is None or val_frac is None or test_frac is None:
        return (
            STATUS_FAILED,
            ["split manifest missing train_fraction, validation_fraction, or test_fraction"],
        )

    if not all(isinstance(value, (int, float)) for value in (train_frac, val_frac, test_frac)):
        return STATUS_FAILED, ["split fractions must be numeric"]

    if any(value < 0 for value in (train_frac, val_frac, test_frac)):
        return STATUS_FAILED, ["split fractions must be non-negative"]

    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        errors.append(f"split fractions sum to {total}, expected 1.0")

    if not manifest.get("split_strategy"):
        errors.append("split manifest missing split_strategy")

    if not manifest.get("leakage_prevention"):
        errors.append("split manifest missing leakage_prevention statement")

    return (STATUS_FAILED, errors) if errors else (STATUS_PASSED, [])


def _validate_forecast_dataset_manifest(payload: dict) -> None:
    """Validate the ForecastDataset.v1 split fields needed by the readiness gate."""
    if payload.get("schema_version") != FORECAST_DATASET_SCHEMA_VERSION:
        raise ValueError("schema_version must be forecast_dataset.v1")
    if not str(payload.get("dataset_id", "")).strip():
        raise ValueError("dataset_id is required")
    if int(payload.get("example_count", 0)) <= 0:
        raise ValueError("example_count must be positive")
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("splits must be a mapping")
    missing_splits = {"train", "validation", "test"} - set(splits)
    if missing_splits:
        raise ValueError(f"splits missing required keys: {sorted(missing_splits)}")
    examples_path = payload.get("examples_path")
    if not isinstance(examples_path, str) or not examples_path.strip():
        raise ValueError("examples_path is required")
    feature_schema = payload.get("feature_schema")
    if not isinstance(feature_schema, dict) or not feature_schema:
        raise ValueError("feature_schema is required")


def _check_split_manifest_dataset(manifest: dict) -> tuple[str, list[str]]:
    """Validate ForecastDataset.v1 split manifests with explicit leak checks."""
    try:
        _validate_forecast_dataset_manifest(manifest)
    except (TypeError, ValueError) as exc:
        return STATUS_FAILED, [f"forecast dataset manifest invalid: {exc}"]

    errors: list[str] = []
    split_policy = manifest.get("split_policy")
    if not isinstance(split_policy, dict):
        return STATUS_FAILED, ["forecast dataset manifest missing split_policy"]

    leakage_prevention = split_policy.get("leakage_prevention")
    if not leakage_prevention:
        errors.append("forecast dataset manifest missing split_policy.leakage_prevention")
    if not isinstance(leakage_prevention, list):
        errors.append("forecast dataset manifest split_policy.leakage_prevention must be a list")

    if errors:
        return STATUS_FAILED, errors
    return STATUS_PASSED, []


def _check_split_manifest(manifest_path: Path) -> tuple[str, list[str]]:
    """Verify split manifest has no leak and explicit split strategy metadata."""
    if not manifest_path.exists():
        return STATUS_BLOCKED, [f"split manifest not found: {manifest_path}"]

    data = _load_yaml(manifest_path)
    if data is None:
        return STATUS_FAILED, [f"split manifest could not be loaded: {manifest_path}"]

    if not isinstance(data, dict):
        return STATUS_FAILED, ["split manifest must be a dictionary"]

    if data.get("schema_version") == FORECAST_DATASET_SCHEMA_VERSION:
        return _check_split_manifest_dataset(data)
    return _check_split_manifest_legacy(data)


def _check_baseline_evidence(  # noqa: C901, PLR0912
    baseline_path: Path,
    baseline_target: str,
) -> tuple[str, list[str]]:
    """Verify baseline evidence has named target ADE and FDE."""
    if not baseline_path.exists():
        return STATUS_BLOCKED, [f"baseline evidence not found: {baseline_path}"]

    data = _load_yaml(baseline_path)
    if data is None:
        return STATUS_FAILED, [f"baseline evidence could not be loaded: {baseline_path}"]

    if not isinstance(data, (dict, list)):
        return STATUS_FAILED, ["baseline evidence must be a dictionary or a list"]

    target = baseline_target.strip().lower() or DEFAULT_BASELINE_TARGET
    baselines = data if isinstance(data, list) else data.get("baselines", {})
    selected = None
    if isinstance(baselines, list):
        for baseline in baselines:
            if not isinstance(baseline, dict):
                return STATUS_FAILED, ["baseline entries must be mappings"]
            name = baseline.get("name")
            if isinstance(name, str) and name.lower() == target:
                selected = baseline
                break
    elif isinstance(baselines, dict):
        for key, value in baselines.items():
            if str(key).lower() == target:
                if isinstance(value, dict):
                    selected = value
                else:
                    return STATUS_FAILED, [f"{target} baseline entry must be a mapping"]
                break

    if selected is None:
        return STATUS_FAILED, [f"baseline evidence missing {target} entry"]

    selected_keys = {str(key).lower() for key in selected}
    missing = []
    if "ade" not in selected_keys:
        missing.append(f"{target} baseline missing ADE metric")
    if "fde" not in selected_keys:
        missing.append(f"{target} baseline missing FDE metric")
    if missing:
        return STATUS_FAILED, missing
    return STATUS_PASSED, []


def _check_horizon_definition(doc_path: Path) -> tuple[str, list[str]]:
    """Verify horizon and timestep recommendations are documented."""
    if not doc_path.exists():
        return STATUS_BLOCKED, [f"horizon definition doc not found: {doc_path}"]
    content = doc_path.read_text(encoding="utf-8")
    errors: list[str] = []
    required = ["horizon_seconds", "horizon_steps", "dt_seconds"]
    for field in required:
        if field not in content:
            errors.append(f"readiness doc missing {field} definition")

    if "horizon_recommendation" not in content:
        errors.append("readiness doc missing horizon_recommendation")
    if "timestep_recommendation" not in content:
        errors.append("readiness doc missing timestep_recommendation")

    if errors:
        return STATUS_FAILED, errors
    return STATUS_PASSED, []


def _is_oracle_tier(observation_tier: object) -> bool:
    """Return whether the tier label is oracle-like."""
    return str(observation_tier).strip().lower().startswith("oracle")


def _check_calibration_report(calibration_report_path: Path) -> tuple[str, list[str]]:
    """Require a calibration report recommendation of continue."""
    if not calibration_report_path.exists():
        return STATUS_BLOCKED, [f"calibration report not found: {calibration_report_path}"]

    payload = _load_json(calibration_report_path)
    if not isinstance(payload, dict):
        return STATUS_FAILED, ["calibration report must be JSON"]

    if payload.get("schema_version") != FORECAST_CALIBRATION_REPORT_SCHEMA_VERSION:
        return STATUS_FAILED, ["calibration report must use ForecastCalibrationReport.v1"]

    recommendation = payload.get("recommendation")
    if not isinstance(recommendation, dict):
        return STATUS_FAILED, ["calibration report missing recommendation block"]

    decision = str(recommendation.get("decision", "")).strip().lower()
    claim_status = str(recommendation.get("claim_status", "")).strip().lower()
    if decision == "":
        return STATUS_FAILED, ["calibration report missing recommendation.decision"]
    if decision != "continue":
        return STATUS_FAILED, [
            f"calibration report recommendation is {decision}, expected continue"
        ]
    if claim_status == "blocked":
        return STATUS_FAILED, ["calibration report claim_status is blocked"]
    return STATUS_PASSED, []


def _check_transferability_report(transferability_report_path: Path) -> tuple[str, list[str]]:
    """Require observation-tier split and non-blocked transferability evidence."""
    if not transferability_report_path.exists():
        return STATUS_BLOCKED, [f"transferability report not found: {transferability_report_path}"]

    payload = _load_json(transferability_report_path)
    if not isinstance(payload, dict):
        return STATUS_FAILED, ["transferability report must be JSON"]

    if payload.get("schema_version") != (FORECAST_TRANSFERABILITY_STRESS_MATRIX_SCHEMA_VERSION):
        return STATUS_FAILED, [
            "transferability report must use ForecastTransferabilityStressMatrix.v1"
        ]

    matrix_rows = payload.get("matrix_rows")
    if not isinstance(matrix_rows, list) or not matrix_rows:
        return STATUS_FAILED, ["transferability report has no matrix_rows"]

    tiers = {row.get("observation_tier") for row in matrix_rows if isinstance(row, dict)}
    has_oracle = any(_is_oracle_tier(tier) for tier in tiers)
    has_deployable = any(not _is_oracle_tier(tier) for tier in tiers)
    if not has_oracle or not has_deployable:
        return STATUS_FAILED, [
            "transferability report missing observation-tier split (oracle + deployable)",
        ]

    recommendation = payload.get("recommendation")
    if not isinstance(recommendation, dict):
        return STATUS_FAILED, ["transferability report missing recommendation block"]

    decision = str(recommendation.get("decision", "")).strip().lower()
    claim_status = str(recommendation.get("claim_status", "")).strip().lower()
    if not decision:
        return STATUS_FAILED, ["transferability report missing recommendation.decision"]
    if decision != "continue":
        return STATUS_FAILED, [
            f"transferability report recommendation is {decision}, expected continue"
        ]
    if claim_status == "blocked":
        return STATUS_FAILED, ["transferability report claim_status is blocked"]
    return STATUS_PASSED, []


def _check_closed_loop_gate(path: Path) -> tuple[str, list[str]]:
    """Require a coupling-gate recommendation of continue."""
    if not path.exists():
        return STATUS_BLOCKED, [f"closed-loop coupling gate artifact not found: {path}"]

    if path.suffix.lower() == ".json":
        payload = _load_json(path)
        if not isinstance(payload, dict):
            return STATUS_FAILED, ["closed-loop gate artifact is malformed JSON"]

        recommendation = payload.get("recommendation")
        if isinstance(recommendation, str):
            if recommendation.lower() == "continue":
                return STATUS_PASSED, []
            return STATUS_FAILED, [
                f"closed-loop gate recommendation is {recommendation}, expected continue"
            ]

        gate_status = payload.get("closed_loop_gate", {}).get("status")
        if gate_status is not None:
            return STATUS_FAILED, [
                f"closed-loop gate status is {gate_status}; expected explicit recommendation continue"
            ]
        return STATUS_FAILED, ["closed-loop gate recommendation missing"]

    if path.suffix.lower() == ".md":
        content = path.read_text(encoding="utf-8")
        status_match = re.search(r"^-\s*status:\s*`?([A-Za-z_]+)`?", content, re.MULTILINE)
        if status_match is None:
            return STATUS_FAILED, ["closed-loop gate status line not found"]
        status = status_match.group(1).strip().lower()
        return STATUS_FAILED, [
            f"closed-loop gate status is {status}; expected explicit recommendation continue"
        ]

    return STATUS_FAILED, ["closed-loop gate artifact must be JSON or Markdown"]


def _register_prerequisite(
    prerequisites: dict[str, dict[str, list[str] | str]],
    name: str,
    status: str,
    messages: list[str],
) -> None:
    """Record a prerequisite result."""
    prerequisites[name] = {"status": status, "messages": messages}


def _summarize_blocking_prerequisites(
    prerequisites: dict[str, dict[str, list[str] | str]],
) -> list[dict[str, object]]:
    """Return blockers that still prevent learned-prediction training readiness."""
    blockers: list[dict[str, object]] = []
    for name, payload in prerequisites.items():
        status = str(payload["status"])
        if status == STATUS_PASSED:
            continue
        blockers.append(
            {
                "name": name,
                "status": status,
                "messages": payload["messages"],
            }
        )
    return blockers


def validate_readiness(
    doc_path: Path,
    registry_path: Path | None = None,
    split_manifest_path: Path | None = None,
    baseline_path: Path | None = None,
    calibration_report_path: Path | None = None,
    transferability_report_path: Path | None = None,
    closed_loop_gate_path: Path | None = None,
    baseline_target: str = DEFAULT_BASELINE_TARGET,
) -> dict:
    """Run all readiness checks and return a structured report."""
    checks: dict[str, tuple[str, list[str]]] = {
        "readiness_doc": _check_readiness_doc(doc_path),
        "trace_registry": (
            STATUS_BLOCKED,
            ["trace registry path not provided"],
        )
        if registry_path is None
        else _check_trace_registry(registry_path),
        "split_manifest": (
            STATUS_BLOCKED,
            ["split manifest path not provided"],
        )
        if split_manifest_path is None
        else _check_split_manifest(split_manifest_path),
        "baseline_evidence": (
            STATUS_BLOCKED,
            ["baseline evidence path not provided"],
        )
        if baseline_path is None
        else _check_baseline_evidence(baseline_path, baseline_target),
        "calibration_report": (
            STATUS_BLOCKED,
            ["calibration report path not provided"],
        )
        if calibration_report_path is None
        else _check_calibration_report(calibration_report_path),
        "transferability_split": (
            STATUS_BLOCKED,
            ["transferability report path not provided"],
        )
        if transferability_report_path is None
        else _check_transferability_report(transferability_report_path),
        "closed_loop_gate": (
            STATUS_BLOCKED,
            ["closed-loop gate path not provided"],
        )
        if closed_loop_gate_path is None
        else _check_closed_loop_gate(closed_loop_gate_path),
        "horizon_timestep": _check_horizon_definition(doc_path),
    }

    prerequisites: dict[str, dict[str, list[str] | str]] = {}
    errors: list[str] = []
    for name, (status, messages) in checks.items():
        _register_prerequisite(prerequisites, name, status, messages)
        if status != STATUS_PASSED:
            errors.extend(messages)

    blocking_prerequisites = _summarize_blocking_prerequisites(prerequisites)
    status = "ready" if not errors else "blocked"
    return {
        "status": status,
        "errors": errors,
        "blocking_prerequisites": blocking_prerequisites,
        "checked": {
            "readiness_doc": str(doc_path),
            "trace_registry": str(registry_path) if registry_path else None,
            "split_manifest": str(split_manifest_path) if split_manifest_path else None,
            "baseline_evidence": str(baseline_path) if baseline_path else None,
            "calibration_report": str(calibration_report_path) if calibration_report_path else None,
            "transferability_report": str(transferability_report_path)
            if transferability_report_path
            else None,
            "closed_loop_gate": str(closed_loop_gate_path) if closed_loop_gate_path else None,
            "required_baseline_target": baseline_target,
        },
        "prerequisites": prerequisites,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Validate learned-prediction readiness before training is unblocked."
    )
    parser.add_argument(
        "--readiness-doc",
        type=Path,
        default=Path("docs/context/issue_2768_learned_prediction_readiness.md"),
        help="Path to the readiness contract document.",
    )
    parser.add_argument(
        "--trace-registry",
        type=Path,
        help="Path to the trace dataset registry YAML.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="Path to the split manifest YAML.",
    )
    parser.add_argument(
        "--baseline-evidence",
        type=Path,
        help="Path to the deterministic baseline evidence YAML.",
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        help="Path to the forecast calibration report JSON.",
    )
    parser.add_argument(
        "--transferability-report",
        type=Path,
        help="Path to the transferability stress-matrix JSON.",
    )
    parser.add_argument(
        "--closed-loop-gate",
        type=Path,
        help="Path to the closed-loop coupling gate artifact (JSON or Markdown).",
    )
    parser.add_argument(
        "--baseline-target",
        type=str,
        default=DEFAULT_BASELINE_TARGET,
        help="Named baseline target that must exist in baseline evidence.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root for resolving relative paths.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON validation report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate learned-prediction readiness and return a shell-friendly exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()

    def _resolve(path: Path | None) -> Path | None:
        if path is None:
            return None
        return path if path.is_absolute() else repo_root / path

    report = validate_readiness(
        doc_path=_resolve(args.readiness_doc) or args.readiness_doc,
        registry_path=_resolve(args.trace_registry),
        split_manifest_path=_resolve(args.split_manifest),
        baseline_path=_resolve(args.baseline_evidence),
        calibration_report_path=_resolve(args.calibration_report),
        transferability_report_path=_resolve(args.transferability_report),
        closed_loop_gate_path=_resolve(args.closed_loop_gate),
        baseline_target=args.baseline_target,
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["status"] == "ready":
        print("learned-prediction readiness: READY")
        for key, payload in report["prerequisites"].items():
            status = payload["status"]
            print(f" - {key}: {status}")
    else:
        print("learned-prediction readiness: BLOCKED")
        print(f"blocking prerequisites: {len(report['blocking_prerequisites'])}")
        for key, payload in report["prerequisites"].items():
            status = payload["status"]
            print(f" - {key}: {status}")
            for err in payload["messages"]:
                print(f"   * {err}")

    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
