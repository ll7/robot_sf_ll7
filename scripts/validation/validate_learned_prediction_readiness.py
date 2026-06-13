"""Validate learned-prediction readiness before training is unblocked.

This validator checks the fail-closed contract defined in
docs/context/issue_2768_learned_prediction_readiness.md. It exits with code 0 only when
all prerequisites are satisfied; otherwise it exits with code 2 and a structured error
message.

Checks performed:
  1. Readiness doc exists and contains the required section headers.
  2. Trace registry file exists (if referenced) with at least one source entry.
  3. Split manifest exists with valid fractions summing to 1.0.
  4. Baseline evidence exists with constant_velocity ADE and FDE metrics.
  5. Horizon definition is present in the readiness doc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_yaml(path: Path) -> dict | list | None:
    """Load a YAML file, returning None if the file does not exist."""
    try:
        import yaml
    except ImportError:
        return None
    if not path.exists():
        return None
    with open(path) as fh:
        return yaml.safe_load(fh)


def _check_readiness_doc(doc_path: Path) -> list[str]:
    """Verify the readiness document exists and contains required sections."""
    errors: list[str] = []
    if not doc_path.exists():
        errors.append(f"readiness doc not found: {doc_path}")
        return errors

    content = doc_path.read_text()
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
    return errors


def _check_trace_registry(registry_path: Path) -> list[str]:
    """Verify trace registry exists and has at least one viable source entry."""
    errors: list[str] = []
    if not registry_path.exists():
        errors.append(f"trace registry not found: {registry_path}")
        return errors

    data = _load_yaml(registry_path)
    if data is None:
        errors.append(f"trace registry could not be loaded: {registry_path}")
        return errors

    sources = data if isinstance(data, list) else data.get("sources", [])
    if not sources:
        errors.append("trace registry has no source entries")
        return sources

    viable = 0
    for src in sources:
        count = src.get("episode_count", 0)
        actors = src.get("actor_types", [])
        if count > 0 and actors:
            viable += 1

    if viable == 0:
        errors.append(
            "trace registry has no source with episode_count > 0 and non-empty actor_types"
        )
    return errors


def _check_split_manifest(manifest_path: Path) -> list[str]:
    """Verify split manifest has valid fractions summing to 1.0."""
    errors: list[str] = []
    if not manifest_path.exists():
        errors.append(f"split manifest not found: {manifest_path}")
        return errors

    data = _load_yaml(manifest_path)
    if data is None:
        errors.append(f"split manifest could not be loaded: {manifest_path}")
        return errors

    train_frac = data.get("train_fraction")
    val_frac = data.get("validation_fraction")
    test_frac = data.get("test_fraction")

    if train_frac is None or val_frac is None or test_frac is None:
        errors.append(
            "split manifest missing train_fraction, validation_fraction, or test_fraction"
        )
        return errors

    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        errors.append(f"split fractions sum to {total}, expected 1.0")

    strategy = data.get("split_strategy")
    if not strategy:
        errors.append("split manifest missing split_strategy")

    leakage = data.get("leakage_prevention")
    if not leakage:
        errors.append("split manifest missing leakage_prevention statement")

    return errors


def _check_baseline_evidence(baseline_path: Path) -> list[str]:
    """Verify baseline evidence has constant_velocity ADE and FDE."""
    errors: list[str] = []
    if not baseline_path.exists():
        errors.append(f"baseline evidence not found: {baseline_path}")
        return errors

    data = _load_yaml(baseline_path)
    if data is None:
        errors.append(f"baseline evidence could not be loaded: {baseline_path}")
        return errors

    baselines = data if isinstance(data, list) else data.get("baselines", {})
    cv = None
    if isinstance(baselines, list):
        for b in baselines:
            if b.get("name") == "constant_velocity":
                cv = b
                break
    elif isinstance(baselines, dict):
        cv = baselines.get("constant_velocity")

    if cv is None:
        errors.append("baseline evidence missing constant_velocity entry")
        return errors

    if "ade" not in cv:
        errors.append("constant_velocity baseline missing ADE metric")
    if "fde" not in cv:
        errors.append("constant_velocity baseline missing FDE metric")

    return errors


def _check_horizon_definition(doc_path: Path) -> list[str]:
    """Verify horizon definition section contains horizon_seconds and horizon_steps."""
    errors: list[str] = []
    if not doc_path.exists():
        return errors

    content = doc_path.read_text()
    if "horizon_seconds" not in content:
        errors.append("readiness doc missing horizon_seconds definition")
    if "horizon_steps" not in content:
        errors.append("readiness doc missing horizon_steps definition")
    return errors


def validate_readiness(
    doc_path: Path,
    registry_path: Path | None = None,
    split_manifest_path: Path | None = None,
    baseline_path: Path | None = None,
) -> dict:
    """Run all readiness checks and return a structured report."""
    all_errors: list[str] = []

    all_errors.extend(_check_readiness_doc(doc_path))

    if registry_path is None:
        all_errors.append("trace registry path not provided")
    else:
        all_errors.extend(_check_trace_registry(registry_path))

    if split_manifest_path is None:
        all_errors.append("split manifest path not provided")
    else:
        all_errors.extend(_check_split_manifest(split_manifest_path))

    if baseline_path is None:
        all_errors.append("baseline evidence path not provided")
    else:
        all_errors.extend(_check_baseline_evidence(baseline_path))

    all_errors.extend(_check_horizon_definition(doc_path))

    status = "ready" if not all_errors else "blocked"
    return {
        "status": status,
        "errors": all_errors,
        "checked": {
            "readiness_doc": str(doc_path),
            "trace_registry": str(registry_path) if registry_path else None,
            "split_manifest": str(split_manifest_path) if split_manifest_path else None,
            "baseline_evidence": str(baseline_path) if baseline_path else None,
        },
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
        help="Path to the train/validation/test split manifest YAML.",
    )
    parser.add_argument(
        "--baseline-evidence",
        type=Path,
        help="Path to the deterministic baseline evidence YAML.",
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

    report = validate_readiness(
        doc_path=args.readiness_doc,
        registry_path=args.trace_registry,
        split_manifest_path=args.split_manifest,
        baseline_path=args.baseline_evidence,
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["status"] == "ready":
        print("learned-prediction readiness: READY")
    else:
        print("learned-prediction readiness: BLOCKED")
        for err in report["errors"]:
            print(f"  - {err}")

    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
