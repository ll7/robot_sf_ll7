#!/usr/bin/env python3
"""Validate the #6970 retained-row contract and audit config exposure."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.paired_effect_metric_contract import (
    REQUIRED_METRIC_NAMES,
    PairedEffectMetricContractError,
    load_json_rows,
    load_paired_effect_metric_contract,
    validate_paired_effect_metric_rows,
)
from robot_sf.common.artifact_paths import get_repository_root


def _repo_path(path: Path, root: Path) -> str:
    """Return a stable repository-relative path when possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _resolve_contract_reference(raw: Any, *, config_path: Path, root: Path) -> Path | None:
    """Resolve a config reference using config-relative then repository-relative semantics."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    reference = Path(raw)
    candidates = (
        (reference,)
        if reference.is_absolute()
        else (config_path.parent / reference, root / reference)
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _paired_contract_reasons(payload: Mapping[str, Any]) -> list[str]:
    """Return reasons a config is relevant to the paired metric exposure audit."""
    reasons: list[str] = []
    report_contract = payload.get("report_contract")
    if (
        isinstance(report_contract, Mapping)
        and report_contract.get("paired_report_builder_issue") == 4598
    ):
        reasons.append("report_contract.paired_report_builder_issue=4598")
    result_contract = payload.get("result_contract")
    if isinstance(result_contract, Mapping):
        required_outputs = result_contract.get("required_outputs")
        if isinstance(required_outputs, Sequence) and not isinstance(
            required_outputs, (str, bytes)
        ):
            output_names = {str(value) for value in required_outputs}
            if "metric_values" in output_names:
                reasons.append("result_contract.required_outputs includes metric_values")
            if output_names.intersection(REQUIRED_METRIC_NAMES):
                reasons.append("result_contract.required_outputs includes paired metric names")
    return reasons


def _audit_configs(root: Path) -> dict[str, Any]:
    """List relevant configs and whether they declare a retained contract reference."""
    entries: list[dict[str, Any]] = []
    for config_path in sorted((root / "configs").rglob("*.yaml")):
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            entries.append(
                {
                    "config": _repo_path(config_path, root),
                    "status": "invalid_yaml",
                    "error": str(exc),
                }
            )
            continue
        if not isinstance(payload, Mapping):
            continue
        reasons = _paired_contract_reasons(payload)
        if not reasons:
            continue
        raw_reference = payload.get("retained_metric_contract")
        resolved_reference = _resolve_contract_reference(
            raw_reference,
            config_path=config_path,
            root=root,
        )
        if raw_reference is None:
            status = "missing_reference"
        elif resolved_reference is None:
            status = "invalid_reference"
        else:
            try:
                load_paired_effect_metric_contract(resolved_reference)
            except (OSError, PairedEffectMetricContractError, TypeError, ValueError) as exc:
                status = "invalid_contract"
                error = str(exc)
            else:
                status = "covered"
                error = None
        entry = {
            "config": _repo_path(config_path, root),
            "status": status,
            "reasons": reasons,
            "retained_metric_contract": (
                _repo_path(resolved_reference, root)
                if resolved_reference is not None
                else raw_reference
            ),
        }
        if status == "invalid_contract":
            entry["error"] = error
        entries.append(entry)
    counts = {
        "covered": sum(entry.get("status") == "covered" for entry in entries),
        "missing_reference": sum(entry.get("status") == "missing_reference" for entry in entries),
        "invalid_reference": sum(entry.get("status") == "invalid_reference" for entry in entries),
        "invalid_contract": sum(entry.get("status") == "invalid_contract" for entry in entries),
        "invalid_yaml": sum(entry.get("status") == "invalid_yaml" for entry in entries),
    }
    return {
        "status": "ok"
        if not any(
            counts[key]
            for key in (
                "missing_reference",
                "invalid_reference",
                "invalid_contract",
                "invalid_yaml",
            )
        )
        else "findings",
        "config_count": len(entries),
        "counts": counts,
        "configs": entries,
        "claim_boundary": (
            "Exposure audit only. Missing references identify follow-up scope and do not imply "
            "that a campaign produced benchmark evidence."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--rows", type=Path, help="Optional JSON list or JSONL retained rows")
    parser.add_argument("--audit-configs", action="store_true")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Include complete per-row validation reports when --rows is supplied",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run contract validation and optional config exposure audit."""
    args = _parser().parse_args(argv)
    root = get_repository_root().resolve()
    result: dict[str, Any] = {
        "status": "ok",
        "contract": {"path": _repo_path(args.contract, root)},
        "rows": {"status": "not_checked"},
    }
    try:
        contract = load_paired_effect_metric_contract(args.contract)
        result["contract"].update(
            {
                "status": "ok",
                "schema_version": contract["schema_version"],
                "required_metric_names": contract["required_metric_names"],
            }
        )
        if args.rows is not None:
            rows = load_json_rows(args.rows)
            rows_report = validate_paired_effect_metric_rows(
                rows,
                contract,
                include_row_reports=args.diagnostic,
            )
            result["rows"] = {
                "path": _repo_path(args.rows, root),
                **rows_report,
            }
            if not rows_report["complete"]:
                result["status"] = "blocked"
        if args.audit_configs:
            result["exposure"] = _audit_configs(root)
            if result["exposure"]["status"] != "ok" and result["status"] == "ok":
                result["status"] = result["exposure"]["status"]
    except (OSError, PairedEffectMetricContractError, TypeError, ValueError, yaml.YAMLError) as exc:
        result["status"] = "invalid"
        result["error"] = str(exc)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result["status"] in {"blocked", "findings"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
