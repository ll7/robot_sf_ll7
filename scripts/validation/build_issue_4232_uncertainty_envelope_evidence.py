#!/usr/bin/env python3
"""Build compact issue #4232 uncertainty-envelope alpha evidence from fixtures.

The builder consumes already-summarized alpha-sweep rows. It does not run a
benchmark campaign, copy raw artifacts, or promote safety, conformal,
deployment, paper, or dissertation claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET = Path("configs/benchmarks/issue_4232_uncertainty_envelope_claim_packet.yaml")
DEFAULT_OUTPUT_DIR = Path(
    "docs/context/evidence/issue_4232_uncertainty_envelope_claim_evidence_2026-07"
)
SCHEMA_VERSION = "issue-4232-uncertainty-envelope-evidence.v1"
COMPARATOR_ARMS = ("envelope_off_alpha_0", "envelope_on_alpha_0")
DELTA_METRICS = (
    "success_rate",
    "collision_rate",
    "near_miss_rate",
    "min_clearance_m",
    "path_efficiency",
    "runtime_seconds",
)
RAW_ARTIFACT_MARKERS = (
    ".jsonl",
    ".mp4",
    ".avi",
    ".mov",
    ".log",
    "slurm",
    "checkpoint",
    "model_cache",
    "raw_episode",
)
FORBIDDEN_CLAIM_MARKERS = (
    "conformal coverage guarantee",
    "conformal guarantee",
    "real-world safety",
    "real world safety",
    "deployment certification",
    "deployment safety",
    "generalized planner superiority",
    "paper claim",
    "dissertation claim",
)


class EvidenceBuildError(ValueError):
    """Raised when compact evidence cannot be built safely."""


def _load_packet_validator() -> Any:
    validator_path = (
        REPO_ROOT / "scripts/validation/check_issue_4232_uncertainty_envelope_claim_packet.py"
    )
    spec = importlib.util.spec_from_file_location("_issue_4232_claim_packet_check", validator_path)
    if spec is None or spec.loader is None:
        raise EvidenceBuildError(f"cannot load packet validator: {validator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceBuildError(f"{path} must contain a YAML mapping")
    return payload


def _load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise EvidenceBuildError("results fixture must not be empty")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise EvidenceBuildError("results fixture must contain a non-empty row list")
    if not all(isinstance(row, dict) for row in rows):
        raise EvidenceBuildError("results fixture rows must be JSON objects")
    return rows


def _finite_number(value: Any, *, row_id: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise EvidenceBuildError(f"{row_id}: {field} must be a finite number")
    return float(value)


def _optional_number(value: Any, *, row_id: str, field: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, row_id=row_id, field=field)


def _row_id(row: Mapping[str, Any]) -> str:
    return (
        f"{row.get('planner_id', '<planner>')}/"
        f"{row.get('scenario_id', '<scenario>')}/"
        f"{row.get('seed', '<seed>')}/"
        f"{row.get('alpha_arm_key', '<arm>')}"
    )


def _as_metrics(row: Mapping[str, Any]) -> dict[str, float | None]:
    row_id = _row_id(row)
    raw_metrics = row.get("metrics", {})
    if raw_metrics is None:
        raw_metrics = {}
    if not isinstance(raw_metrics, Mapping):
        raise EvidenceBuildError(f"{row_id}: metrics must be a mapping")
    metrics: dict[str, float | None] = {}
    for metric in DELTA_METRICS:
        value = row.get(metric, raw_metrics.get(metric))
        metrics[metric] = _optional_number(value, row_id=row_id, field=metric)
    return metrics


def _claim_text_from_rows(rows: Iterable[Mapping[str, Any]], extra_claim_text: str) -> str:
    chunks = [extra_claim_text]
    for row in rows:
        value = row.get("claim_text")
        if value is not None:
            chunks.append(str(value))
    return "\n".join(chunks).lower()


def _check_forbidden_claim_language(
    rows: Sequence[Mapping[str, Any]], extra_claim_text: str
) -> None:
    text = _claim_text_from_rows(rows, extra_claim_text)
    for marker in FORBIDDEN_CLAIM_MARKERS:
        if marker in text:
            raise EvidenceBuildError(f"forbidden claim language present: {marker}")


def _check_raw_artifacts(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        raw_paths = row.get("raw_artifact_paths", [])
        if raw_paths in (None, ""):
            continue
        if isinstance(raw_paths, str):
            raw_paths = [raw_paths]
        if not isinstance(raw_paths, Sequence):
            raise EvidenceBuildError(f"{_row_id(row)}: raw_artifact_paths must be a list")
        for raw_path in raw_paths:
            lowered = str(raw_path).lower()
            if any(marker in lowered for marker in RAW_ARTIFACT_MARKERS):
                raise EvidenceBuildError(
                    f"{_row_id(row)}: raw artifact references are not compact evidence: {raw_path}"
                )


def _packet_arm_lookup(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    arms = packet.get("alpha_arms")
    if not isinstance(arms, Sequence):
        raise EvidenceBuildError("packet alpha_arms must be available")
    return {str(arm["key"]): arm for arm in arms if isinstance(arm, Mapping) and "key" in arm}


def _normalize_rows(
    rows: Sequence[Mapping[str, Any]], packet: Mapping[str, Any]
) -> list[dict[str, Any]]:
    arms = _packet_arm_lookup(packet)
    allowed_statuses = set(packet["row_status_policy"]["allowed_values"])
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str]] = set()
    for row in rows:
        row_id = _row_id(row)
        arm_key = str(row.get("alpha_arm_key", ""))
        if arm_key not in arms:
            raise EvidenceBuildError(f"{row_id}: alpha_arm_key not declared in packet")
        planner_id = str(row.get("planner_id", ""))
        scenario_id = str(row.get("scenario_id", ""))
        if not planner_id or not scenario_id:
            raise EvidenceBuildError(f"{row_id}: planner_id and scenario_id are required")
        seed_raw = row.get("seed")
        if isinstance(seed_raw, bool) or not isinstance(seed_raw, int):
            raise EvidenceBuildError(f"{row_id}: seed must be an integer")
        status = str(row.get("row_status", ""))
        if status not in allowed_statuses:
            raise EvidenceBuildError(f"{row_id}: row_status not declared in packet")
        key = (planner_id, scenario_id, seed_raw, arm_key)
        if key in seen:
            raise EvidenceBuildError(f"{row_id}: duplicate planner/scenario/seed/arm row")
        seen.add(key)
        arm = arms[arm_key]
        normalized.append(
            {
                "planner_id": planner_id,
                "scenario_id": scenario_id,
                "scenario_family": str(row.get("scenario_family", scenario_id)),
                "seed": seed_raw,
                "alpha_arm_key": arm_key,
                "envelope_enabled": bool(arm["pedestrian_uncertainty_envelope_enabled"]),
                "alpha_mps": float(arm["pedestrian_uncertainty_alpha_mps"]),
                "row_status": status,
                "metrics": _as_metrics(row),
                "diagnostics": row.get("diagnostics", {}),
            }
        )
    if not any(row["alpha_arm_key"] == "envelope_off_alpha_0" for row in normalized):
        raise EvidenceBuildError("missing alpha-zero baseline: envelope_off_alpha_0")
    if not any(row["alpha_arm_key"] == "envelope_on_alpha_0" for row in normalized):
        raise EvidenceBuildError("missing alpha-zero regression arm: envelope_on_alpha_0")
    if not any(row["alpha_mps"] > 0.0 for row in normalized):
        raise EvidenceBuildError("missing nonzero alpha arm rows")
    return sorted(
        normalized,
        key=lambda row: (
            row["planner_id"],
            row["scenario_id"],
            row["seed"],
            row["alpha_mps"],
            row["alpha_arm_key"],
        ),
    )


def _benchmark_evidence(row_status: str, packet: Mapping[str, Any]) -> str:
    success_values = set(packet["row_status_policy"]["benchmark_strength_success_values"])
    return "eligible" if row_status in success_values else "excluded"


def _activation_status(row: Mapping[str, Any]) -> tuple[str, int | None, bool | None]:
    diagnostics = row.get("diagnostics", {})
    if diagnostics is None:
        diagnostics = {}
    if not isinstance(diagnostics, Mapping):
        raise EvidenceBuildError(f"{_row_id(row)}: diagnostics must be a mapping")
    if float(row["alpha_mps"]) == 0.0:
        return "not_applicable_alpha_zero", None, None
    count_present = "envelope_activation_count" in diagnostics
    used_present = "effective_radius_used_by_planner" in diagnostics
    count = diagnostics.get("envelope_activation_count")
    used = diagnostics.get("effective_radius_used_by_planner")
    if not count_present and not used_present:
        raise EvidenceBuildError(f"{_row_id(row)}: missing envelope activation diagnostics")
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise EvidenceBuildError(f"{_row_id(row)}: envelope_activation_count must be >= 0")
    if used is not None and not isinstance(used, bool):
        raise EvidenceBuildError(
            f"{_row_id(row)}: effective_radius_used_by_planner must be boolean"
        )
    if count is None and used is None:
        return "unknown_mechanism_activation", None, None
    activated = (count or 0) > 0 or bool(used)
    return ("activated" if activated else "no_mechanism_activation", count, used)


def _metric_table(
    rows: Sequence[dict[str, Any]], packet: Mapping[str, Any]
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in rows:
        metrics = row["metrics"]
        table.append(
            {
                "planner_id": row["planner_id"],
                "scenario_family": row["scenario_family"],
                "scenario_id": row["scenario_id"],
                "seed": row["seed"],
                "alpha_arm_key": row["alpha_arm_key"],
                "envelope_enabled": row["envelope_enabled"],
                "alpha_mps": row["alpha_mps"],
                "row_status": row["row_status"],
                "benchmark_evidence": _benchmark_evidence(row["row_status"], packet),
                **{metric: metrics[metric] for metric in DELTA_METRICS},
            }
        )
    return table


def _paired_delta_table(
    rows: Sequence[dict[str, Any]], packet: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(row["planner_id"], row["scenario_id"], int(row["seed"]))][row["alpha_arm_key"]] = (
            row
        )

    table: list[dict[str, Any]] = []
    for (planner_id, scenario_id, seed), arms in sorted(by_key.items()):
        baseline = arms.get("envelope_off_alpha_0")
        for arm_key, row in sorted(arms.items(), key=lambda item: (item[1]["alpha_mps"], item[0])):
            if arm_key == "envelope_off_alpha_0":
                continue
            comparable = (
                baseline is not None
                and row["row_status"]
                in packet["row_status_policy"]["benchmark_strength_success_values"]
                and baseline["row_status"]
                in packet["row_status_policy"]["benchmark_strength_success_values"]
            )
            delta_row: dict[str, Any] = {
                "planner_id": planner_id,
                "scenario_id": scenario_id,
                "seed": seed,
                "baseline_arm_key": "envelope_off_alpha_0",
                "comparison_arm_key": arm_key,
                "alpha_mps": row["alpha_mps"],
                "pair_status": "paired_comparable" if comparable else "not_comparable",
                "not_comparable_reason": "",
            }
            if baseline is None:
                delta_row["not_comparable_reason"] = "missing_envelope_off_alpha_0_for_seed"
            elif not comparable:
                delta_row["not_comparable_reason"] = "row_status_excluded_from_benchmark_strength"
            for metric in DELTA_METRICS:
                lhs = row["metrics"][metric]
                rhs = baseline["metrics"][metric] if baseline is not None else None
                delta_row[f"{metric}_delta"] = (
                    lhs - rhs if comparable and lhs is not None and rhs is not None else None
                )
            table.append(delta_row)
    return table


def _row_status_audit(
    rows: Sequence[dict[str, Any]], packet: Mapping[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[(row["planner_id"], row["alpha_arm_key"])][row["row_status"]] += 1
    audit: list[dict[str, Any]] = []
    ineligible = set(packet["row_status_policy"]["ineligible_benchmark_strength_values"])
    success = set(packet["row_status_policy"]["benchmark_strength_success_values"])
    for (planner_id, arm_key), counts in sorted(grouped.items()):
        total = sum(counts.values())
        eligible = sum(counts[status] for status in success)
        excluded = sum(counts[status] for status in ineligible)
        audit.append(
            {
                "planner_id": planner_id,
                "alpha_arm_key": arm_key,
                "total_rows": total,
                "benchmark_strength_rows": eligible,
                "excluded_rows": excluded,
                "row_status_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
            }
        )
    return audit


def _activation_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for row in rows:
        status, count, used = _activation_status(row)
        status_counts[status] += 1
        details.append(
            {
                "planner_id": row["planner_id"],
                "scenario_id": row["scenario_id"],
                "seed": row["seed"],
                "alpha_arm_key": row["alpha_arm_key"],
                "alpha_mps": row["alpha_mps"],
                "row_status": row["row_status"],
                "activation_status": status,
                "envelope_activation_count": count,
                "effective_radius_used_by_planner": used,
            }
        )
    return {
        "schema_version": f"{SCHEMA_VERSION}.activation-diagnostics",
        "status_counts": dict(sorted(status_counts.items())),
        "rows": details,
    }


def _runtime_cost_report(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(row["planner_id"], row["scenario_id"], int(row["seed"]))][row["alpha_arm_key"]] = (
            row
        )
    report: list[dict[str, Any]] = []
    for (planner_id, scenario_id, seed), arms in sorted(by_key.items()):
        baseline = arms.get("envelope_off_alpha_0")
        baseline_runtime = baseline["metrics"]["runtime_seconds"] if baseline else None
        for arm_key, row in sorted(arms.items(), key=lambda item: (item[1]["alpha_mps"], item[0])):
            runtime = row["metrics"]["runtime_seconds"]
            report.append(
                {
                    "planner_id": planner_id,
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "alpha_arm_key": arm_key,
                    "alpha_mps": row["alpha_mps"],
                    "runtime_seconds": runtime,
                    "runtime_delta_vs_off_alpha_0": (
                        runtime - baseline_runtime
                        if runtime is not None and baseline_runtime is not None
                        else None
                    ),
                    "row_status": row["row_status"],
                }
            )
    return report


def _csv_text(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return buffer.getvalue()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_outputs(
    *,
    output_dir: Path,
    packet_path: Path,
    packet: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    source_results: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = _metric_table(rows, packet)
    delta_rows = _paired_delta_table(rows, packet)
    audit_rows = _row_status_audit(rows, packet)
    activation = _activation_diagnostics(rows)
    runtime_rows = _runtime_cost_report(rows)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "issue": 4232,
        "evidence_status": "diagnostic_only_fixture_summary",
        "packet": _repo_relative(packet_path),
        "source_results": _repo_relative(source_results),
        "row_count": len(rows),
        "claim_boundary": (
            "Fixture-driven compact evidence builder output only. It does not run a full "
            "benchmark campaign, does not establish conformal calibration, real-world safety, "
            "deployment, generalized planner superiority, paper, or dissertation claims."
        ),
        "raw_artifact_policy": "raw JSONL, videos, logs, checkpoints, and model caches are not copied",
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "pre_registration_packet.json", dict(packet))
    _write(
        output_dir / "alpha_arm_metric_table.csv",
        _csv_text(list(metric_rows[0].keys()), metric_rows),
    )
    _write(
        output_dir / "paired_alpha_delta_table.csv",
        _csv_text(list(delta_rows[0].keys()), delta_rows),
    )
    _write(
        output_dir / "row_status_audit.csv",
        _csv_text(list(audit_rows[0].keys()), audit_rows),
    )
    _write_json(output_dir / "envelope_activation_diagnostics.json", activation)
    _write(
        output_dir / "envelope_activation_diagnostics.md",
        "\n".join(
            [
                "# Envelope Activation Diagnostics",
                "",
                "Evidence status: diagnostic-only fixture summary.",
                "",
                "Status counts:",
                *[
                    f"- `{status}`: {count}"
                    for status, count in sorted(activation["status_counts"].items())
                ],
                "",
                "Rows with `no_mechanism_activation` are blockers for interpreting alpha effects "
                "as mechanism evidence.",
            ]
        )
        + "\n",
    )
    _write(
        output_dir / "runtime_cost_report.csv",
        _csv_text(list(runtime_rows[0].keys()), runtime_rows),
    )
    _write(
        output_dir / "claim_boundary.md",
        "\n".join(
            [
                "# Claim Boundary",
                "",
                "Evidence status: diagnostic-only fixture summary.",
                "",
                "This compact bundle can support review of alpha-arm metrics, paired deltas, "
                "row-status exclusions, activation diagnostics, runtime costs, and checksums.",
                "",
                "Out of scope: full benchmark campaign run, Slurm/GPU submission, conformal "
                "calibration, deployment or real-world safety claims, generalized planner "
                "superiority, and paper/dissertation claim edits.",
            ]
        )
        + "\n",
    )
    _write(
        output_dir / "claim_readiness.md",
        "\n".join(
            [
                "# Claim Readiness",
                "",
                "claim_scope: diagnostic_only",
                "evidence_tier: diagnostic",
                "comparator: envelope_off_alpha_0",
                "baseline: envelope_off_alpha_0",
                "mechanism_activation: see envelope_activation_diagnostics.json",
                "seed_slice_boundary: fixture rows only; no campaign executed by this builder",
                "artifact_provenance: metadata.json and SHA256SUMS",
                "fallback_degraded_limitations: row_status_audit.csv excludes fallback, degraded, "
                "not_available, failed, blocked, and diagnostic_only rows from benchmark strength.",
                "",
                "Readiness result: not ready for benchmark-strength, conformal, real-world safety, "
                "deployment, paper, or dissertation claims.",
            ]
        )
        + "\n",
    )
    _write(
        output_dir / "README.md",
        "\n".join(
            [
                "# Issue #4232 Uncertainty-Envelope Evidence",
                "",
                "This directory is produced from compact alpha-sweep fixture summaries.",
                "",
                "It contains reviewable tables and diagnostics only; raw episode streams, videos, "
                "Slurm logs, checkpoints, and model caches are intentionally excluded.",
            ]
        )
        + "\n",
    )
    checksum_lines = []
    for path in sorted(output_dir.iterdir()):
        if path.name == "SHA256SUMS":
            continue
        checksum_lines.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {_repo_relative(path)}"
        )
    _write(output_dir / "SHA256SUMS", "\n".join(checksum_lines) + "\n")
    return {
        "ok": True,
        "issue": 4232,
        "output_dir": _repo_relative(output_dir),
        "files": sorted(path.name for path in output_dir.iterdir()),
        "row_count": len(rows),
        "paired_delta_rows": len(delta_rows),
        "activation_status_counts": activation["status_counts"],
    }


def build_evidence(
    *,
    packet_path: Path,
    results_path: Path,
    output_dir: Path,
    claim_text: str = "",
) -> dict[str, Any]:
    """Build the compact evidence bundle from one summarized alpha-sweep fixture."""
    packet = _load_yaml(packet_path)
    validator = _load_packet_validator()
    validator.validate_packet(packet)
    raw_rows = _load_rows(results_path)
    _check_forbidden_claim_language(raw_rows, claim_text)
    _check_raw_artifacts(raw_rows)
    rows = _normalize_rows(raw_rows, packet)
    return _write_outputs(
        output_dir=output_dir,
        packet_path=packet_path,
        packet=packet,
        rows=rows,
        source_results=results_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the issue #4232 evidence builder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet", default=str(DEFAULT_PACKET), help="Pre-registration packet YAML."
    )
    parser.add_argument(
        "--results", required=True, help="Compact alpha-sweep summary JSON or JSONL."
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Evidence output directory."
    )
    parser.add_argument(
        "--claim-text",
        default="",
        help="Optional claim text to fail-closed check for forbidden language.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the evidence builder CLI."""
    args = parse_args(argv)
    try:
        report = build_evidence(
            packet_path=Path(args.packet),
            results_path=Path(args.results),
            output_dir=Path(args.output_dir),
            claim_text=args.claim_text,
        )
    except (EvidenceBuildError, ValueError, OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"issue #4232 evidence build failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "issue #4232 compact evidence built: "
            f"rows={report['row_count']} output={report['output_dir']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
