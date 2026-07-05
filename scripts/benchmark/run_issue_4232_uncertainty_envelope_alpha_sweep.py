#!/usr/bin/env python3
"""Run the issue #4232 CPU diagnostic uncertainty-envelope alpha sweep.

This runner executes a bounded local diagnostic slice from the pre-registered
#4232 packet, converts episode JSONL into the compact evidence-builder input
format, and then writes the compact review bundle. It does not submit compute,
does not run the full benchmark campaign by default, and marks local smoke rows
as diagnostic-only unless the caller explicitly opts into benchmark-strength row
status handling.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark import map_runner
from robot_sf.benchmark.runner import load_scenario_matrix
from scripts.validation import build_issue_4232_uncertainty_envelope_evidence as evidence_builder
from scripts.validation import check_issue_4232_uncertainty_envelope_claim_packet as packet_checker

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET = REPO_ROOT / "configs/benchmarks/issue_4232_uncertainty_envelope_claim_packet.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/issue_4232_uncertainty_envelope_diagnostic_smoke"
DEFAULT_SCHEMA = REPO_ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"
DEFAULT_PHASE = "diagnostic-smoke"
DEFAULT_ALPHA_ARMS = (
    "envelope_off_alpha_0",
    "envelope_on_alpha_0",
    "envelope_on_alpha_0p10",
)


class DiagnosticRunError(ValueError):
    """Raised when the issue #4232 diagnostic run cannot be trusted."""


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _load_packet(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DiagnosticRunError("packet must be a YAML mapping")
    packet_checker.validate_packet(payload, repo_root=REPO_ROOT)
    return payload


def _mapping_sequence(value: Any, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DiagnosticRunError(f"{field} must be a sequence")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise DiagnosticRunError(f"{field}[{index}] must be a mapping")
        rows.append(dict(item))
    if not rows:
        raise DiagnosticRunError(f"{field} must not be empty")
    return rows


def _select_by_key(
    rows: Sequence[Mapping[str, Any]], *, keys: Sequence[str], field: str
) -> list[dict[str, Any]]:
    by_key = {str(row.get("key") or row.get("planner_id")): dict(row) for row in rows}
    selected: list[dict[str, Any]] = []
    missing = [key for key in keys if key not in by_key]
    if missing:
        raise DiagnosticRunError(f"unknown {field}: {', '.join(missing)}")
    for key in keys:
        selected.append(by_key[key])
    return selected


def _scenario_identifier(scenario: Mapping[str, Any], index: int) -> str:
    for key in ("id", "scenario_id", "name"):
        value = scenario.get(key)
        if value:
            return str(value)
    return f"scenario_{index}"


def _scenario_family(scenario: Mapping[str, Any], scenario_id: str) -> str:
    metadata = scenario.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("scenario_family", "family"):
            value = metadata.get(key)
            if value:
                return str(value)
    return str(scenario.get("scenario_family") or scenario.get("family") or scenario_id)


def _diagnostic_scenarios(
    packet: Mapping[str, Any],
    *,
    max_scenarios: int,
    max_seeds: int,
) -> list[dict[str, Any]]:
    surface = packet["scenario_surface"]
    matrix_path = REPO_ROOT / str(surface["matrix_path"])
    scenarios = _mapping_sequence(load_scenario_matrix(matrix_path), field="scenario matrix")
    seeds = list(packet["seed_policy"]["seeds"])[:max_seeds]
    if not seeds:
        raise DiagnosticRunError("diagnostic seed slice is empty")

    selected: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios[:max_scenarios]):
        copied = copy.deepcopy(scenario)
        scenario_id = _scenario_identifier(copied, index)
        copied.setdefault("id", scenario_id)
        copied.setdefault("name", scenario_id)
        copied.setdefault("metadata", {})
        if isinstance(copied["metadata"], dict):
            copied["metadata"].setdefault(
                "issue_4232_scenario_family", _scenario_family(copied, scenario_id)
            )
        copied["seeds"] = [int(seed) for seed in seeds]
        sim_config = dict(copied.get("simulation_config") or {})
        sim_config["max_episode_steps"] = int(surface["max_episode_steps"])
        copied["simulation_config"] = sim_config
        selected.append(copied)
    if not selected:
        raise DiagnosticRunError("diagnostic scenario slice is empty")
    return selected


def _arm_scenario(base_scenario: Mapping[str, Any], arm: Mapping[str, Any]) -> dict[str, Any]:
    scenario = copy.deepcopy(dict(base_scenario))
    arm_key = str(arm["key"])
    base_name = str(scenario.get("name") or scenario.get("id") or "scenario")
    scenario["name"] = f"{base_name}__issue4232__{arm_key}"
    scenario["issue_4232_alpha_arm_key"] = arm_key
    scenario.setdefault("metadata", {})
    if isinstance(scenario["metadata"], dict):
        scenario["metadata"]["issue_4232_alpha_arm_key"] = arm_key
    sim_config = dict(scenario.get("simulation_config") or {})
    sim_config["pedestrian_uncertainty_envelope_enabled"] = bool(
        arm["pedestrian_uncertainty_envelope_enabled"]
    )
    sim_config["pedestrian_uncertainty_alpha_mps"] = float(arm["pedestrian_uncertainty_alpha_mps"])
    sim_config["max_episode_steps"] = int(arm["max_episode_steps"])
    scenario["simulation_config"] = sim_config
    return scenario


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DiagnosticRunError(f"map runner did not create {_repo_relative(path)}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise DiagnosticRunError(f"{_repo_relative(path)}:{line_number} must be a JSON object")
        rows.append(payload)
    if not rows:
        raise DiagnosticRunError(f"map runner wrote no episode rows: {_repo_relative(path)}")
    return rows


def _bool_metric(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return 1.0 if float(value) > 0.0 else 0.0
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in {"true", "yes", "1", "success"} else 0.0
    return 0.0


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_metric(record: Mapping[str, Any], *keys: str) -> Any:
    metrics = record.get("metrics")
    for key in keys:
        if key in record:
            return record[key]
        if isinstance(metrics, Mapping) and key in metrics:
            return metrics[key]
    return None


def _episode_row_status(record: Mapping[str, Any], *, default_status: str) -> str:
    raw = str(record.get("row_status") or record.get("status") or "").strip().lower()
    if raw in {"fallback", "degraded", "not_available", "failed", "blocked"}:
        return raw
    return default_status


def _planner_runtime(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return planner runtime diagnostics from compact or map-runner episode records."""
    planner_runtime = record.get("planner_runtime")
    if isinstance(planner_runtime, Mapping):
        return planner_runtime
    algorithm_metadata = record.get("algorithm_metadata")
    if isinstance(algorithm_metadata, Mapping):
        planner_runtime = algorithm_metadata.get("planner_runtime")
        if isinstance(planner_runtime, Mapping):
            return planner_runtime
    return None


def _activation_diagnostics(
    record: Mapping[str, Any], *, arm: Mapping[str, Any], row_status: str
) -> dict[str, Any]:
    alpha = float(arm["pedestrian_uncertainty_alpha_mps"])
    if alpha == 0.0:
        return {}
    diagnostics = {
        "envelope_activation_count": None,
        "effective_radius_used_by_planner": None,
    }
    planner_runtime = _planner_runtime(record)
    if planner_runtime is None:
        return diagnostics
    raw_envelope = planner_runtime.get("pedestrian_uncertainty_envelope")
    if not isinstance(raw_envelope, Mapping):
        return diagnostics
    envelope = dict(raw_envelope)
    used = envelope.get("effective_radius_used_by_planner")
    if isinstance(used, bool):
        diagnostics["effective_radius_used_by_planner"] = used
    count = envelope.get("envelope_activation_count")
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        diagnostics["envelope_activation_count"] = count
    return diagnostics


def _record_identity(
    record: Mapping[str, Any], fallback_scenario: Mapping[str, Any]
) -> tuple[str, int]:
    scenario_id = str(
        record.get("scenario_id")
        or record.get("scenario")
        or fallback_scenario.get("id")
        or fallback_scenario.get("name")
        or "scenario"
    )
    seed_raw = record.get("seed")
    if seed_raw is None:
        seed_raw = record.get("episode_seed")
    if seed_raw is None:
        seed_raw = next(iter(fallback_scenario.get("seeds") or [0]))
    return scenario_id, int(seed_raw)


def _summary_rows_from_episode_jsonl(
    *,
    planner: Mapping[str, Any],
    arm: Mapping[str, Any],
    scenario: Mapping[str, Any],
    episode_jsonl: Path,
    default_status: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenario_family = _scenario_family(scenario, str(scenario.get("id") or scenario.get("name")))
    for record in _jsonl_rows(episode_jsonl):
        scenario_id, seed = _record_identity(record, scenario)
        row_status = _episode_row_status(record, default_status=default_status)
        success = _first_metric(record, "success")
        collision = _first_metric(record, "collision", "collisions")
        near_misses = _first_metric(record, "near_misses", "near_miss")
        min_clearance = _finite_or_none(
            _first_metric(record, "min_clearance_m", "minimum_clearance", "mean_clearance")
        )
        runtime = _finite_or_none(
            _first_metric(record, "runtime_seconds", "runtime_s", "duration_s")
        )
        rows.append(
            {
                "planner_id": str(planner["planner_id"]),
                "scenario_id": scenario_id,
                "scenario_family": scenario_family,
                "seed": seed,
                "alpha_arm_key": str(arm["key"]),
                "row_status": row_status,
                "metrics": {
                    "success_rate": _bool_metric(success),
                    "collision_rate": _bool_metric(collision),
                    "near_miss_rate": _bool_metric(near_misses),
                    "min_clearance_m": min_clearance,
                    "path_efficiency": _finite_or_none(_first_metric(record, "path_efficiency")),
                    "runtime_seconds": runtime,
                },
                "diagnostics": _activation_diagnostics(record, arm=arm, row_status=row_status),
            }
        )
    return rows


def _write_json(path: Path, payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    """Run the bounded local diagnostic alpha sweep and compact evidence builder."""
    packet_path = Path(args.packet)
    output_dir = Path(args.output_dir)
    packet = _load_packet(packet_path)
    planners = _select_by_key(
        _mapping_sequence(packet["planner_families"], field="planner_families"),
        keys=args.planner,
        field="planner",
    )
    arms = _select_by_key(
        _mapping_sequence(packet["alpha_arms"], field="alpha_arms"),
        keys=args.alpha_arm,
        field="alpha arm",
    )
    scenarios = _diagnostic_scenarios(
        packet,
        max_scenarios=args.max_scenarios,
        max_seeds=args.max_seeds,
    )
    default_status = (
        "successful_evidence" if args.allow_benchmark_strength_status else "diagnostic_only"
    )
    raw_root = output_dir / "raw_episode_jsonl"
    summary_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []

    for planner in planners:
        for arm in arms:
            for scenario in scenarios:
                scenario_arm = _arm_scenario(scenario, arm)
                out_path = (
                    raw_root
                    / str(planner["planner_id"])
                    / str(arm["key"])
                    / f"{scenario_arm['name']}.jsonl"
                )
                result = map_runner.run_map_batch(
                    [scenario_arm],
                    out_path,
                    schema_path=DEFAULT_SCHEMA,
                    scenario_path=packet["scenario_surface"]["matrix_path"],
                    algo=str(planner["planner_id"]),
                    algo_config_path=str(REPO_ROOT / str(planner["base_config_path"])),
                    benchmark_profile="experimental",
                    horizon=int(arm["max_episode_steps"]),
                    dt=float(arm["dt"]),
                    record_forces=False,
                    resume=False,
                    workers=1,
                )
                run_records.append(
                    {
                        "planner_id": str(planner["planner_id"]),
                        "alpha_arm_key": str(arm["key"]),
                        "scenario_id": str(scenario.get("id") or scenario.get("name")),
                        "episode_jsonl": _repo_relative(out_path),
                        "map_runner_result": result,
                    }
                )
                summary_rows.extend(
                    _summary_rows_from_episode_jsonl(
                        planner=planner,
                        arm=arm,
                        scenario=scenario,
                        episode_jsonl=out_path,
                        default_status=default_status,
                    )
                )

    results_path = output_dir / "compact_alpha_sweep_rows.json"
    _write_json(results_path, {"rows": summary_rows})
    _write_json(output_dir / "run_manifest.json", {"phase": args.phase, "runs": run_records})
    evidence_report = evidence_builder.build_evidence(
        packet_path=packet_path,
        results_path=results_path,
        output_dir=output_dir / "compact_evidence",
        claim_text=(
            "diagnostic-only CPU smoke output; all packet forbidden claim modes remain excluded"
        ),
    )
    return {
        "ok": True,
        "issue": 4232,
        "phase": args.phase,
        "output_dir": _repo_relative(output_dir),
        "row_count": len(summary_rows),
        "default_row_status": default_status,
        "planners": [str(planner["planner_id"]) for planner in planners],
        "alpha_arms": [str(arm["key"]) for arm in arms],
        "scenario_count": len(scenarios),
        "seed_count": args.max_seeds,
        "claim_boundary": "diagnostic_only_no_benchmark_strength_claim",
        "evidence_report": evidence_report,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the issue #4232 diagnostic runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", default=str(DEFAULT_PACKET), help="Issue #4232 packet YAML.")
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Ignored output root."
    )
    parser.add_argument("--phase", default=DEFAULT_PHASE, choices=(DEFAULT_PHASE,))
    parser.add_argument(
        "--planner", action="append", default=["prediction_mpc"], help="Planner id."
    )
    parser.add_argument(
        "--alpha-arm",
        action="append",
        default=list(DEFAULT_ALPHA_ARMS),
        help="Alpha arm key from packet.",
    )
    parser.add_argument("--max-scenarios", type=int, default=1, help="Bounded scenario slice size.")
    parser.add_argument("--max-seeds", type=int, default=1, help="Bounded seed slice size.")
    parser.add_argument(
        "--allow-benchmark-strength-status",
        action="store_true",
        help="Use successful_evidence row status for local rows. Default is diagnostic_only.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the issue #4232 diagnostic alpha-sweep CLI."""
    args = parse_args(argv)
    try:
        report = run_diagnostic(args)
    except (
        DiagnosticRunError,
        packet_checker.PacketError,
        evidence_builder.EvidenceBuildError,
        OSError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"issue #4232 diagnostic alpha sweep failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "issue #4232 diagnostic alpha sweep complete: "
            f"rows={report['row_count']} output={report['output_dir']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
