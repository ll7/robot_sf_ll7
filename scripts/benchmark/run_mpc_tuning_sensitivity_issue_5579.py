#!/usr/bin/env python3
"""Run the bounded CPU sensitivity study for issue #5579.

The default path is a no-submit config check. The optional run evaluates the two target MPC arms
at the 20 pre-registered points and the four incumbent hybrid arms on the declared held-out
scenario and seed slice. The tuning phase remains available explicitly for bounded local work;
production callers must use the held-out phase. Raw episode output stays under ``output/``; only
compact derived reports should be promoted into ``docs/context/evidence/``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.fallback_policy import availability_payload
from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.benchmark.mpc_tuning_sensitivity import (
    analyze_results,
    build_candidate_plan,
    load_sensitivity_config,
    normalize_episode_record,
    selected_scenarios,
    validate_canary_rows,
    write_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/analysis/issue_5579_mpc_tuning_sensitivity.yaml"
SCHEMA_PATH = REPO_ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"
DEFAULT_OUT_DIR = REPO_ROOT / "output/benchmarks/issue_5579_mpc_tuning_sensitivity"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--check", action="store_true", help="Validate without running episodes")
    parser.add_argument(
        "--canary", action="store_true", help="Run canary check (seed 101, 6/6 eligibility)"
    )
    parser.add_argument(
        "--phase",
        choices=("tuning", "held_out"),
        default="held_out",
        help="Execution scope for a study run; production defaults to the frozen held-out phase",
    )
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row {line_number} in {path} must be an object")
        rows.append(payload)
    return rows


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_effective_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _check_summary(config: dict[str, Any]) -> dict[str, Any]:
    tuning_scenarios = selected_scenarios(config, repo_root=REPO_ROOT, scope_name="tuning_scope")
    held_out_scenarios = selected_scenarios(
        config, repo_root=REPO_ROOT, scope_name="held_out_scope"
    )
    plan = build_candidate_plan(config, repo_root=REPO_ROOT)
    target_points = [entry for entry in plan if entry["target"]]
    incumbents = [entry for entry in plan if not entry["target"]]
    res: dict[str, Any] = {
        "status": "ok",
        "issue": 5579,
        "scenario_count": len(tuning_scenarios),
        "seed_count": len(config["tuning_scope"]["seeds"]),
        "target_arm_count": len(config["target_arms"]),
        "candidate_count": len(config["search"]["candidate_points"]),
        "target_execution_rows": len(target_points),
        "incumbent_execution_rows": len(incumbents),
        "episode_row_bound": len(plan)
        * len(tuning_scenarios)
        * len(config["tuning_scope"]["seeds"]),
        "held_out_scenario_count": len(held_out_scenarios),
        "held_out_seed_count": len(config["held_out_scope"]["seeds"]),
        "held_out_episode_row_bound": len(plan)
        * len(held_out_scenarios)
        * len(config["held_out_scope"]["seeds"]),
    }
    res["tuning_scope"] = {
        "scenario_ids": config["tuning_scope"]["scenario_ids"],
        "seeds": config["tuning_scope"]["seeds"],
        "scenario_list_hash": config["tuning_scope"]["scenario_list_hash"],
    }
    res["held_out_scope"] = {
        "scenario_count": len(config["held_out_scope"]["scenario_ids"]),
        "scenario_ids": config["held_out_scope"]["scenario_ids"],
        "seeds": config["held_out_scope"]["seeds"],
        "scenario_list_hash": config["held_out_scope"]["scenario_list_hash"],
        "excluded_scenarios": config["held_out_scope"]["excluded_scenarios"],
    }
    res["canary"] = {
        "seed": config["canary"]["seed"],
        "required_eligible_episodes": config["canary"]["required_eligible_episodes"],
        "target_eligible_ratio": config["canary"]["target_eligible_ratio"],
        "native_solver_required": True,
    }
    res["inference"] = {
        "inference_population": config["inference"]["inference_population"],
        "resampling_unit": config["inference"]["resampling_unit"],
        "bootstrap_confidence": config["inference"]["bootstrap"]["confidence_level"],
        "bootstrap_replicates": config["inference"]["bootstrap"]["replicates"],
        "multiplicity_method": config["inference"]["multiplicity"]["method"],
        "contrast_count": config["inference"]["multiplicity"]["contrast_count"],
    }
    return res


def run_study(
    config: dict[str, Any], *, out_dir: Path, config_path: Path, phase: str = "held_out"
) -> dict[str, Any]:
    """Execute every declared arm for one frozen phase and build the compact report."""
    if phase not in {"tuning", "held_out"}:
        raise ValueError(f"unsupported sensitivity phase: {phase}")
    scope_name = "tuning_scope" if phase == "tuning" else "held_out_scope"
    scenarios = selected_scenarios(config, repo_root=REPO_ROOT, scope_name=scope_name)
    plan = build_candidate_plan(config, repo_root=REPO_ROOT)
    scope = config[scope_name]
    source_matrix = REPO_ROOT / scope["source_matrix"]
    raw_dir = out_dir / "raw"
    configs_dir = out_dir / "configs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for entry in plan:
        arm_key = str(entry["arm_key"])
        candidate_id = str(entry["candidate_id"])
        safe_id = f"{arm_key}__{candidate_id}"
        if entry["target"]:
            algo_config_path = configs_dir / f"{safe_id}.yaml"
            _write_effective_config(algo_config_path, entry["effective_config"])
        else:
            algo_config_path = REPO_ROOT / str(entry["algo_config_path"])
        episodes_path = raw_dir / f"{safe_id}.jsonl"
        if episodes_path.exists():
            episodes_path.unlink()
        summary = run_map_batch(
            deepcopy(scenarios),
            episodes_path,
            schema_path=SCHEMA_PATH,
            scenario_path=source_matrix,
            algo=str(entry["algo"]),
            algo_config_path=str(algo_config_path),
            horizon=int(scope["horizon"]),
            dt=float(scope["dt"]),
            record_forces=False,
            workers=int(scope["workers"]),
            resume=False,
            benchmark_profile="experimental",
        )
        availability = summary.get("benchmark_availability") or availability_payload(summary)
        for record in _read_jsonl(episodes_path):
            decorated = dict(record)
            decorated["sensitivity_availability"] = availability
            all_rows.append(
                normalize_episode_record(
                    decorated,
                    arm_key=arm_key,
                    candidate_id=candidate_id,
                    expected_config_hash=str(entry["config_sha256_16"]),
                )
            )

    normalized_path = out_dir / "normalized_episode_rows.jsonl"
    normalized_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    report = analyze_results(
        config,
        all_rows,
        repo_root=REPO_ROOT,
        config_path=_display_path(config_path),
        run_commit=_git_head(),
        reproduction_command=(
            "uv run python scripts/benchmark/run_mpc_tuning_sensitivity_issue_5579.py "
            "--config configs/analysis/issue_5579_mpc_tuning_sensitivity.yaml "
            f"--phase {phase} "
            "--out-dir output/benchmarks/issue_5579_mpc_tuning_sensitivity"
        ),
        raw_artifact_root=_display_path(raw_dir),
        scope_name=scope_name,
    )
    report["phase"] = phase
    report["normalized_episode_rows"] = _display_path(normalized_path)
    write_report(report, out_dir)
    return report


def _display_path(path: Path) -> str:
    """Return a stable repository-relative config path when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def run_canary_check(config: dict[str, Any], *, out_dir: Path, config_path: Path) -> dict[str, Any]:
    """Execute and strictly validate the six target canary rows at seed 101."""
    scenarios = selected_scenarios(config, repo_root=REPO_ROOT, scope_name="tuning_scope")
    canary_cfg = config.get("canary", {})
    canary_seed = int(canary_cfg.get("seed", 101))
    required_eligible = int(canary_cfg.get("required_eligible_episodes", 6))

    for scenario in scenarios:
        scenario["seeds"] = [canary_seed]

    plan = build_candidate_plan(config, repo_root=REPO_ROOT)
    canary_plan = [
        entry for entry in plan if entry["target"] and entry["candidate_id"] == "incumbent"
    ]
    scope = config["tuning_scope"]
    source_matrix = REPO_ROOT / scope["source_matrix"]
    raw_dir = out_dir / "raw"
    configs_dir = out_dir / "configs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for entry in canary_plan:
        arm_key = str(entry["arm_key"])
        candidate_id = str(entry["candidate_id"])
        safe_id = f"canary__{arm_key}__{candidate_id}"
        algo_config_path = configs_dir / f"{safe_id}.yaml"
        _write_effective_config(algo_config_path, entry["effective_config"])
        episodes_path = raw_dir / f"{safe_id}.jsonl"
        if episodes_path.exists():
            episodes_path.unlink()
        summary = run_map_batch(
            deepcopy(scenarios),
            episodes_path,
            schema_path=SCHEMA_PATH,
            scenario_path=source_matrix,
            algo=str(entry["algo"]),
            algo_config_path=str(algo_config_path),
            horizon=int(scope["horizon"]),
            dt=float(scope["dt"]),
            record_forces=False,
            workers=int(scope["workers"]),
            resume=False,
            benchmark_profile="experimental",
        )
        availability = summary.get("benchmark_availability") or availability_payload(summary)
        for record in _read_jsonl(episodes_path):
            decorated = dict(record)
            decorated["sensitivity_availability"] = availability
            all_rows.append(
                normalize_episode_record(
                    decorated,
                    arm_key=arm_key,
                    candidate_id=candidate_id,
                    expected_config_hash=str(entry["config_sha256_16"]),
                )
            )
        if canary_cfg.get("stop_on_ineligible") is True:
            partial = validate_canary_rows(
                all_rows,
                scenario_ids=config["tuning_scope"]["scenario_ids"],
                seed=canary_seed,
                required_eligible=required_eligible,
            )
            if partial["invalid_rows"] or partial["duplicate_keys"] or partial["unexpected_keys"]:
                partial["canary_seed"] = canary_seed
                partial["scope_name"] = "tuning_scope"
                partial["config_path"] = _display_path(config_path)
                partial["stop_reason"] = "ineligible_row"
                return partial

    result = validate_canary_rows(
        all_rows,
        scenario_ids=config["tuning_scope"]["scenario_ids"],
        seed=canary_seed,
        required_eligible=required_eligible,
    )
    result["canary_seed"] = canary_seed
    result["scope_name"] = "tuning_scope"
    result["config_path"] = _display_path(config_path)
    return result


def main(argv: list[str] | None = None) -> int:
    """Validate or execute the bounded diagnostic."""
    args = _parse_args(argv)
    config = load_sensitivity_config(args.config, repo_root=REPO_ROOT)
    if args.check:
        print(json.dumps(_check_summary(config), sort_keys=True))
        return 0
    if args.canary:
        canary_res = run_canary_check(config, out_dir=args.out_dir, config_path=args.config)
        print(json.dumps(canary_res, sort_keys=True))
        return 0 if canary_res["status"] == "ok" else 1
    report = run_study(config, out_dir=args.out_dir, config_path=args.config, phase=str(args.phase))
    print(
        json.dumps(
            {
                "status": report["status"],
                "issue": report["issue"],
                "read": report["read"]["decision"],
                "eligible_episode_rows": report["eligible_episode_rows"],
                "excluded_episode_rows": report["excluded_episode_rows"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
