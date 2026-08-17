#!/usr/bin/env python3
"""Run the config-first CALF/LegNav-inspired Robot SF comparator smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.benchmark.calf_legnav_comparator import (
    build_calf_legnav_comparator_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_RUNNER = REPO_ROOT / "scripts/validation/run_policy_search_step_diagnostics.py"
SCHEMA_PATH = REPO_ROOT / "robot_sf/benchmark/schemas/calf_legnav_comparator.v1.json"
CONFIG_SCHEMA_PATH = REPO_ROOT / "robot_sf/benchmark/schemas/calf_legnav_comparator_config.v1.json"
DEFAULT_CONFIG = REPO_ROOT / "configs/benchmarks/issue_7318_calf_legnav_comparator_smoke.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "output/benchmarks/issue_7318_calf_legnav_comparator/latest"


def _repo_path(value: str | Path) -> Path:
    """Resolve a repository-relative path without changing the caller's config."""
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _display_path(path: Path) -> str:
    """Render a path relative to the repository when possible."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one versioned input file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the comparator YAML config."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Comparator config must be a mapping: {path}")
    schema = json.loads(CONFIG_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload), key=lambda error: list(error.path)
    )
    if errors:
        details = "; ".join(f"{list(error.path)}: {error.message}" for error in errors[:5])
        raise ValueError(f"Comparator config failed schema validation: {details}")
    return payload


def _write_funnel(config: dict[str, Any], output_dir: Path) -> Path:
    """Write a disposable one-stage funnel for the canonical diagnostics runner."""
    matrix = _repo_path(str(config["scenario_matrix"]))
    stage = str(config.get("stage", "smoke"))
    funnel = {
        "stage_order": [stage],
        "stages": {
            stage: {
                "scenario_matrix": str(matrix),
                "seed_list": [int(config["seed"])],
                "benchmark_profile": "experimental",
                "horizon": int(config["horizon"]),
                "dt": float(config.get("dt_s", 0.1)),
                "workers": 1,
                "requires_slurm": False,
            }
        },
    }
    path = output_dir / "generated_policy_search_funnel.yaml"
    path.write_text(yaml.safe_dump(funnel, sort_keys=False), encoding="utf-8")
    return path


def _condition_command(
    config: dict[str, Any],
    condition: str,
    *,
    funnel_path: Path,
    output_dir: Path,
) -> list[str]:
    """Build one condition command using the existing policy-search runner."""
    condition_cfg = config["conditions"][condition]
    command = [
        sys.executable,
        str(TRACE_RUNNER),
        "--candidate",
        str(config["candidate"]),
        "--stage",
        str(config.get("stage", "smoke")),
        "--candidate-registry",
        str(_repo_path(str(config["candidate_registry"]))),
        "--funnel-config",
        str(funnel_path),
        "--scenario-name",
        str(config["scenario_name"]),
        "--seed",
        str(int(config["seed"])),
        "--horizon",
        str(int(config["horizon"])),
        "--output-dir",
        str(output_dir / "traces" / condition),
    ]
    if bool(condition_cfg.get("ignore_fixture_visibility", False)):
        command.append("--ignore-fixture-visibility")
    for key, flag in (
        ("position_noise_std_m", "--observation-noise-std-m"),
        ("position_noise_bound_m", "--observation-noise-bound-m"),
        ("missed_detection_probability", "--missed-detection-probability"),
        ("occlusion_distance_m", "--occlusion-distance-m"),
        ("delay_steps", "--observation-delay-steps"),
        ("observation_perturbation_seed", "--observation-perturbation-seed"),
    ):
        if key in condition_cfg and condition_cfg[key] is not None:
            command.extend([flag, str(condition_cfg[key])])
    return command


def _placeholder_trace(config: dict[str, Any]) -> dict[str, Any]:
    """Return an empty trace that keeps a failed comparator report schema-valid."""
    return {
        "candidate": str(config["candidate"]),
        "algo": "unknown",
        "scenario_id": str(config["scenario_name"]),
        "seed": int(config["seed"]),
        "horizon": int(config["horizon"]),
        "steps": [],
        "done_info": {},
        "observation_perturbation_config": {},
        "fallback_degraded_status": {},
    }


def _run_condition(
    config: dict[str, Any],
    condition: str,
    *,
    funnel_path: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run one condition and return its trace plus a compact failure record."""
    command = _condition_command(
        config,
        condition,
        funnel_path=funnel_path,
        output_dir=output_dir,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=float(config.get("timeout_seconds", 180.0)),
        )
    except subprocess.TimeoutExpired as exc:
        return _placeholder_trace(config), {
            "condition": condition,
            "status": "blocked",
            "reason": f"runner timed out after {config.get('timeout_seconds', 180.0)} seconds",
            "stderr_excerpt": str(exc.stderr or "")[-1000:],
            "command": command,
        }

    trace_path = output_dir / "traces" / condition / "trace.json"
    if completed.returncode != 0 or not trace_path.exists():
        return _placeholder_trace(config), {
            "condition": condition,
            "status": "blocked",
            "reason": f"runner exited {completed.returncode} or did not write trace.json",
            "stderr_excerpt": completed.stderr[-1000:],
            "command": command,
        }
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _placeholder_trace(config), {
            "condition": condition,
            "status": "blocked",
            "reason": f"trace could not be loaded: {exc}",
            "command": command,
        }
    if not isinstance(trace, dict):
        return _placeholder_trace(config), {
            "condition": condition,
            "status": "blocked",
            "reason": "trace root is not a JSON object",
            "command": command,
        }
    return trace, None


def _validate_report(report: dict[str, Any]) -> None:
    """Validate a report against its canonical JSON schema."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(report), key=lambda error: list(error.path)
    )
    if errors:
        details = "; ".join(f"{list(error.path)}: {error.message}" for error in errors[:5])
        raise ValueError(f"Comparator report failed schema validation: {details}")


def _markdown(report: dict[str, Any]) -> str:
    """Render a compact human-readable diagnostic handoff."""
    lines = [
        "# CALF/LegNav-inspired comparator smoke",
        "",
        f"- Status: `{report['status']}`",
        f"- Evidence status: `{report['evidence_status']}`",
        f"- Claim boundary: {report['claim_boundary']}",
        f"- Candidate: `{report['provenance']['candidate']}`",
        f"- Scenario/seed: `{report['provenance']['scenario_id']}` / `{report['provenance']['seed']}`",
        "",
        "| Metric | Perfect perception | Sensor limited | Sensor − perfect |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, row in report["paired_metrics"].items():
        left = row["perfect_perception"]["value"]
        right = row["sensor_limited"]["value"]
        delta = row["sensor_minus_perfect"]
        lines.append(
            f"| `{name}` | {left if left is not None else 'n/a'} | {right if right is not None else 'n/a'} | {delta if delta is not None else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "This is a one-seed, one-scenario diagnostic. It does not reproduce the external CALF policy or establish sensor, embodiment, or real-world transfer validity.",
            "",
            "## Unsupported fields",
            "",
        ]
    )
    lines.extend(f"- `{item['field']}`: {item['reason']}" for item in report["unsupported_fields"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse comparator CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run both paired conditions and write a schema-validated report."""
    args = parse_args(argv)
    config_path = _repo_path(args.config)
    output_dir = _repo_path(args.output_dir)
    config = _load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    funnel_path = _write_funnel(config, output_dir)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": "calf_legnav_comparator_plan.v1",
                    "config": _display_path(config_path),
                    "commands": {
                        condition: _condition_command(
                            config,
                            condition,
                            funnel_path=funnel_path,
                            output_dir=output_dir,
                        )
                        for condition in ("perfect_perception", "sensor_limited")
                    },
                },
                sort_keys=True,
            )
        )
        return 0

    traces: dict[str, dict[str, Any]] = {}
    runner_errors: list[dict[str, Any]] = []
    for condition in ("perfect_perception", "sensor_limited"):
        traces[condition], error = _run_condition(
            config,
            condition,
            funnel_path=funnel_path,
            output_dir=output_dir,
        )
        if error is not None:
            runner_errors.append(error)

    input_refs = {
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": _sha256(config_path),
        "scenario_matrix": str(_repo_path(str(config["scenario_matrix"])).relative_to(REPO_ROOT)),
        "scenario_matrix_sha256": _sha256(_repo_path(str(config["scenario_matrix"]))),
        "candidate_registry": str(
            _repo_path(str(config["candidate_registry"])).relative_to(REPO_ROOT)
        ),
    }
    report = build_calf_legnav_comparator_report(
        traces["perfect_perception"],
        traces["sensor_limited"],
        config=config,
        input_refs=input_refs,
    )
    if runner_errors:
        report["status"] = "blocked"
        report["runner_errors"] = runner_errors
    _validate_report(report)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "status": report["status"],
                "evidence_status": report["evidence_status"],
                "summary": _display_path(output_dir / "summary.json"),
                "readme": _display_path(output_dir / "README.md"),
                "runner_error_count": len(runner_errors),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "available" else 2


if __name__ == "__main__":
    raise SystemExit(main())
