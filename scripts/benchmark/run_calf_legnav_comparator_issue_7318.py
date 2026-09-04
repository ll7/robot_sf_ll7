#!/usr/bin/env python3
"""Run the config-first CALF/LegNav-inspired Robot SF comparator smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.benchmark.calf_legnav_comparator import (
    build_calf_legnav_comparator_report,
)
from scripts.validation.run_policy_search_candidate import (
    _resolve_path,
    load_candidate_definition,
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


def _json_sha256(payload: Any) -> str:
    """Return a stable digest for one resolved JSON-compatible configuration."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _registry_checkpoint_sha256(registry_path: Path, model_id: str) -> str:
    """Return the durable registry digest for a resolved learned model."""
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    models = registry.get("models")
    if not isinstance(models, list):
        raise ValueError(f"Model registry has no models list: {registry_path}")
    for entry in models:
        if not isinstance(entry, dict) or str(entry.get("model_id", "")) != model_id:
            continue
        release = entry.get("github_release")
        declared = release.get("sha256") if isinstance(release, dict) else None
        if not isinstance(declared, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", declared):
            raise ValueError(f"Model registry entry has no valid checkpoint SHA-256: {model_id}")
        return declared.lower()
    raise ValueError(f"Model ID is absent from the model registry: {model_id}")


def _normalize_sha256(value: Any) -> str | None:
    """Return a normalized SHA-256 value, or ``None`` when the value is malformed."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if re.fullmatch(r"[0-9a-f]{64}", normalized) else None


def _input_refs(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    """Return resolved input lineage and digests, or raise on missing provenance inputs."""
    scenario_matrix = _repo_path(str(config["scenario_matrix"]))
    candidate_registry = _repo_path(str(config["candidate_registry"]))
    _, candidate_payload, resolved_config, candidate_config_path = load_candidate_definition(
        candidate_registry,
        str(config["candidate"]),
    )
    base_config_path = _resolve_path(
        candidate_config_path.parent,
        candidate_payload.get("base_config_path"),
    )
    if base_config_path is None:
        raise ValueError(f"Candidate has no resolvable base configuration: {config['candidate']}")
    model_id = resolved_config.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f"Resolved candidate has no model_id: {config['candidate']}")
    model_registry = REPO_ROOT / "model/registry.yaml"
    refs = {
        "config": _display_path(config_path),
        "scenario_matrix": _display_path(scenario_matrix),
        "candidate_registry": _display_path(candidate_registry),
        "candidate_config": _display_path(candidate_config_path),
        "base_config": _display_path(base_config_path),
        "model_registry": _display_path(model_registry),
        "checkpoint_model_id": model_id,
        "checkpoint_hash_source_declared": "model_registry.github_release.sha256",
        "checkpoint_sha256_declared": _registry_checkpoint_sha256(model_registry, model_id),
        "resolved_algo_config_sha256": _json_sha256(resolved_config),
    }
    for name, path in (
        ("config", config_path),
        ("scenario_matrix", scenario_matrix),
        ("candidate_registry", candidate_registry),
        ("candidate_config", candidate_config_path),
        ("base_config", base_config_path),
        ("model_registry", model_registry),
    ):
        refs[f"{name}_sha256"] = _sha256(path)
    if resolved_config.get("predictive_foresight_enabled") is True:
        predictive_model_id = resolved_config.get("predictive_foresight_model_id")
        if not isinstance(predictive_model_id, str) or not predictive_model_id.strip():
            raise ValueError(
                "Resolved candidate enables predictive foresight without a model_id: "
                f"{config['candidate']}"
            )
        refs.update(
            {
                "predictive_checkpoint_model_id": predictive_model_id,
                "predictive_checkpoint_sha256_declared": _registry_checkpoint_sha256(
                    model_registry, predictive_model_id
                ),
            }
        )
    return refs


def _runtime_predictive_checkpoint_refs(
    traces: dict[str, dict[str, Any]],
    *,
    expected_sha256: str | None,
) -> dict[str, str]:
    """Return paired predictive-foresight checkpoint provenance and its registry verdict."""
    conditions = ("perfect_perception", "sensor_limited")
    expected = _normalize_sha256(expected_sha256)
    requested_hashes: dict[str, str] = {}
    observed_hashes: dict[str, str] = {}
    runtime_hashes: dict[str, str] = {}
    eligible: dict[str, bool] = {}
    for condition in conditions:
        summary = traces.get(condition, {}).get("planner_summary")
        foresight = summary.get("foresight_prediction") if isinstance(summary, dict) else None
        requested = _normalize_sha256(
            foresight.get("requested_checkpoint_sha256") if isinstance(foresight, dict) else None
        )
        observed = _normalize_sha256(
            foresight.get("observed_checkpoint_sha256") if isinstance(foresight, dict) else None
        )
        is_eligible = (
            isinstance(foresight, dict)
            and foresight.get("load_status") == "loaded"
            and foresight.get("fallback_used") is False
            and requested is not None
            and observed is not None
        )
        requested_hashes[condition] = requested or "unavailable"
        observed_hashes[condition] = observed or "unavailable"
        runtime_hashes[condition] = observed if is_eligible else "unavailable"
        eligible[condition] = is_eligible

    refs = {
        "predictive_checkpoint_sha256_requested_perfect_perception": requested_hashes[
            "perfect_perception"
        ],
        "predictive_checkpoint_sha256_requested_sensor_limited": requested_hashes["sensor_limited"],
        "predictive_checkpoint_sha256_observed_perfect_perception": observed_hashes[
            "perfect_perception"
        ],
        "predictive_checkpoint_sha256_observed_sensor_limited": observed_hashes["sensor_limited"],
        "predictive_checkpoint_sha256_runtime_perfect_perception": runtime_hashes[
            "perfect_perception"
        ],
        "predictive_checkpoint_sha256_runtime_sensor_limited": runtime_hashes["sensor_limited"],
    }
    paired_hash = runtime_hashes["perfect_perception"]
    if paired_hash != "unavailable" and paired_hash == runtime_hashes["sensor_limited"]:
        refs["predictive_checkpoint_sha256_runtime"] = paired_hash
    else:
        refs["predictive_checkpoint_sha256_runtime"] = "unavailable"
    refs["predictive_checkpoint_sha256_matches_declared"] = str(
        expected is not None
        and refs["predictive_checkpoint_sha256_runtime"] != "unavailable"
        and all(
            eligible[condition]
            and requested_hashes[condition] == expected
            and observed_hashes[condition] == expected
            for condition in conditions
        )
    ).lower()
    return refs


def _runtime_checkpoint_refs(
    traces: dict[str, dict[str, Any]],
    *,
    expected_sha256: str | None = None,
    expected_predictive_sha256: str | None = None,
) -> dict[str, str]:
    """Return paired runtime checkpoint hashes and their registry-match verdict."""
    runtime_hashes: dict[str, str] = {}
    runtime_sources: dict[str, str] = {}
    for condition in ("perfect_perception", "sensor_limited"):
        summary = traces.get(condition, {}).get("planner_summary")
        provenance = summary.get("checkpoint_provenance") if isinstance(summary, dict) else None
        raw_sha256 = provenance.get("checkpoint_sha256") if isinstance(provenance, dict) else None
        raw_source = provenance.get("hash_source") if isinstance(provenance, dict) else None
        if (
            isinstance(raw_sha256, str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", raw_sha256)
            and isinstance(raw_source, str)
            and raw_source
            and provenance.get("load_succeeded") is True
        ):
            runtime_hashes[condition] = raw_sha256.lower()
            runtime_sources[condition] = raw_source
        else:
            runtime_hashes[condition] = "unavailable"
            runtime_sources[condition] = "unavailable"

    refs = {
        "checkpoint_sha256_runtime_perfect_perception": runtime_hashes["perfect_perception"],
        "checkpoint_sha256_runtime_sensor_limited": runtime_hashes["sensor_limited"],
        "checkpoint_hash_source_runtime_perfect_perception": runtime_sources["perfect_perception"],
        "checkpoint_hash_source_runtime_sensor_limited": runtime_sources["sensor_limited"],
    }
    paired_hash = runtime_hashes["perfect_perception"]
    if paired_hash != "unavailable" and paired_hash == runtime_hashes["sensor_limited"]:
        refs["checkpoint_sha256_runtime"] = paired_hash
    else:
        refs["checkpoint_sha256_runtime"] = "unavailable"
    refs["checkpoint_sha256_matches_declared"] = str(
        expected_sha256 is not None and refs["checkpoint_sha256_runtime"] == expected_sha256.lower()
    ).lower()
    if expected_predictive_sha256 is not None:
        refs.update(
            _runtime_predictive_checkpoint_refs(
                traces,
                expected_sha256=expected_predictive_sha256,
            )
        )
    return refs


def _runtime_provenance_error(input_refs: dict[str, str]) -> dict[str, Any] | None:
    """Return a blocker when runtime and declared checkpoint provenance disagree."""
    if input_refs.get("checkpoint_sha256_runtime") == "unavailable":
        return {
            "condition": "inputs",
            "status": "blocked",
            "reason": (
                "paired runtime checkpoint provenance is unavailable or did not expose "
                "the same computed digest for both conditions"
            ),
            "command": ["_runtime_checkpoint_refs"],
        }
    if input_refs.get("checkpoint_sha256_matches_declared") != "true":
        return {
            "condition": "inputs",
            "status": "blocked",
            "reason": "runtime checkpoint digest does not match the model registry digest",
            "command": ["_runtime_checkpoint_refs"],
        }
    if "predictive_checkpoint_sha256_declared" in input_refs:
        if input_refs.get("predictive_checkpoint_sha256_runtime") == "unavailable":
            return {
                "condition": "inputs",
                "status": "blocked",
                "reason": (
                    "paired predictive foresight checkpoint provenance is unavailable, "
                    "degraded, or did not expose the same valid digest for both conditions"
                ),
                "command": ["_runtime_checkpoint_refs"],
            }
        if input_refs.get("predictive_checkpoint_sha256_matches_declared") != "true":
            return {
                "condition": "inputs",
                "status": "blocked",
                "reason": (
                    "runtime predictive foresight checkpoint digest does not match "
                    "the model registry digest"
                ),
                "command": ["_runtime_checkpoint_refs"],
            }
    return None


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
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Comparator config must contain only finite JSON-compatible values"
        ) from exc
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

    try:
        input_refs = _input_refs(config_path, config)
        input_refs.update(
            _runtime_checkpoint_refs(
                traces,
                expected_sha256=input_refs.get("checkpoint_sha256_declared"),
                expected_predictive_sha256=input_refs.get("predictive_checkpoint_sha256_declared"),
            )
        )
        runtime_provenance_error = _runtime_provenance_error(input_refs)
    except (OSError, ValueError) as exc:
        runner_errors.append(
            {
                "condition": "inputs",
                "status": "blocked",
                "reason": f"comparator provenance input could not be read: {exc}",
                "command": ["_input_refs"],
            }
        )
        input_refs = {
            "config": _display_path(config_path),
            "scenario_matrix": _display_path(_repo_path(str(config["scenario_matrix"]))),
            "candidate_registry": _display_path(_repo_path(str(config["candidate_registry"]))),
        }
        runtime_provenance_error = None
    try:
        report = build_calf_legnav_comparator_report(
            traces["perfect_perception"],
            traces["sensor_limited"],
            config=config,
            input_refs=input_refs,
        )
    except (TypeError, ValueError) as exc:
        runner_errors.append(
            {
                "condition": "paired",
                "status": "blocked",
                "reason": f"trace pair failed validation: {exc}",
                "command": ["build_calf_legnav_comparator_report"],
            }
        )
        report = build_calf_legnav_comparator_report(
            _placeholder_trace(config),
            _placeholder_trace(config),
            config=config,
            input_refs=input_refs,
        )
    if runner_errors:
        report["status"] = "blocked"
        report["runner_errors"] = runner_errors
    try:
        _validate_report(report)
    except (TypeError, ValueError) as exc:
        runner_errors.append(
            {
                "condition": "paired",
                "status": "blocked",
                "reason": f"trace report failed schema validation: {exc}",
                "command": ["_validate_report"],
            }
        )
        report = build_calf_legnav_comparator_report(
            _placeholder_trace(config),
            _placeholder_trace(config),
            config=config,
            input_refs=input_refs,
        )
        report["status"] = "blocked"
        report["runner_errors"] = runner_errors
        _validate_report(report)
    if runtime_provenance_error is not None:
        runner_errors.append(runtime_provenance_error)
    if runner_errors:
        report["status"] = "blocked"
        report["runner_errors"] = runner_errors
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
