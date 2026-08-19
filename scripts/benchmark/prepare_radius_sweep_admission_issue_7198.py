#!/usr/bin/env python3
"""Prepare a current-source, fail-closed Gate 2 admission packet for #7198.

This is preparation-only: it validates Gate 1, hashes the candidate inputs,
runs zero-episode public preflights, and records private-ops evidence. It never
submits Slurm work and never emits benchmark evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.radius_sweep_manifest import (
    EXPECTED_DT,
    EXPECTED_HORIZON,
    EXPECTED_SCENARIO_COUNT,
    EXPECTED_SEED_RANGE,
    RELEASE_PLANNER_KEYS,
)
from scripts.benchmark.build_radius_sweep_manifest_issue_6642 import (
    _load_yaml,
    build_and_check,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET_SCHEMA = "issue-7198-radius-sweep-admission.v1"
DEFAULT_PACKET_CONFIG = "configs/benchmarks/issue_7198_radius_sweep_admission_v1.yaml"
DEFAULT_OUTPUT_ROOT = "output/issue_7198_radius_sweep_admission"
GATE1_SURFACES = (
    "simulator_collision_geometry",
    "obstacle_pedestrian_contact_logic",
    "feasibility_oracle",
    "metric_metadata_and_output_rows",
    "planner_inputs",
)
ARM_SPECS = (("r0p5", 0.5), ("r0p8", 0.8), ("r1p0", 1.0))
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SUMMARY_FIELDS = (
    "queue_entries",
    "ready_entries",
    "blocked_or_inactive_entries",
    "active_ledger_jobs",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(repo_root: Path, raw_path: str) -> Path:
    """Resolve a tracked path and reject paths outside the candidate checkout."""
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"declared input is outside the repository: {raw_path!r}") from exc
    return resolved


def _repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(command: list[str], *, cwd: Path, timeout: int = 300) -> dict[str, Any]:
    """Run one read-only command and retain its result for the packet."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc), "error": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "error": f"command timed out after {timeout}s",
        }
    except OSError as exc:
        return {"returncode": 126, "stdout": "", "stderr": str(exc), "error": str(exc)}
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": "",
    }


def _command_text(command: list[str]) -> str:
    return shlex.join(command)


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root, timeout=10)
    status = _run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo_root, timeout=10
    )
    head_text = head["stdout"].strip()
    status_lines = [line for line in status["stdout"].splitlines() if line.strip()]
    errors = []
    if head["returncode"] != 0 or not GIT_SHA_RE.fullmatch(head_text):
        errors.append("unable to resolve a 40-character candidate git HEAD")
    if status["returncode"] != 0:
        errors.append("unable to inspect candidate worktree status")
    return {
        "git_head": head_text,
        "working_tree_clean": not status_lines and not errors,
        "status_lines": status_lines,
        "status_line_count": len(status_lines),
        "errors": errors,
    }


def _config_path_strings(value: Any) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for nested in value.values():
            paths.update(_config_path_strings(nested))
    elif isinstance(value, list):
        for nested in value:
            paths.update(_config_path_strings(nested))
    elif isinstance(value, str) and value.startswith(
        ("configs/", "maps/", "model/", "robot_sf/", "scripts/")
    ):
        if Path(value).suffix:
            paths.add(value)
    return paths


def _input_paths(repo_root: Path, packet_path: Path, config: dict[str, Any]) -> list[str]:
    paths = _config_path_strings(config)
    paths.add(_repo_relative(repo_root, packet_path))
    return sorted(paths)


def _checksum_inventory(repo_root: Path, paths: list[str]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    errors: list[str] = []
    for raw_path in paths:
        try:
            path = _repo_path(repo_root, raw_path)
        except ValueError as exc:
            errors.append(str(exc))
            files.append({"path": raw_path, "status": "invalid", "error": str(exc)})
            continue
        if not path.is_file():
            error = f"declared input is missing or not a file: {raw_path}"
            errors.append(error)
            files.append({"path": raw_path, "status": "missing", "error": error})
            continue
        files.append(
            {
                "path": raw_path,
                "status": "verified",
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": "input-checksum-inventory.v1",
        "file_count": len(files),
        "verified_count": sum(item["status"] == "verified" for item in files),
        "errors": errors,
        "files": files,
    }


def _read_gate1_config_state(
    repo_root: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    gate = config["gate1"]
    expected_receipt = str(gate["historical_receipt_sha256"])
    expected_source = str(gate["historical_source_commit"])
    errors: list[str] = []
    arms: list[dict[str, Any]] = []
    for arm_key, radius_m in ARM_SPECS:
        raw_path = str(config["arm_configs"][arm_key])
        try:
            path = _repo_path(repo_root, raw_path)
            payload = _load_yaml(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"{arm_key}: unable to load arm config: {exc}")
            arms.append({"arm_key": arm_key, "path": raw_path, "status": "invalid"})
            continue
        binding = payload.get("radius_sweep")
        if not isinstance(binding, dict):
            binding = {}
            errors.append(f"{arm_key}: radius_sweep metadata is missing")
        checks = {
            "issue": (binding.get("issue"), 6642),
            "parent_issue": (binding.get("parent_issue"), 6600),
            "arm_key": (binding.get("arm_key"), arm_key),
            "radius_m": (binding.get("radius_m"), radius_m),
            "baseline_arm": (binding.get("baseline_arm"), radius_m == 1.0),
            "runtime_binding_status": (binding.get("runtime_binding_status"), "bound_runtime"),
            "binding_contract_version": (
                binding.get("binding_contract_version"),
                "radius_binding_canary.v1",
            ),
            "gate1_canary_issue": (binding.get("gate1_canary_issue"), 6641),
            "gate1_receipt_sha256": (binding.get("gate1_receipt_sha256"), expected_receipt),
            "gate1_source_commit": (binding.get("gate1_source_commit"), expected_source),
        }
        arm_errors = []
        for name, (actual, expected) in checks.items():
            if actual != expected:
                error = f"{arm_key}: {name}={actual!r}, expected {expected!r}"
                errors.append(error)
                arm_errors.append(error)
        arms.append(
            {
                "arm_key": arm_key,
                "radius_m": radius_m,
                "path": raw_path,
                "sha256": _sha256(path),
                "gate1_receipt_sha256": binding.get("gate1_receipt_sha256"),
                "gate1_source_commit": binding.get("gate1_source_commit"),
                "runtime_binding_status": binding.get("runtime_binding_status"),
                "status": "valid" if not arm_errors else "invalid",
            }
        )
    return {
        "historical_receipt_sha256": expected_receipt,
        "historical_source_commit": expected_source,
        "arms": arms,
        "status": "valid" if not errors else "blocked",
    }, errors


def validate_gate1_report(  # noqa: C901
    report: Any, *, packet_config: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Validate schema, identity, radii, and all 15 required binding surfaces."""
    gate = packet_config["gate1"]
    errors: list[str] = []
    if not isinstance(report, dict):
        errors.append(f"Gate 1 report must be a mapping, got {type(report).__name__}")
        report = {}
    for key, expected in (
        ("schema", gate["report_schema"]),
        ("canary_schema", gate["canary_schema"]),
        ("issue", gate["issue"]),
        ("parent_issue", gate["parent_issue"]),
        ("scenario_id", gate["scenario_id"]),
    ):
        if report.get(key) != expected:
            errors.append(f"Gate 1 report {key}={report.get(key)!r}, expected {expected!r}")
    try:
        radii = [float(value) for value in report.get("radii_m", [])]
    except (TypeError, ValueError):
        radii = []
    expected_radii = [float(value) for value in gate["required_radii_m"]]
    if radii != expected_radii:
        errors.append(f"Gate 1 radii {radii!r}, expected {expected_radii!r}")
    if report.get("go") is not True:
        errors.append("Gate 1 report does not have go=true")
    verdicts = report.get("verdicts")
    if not isinstance(verdicts, list) or len(verdicts) != len(expected_radii):
        errors.append("Gate 1 report must contain exactly three radius verdicts")
        verdicts = []
    records: list[dict[str, Any]] = []
    expected_surfaces = list(packet_config["gate1"]["required_surface_names"])
    for expected_radius, verdict in zip(expected_radii, verdicts, strict=False):
        if not isinstance(verdict, dict):
            errors.append(f"Gate 1 verdict for {expected_radius} m is not a mapping")
            continue
        actual_radius = verdict.get("target_radius_m")
        try:
            radius_matches = float(actual_radius) == expected_radius
        except (TypeError, ValueError):
            radius_matches = False
        if not radius_matches:
            errors.append(f"Gate 1 verdict radius {actual_radius!r}, expected {expected_radius!r}")
        surfaces = verdict.get("surfaces")
        valid_surfaces = (
            [item for item in surfaces if isinstance(item, dict)]
            if isinstance(surfaces, list)
            else []
        )
        invalid_surface_count = (
            len(surfaces) - len(valid_surfaces) if isinstance(surfaces, list) else 0
        )
        if invalid_surface_count:
            errors.append(
                f"Gate 1 surfaces at {expected_radius} m contain "
                f"{invalid_surface_count} non-mapping entries"
            )
        names = [item.get("surface") for item in valid_surfaces]
        if names != expected_surfaces:
            errors.append(f"Gate 1 surface roster {names!r}, expected {expected_surfaces!r}")
        if not isinstance(surfaces, list) or len(surfaces) != len(expected_surfaces):
            errors.append(f"Gate 1 must contain five surfaces at {expected_radius} m")
        unbound = (
            [item.get("surface") for item in valid_surfaces if not item.get("bound")]
            if isinstance(surfaces, list)
            else expected_surfaces
        )
        if unbound:
            errors.append(f"Gate 1 has unbound surfaces at {expected_radius} m: {unbound!r}")
        records.append(
            {
                "target_radius_m": actual_radius,
                "go": verdict.get("go"),
                "surface_count": len(surfaces) if isinstance(surfaces, list) else 0,
                "bound_surface_count": (
                    sum(bool(item.get("bound")) for item in valid_surfaces)
                    if isinstance(surfaces, list)
                    else 0
                ),
            }
        )
    return {
        "schema": report.get("schema"),
        "canary_schema": report.get("canary_schema"),
        "issue": report.get("issue"),
        "parent_issue": report.get("parent_issue"),
        "scenario_id": report.get("scenario_id"),
        "radii_m": radii,
        "go": report.get("go"),
        "verdicts": records,
        "status": "valid" if not errors else "blocked",
    }, errors


def _validate_staged_arm_record(index: int, staged_arm: Any) -> list[str]:
    """Validate one staged arm identity and checksum record."""
    if not isinstance(staged_arm, dict):
        return [f"checkpoint preflight arm {index} must be a mapping"]
    errors: list[str] = []
    if staged_arm.get("status") not in {"present_local", "staged"}:
        errors.append(
            f"checkpoint preflight arm {index} status={staged_arm.get('status')!r} is not submit-safe"
        )
    checkpoint_sha256 = staged_arm.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", checkpoint_sha256
    ):
        errors.append(f"checkpoint preflight arm {index} is missing a verified SHA-256")
    return errors


def _validate_submit_safe_checkpoint(checkpoint: Any) -> tuple[dict[str, Any], list[str]]:
    """Validate and retain the staged checkpoint evidence from one arm preflight."""
    errors: list[str] = []
    if not isinstance(checkpoint, dict):
        return {}, ["checkpoint_preflight must be a mapping"]

    if checkpoint.get("mode") != "enforced_staged":
        errors.append(
            f"checkpoint preflight mode={checkpoint.get('mode')!r}, expected 'enforced_staged'"
        )
    if checkpoint.get("stage") is not True:
        errors.append("checkpoint preflight stage must be true for submit-safe admission")
    submit_safe = checkpoint.get("submit_safe") is True
    if not submit_safe:
        errors.append("checkpoint preflight must report submit_safe=true")

    staged_arms = checkpoint.get("arms")
    if not isinstance(staged_arms, list):
        errors.append("checkpoint preflight arms must be a list")
        staged_arms = []
    expected_arm_count = checkpoint.get("checked")
    if expected_arm_count != len(staged_arms):
        errors.append(
            f"checkpoint preflight checked={expected_arm_count!r}, expected {len(staged_arms)} arm records"
        )
    for index, staged_arm in enumerate(staged_arms):
        errors.extend(_validate_staged_arm_record(index, staged_arm))

    return {
        "mode": checkpoint.get("mode"),
        "stage": checkpoint.get("stage"),
        "checked": checkpoint.get("checked"),
        "resolved": checkpoint.get("resolved"),
        "submit_safe": submit_safe,
        "arms": staged_arms,
    }, errors


def validate_preflight_payload(
    payload: Any, *, arm_key: str, radius_m: float, config_sha256: str
) -> dict[str, Any]:
    """Validate one public preflight without accepting any episode output."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        errors.append(f"preflight payload must be a mapping, got {type(payload).__name__}")
        payload = {}
    binding = payload.get("radius_binding")
    if not isinstance(binding, dict):
        errors.append("radius_binding must be a mapping")
        binding = {}
    checks = {
        "schema_version": (payload.get("schema_version"), "benchmark-preflight-validate-config.v1"),
        "config_sha256": (payload.get("config_sha256"), config_sha256),
        "scenario_count": (payload.get("scenario_count"), EXPECTED_SCENARIO_COUNT),
        "planner_count": (payload.get("planner_count"), len(RELEASE_PLANNER_KEYS)),
        "horizon": (payload.get("horizon"), EXPECTED_HORIZON),
        "dt": (payload.get("dt"), EXPECTED_DT),
        "binding_issue": (binding.get("issue"), 6642),
        "binding_parent_issue": (binding.get("parent_issue"), 6600),
        "binding_arm_key": (binding.get("arm_key"), arm_key),
        "binding_radius_m": (binding.get("radius_m"), radius_m),
        "binding_status": (binding.get("status"), "bound_runtime"),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"{name}={actual!r}, expected {expected!r}")
    seed_policy = payload.get("seed_policy")
    if not isinstance(seed_policy, dict):
        errors.append("seed_policy must be a mapping")
        seed_policy = {}
    resolved_seeds = seed_policy.get("resolved_seeds")
    expected_seeds = list(range(EXPECTED_SEED_RANGE[0], EXPECTED_SEED_RANGE[1] + 1))
    if resolved_seeds != expected_seeds:
        errors.append(f"resolved seeds {resolved_seeds!r}, expected {expected_seeds!r}")
    checkpoint, checkpoint_errors = _validate_submit_safe_checkpoint(
        payload.get("checkpoint_preflight")
    )
    errors.extend(checkpoint_errors)
    submit_safe = checkpoint.get("submit_safe") is True
    episode_count = payload.get("episodes", 0)
    if episode_count not in (None, 0):
        errors.append(f"preflight reported nonzero episodes: {episode_count!r}")
    return {
        "campaign_id": payload.get("campaign_id"),
        "config_path": payload.get("config_path"),
        "config_sha256": payload.get("config_sha256"),
        "radius_binding": binding,
        "planner_count": payload.get("planner_count"),
        "scenario_count": payload.get("scenario_count"),
        "seed_range": [expected_seeds[0], expected_seeds[-1]],
        "horizon": payload.get("horizon"),
        "dt": payload.get("dt"),
        "checkpoint_preflight": {
            "mode": checkpoint.get("mode"),
            "stage": checkpoint.get("stage"),
            "checked": checkpoint.get("checked"),
            "resolved": checkpoint.get("resolved"),
            "submit_safe": submit_safe,
            "arms": checkpoint.get("arms", []),
        },
        "episodes": 0,
        "structural_status": "passed" if not errors else "blocked",
        "errors": errors,
    }


def _preflight_command(config_path: str, output_root: Path, campaign_id: str) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "scripts/tools/run_camera_ready_benchmark.py",
        "--config",
        config_path,
        "--output-root",
        str(output_root),
        "--campaign-id",
        campaign_id,
        "--skip-publication-bundle",
        "--mode",
        "preflight",
        "--checkpoint-preflight-mode",
        "enforced_staged",
        "--log-level",
        "ERROR",
    ]


def _run_preflights(
    repo_root: Path,
    packet_config: dict[str, Any],
    output_root: Path,
    blockers: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for arm_key, radius_m in ARM_SPECS:
        config_path = str(packet_config["arm_configs"][arm_key])
        campaign_id = f"issue7198-{arm_key}"
        arm_output_root = output_root / "preflight" / arm_key
        command = _preflight_command(config_path, arm_output_root, campaign_id)
        campaign_root = arm_output_root / campaign_id
        validate_path = campaign_root / "preflight" / "validate_config.json"
        base_record: dict[str, Any] = {
            "arm_key": arm_key,
            "radius_m": radius_m,
            "campaign_id": campaign_id,
            "config_path": config_path,
            "config_sha256": None,
            "command": _command_text(command),
            "validate_config_path": _repo_relative(repo_root, validate_path),
        }
        try:
            config_file = _repo_path(repo_root, config_path)
            if not config_file.is_file():
                raise FileNotFoundError(config_file)
            config_sha = _sha256(config_file)
        except (OSError, ValueError) as exc:
            reason = f"{arm_key}: unable to resolve config for preflight: {exc}"
            blockers.append(reason)
            base_record.update(
                {
                    "returncode": None,
                    "stdout_path": None,
                    "stderr_path": None,
                    "structural_status": "blocked",
                    "errors": [reason],
                    "episodes": 0,
                }
            )
            records.append(base_record)
            continue
        result = _run(command, cwd=repo_root, timeout=900)
        stdout_path = output_root / "logs" / f"preflight_{arm_key}.stdout.txt"
        stderr_path = output_root / "logs" / f"preflight_{arm_key}.stderr.txt"
        _write_text(stdout_path, result["stdout"])
        _write_text(stderr_path, result["stderr"])
        record = {
            **base_record,
            "config_sha256": config_sha,
            "returncode": result["returncode"],
            "stdout_path": _repo_relative(repo_root, stdout_path),
            "stderr_path": _repo_relative(repo_root, stderr_path),
        }
        if result["returncode"] != 0:
            reason = f"{arm_key}: public preflight exited {result['returncode']}"
            blockers.append(reason)
            record.update({"structural_status": "blocked", "errors": [reason], "episodes": 0})
            records.append(record)
            continue
        if not validate_path.is_file():
            reason = f"{arm_key}: preflight did not emit {validate_path}"
            blockers.append(reason)
            record.update({"structural_status": "blocked", "errors": [reason], "episodes": 0})
            records.append(record)
            continue
        try:
            payload = json.loads(validate_path.read_text(encoding="utf-8"))
            validation = validate_preflight_payload(
                payload, arm_key=arm_key, radius_m=radius_m, config_sha256=config_sha
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            reason = f"{arm_key}: malformed validate_config.json: {exc}"
            blockers.append(reason)
            record.update({"structural_status": "blocked", "errors": [reason], "episodes": 0})
            records.append(record)
            continue
        episode_files = [
            _repo_relative(repo_root, path)
            for path in campaign_root.rglob("*")
            if path.is_file()
            and path.name.startswith("episodes")
            and path.suffix in {".jsonl", ".parquet"}
        ]
        if episode_files:
            validation["errors"].append(f"production episode files were emitted: {episode_files!r}")
            validation["structural_status"] = "blocked"
        for error in validation["errors"]:
            blockers.append(f"{arm_key}: {error}")
        if validation["checkpoint_preflight"]["submit_safe"] is not True:
            blockers.append(
                f"{arm_key}: checkpoint preflight is not submit-safe; rerun with "
                "--checkpoint-preflight-mode=enforced_staged"
            )
        record.update(validation)
        record["episode_files"] = episode_files
        records.append(record)
    return records


def _parse_queue_summary(text: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for field in SUMMARY_FIELDS:
        match = re.search(rf"^- {re.escape(field)}:\s*(\d+)\s*$", text, re.MULTILINE)
        summary[field] = int(match.group(1)) if match else None
    return summary


def _parse_route_output(text: str) -> dict[str, Any]:
    selected = re.search(r"^\s+selected:\s*(\S+)\s*$", text, re.MULTILINE)
    elapsed = re.search(r"^\s+- estimated elapsed\s+(\d+)s\s*$", text, re.MULTILINE)
    score = re.search(r"^\s+- score\s+([0-9.]+)\s*$", text, re.MULTILINE)
    return {
        "selected_route": selected.group(1) if selected else None,
        "estimated_elapsed_sec": int(elapsed.group(1)) if elapsed else None,
        "score": float(score.group(1)) if score else None,
        "status": "parsed" if selected else "unparsed",
    }


def _private_ops_snapshot(  # noqa: C901, PLR0912, PLR0915
    repo_root: Path,
    packet_config: dict[str, Any],
    output_root: Path,
    blockers: list[str],
) -> dict[str, Any]:
    private_cfg = packet_config["private_ops"]
    configured_root = os.environ.get(str(private_cfg["root_env"]))
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
    else:
        fallback = Path(str(private_cfg["fallback_relative_path"]))
        root = (repo_root / fallback).resolve()
        if not root.is_dir():
            common_git = _run(["git", "rev-parse", "--git-common-dir"], cwd=repo_root, timeout=10)
            common_path = Path(common_git["stdout"].strip())
            if not common_path.is_absolute():
                common_path = repo_root / common_path
            main_root = common_path.resolve().parent
            root = (main_root.parent / fallback.name).resolve()
    snapshot: dict[str, Any] = {
        "root": str(root),
        "root_source": "environment" if configured_root else "fallback_relative_path",
        "available": root.is_dir(),
        "working_tree_clean": False,
        "queue_summary": {},
        "route": {},
        "scripts": {},
    }
    if not root.is_dir():
        blockers.append(f"private-ops checkout is unavailable: {root}")
        return snapshot

    status = _run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        timeout=20,
    )
    status_lines = [line for line in status["stdout"].splitlines() if line.strip()]
    snapshot["working_tree_clean"] = status["returncode"] == 0 and not status_lines
    snapshot["working_tree_status_line_count"] = len(status_lines)
    snapshot["working_tree_status_sample"] = status_lines[:20]
    if status["returncode"] != 0:
        blockers.append("private-ops working-tree status could not be read")
    elif status_lines:
        blockers.append(
            f"private-ops checkout is dirty ({len(status_lines)} status lines); no clean route is available"
        )

    root_paths = {
        name: root / str(private_cfg[key])
        for name, key in (
            ("queue_summary", "queue_summary_script"),
            ("route", "route_script"),
            ("preflight", "preflight_script"),
            ("submission", "submission_entrypoint"),
        )
    }
    snapshot["scripts"] = {
        name: {"path": str(path), "exists": path.is_file()} for name, path in root_paths.items()
    }
    for name, path in root_paths.items():
        if not path.is_file():
            blockers.append(f"private-ops {name} entrypoint is missing: {path}")

    queue_script = root_paths["queue_summary"]
    if queue_script.is_file():
        command = ["bash", str(queue_script), "--limit", "100"]
        result = _run(command, cwd=repo_root, timeout=60)
        stdout_path = output_root / "private_ops" / "queue_summary.txt"
        stderr_path = output_root / "private_ops" / "queue_summary.stderr.txt"
        _write_text(stdout_path, result["stdout"])
        _write_text(stderr_path, result["stderr"])
        summary = _parse_queue_summary(result["stdout"])
        snapshot["queue_summary"] = {
            "command": _command_text(command),
            "returncode": result["returncode"],
            "summary": summary,
            "stdout_path": _repo_relative(repo_root, stdout_path),
            "stderr_path": _repo_relative(repo_root, stderr_path),
        }
        if result["returncode"] != 0:
            blockers.append("private-ops queue summary failed")
        expected_ready = int(private_cfg["queue_must_have_ready_entries"])
        if summary.get("ready_entries") != expected_ready:
            blockers.append(
                f"private-ops queue has {summary.get('ready_entries')!r} ready entries; "
                f"expected exactly {expected_ready}"
            )

    route_script = root_paths["route"]
    if route_script.is_file():
        command = [
            "python3",
            str(route_script),
            "--job-class",
            str(private_cfg["route_job_class"]),
            "--cpus",
            str(private_cfg["route_cpus"]),
            "--gpus",
            str(private_cfg["route_gpus"]),
            "--mem-gb",
            str(private_cfg["route_mem_gb"]),
            "--limit",
            "100",
            "--explain",
        ]
        if private_cfg.get("route_no_live"):
            command.append("--no-live")
        if private_cfg.get("route_capacity_fill"):
            command.append("--capacity-fill")
        result = _run(command, cwd=repo_root, timeout=240)
        stdout_path = output_root / "private_ops" / "route_job.txt"
        stderr_path = output_root / "private_ops" / "route_job.stderr.txt"
        _write_text(stdout_path, result["stdout"])
        _write_text(stderr_path, result["stderr"])
        route = _parse_route_output(result["stdout"])
        route.update(
            {
                "command": _command_text(command),
                "returncode": result["returncode"],
                "stdout_path": _repo_relative(repo_root, stdout_path),
                "stderr_path": _repo_relative(repo_root, stderr_path),
                "live_capacity_checked": "--no-live" not in command,
            }
        )
        snapshot["route"] = route
        if result["returncode"] not in (0, 2):
            blockers.append("private-ops route evaluation failed")
        if not route.get("selected_route"):
            blockers.append("private-ops route evaluation did not select a route")
        if not route["live_capacity_checked"]:
            blockers.append(
                "private-ops route evidence is static; live capacity remains unverified"
            )
    return snapshot


def _submission_command(
    packet_config: dict[str, Any], private_snapshot: dict[str, Any], *, artifact_manifest: str
) -> str:
    """Render a copyable submission template with an expandable results URI."""
    route = private_snapshot.get("route") or {}
    selected = route.get("selected_route") or "<selected-route>"
    cluster = str(selected).split(":", 1)[0]
    results_expr = "$" + "{ROBOT_SF_RADIUS_SWEEP_RESULTS_URI}/{job_id}"
    command = shlex.join(
        [
            "bash",
            "ops/jobs/scripts/submit_and_record.sh",
            "--cluster",
            cluster,
            "--route-id",
            str(selected),
            "--partition",
            "<route-partition>",
            "--job-class",
            str(packet_config["resources"]["job_class"]),
            "--cpus",
            str(packet_config["resources"]["cpus"]),
            "--gpus",
            str(packet_config["resources"]["gpus"]),
            "--mem-gb",
            str(packet_config["resources"]["mem_gb"]),
            "--config",
            "configs/benchmarks/issue_6642_radius_sweep_arm_<radius>.yaml",
            "--public-issue",
            "ll7/robot_sf_ll7#7198",
            "--remote-results",
            results_expr,
            "--artifact-manifest",
            artifact_manifest,
            "--script",
            "<remote-slurm-script>",
        ]
    )
    return command.replace(shlex.quote(results_expr), results_expr, 1)


def prepare_packet(  # noqa: PLR0915
    *,
    repo_root: Path = REPO_ROOT,
    packet_config_path: Path | None = None,
    output_root: Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Build and write the admission packet; return the packet and process status."""
    packet_config_path = packet_config_path or repo_root / DEFAULT_PACKET_CONFIG
    output_root = output_root or repo_root / DEFAULT_OUTPUT_ROOT
    packet_config = _load_yaml(packet_config_path)
    blockers: list[str] = []

    candidate = _git_snapshot(repo_root)
    blockers.extend(candidate["errors"])
    if not candidate["working_tree_clean"]:
        blockers.append("candidate worktree is not clean; the launch commit is not immutable")

    input_paths = _input_paths(repo_root, packet_config_path, packet_config)
    input_paths.extend(
        [
            "scripts/benchmark/run_radius_binding_canary_issue_6641.py",
            "scripts/benchmark/build_radius_sweep_manifest_issue_6642.py",
            "scripts/tools/run_camera_ready_benchmark.py",
            "robot_sf/benchmark/radius_sweep_manifest.py",
        ]
    )
    checksums = _checksum_inventory(repo_root, sorted(set(input_paths)))
    blockers.extend(checksums["errors"])

    gate1_config, gate1_config_errors = _read_gate1_config_state(repo_root, packet_config)
    blockers.extend(gate1_config_errors)

    manifest_root = output_root / "manifest"
    manifest_command = [
        "uv",
        "run",
        "python",
        "scripts/benchmark/build_radius_sweep_manifest_issue_6642.py",
        "--manifest-config",
        str(packet_config["manifest_config"]),
        "--out",
        _repo_relative(repo_root, manifest_root),
    ]
    try:
        _manifest, manifest_check, manifest_path, check_path = build_and_check(
            _repo_path(repo_root, str(packet_config["manifest_config"])),
            manifest_root,
            repo_root,
            check_only=False,
        )
        manifest_record = {
            "status": "passed" if manifest_check.get("passes") else "blocked",
            "check": manifest_check,
            "manifest_path": _repo_relative(repo_root, manifest_path) if manifest_path else None,
            "check_path": _repo_relative(repo_root, check_path) if check_path else None,
        }
        if not manifest_check.get("passes"):
            blockers.extend(
                f"manifest: {violation}" for violation in manifest_check.get("violations", [])
            )
        if (
            manifest_check.get("expected_total_rows")
            != packet_config["fixed_factors"]["expected_total_rows"]
        ):
            blockers.append("manifest expected row count differs from the admission contract")
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        manifest_record = {"status": "blocked", "error": str(exc)}
        blockers.append(f"manifest check failed: {exc}")

    gate1_output = output_root / "gate1" / "radius_binding_canary_6641_current.json"
    gate1_command = [
        "uv",
        "run",
        "python",
        "scripts/benchmark/run_radius_binding_canary_issue_6641.py",
        "--out-json",
        str(gate1_output),
    ]
    gate1_result = _run(gate1_command, cwd=repo_root, timeout=300)
    gate1_stdout = output_root / "logs" / "gate1.stdout.txt"
    gate1_stderr = output_root / "logs" / "gate1.stderr.txt"
    _write_text(gate1_stdout, gate1_result["stdout"])
    _write_text(gate1_stderr, gate1_result["stderr"])
    gate1_record: dict[str, Any] = {
        "command": _command_text(gate1_command),
        "returncode": gate1_result["returncode"],
        "report_path": _repo_relative(repo_root, gate1_output),
        "stdout_path": _repo_relative(repo_root, gate1_stdout),
        "stderr_path": _repo_relative(repo_root, gate1_stderr),
    }
    gate1_errors: list[str] = []
    if gate1_output.is_file():
        try:
            report = json.loads(gate1_output.read_text(encoding="utf-8"))
            gate1_summary, gate1_errors = validate_gate1_report(report, packet_config=packet_config)
            gate1_record.update(
                {
                    "report_sha256": _sha256(gate1_output),
                    "summary": gate1_summary,
                }
            )
            blockers.extend(gate1_errors)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            blockers.append(f"Gate 1 current report is malformed: {exc}")
            gate1_record["status"] = "blocked"
    else:
        blockers.append("Gate 1 current canary did not emit its report")
    if gate1_result["returncode"] != 0:
        blockers.append(f"Gate 1 canary exited {gate1_result['returncode']}")
    gate1_record.setdefault("status", "valid" if not gate1_errors else "blocked")

    preflights = _run_preflights(repo_root, packet_config, output_root, blockers)
    private_ops = _private_ops_snapshot(repo_root, packet_config, output_root, blockers)
    remote_env = str(packet_config["artifacts"]["remote_results_uri_env"])
    remote_results_uri = os.environ.get(remote_env)
    if not remote_results_uri:
        blockers.append(f"durable remote results route is unset: {remote_env}")

    unique_blockers = list(dict.fromkeys(blockers))
    verdict = "ready_for_authorized_submission" if not unique_blockers else "blocked"
    arm_records: dict[str, dict[str, Any]] = {}
    for arm_key, _radius_m in ARM_SPECS:
        arm_path = packet_config["arm_configs"][arm_key]
        arm_hash = next(
            (
                item["sha256"]
                for item in checksums["files"]
                if item["path"] == arm_path and item["status"] == "verified"
            ),
            None,
        )
        arm_records[arm_key] = {"path": arm_path, "sha256": arm_hash}

    artifact_record = {
        **packet_config["artifacts"],
        "local_packet_root": _repo_relative(repo_root, output_root),
        "public_manifest_path": _repo_relative(repo_root, output_root / "packet.json"),
        "remote_results_uri": remote_results_uri,
        "episodes_run": 0,
    }
    packet = {
        "schema_version": PACKET_SCHEMA,
        "generated_at_utc": _utc_now(),
        "issue": int(packet_config["issue"]),
        "campaign_issue": int(packet_config["campaign_issue"]),
        "parent_issue": int(packet_config["parent_issue"]),
        "title": packet_config["title"],
        "claim_boundary": packet_config["claim_boundary"],
        "verdict": verdict,
        "status": verdict,
        "candidate_commit": candidate,
        "input_checksums": checksums,
        "gate1_config_state": gate1_config,
        "gate1_current_canary": gate1_record,
        "manifest_check": manifest_record,
        "fixed_factors": packet_config["fixed_factors"],
        "arm_configs": arm_records,
        "preflights": preflights,
        "private_ops": private_ops,
        "resources": packet_config["resources"],
        "artifacts": artifact_record,
        "missingness_policy": packet_config["missingness_policy"],
        "commands": {
            "manifest_check": _command_text(manifest_command),
            "gate1_current_canary": _command_text(gate1_command),
            "arm_preflights": [record["command"] for record in preflights],
            "private_submission_entrypoint": _submission_command(
                packet_config,
                private_ops,
                artifact_manifest=artifact_record["public_manifest_path"],
            ),
            "submission_executed": False,
        },
        "blockers": unique_blockers,
        "first_blocker": unique_blockers[0] if unique_blockers else None,
        "production": {
            "submission_authorized": False,
            "submission_executed": False,
            "episodes_run": 0,
            "evidence_status": "not_benchmark_evidence",
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "packet.json", packet)
    return packet, 0 if verdict == "ready_for_authorized_submission" else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the packet configuration and output-root arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-config", type=Path, default=Path(DEFAULT_PACKET_CONFIG))
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the packet and return its fail-closed verdict status."""
    args = parse_args(argv)
    packet_path = (
        args.packet_config if args.packet_config.is_absolute() else REPO_ROOT / args.packet_config
    )
    output_root = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    try:
        packet, status = prepare_packet(
            repo_root=REPO_ROOT, packet_config_path=packet_path, output_root=output_root
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(
            json.dumps(
                {"schema_version": PACKET_SCHEMA, "verdict": "blocked", "error": str(exc)},
                indent=2,
            )
        )
        return 2
    print(json.dumps(packet, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
