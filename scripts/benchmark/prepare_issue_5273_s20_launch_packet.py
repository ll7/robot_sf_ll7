#!/usr/bin/env python3
"""Prepare the deterministic, fail-closed launch packet for issue #5273.

This command inventories the committed S20 per-planner configs, validates their
immutable provenance, and runs the canonical camera-ready *preflight* path. It
never runs a campaign, stages production checkpoints, or submits Slurm. A
blocked packet is still valid evidence: every blocker is recorded and
``submission_allowed`` remains false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from robot_sf.evidence.writers import write_json

SCHEMA_VERSION = "issue-5273-s20-launch-packet.v1"
ISSUE = 5273
PARENT_ISSUE = 5273
DEFAULT_SOURCE_DIRECTORY = (
    "configs/benchmarks/splits/paper_experiment_matrix_v1_scenario_horizons_h500_s20"
)
DEFAULT_MANIFEST = f"{DEFAULT_SOURCE_DIRECTORY}/split_manifest.json"
DEFAULT_OUTPUT = "docs/context/evidence/issue_5273_s20_launch_packet.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HORIZON = re.compile(r"(?:^|[_-])h(?P<steps>[0-9]+)(?:[_./-]|$)", re.IGNORECASE)
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")

CLAIM_BOUNDARY = (
    "Preparation and preflight provenance only. No Slurm job, benchmark campaign, planner "
    "execution, checkpoint staging, or scientific result is claimed."
)
AGGREGATION_CONTRACT = {
    "accepted_row_requirements": {
        "status": "ok",
        "execution_mode": "native",
        "readiness_status": "native",
        "availability_status": "available",
        "benchmark_success": True,
        "provenance": [
            "declared_config_sha256",
            "campaign_manifest_config_sha256",
            "declared_planner_key",
            "expected_row_identity",
        ],
    },
    "declared_adapter_policy": (
        "A declared adapter may be admitted only with explicit adapter provenance and the same "
        "identity checks; no adapter row is admitted by this S20 packet."
    ),
    "excluded_row_classes": [
        "fallback",
        "degraded",
        "failed",
        "missing",
        "duplicate",
        "provenance_invalid",
        "adapter_undeclared",
    ],
    "fail_closed": True,
}

PreflightRunner = Callable[[Path, Path, str], dict[str, Any]]


def _repo_root() -> Path:
    """Return the repository root for this script."""
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    """Return the raw-byte SHA-256 digest for ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value with stable serialization."""
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object with a path-specific error."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping with a path-specific error."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read YAML mapping {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _relative_path(repo_root: Path, path: Path) -> str:
    """Return a portable repository-relative path."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside repository: {path}") from exc


def _resolve_relative(repo_root: Path, value: str, *, field: str) -> Path:
    """Resolve a required repository-relative path and reject traversal."""
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{field} must be repository-relative: {value}")
    resolved = (repo_root / candidate).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes repository: {value}") from exc
    return resolved


def _blocker(code: str, detail: str, *, arm: str | None = None) -> dict[str, str]:
    """Build a stable blocker object."""
    payload = {"code": code, "detail": detail}
    if arm is not None:
        payload["arm"] = arm
    return payload


def _append_blocker(blockers: list[dict[str, str]], item: dict[str, str]) -> None:
    """Append a blocker once."""
    if item not in blockers:
        blockers.append(item)


def _fallback_settings(value: Any, *, path: str = "planner") -> list[dict[str, str]]:
    """Find explicit fallback/degraded settings in a planner config."""
    found: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key in sorted(value):
            child_path = f"{path}.{key}"
            child = value[key]
            key_text = str(key).lower()
            if "fallback" in key_text or "degraded" in key_text:
                found.append({"path": child_path, "value": str(child)})
            elif isinstance(child, str) and child.strip().lower() in {
                "fallback",
                "degraded",
                "allow_fallback",
            }:
                found.append({"path": child_path, "value": child})
            found.extend(_fallback_settings(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_fallback_settings(child, path=f"{path}[{index}]"))
    return found


def _git_commit(repo_root: Path) -> str:
    """Return the exact source commit, or an explicit unknown marker in unit tests."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _canonical_preflight_command(config_path: str, planner_key: str) -> str:
    """Return the reproducible, non-executing preflight command."""
    campaign_id = f"issue-5273-s20-preflight-{planner_key}"
    return (
        "uv run python scripts/tools/run_camera_ready_benchmark.py"
        f" --config {config_path}"
        " --output-root output/benchmarks/issue_5273_s20_preflight"
        f" --campaign-id {campaign_id}"
        " --skip-publication-bundle --mode preflight"
        " --checkpoint-preflight-mode metadata_only"
    )


def _inventory_children(  # noqa: C901, PLR0912, PLR0915
    repo_root: Path, manifest_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    """Validate the split manifest and return child config payloads."""
    blockers: list[dict[str, str]] = []
    try:
        manifest = _read_json(manifest_path)
    except ValueError as exc:
        return {}, [], [_blocker("manifest_invalid", str(exc))]

    source_ref = manifest.get("source_config")
    source_digest = manifest.get("source_sha256")
    if not isinstance(source_ref, str) or not source_ref:
        _append_blocker(
            blockers, _blocker("source_config_missing", "manifest source_config is missing")
        )
    if not isinstance(source_digest, str) or not _SHA256.fullmatch(source_digest):
        _append_blocker(
            blockers, _blocker("source_hash_missing", "manifest source_sha256 is invalid")
        )
    if (
        isinstance(source_ref, str)
        and isinstance(source_digest, str)
        and _SHA256.fullmatch(source_digest)
    ):
        try:
            source_path = _resolve_relative(repo_root, source_ref, field="source_config")
            observed = _sha256(source_path)
            if observed != source_digest:
                _append_blocker(
                    blockers,
                    _blocker(
                        "source_hash_mismatch",
                        f"{source_ref}: expected {source_digest}, observed {observed}",
                    ),
                )
        except (OSError, ValueError) as exc:
            _append_blocker(blockers, _blocker("source_config_invalid", str(exc)))

    children = manifest.get("children")
    if not isinstance(children, list) or not children:
        _append_blocker(
            blockers, _blocker("children_missing", "manifest children must be non-empty")
        )
        return manifest, [], blockers

    manifest_dir = manifest_path.resolve().parent
    declared_names: list[str] = []
    for index, child in enumerate(children):
        if not isinstance(child, dict):
            _append_blocker(
                blockers, _blocker("child_invalid", f"children[{index}] is not an object")
            )
            continue
        filename = child.get("filename")
        if isinstance(filename, str):
            declared_names.append(filename)
    observed_names = sorted(path.name for path in manifest_dir.glob("*.yaml"))
    if sorted(declared_names) != observed_names:
        _append_blocker(
            blockers,
            _blocker(
                "inventory_drift",
                "manifest YAML inventory does not exactly match committed per-arm YAML files",
            ),
        )

    validated: list[dict[str, Any]] = []
    seen_planners: set[str] = set()
    seen_filenames: set[str] = set()
    expected_source = str(source_ref) if isinstance(source_ref, str) else ""
    expected_source_hash = str(source_digest) if isinstance(source_digest, str) else ""
    for index, child in enumerate(children):
        if not isinstance(child, dict):
            continue
        filename = child.get("filename")
        digest = child.get("sha256")
        planners = child.get("planner_keys")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".yaml")
        ):
            _append_blocker(
                blockers,
                _blocker(
                    "child_path_invalid", f"children[{index}].filename is not a safe YAML name"
                ),
            )
            continue
        if filename in seen_filenames:
            _append_blocker(
                blockers, _blocker("duplicate_child", f"duplicate child filename: {filename}")
            )
            continue
        seen_filenames.add(filename)
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            _append_blocker(
                blockers, _blocker("child_hash_missing", f"invalid SHA-256 for {filename}")
            )
            continue
        if not isinstance(planners, list) or len(planners) != 1 or not isinstance(planners[0], str):
            _append_blocker(
                blockers,
                _blocker(
                    "arm_identity_invalid", f"{filename} must declare exactly one planner key"
                ),
            )
            continue
        planner_key = planners[0]
        if planner_key in seen_planners:
            _append_blocker(
                blockers, _blocker("duplicate_arm_identity", planner_key, arm=planner_key)
            )
            continue
        seen_planners.add(planner_key)

        config_path = manifest_dir / filename
        try:
            config_payload = _read_yaml(config_path)
            observed_digest = _sha256(config_path)
        except (OSError, ValueError) as exc:
            _append_blocker(blockers, _blocker("child_config_invalid", str(exc), arm=planner_key))
            continue
        if observed_digest != digest:
            _append_blocker(
                blockers,
                _blocker(
                    "child_hash_mismatch",
                    f"{filename}: expected {digest}, observed {observed_digest}",
                    arm=planner_key,
                ),
            )
        planner_entries = config_payload.get("planners")
        config_planner_keys = (
            [
                entry.get("key")
                for entry in planner_entries
                if isinstance(entry, dict) and isinstance(entry.get("key"), str)
            ]
            if isinstance(planner_entries, list)
            else []
        )
        if config_planner_keys != [planner_key]:
            _append_blocker(
                blockers,
                _blocker(
                    "config_arm_identity_mismatch",
                    f"{filename}: manifest {planner_key!r}, config {config_planner_keys!r}",
                    arm=planner_key,
                ),
            )
        provenance = config_payload.get("split_provenance")
        if not isinstance(provenance, dict) or any(
            provenance.get(field) != expected
            for field, expected in (
                ("source_config", expected_source),
                ("source_sha256", expected_source_hash),
                ("split_mode", manifest.get("split_mode")),
                ("arm_key", planner_key),
                ("arm_total", len(children)),
            )
        ):
            _append_blocker(
                blockers,
                _blocker(
                    "config_provenance_invalid",
                    f"{filename}: split_provenance drift",
                    arm=planner_key,
                ),
            )
        validated.append(
            {
                "planner_key": planner_key,
                "config_path": _relative_path(repo_root, config_path),
                "config_sha256": digest,
                "observed_sha256": observed_digest,
                "config": config_payload,
                "fallback_settings": _fallback_settings(
                    config_payload.get("planners", [{}])[0]
                    if isinstance(config_payload.get("planners"), list)
                    and config_payload.get("planners")
                    else {}
                ),
            }
        )
    validated.sort(key=lambda child: child["planner_key"])
    return manifest, validated, blockers


def _sanitize_error(repo_root: Path, message: str) -> str:
    """Remove machine-specific paths from a preflight error."""
    sanitized = message.replace(str(repo_root), "<repo>")
    return re.sub(r"/tmp/[A-Za-z0-9_.-]+", "<temporary-path>", sanitized)


def _run_canonical_preflight(
    repo_root: Path, config_path: Path, planner_key: str
) -> dict[str, Any]:
    """Run canonical metadata-only preflight without entering campaign execution."""
    command = _canonical_preflight_command(_relative_path(repo_root, config_path), planner_key)
    try:
        with tempfile.TemporaryDirectory(prefix="issue5273-s20-preflight-") as temporary_root:
            completed = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/tools/run_camera_ready_benchmark.py",
                    "--config",
                    _relative_path(repo_root, config_path),
                    "--output-root",
                    temporary_root,
                    "--campaign-id",
                    f"issue-5273-s20-preflight-{planner_key}",
                    "--skip-publication-bundle",
                    "--mode",
                    "preflight",
                    "--checkpoint-preflight-mode",
                    "metadata_only",
                    "--log-level",
                    "ERROR",
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            try:
                result = json.loads(completed.stdout)
            except (TypeError, json.JSONDecodeError) as exc:
                return {
                    "status": "blocked",
                    "command": command,
                    "error_type": type(exc).__name__,
                    "error": (
                        f"canonical preflight exited {completed.returncode} without a JSON result: "
                        f"{_sanitize_error(repo_root, completed.stderr.strip())}"
                    ),
                }
            if completed.returncode != 0 or not isinstance(result, dict):
                reason = result.get("status_reason") if isinstance(result, dict) else None
                return {
                    "status": "blocked",
                    "command": command,
                    "error_type": str(result.get("status", "CanonicalPreflightProcessError"))
                    if isinstance(result, dict)
                    else "CanonicalPreflightProcessError",
                    "error": _sanitize_error(
                        repo_root,
                        str(reason or completed.stderr.strip() or "canonical preflight failed"),
                    ),
                }
            paths = {
                "validate_config": result.get("validate_config_path"),
                "preview_scenarios": result.get("preview_scenarios_path"),
                "matrix_summary": result.get("matrix_summary_json"),
            }
            if not all(isinstance(path, str) and Path(path).is_file() for path in paths.values()):
                return {
                    "status": "blocked",
                    "command": command,
                    "error_type": "CanonicalPreflightArtifactError",
                    "error": "canonical preflight did not emit all required metadata artifacts",
                }
            try:
                return {
                    "status": "passed",
                    "command": command,
                    "validate_config": _read_json(Path(paths["validate_config"])),
                    "preview_scenarios": _read_json(Path(paths["preview_scenarios"])),
                    "matrix_summary": _read_json(Path(paths["matrix_summary"])),
                }
            except (OSError, ValueError) as exc:
                return {
                    "status": "blocked",
                    "command": command,
                    "error_type": type(exc).__name__,
                    "error": _sanitize_error(repo_root, str(exc)),
                }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "blocked",
            "command": command,
            "error_type": type(exc).__name__,
            "error": _sanitize_error(repo_root, str(exc)),
        }


def _expected_identity_block(  # noqa: C901, PLR0912
    child: dict[str, Any], preflight: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Validate canonical preflight payloads and derive deterministic row identities."""
    blockers: list[dict[str, str]] = []
    planner_key = child["planner_key"]
    if preflight.get("status") != "passed":
        _append_blocker(
            blockers,
            _blocker(
                "canonical_preflight_failed",
                f"{preflight.get('error_type', 'PreflightError')}: {preflight.get('error', 'unknown error')}",
                arm=planner_key,
            ),
        )
        return None, blockers

    validate = preflight.get("validate_config")
    preview = preflight.get("preview_scenarios")
    matrix = preflight.get("matrix_summary")
    if (
        not isinstance(validate, dict)
        or not isinstance(preview, dict)
        or not isinstance(matrix, dict)
    ):
        _append_blocker(
            blockers,
            _blocker(
                "preflight_artifact_invalid",
                "canonical preflight artifacts are incomplete",
                arm=planner_key,
            ),
        )
        return None, blockers
    if validate.get("config_sha256") != child["config_sha256"]:
        _append_blocker(
            blockers,
            _blocker(
                "config_drift", "preflight config hash differs from manifest", arm=planner_key
            ),
        )
    if preview.get("truncated") is not False:
        _append_blocker(
            blockers,
            _blocker(
                "scenario_preview_truncated",
                "canonical preview does not enumerate every scenario",
                arm=planner_key,
            ),
        )
    resolved_scenarios = validate.get("scenario_candidates", {}).get("resolved")
    scenarios = preview.get("scenarios")
    if not isinstance(resolved_scenarios, list) or not all(
        isinstance(item, str) and item for item in resolved_scenarios
    ):
        _append_blocker(
            blockers,
            _blocker(
                "scenario_identity_missing",
                "preflight has no resolved scenario identity list",
                arm=planner_key,
            ),
        )
        resolved_scenarios = []
    preview_names = (
        [
            item.get("name")
            for item in scenarios
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        if isinstance(scenarios, list)
        else []
    )
    if preview_names != resolved_scenarios:
        _append_blocker(
            blockers,
            _blocker(
                "scenario_identity_drift",
                "preview scenario identities differ from validation",
                arm=planner_key,
            ),
        )
    if len(set(resolved_scenarios)) != len(resolved_scenarios):
        _append_blocker(
            blockers,
            _blocker(
                "duplicate_row_identity",
                "preflight contains duplicate scenario identities",
                arm=planner_key,
            ),
        )
    seed_policy = validate.get("seed_policy")
    seeds = seed_policy.get("resolved_seeds") if isinstance(seed_policy, dict) else None
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
        _append_blocker(
            blockers,
            _blocker(
                "seed_identity_missing",
                "preflight has no resolved integer seed set",
                arm=planner_key,
            ),
        )
        seeds = []
    if len(set(seeds)) != len(seeds):
        _append_blocker(
            blockers,
            _blocker(
                "duplicate_row_identity", "preflight contains duplicate seeds", arm=planner_key
            ),
        )
    rows = matrix.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        _append_blocker(
            blockers,
            _blocker(
                "matrix_identity_missing",
                "preflight matrix must contain one arm row",
                arm=planner_key,
            ),
        )
        row: dict[str, Any] = {}
    else:
        row = rows[0]
        if row.get("planner_key") != planner_key:
            _append_blocker(
                blockers,
                _blocker(
                    "matrix_planner_mismatch",
                    "matrix planner identity differs from manifest",
                    arm=planner_key,
                ),
            )
        if row.get("resolved_seeds") != seeds or row.get("repeats") != len(seeds):
            _append_blocker(
                blockers,
                _blocker(
                    "matrix_seed_mismatch",
                    "matrix seed identity differs from validation",
                    arm=planner_key,
                ),
            )
        if row.get("scenario_count") != len(resolved_scenarios):
            _append_blocker(
                blockers,
                _blocker(
                    "matrix_scenario_count_mismatch",
                    "matrix scenario count differs from validation",
                    arm=planner_key,
                ),
            )
    kinematics = row.get("kinematics")
    if not isinstance(kinematics, str) or not kinematics:
        _append_blocker(
            blockers,
            _blocker(
                "kinematics_identity_missing", "matrix has no kinematics identity", arm=planner_key
            ),
        )
        kinematics = ""
    identities = [
        {
            "planner_key": planner_key,
            "scenario_id": scenario_id,
            "seed": seed,
            "kinematics": kinematics,
        }
        for scenario_id in resolved_scenarios
        for seed in seeds
    ]
    identities.sort(
        key=lambda item: (
            item["planner_key"],
            item["scenario_id"],
            item["seed"],
            item["kinematics"],
        )
    )
    if blockers:
        return None, blockers
    scenario_matrix = validate.get("scenario_matrix")
    horizon_path = row.get("scenario_horizons_path")
    horizon_match = _HORIZON.search(str(horizon_path or ""))
    horizon = f"h{horizon_match.group('steps')}" if horizon_match else None
    return {
        "scenario_matrix": scenario_matrix,
        "scenario_count": len(resolved_scenarios),
        "scenario_ids": resolved_scenarios,
        "seed_set": seed_policy.get("seed_set") if isinstance(seed_policy, dict) else None,
        "resolved_seeds": seeds,
        "kinematics": [kinematics],
        "scenario_horizons": horizon_path,
        "horizon": horizon,
        "identity_fields": ["planner_key", "scenario_id", "seed", "kinematics"],
        "expected_row_count": len(identities),
        "row_identity_sha256": _canonical_sha256(identities),
        "row_identities": identities,
    }, blockers


def _build_arm(
    repo_root: Path,
    child: dict[str, Any],
    preflight_runner: PreflightRunner,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Build one arm entry and its fail-closed blockers."""
    planner_key = child["planner_key"]
    config_path = repo_root / child["config_path"]
    blockers: list[dict[str, str]] = []
    if child["observed_sha256"] != child["config_sha256"]:
        _append_blocker(
            blockers,
            _blocker(
                "child_hash_mismatch", "config bytes do not match manifest hash", arm=planner_key
            ),
        )
    if child["fallback_settings"]:
        _append_blocker(
            blockers,
            _blocker(
                "fallback_enabled",
                "fallback/degraded policy is configured; arm is diagnostic-only and excluded from the native aggregate",
                arm=planner_key,
            ),
        )
    preflight = preflight_runner(repo_root, config_path, planner_key)
    identity, identity_blockers = _expected_identity_block(child, preflight)
    for item in identity_blockers:
        _append_blocker(blockers, item)
    eligible = not blockers and identity is not None
    arm = {
        "planner_key": planner_key,
        "config_path": child["config_path"],
        "config_sha256": child["config_sha256"],
        "fallback_settings": child["fallback_settings"],
        "canonical_preflight_command": preflight.get(
            "command", _canonical_preflight_command(child["config_path"], planner_key)
        ),
        "preflight": {
            "status": preflight.get("status", "blocked"),
            "error_type": preflight.get("error_type"),
            "error": preflight.get("error"),
        },
        "expected_row_identity": identity,
        "aggregation": {
            "native_aggregation_eligible": eligible,
            "evidence_classification": "native_candidate" if eligible else "excluded",
            "exclusion_reasons": sorted(
                {
                    item["code"]
                    for item in blockers
                    if item.get("arm") == planner_key or item.get("arm") is None
                }
            ),
        },
        "execution": {
            "status": "planned_not_executed",
            "production_execution_performed": False,
            "submission_authorization": "not_granted",
        },
    }
    return arm, blockers


def build_packet(
    repo_root: Path | None = None,
    manifest_path: Path | None = None,
    *,
    preflight_runner: PreflightRunner | None = None,
) -> dict[str, Any]:
    """Build the deterministic launch packet without submitting or running compute."""
    root = (repo_root or _repo_root()).resolve()
    manifest = (manifest_path or root / DEFAULT_MANIFEST).resolve()
    runner = preflight_runner or _run_canonical_preflight
    inventory, children, blockers = _inventory_children(root, manifest)
    arms: list[dict[str, Any]] = []
    for child in children:
        arm, arm_blockers = _build_arm(root, child, runner)
        arms.append(arm)
        for item in arm_blockers:
            _append_blocker(blockers, item)

    identities = [
        identity
        for arm in arms
        if isinstance(arm.get("expected_row_identity"), dict)
        for identity in arm["expected_row_identity"].get("row_identities", [])
    ]
    identity_keys = [
        json.dumps(identity, sort_keys=True, separators=(",", ":")) for identity in identities
    ]
    if len(set(identity_keys)) != len(identity_keys):
        _append_blocker(
            blockers, _blocker("duplicate_row_identity", "duplicate row identity across arms")
        )
    for arm in arms:
        identity = arm.get("expected_row_identity")
        if isinstance(identity, dict):
            # The complete identity list is used for duplicate detection above, while the
            # committed packet carries the deterministic dimensions plus its digest. This keeps
            # evidence reviewable without dropping the exact identity proof.
            identity.pop("row_identities", None)

    successful_identity_blocks = [
        arm["expected_row_identity"]
        for arm in arms
        if isinstance(arm.get("expected_row_identity"), dict)
    ]
    consistency_fields = (
        "scenario_matrix",
        "seed_set",
        "scenario_count",
        "scenario_ids",
        "resolved_seeds",
        "kinematics",
        "scenario_horizons",
        "horizon",
    )
    for field in consistency_fields:
        values = {
            json.dumps(block.get(field), sort_keys=True) for block in successful_identity_blocks
        }
        if len(values) > 1:
            _append_blocker(
                blockers, _blocker("config_drift", f"preflight arms disagree on {field}")
            )
    first_identity = successful_identity_blocks[0] if successful_identity_blocks else {}
    packet_status = "blocked" if blockers else "prepared_not_submitted"
    expected_row_count = sum(
        int(block.get("expected_row_count", 0)) for block in successful_identity_blocks
    )
    manifest_rel = _relative_path(root, manifest) if manifest.is_file() else DEFAULT_MANIFEST
    manifest_digest = _sha256(manifest) if manifest.is_file() else None
    source_ref = inventory.get("source_config")
    source_digest = inventory.get("source_sha256")
    packet = {
        "schema_version": SCHEMA_VERSION,
        "status": packet_status,
        "issue": ISSUE,
        "parent_issue": PARENT_ISSUE,
        "repository_commit": _git_commit(root),
        "source_directory": DEFAULT_SOURCE_DIRECTORY,
        "split_manifest": manifest_rel,
        "split_manifest_sha256": manifest_digest,
        "source_config": source_ref,
        "source_config_sha256": source_digest,
        "arm_count": len(arms),
        "horizon": first_identity.get("horizon"),
        "seed_set": first_identity.get("seed_set"),
        "resolved_seeds": first_identity.get("resolved_seeds", []),
        "scenario_matrix": first_identity.get("scenario_matrix"),
        "scenario_count": first_identity.get("scenario_count"),
        "kinematics": first_identity.get("kinematics", []),
        "expected_row_count": expected_row_count,
        "expected_row_count_complete": len(successful_identity_blocks) == len(arms) and bool(arms),
        "aggregation_contract": AGGREGATION_CONTRACT,
        "claim_boundary": CLAIM_BOUNDARY,
        "execution_authorization": "not_granted",
        "submission_allowed": False,
        "production_execution_performed": False,
        "blockers": sorted(
            blockers,
            key=lambda item: (item.get("code", ""), item.get("arm", ""), item.get("detail", "")),
        ),
        "arms": arms,
    }
    _assert_fail_closed(packet)
    return packet


def _assert_fail_closed(packet: dict[str, Any]) -> None:
    """Reject unsafe packet states before they can be written as evidence."""
    if packet.get("submission_allowed") is not False:
        raise ValueError("launch packet must never enable submission")
    if packet.get("production_execution_performed") is not False:
        raise ValueError("launch packet must never claim production execution")
    identities: list[str] = []
    for arm in packet.get("arms", []):
        if arm.get("fallback_settings") and arm.get("aggregation", {}).get(
            "native_aggregation_eligible"
        ):
            raise ValueError("fallback-enabled arm cannot be native-aggregate eligible")
        identity = arm.get("expected_row_identity")
        if isinstance(identity, dict):
            identities.extend(
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                for row in identity.get("row_identities", [])
            )
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate row identities cannot be emitted")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=None, help=f"Split manifest (default: {DEFAULT_MANIFEST})"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help=f"Packet output (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Build and validate the packet. A blocked packet is valid fail-closed evidence and is not an error.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build and write the issue #5273 launch packet."""
    args = _build_parser().parse_args(argv)
    root = _repo_root()
    manifest = args.manifest or root / DEFAULT_MANIFEST
    output = args.output or root / DEFAULT_OUTPUT
    if not manifest.is_absolute():
        manifest = root / manifest
    if not output.is_absolute():
        output = root / output
    packet = build_packet(root, manifest)
    write_json(output, packet, catalog_area="benchmark_evidence")
    print(
        json.dumps(
            {
                "output": _relative_path(root, output),
                "status": packet["status"],
                "arm_count": packet["arm_count"],
                "expected_row_count": packet["expected_row_count"],
                "blocker_count": len(packet["blockers"]),
                "submission_allowed": packet["submission_allowed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
