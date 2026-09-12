#!/usr/bin/env python3
"""Run a config-driven camera-ready benchmark campaign.

Exit codes preserve fail-closed campaign semantics for non-success outcomes:
- 0: benchmark-success campaign
- 2: unexpected failure, malformed result, or mixed failed/partial-failure outcome
- 3: accepted-unavailable-only campaign outcome (non-success, fail-closed)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf._numerical_thread_env import pin_thread_env_for_determinism

# Apply process-wide numerical thread caps before importing camera-ready modules,
# which transitively import NumPy and may initialize BLAS/OpenMP runtimes.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
pin_thread_env_for_determinism()

from loguru import logger  # noqa: E402

from robot_sf.benchmark.camera_ready._config import (  # noqa: E402
    RadiusSweepBindingPreflightError,
    _load_campaign_scenarios,
)
from robot_sf.benchmark.camera_ready_campaign import (  # noqa: E402
    load_campaign_config,
    prepare_campaign_preflight,
    run_campaign,
)
from robot_sf.benchmark.fallback_policy import campaign_exit_code  # noqa: E402
from robot_sf.benchmark.orca_preflight import OrcaRvo2PreflightError  # noqa: E402
from robot_sf.benchmark.research_answerability import (  # noqa: E402
    DECISION_REQUIRED_PROOF_SURFACES,
    PROOF_BINDING_SCHEMA,
    PROOF_SURFACE_KINDS,
    PROOF_SURFACES,
)
from scripts.tools.record_post_campaign_stage_status import build_stage_status  # noqa: E402
from scripts.validation.run_research_campaign_manifest import (  # noqa: E402
    evaluate_research_manifest_answerability,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for camera-ready campaign execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to camera-ready campaign config YAML.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional campaign base output directory. Defaults to output/benchmarks/camera_ready"
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label suffix embedded into campaign_id.",
    )
    parser.add_argument(
        "--campaign-id",
        type=str,
        default=None,
        help=(
            "Optional exact campaign directory id. Required with --require-answerable so the "
            "research admission binds to the campaign that will execute. Use with "
            "resume-enabled configs to continue an interrupted campaign root."
        ),
    )
    parser.add_argument(
        "--skip-publication-bundle",
        action="store_true",
        help="Skip publication bundle export even if enabled in config.",
    )
    parser.add_argument(
        "--mode",
        choices=("run", "preflight"),
        default="run",
        help="Execution mode: full run or preflight-only artifact generation.",
    )
    parser.add_argument(
        "--research-manifest",
        type=Path,
        default=None,
        help=(
            "Optional research campaign manifest to evaluate before camera-ready admission. "
            "This does not run a campaign or write a research packet."
        ),
    )
    parser.add_argument(
        "--require-answerable",
        action="store_true",
        help=(
            "Fail closed before camera-ready preflight/run unless --research-manifest "
            "evaluates to answerable through its executable proof surfaces."
        ),
    )
    parser.add_argument(
        "--checkpoint-preflight-mode",
        choices=("metadata_only", "enforced_staged"),
        default="metadata_only",
        help=(
            "Arm-checkpoint preflight mode (issue #4613/#4663). 'metadata_only' (default) is the "
            "cheap network-free guard and is NOT submit-safe when any arm is only "
            "stageable_remote. 'enforced_staged' actually downloads and checksum-verifies each "
            "registry checkpoint into the durable cache before continuing; the submit/sbatch "
            "wrapper must use this mode (or run the public "
            "scripts/benchmark/submit_camera_ready_checkpoint_gate.sh) before requeueing. Only "
            "applied to the preflight-only mode path; 'run' mode keeps the cheap guard and "
            "expects checkpoints to be already staged on the compute node."
        ),
    )
    parser.add_argument(
        "--checkpoint-cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory override for staged downloads "
        "(used with --checkpoint-preflight-mode=enforced_staged).",
    )
    parser.add_argument(
        "--checkpoint-registry-path",
        type=Path,
        default=None,
        help="Optional model-registry path override for the arm-checkpoint preflight.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"),
        help="Log level for campaign execution.",
    )
    parser.add_argument(
        "--arm-isolation",
        choices=("in_process", "subprocess"),
        default="in_process",
        help=(
            "Arm isolation mode for campaign execution. 'subprocess' runs each "
            "planner/kinematics variant in a subprocess to ensure full GPU memory "
            "release between arms (issue #4826). 'in_process' runs all arms in the "
            "same process with explicit cleanup."
        ),
    )
    return parser


def _research_admission_proof_error(  # noqa: C901, PLR0912 - ordered admission guards
    *,
    answerability: Mapping[str, Any],
    proof: Mapping[str, Any],
    expected_campaign_id: str,
    expected_config_sha256: str | None,
    expected_execution_inventory: dict[str, Any] | None,
) -> str | None:
    """Reject a weak or mismatched report at the production launch boundary."""
    if answerability.get("decision_capable") is not True:
        return "answerability report did not mark the admitted state as decision_capable"
    if proof.get("executed") is not True or proof.get("status") != "completed":
        return "answerability admission requires a completed executable proof report"
    binding = proof.get("binding")
    if not isinstance(binding, Mapping):
        return "answerability admission omitted its exact proof binding"
    if binding.get("schema_version") != PROOF_BINDING_SCHEMA:
        return f"answerability proof binding schema must be {PROOF_BINDING_SCHEMA}"
    proof_digest = binding.get("proof_digest")
    if (
        not isinstance(proof_digest, str)
        or len(proof_digest) != 64
        or any(character not in "0123456789abcdef" for character in proof_digest.lower())
    ):
        return "answerability admission proof binding has no valid 64-hex proof digest"
    if binding.get("campaign_id") != expected_campaign_id:
        return "answerability proof binding campaign_id does not match the effective campaign"
    if (
        expected_config_sha256 is not None
        and binding.get("config_sha256") != expected_config_sha256
    ):
        return "answerability proof binding config digest does not match the loaded config"
    if (
        expected_execution_inventory is not None
        and binding.get("execution_inventory") != expected_execution_inventory
    ):
        return "answerability proof binding execution inventory does not match the campaign"
    try:
        current_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "answerability admission could not verify the current committed HEAD"
    if binding.get("head_commit") != current_head:
        return "answerability proof binding head_commit does not match the launcher HEAD"
    for field in ("manifest_sha256", "config_sha256"):
        value = binding.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
        ):
            return f"answerability proof binding {field} is not a valid 64-hex SHA-256"
    for field in ("manifest_blob", "config_blob", "head_commit"):
        value = binding.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value.lower())
        ):
            return f"answerability proof binding {field} is not a valid 40-hex Git identity"
    surfaces = proof.get("surfaces")
    bound_surfaces = binding.get("proof_results")
    if not isinstance(surfaces, Mapping) or not isinstance(bound_surfaces, Mapping):
        return "answerability admission proof must include all bound proof surfaces"
    if set(surfaces) != set(PROOF_SURFACES) or set(bound_surfaces) != set(PROOF_SURFACES):
        return "answerability admission proof must name exactly the six proof surfaces"
    for surface in PROOF_SURFACES:
        result = surfaces.get(surface)
        bound_result = bound_surfaces.get(surface)
        if not isinstance(result, Mapping) or not isinstance(bound_result, Mapping):
            return f"answerability proof surface {surface} result must be a mapping"
        if dict(result) != dict(bound_result):
            return f"answerability proof surface {surface} differs from its bound result"
        allowed_kinds = PROOF_SURFACE_KINDS[surface]
        result_kind = result.get("kind")
        if result_kind is not None and result_kind not in allowed_kinds:
            return f"answerability proof surface {surface} has a non-canonical proof kind"
        if result.get("status") == "passed" and result_kind not in allowed_kinds:
            return f"passed answerability proof surface {surface} must identify its canonical kind"
        if result.get("status") not in {
            "passed",
            "unavailable",
            "failed",
            "not_run",
        } or not isinstance(result.get("required"), bool):
            return f"answerability proof surface {surface} has malformed status metadata"
        if result.get("status") == "passed":
            input_path = result.get("proof_input_path")
            input_sha256 = result.get("proof_input_sha256")
            if not isinstance(input_path, str) or not input_path.strip():
                return f"passed answerability proof surface {surface} omitted its input path"
            if (
                not isinstance(input_sha256, str)
                or len(input_sha256) != 64
                or any(character not in "0123456789abcdef" for character in input_sha256.lower())
            ):
                return f"passed answerability proof surface {surface} omitted its input digest"
    for surface in DECISION_REQUIRED_PROOF_SURFACES:
        result = surfaces[surface]
        if result.get("required") is not True or result.get("status") != "passed":
            return f"required answerability proof surface {surface} did not pass"
    return None


def _research_answerability_block(  # noqa: C901
    *,
    manifest_path: Path | None,
    require_answerable: bool,
    mode: str,
    expected_campaign_config: Path,
    expected_config_sha256: str | None = None,
    expected_campaign_id: str | None = None,
    expected_execution_inventory: dict[str, Any] | None = None,
    config_input_drift: bool = False,
) -> dict[str, Any] | None:
    """Return an admission receipt or a fail-closed result for a research gate."""
    if not require_answerable:
        return None
    if manifest_path is None:
        reason = "--require-answerable requires --research-manifest"
        proof: dict[str, Any] = {}
        answerability: dict[str, Any] = {
            "state": "not_declared",
            "decision_capable": False,
            "reasons": [reason],
            "warnings": [],
        }
    elif expected_campaign_id is None:
        reason = (
            "--require-answerable requires --campaign-id so the research admission can bind "
            "to the exact campaign execution identity"
        )
        proof = {}
        answerability = {
            "state": "not_declared",
            "decision_capable": False,
            "reasons": [reason],
            "warnings": [],
        }
    else:
        try:
            if config_input_drift:
                raise ValueError(
                    "camera-ready configuration changed while it was being loaded; "
                    "exact admission binding is unavailable"
                )
            evaluation_kwargs: dict[str, Any] = {
                "execute_validation": True,
                "expected_campaign_config": expected_campaign_config,
                "expected_config_sha256": expected_config_sha256,
                "expected_campaign_id": expected_campaign_id,
                "expected_execution_inventory": expected_execution_inventory,
            }
            report = evaluate_research_manifest_answerability(manifest_path, **evaluation_kwargs)
            if not isinstance(report, dict):
                raise TypeError("answerability evaluator did not return a mapping")
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            reason = f"research answerability admission could not be evaluated: {exc}"
            proof = {}
            answerability = {
                "state": "invalid_contract",
                "decision_capable": False,
                "reasons": [reason],
                "warnings": [],
            }
        else:
            raw_answerability = report.get("answerability")
            answerability = raw_answerability if isinstance(raw_answerability, dict) else {}
            proof = report.get("answerability_proof")
            if not isinstance(proof, dict):
                proof = {}
            reasons = answerability.get("reasons")
            reason = (
                "research answerability gate requires state=answerable, got "
                f"{answerability.get('state', 'unknown')}: {reasons}"
            )
            if answerability.get("state") == "answerable":
                proof_error = _research_admission_proof_error(
                    answerability=answerability,
                    proof=proof,
                    expected_campaign_id=expected_campaign_id,
                    expected_config_sha256=expected_config_sha256,
                    expected_execution_inventory=expected_execution_inventory,
                )
                if proof_error is None:
                    return {
                        "mode": mode,
                        "status": "research_answerability_admitted",
                        "status_reason": "exact manifest/config/proof binding passed",
                        "research_manifest": str(manifest_path),
                        "answerability": answerability,
                        "answerability_proof": proof,
                        "benchmark_success": False,
                        "evidence_status": "not_run",
                    }
                reason = proof_error
                answerability = {
                    **answerability,
                    "state": "blocked_missing_proof",
                    "decision_capable": False,
                    "reasons": [reason],
                }
    return {
        "mode": mode,
        "status": "research_answerability_blocked",
        "status_reason": reason,
        "research_manifest": str(manifest_path) if manifest_path is not None else None,
        "answerability": answerability,
        "answerability_proof": proof,
        "benchmark_success": False,
        "exit_code": 2,
        "campaign_execution_status": "failed",
        "evidence_status": "blocked",
        "row_status_summary": {
            "successful_evidence_rows": 0,
            "accepted_unavailable_rows": 0,
            "unexpected_failed_rows": 0,
            "fallback_or_degraded_rows": 0,
        },
    }


def _config_sha256_if_readable(path: Path) -> str | None:
    """Return a config digest when the loader input is readable at this point."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _execution_inventory(cfg: Any) -> dict[str, Any]:
    """Return the normalized scenario/planner/seed matrix used by the runner."""
    scenarios = _load_campaign_scenarios(cfg)
    return {
        "scenario_ids": sorted(str(scenario["name"]) for scenario in scenarios),
        "planner_ids": sorted(str(planner.key) for planner in cfg.planners),
        "seeds": sorted({int(seed) for scenario in scenarios for seed in scenario["seeds"]}),
        "kinematics": sorted(
            str(value) for value in (cfg.kinematics_matrix or ("differential_drive",))
        ),
    }


def _campaign_contained_path(path: Path, *, campaign_root: Path, field: str) -> Path:
    """Resolve a persistence path and require it to stay below the campaign root."""
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{field} cannot be resolved safely") from exc
    if resolved == campaign_root or campaign_root not in resolved.parents:
        raise ValueError(f"{field} must resolve within campaign_root")
    return resolved


def _stable_persistence_bytes(path: Path, *, field: str) -> bytes:
    """Read one persistence file twice so concurrent mutation fails closed."""
    first = path.read_bytes()
    second = path.read_bytes()
    if first != second:
        raise RuntimeError(f"{field} changed while it was being persisted")
    return first


def _write_new_or_verify(path: Path, payload: bytes, *, field: str) -> None:
    """Create a receipt without overwriting a different concurrent receipt."""
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        existing = _stable_persistence_bytes(path, field=field)
        if existing != payload:
            raise RuntimeError(f"{field} already exists with different bytes") from None
        return
    observed = _stable_persistence_bytes(path, field=field)
    if observed != payload:
        raise RuntimeError(f"{field} changed immediately after it was written")


def _replace_if_unchanged(
    path: Path,
    *,
    expected: bytes,
    replacement: bytes,
    field: str,
) -> None:
    """Atomically replace a summary only when its observed bytes are unchanged."""
    if _stable_persistence_bytes(path, field=field) != expected:
        raise RuntimeError(f"{field} changed before its admission reference was written")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(replacement)
            handle.flush()
            os.fsync(handle.fileno())
        if _stable_persistence_bytes(path, field=field) != expected:
            raise RuntimeError(f"{field} changed during admission persistence")
        os.replace(temporary_path, path)
        temporary_path = None
        if _stable_persistence_bytes(path, field=field) != replacement:
            raise RuntimeError(f"{field} changed immediately after atomic replacement")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _acquire_admission_lock(campaign_root: Path) -> tuple[Path, int]:
    """Acquire a fail-closed per-campaign admission persistence lock."""
    lock_path = campaign_root / ".research_answerability_admission.lock"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        lock_fd = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            "another admission receipt writer is active, or a stale admission lock remains"
        ) from exc
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
    except OSError:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)
        raise
    return lock_path, lock_fd


def _release_admission_lock(lock_path: Path, lock_fd: int) -> None:
    """Release a lock only when its path still names this writer's lock file."""
    try:
        owned = os.fstat(lock_fd)
        current = os.stat(lock_path, follow_symlinks=False)
        if (owned.st_dev, owned.st_ino) == (current.st_dev, current.st_ino):
            lock_path.unlink()
    except FileNotFoundError:
        pass
    finally:
        os.close(lock_fd)


def _build_admission_sidecar(
    admission: dict[str, Any], *, sidecar_path: Path, campaign_root: Path
) -> tuple[bytes, dict[str, str]]:
    """Encode one admission sidecar and its stable receipt identity."""
    encoded = json.dumps(
        admission,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    admission_sha256 = hashlib.sha256(encoded).hexdigest()
    sidecar = {
        "schema_version": "research_answerability_admission.v1",
        "admission": admission,
        "admission_sha256": admission_sha256,
    }
    sidecar_bytes = (json.dumps(sidecar, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    receipt = {
        "sidecar": str(sidecar_path.relative_to(campaign_root)),
        "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        "admission_sha256": admission_sha256,
    }
    return sidecar_bytes, receipt


def _prepare_summary_admission(
    summary_path: Path, *, receipt: Mapping[str, str]
) -> tuple[bytes, bytes]:
    """Validate an existing summary before any sidecar mutation is attempted."""
    summary_bytes = _stable_persistence_bytes(summary_path, field="campaign summary")
    summary = json.loads(summary_bytes.decode("utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("campaign summary must be a JSON object")
    artifacts = summary.get("artifacts")
    if artifacts is None:
        artifacts = {}
    elif not isinstance(artifacts, dict):
        raise ValueError("campaign summary artifacts must be a JSON object")
    existing_receipt = summary.get("research_answerability_admission")
    if existing_receipt is not None and existing_receipt != receipt:
        raise RuntimeError("campaign summary already contains a different admission receipt")
    existing_sidecar = artifacts.get("research_answerability_admission")
    if existing_sidecar is not None and existing_sidecar != receipt["sidecar"]:
        raise RuntimeError("campaign summary already contains a different admission sidecar")
    summary["research_answerability_admission"] = dict(receipt)
    artifacts["research_answerability_admission"] = receipt["sidecar"]
    summary["artifacts"] = artifacts
    replacement = (json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return summary_bytes, replacement


def _persist_answerability_admission(  # noqa: C901
    result: dict[str, Any],
) -> None:
    """Persist a successful admission without path escape or receipt overwrite."""
    admission = result.get("research_answerability_admission")
    summary_value = result.get("summary_json")
    campaign_root_value = result.get("campaign_root")
    if not isinstance(admission, dict):
        raise RuntimeError("research answerability admission payload is missing")
    if not campaign_root_value:
        raise RuntimeError("campaign_root is required to persist research answerability admission")
    try:
        campaign_root = Path(str(campaign_root_value)).resolve()
        if not campaign_root.is_dir():
            raise ValueError("campaign_root must be an existing directory")
        lock_path, lock_fd = _acquire_admission_lock(campaign_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"could not persist research answerability admission: {exc}") from exc

    try:
        summary_path = None
        if summary_value:
            summary_path = _campaign_contained_path(
                Path(str(summary_value)),
                campaign_root=campaign_root,
                field="summary_json",
            )
            if not summary_path.is_file():
                raise ValueError("summary_json must name an existing file")
        sidecar_candidate = (
            summary_path.parent if summary_path is not None else campaign_root / "reports"
        ) / "research_answerability_admission.json"
        sidecar_path = _campaign_contained_path(
            sidecar_candidate,
            campaign_root=campaign_root,
            field="admission sidecar",
        )
        if summary_path is not None and sidecar_path == summary_path:
            raise ValueError("summary_json cannot be the admission sidecar")
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path = _campaign_contained_path(
            sidecar_path,
            campaign_root=campaign_root,
            field="admission sidecar",
        )
        sidecar_bytes, receipt = _build_admission_sidecar(
            admission,
            sidecar_path=sidecar_path,
            campaign_root=campaign_root,
        )
        summary_update = (
            _prepare_summary_admission(summary_path, receipt=receipt)
            if summary_path is not None
            else None
        )
        _write_new_or_verify(sidecar_path, sidecar_bytes, field="admission sidecar")
        if summary_update is not None:
            summary_bytes, summary_replacement = summary_update
            if summary_replacement != summary_bytes:
                _replace_if_unchanged(
                    summary_path,
                    expected=summary_bytes,
                    replacement=summary_replacement,
                    field="campaign summary",
                )
        result["research_answerability_admission_receipt"] = receipt
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not persist research answerability admission: {exc}") from exc
    finally:
        _release_admission_lock(lock_path, lock_fd)


def _finalize_research_admission(
    result: dict[str, Any],
    *,
    research_admission: dict[str, Any],
    expected_campaign_id: str | None,
) -> None:
    """Attach and durably persist an admitted research gate result."""
    result["research_answerability_admission"] = research_admission
    if research_admission.get("status") != "research_answerability_admitted":
        return
    if expected_campaign_id is not None and result.get("campaign_id") != expected_campaign_id:
        result["status"] = "research_answerability_execution_identity_failed"
        result["status_reason"] = (
            "camera-ready result campaign_id does not match the admitted campaign: "
            f"{result.get('campaign_id')!r} != {expected_campaign_id!r}"
        )
        result["benchmark_success"] = False
        result["exit_code"] = 2
        return
    try:
        _persist_answerability_admission(result)
    except RuntimeError as exc:
        logger.error("{}", exc)
        result["status"] = "research_answerability_receipt_failed"
        result["status_reason"] = str(exc)
        result["benchmark_success"] = False
        result["exit_code"] = 2


def main(argv: Sequence[str] | None = None) -> int:
    """Execute camera-ready benchmark campaign from CLI arguments."""
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    parser = _build_parser()
    args = parser.parse_args(raw_argv)

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    config_sha256_before = _config_sha256_if_readable(args.config)
    cfg = load_campaign_config(args.config)
    config_sha256_after = _config_sha256_if_readable(args.config)
    invoked_command = shlex.join([sys.executable, str(Path(__file__)), *raw_argv])
    research_admission = _research_answerability_block(
        manifest_path=args.research_manifest,
        require_answerable=args.require_answerable,
        mode=args.mode,
        expected_campaign_config=args.config,
        expected_config_sha256=config_sha256_after,
        expected_campaign_id=args.campaign_id,
        expected_execution_inventory=(
            _execution_inventory(cfg)
            if args.require_answerable and hasattr(cfg, "scenario_matrix_path")
            else None
        ),
        config_input_drift=(
            config_sha256_before is not None
            and config_sha256_after is not None
            and config_sha256_before != config_sha256_after
        ),
    )
    if research_admission is not None and research_admission.get("status") != (
        "research_answerability_admitted"
    ):
        print(json.dumps(research_admission, indent=2))
        return 2

    try:
        if args.mode == "preflight":
            prepared = prepare_campaign_preflight(
                cfg,
                output_root=args.output_root,
                label=args.label,
                campaign_id=args.campaign_id,
                invoked_command=invoked_command,
                checkpoint_preflight_mode=args.checkpoint_preflight_mode,
                checkpoint_cache_dir=args.checkpoint_cache_dir,
                checkpoint_registry_path=args.checkpoint_registry_path,
            )
            result = {
                "campaign_id": prepared["campaign_id"],
                "campaign_root": str(prepared["campaign_root"]),
                "validate_config_path": str(prepared["validate_config_path"]),
                "preview_scenarios_path": str(prepared["preview_scenarios_path"]),
                "matrix_summary_json": str(prepared["matrix_summary_json_path"]),
                "matrix_summary_csv": str(prepared["matrix_summary_csv_path"]),
                "amv_coverage_json": str(prepared["amv_coverage_json_path"]),
                "amv_coverage_md": str(prepared["amv_coverage_md_path"]),
                "comparability_json": (
                    str(prepared["comparability_json_path"])
                    if prepared.get("comparability_json_path") is not None
                    else None
                ),
                "comparability_md": (
                    str(prepared["comparability_md_path"])
                    if prepared.get("comparability_md_path") is not None
                    else None
                ),
            }
        else:
            result = run_campaign(
                cfg,
                output_root=args.output_root,
                label=args.label,
                campaign_id=args.campaign_id,
                skip_publication_bundle=bool(args.skip_publication_bundle),
                invoked_command=invoked_command,
                arm_isolation=args.arm_isolation,
            )
    except OrcaRvo2PreflightError as exc:
        result = {
            "mode": args.mode,
            "status": "orca_preflight_failed",
            "status_reason": str(exc),
            "benchmark_success": False,
            "exit_code": 2,
            "campaign_execution_status": "failed",
            "evidence_status": "blocked",
            "row_status_summary": {
                "successful_evidence_rows": 0,
                "accepted_unavailable_rows": 0,
                "unexpected_failed_rows": 0,
                "fallback_or_degraded_rows": 0,
            },
        }
    except RadiusSweepBindingPreflightError as exc:
        result = {
            "mode": args.mode,
            "status": "radius_binding_preflight_failed",
            "status_reason": str(exc),
            "benchmark_success": False,
            "exit_code": 2,
            "campaign_execution_status": "failed",
            "evidence_status": "blocked",
            "row_status_summary": {
                "successful_evidence_rows": 0,
                "accepted_unavailable_rows": 0,
                "unexpected_failed_rows": 0,
                "fallback_or_degraded_rows": 0,
            },
        }
    if result is None:
        result = {}
    if research_admission is not None:
        _finalize_research_admission(
            result,
            research_admission=research_admission,
            expected_campaign_id=args.campaign_id,
        )
    print(json.dumps(result, indent=2))
    if args.mode == "preflight" and result.get("status") not in {
        "orca_preflight_failed",
        "radius_binding_preflight_failed",
        "research_answerability_blocked",
        "research_answerability_receipt_failed",
        "research_answerability_execution_identity_failed",
    }:
        return 0
    exit_code = campaign_exit_code(result)
    # Issue #5244: emit the post-campaign stage-status envelope so downstream
    # schedulers/ledgers can classify a completed campaign whose report/analysis
    # stage fails as a separate lane (job_exit_code follows the campaign lane and
    # must not be remapped by a nonzero reporting stage). The campaign exit code is
    # preserved regardless of whether the envelope was written.
    _record_stage_status(result, exit_code)
    return exit_code


def _record_stage_status(result: dict[str, Any], exit_code: int) -> None:
    """Best-effort emit of the post-campaign stage-status envelope.

    A completed campaign with a failed reporting stage must still exit 0 here; the
    envelope simply records the separate report lane. Any failure to write the
    envelope is logged but never changes the campaign exit code.
    """
    campaign_root = result.get("campaign_root") if isinstance(result, dict) else None
    summary_json = result.get("summary_json") if isinstance(result, dict) else None
    if not campaign_root or not summary_json:
        return
    stage_status_path = Path(campaign_root) / "reports" / "post_campaign_stage_status.json"
    try:
        payload = build_stage_status(
            campaign_summary_path=Path(summary_json),
            campaign_exit_code=exit_code,
            stage_name="camera_ready_campaign",
            stage_exit_code=exit_code,
        )
        stage_status_path.parent.mkdir(parents=True, exist_ok=True)
        stage_status_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("post-campaign stage status not recorded: {}", exc)


if __name__ == "__main__":
    raise SystemExit(main())
