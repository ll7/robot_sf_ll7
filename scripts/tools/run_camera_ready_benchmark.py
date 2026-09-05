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
import inspect
import json
import os
import shlex
import sys
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
            }
            if (
                "expected_config_sha256"
                in inspect.signature(evaluate_research_manifest_answerability).parameters
            ):
                evaluation_kwargs["expected_config_sha256"] = expected_config_sha256
            if (
                "expected_campaign_id"
                in inspect.signature(evaluate_research_manifest_answerability).parameters
            ):
                evaluation_kwargs["expected_campaign_id"] = expected_campaign_id
            if (
                "expected_execution_inventory"
                in inspect.signature(evaluate_research_manifest_answerability).parameters
            ):
                evaluation_kwargs["expected_execution_inventory"] = expected_execution_inventory
            report = evaluate_research_manifest_answerability(manifest_path, **evaluation_kwargs)
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
                binding = proof.get("binding")
                if isinstance(binding, dict) and binding.get("proof_digest"):
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
                reason = "answerability admission omitted its exact proof binding"
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


def _persist_answerability_admission(result: dict[str, Any]) -> None:
    """Persist the successful admission beside, and by digest in, the campaign summary."""
    admission = result.get("research_answerability_admission")
    summary_value = result.get("summary_json")
    campaign_root_value = result.get("campaign_root")
    if not isinstance(admission, dict) or not campaign_root_value:
        return
    summary_path = Path(str(summary_value)).resolve() if summary_value else None
    campaign_root = Path(str(campaign_root_value)).resolve()
    try:
        encoded = json.dumps(admission, sort_keys=True, separators=(",", ":")).encode("utf-8")
        admission_sha256 = hashlib.sha256(encoded).hexdigest()
        sidecar_path = (
            summary_path.parent if summary_path is not None else campaign_root / "reports"
        ) / "research_answerability_admission.json"
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar = {
            "schema_version": "research_answerability_admission.v1",
            "admission": admission,
            "admission_sha256": admission_sha256,
        }
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        receipt = {
            "sidecar": str(sidecar_path.relative_to(campaign_root)),
            "sidecar_sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
            "admission_sha256": admission_sha256,
        }
        result["research_answerability_admission_receipt"] = receipt
        if summary_path is not None:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(summary, dict):
                raise ValueError("campaign summary must be a JSON object")
            summary["research_answerability_admission"] = receipt
            artifacts = summary.setdefault("artifacts", {})
            if isinstance(artifacts, dict):
                artifacts["research_answerability_admission"] = receipt["sidecar"]
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not persist research answerability admission: {exc}") from exc


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
        result["research_answerability_admission"] = research_admission
        if research_admission.get("status") == "research_answerability_admitted":
            try:
                _persist_answerability_admission(result)
            except RuntimeError as exc:
                logger.error("{}", exc)
                result["status"] = "research_answerability_receipt_failed"
                result["status_reason"] = str(exc)
                result["benchmark_success"] = False
                result["exit_code"] = 2
    print(json.dumps(result, indent=2))
    if args.mode == "preflight" and result.get("status") not in {
        "orca_preflight_failed",
        "radius_binding_preflight_failed",
        "research_answerability_blocked",
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
