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
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf._numerical_thread_env import pin_thread_env_for_determinism

# Apply process-wide numerical thread caps before importing camera-ready modules,
# which transitively import NumPy and may initialize BLAS/OpenMP runtimes.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
pin_thread_env_for_determinism()

from loguru import logger  # noqa: E402

from robot_sf.benchmark.camera_ready._config import (  # noqa: E402
    RadiusSweepBindingPreflightError,
)
from robot_sf.benchmark.camera_ready_campaign import (  # noqa: E402
    load_campaign_config,
    prepare_campaign_preflight,
    run_campaign,
)
from robot_sf.benchmark.checkpoint_staging_receipt import (  # noqa: E402
    validate_checkpoint_staging_receipt,
)
from robot_sf.benchmark.fallback_policy import campaign_exit_code  # noqa: E402
from robot_sf.benchmark.orca_preflight import OrcaRvo2PreflightError  # noqa: E402
from robot_sf.benchmark.release_acceptance import (  # noqa: E402
    validate_full_benchmark_release_acceptance,
)
from robot_sf.benchmark.release_protocol import (  # noqa: E402
    SCIENTIFIC_CANDIDATE_FROZEN_ARCHIVE_SHA256,
    build_scientific_candidate_identity,
    scientific_candidate_acceptance_view,
)
from robot_sf.benchmark.runtime_smoke_admission import (  # noqa: E402
    validate_runtime_smoke_result,
)
from robot_sf.common.artifact_paths import get_repository_root  # noqa: E402
from scripts.tools.record_post_campaign_stage_status import build_stage_status  # noqa: E402
from scripts.tools.run_benchmark_release import (  # noqa: E402
    _compare_rehearsal_checkpoint_identities,
)
from scripts.validation.check_release_metric_equivalence import (  # noqa: E402
    _read_archive_campaign_manifest,
    _read_archive_manifest,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

FROZEN_0_0_7_ARCHIVE_SHA256 = SCIENTIFIC_CANDIDATE_FROZEN_ARCHIVE_SHA256
FROZEN_0_0_7_SOURCE_SHA = "07f7e8d43084de748915e1b1eb8b2a1603357c6e"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_gate(command: list[str], output_log: Path) -> None:
    with output_log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=get_repository_root(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise ValueError(f"scientific candidate gate failed ({completed.returncode}): {output_log}")


def _finish_scientific_candidate(
    result: dict[str, Any],
    cfg: Any,
    baseline_manifest: dict[str, Any],
    baseline_campaign_manifest: dict[str, Any],
    args: Any,
    checkpoint_receipt: dict[str, Any],
) -> None:
    """Record raw custody and run full acceptance and post-run scientific gates."""
    root = Path(result["campaign_root"]).resolve(strict=True)
    identity = build_scientific_candidate_identity(
        cfg=cfg,
        baseline_manifest=baseline_manifest,
        baseline_campaign_manifest=baseline_campaign_manifest,
        baseline_archive_sha256=FROZEN_0_0_7_ARCHIVE_SHA256,
        source_sha=args.source_commit,
        checkpoint_receipt=checkpoint_receipt,
        checkpoint_receipt_sha256=_sha256(args.checkpoint_receipt),
        runtime_smoke_receipt_sha256=_sha256(args.runtime_smoke_receipt),
        campaign_root=root,
    )
    identity_path = root / "release/scientific_candidate.json"
    _write_json(identity_path, identity)
    report_dir = root / "reports"
    acceptance = validate_full_benchmark_release_acceptance(
        root,
        manifest=scientific_candidate_acceptance_view(identity, cfg),
        campaign_config=cfg,
    )
    _write_json(report_dir / "scientific_candidate_acceptance.json", acceptance)
    if acceptance.get("status") != "valid":
        raise ValueError("full 20,160-row scientific candidate acceptance failed")
    equivalence = report_dir / "metric_equivalence.json"
    _candidate_gate(
        [
            sys.executable,
            str(get_repository_root() / "scripts/validation/check_release_metric_equivalence.py"),
            "--baseline-archive",
            str(args.baseline_archive),
            "--baseline-sha256",
            FROZEN_0_0_7_ARCHIVE_SHA256,
            "--baseline-source-sha",
            FROZEN_0_0_7_SOURCE_SHA,
            "--candidate-root",
            str(root),
            "--candidate-source-sha",
            args.source_commit,
            "--expected-rows",
            "20160",
            "--scientific-candidate",
            "--require-robot-force-metrics",
            "--output",
            str(equivalence),
        ],
        report_dir / "scientific_candidate_equivalence.log",
    )
    force_report = report_dir / "robot_force_validation.json"
    _candidate_gate(
        [
            sys.executable,
            str(get_repository_root() / "scripts/analysis/issue_9668_robot_force_validation.py"),
            "--campaign-root",
            str(root),
            "--expected-source-sha",
            args.source_commit,
            "--expected-episodes",
            "20160",
            "--equivalence-report",
            str(equivalence),
            "--output",
            str(force_report),
        ],
        report_dir / "scientific_candidate_force.log",
    )
    receipt = {
        "schema_version": "benchmark-scientific-candidate-result.v1",
        "status": "accepted_pre_publication",
        "source_sha": args.source_commit,
        "scientific_identity_sha256": identity["scientific_identity_sha256"],
        "baseline_archive_sha256": FROZEN_0_0_7_ARCHIVE_SHA256,
        "identity_file_sha256": _sha256(identity_path),
        "full_acceptance_sha256": _sha256(report_dir / "scientific_candidate_acceptance.json"),
        "metric_equivalence_sha256": _sha256(equivalence),
        "metric_equivalence_log_sha256": _sha256(
            report_dir / "scientific_candidate_equivalence.log"
        ),
        "robot_force_validation_sha256": _sha256(force_report),
        "robot_force_log_sha256": _sha256(report_dir / "scientific_candidate_force.log"),
    }
    _write_json(root / "release/scientific_candidate_result.json", receipt)
    result["scientific_candidate_status"] = "accepted_pre_publication"
    result["scientific_candidate_identity"] = str(identity_path)


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
            "Optional exact campaign directory id. Use with resume-enabled configs to continue "
            "an interrupted campaign root."
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
        "--scientific-candidate",
        action="store_true",
        help="Explicit publication-free 0.0.8 scientific candidate; requires source and custody pins",
    )
    parser.add_argument("--source-commit", help="Exact clean source SHA for candidate mode")
    parser.add_argument("--baseline-archive", type=Path)
    parser.add_argument("--checkpoint-receipt", type=Path)
    parser.add_argument("--runtime-smoke-receipt", type=Path)
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


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    """Execute camera-ready benchmark campaign from CLI arguments."""
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    parser = _build_parser()
    args = parser.parse_args(raw_argv)

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    cfg = load_campaign_config(args.config)
    baseline_manifest: dict[str, Any] | None = None
    baseline_campaign_manifest: dict[str, Any] | None = None
    checkpoint_receipt: dict[str, Any] | None = None
    if args.scientific_candidate:
        if (
            not args.source_commit
            or not args.baseline_archive
            or not args.checkpoint_receipt
            or not args.runtime_smoke_receipt
        ):
            parser.error(
                "--scientific-candidate requires --source-commit, --baseline-archive, "
                "--checkpoint-receipt, and --runtime-smoke-receipt"
            )
        if args.campaign_id is not None:
            parser.error(
                "scientific candidate requires a fresh campaign id; resume is not admitted"
            )
        if args.checkpoint_registry_path is not None:
            parser.error("scientific candidate uses the source-pinned model registry")
        if args.arm_isolation != cfg.arm_isolation:
            parser.error("scientific candidate arm isolation must match the pinned config")
        cfg = replace(cfg, release_tag="", doi="", publication_identity_mode="scientific_candidate")
        if _sha256(args.baseline_archive) != FROZEN_0_0_7_ARCHIVE_SHA256:
            raise ValueError("frozen 0.0.7 archive checksum mismatch")
        baseline_manifest = _read_archive_manifest(args.baseline_archive, FROZEN_0_0_7_SOURCE_SHA)
        baseline_campaign_manifest = _read_archive_campaign_manifest(
            args.baseline_archive, FROZEN_0_0_7_SOURCE_SHA
        )
        checkpoint_receipt = validate_checkpoint_staging_receipt(
            cfg,
            args.checkpoint_receipt,
            campaign_config_path=args.config,
            repo_root=get_repository_root(),
        )
        validate_runtime_smoke_result(
            args.runtime_smoke_receipt,
            repo_root=get_repository_root(),
            expected_source_commit=args.source_commit,
            expected_planner_keys=tuple(planner.key for planner in cfg.planners),
        )
        _, checkpoint_identities_match = _compare_rehearsal_checkpoint_identities(
            checkpoint_receipt,
            args.runtime_smoke_receipt,
            release_receipt_sha256=_sha256(args.checkpoint_receipt),
            runtime_smoke_receipt_sha256=_sha256(args.runtime_smoke_receipt),
        )
        if not checkpoint_identities_match:
            raise ValueError("staged and runtime-smoke checkpoint identities differ")
        build_scientific_candidate_identity(
            cfg=cfg,
            baseline_manifest=baseline_manifest,
            baseline_campaign_manifest=baseline_campaign_manifest,
            baseline_archive_sha256=FROZEN_0_0_7_ARCHIVE_SHA256,
            source_sha=args.source_commit,
            checkpoint_receipt=checkpoint_receipt,
            checkpoint_receipt_sha256=_sha256(args.checkpoint_receipt),
            runtime_smoke_receipt_sha256=_sha256(args.runtime_smoke_receipt),
        )
    invoked_command = shlex.join([sys.executable, str(Path(__file__)), *raw_argv])
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
                skip_publication_bundle=bool(
                    args.skip_publication_bundle or args.scientific_candidate
                ),
                invoked_command=invoked_command,
                arm_isolation=args.arm_isolation,
            )
            if args.scientific_candidate:
                if campaign_exit_code(result) != 0:
                    raise ValueError("campaign did not complete successfully")
                assert (
                    baseline_manifest is not None
                    and baseline_campaign_manifest is not None
                    and checkpoint_receipt is not None
                )
                _finish_scientific_candidate(
                    result,
                    cfg,
                    baseline_manifest,
                    baseline_campaign_manifest,
                    args,
                    checkpoint_receipt,
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
    print(json.dumps(result, indent=2))
    if args.mode == "preflight" and result.get("status") not in {
        "orca_preflight_failed",
        "radius_binding_preflight_failed",
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
