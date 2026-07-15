#!/usr/bin/env python3
"""Reconstruct the job-13274 downstream exit-code remapping (issue #5244).

Slurm job 13274 finished its planner rows but exited 5 under
``snqi_contract.enforcement=warn``. The issue thread narrows the observed exit 5
to one of two public candidates chained *after* the campaign completed:

1. A post-campaign artifact-presence check under ``set -euo pipefail`` that
   re-labels a complete campaign as failed when a report file is missing.
2. A chained SNQI tool (e.g. ``snqi_sensitivity_analysis.py``) emitting
   ``EXIT_OPTIONAL_DEPS_MISSING`` (5) when an optional dependency such as
   matplotlib is missing on the compute node.

Both are "structural conflation" defects: they fire *after* the campaign has
fully completed, so a missing report or a missing optional dep relabels an
otherwise complete campaign as a failed scheduler job and orphans the result.

This harness reconstructs that scenario with a minimal fixture and runs it
through the **adopted** production boundary (the ``robot-sf-post-campaign-stage-status.v1``
envelope plus ``run_post_campaign_stage`` / ``slurm_job_finalize``). It prints the
before/after exit mapping so the responsible layer and the corrected behavior are
reproducible without a compute node.

Run:

    uv run python scripts/tools/reconstruct_job13274_exit_mapping.py \
        --campaign-root /tmp/job13274 --emit-side-by-side

The harness never submits jobs, uploads, or runs a full campaign; it drives the
real production dispatch/serialization primitives against a fake completed
campaign and a ``stage`` exit code of 5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.tools import run_post_campaign_stage, slurm_job_finalize

if TYPE_CHECKING:
    from collections.abc import Sequence

REPORT_REQUIRED_ARTIFACTS = (
    "reports/headline_rows.json",
    "reports/result.json",
)


def _write_completed_campaign_fixture(campaign_root: Path, *, soft_warning: bool) -> Path:
    """Write a minimal completed ``enforcement=warn`` campaign fixture.

    The fixture records a benchmark-success campaign whose SNQI contract carried a
    soft warning (``soft_contract_warning: true``). It mirrors the job-13274 result
    shape: all planner rows completed but the contract surfaced a non-fatal warning.
    """
    reports_dir = campaign_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "campaign_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "soft_contract_warning": soft_warning,
                "warnings": (
                    [
                        "SNQI contract status=warn with snqi_contract.enforcement=warn; "
                        "campaign marked with soft contract warning."
                    ]
                    if soft_warning
                    else []
                ),
                "benchmark_success": True,
                "status": "benchmark_success",
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def reconstruct_before_fix(campaign_root: Path, *, stage_exit_code: int) -> dict[str, Any]:
    """Reproduce the legacy defect: a chained stage exit 5 becomes the job exit.

    Before the #5244 envelope was adopted, a ``set -e``/``set -euo pipefail`` shell
    wrapper propagated the report-stage exit code as the job exit code, so a
    completed campaign (exit 0) was relabeled as a failed job (exit 5).

    Returns the reconstructed ``(legacy_job_exit_code, legacy_classification)`` pair.
    """
    # Legacy wrapper: job_exit_code := last chained stage exit under set -e.
    return {
        "legacy_job_exit_code": stage_exit_code,
        "legacy_classification": "failed" if stage_exit_code != 0 else "success",
    }


def reconstruct_after_fix(
    summary_path: Path,
    campaign_root: Path,
    *,
    campaign_exit_code: int,
    stage_exit_code: int,
) -> dict[str, Any]:
    """Reconstruct the corrected production boundary for the same scenario.

    Runs the real ``run_post_campaign_stage`` primitive so the stage exit 5 is
    recorded in the ``post_campaign_stage`` lane while the campaign lane (and thus
    ``job_exit_code``) stays 0, then finalizes the job through ``slurm_job_finalize``
    which consumes the on-disk envelope.
    """
    envelope_path = campaign_root / "reports" / "post_campaign_stage_status.json"
    stage_script = campaign_root / "stage_fail.sh"
    stage_script.write_text(f"#!/usr/bin/env bash\nexit {stage_exit_code}\n", encoding="utf-8")
    stage_script.chmod(0o755)

    primitive_exit = run_post_campaign_stage.main(
        [
            "--campaign-summary",
            str(summary_path),
            "--campaign-exit-code",
            str(campaign_exit_code),
            "--stage-name",
            "headline_ci_rank_stability_report",
            "--output",
            str(envelope_path),
            "--stage-command",
            str(stage_script),
        ]
    )
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))

    finalize_output = campaign_root / "finalize.json"
    finalize_exit = slurm_job_finalize.main(
        [
            "--issue",
            "5244",
            "--job-id",
            "13274",
            "--job-state",
            "COMPLETED",
            "--expected-artifact",
            str(envelope_path),
            "--post-campaign-stage-status",
            str(envelope_path),
            "--output",
            str(finalize_output),
        ]
    )
    report = json.loads(finalize_output.read_text(encoding="utf-8"))
    return {
        "primitive_exit": primitive_exit,
        "envelope_schema": envelope.get("schema_version"),
        "campaign_exit_code": envelope["campaign"]["exit_code"],
        "job_exit_code": envelope["job_exit_code"],
        "post_campaign_stage_exit_code": envelope["post_campaign_stage"]["exit_code"],
        "post_campaign_stage_status": envelope["post_campaign_stage"]["status"],
        "finalize_exit": finalize_exit,
        "finalize_classification": report["classification"],
        "claim_boundary": report["claim_boundary"],
    }


def _emit_side_by_side(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Print the before/after exit mapping for the reproduced job-13274 trace."""
    print("== job 13274 exit-code reconstruction (issue #5244) ==")
    print(f" schema: {after['envelope_schema']}")
    print()
    print("  BEFORE the #5244 envelope was adopted (legacy set -e wrapper):")
    print(
        f"    job_exit_code = {before['legacy_job_exit_code']} ({before['legacy_classification']})"
    )
    print(
        "    -> a completed campaign (exit 0) was relabeled by the chained "
        "report stage exit 5 and orphaned."
    )
    print()
    print("  AFTER the adopted production boundary:")
    print(f"    campaign_exit_code        = {after['campaign_exit_code']}")
    print(
        f"    job_exit_code             = {after['job_exit_code']} "
        f"(finalize exit {after['finalize_exit']}: {after['finalize_classification']})"
    )
    print(
        f"    post_campaign_stage       = exit {after['post_campaign_stage_exit_code']} "
        f"({after['post_campaign_stage_status']})"
    )
    print(f"    claim_boundary            = {after['claim_boundary']}")
    print()
    print(
        "  Root cause: the post-campaign report/analysis stage ran under "
        "set -e/set -euo pipefail and its exit code propagated as the job exit, "
        "confusing a completed campaign with a failed job. Fix: route the stage "
        "through the envelope so its lane is separate and the campaign exit is "
        "preserved."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Reconstruct the job-13274 exit mapping and emit the side-by-side result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path("output/job13274_reconstruction"),
        help="Root directory for the minimal campaign fixture and reconstructed artifacts.",
    )
    parser.add_argument(
        "--stage-exit-code",
        type=int,
        default=5,
        help="Exit code of the chained post-campaign stage to reproduce (job-13274: 5).",
    )
    parser.add_argument(
        "--no-soft-warning",
        action="store_true",
        help="Fixture a campaign without the soft contract warning (hard-success variant).",
    )
    parser.add_argument(
        "--emit-side-by-side",
        action="store_true",
        help="Print the human-readable before/after exit mapping for the reproduced trace.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    campaign_root = args.campaign_root
    soft_warning = not args.no_soft_warning
    summary_path = _write_completed_campaign_fixture(campaign_root, soft_warning=soft_warning)

    before = reconstruct_before_fix(campaign_root, stage_exit_code=args.stage_exit_code)
    after = reconstruct_after_fix(
        summary_path,
        campaign_root,
        campaign_exit_code=0,
        stage_exit_code=args.stage_exit_code,
    )

    if args.emit_side_by_side:
        _emit_side_by_side(before, after)
    else:
        print(json.dumps({"before": before, "after": after}, indent=2, sort_keys=True))

    # The harness exits 0 when the corrected boundary preserves the campaign exit 0
    # and isolates the stage failure, proving the reproduction is actionable.
    return 0 if after["job_exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
