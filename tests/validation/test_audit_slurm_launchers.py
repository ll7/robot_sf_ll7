"""Tests for SLURM launcher static audit (scripts/validation/audit_slurm_launchers.py)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.validation.audit_slurm_launchers import (
    SCHEMA_VERSION,
    audit_launchers,
    parse_slurm_launcher,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fixtures_dir(tmp_path: Path) -> Path:
    base = tmp_path / "fixtures"
    base.mkdir()

    # 1. Valid launcher
    (base / "valid_launcher.sl").write_text(
        """#!/bin/bash
#SBATCH --job-name=valid_training
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=output/slurm/%j-valid_training.out

set -euo pipefail
CONFIG="configs/training/model.yaml"
if [[ ! -f "$CONFIG" ]]; then
  echo "preflight failed: config missing" >&2
  exit 1
fi
RESUME_RECEIPT="${1:-}"
if [[ -n "$RESUME_RECEIPT" ]]; then
  echo "Using portable resume: $RESUME_RECEIPT"
fi
exec python train_model.py --config "$CONFIG"
""",
        encoding="utf-8",
    )

    # 2. Stale partition token
    (base / "stale_partition.sl").write_text(
        """#!/bin/bash
#SBATCH --job-name=stale_job
#SBATCH --partition=epyc-gpu-test
#SBATCH --time=01:00:00
#SBATCH --output=output/slurm/%j-stale.out
set -euo pipefail
python train_model.py
""",
        encoding="utf-8",
    )

    # 3. Path escape
    (base / "path_escape.sl").write_text(
        """#!/bin/bash
#SBATCH --job-name=path_escape_job
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=output/slurm/%j-escape.out
SCRATCH="/hpc/gpfs2/scratch/user123/work"
echo "scratch is $SCRATCH"
""",
        encoding="utf-8",
    )

    # 4. Missing job-name identity
    (base / "missing_job_name.sl").write_text(
        """#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=output/slurm/%j-no_name.out
python train_model.py
""",
        encoding="utf-8",
    )

    # 5. No timeout
    (base / "no_timeout.sl").write_text(
        """#!/bin/bash
#SBATCH --job-name=no_time_job
#SBATCH --partition=gpu
#SBATCH --output=output/slurm/%j-no_time.out
python train_model.py
""",
        encoding="utf-8",
    )

    # 6. Conflicting GPU request
    (base / "conflicting_gpu.sl").write_text(
        """#!/bin/bash
#SBATCH --job-name=conflict_gpu_job
#SBATCH --partition=standard-cpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=output/slurm/%j-conflict.out
python train_model.py
""",
        encoding="utf-8",
    )

    # 7. Absent preflight
    (base / "absent_preflight.sl").write_text(
        """#!/bin/bash
#SBATCH --job-name=non_preflight_job
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=output/slurm/%j-no_preflight.out
set -euo pipefail
exec python train_model.py
""",
        encoding="utf-8",
    )

    # 8. Non-portable resume
    (base / "non_portable_resume.sl").write_text(
        """#!/bin/bash
#SBATCH --job-name=resume_job
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=output/slurm/%j-resume.out
python train_model.py --resume /local/node01/checkpoint_epoch_50.ckpt
""",
        encoding="utf-8",
    )

    return base


def test_valid_launcher_passes(fixtures_dir: Path) -> None:
    row = parse_slurm_launcher(fixtures_dir / "valid_launcher.sl")
    assert row.status == "pass"
    assert row.job_name == "valid_training"
    assert row.time_limit == "04:00:00"
    assert row.partition == "gpu"
    assert row.cpus_per_task == 8
    assert row.gpus == "gpu:1"
    assert row.has_preflight is True
    assert len(row.violations) == 0


def test_stale_partition_detected(fixtures_dir: Path) -> None:
    row = parse_slurm_launcher(fixtures_dir / "stale_partition.sl")
    assert row.status == "fail"
    codes = [v.code for v in row.violations]
    assert "STALE_PARTITION" in codes


def test_path_escape_detected(fixtures_dir: Path) -> None:
    row = parse_slurm_launcher(fixtures_dir / "path_escape.sl")
    assert row.status == "fail"
    codes = [v.code for v in row.violations]
    assert "HARDCODED_PRIVATE_PATH" in codes


def test_missing_job_name_detected(fixtures_dir: Path) -> None:
    row = parse_slurm_launcher(fixtures_dir / "missing_job_name.sl")
    assert row.status == "fail"
    codes = [v.code for v in row.violations]
    assert "MISSING_JOB_NAME" in codes


def test_no_timeout_detected(fixtures_dir: Path) -> None:
    row = parse_slurm_launcher(fixtures_dir / "no_timeout.sl")
    assert row.status == "fail"
    codes = [v.code for v in row.violations]
    assert "MISSING_TIMEOUT" in codes


def test_conflicting_gpu_detected(fixtures_dir: Path) -> None:
    row = parse_slurm_launcher(fixtures_dir / "conflicting_gpu.sl")
    assert row.status == "fail"
    codes = [v.code for v in row.violations]
    assert "CONFLICTING_GPU_REQUEST" in codes


def test_absent_preflight_flagged(fixtures_dir: Path) -> None:
    row = parse_slurm_launcher(fixtures_dir / "absent_preflight.sl")
    assert row.has_preflight is False
    codes = [v.code for v in row.violations]
    assert "MISSING_PREFLIGHT" in codes


def test_non_portable_resume_detected(fixtures_dir: Path) -> None:
    row = parse_slurm_launcher(fixtures_dir / "non_portable_resume.sl")
    assert row.status == "fail"
    codes = [v.code for v in row.violations]
    assert "NON_PORTABLE_RESUME" in codes


def test_unsafe_output_path_detected(tmp_path: Path) -> None:
    script = tmp_path / "bad_output.sl"
    script.write_text(
        """#!/bin/bash
#SBATCH --job-name=bad_out
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
python train.py
""",
        encoding="utf-8",
    )
    row = parse_slurm_launcher(script)
    assert row.status == "fail"
    codes = [v.code for v in row.violations]
    assert "UNSAFE_OUTPUT_PATH" in codes


def test_audit_report_schema_and_determinism(fixtures_dir: Path) -> None:
    paths = [
        fixtures_dir / "valid_launcher.sl",
        fixtures_dir / "stale_partition.sl",
    ]
    report1 = audit_launchers(paths)
    report2 = audit_launchers(paths)

    assert report1["schema"] == SCHEMA_VERSION
    assert report1["ok"] is False
    assert report1["summary"]["total"] == 2
    assert report1["summary"]["passed"] == 1
    assert report1["summary"]["failed"] == 1
    assert json.dumps(report1, sort_keys=True) == json.dumps(report2, sort_keys=True)


def test_cli_execution(
    fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.validation.audit_slurm_launchers import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_slurm_launchers",
            str(fixtures_dir / "valid_launcher.sl"),
            "--format",
            "json",
            "--check",
        ],
    )
    code = main()
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["ok"] is True
    assert data["summary"]["passed"] == 1
