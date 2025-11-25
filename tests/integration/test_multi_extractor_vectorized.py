"""Integration tests for multi-extractor training with vectorized environments.

This module tests the multi-extractor training script's ability to handle GPU/CUDA
configurations in both available and unavailable scenarios.

Environment Variables:
    CI_EXPECTS_CUDA: Set to "1" in CI environments where CUDA is expected to be
                     available and GPU extractors should run successfully. When not
                     set or set to "0", CUDA extractors are expected to be skipped
                     with appropriate reasons.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import torch
except Exception:  # pragma: no cover - torch is optional in some test environments
    torch = None

# Test configuration constants
CI_EXPECTS_CUDA_ENV = "CI_EXPECTS_CUDA"
CUDA_AVAILABLE_FLAG = "1"

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "multi_extractor_training.py"
CONFIG_GPU = ROOT / "configs" / "scenarios" / "multi_extractor_gpu.yaml"


def _detect_cuda_available() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - cuda detection can raise on exotic setups
        return False


def _resolve_cuda_expectation(env: dict[str, str]) -> bool:
    override = env.get(CI_EXPECTS_CUDA_ENV)
    if override in {"0", "1"}:
        return override == CUDA_AVAILABLE_FLAG
    detected = _detect_cuda_available()
    env[CI_EXPECTS_CUDA_ENV] = CUDA_AVAILABLE_FLAG if detected else "0"
    return detected


@pytest.mark.skipif(not CONFIG_GPU.exists(), reason="GPU multi-extractor config missing")
def test_vectorized_run_handles_cuda_availability(tmp_path):
    env = os.environ.copy()
    env["ROBOT_SF_MULTI_EXTRACTOR_TMP"] = str(tmp_path)
    env["ROBOT_SF_MULTI_EXTRACTOR_TEST_MODE"] = "1"
    cuda_available = _resolve_cuda_expectation(env)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(CONFIG_GPU),
            "--run-id",
            "integration-gpu",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    run_dirs = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    assert run_dirs, (
        f"No output directory created. Stdout: {completed.stdout}\nStderr: {completed.stderr}"
    )

    latest = run_dirs[-1]
    summary_path = latest / "summary.json"
    markdown_path = latest / "summary.md"
    assert summary_path.exists()
    assert markdown_path.exists()

    data = json.loads(summary_path.read_text(encoding="utf-8"))

    statuses = {record["status"] for record in data["extractor_results"]}
    reasons = {record.get("reason", "") for record in data["extractor_results"]}
    worker_modes = {record["worker_mode"] for record in data["extractor_results"]}

    if cuda_available:
        assert statuses == {"success"}
        assert worker_modes == {"vectorized"}
        gpu_records = [
            hardware for hardware in data["hardware_overview"] if hardware.get("gpu_model")
        ]
        if gpu_records:
            for hardware in gpu_records:
                assert hardware.get("gpu_model")
                assert hardware.get("cuda_version")
    else:
        assert statuses <= {"skipped", "success"}
        assert "cuda" in " ".join(reason.lower() for reason in reasons)

    assert data["aggregate_metrics"], "Expected aggregate metrics in summary"
