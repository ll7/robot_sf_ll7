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

# Test configuration constants
CI_EXPECTS_CUDA_ENV = "CI_EXPECTS_CUDA"
CUDA_AVAILABLE_FLAG = "1"

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "multi_extractor_training.py"
CONFIG_GPU = ROOT / "configs" / "scenarios" / "multi_extractor_gpu.yaml"


@pytest.mark.skipif(not CONFIG_GPU.exists(), reason="GPU multi-extractor config missing")
def test_vectorized_run_handles_cuda_availability(tmp_path):
    env = os.environ.copy()
    env["ROBOT_SF_MULTI_EXTRACTOR_TMP"] = str(tmp_path)
    env["ROBOT_SF_MULTI_EXTRACTOR_TEST_MODE"] = "1"

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

    cuda_available = env.get(CI_EXPECTS_CUDA_ENV, "0") == CUDA_AVAILABLE_FLAG

    if cuda_available:
        assert statuses == {"success"}
        assert worker_modes == {"vectorized"}
        for hardware in data["hardware_overview"]:
            assert hardware.get("gpu_model")
            assert hardware.get("cuda_version")
    else:
        assert statuses <= {"skipped", "success"}
        assert "cuda" in " ".join(reason.lower() for reason in reasons)

    assert data["aggregate_metrics"], "Expected aggregate metrics in summary"
