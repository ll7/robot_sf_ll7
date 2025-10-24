import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "multi_extractor_training.py"
CONFIG_DEFAULT = ROOT / "configs" / "scenarios" / "multi_extractor_default.yaml"


@pytest.mark.skipif(not CONFIG_DEFAULT.exists(), reason="default multi-extractor config missing")
def test_single_thread_run_produces_timestamped_summary(tmp_path):
    env = os.environ.copy()
    env["ROBOT_SF_MULTI_EXTRACTOR_TMP"] = str(tmp_path)
    env["ROBOT_SF_MULTI_EXTRACTOR_TEST_MODE"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(CONFIG_DEFAULT),
            "--run-id",
            "integration-single",
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
    assert summary_path.exists(), "summary.json missing from run directory"
    assert markdown_path.exists(), "summary.md missing from run directory"

    data = json.loads(summary_path.read_text(encoding="utf-8"))

    statuses = {record["status"] for record in data["extractor_results"]}
    assert statuses == {"success"}

    worker_modes = {record["worker_mode"] for record in data["extractor_results"]}
    assert worker_modes == {"single-thread"}

    for record in data["extractor_results"]:
        for relative_path in record["artifacts"].values():
            artifact_path = latest / relative_path
            assert artifact_path.exists(), (
                f"Missing artifact path {relative_path} for extractor {record['config_name']}"
            )
