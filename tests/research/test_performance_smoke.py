"""Performance smoke test for research report generation (T077)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_research_report_performance(tmp_path: Path):
    """Ensure the synthetic performance harness completes quickly."""

    out_dir = tmp_path / "perf_report"
    env = os.environ.copy()
    env["RESEARCH_REPORT_PERF_DIR"] = str(out_dir)
    env["RESEARCH_PERF_BUDGET"] = "120"

    start = time.time()
    result = subprocess.run(
        [sys.executable, "scripts/validation/performance_research_report.py"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, stdout
    assert (out_dir / "data" / "metrics.json").exists(), stdout
    assert (out_dir / "metadata.json").exists(), stdout
    # Guardrail against runaway runtime even if the helper budget changes
    assert (time.time() - start) < 10, "performance harness exceeded 10s wall time"
