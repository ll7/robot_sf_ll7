import json
import re
from pathlib import Path

import pytest

from robot_sf.benchmark.cli import cli_main

# Note: We rely on the unified CLI which routes to both scripts with consistent behavior.


def _run_cli(tmp_path: Path, action: str) -> tuple[int, str]:
    out = tmp_path / f"snqi_{action}_small.json"
    episodes = Path("tests/data/snqi/episodes_small.jsonl").absolute()
    baseline = Path("tests/data/snqi/baseline_stats.json").absolute()
    # Route through unified CLI and capture log output via caplog in the test functions
    # We pass a low threshold to ensure the warning triggers deterministically
    code = cli_main(
        [
            "snqi",
            action,
            "--episodes",
            str(episodes),
            "--baseline",
            str(baseline),
            "--output",
            str(out),
            "--seed",
            "123",
            "--bootstrap-samples",
            "10",
            "--bootstrap-confidence",
            "0.9",
        ]
    )
    text = out.read_text(encoding="utf-8")
    # Sanity: output JSON decodes and has summary
    data = json.loads(text)
    assert "summary" in data
    return code, text


@pytest.mark.parametrize("action", ["recompute", "optimize"])
def test_small_dataset_warning_emitted_and_exit_zero(tmp_path: Path, caplog, action: str):
    caplog.set_level("WARNING")
    code, _ = _run_cli(tmp_path, action)
    assert code == 0
    # Look for our small dataset warning message in captured logs
    messages = "\n".join([rec.getMessage() for rec in caplog.records])
    assert re.search(r"Small dataset: using \d+ episodes \(< \d+\)\.", messages)
