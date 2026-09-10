"""Semantic and headless smoke coverage for the shared SNQI example fixture."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from examples.fixtures.snqi.v1.loader import load_fixture, write_summary

ROOT = Path(__file__).resolve().parents[2]


def test_shared_fixture_has_canonical_semantics() -> None:
    """The fixture loader detects ties, denominators, exclusions, and provenance."""
    fixture = load_fixture()
    assert fixture["declared_denominators"] == {
        "input_records": 4,
        "eligible_records": 3,
        "excluded_records": 1,
        "tie_records": 2,
    }


def test_fixture_examples_are_headless_and_summary_stable(tmp_path: Path) -> None:
    """Both fixture modes emit bounded summaries and raster/vector figures."""
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": str(ROOT)}
    summaries: list[bytes] = []
    for name in ("snqi_figures_example.py", "snqi_full_flow.py"):
        out_dir = tmp_path / Path(name).stem
        command = [
            sys.executable,
            str(
                ROOT / ("examples/plotting" if "figures" in name else "examples/benchmarks") / name
            ),
            "--fixture",
            "--out-dir",
            str(out_dir),
        ]
        subprocess.run(command, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
        summary = out_dir / "snqi_fixture_summary.json"
        summaries.append(summary.read_bytes())
        payload = json.loads(summary.read_text(encoding="utf-8"))
        assert payload["provenance"]["source_kind"] == "synthetic_fixture"
        assert payload["provenance"]["evidence_status"] == "diagnostic-only"
        assert "not benchmark" in payload["provenance"]["claim_boundary"]
        assert {Path(path).suffix for path in payload["outputs"]} >= {".png", ".svg"}
        before = summary.read_bytes()
        write_summary(out_dir, example=payload["example"], output_files=payload["outputs"])
        assert summary.read_bytes() == before
    assert summaries[0] != summaries[1]  # summaries identify their example
