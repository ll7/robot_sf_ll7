from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_summary(
    path: Path,
    *,
    candidate: str,
    command_clip: float,
    yaw_saturation: float,
    braking_peak: float,
) -> None:
    path.write_text(
        json.dumps(
            {
                "candidate": candidate,
                "stage": "amv_actuation_smoke",
                "summary": {
                    "success_rate": 0.0,
                    "collision_rate": 0.0,
                    "near_miss_rate": 0.0,
                    "scenario_family": {"classic": {"collision_rate": 0.0}},
                    "synthetic_actuation": {
                        "command_clip_fraction_mean": command_clip,
                        "yaw_rate_saturation_fraction_mean": yaw_saturation,
                        "signed_braking_peak_m_s2_mean": braking_peak,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_compare_policy_search_candidates_includes_synthetic_actuation_metrics(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    gates = tmp_path / "gates.yaml"
    output = tmp_path / "comparison"
    _write_summary(
        first,
        candidate="hybrid_rule_v3_fast_progress",
        command_clip=0.275,
        yaw_saturation=0.0,
        braking_peak=-2.5,
    )
    _write_summary(
        second,
        candidate="actuation_aware_hybrid_rule_v0",
        command_clip=0.1875,
        yaw_saturation=0.0,
        braking_peak=-2.5,
    )
    gates.write_text(
        "baselines:\n"
        "  goal:\n"
        "    success_rate: 0.1\n"
        "    collision_rate: 0.2\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/tools/compare_policy_search_candidates.py",
            str(first),
            str(second),
            "--promotion-gates",
            str(gates),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "comparison.json" in result.stdout
    payload = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    rows = {row["candidate"]: row for row in payload["rows"]}
    assert rows["hybrid_rule_v3_fast_progress"]["command_clip_fraction_mean"] == 0.275
    assert rows["actuation_aware_hybrid_rule_v0"]["yaw_rate_saturation_fraction_mean"] == 0.0
    assert rows["actuation_aware_hybrid_rule_v0"]["signed_braking_peak_m_s2_mean"] == -2.5
    assert rows["goal"]["command_clip_fraction_mean"] is None

    markdown = (output / "comparison.md").read_text(encoding="utf-8")
    assert "Command Clip" in markdown
    assert "Yaw Saturation" in markdown
    assert "0.1875" in markdown
    assert "n/a" in markdown
