"""TODO docstring. Document this module."""

import json
from pathlib import Path

from scripts.snqi_weight_optimization import parse_args, run

FIXTURE_DIR = Path(__file__).parent / "data" / "snqi"
ARTIFACT = Path("model/snqi_canonical_weights_v1.json")


def test_snqi_drift_minimal(tmp_path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    episodes = FIXTURE_DIR / "episodes_small.jsonl"
    baseline = FIXTURE_DIR / "baseline_stats.json"
    out = tmp_path / "opt.json"
    args = parse_args(
        [
            "--episodes",
            str(episodes),
            "--baseline",
            str(baseline),
            "--output",
            str(out),
            "--method",
            "grid",
            "--grid-resolution",
            "3",
            "--max-grid-combinations",
            "500",
            "--seed",
            "123",
            "--validate",
        ],
    )
    code = run(args)
    assert code == 0
    current = json.loads(out.read_text())
    assert ARTIFACT.exists(), "Canonical weights artifact missing; regenerate if refactored."
    with ARTIFACT.open("r", encoding="utf-8") as f:
        artifact = json.load(f)
    artifact_weights = artifact["weights"]
    # Ensure weight keys stable
    assert set(artifact_weights.keys()) == set(current["recommended"]["weights"].keys())
    # Objective should not degrade (allow tiny floating noise)
    assert current["recommended"]["objective_value"] + 1e-9 >= 1.0
