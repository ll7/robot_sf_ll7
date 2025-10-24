import json
from pathlib import Path

from robot_sf.benchmark.snqi.schema import assert_all_finite, validate_snqi
from scripts.snqi_weight_optimization import parse_args, run  # imports sorted

FIXTURE_DIR = Path(__file__).parent / "data" / "snqi"


def test_snqi_minimal_grid(tmp_path):
    episodes = FIXTURE_DIR / "episodes_small.jsonl"
    baseline = FIXTURE_DIR / "baseline_stats.json"
    out = tmp_path / "result.json"
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
            "200",
            "--seed",
            "123",
            "--validate",
            "--ci-placeholder",
        ],
    )
    code = run(args)
    assert code == 0
    data = json.loads(out.read_text())
    # Basic structural checks
    assert "recommended" in data
    assert "_metadata" in data
    assert (
        data["_metadata"].get("confidence_intervals_placeholder", {}).get("status") == "placeholder"
    )
    validate_snqi(data, "optimization", check_finite=True)
    assert_all_finite(data)
