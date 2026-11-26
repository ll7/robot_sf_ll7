"""The validated reference-oracle schedule must contain concrete integers."""

from pathlib import Path

import pytest


@pytest.mark.parametrize("numeric_type", [int, float])
def test_validated_schedule_constructs_admitted_integers(
    tmp_path: Path, numeric_type: type[int] | type[float]
) -> None:
    """Numeric equality at admission must not leak float seeds into execution."""
    root = Path(__file__).resolve().parents[2]
    assert (root / "scripts/validation/run_reference_planner_oracles.py").is_file(), (
        "Reference-oracle runner is absent from this revision"
    )

    import yaml

    from scripts.validation.run_reference_planner_oracles import load_contract

    config_path = root / "configs/validation/issue_9732_reference_oracles_development_v1.yaml"
    candidate = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    candidate["expected_seeds"] = [numeric_type(seed) for seed in range(1001, 1031)]
    path = tmp_path / "numeric-seeds.yaml"
    path.write_text(yaml.safe_dump(candidate), encoding="utf-8")

    config, _ = load_contract(path)

    seeds = config["_resolved"]["seeds"]
    assert seeds == list(range(1001, 1031))
    assert all(type(seed) is int for seed in seeds)
