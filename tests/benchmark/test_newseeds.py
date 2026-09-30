"""Author-approved sealed schedule; static validation only, no environments."""

import hashlib
import random
from pathlib import Path

import pytest

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios, load_campaign_config
from robot_sf.benchmark.release_candidate import _candidate_seed_policy
from scripts.validation.check_seed_holdout_diff import _range_overlaps_holdout

LABEL = "robot_sf_ll7 release 0.0.8 evaluation seeds v1 (sealed 2026-09-30)"
DIGEST = "166597da1e0e813d8a9cdc810f4b85db1407e9286c3113821423e17c50908dc0"
SEALED = (
    50036,
    50140,
    50331,
    50403,
    50813,
    51339,
    51709,
    51767,
    52094,
    52175,
    52257,
    52671,
    52850,
    52971,
    53020,
    53198,
    53239,
    53636,
    53671,
    53779,
    55022,
    55379,
    55568,
    56170,
    56966,
    57077,
    57113,
    57494,
    57943,
    59019,
)
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("config_name", "scenario_count"),
    [
        ("paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml", 48),
        ("paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate.yaml", 48),
        ("paper_experiment_matrix_v2_h600_s30_three_width_doorway_v1.yaml", 3),
    ],
)
def test_release_template_resolves_fresh_sealed_grid(config_name, scenario_count):
    cfg = load_campaign_config(ROOT / "configs/benchmarks" / config_name)
    scenarios = _load_campaign_scenarios(cfg, repository_root=ROOT)
    identities = {(row["name"], seed) for row in scenarios for seed in row["seeds"]}
    assert sorted({seed for _, seed in identities}) == list(SEALED)
    assert len({scenario for scenario, _ in identities}) == scenario_count
    assert len(identities) == scenario_count * 30
    assert sum(arm.enabled for arm in cfg.planners) == 14
    assert 14 * len(identities) == 14 * scenario_count * 30


@pytest.mark.parametrize("seed", SEALED)
def test_diff_guard_refuses_fresh_sealed_range(seed):
    assert _range_overlaps_holdout(f"range({seed}, {seed + 1})")
    assert _range_overlaps_holdout(f"range({seed}, {seed - 1}, -1)")


def test_candidate_rejects_retired_seed_band(tmp_path):
    # seed-holdout: synthetic-fixture begin
    policy = {
        "mode": "seed-set",
        "seed_set": "paper_eval_s30",
        "seed_sets_path": "seeds.yaml",
        "resolved_seeds": list(range(111, 141)),
    }  # seed-holdout: synthetic-fixture
    (tmp_path / "seeds.yaml").write_text(
        "paper_eval_s30: [" + ", ".join(map(str, range(111, 141))) + "]\n"
    )  # seed-holdout: synthetic-fixture
    # seed-holdout: synthetic-fixture end
    with pytest.raises(ValueError, match="sealed 0.0.8"):
        _candidate_seed_policy(tmp_path, {"seed_policy": policy}, {"seed_policy": policy})


def test_seed_derivation_matches_author_pin():
    assert hashlib.sha256(LABEL.encode()).hexdigest() == DIGEST
    assert tuple(sorted(random.Random(int(DIGEST, 16)).sample(range(50000, 60000), 30))) == SEALED


def test_source_module_matches_author_derivation():
    from robot_sf.benchmark import seed_bands

    assert seed_bands.EVAL_SEEDS_0_0_8 == SEALED
    assert seed_bands.EVAL_SEEDS_DERIVATION_LABEL == LABEL
    assert seed_bands.EVAL_SEEDS_DERIVATION_SHA256 == DIGEST
    assert seed_bands.DEV_SEEDS == tuple(range(1001, 1031))
    assert seed_bands.RETIRED_EVAL_SEEDS_0_0_7 == tuple(
        range(111, 141)
    )  # seed-holdout: synthetic-fixture
    assert seed_bands.HELD_OUT_SEEDS == frozenset(
        SEALED + tuple(range(111, 141))
    )  # seed-holdout: synthetic-fixture


@pytest.mark.parametrize("seed", SEALED + tuple(range(111, 141)))  # seed-holdout: synthetic-fixture
def test_guard_refuses_both_sealed_bands(seed, tmp_path):
    from scripts.benchmark.run_issue_9748_v4_tuning import _choose
    from scripts.validation.check_issue_9748_dev_split import RELEASE_SEEDS
    from scripts.validation.check_seed_holdout_diff import check_diff

    assert seed in RELEASE_SEEDS
    # Even a corrupted available-set cannot admit held-out values.
    with pytest.raises(ValueError, match="development set"):
        _choose([seed], [seed], label="seed")
    path = "configs/benchmarks/pilot.yaml"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text(f"seeds:\n  - {seed}\n")
    diff = f"+++ b/{path}\n@@ -0,0 +1,2 @@\n+seeds:\n+  - {seed}\n"
    assert len(check_diff(diff, tmp_path)) == 1


def test_release_protocol_rejects_jointly_repinned_retired_band():
    from dataclasses import replace

    from robot_sf.benchmark.release_protocol import (
        _validate_release_seed_policy,
        load_release_manifest,
    )

    cfg = load_campaign_config(
        ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_three_width_doorway_v1.yaml"
    )
    policy = replace(
        cfg.seed_policy,
        seed_set="paper_eval_s30",
        seed_sets_path=ROOT / "configs/benchmarks/seed_sets_v1.yaml",
    )
    cfg = replace(cfg, seed_policy=policy)
    manifest = load_release_manifest(
        ROOT / "configs/benchmarks/releases/three_width_doorway_release_0_0_8_v1.yaml"
    )
    manifest = replace(
        manifest,
        seed_policy={
            **manifest.seed_policy,
            "seed_set": "paper_eval_s30",
            "seed_sets_path": "../seed_sets_v1.yaml",
        },
    )
    problems = []
    _validate_release_seed_policy(manifest, cfg, problems)
    assert problems == [
        "retired evaluation seeds are forbidden for non-historical releases (D-049)",
        "0.0.8 requires the exact sealed evaluation seeds (D-049)",
    ]


@pytest.mark.parametrize("mode", ["fixed-list", "seed-set", "scenario-default"])
@pytest.mark.parametrize("anonymous", [False, True])
def test_protocol_rejects_resolved_retired_seeds_in_every_mode(mode, anonymous, tmp_path):
    """Validate real scenario resolution, including ranges and anonymous input names."""
    from dataclasses import replace

    import yaml

    from robot_sf.benchmark.release_protocol import (
        load_release_manifest,
        validate_release_manifest,
    )

    cfg = load_campaign_config(
        ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_three_width_doorway_v1.yaml"
    )
    manifest = load_release_manifest(
        ROOT / "configs/benchmarks/releases/three_width_doorway_release_0_0_8_v1.yaml"
    )
    # seed-holdout: synthetic-fixture begin
    retired = list(range(111, 141))
    seed_file = tmp_path / "seeds.yaml"
    seed_file.write_text(yaml.safe_dump({"anonymous": [116]}))
    scenario_file = tmp_path / "matrix.yaml"
    scenarios = _load_campaign_scenarios(cfg, repository_root=ROOT)
    for scenario in scenarios:
        scenario["seeds"] = retired
    scenario_file.write_text(yaml.safe_dump(scenarios))
    # seed-holdout: synthetic-fixture end
    config_path = tmp_path / ("campaign.yaml" if anonymous else "campaign_0_0_8.yaml")
    config_payload = yaml.safe_load(
        (
            ROOT
            / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_three_width_doorway_v1.yaml"
        ).read_text()
    )
    policy = {
        "mode": mode,
        "seed_set": "anonymous" if mode == "seed-set" else None,
        "seeds": retired if mode == "fixed-list" else [],
        "seed_sets_path": str(seed_file),
    }
    config_payload.update(scenario_matrix=str(scenario_file), seed_policy=policy)
    config_path.write_text(yaml.safe_dump(config_payload))
    cfg = replace(
        cfg,
        scenario_matrix_path=scenario_file,
        seed_policy=replace(
            cfg.seed_policy,
            mode=mode,
            seed_set=policy["seed_set"],
            seeds=tuple(policy["seeds"]),
            seed_sets_path=seed_file,
        ),
    )
    manifest = replace(
        manifest,
        seed_policy=policy,
        canonical_campaign_config_path=config_path,
        campaign_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        scenario_matrix_path=scenario_file,
        scenario_matrix_sha256=hashlib.sha256(scenario_file.read_bytes()).hexdigest(),
        seed_sets_sha256=hashlib.sha256(seed_file.read_bytes()).hexdigest(),
        resolved_seeds=tuple(
            [116] if mode == "seed-set" else retired
        ),  # seed-holdout: synthetic-fixture
    )
    if anonymous:
        manifest = replace(manifest, release_tag="future-release", release_id="future-release")
    report = validate_release_manifest(manifest, campaign_config=cfg, repository_root=ROOT)
    expected = ["retired evaluation seeds are forbidden for non-historical releases (D-049)"]
    if not anonymous:
        expected.append("0.0.8 requires the exact sealed evaluation seeds (D-049)")
    assert report["problems"] == expected


def test_stress_gate_resolves_only_dev_seed():
    cfg = load_campaign_config(
        ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_hybrid_stress_smoke.yaml"
    )
    scenarios = _load_campaign_scenarios(cfg, repository_root=ROOT)
    assert {seed for row in scenarios for seed in row["seeds"]} == {1001}
    assert len(scenarios) * sum(arm.enabled for arm in cfg.planners) == 70
