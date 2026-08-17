"""Tests for the leakage-safe parametric curriculum fixture contract."""

from __future__ import annotations

import copy

import pytest

from robot_sf.training.parametric_curriculum import (
    PARAMETRIC_CURRICULUM_DIAGNOSTIC_SCHEMA,
    ParameterDimension,
    ScenarioParameterSpace,
    build_manifest,
    build_parameter_space,
    build_parametric_curriculum_report,
    validate_no_leakage,
    verify_replay,
)


def _space() -> ScenarioParameterSpace:
    """Return the compact six-dimension social-navigation smoke space."""

    return build_parameter_space(
        {
            "density_m2": {"kind": "continuous", "minimum": 0.02, "maximum": 0.12},
            "pedestrian_speed_mps": {"kind": "continuous", "minimum": 0.5, "maximum": 1.8},
            "constriction_ratio": {"kind": "continuous", "minimum": 0.35, "maximum": 0.95},
            "interaction_type": {
                "kind": "categorical",
                "values": ["crossing", "following", "doorway"],
            },
            "robot_speed_mps": {"kind": "continuous", "minimum": 0.5, "maximum": 2.0},
            "adversariality": {"kind": "continuous", "minimum": 0.0, "maximum": 1.0},
        }
    )


def test_parameter_sampling_is_deterministic_and_strategy_specific() -> None:
    """Fixed seeds replay random draws while structured draws traverse declared bounds."""

    space = _space()
    random_a = space.sample(seed=7316, count=4, strategy="random")
    random_b = space.sample(seed=7316, count=4, strategy="random")
    assert random_a == random_b
    structured = space.sample(seed=999, count=3, strategy="structured")
    assert structured[0]["density_m2"] == 0.02
    assert structured[-1]["density_m2"] == 0.12
    assert structured[0]["pedestrian_speed_mps"] == 1.8
    assert structured[-1]["pedestrian_speed_mps"] == 0.5


def test_parameter_space_rejects_missing_extra_and_out_of_range_values() -> None:
    """Parameter vectors must match the declared support exactly."""

    space = _space()
    valid = space.sample(seed=1, count=1, strategy="fixed")[0]
    with pytest.raises(ValueError, match="keys mismatch"):
        space.validate({**valid, "unexpected": 1.0})
    invalid = dict(valid)
    invalid["density_m2"] = 10.0
    with pytest.raises(ValueError, match="finite in"):
        space.validate(invalid)


def test_manifest_leakage_check_fails_closed_on_parameter_overlap() -> None:
    """Different scenario IDs cannot hide an identical train/evaluation vector."""

    space = _space()
    train = build_manifest(
        space,
        split="train",
        seed=1,
        count=1,
        strategy="fixed",
        scenario_prefix="train",
    )
    evaluation_entry = copy.deepcopy(train.entries[0])
    evaluation_entry["scenario_id"] = "evaluation-000"
    from robot_sf.training.parametric_curriculum import CurriculumManifest

    evaluation = CurriculumManifest(
        split="evaluation",
        seed=2,
        strategy="fixed",
        entries=(evaluation_entry,),
    )
    with pytest.raises(ValueError, match="leakage detected"):
        validate_no_leakage(train, evaluation)


def test_manifest_replay_is_exact_and_detects_mutation() -> None:
    """Replay uses the seed and strategy, and a changed entry cannot pass silently."""

    space = _space()
    manifest = build_manifest(
        space,
        split="train",
        seed=7316,
        count=4,
        strategy="structured",
        scenario_prefix="structured-train",
    )
    assert verify_replay(space, manifest) is True
    mutated = list(manifest.entries)
    mutated[0] = {**mutated[0], "parameters": {**mutated[0]["parameters"], "density_m2": 0.03}}
    from robot_sf.training.parametric_curriculum import CurriculumManifest

    with pytest.raises(ValueError, match="replay mismatch"):
        verify_replay(
            space,
            CurriculumManifest(
                split="train",
                seed=manifest.seed,
                strategy=manifest.strategy,
                entries=tuple(mutated),
            ),
        )


def test_report_contains_three_unexecuted_methods_and_independent_hashes() -> None:
    """The report records the three methodology lanes without promoting outcomes."""

    report = build_parametric_curriculum_report(
        _space(), seed=7316, train_count=6, evaluation_count=4
    )
    assert report["schema_version"] == PARAMETRIC_CURRICULUM_DIAGNOSTIC_SCHEMA
    assert [method["method_id"] for method in report["methods"]] == [
        "no_curriculum",
        "random_curriculum",
        "structured_curriculum",
    ]
    assert all(method["replay_verified"] for method in report["methods"])
    assert all(not method["training_executed"] for method in report["methods"])
    assert report["simulator_executed"] is False
    assert len({method["training_manifest"]["sha256"] for method in report["methods"]}) == 3
    assert all(
        method["evaluation_manifest"]["sha256"]
        == report["methods"][0]["evaluation_manifest"]["sha256"]
        for method in report["methods"]
    )


def test_dimension_validation_is_strict() -> None:
    """Dimension constructors reject unsupported or ambiguous declarations."""

    with pytest.raises(ValueError, match="requires finite"):
        ParameterDimension("density", "continuous", minimum=1.0, maximum=1.0)
    with pytest.raises(ValueError, match="duplicate"):
        ParameterDimension("interaction", "categorical", values=("crossing", "crossing"))
