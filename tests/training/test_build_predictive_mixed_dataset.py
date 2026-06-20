"""Regression tests for predictive mixed-dataset feature-schema metadata."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from robot_sf.planner.obstacle_features import (
    PREDICTIVE_EGO_FEATURE_SCHEMA,
    PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME,
    PREDICTIVE_EGO_MOTION_PRODUCER_STANDALONE,
    predictive_feature_schema_metadata,
)
from scripts.training import build_predictive_mixed_dataset as mixed_builder

if TYPE_CHECKING:
    from pathlib import Path


def _write_dataset(
    path: Path,
    *,
    input_dim: int,
    feature_schema: dict[str, object] | None = None,
    sample_offset: int = 0,
) -> None:
    """Write a compact predictive dataset fixture with optional embedded schema metadata."""
    state = np.zeros((2, 3, input_dim), dtype=np.float32)
    for sample_idx in range(state.shape[0]):
        state[sample_idx, :, 0] = sample_offset + sample_idx
    payload: dict[str, object] = {
        "state": state,
        "target": np.zeros((2, 3, 5, 2), dtype=np.float32),
        "mask": np.ones((2, 3), dtype=np.float32),
        "target_mask": np.ones((2, 3, 5), dtype=np.float32),
    }
    if feature_schema is not None:
        payload["feature_schema_json"] = np.asarray(json.dumps(feature_schema, sort_keys=True))
    np.savez_compressed(path, **payload)


def _ego_feature_schema(producer: str | None) -> dict[str, object]:
    """Return predictive ego schema metadata for mixed-dataset tests."""
    return predictive_feature_schema_metadata(
        model_family=PREDICTIVE_EGO_FEATURE_SCHEMA,
        ego_conditioning=True,
        ego_motion_channel_producer=producer,
    )


def _run_builder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    base_path: Path,
    hardcase_path: Path,
    output_path: Path,
    hardcase_repeat: int | None = 2,
    shuffle_seed: int | None = 7,
    weighting_spec: Path | None = None,
) -> None:
    """Run the mixed-dataset builder through a compact monkeypatched CLI namespace."""
    monkeypatch.setattr(
        mixed_builder,
        "parse_args",
        lambda: mixed_builder.argparse.Namespace(
            base_dataset=base_path,
            hardcase_dataset=hardcase_path,
            hardcase_repeat=hardcase_repeat,
            output=output_path,
            shuffle_seed=shuffle_seed,
            weighting_spec=weighting_spec,
        ),
    )

    assert mixed_builder.main() == 0


def test_mixed_builder_propagates_matching_ego_feature_schema_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Mixed ego datasets should preserve comparable producer metadata into the output NPZ."""
    base_path = tmp_path / "predictive_rollouts_base.npz"
    hardcase_path = tmp_path / "predictive_rollouts_hardcase.npz"
    output_path = tmp_path / "predictive_rollouts_mixed.npz"
    schema = _ego_feature_schema(PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME)
    _write_dataset(base_path, input_dim=9, feature_schema=schema)
    _write_dataset(hardcase_path, input_dim=9, feature_schema=schema)

    _run_builder(
        monkeypatch,
        base_path=base_path,
        hardcase_path=hardcase_path,
        output_path=output_path,
    )
    with np.load(output_path) as raw:
        mixed_schema = json.loads(str(raw["feature_schema_json"].item()))
    summary = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))

    assert mixed_schema["ego_motion_channel_producer"]["producer_key"] == (
        PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME
    )
    assert summary["feature_schema"]["ego_motion_channel_producer"]["producer_key"] == (
        PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME
    )
    assert summary["ego_motion_channel_producer"]["producer_key"] == (
        PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME
    )


def test_mixed_builder_rejects_mismatched_ego_producers(monkeypatch, tmp_path: Path) -> None:
    """Mixed ego datasets must fail closed when producer keys disagree."""
    base_path = tmp_path / "predictive_rollouts_base.npz"
    hardcase_path = tmp_path / "predictive_rollouts_hardcase.npz"
    _write_dataset(
        base_path,
        input_dim=9,
        feature_schema=_ego_feature_schema(PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME),
    )
    _write_dataset(
        hardcase_path,
        input_dim=9,
        feature_schema=_ego_feature_schema(PREDICTIVE_EGO_MOTION_PRODUCER_STANDALONE),
    )

    monkeypatch.setattr(
        mixed_builder,
        "parse_args",
        lambda: mixed_builder.argparse.Namespace(
            base_dataset=base_path,
            hardcase_dataset=hardcase_path,
            hardcase_repeat=2,
            output=tmp_path / "predictive_rollouts_mixed.npz",
            shuffle_seed=7,
        ),
    )

    with pytest.raises(ValueError, match="producer mismatch"):
        mixed_builder.main()


def test_mixed_builder_rejects_missing_schema_for_ego_conditioned_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Ego-conditioned mixed datasets must not be created when one input lacks schema metadata."""
    base_path = tmp_path / "predictive_rollouts_base.npz"
    hardcase_path = tmp_path / "predictive_rollouts_hardcase.npz"
    _write_dataset(
        base_path,
        input_dim=9,
        feature_schema=_ego_feature_schema(PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME),
    )
    _write_dataset(hardcase_path, input_dim=9, feature_schema=None)

    monkeypatch.setattr(
        mixed_builder,
        "parse_args",
        lambda: mixed_builder.argparse.Namespace(
            base_dataset=base_path,
            hardcase_dataset=hardcase_path,
            hardcase_repeat=2,
            output=tmp_path / "predictive_rollouts_mixed.npz",
            shuffle_seed=7,
        ),
    )

    with pytest.raises(ValueError, match="require feature_schema_json"):
        mixed_builder.main()


def test_mixed_builder_preserves_legacy_compatibility_without_schema(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Legacy non-ego datasets without schema metadata should still mix successfully."""
    base_path = tmp_path / "predictive_rollouts_base.npz"
    hardcase_path = tmp_path / "predictive_rollouts_hardcase.npz"
    output_path = tmp_path / "predictive_rollouts_mixed.npz"
    _write_dataset(base_path, input_dim=4, feature_schema=None)
    _write_dataset(hardcase_path, input_dim=4, feature_schema=None)

    _run_builder(
        monkeypatch,
        base_path=base_path,
        hardcase_path=hardcase_path,
        output_path=output_path,
    )
    with np.load(output_path) as raw:
        assert "feature_schema_json" not in raw
    summary = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert summary["feature_schema"] is None
    assert summary["feature_schema_json"] is None
    assert summary["ego_motion_channel_producer"] is None


def test_mixed_builder_applies_crossing_conflict_weighting_spec_deterministically(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Crossing-conflict hard-case weighting should be explicit and deterministic."""
    base_path = tmp_path / "predictive_rollouts_base.npz"
    hardcase_path = tmp_path / "predictive_rollouts_crossing_conflict.npz"
    output_a = tmp_path / "predictive_rollouts_mixed_a.npz"
    output_b = tmp_path / "predictive_rollouts_mixed_b.npz"
    spec_path = tmp_path / "crossing_conflict_weighting.yaml"
    schema = _ego_feature_schema(PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME)
    _write_dataset(base_path, input_dim=9, feature_schema=schema, sample_offset=10)
    _write_dataset(hardcase_path, input_dim=9, feature_schema=schema, sample_offset=100)
    spec_path.write_text(
        "\n".join(
            [
                "profile_id: crossing_conflict_hardcase_repeat_test",
                "claim_boundary: launch/config/tooling only; no model-improvement claim",
                "weighting:",
                "  rule: repeat_hardcase_rows",
                "  hardcase_family: crossing_conflict",
                "  hardcase_repeat: 3",
                "  shuffle_seed: 3214",
            ]
        ),
        encoding="utf-8",
    )

    _run_builder(
        monkeypatch,
        base_path=base_path,
        hardcase_path=hardcase_path,
        output_path=output_a,
        hardcase_repeat=None,
        shuffle_seed=None,
        weighting_spec=spec_path,
    )
    _run_builder(
        monkeypatch,
        base_path=base_path,
        hardcase_path=hardcase_path,
        output_path=output_b,
        hardcase_repeat=None,
        shuffle_seed=None,
        weighting_spec=spec_path,
    )

    with np.load(output_a) as raw_a, np.load(output_b) as raw_b:
        np.testing.assert_array_equal(raw_a["state"], raw_b["state"])
        mixed_sample_ids = raw_a["state"][:, 0, 0].astype(int).tolist()
    summary = json.loads(output_a.with_suffix(".json").read_text(encoding="utf-8"))

    assert mixed_sample_ids.count(10) == 1
    assert mixed_sample_ids.count(11) == 1
    assert mixed_sample_ids.count(100) == 3
    assert mixed_sample_ids.count(101) == 3
    assert summary["base_count"] == 2
    assert summary["hard_case_count"] == 2
    assert summary["output_count"] == 8
    assert summary["weighting_profile"] == "crossing_conflict_hardcase_repeat_test"
    assert summary["weighting_rule"]["rule"] == "repeat_hardcase_rows"
    assert summary["weighting_rule"]["hardcase_family"] == "crossing_conflict"
    assert summary["weighting_rule"]["hardcase_repeat"] == 3
    assert summary["shuffle_seed"] == 3214
    assert summary["feature_compatibility"]["status"] == "compatible"
    assert summary["feature_compatibility"]["feature_schema_required"] is True


def test_mixed_builder_rejects_mismatched_feature_schema_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Predictive schemas with different contracts must fail closed before mixing."""
    base_path = tmp_path / "predictive_rollouts_base.npz"
    hardcase_path = tmp_path / "predictive_rollouts_hardcase.npz"
    base_schema = _ego_feature_schema(PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME)
    hardcase_schema = dict(base_schema)
    hardcase_schema["base_schema"] = "predictive_legacy_v1"
    _write_dataset(base_path, input_dim=9, feature_schema=base_schema)
    _write_dataset(hardcase_path, input_dim=9, feature_schema=hardcase_schema)

    monkeypatch.setattr(
        mixed_builder,
        "parse_args",
        lambda: mixed_builder.argparse.Namespace(
            base_dataset=base_path,
            hardcase_dataset=hardcase_path,
            hardcase_repeat=2,
            output=tmp_path / "predictive_rollouts_mixed.npz",
            shuffle_seed=7,
            weighting_spec=None,
        ),
    )

    with pytest.raises(ValueError, match="feature schema mismatch"):
        mixed_builder.main()


def test_mixed_builder_rejects_schema_input_dim_that_disagrees_with_array_width(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Mixed ego metadata must match the actual feature width before schemas are accepted."""
    base_path = tmp_path / "predictive_rollouts_base.npz"
    hardcase_path = tmp_path / "predictive_rollouts_hardcase.npz"
    schema = _ego_feature_schema(PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME)
    wrong_schema = dict(schema)
    wrong_schema["input_dim"] = 15
    _write_dataset(base_path, input_dim=9, feature_schema=wrong_schema)
    _write_dataset(hardcase_path, input_dim=9, feature_schema=wrong_schema)

    monkeypatch.setattr(
        mixed_builder,
        "parse_args",
        lambda: mixed_builder.argparse.Namespace(
            base_dataset=base_path,
            hardcase_dataset=hardcase_path,
            hardcase_repeat=2,
            output=tmp_path / "predictive_rollouts_mixed.npz",
            shuffle_seed=7,
        ),
    )

    with pytest.raises(ValueError, match="array width=9"):
        mixed_builder.main()


def test_mixed_builder_reports_dataset_path_for_invalid_feature_schema_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Invalid schema JSON errors should identify the dataset that failed to parse."""
    base_path = tmp_path / "predictive_rollouts_base.npz"
    hardcase_path = tmp_path / "predictive_rollouts_hardcase.npz"
    schema = _ego_feature_schema(PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME)
    _write_dataset(base_path, input_dim=9, feature_schema=schema)
    np.savez_compressed(
        hardcase_path,
        state=np.zeros((2, 3, 9), dtype=np.float32),
        target=np.zeros((2, 3, 5, 2), dtype=np.float32),
        mask=np.ones((2, 3), dtype=np.float32),
        target_mask=np.ones((2, 3, 5), dtype=np.float32),
        feature_schema_json=np.asarray("{not-json"),
    )

    monkeypatch.setattr(
        mixed_builder,
        "parse_args",
        lambda: mixed_builder.argparse.Namespace(
            base_dataset=base_path,
            hardcase_dataset=hardcase_path,
            hardcase_repeat=2,
            output=tmp_path / "predictive_rollouts_mixed.npz",
            shuffle_seed=7,
        ),
    )

    with pytest.raises(ValueError, match=f"Invalid feature_schema_json in dataset {hardcase_path}"):
        mixed_builder.main()
