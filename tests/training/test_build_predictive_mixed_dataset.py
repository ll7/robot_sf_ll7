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
) -> None:
    """Write a compact predictive dataset fixture with optional embedded schema metadata."""
    payload: dict[str, object] = {
        "state": np.zeros((2, 3, input_dim), dtype=np.float32),
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

    monkeypatch.setattr(
        mixed_builder,
        "parse_args",
        lambda: mixed_builder.argparse.Namespace(
            base_dataset=base_path,
            hardcase_dataset=hardcase_path,
            hardcase_repeat=2,
            output=output_path,
            shuffle_seed=7,
        ),
    )

    assert mixed_builder.main() == 0
    with np.load(output_path) as raw:
        mixed_schema = json.loads(str(raw["feature_schema_json"].item()))
    summary = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))

    assert mixed_schema["ego_motion_channel_producer"]["producer_key"] == (
        PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME
    )
    assert summary["feature_schema"]["ego_motion_channel_producer"]["producer_key"] == (
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

    monkeypatch.setattr(
        mixed_builder,
        "parse_args",
        lambda: mixed_builder.argparse.Namespace(
            base_dataset=base_path,
            hardcase_dataset=hardcase_path,
            hardcase_repeat=2,
            output=output_path,
            shuffle_seed=7,
        ),
    )

    assert mixed_builder.main() == 0
    with np.load(output_path) as raw:
        assert "feature_schema_json" not in raw
    summary = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert summary["feature_schema"] is None
    assert summary["feature_schema_json"] is None
