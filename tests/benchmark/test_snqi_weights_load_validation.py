"""Fail-closed validation regressions for ``SNQIWeights`` JSON/config loading.

Covers issue #3710: ``SNQIWeights.load`` / ``from_dict`` previously raised bare
``KeyError`` on missing provenance metadata and silently accepted malformed
weight values. These tests pin the descriptive, fail-closed contract.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.snqi.types import SNQIWeights

if TYPE_CHECKING:
    from pathlib import Path


def _valid_payload() -> dict[str, object]:
    """Return a minimal well-formed SNQIWeights mapping."""
    return {
        "weights_version": "v1",
        "created_at": "2026-01-01T00:00:00",
        "git_sha": "abc123",
        "baseline_stats_path": "baseline.json",
        "baseline_stats_hash": "deadbeef",
        "normalization_strategy": "median_p95_clamp",
        "bootstrap_params": {"method": "balanced", "seed": 7},
        "components": ["w_success", "w_time"],
        "weights": {"w_success": 1.0, "w_time": 0.8},
    }


def test_from_dict_round_trips_valid_payload() -> None:
    """A well-formed mapping reconstructs without error and coerces weights to float."""
    weights = SNQIWeights.from_dict(_valid_payload())

    assert weights.weights_version == "v1"
    assert weights.bootstrap_params == {"method": "balanced", "seed": 7}
    assert weights.components == ["w_success", "w_time"]
    assert weights.weights == {"w_success": 1.0, "w_time": 0.8}
    assert all(isinstance(v, float) for v in weights.weights.values())


def test_from_dict_rejects_non_mapping() -> None:
    """A JSON array (or any non-mapping) fails closed with a descriptive error."""
    with pytest.raises(ValueError, match="must be a JSON object/mapping"):
        SNQIWeights.from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_from_dict_missing_required_metadata_lists_keys() -> None:
    """Missing provenance metadata raises ValueError naming the absent keys (not KeyError)."""
    payload = _valid_payload()
    del payload["git_sha"]
    del payload["baseline_stats_hash"]

    with pytest.raises(ValueError, match="missing required field") as exc_info:
        SNQIWeights.from_dict(payload)

    message = str(exc_info.value)
    assert "git_sha" in message
    assert "baseline_stats_hash" in message


def test_from_dict_rejects_non_string_metadata() -> None:
    """Non-string provenance metadata fails closed with a descriptive error."""
    payload = _valid_payload()
    payload["weights_version"] = 123

    with pytest.raises(ValueError, match="'weights_version' must be a string"):
        SNQIWeights.from_dict(payload)


@pytest.mark.parametrize("bad_weight", [-1.0, float("nan"), float("inf")])
def test_from_dict_rejects_non_finite_or_negative_weights(bad_weight: float) -> None:
    """Negative or non-finite weight values fail closed."""
    payload = _valid_payload()
    payload["weights"] = {"w_success": bad_weight}

    with pytest.raises(ValueError, match="finite and non-negative"):
        SNQIWeights.from_dict(payload)


@pytest.mark.parametrize("bad_weight", ["1.0", None, True])
def test_from_dict_rejects_non_numeric_weights(bad_weight: object) -> None:
    """Non-numeric weight values (including bools) fail closed."""
    payload = _valid_payload()
    payload["weights"] = {"w_success": bad_weight}

    with pytest.raises(ValueError, match="is not numeric"):
        SNQIWeights.from_dict(payload)


def test_from_dict_rejects_non_mapping_weights() -> None:
    """A weights field that is not a mapping fails closed."""
    payload = _valid_payload()
    payload["weights"] = ["w_success", 1.0]

    with pytest.raises(ValueError, match="'weights' must be a mapping"):
        SNQIWeights.from_dict(payload)


def test_from_dict_rejects_non_mapping_bootstrap_params() -> None:
    """A bootstrap_params field that is not a mapping fails closed."""
    payload = _valid_payload()
    payload["bootstrap_params"] = ["not", "a", "mapping"]

    with pytest.raises(ValueError, match="'bootstrap_params' must be a mapping"):
        SNQIWeights.from_dict(payload)


def test_from_dict_rejects_non_list_components() -> None:
    """A components field that is not a list fails closed."""
    payload = _valid_payload()
    payload["components"] = "w_success"

    with pytest.raises(ValueError, match="'components' must be a list"):
        SNQIWeights.from_dict(payload)


def test_load_round_trips_saved_file(tmp_path: Path) -> None:
    """save() output reloads cleanly through load()."""
    path = tmp_path / "weights.json"
    SNQIWeights.from_dict(_valid_payload()).save(path)

    reloaded = SNQIWeights.load(path)

    assert reloaded.weights_version == "v1"
    assert reloaded.weights == {"w_success": 1.0, "w_time": 0.8}


def test_load_invalid_json_includes_path(tmp_path: Path) -> None:
    """Malformed JSON fails closed with the source path in the message."""
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON") as exc_info:
        SNQIWeights.load(path)

    assert str(path) in str(exc_info.value)


def test_load_malformed_config_includes_path(tmp_path: Path) -> None:
    """A structurally-invalid config surfaces the originating file path."""
    path = tmp_path / "missing_keys.json"
    payload = _valid_payload()
    del payload["weights_version"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required field") as exc_info:
        SNQIWeights.load(path)

    assert str(path) in str(exc_info.value)
