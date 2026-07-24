"""Tests for shared Gymnasium reset metadata helpers."""

from __future__ import annotations

from types import SimpleNamespace

from robot_sf.gym_env.reset_metadata import build_reset_metadata, resolve_map_id


def _config_with_map_pool(
    pool: dict[str, object] | None = None,
    *,
    map_id: str | None = None,
    sim_time: float = 10.0,
    time_per_step: float = 0.1,
    max_sim_steps: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        map_pool=SimpleNamespace(map_defs=pool or {}),
        map_id=map_id,
        sim_config=SimpleNamespace(
            sim_time_in_secs=sim_time,
            time_per_step_in_secs=time_per_step,
            max_sim_steps=max_sim_steps,
        ),
    )


def test_resolve_map_id_finds_by_identity() -> None:
    map_def = object()
    config = _config_with_map_pool({"map_a": map_def})
    assert resolve_map_id(config, map_def) == "map_a"


def test_resolve_map_id_returns_first_match() -> None:
    map_def = object()
    config = _config_with_map_pool({"map_x": object(), "map_y": map_def, "map_z": map_def})
    assert resolve_map_id(config, map_def) == "map_y"


def test_resolve_map_id_falls_back_to_config_map_id() -> None:
    map_def = object()
    config = _config_with_map_pool({"other": object()}, map_id="fallback_id")
    assert resolve_map_id(config, map_def) == "fallback_id"


def test_resolve_map_id_returns_none_when_no_match() -> None:
    config = _config_with_map_pool({"other": object()}, map_id=None)
    assert resolve_map_id(config, object()) is None


def test_resolve_map_id_handles_missing_map_pool() -> None:
    config = SimpleNamespace(map_id="from_config")
    assert resolve_map_id(config, object()) == "from_config"


def test_resolve_map_id_handles_missing_map_id() -> None:
    config = SimpleNamespace()
    assert resolve_map_id(config, object()) is None


def test_resolve_map_id_handles_type_error() -> None:
    config = SimpleNamespace(
        map_pool=SimpleNamespace(map_defs=SimpleNamespace(items=None)), map_id="fallback"
    )
    assert resolve_map_id(config, object()) == "fallback"


def test_build_reset_metadata_contains_required_keys() -> None:
    config = _config_with_map_pool({})
    map_def = object()
    metadata = build_reset_metadata(config, map_def=map_def, seed=42)
    assert set(metadata.keys()) == {
        "map_id",
        "sim_time_in_secs",
        "time_per_step_in_secs",
        "max_sim_steps",
        "seed",
    }


def test_build_reset_metadata_computes_max_sim_steps_from_fallback() -> None:
    config = _config_with_map_pool({}, sim_time=20.0, time_per_step=0.5, max_sim_steps=None)
    metadata = build_reset_metadata(config, map_def=object(), seed=0)
    assert metadata["max_sim_steps"] == 40


def test_build_reset_metadata_honors_explicit_max_sim_steps() -> None:
    config = _config_with_map_pool({}, sim_time=999.0, time_per_step=0.1, max_sim_steps=77)
    metadata = build_reset_metadata(config, map_def=object(), seed=0)
    assert metadata["max_sim_steps"] == 77


def test_build_reset_metadata_casts_floats() -> None:
    config = _config_with_map_pool({}, sim_time=5, time_per_step=1)
    metadata = build_reset_metadata(config, map_def=object(), seed=1)
    assert isinstance(metadata["sim_time_in_secs"], float)
    assert isinstance(metadata["time_per_step_in_secs"], float)


def test_build_reset_metadata_merges_extra() -> None:
    config = _config_with_map_pool({})
    metadata = build_reset_metadata(
        config,
        map_def=object(),
        seed=7,
        extra={"custom_key": "val", "another": 42},
    )
    assert metadata["custom_key"] == "val"
    assert metadata["another"] == 42


def test_build_reset_metadata_extra_overwrites_none() -> None:
    config = _config_with_map_pool({})
    metadata = build_reset_metadata(
        config,
        map_def=object(),
        seed=0,
        extra={"map_id": "override"},
    )
    assert metadata["map_id"] == "override"


def test_build_reset_metadata_seed_preserved() -> None:
    config = _config_with_map_pool({})
    metadata = build_reset_metadata(config, map_def=object(), seed=12345)
    assert metadata["seed"] == 12345
