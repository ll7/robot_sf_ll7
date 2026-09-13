"""Unit tests for benchmark type dataclasses."""

from __future__ import annotations

from datetime import datetime
from types import MappingProxyType

from robot_sf.benchmark.types import (
    CANONICAL_EPISODE_RECORD_FIELDS,
    CANONICAL_EPISODE_RUNTIME_FIELDS,
    CANONICAL_EPISODE_RUNTIME_METRICS,
    EpisodeRecord,
    MetricsBundle,
    ResumeManifest,
    ScenarioSpec,
    SNQIWeights,
)


def test_scenario_spec_to_dict_round_trip() -> None:
    """ScenarioSpec should serialize all declared fields."""
    spec = ScenarioSpec(
        id="s1",
        algo="sf",
        map="classic_doorway",
        episodes=4,
        seed=11,
        notes="smoke",
        algo_config_path="configs/alg/sf.yaml",
        meta={"density": "low"},
    )

    payload = spec.to_dict()

    assert payload["id"] == "s1"
    assert payload["algo"] == "sf"
    assert payload["map"] == "classic_doorway"
    assert payload["episodes"] == 4
    assert payload["seed"] == 11
    assert payload["notes"] == "smoke"
    assert payload["algo_config_path"] == "configs/alg/sf.yaml"
    assert payload["meta"] == {"density": "low"}


def test_metrics_bundle_get_and_copy() -> None:
    """MetricsBundle should support default lookups and defensive dict conversion."""
    bundle = MetricsBundle(values={"success_rate": 0.9})

    assert bundle.get("success_rate") == 0.9
    assert bundle.get("missing", 1.0) == 1.0

    copied = bundle.to_dict()
    copied["success_rate"] = 0.1
    assert bundle.values["success_rate"] == 0.9


def test_episode_record_to_dict_flattens_metrics() -> None:
    """EpisodeRecord serialization should flatten the MetricsBundle field."""
    record = EpisodeRecord(
        version="v1",
        episode_id="ep-1",
        scenario_id="scenario-1",
        seed=42,
        metrics=MetricsBundle(values={"success_rate": 1.0, "snqi": 0.5}),
        algo="sf",
        horizon=128,
        timing={"steps_per_sec": 123.0},
        tags=["smoke"],
        identity={"robot": "r1"},
        raw={"debug": True},
    )

    payload = record.to_dict()

    assert payload["metrics"] == {"success_rate": 1.0, "snqi": 0.5}
    assert payload["algo"] == "sf"
    assert payload["horizon"] == 128
    assert payload["timing"] == {"steps_per_sec": 123.0}
    assert payload["tags"] == ["smoke"]
    assert payload["identity"] == {"robot": "r1"}
    assert payload["raw"] == {"debug": True}


def test_episode_record_canonical_digest_excludes_runtime_surfaces() -> None:
    """Stable episode comparisons ignore timing/debug data, not semantic inputs."""
    first = EpisodeRecord(
        version="v1",
        episode_id="ep-1",
        scenario_id="scenario-1",
        seed=42,
        metrics=MetricsBundle(values={"steps": 4.0, "duration_s": 1.0}),
        algo="zero_action",
        horizon=4,
        timing={"wall_time_s": 1.0},
        tags=["smoke"],
        identity={"scenario": "scenario-1"},
        raw={"debug": "first"},
    )
    second = EpisodeRecord(
        version="v1",
        episode_id="ep-1",
        scenario_id="scenario-1",
        seed=42,
        metrics=MetricsBundle(values={"steps": 4.0, "duration_s": 9.0}),
        algo="zero_action",
        horizon=4,
        timing={"wall_time_s": 9.0},
        tags=["smoke"],
        identity={"scenario": "scenario-1"},
        raw={"debug": "second"},
    )

    assert CANONICAL_EPISODE_RECORD_FIELDS == (
        "version",
        "episode_id",
        "scenario_id",
        "seed",
        "metrics",
        "algo",
        "horizon",
        "tags",
        "identity",
    )
    assert CANONICAL_EPISODE_RUNTIME_FIELDS == frozenset({"timing", "raw"})
    assert CANONICAL_EPISODE_RUNTIME_METRICS == frozenset({"duration_s"})
    assert first.canonical_payload() == second.canonical_payload()
    assert first.canonical_digest() == second.canonical_digest()
    assert "timing" not in first.canonical_payload()
    assert "raw" not in first.canonical_payload()
    assert "duration_s" not in first.canonical_payload()["metrics"]

    changed_seed = EpisodeRecord(
        version="v1",
        episode_id="ep-2",
        scenario_id="scenario-1",
        seed=43,
        metrics=MetricsBundle(values={"steps": 4.0, "duration_s": 1.0}),
        algo="zero_action",
        horizon=4,
        tags=["smoke"],
        identity={"scenario": "scenario-1"},
    )
    assert changed_seed.canonical_digest() != first.canonical_digest()


def test_snqi_weights_to_dict_handles_mapping_and_default_meta() -> None:
    """SNQI weights should convert arbitrary mapping types and normalize meta."""
    weights = SNQIWeights(
        version="v1", weights=MappingProxyType({"collision_rate": 0.7}), meta=None
    )

    payload = weights.to_dict()

    assert payload == {
        "version": "v1",
        "weights": {"collision_rate": 0.7},
        "meta": {},
    }


def test_resume_manifest_to_dict_uses_default_timestamp_and_meta() -> None:
    """ResumeManifest should default meta to empty dict and emit ISO UTC timestamp."""
    manifest = ResumeManifest(version="v1", episodes=["ep-1", "ep-2"], meta=None)

    payload = manifest.to_dict()
    parsed = datetime.fromisoformat(payload["generated_at"])

    assert payload["version"] == "v1"
    assert payload["episodes"] == ["ep-1", "ep-2"]
    assert payload["meta"] == {}
    assert parsed.tzinfo is not None
    assert parsed.microsecond == 0
