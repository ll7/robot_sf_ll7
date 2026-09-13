"""Contract tests for canonical aggregate-cell and metric-source provenance."""

from __future__ import annotations

import typing
from dataclasses import replace
from pathlib import Path

import pytest

from robot_sf.benchmark.aggregate import (
    AggregateCellIdentity,
    AggregateCellProvenance,
    AggregateProvenanceError,
    compute_aggregates,
    read_jsonl,
    resolve_aggregate_cell_provenance,
)
from robot_sf.benchmark.metric_layers import (
    CANONICAL_METRICS,
    METRIC_BINDING_OWNER,
    MetricBindingError,
    MetricDefinition,
    MetricSourceBinding,
    resolve_metric_source_binding,
)


def _episode(
    episode_id: str | None,
    *,
    algo: str = "planner_a",
    collision_rate: float | None = 0.0,
    evidence_eligible: bool = True,
    benchmark_track: str = "state",
) -> dict[str, object]:
    """Build one aggregation-compatible episode with explicit identity and provenance."""
    metrics = {} if collision_rate is None else {"collision_rate": collision_rate}
    return {
        "episode_id": episode_id,
        "scenario_id": "crossing",
        "scenario_params": {"algo": algo},
        "benchmark_track": benchmark_track,
        "algorithm_metadata": {"foresight_prediction": {"evidence_eligible": evidence_eligible}},
        "metrics": metrics,
    }


def _identity(*, group: str = "planner_a", metric: str = "collision_rate") -> AggregateCellIdentity:
    """Return the canonical selector used by focused provenance tests."""
    return AggregateCellIdentity(group_identity=group, metric_id=metric, statistic="mean")


def test_aggregate_cell_provenance_uses_exact_canonical_numeric_contributors() -> None:
    """Only eligible rows with the selected numeric metric contribute to the reported cell."""
    records = [
        _episode("ep-2", collision_rate=1.0),
        _episode("ep-missing", collision_rate=None),
        _episode("ep-ineligible", collision_rate=0.0, evidence_eligible=False),
        _episode("ep-1", collision_rate=0.0),
        _episode("other-group", algo="planner_b", collision_rate=1.0),
    ]

    aggregate = compute_aggregates(records)
    provenance = resolve_aggregate_cell_provenance(records, _identity())

    assert provenance.value == aggregate["planner_a"]["collision_rate"]["mean"] == 0.5
    assert provenance.contributor_episode_ids == ("ep-1", "ep-2")
    assert provenance.metric_binding.metric_id == "collision_rate"


def test_golden_fixture_alias_source_resolves_to_actual_aggregate_contributors() -> None:
    """A canonical ID resolves through the metric-layer alias on the real aggregate fixture."""
    fixture = Path(__file__).parents[2] / "tests/fixtures/benchmark/golden/aggregate_episodes.jsonl"
    records = read_jsonl(fixture)

    aggregate = compute_aggregates(records)
    provenance = resolve_aggregate_cell_provenance(records, _identity(group="orca"))

    assert aggregate["orca"]["collisions"]["mean"] == pytest.approx(0.5)
    assert provenance.value == pytest.approx(0.5)
    assert provenance.contributor_episode_ids == (
        "golden-orca-corridor-17",
        "golden-orca-crossing-23",
    )
    assert provenance.metric_binding.source_field_paths == (
        "metrics.collision_rate",
        "metrics.collisions",
        "outcome.collision_event",
    )


def test_derived_metric_source_resolves_without_serialized_aggregate_changes() -> None:
    """A metric-layer outcome derivation supplies a provenance-only canonical cell."""
    records = [
        {
            **_episode("derived-safe", collision_rate=None),
            "outcome": {"collision_event": False},
        },
        {
            **_episode("derived-collision", collision_rate=None),
            "outcome": {"collision_event": True},
        },
    ]

    aggregate = compute_aggregates(records)
    provenance = resolve_aggregate_cell_provenance(records, _identity())

    assert aggregate["planner_a"] == {}
    assert provenance.value == pytest.approx(0.5)
    assert provenance.contributor_episode_ids == ("derived-collision", "derived-safe")


def test_aggregate_cell_provenance_annotations_are_runtime_resolvable() -> None:
    """The public provenance dataclass can be inspected by runtime consumers."""
    hints = typing.get_type_hints(AggregateCellProvenance)

    assert hints["metric_binding"] is MetricSourceBinding


def test_eligibility_and_grouping_changes_flow_through_canonical_aggregation() -> None:
    """Eligibility and group inputs alter membership through the production aggregation path."""
    records = [_episode("ep-1"), _episode("ep-2", evidence_eligible=False)]
    eligible_only = resolve_aggregate_cell_provenance(records, _identity())

    records[1]["algorithm_metadata"] = {"foresight_prediction": {"evidence_eligible": True}}
    both_eligible = resolve_aggregate_cell_provenance(records, _identity())
    regrouped = resolve_aggregate_cell_provenance(
        records,
        _identity(group="crossing"),
        group_by="scenario_id",
    )

    assert eligible_only.contributor_episode_ids == ("ep-1",)
    assert both_eligible.contributor_episode_ids == ("ep-1", "ep-2")
    assert regrouped.contributor_episode_ids == both_eligible.contributor_episode_ids


def test_diagnostic_observation_tracks_remain_separate_group_identities() -> None:
    """Diagnostic cross-track mode exposes the exact existing track-qualified group labels."""
    records = [
        _episode("state-ep", benchmark_track="state"),
        _episode("image-ep", benchmark_track="image"),
    ]

    provenance = resolve_aggregate_cell_provenance(
        records,
        _identity(group="image :: planner_a"),
        observation_track_mode="diagnostic-cross-track",
    )

    assert provenance.contributor_episode_ids == ("image-ep",)


@pytest.mark.parametrize(
    ("records", "reason"),
    [
        ([_episode("duplicate"), _episode("duplicate")], "duplicate_episode_identity"),
        ([_episode(None)], "missing_episode_identity"),
        ([_episode("ep", collision_rate=None)], "missing_contributor_row"),
    ],
)
def test_aggregate_provenance_rejects_ambiguous_or_missing_contributor_identity(
    records: list[dict[str, object]],
    reason: str,
) -> None:
    """Duplicate, missing, and absent contributor identities fail closed with stable reasons."""
    with pytest.raises(AggregateProvenanceError) as exc_info:
        resolve_aggregate_cell_provenance(records, _identity())

    assert exc_info.value.reason == reason


@pytest.mark.parametrize("metric_id", ["Collision rate", "collision rate", "totally_unknown"])
def test_metric_selection_rejects_display_names_and_unknown_ids(metric_id: str) -> None:
    """Only exact stable metric IDs select aggregate cells."""
    with pytest.raises(AggregateProvenanceError) as exc_info:
        resolve_aggregate_cell_provenance([_episode("ep")], _identity(metric=metric_id))

    assert exc_info.value.reason == "unknown_metric_id"


def test_metric_binding_returns_existing_metadata_and_explicit_missing_unit() -> None:
    """The resolver reports owner/source/direction and does not invent unavailable units."""
    binding = resolve_metric_source_binding("collision_rate")

    assert binding.source_field_paths == (
        "metrics.collision_rate",
        "metrics.collisions",
        "outcome.collision_event",
    )
    assert binding.owner == METRIC_BINDING_OWNER
    assert binding.status == "available"
    assert binding.direction == "lower_is_better"
    assert binding.direction_status == "available"
    assert binding.unit is None
    assert binding.unit_status == "unavailable"


def test_metric_binding_rejects_stale_and_conflicting_registry_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expected-binding drift and a key/definition conflict both fail closed."""
    current = resolve_metric_source_binding("collision_rate")
    stale = replace(current, source_field_paths=("metrics.old_collision_rate",))
    with pytest.raises(MetricBindingError) as drift_error:
        resolve_metric_source_binding("collision_rate", expected=stale)
    assert drift_error.value.reason == "metric_binding_drift"

    monkeypatch.setitem(
        CANONICAL_METRICS,
        "collision_rate",
        replace(CANONICAL_METRICS["collision_rate"], name="collision_rate_display"),
    )
    with pytest.raises(MetricBindingError) as conflict_error:
        resolve_metric_source_binding("collision_rate")
    assert conflict_error.value.reason == "conflicting_metric_identity"


def test_metric_binding_rejects_conflicting_source_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicate canonical source paths are a conflicting binding, not aliases."""
    definition = CANONICAL_METRICS["collision_rate"]
    monkeypatch.setitem(
        CANONICAL_METRICS,
        "collision_rate",
        replace(definition, source_keys=("metrics.collisions", "metrics.collisions")),
    )

    with pytest.raises(MetricBindingError) as exc_info:
        resolve_metric_source_binding("collision_rate")

    assert exc_info.value.reason == "conflicting_source_binding"


def test_metric_binding_keeps_unsupported_source_channel_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered metric without a raw source channel is unsupported rather than guessed."""
    monkeypatch.setitem(
        CANONICAL_METRICS,
        "derived_only",
        MetricDefinition(
            name="derived_only",
            layer="operational",
            source_keys=(),
            reduction="mean",
            higher_is_better=None,
            description="Test-only derived metric without a canonical episode source.",
            source_kind="derived",
        ),
    )

    binding = resolve_metric_source_binding("derived_only")

    assert binding.status == "unsupported"
    assert binding.source_field_paths == ()
    assert binding.direction is None
    assert binding.direction_status == "unavailable"
