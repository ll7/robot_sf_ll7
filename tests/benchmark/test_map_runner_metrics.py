"""Direct contract tests for ``robot_sf.benchmark.map_runner_metrics``.

These tests lock the metric-contract boundaries used by map-based benchmark runs:
pedestrian-impact control normalization, exact/fallback collision extraction, the
aggregate collision summary's provenance and denominator semantics, and the
flag-floor behavior that preserves exact environment collision events.

All inputs are small in-memory record dictionaries; no benchmark artifacts are read.
"""

from __future__ import annotations

import math

import pytest

from robot_sf.benchmark.map_runner_metrics import (
    _episode_collision_value,
    _exact_collision_event,
    _finite_float,
    collision_metric_value,
    floor_collision_metrics_from_flags,
    normalize_pedestrian_impact_controls,
    summarize_collision_metrics,
)

# ---------------------------------------------------------------------------
# normalize_pedestrian_impact_controls
# ---------------------------------------------------------------------------


class TestNormalizePedestrianImpactControls:
    """Lock pedestrian-impact control coercion and opt-in validation."""

    def test_returns_coerced_pair_when_opt_in_disabled(self) -> None:
        """Without the experimental opt-in, controls are only coerced, never validated."""
        radius, window = normalize_pedestrian_impact_controls(
            experimental_ped_impact=False,
            ped_impact_radius_m="1.5",
            ped_impact_window_steps=2.5,
        )

        assert radius == pytest.approx(1.5)
        # ``int(float(2.5))`` truncates toward zero without raising.
        assert window == 2

    def test_passes_through_non_finite_when_opt_in_disabled(self) -> None:
        """Non-finite/malformed controls flow through unvalidated when the opt-in is off."""
        radius, window = normalize_pedestrian_impact_controls(
            experimental_ped_impact=False,
            ped_impact_radius_m=float("nan"),
            ped_impact_window_steps=3,
        )

        assert math.isnan(radius)
        assert window == 3

    def test_returns_validated_pair_when_opt_in_enabled(self) -> None:
        """A finite positive radius and integer window pass opt-in validation unchanged."""
        radius, window = normalize_pedestrian_impact_controls(
            experimental_ped_impact=True,
            ped_impact_radius_m=1.5,
            ped_impact_window_steps=4,
        )

        assert radius == pytest.approx(1.5)
        assert window == 4

    def test_accepts_float_valued_integer_window_when_opt_in_enabled(self) -> None:
        """A float that is exactly integer-valued satisfies the opt-in window contract."""
        radius, window = normalize_pedestrian_impact_controls(
            experimental_ped_impact=True,
            ped_impact_radius_m=1.0,
            ped_impact_window_steps=3.0,
        )

        assert radius == pytest.approx(1.0)
        assert window == 3

    @pytest.mark.parametrize("bad_radius", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_radius_when_opt_in_enabled(self, bad_radius: float) -> None:
        """Non-finite radii must fail fast under the experimental opt-in."""
        with pytest.raises(ValueError, match="ped_impact_radius_m must be a finite value > 0."):
            normalize_pedestrian_impact_controls(
                experimental_ped_impact=True,
                ped_impact_radius_m=bad_radius,
                ped_impact_window_steps=4,
            )

    @pytest.mark.parametrize("bad_radius", [0.0, -0.25])
    def test_rejects_non_positive_radius_when_opt_in_enabled(self, bad_radius: float) -> None:
        """Zero or negative radii must fail fast under the experimental opt-in."""
        with pytest.raises(ValueError, match="ped_impact_radius_m must be a finite value > 0."):
            normalize_pedestrian_impact_controls(
                experimental_ped_impact=True,
                ped_impact_radius_m=bad_radius,
                ped_impact_window_steps=4,
            )

    @pytest.mark.parametrize(
        ("bad_window", "expected"),
        [
            (float("nan"), ValueError),
            (float("inf"), OverflowError),
        ],
    )
    def test_rejects_non_finite_window_when_opt_in_enabled(
        self, bad_window: float, expected: type[BaseException]
    ) -> None:
        """Non-finite windows fail fast at the unconditional int-coercion step.

        The window control is coerced via ``int(float(...))`` before the opt-in
        validation block, so a non-finite window raises from that conversion
        (``ValueError`` for NaN, ``OverflowError`` for infinity) rather than from
        the friendly validation message. This is a fail-closed contract: a
        non-finite window can never produce a usable step count.
        """
        with pytest.raises(expected):
            normalize_pedestrian_impact_controls(
                experimental_ped_impact=True,
                ped_impact_radius_m=1.0,
                ped_impact_window_steps=bad_window,
            )

    @pytest.mark.parametrize(
        ("bad_window", "expected"),
        [
            (float("nan"), ValueError),
            (float("inf"), OverflowError),
        ],
    )
    def test_rejects_non_finite_window_even_when_opt_in_disabled(
        self, bad_window: float, expected: type[BaseException]
    ) -> None:
        """Non-finite windows are rejected without the opt-in, unlike non-finite radii.

        The unconditional ``int(float(...))`` coercion runs before the opt-in gate,
        so a non-finite window fails fast regardless of the experimental flag. A
        non-finite radius, by contrast, passes through unvalidated when the opt-in
        is off (see ``test_passes_through_non_finite_when_opt_in_disabled``).
        """
        with pytest.raises(expected):
            normalize_pedestrian_impact_controls(
                experimental_ped_impact=False,
                ped_impact_radius_m=1.0,
                ped_impact_window_steps=bad_window,
            )

    def test_rejects_non_integer_window_when_opt_in_enabled(self) -> None:
        """Fractional windows must fail fast under the experimental opt-in."""
        with pytest.raises(ValueError, match="ped_impact_window_steps must be an integer >= 1."):
            normalize_pedestrian_impact_controls(
                experimental_ped_impact=True,
                ped_impact_radius_m=1.0,
                ped_impact_window_steps=2.5,
            )

    @pytest.mark.parametrize("bad_window", [0, -1, 0.0])
    def test_rejects_window_below_one_when_opt_in_enabled(self, bad_window: float) -> None:
        """Windows below one must fail fast under the experimental opt-in."""
        with pytest.raises(ValueError, match="ped_impact_window_steps must be an integer >= 1."):
            normalize_pedestrian_impact_controls(
                experimental_ped_impact=True,
                ped_impact_radius_m=1.0,
                ped_impact_window_steps=bad_window,
            )

    def test_validates_radius_before_window(self) -> None:
        """When both controls are invalid, the radius check fires first."""
        with pytest.raises(ValueError, match="ped_impact_radius_m"):
            normalize_pedestrian_impact_controls(
                experimental_ped_impact=True,
                ped_impact_radius_m=0.0,
                ped_impact_window_steps=0,
            )


# ---------------------------------------------------------------------------
# collision_metric_value
# ---------------------------------------------------------------------------


class TestCollisionMetricValue:
    """Lock the sampled collision-metric coercion and fail-closed-to-zero behavior."""

    def test_returns_stored_finite_value(self) -> None:
        """A stored finite numeric value is returned as a float."""
        assert collision_metric_value({"collisions": 2.0}, "collisions") == pytest.approx(2.0)

    def test_treats_missing_key_as_zero(self) -> None:
        """A missing metric key fails closed to zero instead of raising."""
        assert collision_metric_value({}, "collisions") == 0.0

    def test_treats_null_value_as_zero(self) -> None:
        """An explicit null metric value fails closed to zero."""
        assert collision_metric_value({"collisions": None}, "collisions") == 0.0

    def test_coerces_numeric_string(self) -> None:
        """Numeric strings are coerced to their finite float value."""
        assert collision_metric_value({"collisions": "3.5"}, "collisions") == pytest.approx(3.5)

    def test_treats_non_numeric_string_as_zero(self) -> None:
        """Non-numeric strings fail closed to zero rather than propagating."""
        assert collision_metric_value({"collisions": "abc"}, "collisions") == 0.0

    def test_treats_uncoercible_type_as_zero(self) -> None:
        """Uncoercible container types fail closed to zero rather than raising."""
        assert collision_metric_value({"collisions": [1, 2]}, "collisions") == 0.0
        assert collision_metric_value({"collisions": {"nested": 1}}, "collisions") == 0.0

    @pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf"), "inf"])
    def test_treats_non_finite_as_zero(self, non_finite: object) -> None:
        """Non-finite values (including the ``inf`` string) fail closed to zero."""
        assert collision_metric_value({"collisions": non_finite}, "collisions") == 0.0


# ---------------------------------------------------------------------------
# _finite_float (metric fallback / unavailability primitive)
# ---------------------------------------------------------------------------


class TestFiniteFloat:
    """Lock the finite-float coercion primitive that drives metric fallback."""

    def test_returns_coerced_finite_value(self) -> None:
        """Finite numeric inputs coerce to a finite float."""
        assert _finite_float("2.5") == pytest.approx(2.5)
        assert _finite_float(3) == pytest.approx(3.0)

    @pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
    def test_returns_none_for_non_finite(self, non_finite: float) -> None:
        """Non-finite values map to ``None`` so callers can treat them as unavailable."""
        assert _finite_float(non_finite) is None

    @pytest.mark.parametrize("malformed", [None, "abc", [1.0], {"x": 1.0}])
    def test_returns_none_for_uncoercible(self, malformed: object) -> None:
        """Uncoercible inputs map to ``None`` instead of raising."""
        assert _finite_float(malformed) is None


# ---------------------------------------------------------------------------
# _exact_collision_event (exact flag extraction)
# ---------------------------------------------------------------------------


class TestExactCollisionEvent:
    """Lock extraction of the exact environment collision flag from a record."""

    @pytest.mark.parametrize("flag,expected", [(True, True), (False, False), (1, True), (0, False)])
    def test_returns_bool_for_present_flag(self, flag: object, expected: bool) -> None:
        """A present, non-null flag is coerced to its boolean value."""
        assert _exact_collision_event({"outcome": {"collision_event": flag}}) is expected

    def test_returns_none_for_missing_outcome(self) -> None:
        """Records without an outcome block report the exact flag as unavailable."""
        assert _exact_collision_event({"metrics": {}}) is None

    def test_returns_none_for_non_dict_outcome(self) -> None:
        """A non-dict outcome block reports the exact flag as unavailable."""
        assert _exact_collision_event({"outcome": "collision"}) is None

    def test_returns_none_when_flag_key_absent(self) -> None:
        """An outcome block without the collision-event key reports unavailable."""
        assert _exact_collision_event({"outcome": {"success": True}}) is None

    def test_returns_none_for_null_flag(self) -> None:
        """A null collision-event flag is treated as unavailable, not as falsy."""
        assert _exact_collision_event({"outcome": {"collision_event": None}}) is None


# ---------------------------------------------------------------------------
# _episode_collision_value (per-episode extraction with exact-event floor)
# ---------------------------------------------------------------------------


class TestEpisodeCollisionValue:
    """Lock per-episode collision extraction, key priority, and the exact-event floor."""

    def test_prefers_collisions_key(self) -> None:
        """The ``collisions`` metric key has the highest extraction priority."""
        value, source = _episode_collision_value(
            {
                "metrics": {
                    "collisions": 1.0,
                    "total_collision_count": 5.0,
                    "collision_count": 9.0,
                }
            }
        )

        assert value == pytest.approx(1.0)
        assert source == "episode.metrics.collisions"

    def test_falls_back_to_total_collision_count(self) -> None:
        """When ``collisions`` is absent, ``total_collision_count`` is selected."""
        value, source = _episode_collision_value(
            {"metrics": {"total_collision_count": 4.0, "collision_count": 9.0}}
        )

        assert value == pytest.approx(4.0)
        assert source == "episode.metrics.total_collision_count"

    def test_falls_back_to_collision_count(self) -> None:
        """When no higher-priority key is present, ``collision_count`` is selected."""
        value, source = _episode_collision_value({"metrics": {"collision_count": 7.0}})

        assert value == pytest.approx(7.0)
        assert source == "episode.metrics.collision_count"

    def test_skips_present_but_unavailable_primary_key(self) -> None:
        """A present-but-null primary key falls through to the next finite key."""
        value, source = _episode_collision_value(
            {"metrics": {"collisions": None, "total_collision_count": 2.0}}
        )

        assert value == pytest.approx(2.0)
        assert source == "episode.metrics.total_collision_count"

    def test_skips_non_finite_primary_key(self) -> None:
        """A non-finite primary key falls through to the next finite key."""
        value, source = _episode_collision_value(
            {
                "metrics": {
                    "collisions": float("nan"),
                    "total_collision_count": 2.0,
                }
            }
        )

        assert value == pytest.approx(2.0)
        assert source == "episode.metrics.total_collision_count"

    def test_floors_zero_sampled_metric_when_exact_event_fired(self) -> None:
        """A zero sampled metric is floored to one when the exact event fired."""
        value, source = _episode_collision_value(
            {"metrics": {"collisions": 0.0}, "outcome": {"collision_event": True}}
        )

        assert value == pytest.approx(1.0)
        assert source == "episode.outcome.collision_event"

    def test_preserves_larger_sampled_metric_when_exact_event_fired(self) -> None:
        """A sampled count above the exact-event floor is never reduced."""
        value, source = _episode_collision_value(
            {"metrics": {"collisions": 3.0}, "outcome": {"collision_event": True}}
        )

        assert value == pytest.approx(3.0)
        assert source == "episode.metrics.collisions"

    def test_keeps_zero_sampled_metric_when_no_exact_event(self) -> None:
        """A zero sampled metric stays zero when the exact flag is ``False``."""
        value, source = _episode_collision_value(
            {"metrics": {"collisions": 0.0}, "outcome": {"collision_event": False}}
        )

        assert value == pytest.approx(0.0)
        assert source == "episode.metrics.collisions"

    def test_uses_exact_flag_when_no_metric_keys_present(self) -> None:
        """Without sampled metric keys, the exact flag drives value and provenance."""
        value_true, source_true = _episode_collision_value({"outcome": {"collision_event": True}})
        value_false, source_false = _episode_collision_value(
            {"outcome": {"collision_event": False}}
        )

        assert value_true == pytest.approx(1.0)
        assert source_true == "episode.outcome.collision_event"
        assert value_false == pytest.approx(0.0)
        assert source_false == "episode.outcome.collision_event"

    def test_returns_none_when_no_metrics_and_no_event(self) -> None:
        """Records with neither metric keys nor an exact flag report no collision value."""
        assert _episode_collision_value({"metrics": {}}) == (None, None)
        assert _episode_collision_value({}) == (None, None)

    def test_returns_none_when_metric_keys_all_unavailable_and_no_event(self) -> None:
        """Present-but-unavailable metric keys without an exact flag report no value."""
        assert _episode_collision_value(
            {"metrics": {"collisions": None, "total_collision_count": float("nan")}}
        ) == (None, None)


# ---------------------------------------------------------------------------
# summarize_collision_metrics (aggregate provenance and denominator semantics)
# ---------------------------------------------------------------------------


class TestSummarizeCollisionMetrics:
    """Lock aggregate collision counts, rates, provenance, and denominator semantics."""

    def test_empty_records_return_not_available_with_zero_denominator(self) -> None:
        """An empty record list reports collision as unavailable with a zero denominator."""
        summary = summarize_collision_metrics([])

        assert summary == {
            "collision": "not_available",
            "collision_count": "not_available",
            "collision_rate": "not_available",
            "collision_status": {
                "status": "not_available",
                "reason": "no successful episode records were available for aggregation",
                "denominator": 0,
                "source": None,
            },
        }

    def test_all_available_summary_counts_rate_and_provenance(self) -> None:
        """A fully available record set reports counts, rate, and a single source."""
        summary = summarize_collision_metrics(
            [
                {"metrics": {"collisions": 1.0}},
                {"metrics": {"collisions": 0.0}},
            ]
        )

        assert summary["collision"] == pytest.approx(1.0)
        assert summary["collision_count"] == pytest.approx(1.0)
        # Rate denominator is the available-records count, not the raw record count.
        assert summary["collision_rate"] == pytest.approx(0.5)
        assert summary["collision_status"] == {
            "status": "available",
            "reason": None,
            "denominator": 2,
            "source": "episode.metrics.collisions",
        }

    def test_rate_denominator_uses_available_records_not_record_count(self) -> None:
        """Skipped records lower the rate denominator but not the aggregate record count."""
        summary = summarize_collision_metrics(
            [
                {"metrics": {"collisions": 1.0}},
                # No metric keys and no exact flag: skipped, not counted as available.
                {"metrics": {}},
            ]
        )

        assert summary["collision_count"] == pytest.approx(1.0)
        # One collided episode over one available record -> rate 1.0, not 0.5.
        assert summary["collision_rate"] == pytest.approx(1.0)
        assert summary["collision_status"] == {
            "status": "partial",
            "reason": "some successful records lacked collision metrics",
            "denominator": 2,
            "source": "episode.metrics.collisions",
        }

    def test_count_sums_metric_values_not_collided_episodes(self) -> None:
        """The aggregate count sums per-episode values, which may exceed the episode count."""
        summary = summarize_collision_metrics(
            [
                {"metrics": {"collisions": 3.0}},
                {"metrics": {"collisions": 2.0}},
            ]
        )

        assert summary["collision_count"] == pytest.approx(5.0)
        assert summary["collision_rate"] == pytest.approx(1.0)
        assert summary["collision_status"]["denominator"] == 2
        assert summary["collision_status"]["status"] == "available"

    def test_all_unavailable_returns_not_available_with_positive_denominator(self) -> None:
        """Records that emit no collision value report unavailable with a distinct reason."""
        summary = summarize_collision_metrics(
            [
                {"metrics": {}},
                {"outcome": {}},
            ]
        )

        assert summary["collision"] == "not_available"
        assert summary["collision_count"] == "not_available"
        assert summary["collision_rate"] == "not_available"
        assert summary["collision_status"] == {
            "status": "not_available",
            "reason": "successful episode records did not emit collision metrics",
            "denominator": 2,
            "source": None,
        }

    def test_source_is_sorted_and_comma_joined_across_mixed_records(self) -> None:
        """Provenance sources are de-duplicated and emitted in sorted, comma-joined order."""
        summary = summarize_collision_metrics(
            [
                {"metrics": {"collisions": 1.0}, "outcome": {"collision_event": True}},
                # No metric keys: provenance falls back to the exact-event source.
                {"outcome": {"collision_event": False}},
            ]
        )

        assert summary["collision_count"] == pytest.approx(1.0)
        assert summary["collision_rate"] == pytest.approx(0.5)
        assert summary["collision_status"]["source"] == (
            "episode.metrics.collisions,episode.outcome.collision_event"
        )
        assert summary["collision_status"]["status"] == "available"

    def test_exact_event_floor_propagates_into_aggregate_count(self) -> None:
        """An exact collision event floors a zero sampled metric within the aggregate."""
        summary = summarize_collision_metrics(
            [
                {"metrics": {"collisions": 0.0}, "outcome": {"collision_event": True}},
                {"metrics": {"collisions": 0.0}, "outcome": {"collision_event": False}},
            ]
        )

        assert summary["collision_count"] == pytest.approx(1.0)
        assert summary["collision_rate"] == pytest.approx(0.5)
        assert summary["collision_status"]["status"] == "available"
        assert "episode.outcome.collision_event" in summary["collision_status"]["source"]


# ---------------------------------------------------------------------------
# floor_collision_metrics_from_flags (exact-flag floor without reduction)
# ---------------------------------------------------------------------------


def _empty_collision_metrics() -> dict[str, float]:
    """Return a fresh zeroed collision-metrics dict for floor tests."""
    return {
        "ped_collision_count": 0.0,
        "obstacle_collision_count": 0.0,
        "agent_collision_count": 0.0,
        "total_collision_count": 0.0,
        "collisions": 0.0,
        "wall_collisions": 0.0,
    }


class TestFloorCollisionMetricsFromFlags:
    """Lock exact-flag flooring that never reduces an already higher metric."""

    def test_no_flags_leaves_metrics_untouched(self) -> None:
        """With no collision flags, the metrics dict is left unchanged."""
        metrics = _empty_collision_metrics()
        original = dict(metrics)

        floor_collision_metrics_from_flags(
            metrics,
            collision_seen=False,
            ped_collision_seen=False,
            obstacle_collision_seen=False,
            robot_collision_seen=False,
        )

        assert metrics == original

    def test_floors_each_typed_key_to_one_when_seen_and_zero(self) -> None:
        """Each typed collision flag floors its zero count to one."""
        metrics = _empty_collision_metrics()

        floor_collision_metrics_from_flags(
            metrics,
            collision_seen=False,
            ped_collision_seen=True,
            obstacle_collision_seen=True,
            robot_collision_seen=True,
        )

        assert metrics["ped_collision_count"] == pytest.approx(1.0)
        assert metrics["obstacle_collision_count"] == pytest.approx(1.0)
        assert metrics["agent_collision_count"] == pytest.approx(1.0)
        # typed total (3.0) beats the zero sampled totals.
        assert metrics["total_collision_count"] == pytest.approx(3.0)
        assert metrics["collisions"] == pytest.approx(3.0)
        # Obstacle flag also floors wall collisions from the obstacle count.
        assert metrics["wall_collisions"] == pytest.approx(1.0)

    def test_does_not_reduce_already_higher_typed_metric(self) -> None:
        """A typed collision count above the floor is preserved."""
        metrics = _empty_collision_metrics()
        metrics["ped_collision_count"] = 2.0

        floor_collision_metrics_from_flags(
            metrics,
            collision_seen=True,
            ped_collision_seen=True,
            obstacle_collision_seen=False,
            robot_collision_seen=False,
        )

        assert metrics["ped_collision_count"] == pytest.approx(2.0)
        assert metrics["total_collision_count"] == pytest.approx(2.0)
        assert metrics["collisions"] == pytest.approx(2.0)

    def test_aggregate_never_reduces_higher_sampled_total(self) -> None:
        """Typed floors must not lower an already higher sampled aggregate."""
        metrics = _empty_collision_metrics()
        metrics["collisions"] = 5.0
        metrics["total_collision_count"] = 5.0

        floor_collision_metrics_from_flags(
            metrics,
            collision_seen=True,
            ped_collision_seen=True,
            obstacle_collision_seen=False,
            robot_collision_seen=False,
        )

        assert metrics["ped_collision_count"] == pytest.approx(1.0)
        assert metrics["collisions"] == pytest.approx(5.0)
        assert metrics["total_collision_count"] == pytest.approx(5.0)

    def test_obstacle_flag_floors_wall_collisions_from_obstacle_count(self) -> None:
        """The obstacle flag floors wall collisions to the obstacle count when both are zero."""
        metrics = _empty_collision_metrics()

        floor_collision_metrics_from_flags(
            metrics,
            collision_seen=False,
            ped_collision_seen=False,
            obstacle_collision_seen=True,
            robot_collision_seen=False,
        )

        assert metrics["obstacle_collision_count"] == pytest.approx(1.0)
        assert metrics["wall_collisions"] == pytest.approx(1.0)
        assert metrics["total_collision_count"] == pytest.approx(1.0)
        assert metrics["collisions"] == pytest.approx(1.0)

    def test_obstacle_flag_does_not_overwrite_higher_wall_collisions(self) -> None:
        """A higher sampled wall-collision count survives the obstacle floor."""
        metrics = _empty_collision_metrics()
        metrics["wall_collisions"] = 3.0

        floor_collision_metrics_from_flags(
            metrics,
            collision_seen=False,
            ped_collision_seen=False,
            obstacle_collision_seen=True,
            robot_collision_seen=False,
        )

        assert metrics["obstacle_collision_count"] == pytest.approx(1.0)
        assert metrics["wall_collisions"] == pytest.approx(3.0)
        assert metrics["total_collision_count"] == pytest.approx(3.0)
        assert metrics["collisions"] == pytest.approx(3.0)

    def test_untyped_collision_seen_floors_total_and_collisions(self) -> None:
        """A bare collision flag floors only the aggregate totals, leaving typed keys alone."""
        metrics = _empty_collision_metrics()

        floor_collision_metrics_from_flags(
            metrics,
            collision_seen=True,
            ped_collision_seen=False,
            obstacle_collision_seen=False,
            robot_collision_seen=False,
        )

        assert metrics["total_collision_count"] == pytest.approx(1.0)
        assert metrics["collisions"] == pytest.approx(1.0)
        assert metrics["ped_collision_count"] == pytest.approx(0.0)
        assert metrics["obstacle_collision_count"] == pytest.approx(0.0)
        assert metrics["agent_collision_count"] == pytest.approx(0.0)
        # The untyped branch does not touch wall collisions.
        assert metrics["wall_collisions"] == pytest.approx(0.0)

    def test_typed_flags_take_precedence_over_untyped_collision_seen(self) -> None:
        """Typed flags drive the aggregate even when the bare collision flag is also set."""
        metrics = _empty_collision_metrics()

        floor_collision_metrics_from_flags(
            metrics,
            collision_seen=True,
            ped_collision_seen=True,
            obstacle_collision_seen=True,
            robot_collision_seen=False,
        )

        # Two typed collisions -> aggregate 2.0, not the untyped floor of 1.0.
        assert metrics["total_collision_count"] == pytest.approx(2.0)
        assert metrics["collisions"] == pytest.approx(2.0)
        assert metrics["ped_collision_count"] == pytest.approx(1.0)
        assert metrics["obstacle_collision_count"] == pytest.approx(1.0)

    def test_missing_keys_are_created_by_the_floor(self) -> None:
        """An empty metrics dict receives floored typed and aggregate keys."""
        metrics: dict[str, float] = {}

        floor_collision_metrics_from_flags(
            metrics,
            collision_seen=True,
            ped_collision_seen=True,
            obstacle_collision_seen=False,
            robot_collision_seen=False,
        )

        assert metrics["ped_collision_count"] == pytest.approx(1.0)
        assert metrics["total_collision_count"] == pytest.approx(1.0)
        assert metrics["collisions"] == pytest.approx(1.0)
