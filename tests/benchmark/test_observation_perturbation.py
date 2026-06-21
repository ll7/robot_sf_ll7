"""Tests for bounded observation-perturbation helper."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark.observation_perturbation import (
    EVIDENCE_IDEAL,
    EVIDENCE_PERCEPTION_LIMITED,
    NOISE_PROFILE_DELAYED_OBSERVATION,
    NOISE_PROFILE_FALSE_POSITIVE,
    NOISE_PROFILE_FIXTURE_VISIBILITY,
    NOISE_PROFILE_GAUSSIAN,
    NOISE_PROFILE_MISSED_DETECTION,
    NOISE_PROFILE_NONE,
    NOISE_PROFILE_OCCLUSION_MASK,
    ObservationPerturbationSpec,
    ObservationPerturbationState,
    perturb_ground_truth,
)


def _simple_actors() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return a 3-actor fixture."""
    positions = np.array([[1.0, 2.0], [5.0, 5.0], [10.0, 0.0]])
    velocities = np.array([[0.1, 0.0], [-0.2, 0.3], [0.0, 0.5]])
    ids = ["ped_A", "ped_B", "ped_C"]
    return positions, velocities, ids


class TestNoopSpec:
    """No-noise config must reproduce ground truth deterministically."""

    def test_noop_returns_ground_truth_identically(self) -> None:
        """Zero-noise spec should output ground truth as observed state."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec()
        assert spec.is_noop

        result = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)

        np.testing.assert_array_equal(result["observed"]["positions"], pos)
        np.testing.assert_array_equal(result["observed"]["velocities"], vel)
        assert result["observed"]["ids"] == ids
        assert result["missing_ids"] == []
        assert result["metadata"]["evidence_class"] == EVIDENCE_IDEAL
        assert result["metadata"]["noise_profile"] == NOISE_PROFILE_NONE
        assert result["metadata"]["missed_actor_count"] == 0
        assert result["metadata"]["occluded_actor_count"] == 0

    def test_noop_ground_truth_dict_matches_input(self) -> None:
        """Ground-truth payload should match input arrays."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec()

        result = perturb_ground_truth(pos, vel, ids, spec=spec)

        np.testing.assert_array_equal(result["ground_truth"]["positions"], pos)
        np.testing.assert_array_equal(result["ground_truth"]["velocities"], vel)
        assert result["ground_truth"]["ids"] == ids

    def test_noop_is_deterministic_across_calls(self) -> None:
        """Repeated calls with same input should produce identical output."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec()

        first = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)
        second = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)

        np.testing.assert_array_equal(
            first["observed"]["positions"], second["observed"]["positions"]
        )

    def test_noop_metadata_shape(self) -> None:
        """Metadata should contain the canonical set of keys."""
        pos, vel, ids = _simple_actors()
        result = perturb_ground_truth(pos, vel, ids, spec=ObservationPerturbationSpec())
        meta = result["metadata"]
        assert set(meta.keys()) == {
            "noise_profile",
            "evidence_class",
            "position_noise_std_m",
            "position_noise_bound_m",
            "missed_detection_probability",
            "missed_actor_count",
            "occluded_actor_count",
            "visibility_hidden_actor_count",
            "false_positive_actor_count",
            "delay_steps",
            "step",
            "actor_count",
            "observed_actor_count",
        }


class TestBoundedGaussianNoise:
    """Bounded Gaussian position noise must respect std and bound."""

    def test_noise_perturbs_positions(self) -> None:
        """Non-zero noise should change positions but keep velocities/ids."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(
            position_noise_std_m=0.5, position_noise_bound_m=1.0, seed=42
        )

        result = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)

        assert result["metadata"]["evidence_class"] == EVIDENCE_PERCEPTION_LIMITED
        assert result["metadata"]["noise_profile"] == NOISE_PROFILE_GAUSSIAN
        assert result["metadata"]["missed_actor_count"] == 0
        assert result["observed"]["positions"].shape == pos.shape
        assert not np.allclose(result["observed"]["positions"], pos)
        np.testing.assert_array_equal(result["observed"]["velocities"], vel)
        assert result["observed"]["ids"] == ids

    def test_noise_is_bounded(self) -> None:
        """Per-axis displacement should not exceed the configured bound."""
        pos, vel, ids = _simple_actors()
        bound = 0.25
        spec = ObservationPerturbationSpec(
            position_noise_std_m=10.0, position_noise_bound_m=bound, seed=99
        )

        result = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)
        diff = np.abs(result["observed"]["positions"] - pos)
        assert np.all(diff <= bound + 1e-10)

    def test_noise_is_deterministic_for_same_seed(self) -> None:
        """Same seed and step should produce identical noise."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(
            position_noise_std_m=0.3, position_noise_bound_m=0.5, seed=7
        )

        a = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)
        b = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)

        np.testing.assert_array_equal(a["observed"]["positions"], b["observed"]["positions"])

    def test_noise_differs_across_steps(self) -> None:
        """Different step indices should produce different noise."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(
            position_noise_std_m=0.3, position_noise_bound_m=0.5, seed=7
        )

        a = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)
        b = perturb_ground_truth(pos, vel, ids, spec=spec, step=1)

        assert not np.allclose(a["observed"]["positions"], b["observed"]["positions"])


class TestMissedDetections:
    """Missed detection probability should drop whole actors."""

    def test_full_miss_drops_all(self) -> None:
        """Probability=1.0 should remove all actors from observed state."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(missed_detection_probability=1.0, seed=0)

        result = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)

        assert result["metadata"]["missed_actor_count"] == 3
        assert result["observed"]["positions"].shape[0] == 0
        assert result["missing_ids"] == ids
        assert result["metadata"]["observed_actor_count"] == 0
        assert result["metadata"]["noise_profile"] == NOISE_PROFILE_MISSED_DETECTION

    def test_zero_miss_keeps_all(self) -> None:
        """Probability=0.0 should keep all actors."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(missed_detection_probability=0.0, seed=0)

        result = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)

        assert result["metadata"]["missed_actor_count"] == 0
        assert result["observed"]["positions"].shape[0] == 3
        assert result["missing_ids"] == []

    def test_partial_miss(self) -> None:
        """Non-trivial probability should drop at least one but not all actors."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(missed_detection_probability=0.5, seed=42)

        result = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)

        observed_count = result["observed"]["positions"].shape[0]
        assert 0 <= observed_count <= 3
        assert result["metadata"]["missed_actor_count"] == 3 - observed_count
        assert len(result["missing_ids"]) == result["metadata"]["missed_actor_count"]
        assert result["metadata"]["evidence_class"] == EVIDENCE_PERCEPTION_LIMITED

    def test_miss_with_zero_std_has_no_position_noise(self) -> None:
        """Missed-only spec should not add position noise."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(
            position_noise_std_m=0.0, missed_detection_probability=1.0, seed=0
        )

        result = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)
        assert result["observed"]["positions"].shape[0] == 0


class TestOcclusionMask:
    """Occlusion mask should zero out positions/velocities of occluded actors."""

    def test_occlusion_zeros_positions(self) -> None:
        """Occluded actors should have zeroed positions in observed state."""
        pos, vel, ids = _simple_actors()
        mask = np.array([True, False, True])  # A and C occluded
        spec = ObservationPerturbationSpec(occlusion_mask=mask, seed=0)

        result = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)

        assert result["metadata"]["occluded_actor_count"] == 2
        assert result["metadata"]["noise_profile"] == NOISE_PROFILE_OCCLUSION_MASK
        assert result["observed"]["positions"].shape[0] == 3
        np.testing.assert_array_equal(result["observed"]["ids"], ids)
        np.testing.assert_array_equal(result["observed"]["positions"][0], [0.0, 0.0])
        np.testing.assert_array_equal(result["observed"]["positions"][2], [0.0, 0.0])
        np.testing.assert_array_equal(result["observed"]["positions"][1], pos[1])

    def test_occlusion_does_not_drop_actors(self) -> None:
        """All-occluded mask should keep all actors in observed state."""
        pos, vel, ids = _simple_actors()
        mask = np.array([True, True, True])
        spec = ObservationPerturbationSpec(occlusion_mask=mask, seed=0)

        result = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)

        assert result["metadata"]["occluded_actor_count"] == 3
        assert result["observed"]["positions"].shape[0] == 3
        assert result["missing_ids"] == []

    def test_occlusion_with_noise_zeros_only_occluded(self) -> None:
        """Noise applies to non-occluded; occluded stay zeroed."""
        pos, vel, ids = _simple_actors()
        mask = np.array([False, True, False])
        spec = ObservationPerturbationSpec(
            position_noise_std_m=0.5,
            position_noise_bound_m=1.0,
            occlusion_mask=mask,
            seed=42,
        )

        result = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)

        assert result["metadata"]["occluded_actor_count"] == 1
        np.testing.assert_array_equal(result["observed"]["positions"][1], [0.0, 0.0])
        assert not np.allclose(result["observed"]["positions"][0], pos[0])
        assert not np.allclose(result["observed"]["positions"][2], pos[2])

    def test_occlusion_mask_length_mismatch_raises(self) -> None:
        """Occlusion mask with wrong length should raise ValueError."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(occlusion_mask=np.array([True, False]), seed=0)

        with pytest.raises(ValueError, match="occlusion_mask length"):
            perturb_ground_truth(pos, vel, ids, spec=spec)


class TestFalsePositiveActorInjection:
    """False-positive actors should appear only in observed state."""

    def test_false_positive_actor_is_observed_only(self) -> None:
        """Injected actors should not modify the ground-truth payload."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(
            false_positive_positions=np.array([[2.0, 3.0]]),
            false_positive_velocities=np.array([[0.0, 0.0]]),
            false_positive_ids=["fp_close"],
            seed=0,
        )

        result = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)

        assert result["metadata"]["noise_profile"] == NOISE_PROFILE_FALSE_POSITIVE
        assert result["metadata"]["false_positive_actor_count"] == 1
        assert result["metadata"]["actor_count"] == 3
        assert result["metadata"]["observed_actor_count"] == 4
        np.testing.assert_array_equal(result["ground_truth"]["positions"], pos)
        assert result["ground_truth"]["ids"] == ids
        np.testing.assert_array_equal(result["observed"]["positions"][-1], [2.0, 3.0])
        np.testing.assert_array_equal(result["observed"]["velocities"][-1], [0.0, 0.0])
        assert result["observed"]["ids"] == [*ids, "fp_close"]
        assert result["missing_ids"] == []

    def test_false_positive_defaults_zero_velocity_and_ids(self) -> None:
        """Spec defaults should be deterministic and reviewable."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(false_positive_positions=[[2.0, 3.0], [4.0, 5.0]])

        result = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)

        assert result["observed"]["ids"][-2:] == ["false_positive_0", "false_positive_1"]
        np.testing.assert_array_equal(result["observed"]["velocities"][-2:], np.zeros((2, 2)))
        assert result["metadata"]["false_positive_actor_count"] == 2

    def test_false_positive_can_combine_with_missed_detections(self) -> None:
        """False positives should be counted separately from false negatives."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(
            missed_detection_probability=1.0,
            false_positive_positions=[[9.0, 9.0]],
            seed=0,
        )

        result = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)

        assert result["metadata"]["missed_actor_count"] == 3
        assert result["metadata"]["false_positive_actor_count"] == 1
        assert result["metadata"]["observed_actor_count"] == 1
        assert result["observed"]["ids"] == ["false_positive_0"]
        assert result["missing_ids"] == ids


class TestFixtureVisibility:
    """Scenario fixture visibility should hide actors without changing ground truth."""

    def test_visibility_mask_drops_observed_actors_only(self) -> None:
        """Fixture-hidden actors should be absent from the observed planner input."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(visibility_mask=np.array([False, True, False]))

        result = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)

        np.testing.assert_array_equal(result["ground_truth"]["positions"], pos)
        assert result["ground_truth"]["ids"] == ids
        assert result["observed"]["ids"] == ["ped_B"]
        np.testing.assert_array_equal(result["observed"]["positions"], pos[1:2])
        assert result["missing_ids"] == ["ped_A", "ped_C"]
        assert result["metadata"]["visibility_hidden_actor_count"] == 2
        assert result["metadata"]["missed_actor_count"] == 0
        assert result["metadata"]["noise_profile"] == NOISE_PROFILE_FIXTURE_VISIBILITY

    def test_seeded_delay_buffer_preserves_first_observed_boundary(self) -> None:
        """An empty seeded delay buffer should delay first visibility by two steps."""
        pos = np.array([[1.0, 2.0]])
        vel = np.array([[0.1, 0.0]])
        ids = ["ped_A"]
        state = ObservationPerturbationState(delay_steps=2)
        state.reset(
            initial_obs={
                "positions": np.zeros((0, 2), dtype=float),
                "velocities": np.zeros((0, 2), dtype=float),
                "ids": [],
            }
        )
        observed_counts = []
        for step in range(8):
            visibility_mask = np.array([False]) if step < 5 else None
            spec = ObservationPerturbationSpec(
                delay_steps=2,
                visibility_mask=visibility_mask,
            )
            result = perturb_ground_truth(pos + step, vel, ids, spec=spec, step=step, state=state)
            observed_counts.append(result["metadata"]["observed_actor_count"])

        assert observed_counts[:7] == [0, 0, 0, 0, 0, 0, 0]
        assert observed_counts[7] == 1


class TestDelayBehavior:
    """Delay buffer should lag observed state behind ground truth."""

    def test_delay_requires_state_object(self) -> None:
        """Missing state with delay_steps > 0 should raise ValueError."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(delay_steps=2, seed=0)

        with pytest.raises(ValueError, match="ObservationPerturbationState required"):
            perturb_ground_truth(pos, vel, ids, spec=spec)

    def test_delay_1_returns_previous_step(self) -> None:
        """With delay=1, step N should return step N-1 observed state."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(delay_steps=1, seed=0)
        state = ObservationPerturbationState(delay_steps=1)

        # Step 0: buffer fills, returns GT (buffer not yet full)
        r0 = perturb_ground_truth(pos, vel, ids, spec=spec, step=0, state=state)
        np.testing.assert_array_equal(r0["observed"]["positions"], pos)
        assert r0["metadata"]["delay_steps"] == 1

        # Step 1: returns step 0 observed
        pos2 = pos + 1.0
        r1 = perturb_ground_truth(pos2, vel, ids, spec=spec, step=1, state=state)
        np.testing.assert_array_equal(r1["observed"]["positions"], pos)

    def test_delay_2_lags_by_two_steps(self) -> None:
        """With delay=2, step N should return step N-2 observed state."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(delay_steps=2, seed=0)
        state = ObservationPerturbationState(delay_steps=2)

        r0 = perturb_ground_truth(pos, vel, ids, spec=spec, step=0, state=state)
        r1 = perturb_ground_truth(pos + 1, vel, ids, spec=spec, step=1, state=state)
        r2 = perturb_ground_truth(pos + 2, vel, ids, spec=spec, step=2, state=state)

        # Step 0: buffer not full, returns GT
        np.testing.assert_array_equal(r0["observed"]["positions"], pos)
        # Step 1: buffer not full, returns GT
        np.testing.assert_array_equal(r1["observed"]["positions"], pos + 1)
        # Step 2: buffer full, returns step 0
        np.testing.assert_array_equal(r2["observed"]["positions"], pos)

    def test_delay_preserves_metadata_fields(self) -> None:
        """Delayed results should carry valid metadata."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(delay_steps=1, seed=0)
        state = ObservationPerturbationState(delay_steps=1)

        perturb_ground_truth(pos, vel, ids, spec=spec, step=0, state=state)
        r1 = perturb_ground_truth(pos + 1, vel, ids, spec=spec, step=1, state=state)

        assert r1["metadata"]["delay_steps"] == 1
        assert r1["metadata"]["evidence_class"] == EVIDENCE_PERCEPTION_LIMITED
        assert r1["metadata"]["noise_profile"] == NOISE_PROFILE_DELAYED_OBSERVATION
        assert r1["metadata"]["visibility_hidden_actor_count"] == 0
        assert r1["metadata"]["false_positive_actor_count"] == 0

    def test_delay_reports_missing_ids_from_delayed_snapshot(self) -> None:
        """Delayed results should report missing IDs from the returned observed snapshot."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(missed_detection_probability=1.0, delay_steps=1, seed=0)
        state = ObservationPerturbationState(delay_steps=1)

        perturb_ground_truth(pos, vel, ids, spec=spec, step=0, state=state)
        r1 = perturb_ground_truth(pos + 1, vel, ids, spec=spec, step=1, state=state)

        assert r1["observed"]["positions"].shape[0] == 0
        assert r1["missing_ids"] == ids


class TestSpecValidation:
    """Spec constructor should reject invalid parameters."""

    def test_negative_noise_std_raises(self) -> None:
        """Negative position noise std should raise ValueError."""
        with pytest.raises(ValueError, match="position_noise_std_m"):
            ObservationPerturbationSpec(position_noise_std_m=-1.0)

    def test_negative_bound_raises(self) -> None:
        """Negative noise bound should raise ValueError."""
        with pytest.raises(ValueError, match="position_noise_bound_m"):
            ObservationPerturbationSpec(position_noise_bound_m=-0.5)

    def test_bad_miss_prob_raises(self) -> None:
        """Missed detection probability > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="missed_detection_probability"):
            ObservationPerturbationSpec(missed_detection_probability=1.5)

    def test_negative_delay_raises(self) -> None:
        """Negative delay_steps should raise ValueError."""
        with pytest.raises(ValueError, match="delay_steps"):
            ObservationPerturbationSpec(delay_steps=-1)

    def test_bad_false_positive_position_shape_raises(self) -> None:
        """False-positive positions should be an (N, 2) array."""
        with pytest.raises(ValueError, match="false_positive_positions"):
            ObservationPerturbationSpec(false_positive_positions=[1.0, 2.0, 3.0])

    def test_bad_false_positive_velocity_shape_raises(self) -> None:
        """False-positive velocities must match false-positive actor count."""
        with pytest.raises(ValueError, match="false_positive_velocities"):
            ObservationPerturbationSpec(
                false_positive_positions=[[1.0, 2.0]],
                false_positive_velocities=[[0.0, 0.0], [1.0, 0.0]],
            )

    def test_bad_false_positive_id_count_raises(self) -> None:
        """False-positive IDs must match false-positive actor count."""
        with pytest.raises(ValueError, match="false_positive_ids"):
            ObservationPerturbationSpec(
                false_positive_positions=[[1.0, 2.0]],
                false_positive_ids=["fp_0", "fp_1"],
            )


class TestInputValidation:
    """perturb_ground_truth should reject count mismatches."""

    def test_position_velocity_count_mismatch(self) -> None:
        """Mismatched position/velocity arrays should raise ValueError."""
        pos = np.array([[1.0, 2.0], [3.0, 4.0]])
        vel = np.array([[0.1, 0.0]])
        with pytest.raises(ValueError, match="Actor count mismatch"):
            perturb_ground_truth(pos, vel, ["a", "b"], spec=ObservationPerturbationSpec())

    def test_id_count_mismatch(self) -> None:
        """Mismatched ID list length should raise ValueError."""
        pos = np.array([[1.0, 2.0]])
        vel = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError, match="Actor count mismatch"):
            perturb_ground_truth(pos, vel, ["a", "b"], spec=ObservationPerturbationSpec())


class TestTraceShapeSeparation:
    """Ground truth and observed payloads must be distinct objects in trace output."""

    def test_ground_truth_unmodified_by_noise(self) -> None:
        """Ground truth should not be altered by noise application."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(
            position_noise_std_m=1.0, position_noise_bound_m=2.0, seed=42
        )

        result = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)

        np.testing.assert_array_equal(result["ground_truth"]["positions"], pos)
        np.testing.assert_array_equal(result["ground_truth"]["velocities"], vel)
        assert result["ground_truth"]["ids"] == ids
        assert not np.allclose(result["observed"]["positions"], pos)

    def test_ground_truth_unmodified_by_missed_detections(self) -> None:
        """Ground truth should retain all actors even when all are missed."""
        pos, vel, ids = _simple_actors()
        spec = ObservationPerturbationSpec(missed_detection_probability=1.0, seed=0)

        result = perturb_ground_truth(pos, vel, ids, spec=spec, step=0)

        assert result["ground_truth"]["positions"].shape[0] == 3
        assert result["ground_truth"]["ids"] == ids
        assert result["observed"]["positions"].shape[0] == 0

    def test_metadata_evidence_class_always_set(self) -> None:
        """Evidence class should distinguish ideal vs perception-limited results."""
        pos, vel, ids = _simple_actors()
        noop = perturb_ground_truth(pos, vel, ids, spec=ObservationPerturbationSpec())
        noisy = perturb_ground_truth(
            pos,
            vel,
            ids,
            spec=ObservationPerturbationSpec(position_noise_std_m=0.1, seed=0),
        )

        assert noop["metadata"]["evidence_class"] == EVIDENCE_IDEAL
        assert noisy["metadata"]["evidence_class"] == EVIDENCE_PERCEPTION_LIMITED


class TestStateReset:
    """ObservationPerturbationState.reset should clear delay buffer."""

    def test_reset_clears_buffer(self) -> None:
        """Reset should empty the delay buffer so next call uses fresh GT."""
        state = ObservationPerturbationState(delay_steps=2)
        spec = ObservationPerturbationSpec(delay_steps=2, seed=0)
        pos, vel, ids = _simple_actors()

        perturb_ground_truth(pos, vel, ids, spec=spec, step=0, state=state)
        perturb_ground_truth(pos + 1, vel, ids, spec=spec, step=1, state=state)
        perturb_ground_truth(pos + 2, vel, ids, spec=spec, step=2, state=state)

        state.reset()
        r = perturb_ground_truth(pos + 10, vel, ids, spec=spec, step=3, state=state)
        np.testing.assert_array_equal(r["observed"]["positions"], pos + 10)

    def test_reset_with_initial_obs(self) -> None:
        """Reset with initial_obs should seed the buffer."""
        state = ObservationPerturbationState(delay_steps=1)
        spec = ObservationPerturbationSpec(delay_steps=1, seed=0)
        pos, vel, ids = _simple_actors()

        state.reset(
            initial_obs={
                "positions": pos + 99,
                "velocities": vel,
                "ids": ids,
            }
        )
        r = perturb_ground_truth(pos, vel, ids, spec=spec, step=0, state=state)
        # Buffer has initial_obs, delay=1: should return the seeded observation
        np.testing.assert_array_equal(r["observed"]["positions"], pos + 99)

    def test_reset_with_initial_obs_copies_seeded_snapshots(self) -> None:
        """Seeded delay-buffer warmup should not share mutable snapshot references."""
        state = ObservationPerturbationState(delay_steps=2)
        pos, vel, ids = _simple_actors()
        initial_obs = {
            "positions": pos.copy(),
            "velocities": vel.copy(),
            "ids": list(ids),
        }

        state.reset(initial_obs=initial_obs)
        initial_obs["ids"].append("mutated")
        initial_obs["positions"][0, 0] = 99.0

        first, second = list(state._delay_buffer)
        assert first is not second
        assert first["ids"] == ids
        assert second["ids"] == ids
        np.testing.assert_array_equal(first["positions"], pos)
        np.testing.assert_array_equal(second["positions"], pos)
