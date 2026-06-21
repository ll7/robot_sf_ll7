"""Bounded observation-perturbation helper for local policy inputs.

Provides a pure, stateless-per-step function that separates ground-truth
actor state from a noisy/occluded/delayed observed state.  Designed for
diagnostic trace and report payloads that must keep ``ideal_state`` and
``perception_limited`` rows distinct.

This module is a *benchmark diagnostic tool*, not a sensor model or
hardware abstraction.  Evidence class labels are metadata only and
carry no sensor-certification semantics.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

NOISE_PROFILE_NONE = "none"
NOISE_PROFILE_GAUSSIAN = "bounded_gaussian"
NOISE_PROFILE_MISSED_DETECTION = "missed_detection"
NOISE_PROFILE_OCCLUSION_MASK = "occlusion_mask"
NOISE_PROFILE_DELAYED_OBSERVATION = "delayed_observation"
NOISE_PROFILE_FIXTURE_VISIBILITY = "fixture_visibility"
EVIDENCE_IDEAL = "ideal_state"
EVIDENCE_PERCEPTION_LIMITED = "perception_limited"


@dataclass(frozen=True)
class ObservationPerturbationSpec:
    """Configuration for observation perturbation applied to ground-truth actor state."""

    position_noise_std_m: float = 0.0
    position_noise_bound_m: float = 0.0
    missed_detection_probability: float = 0.0
    occlusion_mask: np.ndarray | None = None
    delay_steps: int = 0
    visibility_mask: np.ndarray | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        if self.position_noise_std_m < 0.0:
            raise ValueError("position_noise_std_m must be >= 0")
        if self.position_noise_bound_m < 0.0:
            raise ValueError("position_noise_bound_m must be >= 0")
        if not 0.0 <= self.missed_detection_probability <= 1.0:
            raise ValueError("missed_detection_probability must be between 0 and 1")
        if self.delay_steps < 0:
            raise ValueError("delay_steps must be >= 0")

    @property
    def is_noop(self) -> bool:
        """Return True when the spec produces ground-truth output deterministically."""
        return (
            self.position_noise_std_m <= 0.0
            and self.missed_detection_probability <= 0.0
            and self.occlusion_mask is None
            and self.delay_steps <= 0
            and self.visibility_mask is None
        )

    @property
    def noise_profile(self) -> str:
        """Return a short label for the active noise profile."""
        if self.is_noop:
            return NOISE_PROFILE_NONE
        if self.position_noise_std_m > 0.0:
            return NOISE_PROFILE_GAUSSIAN
        if self.missed_detection_probability > 0.0:
            return NOISE_PROFILE_MISSED_DETECTION
        if self.occlusion_mask is not None:
            return NOISE_PROFILE_OCCLUSION_MASK
        if self.delay_steps > 0:
            return NOISE_PROFILE_DELAYED_OBSERVATION
        if self.visibility_mask is not None:
            return NOISE_PROFILE_FIXTURE_VISIBILITY
        return NOISE_PROFILE_NONE


@dataclass
class ObservationPerturbationState:
    """Mutable state for delay-buffer management across steps.

    The caller owns this object and passes it to ``perturb_ground_truth``.
    For specs with ``delay_steps > 0``, the buffer stores recent observed
    snapshots and the observed state lags behind the latest ground truth.
    """

    delay_steps: int = 0
    _delay_buffer: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=1))

    def __post_init__(self) -> None:
        """Ensure the buffer capacity matches the configured delay."""
        if self.delay_steps < 0:
            raise ValueError("delay_steps must be >= 0")
        capacity = max(1, self.delay_steps + 1)
        if getattr(self._delay_buffer, "maxlen", 1) != capacity:
            existing = list(self._delay_buffer)
            self._delay_buffer = deque(existing, maxlen=capacity)

    def reset(self, initial_obs: dict[str, Any] | None = None) -> None:
        """Reset the delay buffer, optionally seeding warmup with an initial observation."""
        capacity = max(1, self.delay_steps + 1)
        self._delay_buffer = deque(maxlen=capacity)
        if initial_obs is not None:
            for _ in range(max(1, self.delay_steps)):
                self._delay_buffer.append(deepcopy(initial_obs))


def _make_rng(spec: ObservationPerturbationSpec, step: int) -> np.random.Generator | None:
    """Create a deterministic RNG for one perturbation step.

    Returns:
        NumPy generator or None for no-op specs.
    """
    if spec.is_noop:
        return None
    if spec.seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence([spec.seed, step]))


def _apply_bounded_gaussian(
    positions: np.ndarray,
    *,
    std: float,
    bound: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return positions with bounded additive Gaussian noise.

    Each axis is perturbed independently.  The displacement is clipped
    to ``[-bound, bound]`` per axis.
    """
    if std <= 0.0 or positions.size == 0:
        return positions
    noise = rng.normal(0.0, std, size=positions.shape)
    if bound > 0.0:
        noise = np.clip(noise, -bound, bound)
    return positions + noise


@dataclass(frozen=True)
class _DelayedResultContext:
    """Internal bundle for delay-buffer result construction."""

    gt_pos: np.ndarray
    gt_vel: np.ndarray
    actor_ids: list[str]
    observed_snapshot: dict[str, Any]
    state: ObservationPerturbationState
    delay: int
    step: int
    missed_mask: np.ndarray
    occluded_mask: np.ndarray
    visibility_hidden_mask: np.ndarray
    n_actors: int
    spec: ObservationPerturbationSpec


def _build_delayed_result(ctx: _DelayedResultContext) -> dict[str, Any]:
    """Build the perturbation result with delay-buffer bookkeeping.

    Returns:
        Complete perturbation result dictionary with delayed observed state.
    """
    ctx.state._delay_buffer.append(ctx.observed_snapshot)
    if len(ctx.state._delay_buffer) > ctx.delay:
        delayed_snapshot = ctx.state._delay_buffer[0]
    else:
        delayed_snapshot = {
            "positions": ctx.gt_pos.copy(),
            "velocities": ctx.gt_vel.copy(),
            "ids": list(ctx.actor_ids),
        }
    observed_out = delayed_snapshot
    observed_actor_count = observed_out["positions"].shape[0]
    missing_ids = [aid for aid in ctx.actor_ids if aid not in observed_out["ids"]]
    return {
        "ground_truth": {
            "positions": ctx.gt_pos,
            "velocities": ctx.gt_vel,
            "ids": list(ctx.actor_ids),
        },
        "observed": observed_out,
        "missing_ids": missing_ids,
        "metadata": {
            "noise_profile": ctx.spec.noise_profile,
            "evidence_class": EVIDENCE_PERCEPTION_LIMITED,
            "position_noise_std_m": ctx.spec.position_noise_std_m,
            "position_noise_bound_m": ctx.spec.position_noise_bound_m,
            "missed_detection_probability": ctx.spec.missed_detection_probability,
            "missed_actor_count": int(ctx.missed_mask.sum()),
            "occluded_actor_count": int(ctx.occluded_mask.sum()),
            "visibility_hidden_actor_count": int(ctx.visibility_hidden_mask.sum()),
            "delay_steps": ctx.delay,
            "step": ctx.step,
            "actor_count": ctx.n_actors,
            "observed_actor_count": observed_actor_count,
        },
    }


def _compute_observed_state(
    *,
    gt_pos: np.ndarray,
    gt_vel: np.ndarray,
    actor_ids: list[str],
    n_actors: int,
    spec: ObservationPerturbationSpec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Compute the observed state after missed detections, noise, and occlusion.

    Returns:
        Tuple of observed state plus missed, occluded, and fixture-hidden masks.
    """
    missed_mask = np.zeros(n_actors, dtype=bool)
    occluded_mask = np.zeros(n_actors, dtype=bool)
    visibility_hidden_mask = np.zeros(n_actors, dtype=bool)

    # --- missed detections ---
    ids_out = list(actor_ids)
    if spec.missed_detection_probability > 0.0 and n_actors > 0:
        keep = rng.random(n_actors) >= spec.missed_detection_probability
        missed_mask = ~keep

    # --- bounded gaussian noise ---
    obs_pos_full = gt_pos.copy()
    if spec.position_noise_std_m > 0.0 and n_actors > 0:
        obs_pos_full = _apply_bounded_gaussian(
            gt_pos,
            std=spec.position_noise_std_m,
            bound=spec.position_noise_bound_m,
            rng=rng,
        )

    # --- occlusion mask ---
    if spec.occlusion_mask is not None:
        ext_mask = np.asarray(spec.occlusion_mask, dtype=bool).reshape(-1)
        if ext_mask.shape[0] != n_actors:
            raise ValueError(f"occlusion_mask length {ext_mask.shape[0]} != actor count {n_actors}")
        occluded_mask = ext_mask & ~missed_mask

    # --- scenario fixture visibility mask ---
    if spec.visibility_mask is not None:
        visible = np.asarray(spec.visibility_mask, dtype=bool).reshape(-1)
        if visible.shape[0] != n_actors:
            raise ValueError(f"visibility_mask length {visible.shape[0]} != actor count {n_actors}")
        visibility_hidden_mask = ~visible
        occluded_mask = occluded_mask & ~visibility_hidden_mask

    # Build observed array: drop missed/fixture-hidden, zero occluded positions
    drop_mask = missed_mask | visibility_hidden_mask
    if drop_mask.any():
        obs_pos = obs_pos_full[~drop_mask]
        obs_vel_out = gt_vel[~drop_mask]
        ids_out = [aid for aid, keep in zip(actor_ids, ~drop_mask, strict=True) if keep]
    else:
        obs_pos = obs_pos_full
        obs_vel_out = gt_vel.copy()
        ids_out = list(actor_ids)

    # Zero out occluded actor positions in the observed view
    for i, aid in enumerate(ids_out):
        orig_idx = actor_ids.index(aid)
        if occluded_mask[orig_idx]:
            obs_pos[i] = 0.0
            obs_vel_out[i] = 0.0

    return obs_pos, obs_vel_out, ids_out, missed_mask, occluded_mask, visibility_hidden_mask


def perturb_ground_truth(
    ground_truth_positions: np.ndarray | list,
    ground_truth_velocities: np.ndarray | list,
    actor_ids: list[str],
    *,
    spec: ObservationPerturbationSpec,
    step: int = 0,
    state: ObservationPerturbationState | None = None,
) -> dict[str, Any]:
    """Separate ground-truth actor state from a perturbed observed state.

    Args:
        ground_truth_positions: ``(N, 2)`` array of actor [x, y] positions.
        ground_truth_velocities: ``(N, 2)`` array of actor [vx, vy] velocities.
        actor_ids: List of ``N`` actor identifier strings.
        spec: Perturbation configuration.
        step: Current simulation step index (used for RNG seeding).
        state: Mutable delay-buffer state.  Required when ``spec.delay_steps > 0``.

    Returns:
        Dictionary with keys:
        - ``ground_truth``: dict of position/velocity/id arrays (always unmodified).
        - ``observed``: dict of perturbed position/velocity/id arrays.
        - ``missing_ids``: list of actor IDs not present in the observed state.
        - ``metadata``: dict with noise profile, evidence class, and per-actor flags.

    Raises:
        ValueError: When actor count mismatches or delay state is missing.
    """
    gt_pos = np.asarray(ground_truth_positions, dtype=np.float64).reshape(-1, 2).copy()
    gt_vel = np.asarray(ground_truth_velocities, dtype=np.float64).reshape(-1, 2).copy()
    n_actors = gt_pos.shape[0]

    if gt_vel.shape[0] != n_actors:
        raise ValueError(
            f"Actor count mismatch: {n_actors} positions vs {gt_vel.shape[0]} velocities"
        )
    if len(actor_ids) != n_actors:
        raise ValueError(f"Actor count mismatch: {n_actors} positions vs {len(actor_ids)} ids")

    gt_payload = {
        "positions": gt_pos,
        "velocities": gt_vel,
        "ids": list(actor_ids),
    }

    if spec.is_noop:
        return {
            "ground_truth": gt_payload,
            "observed": {
                "positions": gt_pos.copy(),
                "velocities": gt_vel.copy(),
                "ids": list(actor_ids),
            },
            "missing_ids": [],
            "metadata": {
                "noise_profile": NOISE_PROFILE_NONE,
                "evidence_class": EVIDENCE_IDEAL,
                "position_noise_std_m": 0.0,
                "position_noise_bound_m": 0.0,
                "missed_detection_probability": 0.0,
                "missed_actor_count": 0,
                "occluded_actor_count": 0,
                "visibility_hidden_actor_count": 0,
                "delay_steps": 0,
                "step": step,
                "actor_count": n_actors,
                "observed_actor_count": n_actors,
            },
        }

    rng = _make_rng(spec, step)

    (
        obs_pos,
        obs_vel_out,
        ids_out,
        missed_mask,
        occluded_mask,
        visibility_hidden_mask,
    ) = _compute_observed_state(
        gt_pos=gt_pos,
        gt_vel=gt_vel,
        actor_ids=actor_ids,
        n_actors=n_actors,
        spec=spec,
        rng=rng,
    )

    observed_snapshot = {
        "positions": obs_pos.copy(),
        "velocities": obs_vel_out.copy(),
        "ids": list(ids_out),
    }

    delay = spec.delay_steps
    if delay > 0:
        if state is None:
            raise ValueError("ObservationPerturbationState required when delay_steps > 0")
        return _build_delayed_result(
            _DelayedResultContext(
                gt_pos=gt_pos,
                gt_vel=gt_vel,
                actor_ids=actor_ids,
                observed_snapshot=observed_snapshot,
                state=state,
                delay=delay,
                step=step,
                missed_mask=missed_mask,
                occluded_mask=occluded_mask,
                visibility_hidden_mask=visibility_hidden_mask,
                n_actors=n_actors,
                spec=spec,
            )
        )

    missing_ids = [aid for aid in actor_ids if aid not in ids_out]
    return {
        "ground_truth": gt_payload,
        "observed": observed_snapshot,
        "missing_ids": missing_ids,
        "metadata": {
            "noise_profile": spec.noise_profile,
            "evidence_class": EVIDENCE_PERCEPTION_LIMITED,
            "position_noise_std_m": spec.position_noise_std_m,
            "position_noise_bound_m": spec.position_noise_bound_m,
            "missed_detection_probability": spec.missed_detection_probability,
            "missed_actor_count": int(missed_mask.sum()),
            "occluded_actor_count": int(occluded_mask.sum()),
            "visibility_hidden_actor_count": int(visibility_hidden_mask.sum()),
            "delay_steps": 0,
            "step": step,
            "actor_count": n_actors,
            "observed_actor_count": obs_pos.shape[0],
        },
    }
