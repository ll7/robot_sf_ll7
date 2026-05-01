"""Candidate samplers for adversarial search."""

from __future__ import annotations

from random import Random
from typing import Protocol

from robot_sf.adversarial.config import CandidateSpec, SearchSpaceConfig


class CandidateSampler(Protocol):
    """Protocol for optimizer-backed candidate samplers."""

    def sample(self) -> CandidateSpec:
        """Return the next candidate."""


class RandomCandidateSampler:
    """Dependency-light random-search sampler."""

    def __init__(self, search_space: SearchSpaceConfig, *, seed: int) -> None:
        """Initialize the sampler with a search space and deterministic seed."""
        self._search_space = search_space
        self._rng = Random(seed)

    def sample(self) -> CandidateSpec:
        """Return the next random candidate."""
        return self._search_space.sample_candidate(self._rng)
