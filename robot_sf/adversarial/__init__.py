"""Programmable adversarial scenario search helpers."""

from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    MultiPedAdversarialConfig,
    MultiPedCandidateSpec,
    Pose2D,
    SearchConfig,
    SearchRunResult,
    SearchSpaceConfig,
)
from robot_sf.adversarial.materialize import materialize_multi_ped_single_pedestrian_overrides
from robot_sf.adversarial.samplers import CoordinateRefinementSampler, RandomCandidateSampler
from robot_sf.adversarial.search import run_adversarial_search

__all__ = [
    "CandidateEvaluation",
    "CandidateSpec",
    "CoordinateRefinementSampler",
    "MultiPedAdversarialConfig",
    "MultiPedCandidateSpec",
    "Pose2D",
    "RandomCandidateSampler",
    "SearchConfig",
    "SearchRunResult",
    "SearchSpaceConfig",
    "materialize_multi_ped_single_pedestrian_overrides",
    "run_adversarial_search",
]
